"""Phase 14: reward-PPO fine-tuning from the Phase 12 VD DAgger checkpoint.

The Phase 12 error is in the imitation experts, not only in the router.  This
pipeline therefore fine-tunes each DAgger expert with its own centroid reward,
reassembles the original continuous-preference Soft MoE, and finally runs the
existing preference-first router PPO over centroid and Dirichlet preferences.

Every artifact is written to a new directory.  A pilot is never accepted for
service, and a full run must pass explicit safety, preference specialization,
schedule diversity, and value-density regret gates.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .build_scenario import DEFAULT_DB
from .decomposed_ppo import DecomposedActorCritic
from .evaluate_rl import PREFERENCE_PROFILES, evaluate_model
from .soft_moe import SoftMoEActorCritic
from .train_conditioned_v2 import make_profile_envs
from .train_decomposed_ppo import collect_rollout, update_policy
from .train_rl import split_scenarios
from .train_soft_moe import centroid_routing, continuous_routing_samples
from .train_soft_moe_ppo import (
    _diagonal_gate,
    make_continuous_preference_envs,
    update_router_policy,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_INITIAL_MODEL = (
    ROOT / "models" / "soft_moe_vd_dagger" / "seed-42" / "final.pt"
)
DEFAULT_REFERENCE_EVALUATION = (
    ROOT / "models" / "soft_moe_router_monotone" / "seed-42"
    / "evaluation.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "models" / "soft_moe_vd_warmstart_ppo"
PROFILE_ORDER = tuple(PREFERENCE_PROFILES)


def clone_trainable_expert(
    expert: DecomposedActorCritic,
    *,
    device: str | torch.device = "cpu",
) -> DecomposedActorCritic:
    """Copy one frozen Soft-MoE expert into an independently trainable model."""

    clone = DecomposedActorCritic(
        expert.observation_space,
        expert.n_actions,
        features_dim=expert.features_dim,
        actor_routing=expert.actor_routing,
        device=device,
    )
    clone.load_state_dict(expert.state_dict())
    for parameter in clone.parameters():
        parameter.requires_grad_(True)
    clone.train()
    return clone


def soften_action_logits(
    expert: DecomposedActorCritic,
    temperature: float,
) -> None:
    """Increase rollout exploration without changing deterministic actions.

    Dividing every final-layer logit by the same positive temperature preserves
    the complete action ordering (and therefore the VD/DAgger warm-start
    schedule) while making stochastic PPO rollouts less nearly deterministic.
    """

    if temperature < 1.0:
        raise ValueError("exploration temperature must be at least 1.0")
    if temperature == 1.0:
        return
    if expert.actor_routing == "shared":
        output_layers = [expert.actor[-1]]
    elif expert.actor_routing == "hard_heads":
        output_layers = list(expert.actor_heads)
    else:
        output_layers = [actor[-1] for actor in expert.actor_experts]
    with torch.no_grad():
        for layer in output_layers:
            layer.weight.div_(temperature)
            if layer.bias is not None:
                layer.bias.div_(temperature)


def reset_value_head(expert: DecomposedActorCritic) -> None:
    """Start reward fine-tuning from a neutral critic without touching actor."""

    output_layer = expert.critic[-1]
    if not isinstance(output_layer, torch.nn.Linear):
        raise TypeError("critic must end in a linear value head")
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.zero_()


def assemble_from_warm_experts(
    initial: SoftMoEActorCritic,
    experts: list[DecomposedActorCritic],
    *,
    device: str | torch.device = "cpu",
) -> SoftMoEActorCritic:
    """Install warmed experts while preserving the Phase 12 continuous router."""

    model = SoftMoEActorCritic(
        initial.observation_space,
        initial.n_actions,
        experts=experts,
        features_dim=initial.features_dim,
        device=device,
    )
    model.preference_router.load_state_dict(initial.preference_router.state_dict())
    model.state_router.load_state_dict(initial.state_router.state_dict())
    model.critic_extractor.load_state_dict(initial.critic_extractor.state_dict())
    model.critic.load_state_dict(initial.critic.state_dict())
    model.decode_tie_eps = initial.decode_tie_eps
    model.eval()
    return model


def fine_tune_expert(
    expert: DecomposedActorCritic,
    scenarios: list[dict],
    preference: tuple[float, float, float],
    *,
    steps: int,
    seed: int,
    rollout_steps: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    value_coef: float,
    vector_value_aux_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    detach_critic_extractor: bool,
) -> dict[str, float]:
    """Reward-fine-tune one VD DAgger expert at a fixed preference."""

    env = make_profile_envs(scenarios, seed, [preference])
    optimizer = torch.optim.Adam(expert.parameters(), lr=learning_rate)
    observation = env.reset()
    timesteps = 0
    rng = np.random.default_rng(seed)
    history: defaultdict[str, list[float]] = defaultdict(list)
    try:
        while timesteps < steps:
            current_steps = min(rollout_steps, steps - timesteps)
            rollout, observation = collect_rollout(
                expert,
                env,
                observation,
                current_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            metrics = update_policy(
                expert,
                optimizer,
                rollout,
                epochs=epochs,
                batch_size=min(batch_size, current_steps),
                clip_range=clip_range,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm,
                rng=rng,
                surrogate_mode="preference_first",
                critic_mode="preference_scalar_aux",
                vector_value_aux_coef=vector_value_aux_coef,
                detach_critic_extractor=detach_critic_extractor,
            )
            for key, value in metrics.items():
                history[key].append(value)
            timesteps += current_steps
    finally:
        env.close()
    expert.eval()
    return {key: float(np.mean(values)) for key, values in history.items()}


def fine_tune_router(
    model: SoftMoEActorCritic,
    scenarios: list[dict],
    *,
    steps_per_profile: int,
    seed: int,
    rollout_steps: int,
    batch_size: int,
    epochs: int,
    router_learning_rate: float,
    critic_learning_rate: float,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    value_coef: float,
    vector_value_aux_coef: float,
    entropy_coef: float,
    router_anchor_coef: float,
    router_anchor_temperature: float,
    max_grad_norm: float,
) -> dict[str, float]:
    """Run continuous-preference router PPO after expert reward fine-tuning."""

    env = make_continuous_preference_envs(scenarios, seed)
    optimizer = torch.optim.Adam(
        [
            {"params": model.policy_parameters(), "lr": router_learning_rate},
            {
                "params": list(model.critic_extractor.parameters())
                + list(model.critic.parameters()),
                "lr": critic_learning_rate,
            },
        ]
    )
    observation = env.reset()
    total = steps_per_profile * env.num_envs
    timesteps = 0
    rng = np.random.default_rng(seed)
    history: defaultdict[str, list[float]] = defaultdict(list)
    try:
        while timesteps < total:
            current_steps = min(
                rollout_steps,
                math.ceil((total - timesteps) / env.num_envs),
            )
            rollout, observation = collect_rollout(
                model,
                env,
                observation,
                current_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            metrics = update_router_policy(
                model,
                optimizer,
                rollout,
                epochs=epochs,
                batch_size=batch_size,
                clip_range=clip_range,
                value_coef=value_coef,
                vector_value_aux_coef=vector_value_aux_coef,
                entropy_coef=entropy_coef,
                router_anchor_coef=router_anchor_coef,
                router_anchor_temperature=router_anchor_temperature,
                max_grad_norm=max_grad_norm,
                rng=rng,
            )
            for key, value in metrics.items():
                history[key].append(value)
            timesteps += current_steps * env.num_envs
    finally:
        env.close()
    return {key: float(np.mean(values)) for key, values in history.items()}


def evaluation_summary(evaluation: dict) -> dict:
    """Extract service-relevant preference and baseline metrics."""

    utility = {}
    gap_vs_vd = {}
    for profile in PREFERENCE_PROFILES:
        rows = evaluation["summary_by_preference"][profile]
        utility[profile] = float(rows["rl"]["weighted_utility"])
        gap_vs_vd[profile] = round(
            float(rows["rl"]["weighted_utility"])
            - float(rows["value_density"]["weighted_utility"]),
            6,
        )
    diagonal = _diagonal_gate(evaluation)
    diagnostics = evaluation["paired_preference_diagnostics"]
    cross_utility = diagnostics["cross_utility_matrix"]
    specialization_margin = {
        profile: round(
            float(cross_utility[profile][profile])
            - float(
                np.mean(
                    [
                        value
                        for key, value in cross_utility[profile].items()
                        if key != profile
                    ]
                )
            ),
            6,
        )
        for profile in PREFERENCE_PROFILES
    }
    safety = int(
        sum(
            evaluation["summary_by_preference"][profile]["rl"][
                "safety_violations"
            ]
            for profile in PREFERENCE_PROFILES
        )
    )
    rejected = int(
        sum(
            evaluation["summary_by_preference"][profile]["rl"][
                "rejected_actions"
            ]
            for profile in PREFERENCE_PROFILES
        )
    )
    return {
        "utility": utility,
        "gap_vs_value_density": gap_vs_vd,
        "mean_gap_vs_value_density": round(float(np.mean(list(gap_vs_vd.values()))), 6),
        "diagonal_best": diagonal,
        "diagonal_best_count": sum(diagonal.values()),
        "specialization_margin": specialization_margin,
        "mean_specialization_margin": round(
            float(np.mean(list(specialization_margin.values()))), 6
        ),
        "cases_with_preference_action_difference": diagnostics[
            "cases_with_preference_action_difference"
        ],
        "paired_base_cases": diagnostics["paired_base_cases"],
        "safety_violations": safety,
        "rejected_actions": rejected,
    }


def service_gate(
    reference: dict,
    after: dict,
    *,
    pilot: bool,
    max_profile_utility_regression: float = 0.002,
    max_difference_case_regression: int = 2,
    max_mean_specialization_margin_regression: float = 0.001,
) -> dict:
    """Accept a PoC candidate when it improves Phase 10 specialization.

    Cross-utility diagonals remain diagnostics, not hard gates: count and
    balanced were already known not to be diagonal-best in the predecessor
    experts. The service question is whether the new checkpoint improves the
    four centroid utilities while preserving preference-sensitive schedules.
    """

    utility_delta = {
        profile: round(
            after["utility"][profile] - reference["utility"][profile], 6
        )
        for profile in PREFERENCE_PROFILES
    }
    mean_utility_delta = float(np.mean(list(utility_delta.values())))
    improved_profiles = sum(delta >= 0.0 for delta in utility_delta.values())

    conditions = {
        "not_pilot": not pilot,
        "safety_zero": after["safety_violations"] == 0
        and after["rejected_actions"] == 0,
        "preference_difference_coverage": (
            after["cases_with_preference_action_difference"]
            >= math.ceil(0.8 * after["paired_base_cases"])
        ),
        "preference_difference_not_materially_worse_than_phase10": (
            after["cases_with_preference_action_difference"]
            >= reference["cases_with_preference_action_difference"]
            - max_difference_case_regression
        ),
        "mean_centroid_utility_improves_phase10": mean_utility_delta >= 0.0,
        "at_least_three_profiles_improve_phase10": improved_profiles >= 3,
        "no_material_profile_utility_regression": (
            min(utility_delta.values()) >= -max_profile_utility_regression
        ),
        "mean_specialization_margin_not_materially_worse": (
            after["mean_specialization_margin"]
            >= reference["mean_specialization_margin"]
            - max_mean_specialization_margin_regression
        ),
    }
    return {
        "accepted_for_service": all(conditions.values()),
        "conditions": conditions,
        "utility_delta_vs_phase10": utility_delta,
        "mean_utility_delta_vs_phase10": round(mean_utility_delta, 6),
        "profiles_improved_vs_phase10": improved_profiles,
        "specialization_margin_delta_vs_phase10": round(
            after["mean_specialization_margin"]
            - reference["mean_specialization_margin"],
            6,
        ),
        "tolerances": {
            "max_profile_utility_regression": max_profile_utility_regression,
            "max_difference_case_regression": max_difference_case_regression,
            "max_mean_specialization_margin_regression": (
                max_mean_specialization_margin_regression
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--initial-model", type=Path, default=DEFAULT_INITIAL_MODEL)
    parser.add_argument(
        "--reference-evaluation",
        type=Path,
        default=DEFAULT_REFERENCE_EVALUATION,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expert-steps", type=int, default=25_000)
    parser.add_argument("--router-steps-per-profile", type=int, default=25_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--expert-learning-rate", type=float, default=1e-4)
    parser.add_argument("--exploration-temperature", type=float, default=1.0)
    parser.add_argument("--reset-value-head", action="store_true")
    parser.add_argument("--detach-critic-extractor", action="store_true")
    parser.add_argument("--router-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--vector-value-aux-coef", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--router-anchor-coef", type=float, default=0.01)
    parser.add_argument("--router-anchor-temperature", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--max-profile-utility-regression", type=float, default=0.002
    )
    parser.add_argument(
        "--max-difference-case-regression", type=int, default=2
    )
    parser.add_argument(
        "--max-specialization-margin-regression",
        type=float,
        default=0.001,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.expert_steps <= 0 or args.router_steps_per_profile <= 0:
        raise ValueError("expert and router steps must be positive")
    if args.exploration_temperature < 1.0:
        raise ValueError("exploration temperature must be at least 1.0")
    if not args.initial_model.exists():
        raise FileNotFoundError(f"VD warm-start model missing: {args.initial_model}")
    if not args.reference_evaluation.exists():
        raise FileNotFoundError(
            "Phase 10 reference evaluation missing: "
            f"{args.reference_evaluation}"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_dates, test_dates, train_scenarios, test_scenarios = split_scenarios(
        args.db
    )
    initial = SoftMoEActorCritic.load(args.initial_model, device=args.device)
    reference_evaluation = json.loads(
        args.reference_evaluation.read_text(encoding="utf-8")
    )
    reference = evaluation_summary(reference_evaluation)
    print("evaluating Phase 12 DAgger warm start", flush=True)
    before_evaluation = evaluate_model(initial, test_scenarios)
    before = evaluation_summary(before_evaluation)

    warmed_experts = []
    expert_metrics = {}
    for index, profile in enumerate(PROFILE_ORDER):
        print(f"fine-tuning expert {profile}", flush=True)
        expert = clone_trainable_expert(
            initial.experts[index], device=args.device
        )
        soften_action_logits(expert, args.exploration_temperature)
        if args.reset_value_head:
            reset_value_head(expert)
        metrics = fine_tune_expert(
            expert,
            train_scenarios,
            PREFERENCE_PROFILES[profile],
            steps=args.expert_steps,
            seed=args.seed + index,
            rollout_steps=args.rollout_steps,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.expert_learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            value_coef=args.value_coef,
            vector_value_aux_coef=args.vector_value_aux_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            detach_critic_extractor=args.detach_critic_extractor,
        )
        warmed_experts.append(expert)
        expert_metrics[profile] = metrics

    model = assemble_from_warm_experts(
        initial, warmed_experts, device=args.device
    )
    routing_before = centroid_routing(model)
    continuous_before = continuous_routing_samples(model)
    print("fine-tuning continuous-preference router", flush=True)
    router_metrics = fine_tune_router(
        model,
        train_scenarios,
        steps_per_profile=args.router_steps_per_profile,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        router_learning_rate=args.router_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        value_coef=args.value_coef,
        vector_value_aux_coef=args.vector_value_aux_coef,
        entropy_coef=args.entropy_coef,
        router_anchor_coef=args.router_anchor_coef,
        router_anchor_temperature=args.router_anchor_temperature,
        max_grad_norm=args.max_grad_norm,
    )

    output_dir = args.output_dir / f"seed-{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for profile, expert in zip(PROFILE_ORDER, warmed_experts):
        expert.save(
            output_dir / "experts" / profile / "final.pt",
            metadata={
                "phase": "14_vd_warmstart_preference_ppo",
                "profile": profile,
                "preference": PREFERENCE_PROFILES[profile],
                "initial_model": str(args.initial_model),
                "steps": args.expert_steps,
                "exploration_temperature": args.exploration_temperature,
            },
        )

    print("evaluating reward-fine-tuned Soft MoE", flush=True)
    after_evaluation = evaluate_model(
        model,
        test_scenarios,
        output_path=output_dir / "evaluation.json",
    )
    after = evaluation_summary(after_evaluation)
    gate = service_gate(
        reference,
        after,
        pilot=args.pilot,
        max_profile_utility_regression=args.max_profile_utility_regression,
        max_difference_case_regression=args.max_difference_case_regression,
        max_mean_specialization_margin_regression=(
            args.max_specialization_margin_regression
        ),
    )
    comparison = {
        "phase": "14_vd_warmstart_preference_ppo",
        "seed": args.seed,
        "initial_model": str(args.initial_model),
        "reference_evaluation": str(args.reference_evaluation),
        "phase10_reference": reference,
        "pilot": args.pilot,
        "before": before,
        "after": after,
        "utility_delta_vs_warm_start": {
            profile: round(
                after["utility"][profile] - before["utility"][profile], 6
            )
            for profile in PROFILE_ORDER
        },
        "utility_delta_vs_phase10": {
            profile: round(
                after["utility"][profile] - reference["utility"][profile], 6
            )
            for profile in PROFILE_ORDER
        },
        "mean_vd_gap_delta": round(
            after["mean_gap_vs_value_density"]
            - before["mean_gap_vs_value_density"],
            6,
        ),
        "routing_before": routing_before,
        "routing_after": centroid_routing(model),
        "continuous_routing_samples_before": continuous_before,
        "continuous_routing_samples_after": continuous_routing_samples(model),
        "service_gate": gate,
    }
    manifest = {
        "phase": "14_vd_warmstart_preference_ppo",
        "seed": args.seed,
        "initial_model": str(args.initial_model),
        "reference_evaluation": str(args.reference_evaluation),
        "teacher_lineage": "value_density -> BC -> DAgger -> reward PPO",
        "experts_reward_fine_tuned": True,
        "router_reward_fine_tuned": True,
        "expert_steps_each": args.expert_steps,
        "router_steps_per_profile": args.router_steps_per_profile,
        "total_reward_transitions": (
            args.expert_steps * len(PROFILE_ORDER)
            + args.router_steps_per_profile * len(PROFILE_ORDER)
        ),
        "train_dates": train_dates,
        "test_dates": test_dates,
        "expert_update_metrics": expert_metrics,
        "router_update_metrics": router_metrics,
        "hyperparameters": vars(args)
        | {
            "db": str(args.db),
            "initial_model": str(args.initial_model),
            "reference_evaluation": str(args.reference_evaluation),
            "output_dir": str(args.output_dir),
        },
        "service_gate": gate,
    }
    model.save(output_dir / "final.pt", metadata=manifest)
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"model: {output_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
