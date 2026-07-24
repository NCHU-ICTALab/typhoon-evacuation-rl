"""Train decomposed Maskable PPO with vector critics and late scalarization."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sb3_contrib.common.maskable.utils import get_action_masks

from .build_scenario import DEFAULT_DB
from .decomposed_ppo import (
    OBJECTIVE_NAMES,
    DecomposedActorCritic,
    VectorRollout,
    compute_vector_gae,
    gradient_cosine,
    project_conflicting_gradients,
)
from .evaluate_rl import PREFERENCE_PROFILES, evaluate_model
from .train_conditioned_v2 import (
    _specialist_regret,
    make_balanced_envs,
    make_profile_envs,
)
from .train_rl import split_scenarios


def collect_rollout(
    model: DecomposedActorCritic,
    env,
    observation: np.ndarray,
    rollout_steps: int,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[VectorRollout, np.ndarray]:
    observations = []
    actions = []
    log_probs = []
    masks = []
    objective_rewards = []
    values = []
    dones = []
    preferences = []

    for _ in range(rollout_steps):
        action_masks = get_action_masks(env)
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=model.device
        )
        mask_tensor = torch.as_tensor(action_masks, device=model.device)
        with torch.no_grad():
            action_tensor, log_prob_tensor, value_tensor = model.act(
                observation_tensor, mask_tensor, deterministic=False
            )
        next_observation, _, done, infos = env.step(
            action_tensor.cpu().numpy()
        )
        vector_reward = np.asarray(
            [info["objective_reward"] for info in infos], dtype=np.float32
        )

        observations.append(observation.copy())
        actions.append(action_tensor.cpu().numpy())
        log_probs.append(log_prob_tensor.cpu().numpy())
        masks.append(np.asarray(action_masks, dtype=bool))
        objective_rewards.append(vector_reward)
        values.append(value_tensor.cpu().numpy())
        dones.append(np.asarray(done, dtype=np.float32))
        preferences.append(observation[:, -3:].copy())
        observation = next_observation

    with torch.no_grad():
        last_values = model.values(
            torch.as_tensor(
                observation, dtype=torch.float32, device=model.device
            )
        ).cpu().numpy()
    rewards_array = np.asarray(objective_rewards, dtype=np.float32)
    values_array = np.asarray(values, dtype=np.float32)
    dones_array = np.asarray(dones, dtype=np.float32)
    advantages, returns = compute_vector_gae(
        rewards_array,
        values_array,
        dones_array,
        last_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    rollout = VectorRollout(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        old_log_probs=np.asarray(log_probs, dtype=np.float32),
        action_masks=np.asarray(masks, dtype=bool),
        objective_rewards=rewards_array,
        values=values_array,
        dones=dones_array,
        preferences=np.asarray(preferences, dtype=np.float32),
        advantages=advantages,
        returns=returns,
    )
    return rollout, observation


def update_policy(
    model: DecomposedActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: VectorRollout,
    *,
    epochs: int,
    batch_size: int,
    clip_range: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    rng: np.random.Generator,
    gradient_surgery: str = "none",
    pcgrad_rng: np.random.Generator | None = None,
) -> dict:
    assert rollout.advantages is not None and rollout.returns is not None
    if gradient_surgery not in {"none", "pcgrad"}:
        raise ValueError("gradient_surgery must be 'none' or 'pcgrad'")
    projection_rng = pcgrad_rng if pcgrad_rng is not None else rng
    count = math.prod(rollout.actions.shape)
    observations = rollout.observations.reshape(count, -1)
    actions = rollout.actions.reshape(count)
    old_log_probs = rollout.old_log_probs.reshape(count)
    action_masks = rollout.action_masks.reshape(count, model.n_actions)
    preferences = rollout.preferences.reshape(count, 3)
    advantages = rollout.advantages.reshape(count, 3)
    returns = rollout.returns.reshape(count, 3)

    # Normalize each objective independently before any preference weighting.
    advantages = (advantages - advantages.mean(axis=0, keepdims=True)) / (
        advantages.std(axis=0, keepdims=True) + 1e-8
    )
    tensors = {
        "observations": torch.as_tensor(
            observations, dtype=torch.float32, device=model.device
        ),
        "actions": torch.as_tensor(actions, device=model.device),
        "old_log_probs": torch.as_tensor(
            old_log_probs, dtype=torch.float32, device=model.device
        ),
        "action_masks": torch.as_tensor(action_masks, device=model.device),
        "preferences": torch.as_tensor(
            preferences, dtype=torch.float32, device=model.device
        ),
        "advantages": torch.as_tensor(
            advantages, dtype=torch.float32, device=model.device
        ),
        "returns": torch.as_tensor(
            returns, dtype=torch.float32, device=model.device
        ),
    }

    actor_parameters = model.policy_parameters()
    metrics = defaultdict(list)
    cosine_recorded = False
    for _ in range(epochs):
        indices = rng.permutation(count)
        for start in range(0, count, batch_size):
            batch = indices[start : start + batch_size]
            log_prob, entropy, predicted_values = model.evaluate_actions(
                tensors["observations"][batch],
                tensors["actions"][batch],
                tensors["action_masks"][batch],
            )
            ratio = torch.exp(log_prob - tensors["old_log_probs"][batch])
            batch_advantages = tensors["advantages"][batch]
            unclipped = ratio[:, None] * batch_advantages
            clipped = torch.clamp(
                ratio, 1.0 - clip_range, 1.0 + clip_range
            )[:, None] * batch_advantages
            objective_surrogate = torch.minimum(unclipped, clipped)
            objective_policy_losses = -objective_surrogate.mean(dim=0)
            weighted_objective_policy_losses = -(
                tensors["preferences"][batch] * objective_surrogate
            ).mean(dim=0)

            # Late scalarization: clipping happens per objective first, then
            # each transition's preference combines the stabilized surrogates.
            policy_loss = -torch.sum(
                tensors["preferences"][batch] * objective_surrogate, dim=1
            ).mean()
            value_loss = torch.nn.functional.mse_loss(
                predicted_values, tensors["returns"][batch]
            )
            entropy_loss = -entropy.mean()
            loss = (
                policy_loss
                + value_coef * value_loss
                + entropy_coef * entropy_loss
            )

            record_gradient_diagnostics = (
                gradient_surgery == "pcgrad" or not cosine_recorded
            )
            objective_gradients = None
            if record_gradient_diagnostics:
                objective_gradients = [
                    list(
                        torch.autograd.grad(
                            weighted_objective_policy_losses[index],
                            actor_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                    )
                    for index in range(3)
                ]
                pairs = {
                    "count_gt": (0, 1),
                    "count_risk": (0, 2),
                    "gt_risk": (1, 2),
                }
                for name, (left, right) in pairs.items():
                    cosine = gradient_cosine(
                        objective_gradients[left], objective_gradients[right]
                    )
                    metrics[f"gradient_cosine_{name}"].append(cosine)
                    metrics[f"gradient_negative_fraction_{name}"].append(
                        float(cosine < 0.0)
                    )
                cosine_recorded = True

            optimizer.zero_grad()
            if gradient_surgery == "pcgrad":
                assert objective_gradients is not None
                merged_policy_gradients, projection_metrics = (
                    project_conflicting_gradients(
                        objective_gradients,
                        rng=projection_rng,
                    )
                )
                auxiliary_loss = (
                    value_coef * value_loss + entropy_coef * entropy_loss
                )
                auxiliary_loss.backward()
                for parameter, policy_gradient in zip(
                    actor_parameters, merged_policy_gradients
                ):
                    if parameter.grad is None:
                        parameter.grad = policy_gradient.clone()
                    else:
                        parameter.grad.add_(policy_gradient)
                for key, value in projection_metrics.items():
                    metrics[key].append(value)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approximate_kl = torch.mean(
                    (torch.exp(log_prob - tensors["old_log_probs"][batch]) - 1)
                    - (log_prob - tensors["old_log_probs"][batch])
                )
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1.0) > clip_range).float()
                )
            metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
            metrics["value_loss"].append(float(value_loss.detach().cpu()))
            metrics["entropy"].append(float(entropy.mean().detach().cpu()))
            metrics["approximate_kl"].append(float(approximate_kl.cpu()))
            metrics["clip_fraction"].append(float(clip_fraction.cpu()))
            for index, name in enumerate(OBJECTIVE_NAMES):
                metrics[f"policy_loss_{name}"].append(
                    float(objective_policy_losses[index].detach().cpu())
                )
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def _phase_comparison(evaluation: dict, seed: int) -> dict:
    model_root = Path(__file__).resolve().parent / "models"
    specialist_path = (
        model_root / "specialists" / f"comparison-seed-{seed}.json"
    )
    phase1_path = model_root / "conditioned_v2" / f"seed-{seed}" / "evaluation.json"
    result = {
        "paired_preference_diagnostics": evaluation[
            "paired_preference_diagnostics"
        ],
        "regret_vs_specialists": _specialist_regret(
            evaluation, specialist_path
        ),
    }
    if phase1_path.exists():
        phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
        result["utility_delta_vs_phase1"] = {
            profile: round(
                evaluation["summary_by_preference"][profile]["rl"][
                    "weighted_utility"
                ]
                - phase1["summary_by_preference"][profile]["rl"][
                    "weighted_utility"
                ],
                6,
            )
            for profile in PREFERENCE_PROFILES
        }
        result["preference_difference_cases_delta_vs_phase1"] = (
            evaluation["paired_preference_diagnostics"][
                "cases_with_preference_action_difference"
            ]
            - phase1["paired_preference_diagnostics"][
                "cases_with_preference_action_difference"
            ]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-per-profile", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--gradient-surgery", choices=("none", "pcgrad"), default="none"
    )
    parser.add_argument(
        "--actor-routing",
        choices=("shared", "hard_heads", "full_experts"),
        default="shared",
    )
    parser.add_argument(
        "--fixed-profile", choices=tuple(PREFERENCE_PROFILES), default=None
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    if args.steps_per_profile <= 0:
        raise ValueError("--steps-per-profile must be positive")
    if args.gradient_surgery == "pcgrad" and args.actor_routing != "shared":
        raise ValueError("PCGrad + routed actors is outside this controlled ablation")
    if args.fixed_profile is not None and (
        args.actor_routing != "shared" or args.gradient_surgery != "none"
    ):
        raise ValueError(
            "fixed-profile diagnostic requires shared actor and no gradient surgery"
        )
    if args.output_dir is None:
        if args.fixed_profile is not None:
            model_family = (
                Path("decomposed_ppo_fixed_profiles") / args.fixed_profile
            )
        elif args.actor_routing == "full_experts":
            model_family = "decomposed_ppo_full_experts"
        elif args.actor_routing == "hard_heads":
            model_family = "decomposed_ppo_hard_heads"
        elif args.gradient_surgery == "pcgrad":
            model_family = "decomposed_ppo_pcgrad"
        else:
            model_family = "decomposed_ppo"
        args.output_dir = Path(__file__).resolve().parent / "models" / model_family

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    pcgrad_rng = np.random.default_rng(args.seed + 1_000_003)
    train_dates, test_dates, train_scenarios, test_scenarios = split_scenarios(
        args.db
    )
    if args.fixed_profile is None:
        training_profiles = PREFERENCE_PROFILES
        env = make_balanced_envs(train_scenarios, args.seed)
    else:
        training_profiles = {
            args.fixed_profile: PREFERENCE_PROFILES[args.fixed_profile]
        }
        env = make_profile_envs(
            train_scenarios,
            args.seed,
            list(training_profiles.values()),
        )
    model = DecomposedActorCritic(
        env.observation_space,
        int(env.action_space.n),
        actor_routing=args.actor_routing,
        device=args.device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    observation = env.reset()
    n_envs = env.num_envs
    total_timesteps = args.steps_per_profile * n_envs
    timesteps = 0
    next_report = max(10_000, total_timesteps // 20)
    cosine_history = defaultdict(list)
    last_metrics = {}
    print(
        "Decomposed PPO: "
        f"actor={args.actor_routing}, surgery={args.gradient_surgery}, "
        f"fixed_profile={args.fixed_profile or 'none'}, profiles={n_envs}, "
        f"steps/profile={args.steps_per_profile:,}, "
        f"total={total_timesteps:,}, seed={args.seed}",
        flush=True,
    )
    while timesteps < total_timesteps:
        rollout_steps = min(
            args.rollout_steps,
            math.ceil((total_timesteps - timesteps) / n_envs),
        )
        rollout, observation = collect_rollout(
            model,
            env,
            observation,
            rollout_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        last_metrics = update_policy(
            model,
            optimizer,
            rollout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            clip_range=args.clip_range,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            rng=rng,
            gradient_surgery=args.gradient_surgery,
            pcgrad_rng=pcgrad_rng,
        )
        timesteps += rollout_steps * n_envs
        for key, value in last_metrics.items():
            if key.startswith("gradient_") or key.startswith("pcgrad_"):
                cosine_history[key].append(value)
        if timesteps >= next_report or timesteps >= total_timesteps:
            print(
                f"training: {timesteps:,}/{total_timesteps:,} "
                f"policy={last_metrics['policy_loss']:.4f} "
                f"value={last_metrics['value_loss']:.4f}",
                flush=True,
            )
            while next_report <= timesteps:
                next_report += max(10_000, total_timesteps // 20)

    output_dir = args.output_dir / f"seed-{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    gradient_cosines = {
        key: round(float(np.mean(values)), 6)
        for key, values in cosine_history.items()
        if key.startswith("gradient_cosine_")
    }
    gradient_conflicts = {
        key: round(float(np.mean(values)), 6)
        for key, values in cosine_history.items()
        if key.startswith("gradient_negative_fraction_")
    }
    manifest = {
        "profile": (
            args.fixed_profile
            if args.fixed_profile is not None
            else (
                f"conditioned_decomposed_ppo_{args.actor_routing}_"
                f"{args.gradient_surgery}"
            )
        ),
        "strategy": "vector_critic_per_objective_gae_late_scalarization"
        + ("_fixed_profile" if args.fixed_profile is not None else "")
        + ("_hard_preference_heads" if args.actor_routing == "hard_heads" else "")
        + ("_full_preference_experts" if args.actor_routing == "full_experts" else "")
        + ("_pcgrad_shared_policy" if args.gradient_surgery == "pcgrad" else ""),
        "actor_routing": args.actor_routing,
        "gradient_surgery": args.gradient_surgery,
        "fixed_profile": args.fixed_profile,
        "steps_per_profile": args.steps_per_profile,
        "total_timesteps": total_timesteps,
        "seed": args.seed,
        "profiles": training_profiles,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "hyperparameters": {
            "rollout_steps": args.rollout_steps,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "value_coef": args.value_coef,
            "entropy_coef": args.entropy_coef,
            "max_grad_norm": args.max_grad_norm,
            **(
                {"pcgrad_projection_seed": args.seed + 1_000_003}
                if args.gradient_surgery == "pcgrad"
                else {}
            ),
        },
        "gradient_cosines": gradient_cosines,
        "gradient_negative_fractions": gradient_conflicts,
        "pcgrad_projection_fraction": round(
            float(np.mean(cosine_history["pcgrad_projection_fraction"])), 6
        )
        if cosine_history["pcgrad_projection_fraction"]
        else 0.0,
        "last_update_metrics": last_metrics,
    }
    model_path = output_dir / "final.pt"
    model.save(model_path, metadata=manifest)
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.skip_evaluation:
        model.eval()
        evaluation = evaluate_model(
            model,
            test_scenarios,
            output_path=output_dir / "evaluation.json",
        )
        comparison = _phase_comparison(evaluation, args.seed)
        (output_dir / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
    env.close()
    print(f"model: {model_path}")


if __name__ == "__main__":
    main()
