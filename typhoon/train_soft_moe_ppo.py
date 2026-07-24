"""Phase 9B: train the Soft MoE router with frozen experts and PPO."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .build_scenario import DEFAULT_DB
from .decomposed_ppo import ROUTING_PREFERENCES, VectorRollout
from .env import TyphoonEvacuationEnv
from .evaluate_rl import PREFERENCE_PROFILES, evaluate_model
from .soft_moe import SoftMoEActorCritic
from .train_decomposed_ppo import (
    collect_rollout,
    critic_loss_components,
    preference_first_advantages,
    single_clipped_surrogate,
)
from .train_rl import split_scenarios
from .train_soft_moe import (
    centroid_routing,
    continuous_routing_samples,
    soft_routing_targets,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_INITIAL_MODEL = (
    ROOT / "models" / "soft_moe_router_distilled" / "seed-42" / "final.pt"
)
DEFAULT_OUTPUT_DIR = ROOT / "models" / "soft_moe_router_ppo"


class ContinuousPreferenceTrainingEnv(TyphoonEvacuationEnv):
    """Use paired conditions with a balanced centroid/continuous preference mix."""

    def __init__(
        self,
        scenarios: list[dict],
        *,
        anchor_preference: tuple[float, float, float],
        base_seed: int,
        lane_index: int,
        anchor_fraction: float = 0.5,
        dirichlet_alpha: float = 0.7,
    ):
        super().__init__(scenarios, randomize=False)
        if not 0.0 <= anchor_fraction <= 1.0:
            raise ValueError("anchor_fraction must be in [0, 1]")
        if dirichlet_alpha <= 0.0:
            raise ValueError("dirichlet_alpha must be positive")
        self.anchor_preference = tuple(float(value) for value in anchor_preference)
        self.base_seed = int(base_seed)
        self.lane_index = int(lane_index)
        self.anchor_fraction = float(anchor_fraction)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.episode_index = 0

    def reset(self, *, seed=None, options=None):
        episode_seed = self.base_seed + self.episode_index
        condition_rng = np.random.default_rng(episode_seed)
        preference_rng = np.random.default_rng(
            self.base_seed
            + 1_000_003 * (self.lane_index + 1)
            + self.episode_index
        )
        if preference_rng.random() < self.anchor_fraction:
            preference = self.anchor_preference
        else:
            preference = tuple(
                preference_rng.dirichlet(
                    np.full(3, self.dirichlet_alpha)
                ).tolist()
            )
        paired_options = {
            "scenario_index": int(
                condition_rng.integers(0, len(self.base_scenarios))
            ),
            "preference": preference,
            "closure_hour": float(
                condition_rng.uniform(*self.closure_range)
            ),
            "tug_capacity": int(
                condition_rng.integers(
                    self.tug_range[0], self.tug_range[1] + 1
                )
            ),
            "demand_compression": float(
                condition_rng.uniform(*self.compression_range)
            ),
            "jitter": True,
        }
        if options:
            paired_options.update(options)
        self.episode_index += 1
        return super().reset(seed=episode_seed, options=paired_options)


def make_continuous_preference_envs(
    scenarios: list[dict],
    seed: int,
    *,
    anchor_fraction: float = 0.5,
    dirichlet_alpha: float = 0.7,
) -> DummyVecEnv:
    factories = []
    for lane_index, preference in enumerate(ROUTING_PREFERENCES):

        def factory(
            lane=lane_index,
            anchor=preference,
        ):
            env = ContinuousPreferenceTrainingEnv(
                scenarios,
                anchor_preference=anchor,
                base_seed=seed,
                lane_index=lane,
                anchor_fraction=anchor_fraction,
                dirichlet_alpha=dirichlet_alpha,
            )
            env = ActionMasker(
                env, lambda wrapped: wrapped.unwrapped.action_masks()
            )
            return Monitor(env)

        factories.append(factory)
    return DummyVecEnv(factories)


def update_router_policy(
    model: SoftMoEActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: VectorRollout,
    *,
    epochs: int,
    batch_size: int,
    clip_range: float,
    value_coef: float,
    vector_value_aux_coef: float,
    entropy_coef: float,
    router_anchor_coef: float,
    router_anchor_temperature: float,
    max_grad_norm: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Update only the router and critic with one preference-first PPO clip."""
    if rollout.advantages is None or rollout.returns is None:
        raise ValueError("rollout must contain vector advantages and returns")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if router_anchor_coef < 0.0:
        raise ValueError("router_anchor_coef must be non-negative")

    count = math.prod(rollout.actions.shape)
    observations = rollout.observations.reshape(count, -1)
    actions = rollout.actions.reshape(count)
    old_log_probs = rollout.old_log_probs.reshape(count)
    action_masks = rollout.action_masks.reshape(count, model.n_actions)
    preferences = rollout.preferences.reshape(count, 3)
    vector_advantages = rollout.advantages.reshape(count, 3)
    returns = rollout.returns.reshape(count, 3)
    scalar_advantages, _ = preference_first_advantages(
        vector_advantages, preferences
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
            scalar_advantages, dtype=torch.float32, device=model.device
        ),
        "returns": torch.as_tensor(
            returns, dtype=torch.float32, device=model.device
        ),
    }

    trainable_parameters = model.trainable_parameters()
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    model.train()
    for _ in range(epochs):
        indices = rng.permutation(count)
        for start in range(0, count, batch_size):
            batch = indices[start : start + batch_size]
            batch_observations = tensors["observations"][batch]
            log_prob, action_entropy, predicted_values = model.evaluate_actions(
                batch_observations,
                tensors["actions"][batch],
                tensors["action_masks"][batch],
            )
            ratio = torch.exp(
                log_prob - tensors["old_log_probs"][batch]
            )
            policy_loss = -single_clipped_surrogate(
                ratio,
                tensors["advantages"][batch],
                clip_range,
            ).mean()
            value_loss, scalar_value_loss, vector_value_loss = (
                critic_loss_components(
                    predicted_values,
                    tensors["returns"][batch],
                    tensors["preferences"][batch],
                    critic_mode="preference_scalar_aux",
                    vector_aux_coef=vector_value_aux_coef,
                )
            )
            routing_logits = model.routing_logits(batch_observations)
            routing_targets = soft_routing_targets(
                tensors["preferences"][batch],
                temperature=router_anchor_temperature,
            )
            router_anchor_loss = -torch.sum(
                routing_targets * torch.log_softmax(routing_logits, dim=-1),
                dim=-1,
            ).mean()
            routing_weights = torch.softmax(routing_logits, dim=-1)
            routing_entropy = -torch.sum(
                routing_weights
                * torch.log(routing_weights.clamp_min(1e-12)),
                dim=-1,
            ).mean()
            entropy_loss = -action_entropy.mean()
            loss = (
                policy_loss
                + value_coef * value_loss
                + entropy_coef * entropy_loss
                + router_anchor_coef * router_anchor_loss
            )

            optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, max_grad_norm
            )
            optimizer.step()

            with torch.no_grad():
                log_ratio = log_prob - tensors["old_log_probs"][batch]
                approximate_kl = torch.mean(
                    (torch.exp(log_ratio) - 1.0) - log_ratio
                )
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1.0) > clip_range).float()
                )
            metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
            metrics["value_loss"].append(float(value_loss.detach().cpu()))
            metrics["scalar_value_loss"].append(
                float(scalar_value_loss.detach().cpu())
            )
            metrics["vector_value_loss"].append(
                float(vector_value_loss.detach().cpu())
            )
            metrics["action_entropy"].append(
                float(action_entropy.mean().detach().cpu())
            )
            metrics["routing_entropy"].append(
                float(routing_entropy.detach().cpu())
            )
            metrics["router_anchor_loss"].append(
                float(router_anchor_loss.detach().cpu())
            )
            metrics["approximate_kl"].append(float(approximate_kl.cpu()))
            metrics["clip_fraction"].append(float(clip_fraction.cpu()))
            metrics["gradient_norm"].append(float(gradient_norm.detach().cpu()))
    model.eval()
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def _diagonal_gate(evaluation: dict) -> dict[str, bool]:
    matrix = evaluation["paired_preference_diagnostics"][
        "cross_utility_matrix"
    ]
    return {
        profile: max(values, key=values.get) == profile
        for profile, values in matrix.items()
    }


def _phase9a_utility(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        profile: float(values["weighted_utility"])
        for profile, values in report.get("summary_by_preference", {}).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--initial-model", type=Path, default=DEFAULT_INITIAL_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-per-profile", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
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
    parser.add_argument("--anchor-fraction", type=float, default=0.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.7)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    if args.steps_per_profile <= 0:
        raise ValueError("--steps-per-profile must be positive")
    if not args.initial_model.exists():
        raise FileNotFoundError(f"initial Phase 9A model missing: {args.initial_model}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_dates, test_dates, train_scenarios, test_scenarios = split_scenarios(
        args.db
    )
    env = make_continuous_preference_envs(
        train_scenarios,
        args.seed,
        anchor_fraction=args.anchor_fraction,
        dirichlet_alpha=args.dirichlet_alpha,
    )
    model = SoftMoEActorCritic.load(args.initial_model, device=args.device)
    if model.observation_space.shape != env.observation_space.shape:
        raise ValueError("Phase 9A model/environment observation mismatch")
    if model.n_actions != int(env.action_space.n):
        raise ValueError("Phase 9A model/environment action mismatch")

    router_parameters = model.policy_parameters()
    critic_parameters = list(model.critic_extractor.parameters()) + list(
        model.critic.parameters()
    )
    optimizer = torch.optim.Adam(
        [
            {"params": router_parameters, "lr": args.router_learning_rate},
            {"params": critic_parameters, "lr": args.critic_learning_rate},
        ]
    )
    routing_before = centroid_routing(model)
    routing_samples_before = continuous_routing_samples(model)
    observation = env.reset()
    n_envs = env.num_envs
    total_timesteps = args.steps_per_profile * n_envs
    timesteps = 0
    report_interval = max(10_000, total_timesteps // 20)
    next_report = report_interval
    metric_history: defaultdict[str, list[float]] = defaultdict(list)
    last_metrics: dict[str, float] = {}
    print(
        "Phase 9B Soft MoE router PPO: "
        f"experts=frozen, profiles={n_envs}, "
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
        last_metrics = update_router_policy(
            model,
            optimizer,
            rollout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            clip_range=args.clip_range,
            value_coef=args.value_coef,
            vector_value_aux_coef=args.vector_value_aux_coef,
            entropy_coef=args.entropy_coef,
            router_anchor_coef=args.router_anchor_coef,
            router_anchor_temperature=args.router_anchor_temperature,
            max_grad_norm=args.max_grad_norm,
            rng=rng,
        )
        timesteps += rollout_steps * n_envs
        for key, value in last_metrics.items():
            metric_history[key].append(value)
        if timesteps >= next_report or timesteps >= total_timesteps:
            print(
                f"training: {timesteps:,}/{total_timesteps:,} "
                f"policy={last_metrics['policy_loss']:.4f} "
                f"value={last_metrics['value_loss']:.4f} "
                f"anchor={last_metrics['router_anchor_loss']:.4f} "
                f"kl={last_metrics['approximate_kl']:.5f}",
                flush=True,
            )
            while next_report <= timesteps:
                next_report += report_interval

    output_dir = args.output_dir / f"seed-{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    routing_after = centroid_routing(model)
    routing_samples_after = continuous_routing_samples(model)
    manifest = {
        "profile": "continuous_preference_soft_moe",
        "phase": "9b_frozen_experts_router_ppo",
        "seed": args.seed,
        "initial_model": str(args.initial_model),
        "experts_frozen": True,
        "trainable": ["preference_router", "state_router", "vector_critic"],
        "surrogate": "preference_first_scalar_advantage_single_clip",
        "critic": "preference_scalar_loss_plus_vector_auxiliary",
        "steps_per_profile": args.steps_per_profile,
        "total_timesteps": total_timesteps,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "preference_sampling": {
            "centroid_anchor_fraction": args.anchor_fraction,
            "continuous_dirichlet_fraction": 1.0 - args.anchor_fraction,
            "dirichlet_alpha": args.dirichlet_alpha,
        },
        "hyperparameters": {
            "rollout_steps": args.rollout_steps,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "router_learning_rate": args.router_learning_rate,
            "critic_learning_rate": args.critic_learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "value_coef": args.value_coef,
            "vector_value_aux_coef": args.vector_value_aux_coef,
            "entropy_coef": args.entropy_coef,
            "router_anchor_coef": args.router_anchor_coef,
            "router_anchor_temperature": args.router_anchor_temperature,
            "max_grad_norm": args.max_grad_norm,
        },
        "routing_before": routing_before,
        "routing_after": routing_after,
        "continuous_routing_samples_before": routing_samples_before,
        "continuous_routing_samples_after": routing_samples_after,
        "mean_update_metrics": {
            key: float(np.mean(values))
            for key, values in metric_history.items()
        },
        "last_update_metrics": last_metrics,
    }
    model_path = model.save(output_dir / "final.pt", metadata=manifest)
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.skip_evaluation:
        evaluation = evaluate_model(
            model,
            test_scenarios,
            output_path=output_dir / "evaluation.json",
        )
        diagonal = _diagonal_gate(evaluation)
        summary = {
            profile: evaluation["summary_by_preference"][profile]["rl"]
            for profile in PREFERENCE_PROFILES
        }
        phase9a_path = args.initial_model.parent / "comparison.json"
        phase9a_utility = _phase9a_utility(phase9a_path)
        utility_delta = {
            profile: round(
                summary[profile]["weighted_utility"]
                - phase9a_utility.get(
                    profile, summary[profile]["weighted_utility"]
                ),
                6,
            )
            for profile in PREFERENCE_PROFILES
        }
        safety_violations = int(
            sum(values["safety_violations"] for values in summary.values())
        )
        rejected_actions = int(
            sum(values["rejected_actions"] for values in summary.values())
        )
        gt_risk_gate = diagonal["gt"] and diagonal["risk"]
        performance_floor = all(
            utility_delta[profile] >= -0.005 for profile in ("gt", "risk")
        )
        comparison = {
            "phase": "9b_frozen_experts_router_ppo",
            "seed": args.seed,
            "paired_base_cases": evaluation["evaluation"][
                "paired_base_cases"
            ],
            "routing_before": routing_before,
            "routing_after": routing_after,
            "continuous_routing_samples_before": routing_samples_before,
            "continuous_routing_samples_after": routing_samples_after,
            "diagonal_best": diagonal,
            "diagonal_best_count": sum(diagonal.values()),
            "gt_risk_gate": gt_risk_gate,
            "performance_floor_vs_phase9a": performance_floor,
            "accepted_for_frontend": (
                gt_risk_gate
                and performance_floor
                and safety_violations == 0
                and rejected_actions == 0
            ),
            "utility_delta_vs_phase9a": utility_delta,
            "summary_by_preference": summary,
            "safety_violations": safety_violations,
            "rejected_actions": rejected_actions,
            "paired_preference_diagnostics": evaluation[
                "paired_preference_diagnostics"
            ],
        }
        (output_dir / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
    env.close()
    print(f"model: {model_path}")


if __name__ == "__main__":
    main()
