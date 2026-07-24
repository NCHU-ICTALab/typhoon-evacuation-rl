"""Build one continuous-preference Soft MoE from Phase 8 experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .build_scenario import DEFAULT_DB
from .decomposed_ppo import ROUTING_PREFERENCES
from .evaluate_rl import PREFERENCE_PROFILES, evaluate_model
from .soft_moe import SoftMoEActorCritic
from .train_rl import split_scenarios


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERT_ROOT = (
    ROOT
    / "models"
    / "decomposed_ppo_preference_first_scalar_critic_fixed_profiles"
)


def soft_routing_targets(
    preferences: torch.Tensor, *, temperature: float
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    centroids = torch.as_tensor(
        ROUTING_PREFERENCES,
        dtype=preferences.dtype,
        device=preferences.device,
    )
    distances = torch.sum(
        (preferences[:, None, :] - centroids[None, :, :]).square(),
        dim=-1,
    )
    return torch.softmax(-distances / temperature, dim=-1)


def pretrain_preference_router(
    model: SoftMoEActorCritic,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    seed: int,
) -> dict[str, float]:
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(
        model.preference_router.parameters(), lr=learning_rate
    )
    centroids = np.asarray(ROUTING_PREFERENCES, dtype=np.float32)
    losses = []
    model.train()
    for _ in range(steps):
        continuous_count = batch_size // 2
        continuous = rng.dirichlet(
            np.full(3, 0.7), size=continuous_count
        ).astype(np.float32)
        centroid_indices = rng.integers(
            0, len(centroids), size=batch_size - continuous_count
        )
        preferences = np.concatenate(
            (continuous, centroids[centroid_indices]), axis=0
        )
        rng.shuffle(preferences)
        preference_tensor = torch.as_tensor(
            preferences, device=model.device
        )
        targets = soft_routing_targets(
            preference_tensor, temperature=temperature
        )
        log_weights = torch.log_softmax(
            model.preference_router(preference_tensor), dim=-1
        )
        loss = -torch.sum(targets * log_weights, dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_loss": float(np.mean(losses)),
    }


def centroid_routing(model: SoftMoEActorCritic) -> dict[str, list[float]]:
    observations = np.zeros((len(PREFERENCE_PROFILES), 344), dtype=np.float32)
    observations[:, -3:] = np.asarray(
        list(PREFERENCE_PROFILES.values()), dtype=np.float32
    )
    weights = model.predict_routing(observations)
    return {
        profile: np.round(weights[index], 6).tolist()
        for index, profile in enumerate(PREFERENCE_PROFILES)
    }


def continuous_routing_samples(
    model: SoftMoEActorCritic,
) -> dict[str, dict[str, list[float]]]:
    samples = {
        "count_gt_midpoint": [0.5, 0.5, 0.0],
        "gt_risk_midpoint": [0.0, 0.5, 0.5],
        "count_risk_midpoint": [0.5, 0.0, 0.5],
        "custom_20_30_50": [0.2, 0.3, 0.5],
    }
    observations = np.zeros((len(samples), 344), dtype=np.float32)
    observations[:, -3:] = np.asarray(list(samples.values()), dtype=np.float32)
    routing = model.predict_routing(observations)
    return {
        name: {
            "preference": preference,
            "routing": np.round(routing[index], 6).tolist(),
        }
        for index, (name, preference) in enumerate(samples.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--expert-root", type=Path, default=DEFAULT_EXPERT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models" / "soft_moe_router_distilled",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--router-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    expert_paths = [
        args.expert_root / profile / f"seed-{args.seed}" / "final.pt"
        for profile in PREFERENCE_PROFILES
    ]
    model = SoftMoEActorCritic.from_expert_paths(
        expert_paths, device=args.device
    )
    router_training = pretrain_preference_router(
        model,
        steps=args.router_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
    )
    routing = centroid_routing(model)
    routing_samples = continuous_routing_samples(model)
    output_dir = args.output_dir / f"seed-{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "profile": "continuous_preference_soft_moe",
        "phase": "9a_frozen_experts_distilled_router",
        "seed": args.seed,
        "expert_profiles": list(PREFERENCE_PROFILES),
        "expert_paths": [str(path) for path in expert_paths],
        "experts_frozen": True,
        "router_training_kind": "supervised_preference_distillation_not_rl",
        "router_inputs": ["preference", "global_state"],
        "state_router_initialized_zero": True,
        "router_training": {
            "steps": args.router_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            **router_training,
        },
        "centroid_routing": routing,
        "continuous_routing_samples": routing_samples,
    }
    model_path = model.save(output_dir / "final.pt", metadata=manifest)
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.skip_evaluation:
        _, _, _, test_scenarios = split_scenarios(args.db)
        evaluation = evaluate_model(
            model,
            test_scenarios,
            output_path=output_dir / "evaluation.json",
        )
        cross = evaluation["paired_preference_diagnostics"][
            "cross_utility_matrix"
        ]
        diagonal = {
            profile: max(values, key=values.get) == profile
            for profile, values in cross.items()
        }
        comparison = {
            "phase": "9a_frozen_experts_distilled_router",
            "seed": args.seed,
            "paired_base_cases": evaluation["evaluation"]["paired_base_cases"],
            "centroid_routing": routing,
            "continuous_routing_samples": routing_samples,
            "diagonal_best": diagonal,
            "diagonal_best_count": sum(diagonal.values()),
            "gt_risk_gate": diagonal["gt"] and diagonal["risk"],
            "summary_by_preference": {
                profile: evaluation["summary_by_preference"][profile]["rl"]
                for profile in PREFERENCE_PROFILES
            },
            "safety_violations": int(
                sum(
                    evaluation["summary_by_preference"][profile]["rl"][
                        "safety_violations"
                    ]
                    for profile in PREFERENCE_PROFILES
                )
            ),
            "rejected_actions": int(
                sum(
                    evaluation["summary_by_preference"][profile]["rl"][
                        "rejected_actions"
                    ]
                    for profile in PREFERENCE_PROFILES
                )
            ),
            "paired_preference_diagnostics": evaluation[
                "paired_preference_diagnostics"
            ],
        }
        (output_dir / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"model: {model_path}")


if __name__ == "__main__":
    main()
