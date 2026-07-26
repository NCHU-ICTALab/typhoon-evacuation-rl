"""Phase 12b: DAgger the value-density teacher into the soft-MoE.

Plain behavior cloning of value-density reaches ~97% per-step action accuracy but
still trails value-density by ~10% utility: the small errors compound on the
decisive tight-closure dispatches (covariate shift). DAgger fixes this by rolling
out the current student, labelling the states IT visits with the teacher's
action, aggregating, and retraining. Writes only to fresh directories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .baselines import select_action
from .build_scenario import DEFAULT_DB
from .decomposed_ppo import ROUTING_PREFERENCES
from .env import TyphoonEvacuationEnv
from .soft_moe import SoftMoEActorCritic
from .train_behavior_clone import (
    _PROFILE_ORDER,
    collect_teacher_dataset,
    train_expert,
)
from .train_rl import split_scenarios
from .train_soft_moe import (
    centroid_routing,
    continuous_routing_samples,
    pretrain_preference_router,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "models" / "soft_moe_vd_dagger"


def collect_student_dataset(
    model: SoftMoEActorCritic,
    scenarios: list[dict],
    preference: tuple[float, float, float],
    *,
    episodes: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Roll out the student; label each visited state with the teacher's action."""
    env = TyphoonEvacuationEnv(scenarios, randomize=True)
    rng = np.random.default_rng(seed)
    observations, masks, actions = [], [], []
    for episode in range(episodes):
        options = {
            "scenario_index": int(rng.integers(0, len(scenarios))),
            "preference": preference,
            "closure_hour": float(rng.uniform(*env.closure_range)),
            "tug_capacity": int(rng.integers(env.tug_range[0], env.tug_range[1] + 1)),
            "demand_compression": float(rng.uniform(*env.compression_range)),
            "jitter": True,
        }
        obs, _ = env.reset(seed=seed + episode, options=options)
        done = False
        while not done:
            mask = env.action_masks()
            teacher_action = select_action(env, "value_density")
            observations.append(obs.astype(np.float32))
            masks.append(mask.astype(bool))
            actions.append(int(teacher_action))
            student_action, _ = model.predict(
                obs, deterministic=True, action_masks=mask
            )
            obs, _, done, _, _ = env.step(int(student_action))
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "action_masks": np.asarray(masks, dtype=bool),
        "actions": np.asarray(actions, dtype=np.int64),
    }


def _concat(datasets: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([d[key] for d in datasets], axis=0)
        for key in ("observations", "action_masks", "actions")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-episodes", type=int, default=400)
    parser.add_argument("--student-episodes", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--router-steps", type=int, default=4000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    train_dates, test_dates, train_scenarios, _ = split_scenarios(args.db)
    probe = TyphoonEvacuationEnv(train_scenarios, randomize=False)
    observation_space = probe.observation_space
    n_actions = int(probe.action_space.n)

    # Round 0: seed each expert's dataset with teacher-only data.
    datasets = {
        profile: [
            collect_teacher_dataset(
                train_scenarios,
                ROUTING_PREFERENCES[index],
                episodes=args.teacher_episodes,
                seed=args.seed + index,
            )
        ]
        for index, profile in enumerate(_PROFILE_ORDER)
    }

    metrics_by_round = []
    model = None
    for dagger_round in range(args.rounds + 1):
        experts = []
        round_metrics = {}
        for index, profile in enumerate(_PROFILE_ORDER):
            expert, metrics = train_expert(
                observation_space,
                n_actions,
                _concat(datasets[profile]),
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed + index + 100 * dagger_round,
                device=args.device,
            )
            experts.append(expert)
            round_metrics[profile] = metrics
        model = SoftMoEActorCritic(
            observation_space, n_actions, experts=experts, device=args.device
        )
        pretrain_preference_router(
            model,
            steps=args.router_steps,
            batch_size=256,
            learning_rate=1e-3,
            temperature=0.02,
            seed=args.seed,
        )
        metrics_by_round.append(round_metrics)
        print(
            f"round {dagger_round}: "
            + " ".join(
                f"{p}={round_metrics[p]['final_accuracy']:.4f}" for p in _PROFILE_ORDER
            ),
            flush=True,
        )
        if dagger_round == args.rounds:
            break
        # Aggregate student-visited states labelled by the teacher.
        for index, profile in enumerate(_PROFILE_ORDER):
            datasets[profile].append(
                collect_student_dataset(
                    model,
                    train_scenarios,
                    ROUTING_PREFERENCES[index],
                    episodes=args.student_episodes,
                    seed=args.seed + index + 1000 * (dagger_round + 1),
                )
            )

    output_dir = args.output_dir / f"seed-{args.seed}"
    manifest = {
        "phase": "12b_value_density_dagger",
        "seed": args.seed,
        "teacher": "value_density",
        "rounds": args.rounds,
        "profiles": list(_PROFILE_ORDER),
        "train_dates": train_dates,
        "test_dates": test_dates,
        "accuracy_by_round": [
            {p: metrics_by_round[r][p]["final_accuracy"] for p in _PROFILE_ORDER}
            for r in range(len(metrics_by_round))
        ],
        "centroid_routing": centroid_routing(model),
        "continuous_routing_samples": continuous_routing_samples(model),
        "hyperparameters": vars(args) | {"db": str(args.db), "output_dir": str(args.output_dir)},
    }
    model_path = model.save(output_dir / "final.pt", metadata=manifest)
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["accuracy_by_round"], ensure_ascii=False))
    print(f"model: {model_path}")


if __name__ == "__main__":
    main()
