"""Train one preference-gated policy with balanced four-profile rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .build_scenario import DEFAULT_DB
from .env import TyphoonEvacuationEnv
from .evaluate_rl import PREFERENCE_PROFILES, evaluate_model
from .policy import PreferenceGatedExtractor
from .train_rl import TrainingProgress, split_scenarios


class PairedProfileTrainingEnv(TyphoonEvacuationEnv):
    """Give every profile the same deterministic scenario stream by episode index."""

    def __init__(
        self,
        scenarios: list[dict],
        *,
        preference: tuple[float, float, float],
        base_seed: int,
    ):
        super().__init__(
            scenarios,
            randomize=False,
            fixed_preference=preference,
        )
        self.profile_preference = preference
        self.base_seed = int(base_seed)
        self.episode_index = 0

    def reset(self, *, seed=None, options=None):
        episode_seed = self.base_seed + self.episode_index
        rng = np.random.default_rng(episode_seed)
        paired_options = {
            "scenario_index": int(rng.integers(0, len(self.base_scenarios))),
            "preference": self.profile_preference,
            "closure_hour": float(rng.uniform(*self.closure_range)),
            "tug_capacity": int(
                rng.integers(self.tug_range[0], self.tug_range[1] + 1)
            ),
            "demand_compression": float(rng.uniform(*self.compression_range)),
            "jitter": True,
        }
        if options:
            paired_options.update(options)
        self.episode_index += 1
        return super().reset(seed=episode_seed, options=paired_options)


def make_balanced_envs(scenarios: list[dict], seed: int) -> DummyVecEnv:
    factories = []
    for preference in PREFERENCE_PROFILES.values():
        def factory(profile_preference=preference):
            env = PairedProfileTrainingEnv(
                scenarios,
                preference=profile_preference,
                base_seed=seed,
            )
            env = ActionMasker(
                env, lambda wrapped: wrapped.unwrapped.action_masks()
            )
            return Monitor(env)

        factories.append(factory)
    return DummyVecEnv(factories)


def _specialist_regret(evaluation: dict, specialist_path: Path) -> dict | None:
    if not specialist_path.exists():
        return None
    specialists = json.loads(specialist_path.read_text(encoding="utf-8"))
    result = {}
    for profile in PREFERENCE_PROFILES:
        conditioned = evaluation["summary_by_preference"][profile]["rl"][
            "weighted_utility"
        ]
        specialist = specialists["cross_utility_matrix"][profile][profile]
        result[profile] = {
            "conditioned_utility": conditioned,
            "specialist_utility": specialist,
            "relative_regret": round(
                (specialist - conditioned) / max(abs(specialist), 1e-9), 6
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-per-profile", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "conditioned_v2",
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    if args.steps_per_profile <= 0:
        raise ValueError("--steps-per-profile must be positive")

    train_dates, test_dates, train_scenarios, test_scenarios = split_scenarios(
        args.db
    )
    env = make_balanced_envs(train_scenarios, args.seed)
    total_timesteps = args.steps_per_profile * len(PREFERENCE_PROFILES)
    rollout_steps = min(512, max(64, args.steps_per_profile // 20))
    batch_size = min(256, rollout_steps * len(PREFERENCE_PROFILES))
    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=0,
        n_steps=rollout_steps,
        batch_size=batch_size,
        learning_rate=3e-4,
        gamma=0.995,
        ent_coef=0.01,
        policy_kwargs={
            "features_extractor_class": PreferenceGatedExtractor,
            "features_extractor_kwargs": {
                "n_vessels": 30,
                "vessel_features": 11,
                "global_features": 14,
                "features_dim": 256,
            },
            "net_arch": {"pi": [256, 128], "vf": [256, 128]},
        },
    )
    print(
        "Conditioned v2: "
        f"profiles={len(PREFERENCE_PROFILES)}, "
        f"steps/profile={args.steps_per_profile:,}, "
        f"total={total_timesteps:,}, seed={args.seed}",
        flush=True,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=TrainingProgress(max(10_000, total_timesteps // 20)),
        progress_bar=False,
    )

    output_dir = args.output_dir / f"seed-{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "final.zip"
    model.save(model_path)
    manifest = {
        "profile": "conditioned_v2",
        "strategy": "balanced_profiles_with_preference_film_gating",
        "steps_per_profile": args.steps_per_profile,
        "total_timesteps": total_timesteps,
        "seed": args.seed,
        "profiles": PREFERENCE_PROFILES,
        "paired_scenario_stream": True,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "observation_size": int(env.observation_space.shape[0]),
        "actions": int(env.action_space.n),
    }
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
        specialist_path = (
            Path(__file__).resolve().parent
            / "models"
            / "specialists"
            / f"comparison-seed-{args.seed}.json"
        )
        comparison = {
            "paired_preference_diagnostics": evaluation[
                "paired_preference_diagnostics"
            ],
            "regret_vs_specialists": _specialist_regret(
                evaluation, specialist_path
            ),
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
