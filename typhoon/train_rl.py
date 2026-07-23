"""Train conditioned or fixed-preference MaskablePPO policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .build_scenario import DEFAULT_DB, available_scenario_dates, build_scenario_pool
from .env import TyphoonEvacuationEnv
from .evaluate_rl import PREFERENCE_PROFILES, evaluate_model


class TrainingProgress(BaseCallback):
    def __init__(self, every: int):
        super().__init__()
        self.every = max(1, every)
        self.next_report = self.every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_report:
            self.next_report += self.every
            print(f"training: {self.num_timesteps:,} steps", flush=True)
        return True


def make_training_env(
    scenarios: list[dict],
    seed: int,
    preference: tuple[float, float, float] | None = None,
):
    env = TyphoonEvacuationEnv(
        scenarios, randomize=True, fixed_preference=preference
    )
    env = ActionMasker(env, lambda wrapped: wrapped.unwrapped.action_masks())
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def split_scenarios(db_path: Path) -> tuple[list[str], list[str], list[dict], list[dict]]:
    dates = available_scenario_dates(db_path, min_vessels=30)
    if len(dates) < 2:
        raise ValueError("at least two usable offline dates are required")
    test_count = min(5, max(1, len(dates) // 5))
    train_dates, test_dates = dates[:-test_count], dates[-test_count:]
    train_scenarios = build_scenario_pool(
        db_path=db_path, dates=train_dates, max_vessels=30
    )
    test_scenarios = build_scenario_pool(
        db_path=db_path, dates=test_dates, max_vessels=30
    )
    return train_dates, test_dates, train_scenarios, test_scenarios


def train_policy(
    *,
    profile: str,
    steps: int,
    seed: int,
    db_path: Path,
    output_dir: Path,
    evaluate: bool = True,
) -> dict:
    if profile != "conditioned" and profile not in PREFERENCE_PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    if steps <= 0:
        raise ValueError("steps must be positive")

    fixed_preference = (
        None if profile == "conditioned" else PREFERENCE_PROFILES[profile]
    )
    train_dates, test_dates, train_scenarios, test_scenarios = split_scenarios(
        db_path
    )
    env = make_training_env(train_scenarios, seed, fixed_preference)
    n_steps = min(512, max(64, steps))
    batch_size = min(128, n_steps)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=3e-4,
        gamma=0.995,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
    )
    print(
        f"Typhoon RL: profile={profile}, steps={steps:,}, seed={seed}, "
        f"obs={env.observation_space.shape[0]}, actions={env.action_space.n}",
        flush=True,
    )
    model.learn(
        total_timesteps=steps,
        callback=TrainingProgress(max(2_000, steps // 10)),
        progress_bar=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "final.zip"
    model.save(model_path)
    report = None
    if evaluate:
        report = evaluate_model(
            model, test_scenarios, output_path=output_dir / "evaluation.json"
        )

    manifest = {
        "profile": profile,
        "preference": fixed_preference,
        "steps": steps,
        "seed": seed,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "observation_size": int(env.observation_space.shape[0]),
        "actions": int(env.action_space.n),
        "reward_contract": "vector_objectives_then_scalarize",
        "risk_contract": "synthetic_independent_proxy_not_official_priority",
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    env.close()
    print(f"model: {model_path}")
    if report is not None:
        own = (
            report["summary"]
            if profile == "conditioned"
            else report["summary_by_preference"][profile]
        )
        print(json.dumps(own, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--profile",
        choices=("conditioned", *PREFERENCE_PROFILES),
        default="conditioned",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.profile == "conditioned":
        output_dir = Path(__file__).resolve().parent / "models"
    else:
        output_dir = (
            Path(__file__).resolve().parent
            / "models"
            / "specialists"
            / args.profile
            / f"seed-{args.seed}"
        )
    train_policy(
        profile=args.profile,
        steps=args.steps,
        seed=args.seed,
        db_path=args.db,
        output_dir=output_dir,
        evaluate=not args.skip_evaluation,
    )


if __name__ == "__main__":
    main()
