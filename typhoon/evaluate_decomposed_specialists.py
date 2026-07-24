"""Paired evaluation for custom decomposed PPO fixed-profile models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_scenario import (
    DEFAULT_DB,
    available_scenario_dates,
    build_scenario_pool,
)
from .decomposed_ppo import DecomposedActorCritic
from .evaluate_rl import PREFERENCE_PROFILES
from .evaluate_specialists import evaluate_specialists


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(__file__).resolve().parent
        / "models"
        / "decomposed_ppo_fixed_profiles",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--algorithm",
        default="custom_decomposed_ppo_fixed_profile",
        help="Label stored in the comparison report.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PREFERENCE_PROFILES),
        default=list(PREFERENCE_PROFILES),
        help="Subset of fixed-profile models to load.",
    )
    args = parser.parse_args()

    dates = available_scenario_dates(args.db, min_vessels=30)
    test_dates = dates[-5:] if len(dates) >= 5 else dates
    scenarios = build_scenario_pool(
        db_path=args.db,
        dates=test_dates,
        max_vessels=30,
    )
    models = {
        profile: DecomposedActorCritic.load(
            args.model_root / profile / f"seed-{args.seed}" / "final.pt"
        )
        for profile in args.profiles
    }
    report = evaluate_specialists(models, scenarios)
    report["algorithm"] = args.algorithm
    report["seed"] = args.seed
    output = args.output or args.model_root / f"comparison-seed-{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
