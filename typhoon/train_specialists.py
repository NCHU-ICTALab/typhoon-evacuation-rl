"""Train the four fixed-preference diagnostic specialist policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from .build_scenario import DEFAULT_DB
from .evaluate_rl import PREFERENCE_PROFILES
from .train_rl import train_policy


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one integer seed")
    return seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seeds", type=_parse_seeds, default=[42])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "specialists",
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    for seed in args.seeds:
        for profile in PREFERENCE_PROFILES:
            train_policy(
                profile=profile,
                steps=args.steps,
                seed=seed,
                db_path=args.db,
                output_dir=args.output_root / profile / f"seed-{seed}",
                evaluate=not args.skip_evaluation,
            )


if __name__ == "__main__":
    main()
