"""Paired held-out comparison of fixed-preference specialist policies."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from .build_scenario import DEFAULT_DB, available_scenario_dates, build_scenario_pool
from .env import TyphoonEvacuationEnv
from .evaluate_rl import PREFERENCE_PROFILES, _utility, evaluation_cases, run_rl


def evaluate_specialists(
    models: dict[str, MaskablePPO], scenarios: list[dict]
) -> dict:
    env = TyphoonEvacuationEnv(scenarios, randomize=False)
    all_cases = evaluation_cases(len(scenarios))
    rows: dict[str, list[dict]] = {profile: [] for profile in models}

    for profile, model in models.items():
        for seed, options in all_cases:
            if options["preference_name"] != profile:
                continue
            rows[profile].append(run_rl(model, env, seed=seed, options=options))

    cross_utility: dict[str, dict[str, float]] = {}
    for eval_profile, weights in PREFERENCE_PROFILES.items():
        cross_utility[eval_profile] = {
            producer: round(
                float(np.mean([_utility(row, weights) for row in profile_rows])),
                6,
            )
            for producer, profile_rows in rows.items()
        }

    grouped = defaultdict(dict)
    for profile, profile_rows in rows.items():
        for row in profile_rows:
            grouped[row["case_id"]][profile] = row
    trace_histogram = Counter()
    schedule_histogram = Counter()
    for group in grouped.values():
        trace_histogram[
            len({tuple(row["action_trace"]) for row in group.values()})
        ] += 1
        schedule_histogram[
            len({tuple(row["schedule_signature"]) for row in group.values()})
        ] += 1

    metrics = (
        "weighted_utility",
        "evacuated_count",
        "evacuated_gt",
        "evacuated_risk_points",
        "remaining_count",
        "remaining_gt",
        "remaining_risk_points",
        "safety_violations",
        "rejected_actions",
    )
    summary = {
        profile: {
            metric: round(
                float(np.mean([row[metric] for row in profile_rows])), 4
            )
            for metric in metrics
        }
        for profile, profile_rows in rows.items()
    }
    diagonal_best = {
        eval_profile: max(values, key=values.get) == eval_profile
        for eval_profile, values in cross_utility.items()
    }
    return {
        "profiles": PREFERENCE_PROFILES,
        "held_out_dates": len(scenarios),
        "paired_cases_per_profile": len(all_cases) // len(PREFERENCE_PROFILES),
        "summary": summary,
        "cross_utility_matrix": cross_utility,
        "diagonal_best": diagonal_best,
        "diagonal_best_count": sum(diagonal_best.values()),
        "action_trace_unique_count_histogram": {
            str(key): value for key, value in sorted(trace_histogram.items())
        },
        "schedule_unique_count_histogram": {
            str(key): value for key, value in sorted(schedule_histogram.items())
        },
        "cases_with_multiple_schedules": sum(
            value for key, value in schedule_histogram.items() if key > 1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "specialists",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dates = available_scenario_dates(args.db, min_vessels=30)
    test_dates = dates[-5:] if len(dates) >= 5 else dates
    scenarios = build_scenario_pool(
        db_path=args.db, dates=test_dates, max_vessels=30
    )
    models = {
        profile: MaskablePPO.load(
            args.model_root / profile / f"seed-{args.seed}" / "final.zip"
        )
        for profile in PREFERENCE_PROFILES
    }
    report = evaluate_specialists(models, scenarios)
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
