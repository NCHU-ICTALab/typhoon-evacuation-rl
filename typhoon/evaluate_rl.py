"""Evaluate conditioned or specialist policies with paired preference cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from .baselines import run_baseline
from .build_scenario import DEFAULT_DB, available_scenario_dates, build_scenario_pool
from .env import TyphoonEvacuationEnv


METRICS = (
    "episode_return",
    "safety_violations",
    "rejected_actions",
    "evacuated_count",
    "evacuated_gt",
    "evacuated_risk_points",
    "remaining_count",
    "remaining_gt",
    "remaining_risk_points",
    "tug_utilization",
    "entrance_utilization",
    "weighted_utility",
)

PREFERENCE_PROFILES = {
    "count": (0.70, 0.15, 0.15),
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "gt": (0.15, 0.70, 0.15),
    "risk": (0.15, 0.15, 0.70),
}


def evaluation_cases(n_scenarios: int = 1) -> list[tuple[int, dict]]:
    """Use exactly the same random seed for every preference in a base case."""
    cases: list[tuple[int, dict]] = []
    case_index = 0
    for scenario_index in range(n_scenarios):
        for closure in (4.0, 6.0, 8.0):
            for tugs in (4, 8):
                for compression in (2.0, 3.5):
                    seed = 10_000 + case_index
                    case_id = (
                        f"s{scenario_index}-c{closure:g}-t{tugs}-d{compression:g}"
                    )
                    for preference_name, preference in PREFERENCE_PROFILES.items():
                        cases.append(
                            (
                                seed,
                                {
                                    "case_id": case_id,
                                    "scenario_index": scenario_index,
                                    "preference_name": preference_name,
                                    "preference": preference,
                                    "closure_hour": closure,
                                    "tug_capacity": tugs,
                                    "demand_compression": compression,
                                    "jitter": True,
                                },
                            )
                        )
                    case_index += 1
    return cases


def run_rl(model, env, *, seed: int, options: dict) -> dict:
    obs, _ = env.reset(seed=seed, options=options)
    terminated = truncated = False
    episode_return = 0.0
    objective_return = np.zeros(3, dtype=float)
    action_trace: list[int] = []
    last_info = {}
    while not (terminated or truncated):
        action, _ = model.predict(
            obs, deterministic=True, action_masks=env.action_masks()
        )
        action = int(action)
        action_trace.append(action)
        obs, reward, terminated, truncated, last_info = env.step(action)
        episode_return += float(reward)
        objective_return += np.asarray(last_info["objective_reward"], dtype=float)
    return {
        "policy": "rl",
        "case_id": options["case_id"],
        "preference_name": options["preference_name"],
        "seed": seed,
        "episode_return": round(episode_return, 6),
        "objective_return": np.round(objective_return, 6).tolist(),
        "action_trace": action_trace,
        "schedule_signature": [job["ship_id"] for job in env.schedule],
        **last_info["kpi"],
    }


def _aggregate(rows: list[dict]) -> dict:
    return {
        metric: round(float(np.mean([row[metric] for row in rows])), 4)
        for metric in METRICS
    }


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def _utility(row: dict, preference: tuple[float, float, float]) -> float:
    total_count = row["evacuated_count"] + row["remaining_count"]
    total_gt = row["evacuated_gt"] + row["remaining_gt"]
    total_risk = row["evacuated_risk_points"] + row["remaining_risk_points"]
    objectives = np.asarray(
        [
            row["evacuated_count"] / max(total_count, 1),
            row["evacuated_gt"] / max(total_gt, 1e-9),
            row["evacuated_risk_points"] / max(total_risk, 1),
        ],
        dtype=float,
    )
    return float(np.dot(np.asarray(preference, dtype=float), objectives))


def _paired_diagnostics(rl_rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rl_rows:
        grouped[row["case_id"]].append(row)

    trace_counts = Counter()
    schedule_counts = Counter()
    for group in grouped.values():
        trace_counts[len({tuple(row["action_trace"]) for row in group})] += 1
        schedule_counts[
            len({tuple(row["schedule_signature"]) for row in group})
        ] += 1

    cross_utility: dict[str, dict[str, float]] = {}
    for eval_name, eval_weights in PREFERENCE_PROFILES.items():
        cross_utility[eval_name] = {}
        for producing_name in PREFERENCE_PROFILES:
            values = [
                _utility(row, eval_weights)
                for row in rl_rows
                if row["preference_name"] == producing_name
            ]
            cross_utility[eval_name][producing_name] = round(
                float(np.mean(values)), 6
            )

    return {
        "paired_base_cases": len(grouped),
        "action_trace_unique_count_histogram": {
            str(key): value for key, value in sorted(trace_counts.items())
        },
        "schedule_unique_count_histogram": {
            str(key): value for key, value in sorted(schedule_counts.items())
        },
        "cases_with_preference_action_difference": sum(
            count for unique, count in trace_counts.items() if unique > 1
        ),
        "cross_utility_matrix": cross_utility,
    }


def evaluate_model(
    model,
    scenario: dict | list[dict],
    *,
    output_path: str | Path | None = None,
) -> dict:
    env = TyphoonEvacuationEnv(scenario, randomize=False)
    rows = {"rl": [], "fcfs": [], "risk_aware": [], "value_density": []}
    n_scenarios = len(scenario) if isinstance(scenario, list) else 1
    cases = evaluation_cases(n_scenarios)
    for seed, options in cases:
        rows["rl"].append(run_rl(model, env, seed=seed, options=options))
        for policy in ("fcfs", "risk_aware", "value_density"):
            baseline = run_baseline(env, policy, seed=seed, options=options)
            baseline["case_id"] = options["case_id"]
            baseline["preference_name"] = options["preference_name"]
            baseline["seed"] = seed
            rows[policy].append(baseline)

    summary = {policy: _aggregate(policy_rows) for policy, policy_rows in rows.items()}
    summary_by_preference = {}
    for preference_name in PREFERENCE_PROFILES:
        summary_by_preference[preference_name] = {}
        for policy, policy_rows in rows.items():
            selected = [
                row
                for row in policy_rows
                if row["preference_name"] == preference_name
            ]
            metrics = _aggregate(selected)
            metrics["weighted_utility_ci95"] = round(
                _ci95([row["weighted_utility"] for row in selected]), 4
            )
            summary_by_preference[preference_name][policy] = metrics

    result = {
        "evaluation": {
            "episodes_per_policy": len(cases),
            "paired_base_cases": len(cases) // len(PREFERENCE_PROFILES),
            "offline_dates": n_scenarios,
            "closure_hours": [4.0, 6.0, 8.0],
            "tug_capacities": [4, 8],
            "demand_compressions": [2.0, 3.5],
            "preference_profiles": PREFERENCE_PROFILES,
            "weather_and_resources": "synthetic_training_only",
            "vessels": "offline_public_source_snapshot",
        },
        "summary": summary,
        "summary_by_preference": summary_by_preference,
        "paired_preference_diagnostics": _paired_diagnostics(rows["rl"]),
        "delta_rl_minus_fcfs": {
            metric: round(summary["rl"][metric] - summary["fcfs"][metric], 4)
            for metric in METRICS
        },
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "final.zip",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dates = available_scenario_dates(args.db, min_vessels=30)
    test_dates = dates[-5:] if len(dates) >= 5 else dates
    scenario = build_scenario_pool(db_path=args.db, dates=test_dates, max_vessels=30)
    model = MaskablePPO.load(args.model)
    report_path = args.output or (
        Path(__file__).resolve().parent / "models" / "evaluation.json"
    )
    result = evaluate_model(model, scenario, output_path=report_path)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(
        json.dumps(
            result["paired_preference_diagnostics"], ensure_ascii=False, indent=2
        )
    )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
