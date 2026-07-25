"""Phase 10 diagnostic: is the soft-MoE routing monotone in each objective?

Along a simplex edge where one objective's preference weight strictly
increases, the evacuated amount of that objective should not decrease on the
same held-out case. This module sweeps the three edges, counts adjacent
non-monotonic pairs and endpoint inversions, and reports a gate boolean. It can
evaluate the plain argmax decode or the preference-consistent tie-aware decode
(``--tie-eps``), so the effect of the Phase 10 decode fix is measurable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .build_scenario import DEFAULT_DB
from .env import TyphoonEvacuationEnv
from .evaluate_rl import evaluation_cases, run_rl
from .evaluate_soft_moe_grid import DEFAULT_AFTER_MODEL
from .soft_moe import SoftMoEActorCritic
from .train_rl import split_scenarios


# (increasing objective, evacuated KPI, index in the 3-weight vector, fixed
# objective index, fixed weight). Each edge moves mass onto ``rising`` while the
# third objective's weight stays fixed.
EDGES = (
    ("gt", "evacuated_gt", 1, 2, 0.15),  # count -> gt, risk fixed
    ("risk", "evacuated_risk_points", 2, 1, 0.15),  # count -> risk, gt fixed
    ("count", "evacuated_count", 0, 1, 0.15),  # risk -> count, gt fixed
)


def _edge_preferences(
    rising_index: int, fixed_index: int, fixed_weight: float, step: float
) -> list[tuple[float, float, float]]:
    other_index = 3 - rising_index - fixed_index
    budget = 1.0 - fixed_weight
    points = []
    t = 0.0
    while t <= 1.0 + 1e-9:
        weights = [0.0, 0.0, 0.0]
        weights[fixed_index] = fixed_weight
        weights[rising_index] = round(budget * t, 6)
        weights[other_index] = round(budget * (1.0 - t), 6)
        points.append(tuple(weights))
        t += step
    return points


def _paired_base_cases(n_scenarios: int) -> list[tuple[int, dict]]:
    seen: set[str] = set()
    cases = []
    for seed, options in evaluation_cases(n_scenarios):
        if options["case_id"] in seen:
            continue
        seen.add(options["case_id"])
        cases.append((seed, options))
    return cases


def evaluate_monotonicity(
    model,
    scenarios: list[dict],
    *,
    step: float = 0.25,
    tie_eps: float | None = None,
) -> dict:
    if tie_eps is not None:
        model.decode_tie_eps = float(tie_eps)
    env = TyphoonEvacuationEnv(scenarios, randomize=False)
    base_cases = _paired_base_cases(len(scenarios))
    edge_reports = []
    total_pairs = total_violations = total_cases = total_inversions = 0
    for rising, kpi_key, rising_index, fixed_index, fixed_weight in EDGES:
        preferences = _edge_preferences(
            rising_index, fixed_index, fixed_weight, step
        )
        adj_pairs = adj_violations = inversions = flat = 0
        for seed, options in base_cases:
            series = []
            for preference in preferences:
                row = run_rl(
                    model,
                    env,
                    seed=seed,
                    options={
                        **options,
                        "preference_name": f"mono-{rising}",
                        "preference": preference,
                    },
                )
                series.append(float(row[kpi_key]))
            for a in range(len(series) - 1):
                adj_pairs += 1
                if series[a + 1] < series[a] - 1e-6:
                    adj_violations += 1
            if series[0] > series[-1] + 1e-6:
                inversions += 1
            if max(series) - min(series) <= 1e-6:
                flat += 1
        edge_reports.append(
            {
                "edge": f"rising_{rising}",
                "kpi": kpi_key,
                "sweep_points": len(preferences),
                "cases": len(base_cases),
                "adjacent_pairs": adj_pairs,
                "adjacent_violations": adj_violations,
                "adjacent_violation_rate": round(adj_violations / max(adj_pairs, 1), 4),
                "endpoint_inversions": inversions,
                "endpoint_inversion_rate": round(inversions / max(len(base_cases), 1), 4),
                "flat_cases": flat,
            }
        )
        total_pairs += adj_pairs
        total_violations += adj_violations
        total_cases += len(base_cases)
        total_inversions += inversions
    env.close()
    adj_rate = round(total_violations / max(total_pairs, 1), 4)
    inv_rate = round(total_inversions / max(total_cases, 1), 4)
    return {
        "grid_step": step,
        "tie_eps": float(model.decode_tie_eps),
        "edges": edge_reports,
        "overall_adjacent_violation_rate": adj_rate,
        "overall_endpoint_inversion_rate": inv_rate,
        # Phase 10 acceptance targets both below 5%.
        "monotonicity_gate": adj_rate < 0.05 and inv_rate < 0.05,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--model", type=Path, default=DEFAULT_AFTER_MODEL)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument(
        "--tie-eps",
        type=float,
        default=None,
        help="Override decode tie-break epsilon (default: model's stored value).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    _, _, _, test_scenarios = split_scenarios(args.db)
    model = SoftMoEActorCritic.load(args.model, device=args.device)
    result = evaluate_monotonicity(
        model, test_scenarios, step=args.step, tie_eps=args.tie_eps
    )
    output = args.output or args.model.parent / "preference_monotonicity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
