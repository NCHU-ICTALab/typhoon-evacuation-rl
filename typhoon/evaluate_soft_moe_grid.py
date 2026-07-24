"""Paired continuous-preference evaluation for Phase 9A and Phase 9B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .build_scenario import DEFAULT_DB
from .env import TyphoonEvacuationEnv
from .evaluate_rl import evaluation_cases, run_rl
from .soft_moe import SoftMoEActorCritic
from .train_rl import split_scenarios


ROOT = Path(__file__).resolve().parent
DEFAULT_BEFORE_MODEL = (
    ROOT / "models" / "soft_moe_router_distilled" / "seed-42" / "final.pt"
)
DEFAULT_AFTER_MODEL = (
    ROOT / "models" / "soft_moe_router_ppo" / "seed-42" / "final.pt"
)


def simplex_grid(step: float = 0.25) -> list[tuple[float, float, float]]:
    if step <= 0.0 or step > 1.0:
        raise ValueError("step must be in (0, 1]")
    divisions = round(1.0 / step)
    if not np.isclose(divisions * step, 1.0):
        raise ValueError("step must evenly divide 1.0")
    return [
        (count / divisions, gt / divisions, risk / divisions)
        for count in range(divisions + 1)
        for gt in range(divisions - count + 1)
        for risk in [divisions - count - gt]
    ]


def _paired_base_cases(n_scenarios: int) -> list[tuple[int, dict]]:
    result = []
    seen = set()
    for seed, options in evaluation_cases(n_scenarios):
        if options["case_id"] in seen:
            continue
        seen.add(options["case_id"])
        result.append((seed, options))
    return result


def evaluate_continuous_grid(
    before_model,
    after_model,
    scenarios: list[dict] | dict,
    *,
    step: float = 0.25,
) -> dict:
    n_scenarios = len(scenarios) if isinstance(scenarios, list) else 1
    base_cases = _paired_base_cases(n_scenarios)
    preferences = simplex_grid(step)
    before_env = TyphoonEvacuationEnv(scenarios, randomize=False)
    after_env = TyphoonEvacuationEnv(scenarios, randomize=False)
    point_results = []
    total_action_differences = 0
    total_schedule_differences = 0
    total_safety_violations = 0
    total_rejected_actions = 0

    for index, preference in enumerate(preferences):
        label = "grid-" + "-".join(f"{value:.2f}" for value in preference)
        before_rows = []
        after_rows = []
        for seed, base_options in base_cases:
            options = {
                **base_options,
                "preference_name": label,
                "preference": preference,
            }
            before = run_rl(
                before_model, before_env, seed=seed, options=options
            )
            after = run_rl(
                after_model, after_env, seed=seed, options=options
            )
            before_rows.append(before)
            after_rows.append(after)
            total_action_differences += int(
                before["action_trace"] != after["action_trace"]
            )
            total_schedule_differences += int(
                before["schedule_signature"] != after["schedule_signature"]
            )
            total_safety_violations += int(before["safety_violations"])
            total_safety_violations += int(after["safety_violations"])
            total_rejected_actions += int(before["rejected_actions"])
            total_rejected_actions += int(after["rejected_actions"])

        before_utility = float(
            np.mean([row["weighted_utility"] for row in before_rows])
        )
        after_utility = float(
            np.mean([row["weighted_utility"] for row in after_rows])
        )
        point_results.append(
            {
                "index": index,
                "preference": list(preference),
                "phase9a_utility": round(before_utility, 6),
                "phase9b_utility": round(after_utility, 6),
                "utility_delta": round(after_utility - before_utility, 6),
                "action_difference_cases": sum(
                    before["action_trace"] != after["action_trace"]
                    for before, after in zip(before_rows, after_rows)
                ),
                "schedule_difference_cases": sum(
                    before["schedule_signature"]
                    != after["schedule_signature"]
                    for before, after in zip(before_rows, after_rows)
                ),
            }
        )

    deltas = [item["utility_delta"] for item in point_results]
    before_env.close()
    after_env.close()
    tolerance = 1e-6
    mean_delta = round(float(np.mean(deltas)), 6)
    min_delta = round(float(np.min(deltas)), 6)
    return {
        "grid_step": step,
        "grid_points": len(preferences),
        "paired_base_cases": len(base_cases),
        "episodes_per_model": len(preferences) * len(base_cases),
        "points_improved": sum(delta > tolerance for delta in deltas),
        "points_tied": sum(abs(delta) <= tolerance for delta in deltas),
        "points_worse": sum(delta < -tolerance for delta in deltas),
        "mean_utility_delta": mean_delta,
        "min_utility_delta": min_delta,
        "max_utility_delta": round(float(np.max(deltas)), 6),
        "action_difference_cases": total_action_differences,
        "schedule_difference_cases": total_schedule_differences,
        "safety_violations": total_safety_violations,
        "rejected_actions": total_rejected_actions,
        "continuous_grid_gate": (
            mean_delta >= 0.0
            and min_delta >= -0.002
            and total_safety_violations == 0
            and total_rejected_actions == 0
        ),
        "point_results": point_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--before-model", type=Path, default=DEFAULT_BEFORE_MODEL)
    parser.add_argument("--after-model", type=Path, default=DEFAULT_AFTER_MODEL)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    _, _, _, test_scenarios = split_scenarios(args.db)
    before_model = SoftMoEActorCritic.load(
        args.before_model, device=args.device
    )
    after_model = SoftMoEActorCritic.load(
        args.after_model, device=args.device
    )
    result = evaluate_continuous_grid(
        before_model,
        after_model,
        test_scenarios,
        step=args.step,
    )
    output = args.output or args.after_model.parent / "continuous_grid.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in result.items() if key != "point_results"}
    comparison_path = args.after_model.parent / "comparison.json"
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        comparison["continuous_grid"] = summary
        comparison["accepted_for_frontend"] = bool(
            comparison.get("accepted_for_frontend", False)
            and result["continuous_grid_gate"]
        )
        comparison_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
