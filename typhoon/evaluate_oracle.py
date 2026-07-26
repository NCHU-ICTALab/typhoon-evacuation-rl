"""Strict optimization ceiling for the deterministic evacuation problem.

The weighted utility depends only on *which* vessels are evacuated (each vessel
contributes a fixed non-negative ``v_i = pref . [1/N, gt_i/GT, risk_i/risk]``),
so the optimum is the maximum-weight *schedulable* subset under the same hard
mask the env enforces (readiness, closure deadline, tug and entrance capacity
over time). This module finds it by branch-and-bound over a pure-array mirror of
the env dynamics — no env clone — seeded with the value-density incumbent and
pruned with an admissible time-feasibility relaxation. Dispatching A then B
reaches the same state as B then A, so a visited-state set collapses the
permutations.

Every value the search returns is an achievable schedule, hence a valid lower
bound on the true optimum: wherever it exceeds a rule policy, headroom above that
rule is proven even if global optimality is not (``proven`` records which). On
tight-closure cases the search proves optimality outright.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .baselines import run_baseline
from .build_scenario import DEFAULT_DB
from .env import TyphoonEvacuationEnv
from .evaluate_rl import PREFERENCE_PROFILES, evaluation_cases, run_rl
from .soft_moe import SoftMoEActorCritic
from .train_rl import split_scenarios

EPS = 1e-9


def _vessel_values(env: TyphoonEvacuationEnv) -> np.ndarray:
    preference = env.preference
    return np.array(
        [
            float(preference[0]) / env.n_vessels
            + float(preference[1]) * float(v["gross_tonnage"]) / env.total_gt
            + float(preference[2]) * int(v["risk_points"]) / env.total_risk
            for v in env.vessels
        ]
    )


def optimal_utility(
    scenarios: list[dict],
    seed: int,
    options: dict,
    *,
    node_cap: int = 1_000_000,
) -> dict:
    """Return the best schedulable-subset utility, with a proven-optimal flag."""
    env = TyphoonEvacuationEnv(scenarios, randomize=False)
    env.reset(seed=seed, options=options)
    n = env.n_vessels
    ready = np.array([float(v["ready_hour"]) for v in env.vessels])
    transit = np.array([float(v["transit_hour"]) for v in env.vessels])
    tugs = np.array([int(v["tugs"]) for v in env.vessels])
    entrance = np.array(
        [0 if str(v["entrance"]) == "1" else 1 for v in env.vessels]
    )
    values = _vessel_values(env)
    closure = env.closure_hour
    tug_capacity = env.tug_capacity
    entrance_capacity = [
        int(env.entrance_capacity.get("1", 1)),
        int(env.entrance_capacity.get("2", 1)),
    ]

    incumbent = run_baseline(env, "value_density", seed=seed, options=options)
    best = float(incumbent["weighted_utility"])
    visited: set = set()
    counter = {"nodes": 0, "proven": True}
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 200_000))

    def next_event(now: float, active: tuple, scheduled: int):
        candidates = [f for (f, _, _) in active if now + EPS < f <= closure + EPS]
        for i in range(n):
            if not (scheduled >> i) & 1 and now + EPS < ready[i] < closure - EPS:
                candidates.append(ready[i])
        return min(candidates) if candidates else None

    def search(now: float, active: tuple, scheduled: int, partial: float):
        nonlocal best
        if counter["nodes"] > node_cap:
            counter["proven"] = False
            return
        counter["nodes"] += 1
        if partial > best:
            best = partial
        active = tuple(sorted((f, e, t) for (f, e, t) in active if f > now + EPS))
        key = (scheduled, round(now, 4), active)
        if key in visited:
            return
        visited.add(key)
        upper = partial
        for i in range(n):
            if not (scheduled >> i) & 1 and max(now, ready[i]) + transit[i] <= closure + EPS:
                upper += values[i]
        if upper <= best + 1e-12:
            return
        used_tugs = sum(t for (_, _, t) in active)
        used_entrances = [0, 0]
        for (_, e, _) in active:
            used_entrances[e] += 1
        feasible = [
            i
            for i in range(n)
            if not (scheduled >> i) & 1
            and ready[i] <= now + EPS
            and now + transit[i] <= closure + EPS
            and used_tugs + tugs[i] <= tug_capacity
            and used_entrances[entrance[i]] < entrance_capacity[entrance[i]]
        ]
        feasible.sort(key=lambda i: values[i], reverse=True)
        for i in feasible:
            child = tuple(
                sorted(active + ((now + transit[i], entrance[i], tugs[i]),))
            )
            search(now, child, scheduled | (1 << i), partial + values[i])
        event = next_event(now, active, scheduled)
        if event is not None:
            search(event, active, scheduled, partial)

    search(float(min(ready)), tuple(), 0, 0.0)
    sys.setrecursionlimit(limit)
    return {
        "optimal_utility": round(best, 6),
        "proven_optimal": counter["proven"],
        "nodes": counter["nodes"],
    }


def _paired_base_cases(n_scenarios: int) -> list[tuple[int, dict]]:
    seen: set = set()
    cases = []
    for seed, options in evaluation_cases(n_scenarios):
        if options["case_id"] in seen:
            continue
        seen.add(options["case_id"])
        cases.append((seed, options))
    return cases


def evaluate_oracle_gap(
    model,
    scenarios: list[dict],
    *,
    node_cap: int = 1_000_000,
) -> dict:
    env = TyphoonEvacuationEnv(scenarios, randomize=False)
    base_cases = _paired_base_cases(len(scenarios))
    rule_policies = ("fcfs", "risk_aware", "value_density")
    per_preference = {}
    for pname, preference in PREFERENCE_PROFILES.items():
        rows = []
        for seed, options in base_cases:
            case = {**options, "preference_name": pname, "preference": preference}
            rl = run_rl(model, env, seed=seed, options=case)["weighted_utility"]
            rules = {
                policy: run_baseline(env, policy, seed=seed, options=case)[
                    "weighted_utility"
                ]
                for policy in rule_policies
            }
            best_rule = max(rules.values())
            oracle = optimal_utility(scenarios, seed, case, node_cap=node_cap)
            rows.append(
                {
                    "case_id": options["case_id"],
                    "rl": rl,
                    "value_density": rules["value_density"],
                    "best_rule": best_rule,
                    "oracle": oracle["optimal_utility"],
                    "proven": oracle["proven_optimal"],
                }
            )
        proven = [r for r in rows if r["proven"]]
        headroom = [r for r in rows if r["oracle"] - r["value_density"] > 1e-6]

        def mean(items, key):
            return round(float(np.mean([r[key] for r in items])), 6) if items else None

        per_preference[pname] = {
            "cases": len(rows),
            "proven_cases": len(proven),
            "cases_with_headroom_above_value_density": len(headroom),
            "mean_rl": mean(rows, "rl"),
            "mean_value_density": mean(rows, "value_density"),
            "mean_best_rule": mean(rows, "best_rule"),
            "mean_oracle_lower_bound": mean(rows, "oracle"),
            "mean_rl_gap_vs_value_density": round(
                mean(rows, "rl") - mean(rows, "value_density"), 6
            ),
            "mean_rl_regret_vs_oracle": round(
                mean(rows, "oracle") - mean(rows, "rl"), 6
            ),
            "mean_value_density_regret_vs_oracle_proven": (
                round(mean(proven, "oracle") - mean(proven, "value_density"), 6)
                if proven
                else None
            ),
            "rl_beats_or_ties_value_density_cases": sum(
                r["rl"] >= r["value_density"] - 1e-6 for r in rows
            ),
        }
    return {
        "node_cap": node_cap,
        "paired_base_cases": len(base_cases),
        "per_preference": per_preference,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent
        / "models"
        / "soft_moe_router_ppo"
        / "seed-42"
        / "final.pt",
    )
    parser.add_argument("--node-cap", type=int, default=1_000_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    _, _, _, test_scenarios = split_scenarios(args.db)
    model = SoftMoEActorCritic.load(args.model, device=args.device)
    result = evaluate_oracle_gap(model, test_scenarios, node_cap=args.node_cap)
    output = args.output or args.model.parent / "oracle_gap.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
