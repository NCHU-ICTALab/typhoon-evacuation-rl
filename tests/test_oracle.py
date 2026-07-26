import numpy as np

from typhoon.baselines import run_baseline
from typhoon.env import TyphoonEvacuationEnv
from typhoon.evaluate_oracle import optimal_utility


def _scenario():
    return {
        "metadata": {"query_date": "oracle-test"},
        "resources": {"tug_capacity": 3, "entrance_capacity": {"1": 1, "2": 1}},
        "vessels": [
            {
                "ship_id": f"ship-{i}",
                "name": f"SHIP {i}",
                "gross_tonnage": 5_000.0 + i * 4_000.0,
                "entrance": str(i % 2 + 1),
                "ready_hour": 0.0 if i < 4 else 1.0,
                "transit_hour": 0.5,
                "tugs": 1 + (i % 2),
                "risk_points": i % 5 + 1,
            }
            for i in range(6)
        ],
    }


def _options(preference):
    return {
        "preference": preference,
        "closure_hour": 3.0,
        "tug_capacity": 3,
        "demand_compression": 1.0,
        "jitter": False,
        "deterministic": True,
    }


def test_oracle_dominates_every_rule_policy_and_proves_small_case():
    scenarios = [_scenario()]
    env = TyphoonEvacuationEnv(scenarios, randomize=False)
    for preference in ((1 / 3, 1 / 3, 1 / 3), (0.15, 0.70, 0.15), (0.15, 0.15, 0.70)):
        options = _options(preference)
        oracle = optimal_utility(scenarios, seed=0, options=options)
        rule_utils = [
            run_baseline(env, policy, seed=0, options=options)["weighted_utility"]
            for policy in ("fcfs", "risk_aware", "value_density")
        ]
        assert oracle["proven_optimal"] is True
        assert oracle["optimal_utility"] >= max(rule_utils) - 1e-9


def test_oracle_matches_bruteforce_on_tiny_instance():
    # With three same-entrance vessels and one tug/entrance slot, only one job runs
    # at a time; the optimum is a plain enumeration of feasible dispatch subsets.
    scenarios = [
        {
            "metadata": {"query_date": "tiny"},
            "resources": {"tug_capacity": 1, "entrance_capacity": {"1": 1, "2": 1}},
            "vessels": [
                {
                    "ship_id": f"s{i}",
                    "name": f"S{i}",
                    "gross_tonnage": gt,
                    "entrance": "1",
                    "ready_hour": 0.0,
                    "transit_hour": 0.4,
                    "tugs": 1,
                    "risk_points": 1,
                }
                for i, gt in enumerate((10_000.0, 20_000.0, 30_000.0))
            ],
        }
    ]
    options = {
        "preference": (0.0, 1.0, 0.0),  # pure GT
        "closure_hour": 1.0,
        "tug_capacity": 1,
        "demand_compression": 1.0,
        "jitter": False,
        "deterministic": True,
    }
    oracle = optimal_utility(scenarios, seed=0, options=options)
    # closure 1.0 / transit 0.4 with one slot => at most two vessels; the GT
    # optimum evacuates the two largest (20k + 30k of 60k total) => 0.8333.
    assert oracle["proven_optimal"] is True
    assert oracle["optimal_utility"] == round(50_000.0 / 60_000.0, 6)
