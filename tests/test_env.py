import numpy as np
from gymnasium.utils.env_checker import check_env

from typhoon.baselines import run_baseline
from typhoon.env import TyphoonEvacuationEnv


def _scenario():
    return {
        "resources": {"tug_capacity": 2, "entrance_capacity": {"1": 1}},
        "vessels": [
            {
                "ship_id": "small-first",
                "gross_tonnage": 10_000.0,
                "entrance": "1",
                "ready_hour": 0.0,
                "transit_hour": 2.0,
                "tugs": 1,
                "risk_points": 1,
            },
            {
                "ship_id": "large-risk",
                "gross_tonnage": 180_000.0,
                "entrance": "1",
                "ready_hour": 0.0,
                "transit_hour": 2.0,
                "tugs": 2,
                "risk_points": 5,
            },
        ],
    }


def test_env_passes_gym_checker():
    env = TyphoonEvacuationEnv(_scenario(), randomize=False)
    check_env(env, skip_render_check=True)


def test_mask_blocks_overlapping_entrance_and_deadline():
    env = TyphoonEvacuationEnv(_scenario(), randomize=False)
    env.reset(
        seed=1,
        options={
            "closure_hour": 3.0,
            "tug_capacity": 2,
            "demand_compression": 1.0,
            "jitter": False,
        },
    )
    assert np.array_equal(env.action_masks(), [True, True, True])
    _, _, done, _, _ = env.step(1)
    assert not done
    assert np.array_equal(env.action_masks(), [False, False, True])
    _, _, done, _, info = env.step(env.wait_action)
    assert done
    assert info["kpi"]["safety_violations"] == 0
    assert info["kpi"]["evacuated_risk_points"] == 5


def test_risk_baseline_preserves_more_risk_than_fcfs():
    options = {
        "closure_hour": 3.0,
        "tug_capacity": 2,
        "demand_compression": 1.0,
        "jitter": False,
    }
    env = TyphoonEvacuationEnv(_scenario(), randomize=False)
    fcfs = run_baseline(env, "fcfs", seed=1, options=options)
    risk = run_baseline(env, "risk_aware", seed=1, options=options)
    assert fcfs["evacuated_count"] == risk["evacuated_count"] == 1
    assert risk["evacuated_risk_points"] > fcfs["evacuated_risk_points"]
    assert risk["safety_violations"] == 0


def test_preference_is_observed_and_reward_is_normalized():
    env = TyphoonEvacuationEnv(_scenario(), randomize=False)
    observation, _ = env.reset(
        seed=1,
        options={
            "closure_hour": 3.0,
            "tug_capacity": 2,
            "demand_compression": 1.0,
            "jitter": False,
            "preference": (0.2, 0.3, 0.5),
        },
    )
    assert observation.shape == (2 * 11 + 14,)
    assert np.allclose(observation[-3:], [0.2, 0.3, 0.5])
    assert np.isclose(
        sum(env.dispatch_value(vessel) for vessel in env.vessels), 10.0
    )


def test_resource_release_times_are_part_of_observation():
    env = TyphoonEvacuationEnv(_scenario(), randomize=False)
    env.reset(
        seed=1,
        options={
            "closure_hour": 4.0,
            "tug_capacity": 2,
            "demand_compression": 1.0,
            "jitter": False,
        },
    )
    observation, _, _, _, _ = env.step(1)
    global_features = observation[-14:]
    assert global_features[8] == 0.5
    assert global_features[9] == 0.5
