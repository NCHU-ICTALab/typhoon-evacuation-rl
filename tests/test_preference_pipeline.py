from collections import defaultdict

import numpy as np

from typhoon.build_scenario import _synthetic_risk_profile
from typhoon.env import TyphoonEvacuationEnv, VectorRewardWrapper
from typhoon.evaluate_rl import PREFERENCE_PROFILES, evaluation_cases


def _scenario():
    return {
        "metadata": {"query_date": "preference-test"},
        "resources": {"tug_capacity": 2, "entrance_capacity": {"1": 1}},
        "vessels": [
            {
                "ship_id": "count-choice",
                "gross_tonnage": 10_000.0,
                "entrance": "1",
                "ready_hour": 0.0,
                "transit_hour": 1.0,
                "tugs": 1,
                "risk_points": 5,
            },
            {
                "ship_id": "gt-choice",
                "gross_tonnage": 180_000.0,
                "entrance": "1",
                "ready_hour": 0.0,
                "transit_hour": 1.0,
                "tugs": 2,
                "risk_points": 1,
            },
        ],
    }


def _options():
    return {
        "closure_hour": 2.0,
        "tug_capacity": 2,
        "demand_compression": 1.0,
        "jitter": False,
    }


def test_scalar_reward_is_preference_dot_vector_reward():
    preference = (0.2, 0.3, 0.5)
    env = TyphoonEvacuationEnv(
        _scenario(), randomize=False, fixed_preference=preference
    )
    env.reset(seed=7, options=_options())
    _, scalar_reward, _, _, info = env.step(0)
    vector_reward = np.asarray(info["objective_reward"])
    assert vector_reward.shape == (3,)
    assert np.isclose(scalar_reward, np.dot(preference, vector_reward))


def test_vector_wrapper_exposes_three_objectives():
    base = TyphoonEvacuationEnv(_scenario(), randomize=False)
    env = VectorRewardWrapper(base)
    env.reset(seed=7, options=_options())
    _, reward, _, _, _ = env.step(0)
    assert reward.shape == (3,)
    assert env.reward_space.contains(reward)


def test_fixed_profile_overrides_random_dirichlet_sampling():
    preference = PREFERENCE_PROFILES["risk"]
    env = TyphoonEvacuationEnv(
        _scenario(), randomize=True, fixed_preference=preference
    )
    observation, _ = env.reset(seed=99, options=_options())
    assert np.allclose(observation[-3:], preference)


def test_evaluation_uses_identical_seed_inside_each_paired_case():
    groups = defaultdict(list)
    for seed, options in evaluation_cases(2):
        groups[options["case_id"]].append((seed, options["preference_name"]))
    assert len(groups) == 24
    for values in groups.values():
        assert len(values) == len(PREFERENCE_PROFILES)
        assert len({seed for seed, _ in values}) == 1
        assert {name for _, name in values} == set(PREFERENCE_PROFILES)


def test_synthetic_risk_is_reproducible_and_not_a_gt_function():
    first = _synthetic_risk_profile("S1", "V1", "B01", "1")
    second = _synthetic_risk_profile("S1", "V1", "B01", "1")
    assert first == second
    profiles = {
        _synthetic_risk_profile(f"S{i}", "V1", "B01", "1")[0]
        for i in range(30)
    }
    assert len(profiles) >= 3


def test_specialist_evaluator_uses_paired_cases():
    from typhoon.evaluate_specialists import evaluate_specialists

    class FirstFeasibleModel:
        def predict(self, observation, *, deterministic, action_masks):
            feasible = np.flatnonzero(action_masks[:-1])
            return (int(feasible[0]) if len(feasible) else len(action_masks) - 1), None

    models = {profile: FirstFeasibleModel() for profile in PREFERENCE_PROFILES}
    report = evaluate_specialists(models, [_scenario()])
    assert report["paired_cases_per_profile"] == 12
    assert report["cases_with_multiple_schedules"] == 0
    assert all(
        metrics["safety_violations"] == 0
        for metrics in report["summary"].values()
    )


def test_phase1_profiles_receive_same_scenario_stream():
    from typhoon.train_conditioned_v2 import PairedProfileTrainingEnv

    count_env = PairedProfileTrainingEnv(
        [_scenario()], preference=PREFERENCE_PROFILES["count"], base_seed=123
    )
    risk_env = PairedProfileTrainingEnv(
        [_scenario()], preference=PREFERENCE_PROFILES["risk"], base_seed=123
    )
    count_observation, _ = count_env.reset()
    risk_observation, _ = risk_env.reset()
    assert np.allclose(count_observation[:-3], risk_observation[:-3])
    assert np.allclose(count_observation[-3:], PREFERENCE_PROFILES["count"])
    assert np.allclose(risk_observation[-3:], PREFERENCE_PROFILES["risk"])
    assert count_env.closure_hour == risk_env.closure_hour
    assert count_env.tug_capacity == risk_env.tug_capacity


def test_fixed_profile_vec_env_contains_only_requested_preference():
    from typhoon.train_conditioned_v2 import make_profile_envs

    env = make_profile_envs(
        [_scenario()],
        123,
        [PREFERENCE_PROFILES["gt"]],
    )
    observation = env.reset()
    assert env.num_envs == 1
    assert np.allclose(observation[0, -3:], PREFERENCE_PROFILES["gt"])
    env.close()


def test_preference_gating_changes_ship_representation():
    import torch

    from typhoon.policy import PreferenceGatedExtractor

    env = TyphoonEvacuationEnv(_scenario(), randomize=False)
    # The production extractor is configured for 30 vessels; use a matching
    # synthetic observation space without requiring the historical database.
    space = type(env.observation_space)(
        low=-2.0, high=5.0, shape=(344,), dtype=np.float32
    )
    extractor = PreferenceGatedExtractor(space)
    observations = torch.zeros((2, 344), dtype=torch.float32)
    observations[:, :330] = 0.5
    observations[0, -3:] = torch.tensor(PREFERENCE_PROFILES["count"])
    observations[1, -3:] = torch.tensor(PREFERENCE_PROFILES["risk"])
    features = extractor(observations)
    assert features.shape == (2, 256)
    assert not torch.allclose(features[0], features[1])
