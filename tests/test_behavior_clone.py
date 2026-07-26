import numpy as np

from typhoon.decomposed_ppo import ROUTING_PREFERENCES
from typhoon.env import TyphoonEvacuationEnv
from typhoon.soft_moe import SoftMoEActorCritic
from typhoon.train_behavior_clone import collect_teacher_dataset, train_expert
from typhoon.train_dagger import collect_student_dataset


def _scenario():
    return {
        "metadata": {"query_date": "bc-test"},
        "resources": {"tug_capacity": 8, "entrance_capacity": {"1": 1, "2": 1}},
        "vessels": [
            {
                "ship_id": f"ship-{i}",
                "name": f"SHIP {i}",
                "gross_tonnage": 10_000.0 + i * 1_000.0,
                "entrance": str(i % 2 + 1),
                "ready_hour": i * 0.1,
                "transit_hour": 0.5,
                "tugs": 1,
                "risk_points": i % 5 + 1,
            }
            for i in range(30)
        ],
    }


def test_collect_teacher_dataset_shapes():
    scenarios = [_scenario()]
    data = collect_teacher_dataset(
        scenarios, (0.15, 0.70, 0.15), episodes=5, seed=0
    )
    n = len(data["actions"])
    assert n > 0
    assert data["observations"].shape == (n, 344)
    assert data["action_masks"].shape == (n, 31)
    # every recorded teacher action is feasible under its own mask
    assert all(data["action_masks"][i, data["actions"][i]] for i in range(n))


def test_train_expert_learns_the_teacher():
    scenarios = [_scenario()]
    env = TyphoonEvacuationEnv(scenarios, randomize=False)
    data = collect_teacher_dataset(
        scenarios, (0.15, 0.70, 0.15), episodes=40, seed=1
    )
    expert, metrics = train_expert(
        env.observation_space,
        int(env.action_space.n),
        data,
        epochs=8,
        batch_size=128,
        learning_rate=1e-3,
        seed=1,
    )
    assert metrics["final_accuracy"] > 0.6  # clearly imitating, not chance


def test_collect_student_dataset_labels_with_teacher():
    scenarios = [_scenario()]
    env = TyphoonEvacuationEnv(scenarios, randomize=False)
    model = SoftMoEActorCritic(env.observation_space, int(env.action_space.n))
    data = collect_student_dataset(
        model, scenarios, ROUTING_PREFERENCES[2], episodes=3, seed=2
    )
    n = len(data["actions"])
    assert n > 0
    # labels are the teacher's feasible actions on student-visited states
    assert all(data["action_masks"][i, data["actions"][i]] for i in range(n))
