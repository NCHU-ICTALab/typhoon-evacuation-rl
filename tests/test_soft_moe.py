import numpy as np
import torch
from gymnasium import spaces
import pytest

from typhoon.soft_moe import SoftMoEActorCritic
from typhoon.train_soft_moe import soft_routing_targets


def _space():
    return spaces.Box(low=-2.0, high=5.0, shape=(344,), dtype=np.float32)


def _observation(preference=(1 / 3, 1 / 3, 1 / 3)):
    observation = np.zeros((1, 344), dtype=np.float32)
    observation[:, :330] = 0.25
    observation[:, -3:] = preference
    return observation


def test_soft_moe_freezes_experts_and_normalizes_router_weights():
    model = SoftMoEActorCritic(_space(), 31)
    weights = model.predict_routing(_observation())

    assert weights.shape == (1, 4)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert all(
        not parameter.requires_grad
        for expert in model.experts
        for parameter in expert.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.policy_parameters()
    )


def test_soft_router_selects_the_requested_expert_policy():
    model = SoftMoEActorCritic(_space(), 31)
    expected_actions = [3, 7, 11, 15]
    with torch.no_grad():
        for expert, action in zip(model.experts, expected_actions):
            output = expert.actor[-1]
            output.weight.zero_()
            output.bias.fill_(-20.0)
            output.bias[action] = 20.0
        for parameter in model.preference_router.parameters():
            parameter.zero_()
        model.preference_router[-1].bias.fill_(-20.0)
        model.preference_router[-1].bias[2] = 20.0

    action, _ = model.predict(
        _observation((0.15, 0.70, 0.15))[0],
        deterministic=True,
        action_masks=np.ones(31, dtype=bool),
    )
    routing = model.predict_routing(_observation((0.15, 0.70, 0.15)))[0]

    assert action == expected_actions[2]
    assert int(np.argmax(routing)) == 2


def test_soft_moe_never_selects_a_masked_action():
    model = SoftMoEActorCritic(_space(), 31)
    with torch.no_grad():
        for expert in model.experts:
            output = expert.actor[-1]
            output.weight.zero_()
            output.bias.fill_(-20.0)
            output.bias[5] = 20.0

    mask = np.ones(31, dtype=bool)
    mask[5] = False
    action, _ = model.predict(
        _observation()[0], deterministic=True, action_masks=mask
    )
    assert action != 5


def test_soft_moe_rejects_an_empty_action_mask():
    model = SoftMoEActorCritic(_space(), 31)
    with pytest.raises(ValueError, match="at least one valid action"):
        model.predict(
            _observation()[0],
            deterministic=True,
            action_masks=np.zeros(31, dtype=bool),
        )


def test_soft_moe_checkpoint_round_trip(tmp_path):
    model = SoftMoEActorCritic(_space(), 31)
    with torch.no_grad():
        model.preference_router[-1].bias.copy_(
            torch.tensor([-1.0, 0.0, 1.0, 2.0])
        )
    path = model.save(tmp_path / "soft-moe.pt", metadata={"phase": 9})
    loaded = SoftMoEActorCritic.load(path)

    before = model.predict_routing(_observation())
    after = loaded.predict_routing(_observation())
    assert np.allclose(before, after)
    assert all(
        not parameter.requires_grad
        for expert in loaded.experts
        for parameter in expert.parameters()
    )


def test_soft_routing_targets_are_normalized_and_centroid_aligned():
    preferences = torch.tensor(
        [
            [0.70, 0.15, 0.15],
            [1 / 3, 1 / 3, 1 / 3],
            [0.15, 0.70, 0.15],
            [0.15, 0.15, 0.70],
        ],
        dtype=torch.float32,
    )
    targets = soft_routing_targets(preferences, temperature=0.02)

    assert torch.allclose(targets.sum(dim=-1), torch.ones(4))
    assert targets.argmax(dim=-1).tolist() == [0, 1, 2, 3]
