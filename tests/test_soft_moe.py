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


def _two_way_tie_model():
    """Experts that put equal mass on dispatch actions 0 and 1."""
    model = SoftMoEActorCritic(_space(), 31)
    with torch.no_grad():
        for expert in model.experts:
            output = expert.actor[-1]
            output.weight.zero_()
            output.bias.fill_(-20.0)
            output.bias[0] = 20.0
            output.bias[1] = 20.0
    return model


def test_tie_aware_decode_breaks_ties_by_preference():
    model = _two_way_tie_model()
    observation = _observation((0.15, 0.70, 0.15))
    observation[:, :330] = 0.0
    observation[0, 0 * 11 + 0] = 0.1  # vessel 0 has the lower GT feature
    observation[0, 1 * 11 + 0] = 0.9  # vessel 1 has the higher GT feature
    mask = np.zeros(31, dtype=bool)
    mask[0] = True
    mask[1] = True

    plain, _ = model.predict(
        observation[0], deterministic=True, action_masks=mask, tie_eps=0.0
    )
    tie_aware, _ = model.predict(
        observation[0], deterministic=True, action_masks=mask, tie_eps=0.01
    )

    assert plain == 0  # argmax resolves an exact tie to the lower index
    assert tie_aware == 1  # tie-break prefers the higher-GT vessel under a GT weight


def test_tie_aware_decode_keeps_a_confident_argmax():
    model = SoftMoEActorCritic(_space(), 31)
    with torch.no_grad():
        for expert in model.experts:
            output = expert.actor[-1]
            output.weight.zero_()
            output.bias.fill_(-20.0)
            output.bias[2] = 20.0  # every expert is confident about action 2
    observation = _observation((0.15, 0.70, 0.15))
    observation[:, :330] = 0.0
    observation[0, 0 * 11 + 0] = 0.9  # a high-GT vessel exists at index 0
    mask = np.zeros(31, dtype=bool)
    mask[[0, 1, 2]] = True

    action, _ = model.predict(
        observation[0], deterministic=True, action_masks=mask, tie_eps=0.01
    )
    assert action == 2  # a wide probability gap is not a tie, so it is not overridden


def test_tie_aware_decode_never_selects_a_masked_action():
    model = _two_way_tie_model()
    observation = _observation((0.15, 0.70, 0.15))
    mask = np.ones(31, dtype=bool)
    mask[1] = False  # the higher-scoring tie candidate is masked out
    action, _ = model.predict(
        observation[0], deterministic=True, action_masks=mask, tie_eps=0.05
    )
    assert action != 1


def test_decode_tie_eps_round_trips(tmp_path):
    model = SoftMoEActorCritic(_space(), 31)
    model.decode_tie_eps = 0.007
    path = model.save(tmp_path / "soft-moe.pt")
    loaded = SoftMoEActorCritic.load(path)
    assert loaded.decode_tie_eps == pytest.approx(0.007)


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
