import numpy as np
import torch
from gymnasium import spaces

from typhoon.decomposed_ppo import VectorRollout
from typhoon.soft_moe import SoftMoEActorCritic
from typhoon.train_soft_moe_ppo import (
    ContinuousPreferenceTrainingEnv,
    router_monotonicity_loss,
    update_router_policy,
)


def _space():
    return spaces.Box(low=-2.0, high=5.0, shape=(344,), dtype=np.float32)


def _scenario():
    return {
        "metadata": {"query_date": "phase9b-test"},
        "resources": {
            "tug_capacity": 8,
            "entrance_capacity": {"1": 1, "2": 1},
        },
        "vessels": [
            {
                "ship_id": f"ship-{index}",
                "name": f"SHIP {index}",
                "gross_tonnage": 10_000.0 + index * 1_000.0,
                "entrance": str(index % 2 + 1),
                "ready_hour": index * 0.1,
                "transit_hour": 0.5,
                "tugs": 1,
                "risk_points": index % 5 + 1,
            }
            for index in range(30)
        ],
    }


def _rollout(model: SoftMoEActorCritic) -> VectorRollout:
    rng = np.random.default_rng(11)
    steps, envs = 4, 4
    observations = rng.normal(0.0, 0.2, size=(steps, envs, 344)).astype(
        np.float32
    )
    preferences = rng.dirichlet(np.full(3, 0.7), size=(steps, envs)).astype(
        np.float32
    )
    observations[:, :, -3:] = preferences
    masks = np.ones((steps, envs, 31), dtype=bool)
    flat_observations = torch.as_tensor(observations.reshape(-1, 344))
    flat_masks = torch.as_tensor(masks.reshape(-1, 31))
    with torch.no_grad():
        actions, log_probs, values = model.act(
            flat_observations, flat_masks, deterministic=False
        )
    values_array = values.numpy().reshape(steps, envs, 3)
    advantages = rng.normal(0.0, 1.0, size=(steps, envs, 3)).astype(
        np.float32
    )
    returns = values_array + advantages
    return VectorRollout(
        observations=observations,
        actions=actions.numpy().reshape(steps, envs),
        old_log_probs=log_probs.numpy().reshape(steps, envs),
        action_masks=masks,
        objective_rewards=np.zeros((steps, envs, 3), dtype=np.float32),
        values=values_array,
        dones=np.zeros((steps, envs), dtype=np.float32),
        preferences=preferences,
        advantages=advantages,
        returns=returns,
    )


def test_continuous_training_env_preserves_anchor_and_samples_simplex():
    anchor = (0.70, 0.15, 0.15)
    anchored = ContinuousPreferenceTrainingEnv(
        [_scenario()],
        anchor_preference=anchor,
        base_seed=42,
        lane_index=0,
        anchor_fraction=1.0,
    )
    observation, _ = anchored.reset()
    assert np.allclose(observation[-3:], anchor)

    continuous = ContinuousPreferenceTrainingEnv(
        [_scenario()],
        anchor_preference=anchor,
        base_seed=42,
        lane_index=0,
        anchor_fraction=0.0,
    )
    preferences = [continuous.reset()[0][-3:] for _ in range(4)]
    assert all(np.isclose(preference.sum(), 1.0) for preference in preferences)
    assert any(not np.allclose(preference, anchor) for preference in preferences)


def _high_gt_observation(preference):
    observation = np.zeros((1, 344), dtype=np.float32)
    observation[0, 0 * 11 + 0] = 1.0  # vessel 0 is the high-GT ship
    observation[0, 1 * 11 + 0] = 0.0  # vessel 1 is the low-GT ship
    observation[0, -3:] = preference
    return torch.as_tensor(observation)


def _mask_two_dispatches():
    mask = np.zeros((1, 31), dtype=bool)
    mask[0, 0] = True
    mask[0, 1] = True
    return torch.as_tensor(mask)


def test_expert_expected_objectives_are_detached_and_shaped():
    model = SoftMoEActorCritic(_space(), 31)
    expected = model.expert_expected_objectives(
        _high_gt_observation((0.4, 0.3, 0.3)), _mask_two_dispatches()
    )
    assert expected.shape == (1, 4, 3)
    assert not expected.requires_grad


def test_hinge_monotonicity_loss_zero_when_routing_ignores_preference():
    model = SoftMoEActorCritic(_space(), 31)
    with torch.no_grad():
        for parameter in model.preference_router.parameters():
            parameter.zero_()  # routing no longer depends on the preference
    loss = router_monotonicity_loss(
        model,
        _high_gt_observation((0.4, 0.3, 0.3)),
        _mask_two_dispatches(),
        torch.tensor([[0.4, 0.3, 0.3]]),
        delta=0.1,
        mode="hinge",
    )
    assert float(loss.detach()) < 1e-6


def test_monotonicity_loss_penalizes_misordered_routing_and_grads_router_only():
    model = SoftMoEActorCritic(_space(), 31)
    with torch.no_grad():
        # Expert 2 (gt) dispatches the high-GT vessel; the others dispatch the low.
        for expert in model.experts:
            output = expert.actor[-1]
            output.weight.zero_()
            output.bias.fill_(-20.0)
            output.bias[1] = 20.0
        model.experts[2].actor[-1].bias[1] = -20.0
        model.experts[2].actor[-1].bias[0] = 20.0
        # Route AWAY from the gt expert as the gt weight rises (mis-ordered).
        for parameter in model.preference_router.parameters():
            parameter.zero_()
        model.preference_router[0].weight[0, 1] = 1.0  # hidden0 = tanh(gt weight)
        model.preference_router[2].weight[2, 0] = -8.0  # expert-2 logit falls with it

    observation = _high_gt_observation((0.4, 0.3, 0.3))
    loss = router_monotonicity_loss(
        model,
        observation,
        _mask_two_dispatches(),
        torch.tensor([[0.4, 0.3, 0.3]]),
        delta=0.1,
    )
    assert float(loss.detach()) > 0.0

    loss.backward()
    assert model.preference_router[2].weight.grad is not None
    assert torch.any(model.preference_router[2].weight.grad != 0)
    assert all(
        parameter.grad is None
        for expert in model.experts
        for parameter in expert.parameters()
    )


def test_router_ppo_updates_router_and_critic_but_not_experts():
    torch.manual_seed(4)
    model = SoftMoEActorCritic(_space(), 31)
    rollout = _rollout(model)
    expert_before = {
        name: value.detach().clone()
        for name, value in model.experts.state_dict().items()
    }
    router_before = [
        parameter.detach().clone() for parameter in model.policy_parameters()
    ]
    critic_before = [
        parameter.detach().clone() for parameter in model.critic.parameters()
    ]
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-3)

    metrics = update_router_policy(
        model,
        optimizer,
        rollout,
        epochs=2,
        batch_size=8,
        clip_range=0.2,
        value_coef=0.5,
        vector_value_aux_coef=0.1,
        entropy_coef=0.005,
        router_anchor_coef=0.02,
        router_anchor_temperature=0.02,
        max_grad_norm=0.5,
        rng=np.random.default_rng(7),
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(router_before, model.policy_parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, model.critic.parameters())
    )
    assert all(
        torch.equal(expert_before[name], value)
        for name, value in model.experts.state_dict().items()
    )
    assert all(
        parameter.grad is None
        for expert in model.experts
        for parameter in expert.parameters()
    )
    assert metrics["router_anchor_loss"] > 0.0
    assert np.isfinite(metrics["approximate_kl"])
