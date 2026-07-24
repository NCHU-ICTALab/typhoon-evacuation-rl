import numpy as np
import pytest
import torch
from gymnasium import spaces

from typhoon.decomposed_ppo import (
    DecomposedActorCritic,
    VectorRollout,
    compute_vector_gae,
    project_conflicting_gradients,
)
from typhoon.train_decomposed_ppo import update_policy


def _space():
    return spaces.Box(low=-2.0, high=5.0, shape=(344,), dtype=np.float32)


def _observations(count=2):
    observations = np.zeros((count, 344), dtype=np.float32)
    observations[:, :330] = 0.25
    observations[0::2, -3:] = (0.70, 0.15, 0.15)
    observations[1::2, -3:] = (0.15, 0.15, 0.70)
    return observations


def test_vector_gae_keeps_objectives_separate_at_terminal():
    rewards = np.asarray([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    values = np.zeros_like(rewards)
    dones = np.ones((1, 1), dtype=np.float32)
    last_values = np.asarray([[9.0, 9.0, 9.0]], dtype=np.float32)
    advantages, returns = compute_vector_gae(
        rewards,
        values,
        dones,
        last_values,
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert np.allclose(advantages, rewards)
    assert np.allclose(returns, rewards)


def test_decomposed_policy_has_three_values_and_respects_mask():
    model = DecomposedActorCritic(_space(), 31)
    observation = _observations(1)
    mask = np.zeros((1, 31), dtype=bool)
    mask[0, 7] = True
    action, _ = model.predict(
        observation[0], deterministic=True, action_masks=mask[0]
    )
    with torch.no_grad():
        values = model.values(torch.as_tensor(observation))
    assert action == 7
    assert values.shape == (1, 3)


def test_hard_heads_start_equal_and_route_each_profile():
    model = DecomposedActorCritic(_space(), 31, actor_routing="hard_heads")
    first_state = model.actor_heads[0].state_dict()
    for head in model.actor_heads[1:]:
        assert all(
            torch.equal(first_state[name], value)
            for name, value in head.state_dict().items()
        )

    expected_actions = [2, 5, 8, 11]
    with torch.no_grad():
        for head, action in zip(model.actor_heads, expected_actions):
            head.weight.zero_()
            head.bias.fill_(-10.0)
            head.bias[action] = 10.0
    observations = np.zeros((4, 344), dtype=np.float32)
    observations[:, :330] = 0.25
    observations[:, -3:] = np.asarray(
        [
            (0.70, 0.15, 0.15),
            (1 / 3, 1 / 3, 1 / 3),
            (0.15, 0.70, 0.15),
            (0.15, 0.15, 0.70),
        ],
        dtype=np.float32,
    )
    actions, _ = model.predict(
        observations,
        deterministic=True,
        action_masks=np.ones((4, 31), dtype=bool),
    )
    assert actions.tolist() == expected_actions


def test_hard_head_checkpoint_round_trip(tmp_path):
    model = DecomposedActorCritic(_space(), 31, actor_routing="hard_heads")
    loaded = DecomposedActorCritic.load(model.save(tmp_path / "hard-heads.pt"))
    assert loaded.actor_routing == "hard_heads"
    assert len(loaded.actor_heads) == 4


def test_full_experts_start_equal_and_route_each_profile():
    model = DecomposedActorCritic(_space(), 31, actor_routing="full_experts")
    first_state = model.actor_experts[0].state_dict()
    for expert in model.actor_experts[1:]:
        assert all(
            torch.equal(first_state[name], value)
            for name, value in expert.state_dict().items()
        )

    expected_actions = [3, 6, 9, 12]
    with torch.no_grad():
        for expert, action in zip(model.actor_experts, expected_actions):
            output = expert[-1]
            output.weight.zero_()
            output.bias.fill_(-10.0)
            output.bias[action] = 10.0
    observations = np.zeros((4, 344), dtype=np.float32)
    observations[:, :330] = 0.25
    observations[:, -3:] = np.asarray(
        [
            (0.70, 0.15, 0.15),
            (1 / 3, 1 / 3, 1 / 3),
            (0.15, 0.70, 0.15),
            (0.15, 0.15, 0.70),
        ],
        dtype=np.float32,
    )
    actions, _ = model.predict(
        observations,
        deterministic=True,
        action_masks=np.ones((4, 31), dtype=bool),
    )
    assert actions.tolist() == expected_actions


def test_full_expert_checkpoint_round_trip(tmp_path):
    model = DecomposedActorCritic(_space(), 31, actor_routing="full_experts")
    loaded = DecomposedActorCritic.load(model.save(tmp_path / "full-experts.pt"))
    assert loaded.actor_routing == "full_experts"
    assert len(loaded.actor_experts) == 4


def test_pcgrad_projects_opposed_gradients_and_preserves_aligned_gradients():
    opposed, opposed_metrics = project_conflicting_gradients(
        [[torch.tensor([1.0, 0.0])], [torch.tensor([-1.0, 0.0])]],
        rng=np.random.default_rng(4),
    )
    assert torch.allclose(opposed[0], torch.zeros(2), atol=1e-6)
    assert opposed_metrics["pcgrad_projection_fraction"] == 1.0

    aligned, aligned_metrics = project_conflicting_gradients(
        [[torch.tensor([1.0, 0.0])], [torch.tensor([2.0, 0.0])]],
        rng=np.random.default_rng(4),
    )
    assert torch.allclose(aligned[0], torch.tensor([3.0, 0.0]))
    assert aligned_metrics["pcgrad_projection_fraction"] == 0.0


@pytest.mark.parametrize("gradient_surgery", ["none", "pcgrad"])
def test_one_decomposed_update_reports_objective_losses_and_cosines(
    gradient_surgery,
):
    torch.manual_seed(3)
    model = DecomposedActorCritic(_space(), 31)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    observations = _observations(2).reshape(2, 1, 344)
    masks = np.ones((2, 1, 31), dtype=bool)
    with torch.no_grad():
        actions, log_probs, values = model.act(
            torch.as_tensor(observations.reshape(2, 344)),
            torch.as_tensor(masks.reshape(2, 31)),
        )
    rewards = np.asarray(
        [[[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]], dtype=np.float32
    )
    dones = np.ones((2, 1), dtype=np.float32)
    advantages, returns = compute_vector_gae(
        rewards,
        values.numpy().reshape(2, 1, 3),
        dones,
        np.zeros((1, 3), dtype=np.float32),
        gamma=0.99,
        gae_lambda=0.95,
    )
    rollout = VectorRollout(
        observations=observations,
        actions=actions.numpy().reshape(2, 1),
        old_log_probs=log_probs.numpy().reshape(2, 1),
        action_masks=masks,
        objective_rewards=rewards,
        values=values.numpy().reshape(2, 1, 3),
        dones=dones,
        preferences=observations[:, :, -3:],
        advantages=advantages,
        returns=returns,
    )
    metrics = update_policy(
        model,
        optimizer,
        rollout,
        epochs=1,
        batch_size=2,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        rng=np.random.default_rng(3),
        gradient_surgery=gradient_surgery,
    )
    assert set(("policy_loss_count", "policy_loss_gt", "policy_loss_risk")) <= set(
        metrics
    )
    assert set(
        (
            "gradient_cosine_count_gt",
            "gradient_cosine_count_risk",
            "gradient_cosine_gt_risk",
        )
    ) <= set(metrics)
    if gradient_surgery == "pcgrad":
        assert "pcgrad_projection_fraction" in metrics
    assert all(np.isfinite(value) for value in metrics.values())


def test_decomposed_checkpoint_round_trip(tmp_path):
    model = DecomposedActorCritic(_space(), 31)
    path = model.save(tmp_path / "model.pt", metadata={"phase": 2})
    loaded = DecomposedActorCritic.load(path)
    observation = _observations(1)[0]
    mask = np.zeros(31, dtype=bool)
    mask[11] = True
    action, _ = loaded.predict(
        observation, deterministic=True, action_masks=mask
    )
    assert action == 11
