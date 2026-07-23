import numpy as np
import torch
from gymnasium import spaces

from typhoon.decomposed_ppo import (
    DecomposedActorCritic,
    VectorRollout,
    compute_vector_gae,
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


def test_one_decomposed_update_reports_objective_losses_and_cosines():
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
