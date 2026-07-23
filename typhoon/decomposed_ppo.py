"""A compact decomposed, action-masked PPO implementation for three objectives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from torch.distributions import Categorical

from .policy import PreferenceGatedExtractor


OBJECTIVE_NAMES = ("count", "gt", "risk")


class DecomposedActorCritic(torch.nn.Module):
    """One preference-conditioned actor and a three-head vector critic."""

    def __init__(
        self,
        observation_space: spaces.Box,
        n_actions: int,
        *,
        features_dim: int = 256,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = spaces.Discrete(n_actions)
        self.n_actions = int(n_actions)
        self.features_dim = int(features_dim)
        self.extractor = PreferenceGatedExtractor(
            observation_space,
            n_vessels=30,
            vessel_features=11,
            global_features=14,
            features_dim=features_dim,
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(features_dim, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, n_actions),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(features_dim, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, len(OBJECTIVE_NAMES)),
        )
        self.device = torch.device(device)
        self.to(self.device)

    def _features(self, observations: torch.Tensor) -> torch.Tensor:
        return self.extractor(observations)

    def _distribution(
        self, observations: torch.Tensor, action_masks: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        features = self._features(observations)
        logits = self.actor(features)
        masked_logits = logits.masked_fill(~action_masks.bool(), -1e9)
        return Categorical(logits=masked_logits), features

    def act(
        self,
        observations: torch.Tensor,
        action_masks: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, features = self._distribution(observations, action_masks)
        actions = (
            distribution.probs.argmax(dim=-1)
            if deterministic
            else distribution.sample()
        )
        return actions, distribution.log_prob(actions), self.critic(features)

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, features = self._distribution(observations, action_masks)
        return (
            distribution.log_prob(actions),
            distribution.entropy(),
            self.critic(features),
        )

    def values(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(self._features(observations))

    def predict(self, observation, *, deterministic=True, action_masks=None):
        array = np.asarray(observation, dtype=np.float32)
        single = array.ndim == 1
        if single:
            array = array[None, :]
        if action_masks is None:
            masks = np.ones((len(array), self.n_actions), dtype=bool)
        else:
            masks = np.asarray(action_masks, dtype=bool)
            if masks.ndim == 1:
                masks = masks[None, :]
        with torch.no_grad():
            actions, _, _ = self.act(
                torch.as_tensor(array, device=self.device),
                torch.as_tensor(masks, device=self.device),
                deterministic=deterministic,
            )
        result = actions.cpu().numpy()
        return (int(result[0]) if single else result), None

    def save(self, path: str | Path, *, metadata: dict | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "observation_shape": self.observation_space.shape,
                "n_actions": self.n_actions,
                "features_dim": self.features_dim,
                "metadata": metadata or {},
            },
            target,
        )
        return target

    @classmethod
    def load(
        cls, path: str | Path, *, device: str | torch.device = "cpu"
    ) -> "DecomposedActorCritic":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        observation_space = spaces.Box(
            low=-2.0,
            high=5.0,
            shape=tuple(checkpoint["observation_shape"]),
            dtype=np.float32,
        )
        model = cls(
            observation_space,
            int(checkpoint["n_actions"]),
            features_dim=int(checkpoint["features_dim"]),
            device=device,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


@dataclass
class VectorRollout:
    observations: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    action_masks: np.ndarray
    objective_rewards: np.ndarray
    values: np.ndarray
    dones: np.ndarray
    preferences: np.ndarray
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None


def compute_vector_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute independent GAE/returns for count, GT, and risk."""
    if rewards.shape != values.shape or rewards.shape[-1] != 3:
        raise ValueError("rewards and values must share shape (steps, envs, 3)")
    if dones.shape != rewards.shape[:2] or last_values.shape != rewards.shape[1:]:
        raise ValueError("dones or last_values shape does not match rollout")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros_like(last_values, dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        next_values = last_values if step == rewards.shape[0] - 1 else values[step + 1]
        nonterminal = (1.0 - dones[step]).astype(np.float32)[:, None]
        delta = rewards[step] + gamma * next_values * nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[step] = last_gae
    return advantages, advantages + values


def gradient_cosine(
    left: list[torch.Tensor | None], right: list[torch.Tensor | None]
) -> float:
    left_flat = torch.cat([item.reshape(-1) for item in left if item is not None])
    right_flat = torch.cat([item.reshape(-1) for item in right if item is not None])
    denominator = left_flat.norm() * right_flat.norm()
    if denominator.item() <= 1e-12:
        return 0.0
    return float(torch.dot(left_flat, right_flat).div(denominator).detach().cpu())
