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
ROUTING_PREFERENCES = (
    (0.70, 0.15, 0.15),
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (0.15, 0.70, 0.15),
    (0.15, 0.15, 0.70),
)


class DecomposedActorCritic(torch.nn.Module):
    """One preference-conditioned actor and a three-head vector critic."""

    def __init__(
        self,
        observation_space: spaces.Box,
        n_actions: int,
        *,
        features_dim: int = 256,
        actor_routing: str = "shared",
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = spaces.Discrete(n_actions)
        self.n_actions = int(n_actions)
        self.features_dim = int(features_dim)
        if actor_routing not in {"shared", "hard_heads"}:
            raise ValueError("actor_routing must be 'shared' or 'hard_heads'")
        self.actor_routing = actor_routing
        self.extractor = PreferenceGatedExtractor(
            observation_space,
            n_vessels=30,
            vessel_features=11,
            global_features=14,
            features_dim=features_dim,
        )
        if actor_routing == "shared":
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(features_dim, 256),
                torch.nn.Tanh(),
                torch.nn.Linear(256, 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, n_actions),
            )
        else:
            self.actor_trunk = torch.nn.Sequential(
                torch.nn.Linear(features_dim, 256),
                torch.nn.Tanh(),
                torch.nn.Linear(256, 128),
                torch.nn.Tanh(),
            )
            self.actor_heads = torch.nn.ModuleList(
                [torch.nn.Linear(128, n_actions)]
            )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(features_dim, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, len(OBJECTIVE_NAMES)),
        )
        if actor_routing == "hard_heads":
            for _ in range(len(ROUTING_PREFERENCES) - 1):
                head = torch.nn.Linear(128, n_actions)
                head.load_state_dict(self.actor_heads[0].state_dict())
                self.actor_heads.append(head)
        self.device = torch.device(device)
        self.to(self.device)

    def _features(self, observations: torch.Tensor) -> torch.Tensor:
        return self.extractor(observations)

    def policy_parameters(self) -> list[torch.nn.Parameter]:
        modules = (
            [self.extractor, self.actor]
            if self.actor_routing == "shared"
            else [self.extractor, self.actor_trunk, self.actor_heads]
        )
        return [parameter for module in modules for parameter in module.parameters()]

    def _policy_logits(
        self, features: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
        if self.actor_routing == "shared":
            return self.actor(features)
        hidden = self.actor_trunk(features)
        all_logits = torch.stack(
            [head(hidden) for head in self.actor_heads], dim=1
        )
        routing_preferences = torch.as_tensor(
            ROUTING_PREFERENCES,
            dtype=observations.dtype,
            device=observations.device,
        )
        distances = torch.sum(
            (observations[:, None, -3:] - routing_preferences[None, :, :]).square(),
            dim=-1,
        )
        head_indices = distances.argmin(dim=1)
        return all_logits[
            torch.arange(len(observations), device=observations.device), head_indices
        ]

    def _distribution(
        self, observations: torch.Tensor, action_masks: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        features = self._features(observations)
        logits = self._policy_logits(features, observations)
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
                "actor_routing": self.actor_routing,
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
            actor_routing=checkpoint.get("actor_routing", "shared"),
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


def project_conflicting_gradients(
    objective_gradients: list[list[torch.Tensor | None]],
    *,
    rng: np.random.Generator,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Apply PCGrad to task gradients and return their summed update.

    Projection is limited to negative dot products. Each task is projected
    against the original gradient of the other tasks in a randomized order,
    following the PCGrad algorithm. The caller remains responsible for adding
    critic and entropy gradients.
    """
    if len(objective_gradients) < 2:
        raise ValueError("PCGrad requires at least two objective gradients")
    width = len(objective_gradients[0])
    if width == 0 or any(len(items) != width for items in objective_gradients):
        raise ValueError("objective gradients must have equal non-zero length")

    dense: list[list[torch.Tensor]] = []
    for objective in objective_gradients:
        dense_objective = []
        for index, item in enumerate(objective):
            if item is None:
                template = next(
                    (
                        candidate[index]
                        for candidate in objective_gradients
                        if candidate[index] is not None
                    ),
                    None,
                )
                if template is None:
                    raise ValueError("a parameter is unused by every objective")
                item = torch.zeros_like(template)
            dense_objective.append(item.detach().clone())
        dense.append(dense_objective)

    projected = [[item.clone() for item in objective] for objective in dense]
    comparisons = 0
    projections = 0
    for left_index in range(len(projected)):
        order = rng.permutation(len(dense))
        for right_index in order:
            if left_index == int(right_index):
                continue
            comparisons += 1
            dot = sum(
                torch.sum(left * right)
                for left, right in zip(projected[left_index], dense[int(right_index)])
            )
            if dot.item() >= 0.0:
                continue
            norm_squared = sum(
                torch.sum(right.square()) for right in dense[int(right_index)]
            )
            if norm_squared.item() <= 1e-12:
                continue
            coefficient = dot / norm_squared
            projected[left_index] = [
                left - coefficient * right
                for left, right in zip(
                    projected[left_index], dense[int(right_index)]
                )
            ]
            projections += 1

    merged = [sum(items) for items in zip(*projected)]
    return merged, {
        "pcgrad_projection_count": float(projections),
        "pcgrad_projection_fraction": float(projections / max(comparisons, 1)),
    }
