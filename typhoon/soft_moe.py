"""Soft mixture-of-experts policy initialized from Phase 8 specialists."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from torch.distributions import Categorical

from .decomposed_ppo import (
    OBJECTIVE_NAMES,
    ROUTING_PREFERENCES,
    DecomposedActorCritic,
)
from .policy import PreferenceGatedExtractor


class SoftMoEActorCritic(torch.nn.Module):
    """One checkpoint with frozen Phase 8 experts and a trainable soft router."""

    def __init__(
        self,
        observation_space: spaces.Box,
        n_actions: int,
        *,
        experts: list[DecomposedActorCritic] | None = None,
        features_dim: int = 256,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = spaces.Discrete(n_actions)
        self.n_actions = int(n_actions)
        self.features_dim = int(features_dim)
        self.vessel_end = 30 * 11
        if observation_space.shape != (344,):
            raise ValueError("Soft MoE currently requires the fixed 30-vessel layout")

        if experts is None:
            experts = [
                DecomposedActorCritic(
                    observation_space,
                    n_actions,
                    features_dim=features_dim,
                )
                for _ in ROUTING_PREFERENCES
            ]
        if len(experts) != len(ROUTING_PREFERENCES):
            raise ValueError("Soft MoE requires exactly four experts")
        for expert in experts:
            if (
                expert.observation_space.shape != observation_space.shape
                or expert.n_actions != n_actions
            ):
                raise ValueError("expert observation/action space mismatch")
            for parameter in expert.parameters():
                parameter.requires_grad_(False)
            expert.eval()
        self.experts = torch.nn.ModuleList(experts)

        self.preference_router = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, len(experts)),
        )
        self.state_router = torch.nn.Sequential(
            torch.nn.Linear(11, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, len(experts)),
        )
        torch.nn.init.zeros_(self.state_router[-1].weight)
        torch.nn.init.zeros_(self.state_router[-1].bias)

        self.critic_extractor = PreferenceGatedExtractor(
            observation_space,
            n_vessels=30,
            vessel_features=11,
            global_features=14,
            features_dim=features_dim,
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(features_dim, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, len(OBJECTIVE_NAMES)),
        )
        # Start the future router-PPO critic from the balanced Phase 8 expert
        # instead of an unrelated random baseline.
        self.critic_extractor.load_state_dict(experts[1].extractor.state_dict())
        self.critic.load_state_dict(experts[1].critic.state_dict())
        self.device = torch.device(device)
        self.to(self.device)
        self._freeze_experts()

    @classmethod
    def from_expert_paths(
        cls,
        paths: list[str | Path],
        *,
        device: str | torch.device = "cpu",
    ) -> "SoftMoEActorCritic":
        experts = [DecomposedActorCritic.load(path, device=device) for path in paths]
        first = experts[0]
        return cls(
            first.observation_space,
            first.n_actions,
            experts=experts,
            features_dim=first.features_dim,
            device=device,
        )

    def _freeze_experts(self) -> None:
        for expert in self.experts:
            expert.eval()
            for parameter in expert.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_experts()
        return self

    def routing_logits(self, observations: torch.Tensor) -> torch.Tensor:
        preference = observations[:, -3:]
        state = observations[:, self.vessel_end : -3]
        return self.preference_router(preference) + self.state_router(state)

    def routing_weights(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.routing_logits(observations), dim=-1)

    def policy_parameters(self) -> list[torch.nn.Parameter]:
        return [
            parameter
            for module in (self.preference_router, self.state_router)
            for parameter in module.parameters()
        ]

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def _expert_probabilities(
        self, observations: torch.Tensor, action_masks: torch.Tensor
    ) -> torch.Tensor:
        probabilities = []
        with torch.no_grad():
            for index, expert in enumerate(self.experts):
                # Fixed-profile experts only saw their own centroid during
                # training. The router receives the continuous preference;
                # each expert remains evaluated at its teacher centroid.
                expert_observations = observations.clone()
                expert_observations[:, -3:] = torch.as_tensor(
                    ROUTING_PREFERENCES[index],
                    dtype=observations.dtype,
                    device=observations.device,
                )
                features = expert._features(expert_observations)
                logits = expert._policy_logits(features, expert_observations)
                masked_logits = logits.masked_fill(~action_masks.bool(), -1e9)
                probabilities.append(torch.softmax(masked_logits, dim=-1))
        return torch.stack(probabilities, dim=1)

    def _distribution(
        self, observations: torch.Tensor, action_masks: torch.Tensor
    ) -> Categorical:
        weights = self.routing_weights(observations)
        expert_probabilities = self._expert_probabilities(
            observations, action_masks
        )
        mixture = torch.sum(weights[:, :, None] * expert_probabilities, dim=1)
        valid_actions = action_masks.bool()
        if not torch.all(valid_actions.any(dim=-1)):
            raise ValueError("each observation must have at least one valid action")
        # Keep invalid actions at exactly zero. Clamping every probability would
        # otherwise re-introduce a tiny chance of choosing a masked action.
        mixture = mixture.masked_fill(~valid_actions, 0.0)
        mixture = mixture / mixture.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return Categorical(probs=mixture)

    def act(
        self,
        observations: torch.Tensor,
        action_masks: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observations, action_masks)
        actions = (
            distribution.probs.argmax(dim=-1)
            if deterministic
            else distribution.sample()
        )
        values = self.critic(self.critic_extractor(observations))
        return actions, distribution.log_prob(actions), values

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observations, action_masks)
        values = self.critic(self.critic_extractor(observations))
        return distribution.log_prob(actions), distribution.entropy(), values

    def values(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(self.critic_extractor(observations))

    def predict(self, observation, *, deterministic=True, action_masks=None):
        array = np.asarray(observation, dtype=np.float32)
        single = array.ndim == 1
        if single:
            array = array[None, :]
        masks = (
            np.ones((len(array), self.n_actions), dtype=bool)
            if action_masks is None
            else np.asarray(action_masks, dtype=bool)
        )
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

    def predict_routing(self, observation) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        with torch.no_grad():
            weights = self.routing_weights(
                torch.as_tensor(array, device=self.device)
            )
        return weights.cpu().numpy()

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
    ) -> "SoftMoEActorCritic":
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
