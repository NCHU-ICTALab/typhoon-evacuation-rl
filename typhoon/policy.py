"""Preference-aware policy components for conditioned typhoon scheduling."""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PreferenceGatedExtractor(BaseFeaturesExtractor):
    """Encode preference separately and use FiLM to modulate every vessel."""

    def __init__(
        self,
        observation_space: spaces.Box,
        n_vessels: int = 30,
        vessel_features: int = 11,
        global_features: int = 14,
        vessel_embedding: int = 32,
        features_dim: int = 256,
    ):
        expected = n_vessels * vessel_features + global_features
        if observation_space.shape != (expected,):
            raise ValueError(
                f"expected flat observation ({expected},), got {observation_space.shape}"
            )
        super().__init__(observation_space, features_dim)
        self.n_vessels = n_vessels
        self.vessel_features = vessel_features
        self.vessel_end = n_vessels * vessel_features

        self.vessel_encoder = torch.nn.Sequential(
            torch.nn.Linear(vessel_features, vessel_embedding),
            torch.nn.ReLU(),
            torch.nn.Linear(vessel_embedding, vessel_embedding),
            torch.nn.ReLU(),
        )
        self.preference_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
        )
        self.film = torch.nn.Linear(32, 2 * vessel_embedding)
        self.fleet_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_vessels * vessel_embedding, 192),
            torch.nn.ReLU(),
        )
        # Eleven non-preference global features plus the preference embedding.
        self.global_encoder = torch.nn.Sequential(
            torch.nn.Linear((global_features - 3) + 32, 64),
            torch.nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        vessels = observations[:, : self.vessel_end].reshape(
            -1, self.n_vessels, self.vessel_features
        )
        global_without_preference = observations[:, self.vessel_end : -3]
        preference = observations[:, -3:]

        preference_embedding = self.preference_encoder(preference)
        gamma, beta = self.film(preference_embedding).chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(1)
        beta = beta.unsqueeze(1)

        vessel_embedding = self.vessel_encoder(vessels)
        gated_vessels = vessel_embedding * (1.0 + gamma) + beta
        fleet_features = self.fleet_encoder(gated_vessels.flatten(start_dim=1))
        global_features = self.global_encoder(
            torch.cat((global_without_preference, preference_embedding), dim=-1)
        )
        return torch.cat((fleet_features, global_features), dim=-1)
