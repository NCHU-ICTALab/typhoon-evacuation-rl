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
        # Deterministic decoding: 0.0 keeps the plain argmax. A positive value
        # enables preference-consistent tie-breaking among near-equal actions
        # (Phase 10 monotonicity fix).
        self.decode_tie_eps = 0.0
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

    def _tie_break_scores(self, observations: torch.Tensor) -> torch.Tensor:
        """Preference-weighted per-action score used only to break decode ties.

        The per-vessel ``log1p(GT)`` and ``risk/5`` observation features are
        strictly monotone in ``GT_i / GT_total`` and ``risk_i / risk_total``
        within a single state, so ranking tied dispatch actions by
        ``w_gt * log_gt + w_risk * risk`` matches the preference-weighted
        immediate objective ordering (the constant ``1/N`` count term does not
        discriminate between vessels). The wait action gets a fixed low score so
        an equally-probable dispatch is preferred within a tie.
        """
        batch = observations.shape[0]
        vessels = observations[:, : self.vessel_end].view(batch, -1, 11)
        log_gt = vessels[:, :, 0]
        risk = vessels[:, :, 3]
        preference = observations[:, -3:]
        w_gt = preference[:, 1:2]
        w_risk = preference[:, 2:3]
        dispatch = w_gt * log_gt + w_risk * risk
        wait = torch.full(
            (batch, 1), -1.0, dtype=dispatch.dtype, device=dispatch.device
        )
        return torch.cat([dispatch, wait], dim=-1)

    def _action_objective_proxy(self, observations: torch.Tensor) -> torch.Tensor:
        """Per-action ``[count, gt, risk]`` proxy used by the monotonicity loss.

        Count is 1 for every dispatch, and GT/risk reuse the per-vessel
        ``log1p(GT)`` and ``risk/5`` observation features, which are strictly
        monotone in the true normalized objectives within one state. The wait
        action contributes zero to every objective. Shape ``(batch, actions, 3)``.
        """
        batch = observations.shape[0]
        vessels = observations[:, : self.vessel_end].view(batch, -1, 11)
        log_gt = vessels[:, :, 0]
        risk = vessels[:, :, 3]
        count = torch.ones_like(log_gt)
        wait = torch.zeros((batch, 1), dtype=log_gt.dtype, device=log_gt.device)
        obj_count = torch.cat([count, wait], dim=-1)
        obj_gt = torch.cat([log_gt, wait], dim=-1)
        obj_risk = torch.cat([risk, wait], dim=-1)
        return torch.stack([obj_count, obj_gt, obj_risk], dim=-1)

    def expert_expected_objectives(
        self, observations: torch.Tensor, action_masks: torch.Tensor
    ) -> torch.Tensor:
        """Frozen per-expert expected immediate objective, shape ``(batch, 4, 3)``.

        ``m[b, k, j] = sum_a p_k(a | s_b) * O_j(a)``. The experts are frozen and
        the objective proxy carries no parameters, so the result is detached: the
        monotonicity loss differentiates only through the router weights.
        """
        expert_probabilities = self._expert_probabilities(observations, action_masks)
        objective = self._action_objective_proxy(observations)
        return torch.einsum("bka,baj->bkj", expert_probabilities, objective).detach()

    def _tie_aware_argmax(
        self, probs: torch.Tensor, observations: torch.Tensor, tie_eps: float
    ) -> torch.Tensor:
        top = probs.max(dim=-1, keepdim=True).values
        tie = probs >= (top - tie_eps)
        scores = self._tie_break_scores(observations)
        masked = torch.where(tie, scores, torch.full_like(scores, float("-inf")))
        return masked.argmax(dim=-1)

    def act(
        self,
        observations: torch.Tensor,
        action_masks: torch.Tensor,
        *,
        deterministic: bool = False,
        tie_eps: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observations, action_masks)
        if deterministic:
            eps = self.decode_tie_eps if tie_eps is None else float(tie_eps)
            actions = (
                self._tie_aware_argmax(distribution.probs, observations, eps)
                if eps > 0.0
                else distribution.probs.argmax(dim=-1)
            )
        else:
            actions = distribution.sample()
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

    def predict(
        self, observation, *, deterministic=True, action_masks=None, tie_eps=None
    ):
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
                tie_eps=tie_eps,
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
                "decode_tie_eps": float(self.decode_tie_eps),
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
        model.decode_tie_eps = float(checkpoint.get("decode_tie_eps", 0.0))
        model.eval()
        return model
