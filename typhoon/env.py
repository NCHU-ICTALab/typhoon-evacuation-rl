"""Maskable Gymnasium environment for offline typhoon evacuation training."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TyphoonEvacuationEnv(gym.Env):
    """Dispatch vessels before a hard port-closure deadline."""

    metadata = {"render_modes": []}
    VESSEL_FEATURES = 11
    # Eight resource/time features, three resource-release features, and three
    # operator-preference weights.
    GLOBAL_FEATURES = 14

    def __init__(
        self,
        base_scenario: dict | list[dict],
        *,
        randomize: bool = True,
        fixed_preference: tuple[float, float, float] | None = None,
        closure_range: tuple[float, float] = (3.0, 10.0),
        tug_range: tuple[int, int] = (4, 10),
        compression_range: tuple[float, float] = (1.5, 4.0),
    ):
        super().__init__()
        scenarios = base_scenario if isinstance(base_scenario, list) else [base_scenario]
        if not scenarios or not all(item.get("vessels") for item in scenarios):
            raise ValueError("base_scenario must contain at least one vessel")
        sizes = {len(item["vessels"]) for item in scenarios}
        if len(sizes) != 1:
            raise ValueError("all base scenarios must contain the same vessel count")
        self.base_scenarios = deepcopy(scenarios)
        self.base_scenario = self.base_scenarios[0]
        self.n_vessels = len(self.base_scenario["vessels"])
        self.wait_action = self.n_vessels
        self.randomize = randomize
        self.fixed_preference = fixed_preference
        self.closure_range = closure_range
        self.tug_range = tug_range
        self.compression_range = compression_range

        self.action_space = spaces.Discrete(self.n_vessels + 1)
        obs_size = self.n_vessels * self.VESSEL_FEATURES + self.GLOBAL_FEATURES
        self.observation_space = spaces.Box(
            low=-2.0, high=5.0, shape=(obs_size,), dtype=np.float32
        )

        self.vessels: list[dict] = []
        self.closure_hour = 0.0
        self.tug_capacity = 0
        self.entrance_capacity: dict[str, int] = {}
        self.now = 0.0
        self.active: list[dict] = []
        self.schedule: list[dict] = []
        self.unscheduled: set[int] = set()
        self.rejected_actions = 0
        self._episode_done = False
        self.preference = np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        self.total_gt = 1.0
        self.total_risk = 1.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        randomized = self.randomize and not options.get("deterministic", False)

        if "scenario_index" in options:
            scenario_index = int(options["scenario_index"])
        elif randomized:
            scenario_index = int(self.np_random.integers(0, len(self.base_scenarios)))
        else:
            scenario_index = 0
        self.base_scenario = self.base_scenarios[
            scenario_index % len(self.base_scenarios)
        ]

        preference = options.get("preference", self.fixed_preference)
        if preference is None and randomized:
            weights = self.np_random.dirichlet(np.ones(3))
        elif preference is None:
            weights = np.ones(3)
        elif isinstance(preference, dict):
            weights = np.asarray(
                [
                    preference.get("count", 0.0),
                    preference.get("gt", 0.0),
                    preference.get("risk", 0.0),
                ],
                dtype=float,
            )
        else:
            weights = np.asarray(preference, dtype=float)
        if weights.shape != (3,) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("preference must be three non-negative weights")
        self.preference = (weights / weights.sum()).astype(np.float32)

        if "closure_hour" in options:
            self.closure_hour = float(options["closure_hour"])
        elif randomized:
            self.closure_hour = float(self.np_random.uniform(*self.closure_range))
        else:
            self.closure_hour = 6.0

        if "tug_capacity" in options:
            self.tug_capacity = int(options["tug_capacity"])
        elif randomized:
            lo, hi = self.tug_range
            self.tug_capacity = int(self.np_random.integers(lo, hi + 1))
        else:
            self.tug_capacity = int(
                self.base_scenario["resources"].get("tug_capacity", 8)
            )

        if "demand_compression" in options:
            compression = float(options["demand_compression"])
        elif randomized:
            compression = float(self.np_random.uniform(*self.compression_range))
        else:
            compression = 2.5
        if self.closure_hour <= 0 or self.tug_capacity <= 0 or compression <= 0:
            raise ValueError("closure, tug capacity, and compression must be positive")

        self.entrance_capacity = {
            str(k): int(v)
            for k, v in self.base_scenario["resources"]["entrance_capacity"].items()
        }
        jitter = bool(options.get("jitter", randomized))
        self.vessels = []
        for source in self.base_scenario["vessels"]:
            vessel = deepcopy(source)
            ready = float(source["ready_hour"]) / compression
            transit = float(source["transit_hour"])
            if jitter:
                ready += float(self.np_random.normal(0.0, 0.12))
                transit *= float(self.np_random.uniform(0.9, 1.1))
            vessel["ready_hour"] = max(0.0, ready)
            vessel["transit_hour"] = max(0.05, transit)
            self.vessels.append(vessel)

        self.total_gt = max(
            1.0, sum(float(v["gross_tonnage"]) for v in self.vessels)
        )
        self.total_risk = max(
            1.0, sum(float(v["risk_points"]) for v in self.vessels)
        )

        self.now = min(float(v["ready_hour"]) for v in self.vessels)
        self.active = []
        self.schedule = []
        self.unscheduled = set(range(self.n_vessels))
        self.rejected_actions = 0
        self._episode_done = False
        return self._observation(), self._info()

    def _release_finished(self) -> None:
        eps = 1e-9
        self.active = [j for j in self.active if j["finish_hour"] > self.now + eps]

    def _resource_use(self) -> tuple[int, Counter]:
        self._release_finished()
        return (
            sum(int(job["tugs"]) for job in self.active),
            Counter(str(job["entrance"]) for job in self.active),
        )

    def _is_feasible(self, index: int) -> bool:
        if index not in self.unscheduled or self._episode_done:
            return False
        vessel = self.vessels[index]
        eps = 1e-9
        if float(vessel["ready_hour"]) > self.now + eps:
            return False
        if self.now + float(vessel["transit_hour"]) > self.closure_hour + eps:
            return False
        used_tugs, used_entrances = self._resource_use()
        entrance = str(vessel["entrance"])
        return (
            used_tugs + int(vessel["tugs"]) <= self.tug_capacity
            and used_entrances[entrance] < self.entrance_capacity.get(entrance, 1)
        )

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.n_vessels + 1, dtype=bool)
        if self._episode_done:
            mask[self.wait_action] = True
            return mask
        for i in self.unscheduled:
            mask[i] = self._is_feasible(i)
        mask[self.wait_action] = True
        return mask

    def _next_event(self) -> float | None:
        eps = 1e-9
        future = [
            float(job["finish_hour"])
            for job in self.active
            if self.now + eps < float(job["finish_hour"]) <= self.closure_hour + eps
        ]
        future.extend(
            float(self.vessels[i]["ready_hour"])
            for i in self.unscheduled
            if self.now + eps
            < float(self.vessels[i]["ready_hour"])
            < self.closure_hour - eps
        )
        return min(future) if future else None

    def dispatch_objectives(self, vessel: dict) -> np.ndarray:
        """Return count, GT, and risk reward before preference scalarization."""
        return 10.0 * np.asarray(
            [
                1.0 / self.n_vessels,
                float(vessel["gross_tonnage"]) / self.total_gt,
                float(vessel["risk_points"]) / self.total_risk,
            ],
            dtype=np.float32,
        )

    def dispatch_value(self, vessel: dict) -> float:
        """Scalarized utility retained for MaskablePPO compatibility."""
        return float(np.dot(self.preference, self.dispatch_objectives(vessel)))

    def step(self, action):
        if self._episode_done:
            raise RuntimeError("step() called after episode termination")
        action = int(action)
        objective_reward = np.zeros(3, dtype=np.float32)

        if action == self.wait_action:
            next_event = self._next_event()
            if next_event is None:
                delta = max(0.0, self.closure_hour - self.now)
                self.now = self.closure_hour
                self._episode_done = True
            else:
                delta = max(0.0, next_event - self.now)
                self.now = next_event
            wait_cost = 0.02 * delta / max(self.closure_hour, 1e-6)
            objective_reward -= wait_cost
        elif 0 <= action < self.n_vessels and self._is_feasible(action):
            vessel = self.vessels[action]
            finish = self.now + float(vessel["transit_hour"])
            record = {
                "index": action,
                "ship_id": vessel["ship_id"],
                "gross_tonnage": float(vessel["gross_tonnage"]),
                "risk_points": int(vessel["risk_points"]),
                "tugs": int(vessel["tugs"]),
                "entrance": str(vessel["entrance"]),
                "start_hour": self.now,
                "finish_hour": finish,
            }
            self.schedule.append(record)
            self.active.append(record)
            self.unscheduled.remove(action)
            objective_reward += self.dispatch_objectives(vessel)
            if not self.unscheduled:
                self._episode_done = True
        else:
            # Unsafe/unavailable dispatches are rejected for unmasked callers.
            self.rejected_actions += 1
            objective_reward -= 5.0
            next_event = self._next_event()
            if next_event is None:
                self.now = self.closure_hour
                self._episode_done = True
            else:
                self.now = next_event

        reward = float(np.dot(self.preference, objective_reward))

        if self.now >= self.closure_hour - 1e-9:
            self._episode_done = True
        if not self._episode_done:
            can_dispatch = any(self._is_feasible(i) for i in self.unscheduled)
            if not can_dispatch and self._next_event() is None:
                self._episode_done = True
        info = self._info()
        info["objective_reward"] = objective_reward.tolist()
        if self._episode_done:
            info["kpi"] = self.kpis()
        return self._observation(), float(reward), self._episode_done, False, info

    def _observation(self) -> np.ndarray:
        mask = self.action_masks()[:-1]
        values: list[float] = []
        closure = max(self.closure_hour, 1e-6)
        for i, vessel in enumerate(self.vessels):
            ready = float(vessel["ready_hour"])
            transit = float(vessel["transit_hour"])
            values.extend(
                [
                    np.log1p(float(vessel["gross_tonnage"])) / np.log1p(250_000.0),
                    int(vessel["tugs"]) / 4.0,
                    transit / 1.5,
                    int(vessel["risk_points"]) / 5.0,
                    float(str(vessel["entrance"]) == "1"),
                    float(str(vessel["entrance"]) == "2"),
                    np.clip(ready / closure, 0.0, 2.0),
                    np.clip(max(self.now - ready, 0.0) / closure, 0.0, 2.0),
                    np.clip((closure - self.now - transit) / closure, -1.0, 1.0),
                    float(i in self.unscheduled),
                    float(mask[i]),
                ]
            )

        used_tugs, used_entrances = self._resource_use()
        next_tug_release = min(
            [float(job["finish_hour"]) for job in self.active] or [self.now]
        )
        entrance_release = {}
        for entrance in ("1", "2"):
            entrance_release[entrance] = min(
                [
                    float(job["finish_hour"])
                    for job in self.active
                    if str(job["entrance"]) == entrance
                ]
                or [self.now]
            )
        values.extend(
            [
                np.clip(self.now / closure, 0.0, 1.0),
                np.clip((closure - self.now) / closure, 0.0, 1.0),
                np.clip(
                    (self.tug_capacity - used_tugs) / self.tug_capacity, 0.0, 1.0
                ),
                float(used_entrances["1"] < self.entrance_capacity.get("1", 1)),
                float(used_entrances["2"] < self.entrance_capacity.get("2", 1)),
                len(self.unscheduled) / self.n_vessels,
                len(self.active) / self.n_vessels,
                np.count_nonzero(mask) / self.n_vessels,
                np.clip((next_tug_release - self.now) / closure, 0.0, 1.0),
                np.clip((entrance_release["1"] - self.now) / closure, 0.0, 1.0),
                np.clip((entrance_release["2"] - self.now) / closure, 0.0, 1.0),
                *self.preference.tolist(),
            ]
        )
        return np.asarray(values, dtype=np.float32)

    def kpis(self) -> dict:
        remaining = [self.vessels[i] for i in self.unscheduled]
        tug_hours = sum(
            float(job["finish_hour"] - job["start_hour"]) * int(job["tugs"])
            for job in self.schedule
        )
        entrance_hours = sum(
            float(job["finish_hour"] - job["start_hour"]) for job in self.schedule
        )
        n_entrances = max(1, len(self.entrance_capacity))
        return {
            "safety_violations": 0,
            "rejected_actions": self.rejected_actions,
            "evacuated_count": len(self.schedule),
            "evacuated_gt": round(sum(j["gross_tonnage"] for j in self.schedule), 1),
            "evacuated_risk_points": sum(j["risk_points"] for j in self.schedule),
            "remaining_count": len(remaining),
            "remaining_gt": round(
                sum(float(v["gross_tonnage"]) for v in remaining), 1
            ),
            "remaining_risk_points": sum(
                int(v["risk_points"]) for v in remaining
            ),
            "tug_utilization": round(
                tug_hours / max(self.tug_capacity * self.closure_hour, 1e-6), 4
            ),
            "entrance_utilization": round(
                entrance_hours / max(n_entrances * self.closure_hour, 1e-6), 4
            ),
            "closure_hour": round(self.closure_hour, 3),
            "tug_capacity": self.tug_capacity,
            "preference_count": round(float(self.preference[0]), 4),
            "preference_gt": round(float(self.preference[1]), 4),
            "preference_risk": round(float(self.preference[2]), 4),
            "weighted_utility": round(
                float(
                    self.preference[0] * len(self.schedule) / self.n_vessels
                    + self.preference[1]
                    * sum(j["gross_tonnage"] for j in self.schedule)
                    / self.total_gt
                    + self.preference[2]
                    * sum(j["risk_points"] for j in self.schedule)
                    / self.total_risk
                ),
                6,
            ),
            "scenario_date": self.base_scenario.get("metadata", {}).get(
                "query_date", "test"
            ),
        }

    def _info(self) -> dict:
        return {
            "time_hour": self.now,
            "closure_hour": self.closure_hour,
            "scheduled_count": len(self.schedule),
            "remaining_count": len(self.unscheduled),
        }


class VectorRewardWrapper(gym.Wrapper):
    """Expose the base environment objective vector for MORL algorithms."""

    def __init__(self, env: TyphoonEvacuationEnv):
        super().__init__(env)
        self.reward_space = spaces.Box(
            low=np.full(3, -5.0, dtype=np.float32),
            high=np.full(3, 10.0, dtype=np.float32),
            dtype=np.float32,
        )

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        reward = np.asarray(info["objective_reward"], dtype=np.float32)
        return observation, reward, terminated, truncated, info
