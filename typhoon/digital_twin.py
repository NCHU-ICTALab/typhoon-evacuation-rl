"""Typed contract for digital-twin evacuation scheduling snapshots.

The first integration slice is intentionally stateless.  A digital twin sends
one decision-epoch snapshot and re-sends an updated snapshot whenever vessel
readiness, resources, or the closure deadline changes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CURRENT_VESSEL_SLOTS = 30
SUPPORTED_ENTRANCES = {"1", "2"}
TRAINED_TRANSIT_HOUR_RANGE = (0.36, 1.65)


class DigitalTwinVessel(BaseModel):
    """One not-yet-dispatched vessel at the current decision epoch."""

    model_config = ConfigDict(extra="forbid")

    ship_id: str = Field(min_length=1, max_length=128)
    name: str | None = Field(default=None, max_length=200)
    gross_tonnage: float = Field(gt=0.0, le=250_000.0)
    entrance: str
    ready_at: datetime
    transit_hours: float = Field(gt=0.0, le=12.0)
    tugs: int = Field(ge=1, le=4)
    risk_points: int = Field(ge=1, le=5)
    risk_components: dict[str, float | int | bool] = Field(default_factory=dict)

    @field_validator("entrance")
    @classmethod
    def validate_entrance(cls, entrance: str) -> str:
        value = str(entrance)
        if value not in SUPPORTED_ENTRANCES:
            raise ValueError("entrance must be '1' or '2'")
        return value

    @field_validator("ready_at")
    @classmethod
    def validate_ready_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ready_at must include a timezone offset")
        return value


class DigitalTwinResources(BaseModel):
    """Resources available to the evacuation scheduler."""

    model_config = ConfigDict(extra="forbid")

    tug_capacity: int = Field(ge=2, le=12)
    entrance_capacity: dict[str, int] = Field(
        default_factory=lambda: {"1": 1, "2": 1}
    )

    @field_validator("entrance_capacity")
    @classmethod
    def validate_entrance_capacity(cls, value: dict[str, int]) -> dict[str, int]:
        normalized = {str(key): int(capacity) for key, capacity in value.items()}
        if set(normalized) != SUPPORTED_ENTRANCES:
            raise ValueError("entrance_capacity must contain exactly entrances 1 and 2")
        if any(capacity != 1 for capacity in normalized.values()):
            raise ValueError(
                "the current checkpoints support capacity 1 for each entrance"
            )
        return normalized


class DigitalTwinPreference(BaseModel):
    """One continuous preference requested from the latest Soft MoE."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=r"^[A-Za-z0-9_-]+$", min_length=1, max_length=48)
    label: str | None = Field(default=None, max_length=80)
    weights: list[Annotated[float, Field(ge=0.0)]] = Field(
        min_length=3, max_length=3
    )

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, weights: list[float]) -> list[float]:
        if not all(np.isfinite(weights)) or sum(weights) <= 0.0:
            raise ValueError("weights must have a positive finite sum")
        return weights

    def normalized_weights(self) -> tuple[float, float, float]:
        values = np.asarray(self.weights, dtype=np.float64)
        values /= values.sum()
        return tuple(float(value) for value in values)


class DigitalTwinScenarioRequest(BaseModel):
    """A complete stateless snapshot for RL Pareto scheduling."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    observed_at: datetime
    closure_at: datetime
    resources: DigitalTwinResources
    vessels: list[DigitalTwinVessel] = Field(
        min_length=CURRENT_VESSEL_SLOTS,
        max_length=CURRENT_VESSEL_SLOTS,
    )
    continuous_preferences: list[DigitalTwinPreference] = Field(
        default_factory=list, max_length=12
    )

    @model_validator(mode="after")
    def validate_snapshot(self):
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must include a timezone offset")
        if self.closure_at.tzinfo is None or self.closure_at.utcoffset() is None:
            raise ValueError("closure_at must include a timezone offset")

        remaining = self.closure_hours
        if not 2.0 <= remaining <= 12.0:
            raise ValueError(
                "closure_at must be 2 to 12 hours after observed_at for the "
                "current checkpoints"
            )

        ship_ids = [vessel.ship_id for vessel in self.vessels]
        if len(set(ship_ids)) != len(ship_ids):
            raise ValueError("ship_id values must be unique")
        if any(vessel.tugs > self.resources.tug_capacity for vessel in self.vessels):
            raise ValueError("a vessel cannot require more tugs than tug_capacity")

        reserved_keys = {"count", "balanced", "gt", "risk"}
        custom_keys = [preference.key for preference in self.continuous_preferences]
        if reserved_keys.intersection(custom_keys):
            raise ValueError(
                "continuous preference keys cannot reuse count/balanced/gt/risk"
            )
        if len(set(custom_keys)) != len(custom_keys):
            raise ValueError("continuous preference keys must be unique")
        return self

    @property
    def closure_hours(self) -> float:
        return (self.closure_at - self.observed_at).total_seconds() / 3600.0


@runtime_checkable
class ServiceTimeSource(Protocol):
    """Injectable source of vessel transit/service time."""

    source_id: str

    def transit_hours(self, vessel: DigitalTwinVessel) -> float:
        """Return the expected outbound transit duration in hours."""


class RequestServiceTimeSource:
    """Use service times supplied by the digital twin in the request."""

    source_id = "digital-twin-request.transit_hours.v1"

    def transit_hours(self, vessel: DigitalTwinVessel) -> float:
        return float(vessel.transit_hours)


def build_digital_twin_scenario(
    request: DigitalTwinScenarioRequest,
    *,
    service_time_source: ServiceTimeSource | None = None,
) -> dict:
    """Convert the typed snapshot into the existing environment contract."""

    source = service_time_source or RequestServiceTimeSource()
    vessels = []
    for vessel in request.vessels:
        ready_hour = max(
            0.0,
            (vessel.ready_at - request.observed_at).total_seconds() / 3600.0,
        )
        transit_hour = float(source.transit_hours(vessel))
        if not np.isfinite(transit_hour) or transit_hour <= 0.0:
            raise ValueError(
                f"service time source returned an invalid duration for "
                f"{vessel.ship_id}"
            )
        vessels.append(
            {
                "ship_id": vessel.ship_id,
                "name": vessel.name or vessel.ship_id,
                "gross_tonnage": float(vessel.gross_tonnage),
                "entrance": vessel.entrance,
                "ready_hour": ready_hour,
                "transit_hour": transit_hour,
                "tugs": int(vessel.tugs),
                "risk_points": int(vessel.risk_points),
                "risk_components": dict(vessel.risk_components),
            }
        )

    return {
        "metadata": {
            "name": "digital twin evacuation decision snapshot",
            "request_id": request.request_id,
            "observed_at": request.observed_at.isoformat(),
            "closure_at": request.closure_at.isoformat(),
            "query_date": request.observed_at.date().isoformat(),
            "service_time_source": source.source_id,
            "data_classes": {
                "vessels": "digital_twin_input",
                "resources": "digital_twin_input",
                "closure": "digital_twin_input",
                "risk_points": "digital_twin_input",
            },
        },
        "resources": {
            "tug_capacity": request.resources.tug_capacity,
            "entrance_capacity": dict(request.resources.entrance_capacity),
        },
        "vessels": vessels,
    }


def digital_twin_options(
    request: DigitalTwinScenarioRequest,
    preference: tuple[float, float, float],
) -> dict:
    """Return deterministic environment options for a snapshot."""

    return {
        "closure_hour": request.closure_hours,
        "tug_capacity": request.resources.tug_capacity,
        "demand_compression": 1.0,
        "preference": preference,
        "jitter": False,
        "deterministic": True,
    }


def request_warnings(request: DigitalTwinScenarioRequest) -> list[str]:
    """Describe accepted values that are outside the current training support."""

    lower, upper = TRAINED_TRANSIT_HOUR_RANGE
    outside_transit = sum(
        not lower <= vessel.transit_hours <= upper for vessel in request.vessels
    )
    unavailable_before_closure = sum(
        vessel.ready_at >= request.closure_at for vessel in request.vessels
    )
    warnings = []
    if outside_transit:
        warnings.append(
            f"{outside_transit} vessel service times are outside the approximate "
            f"training range {lower:.2f}-{upper:.2f} hours"
        )
    if unavailable_before_closure:
        warnings.append(
            f"{unavailable_before_closure} vessels are not ready before closure"
        )
    return warnings


def capability_manifest() -> dict:
    """Machine-readable boundary for the first digital-twin integration."""

    return {
        "contract_version": "phase13.snapshot.v1",
        "service_goal": "rl_multi_preference_pareto",
        "policy_candidates": {
            "discrete": ["count", "balanced", "gt", "risk"],
            "continuous_soft_moe": True,
            "single_checkpoint": True,
            "checkpoint_family": "phase14-vd-warmstart-preference-ppo",
            "baseline_candidates_included": False,
            "heuristic_fallback": False,
        },
        "input": {
            "mode": "stateless_receding_horizon_snapshot",
            "vessel_slots": CURRENT_VESSEL_SLOTS,
            "entrances": sorted(SUPPORTED_ENTRANCES),
            "entrance_capacity": {"1": 1, "2": 1},
            "closure_hours": {"min": 2.0, "max": 12.0},
            "service_time_source": "injectable; request transit_hours by default",
            "timestamps_require_timezone": True,
        },
        "output": {
            "all_rl_candidates": True,
            "nondominated_rl_pareto_set": True,
            "next_action_per_pareto_candidate": True,
            "safety_kpis": True,
        },
        "safety": {
            "hard_action_mask": [
                "closure_deadline",
                "tug_capacity",
                "entrance_capacity",
                "vessel_readiness",
            ],
            "unsafe_action_fallback": False,
        },
        "limitations": {
            "active_operations_supported": False,
            "variable_vessel_count_supported": False,
            "current_model_is_experimental": True,
            "phase14_relative_poc_gate_passed": True,
            "single_seed_validation": True,
        },
    }
