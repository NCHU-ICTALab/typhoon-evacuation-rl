from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import typhoon.api as typhoon_api
from typhoon.digital_twin import (
    DigitalTwinScenarioRequest,
    build_digital_twin_scenario,
    capability_manifest,
)


class FirstFeasiblePolicy:
    def __init__(self):
        self.calls = 0
        self.observation_space = SimpleNamespace(shape=(344,))
        self.action_space = SimpleNamespace(n=31)

    def predict(self, observation, *, deterministic, action_masks):
        self.calls += 1
        feasible = np.flatnonzero(action_masks[:-1])
        action = int(feasible[0]) if len(feasible) else len(action_masks) - 1
        return action, None


class PreferenceAwarePolicy(FirstFeasiblePolicy):
    def predict_routing(self, observation):
        count, gt, risk = observation[-3:]
        return np.asarray([[count, 0.0, gt, risk]], dtype=np.float32)


def _request_payload(*, vessel_count=30, continuous=True):
    observed_at = datetime(2026, 7, 26, 8, tzinfo=timezone(timedelta(hours=8)))
    vessels = []
    for index in range(vessel_count):
        vessels.append(
            {
                "ship_id": f"ship-{index:02d}",
                "name": f"SHIP {index:02d}",
                "gross_tonnage": 10_000 + index * 1_000,
                "entrance": "1" if index % 2 == 0 else "2",
                "ready_at": (observed_at + timedelta(minutes=index)).isoformat(),
                "transit_hours": 0.5,
                "tugs": 1,
                "risk_points": 1 + index % 5,
            }
        )
    payload = {
        "request_id": "twin-snapshot-001",
        "observed_at": observed_at.isoformat(),
        "closure_at": (observed_at + timedelta(hours=4)).isoformat(),
        "resources": {
            "tug_capacity": 8,
            "entrance_capacity": {"1": 1, "2": 1},
        },
        "vessels": vessels,
    }
    if continuous:
        payload["continuous_preferences"] = [
            {
                "key": "gt-risk",
                "label": "GT／風險折衷",
                "weights": [0.1, 0.45, 0.45],
            }
        ]
    return payload


def test_snapshot_contract_requires_current_30_vessel_shape():
    with pytest.raises(ValidationError, match="at least 30"):
        DigitalTwinScenarioRequest.model_validate(
            _request_payload(vessel_count=29)
        )


def test_snapshot_contract_rejects_duplicate_ship_ids():
    payload = _request_payload()
    payload["vessels"][1]["ship_id"] = payload["vessels"][0]["ship_id"]
    with pytest.raises(ValidationError, match="ship_id values must be unique"):
        DigitalTwinScenarioRequest.model_validate(payload)


def test_service_time_source_is_injectable():
    request = DigitalTwinScenarioRequest.model_validate(_request_payload())

    class FixedSource:
        source_id = "test-fixed-source"

        def transit_hours(self, vessel):
            return 0.75

    scenario = build_digital_twin_scenario(
        request, service_time_source=FixedSource()
    )
    assert scenario["metadata"]["service_time_source"] == "test-fixed-source"
    assert {vessel["transit_hour"] for vessel in scenario["vessels"]} == {0.75}
    assert scenario["vessels"][0]["ready_hour"] == 0.0


def test_rl_pareto_service_contains_no_rule_candidates():
    request = DigitalTwinScenarioRequest.model_validate(_request_payload())
    model = PreferenceAwarePolicy()

    result = typhoon_api.calculate_digital_twin(
        request,
        model=model,
    )

    assert result["service_goal"] == "rl_multi_preference_pareto"
    assert result["candidate_count"] == 5
    assert all(candidate["method"] == "rl" for candidate in result["rl_candidates"])
    assert all(
        candidate["policy_source"]
        == "phase10-monotonicity-regularized-soft-moe"
        for candidate in result["rl_candidates"]
    )
    assert result["model"]["family"] == result["rl_candidates"][0]["policy_source"]
    assert "baselines" not in result
    assert result["pareto_count"] >= 1
    assert model.calls > 0


def test_capabilities_and_step_endpoint(monkeypatch):
    model = PreferenceAwarePolicy()
    monkeypatch.setattr(typhoon_api, "get_soft_moe", lambda: model)
    client = TestClient(typhoon_api.app)

    capabilities = client.get("/api/typhoon/digital-twin/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json() == capability_manifest()
    assert (
        capabilities.json()["policy_candidates"]["baseline_candidates_included"]
        is False
    )

    health = client.get("/api/typhoon/digital-twin/health")
    assert health.status_code == 200
    assert health.json()["model_family"] == (
        "phase10-monotonicity-regularized-soft-moe"
    )
    assert health.json()["observation_size"] == 344
    assert health.json()["actions"] == 31
    assert health.json()["model_sha256"] == (
        "e6abde454606082b51ef88e5da503a5be4505ec088469ffaf86b30ff6ae0dbf5"
    )

    response = client.post(
        "/api/typhoon/digital-twin/step",
        json=_request_payload(continuous=False),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "python-rl-multi-pareto"
    assert body["next_actions"]
    assert all(item["next_decision"] is not None for item in body["next_actions"])
