from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

import typhoon.api as typhoon_api


def _scenario():
    return {
        "metadata": {"query_date": "held-out-test"},
        "resources": {
            "tug_capacity": 2,
            "entrance_capacity": {"1": 1, "2": 1},
        },
        "vessels": [
            {
                "ship_id": "first",
                "name": "FIRST 一號",
                "gross_tonnage": 10_000.0,
                "entrance": "1",
                "ready_hour": 0.0,
                "transit_hour": 1.0,
                "tugs": 1,
                "risk_points": 1,
            },
            {
                "ship_id": "second",
                "name": "SECOND 二號",
                "gross_tonnage": 100_000.0,
                "entrance": "2",
                "ready_hour": 0.0,
                "transit_hour": 1.0,
                "tugs": 1,
                "risk_points": 4,
            },
        ],
    }


class FakeMaskablePPO:
    def __init__(self):
        self.calls = 0
        self.observation_space = SimpleNamespace(shape=(36,))
        self.action_space = SimpleNamespace(n=3)

    def predict(self, observation, *, deterministic, action_masks):
        self.calls += 1
        assert deterministic is True
        feasible = np.flatnonzero(action_masks[:-1])
        action = int(feasible[0]) if len(feasible) else len(action_masks) - 1
        return action, None


def test_calculate_executes_model_and_python_baselines():
    model = FakeMaskablePPO()
    result = typhoon_api.calculate(
        typhoon_api.ScheduleRequest(
            closure_hour=3,
            tug_capacity=2,
            demand_compression=1,
            preference="risk",
        ),
        model=model,
        scenario=_scenario(),
    )

    assert result["engine"] == "python-maskableppo"
    assert model.calls > 0
    assert set(result["rl_results"]) == {"count", "balanced", "gt", "risk"}
    assert set(result["baselines"]) == {"fcfs", "risk_aware", "value_density"}
    assert result["rl_results"]["risk"]["kpi"]["safety_violations"] == 0
    assert result["scenario"]["date"] == "held-out-test"


def test_schedule_endpoint_never_silently_falls_back(monkeypatch):
    model = FakeMaskablePPO()
    monkeypatch.setattr(typhoon_api, "get_model", lambda: model)
    monkeypatch.setattr(typhoon_api, "get_scenario", _scenario)
    client = TestClient(typhoon_api.app)

    response = client.post(
        "/api/typhoon/schedule",
        json={
            "closure_hour": 3,
            "tug_capacity": 2,
            "demand_compression": 1,
            "preference": "balanced",
        },
    )
    assert response.status_code == 200
    assert response.json()["engine"] == "python-maskableppo"
    assert model.calls > 0

    def missing_model():
        raise FileNotFoundError("missing final.zip")

    monkeypatch.setattr(typhoon_api, "get_model", missing_model)
    failed = client.post("/api/typhoon/schedule", json={})
    assert failed.status_code == 503
    assert "missing final.zip" in failed.json()["detail"]
