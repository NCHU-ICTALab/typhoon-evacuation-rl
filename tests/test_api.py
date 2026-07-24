import json
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


class FakeSoftMoE(FakeMaskablePPO):
    def predict_routing(self, observation):
        preference = np.asarray(observation[-3:], dtype=np.float32)
        return np.asarray(
            [[preference[0], 0.0, preference[1], preference[2]]],
            dtype=np.float32,
        )


def test_calculate_executes_model_and_python_baselines():
    models = {
        profile: FakeMaskablePPO()
        for profile in ("count", "balanced", "gt", "risk")
    }
    result = typhoon_api.calculate(
        typhoon_api.ScheduleRequest(
            closure_hour=3,
            tug_capacity=2,
            demand_compression=1,
            preference="risk",
        ),
        models=models,
        scenario=_scenario(),
    )

    assert result["engine"] == "python-decomposed-ppo-phase8-experts"
    assert all(model.calls > 0 for model in models.values())
    assert set(result["rl_results"]) == {"count", "balanced", "gt", "risk"}
    assert all(
        result["rl_results"][profile]["expert_profile"] == profile
        for profile in models
    )
    assert set(result["baselines"]) == {"fcfs", "risk_aware", "value_density"}
    assert result["rl_results"]["risk"]["kpi"]["safety_violations"] == 0
    assert result["scenario"]["date"] == "held-out-test"


def test_calculate_runs_one_checkpoint_with_normalized_continuous_preference():
    models = {
        profile: FakeMaskablePPO()
        for profile in ("count", "balanced", "gt", "risk")
    }
    soft_model = FakeSoftMoE()
    result = typhoon_api.calculate(
        typhoon_api.ScheduleRequest(
            closure_hour=3,
            tug_capacity=2,
            demand_compression=1,
            preference_weights=[2, 3, 5],
        ),
        models=models,
        soft_model=soft_model,
        scenario=_scenario(),
    )

    continuous = result["continuous_result"]
    assert soft_model.calls > 0
    assert result["engine"] == "python-phase8-experts+phase9b-soft-moe"
    assert result["selected_weights"] == [0.2, 0.3, 0.5]
    assert continuous["preference"]["key"] == "continuous"
    assert continuous["preference"]["weights"] == [0.2, 0.3, 0.5]
    assert continuous["routing_weights_initial"] == [0.2, 0.0, 0.3, 0.5]
    assert continuous["kpi"]["safety_violations"] == 0


def test_schedule_endpoint_never_silently_falls_back(monkeypatch):
    models = {
        profile: FakeMaskablePPO()
        for profile in ("count", "balanced", "gt", "risk")
    }
    monkeypatch.setattr(typhoon_api, "get_models", lambda: models)
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
    assert response.json()["engine"] == "python-decomposed-ppo-phase8-experts"
    assert all(model.calls > 0 for model in models.values())

    def missing_models():
        raise FileNotFoundError("missing Phase 8 experts")

    monkeypatch.setattr(typhoon_api, "get_models", missing_models)
    failed = client.post("/api/typhoon/schedule", json={})
    assert failed.status_code == 503
    assert "missing Phase 8 experts" in failed.json()["detail"]


def test_schedule_rejects_invalid_continuous_preference():
    client = TestClient(typhoon_api.app)
    response = client.post(
        "/api/typhoon/schedule",
        json={"preference_weights": [0, 0, 0]},
    )
    assert response.status_code == 422


def test_frontend_identifies_phase8_expert_routing():
    client = TestClient(typhoon_api.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Phase 8 Python RL experts" in response.text
    assert "Phase 8 held-out 紀錄" in response.text
    assert "實驗性連續偏好" in response.text
    assert "Phase 9B RL-trained Soft MoE" in response.text


def test_validation_record_falls_back_to_tracked_model_card(
    monkeypatch, tmp_path
):
    comparison = tmp_path / "missing-comparison.json"
    model_card = tmp_path / "model-card.json"
    model_card.write_text(
        json.dumps(
            {
                "available": True,
                "seed": 42,
                "gt_risk_gate": True,
                "safety_violations": 0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(typhoon_api, "COMPARISON_PATH", comparison)
    monkeypatch.setattr(typhoon_api, "MODEL_CARD_PATH", model_card)
    typhoon_api.get_validation_record.cache_clear()
    try:
        record = typhoon_api.get_validation_record()
        assert record["gt_risk_gate"] is True
        assert record["safety_violations"] == 0
    finally:
        typhoon_api.get_validation_record.cache_clear()


def test_soft_moe_validation_falls_back_to_phase9b_model_card(
    monkeypatch, tmp_path
):
    comparison = tmp_path / "missing-phase9b-comparison.json"
    model_card = tmp_path / "phase9b-model-card.json"
    model_card.write_text(
        json.dumps(
            {
                "available": True,
                "phase": "9b_frozen_experts_router_ppo",
                "seed": 42,
                "gt_risk_gate": True,
                "accepted_for_frontend": True,
                "safety_violations": 0,
                "continuous_grid": {
                    "grid_points": 15,
                    "continuous_grid_gate": True,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        typhoon_api, "SOFT_MOE_COMPARISON_PATH", comparison
    )
    monkeypatch.setattr(typhoon_api, "SOFT_MOE_CARD_PATH", model_card)
    typhoon_api.get_soft_moe_validation_record.cache_clear()
    try:
        record = typhoon_api.get_soft_moe_validation_record()
        assert record["phase"] == "9b_frozen_experts_router_ppo"
        assert record["accepted_for_frontend"] is True
        assert record["continuous_grid"]["continuous_grid_gate"] is True
    finally:
        typhoon_api.get_soft_moe_validation_record.cache_clear()
