import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import typhoon.api as typhoon_api
from typhoon.digital_twin import DigitalTwinScenarioRequest


ROOT = Path(__file__).resolve().parents[1]


class CloneReadyModel:
    observation_space = SimpleNamespace(shape=(344,))
    action_space = SimpleNamespace(n=31)


def test_tracked_phase14_checkpoint_matches_service_sha():
    checkpoint = (
        ROOT
        / "typhoon"
        / "models"
        / "soft_moe_vd_warmstart_ppo"
        / "seed-42"
        / "service_candidate.pt"
    )
    assert checkpoint.is_file()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
        typhoon_api.SOFT_MOE_SHA256
    )


def test_demo_snapshot_is_a_valid_frontend_request():
    payload = json.loads(
        (ROOT / "examples" / "digital_twin_request.json").read_text(
            encoding="utf-8"
        )
    )
    request = DigitalTwinScenarioRequest.model_validate(payload)
    assert len(request.vessels) == 30
    assert request.closure_hours == 4.0
    assert len(request.continuous_preferences) == 2


def test_default_health_reports_phase14(monkeypatch):
    monkeypatch.setattr(
        typhoon_api,
        "get_soft_moe",
        lambda: CloneReadyModel(),
    )
    client = TestClient(typhoon_api.app)
    response = client.get("/api/typhoon/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["engine"] == "python-rl-multi-pareto"
    assert body["model_family"] == "phase14-vd-warmstart-preference-ppo"
    assert body["model_sha256"] == typhoon_api.SOFT_MOE_SHA256
    assert body["observation_size"] == 344
    assert body["actions"] == 31


def test_cors_origins_are_configurable(monkeypatch):
    monkeypatch.setenv(
        "TYPHOON_CORS_ORIGINS",
        "http://localhost:3000, https://twin.example",
    )
    assert typhoon_api._cors_origins() == [
        "http://localhost:3000",
        "https://twin.example",
    ]
