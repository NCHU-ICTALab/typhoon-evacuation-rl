"""Smoke-test a running clone-ready Phase 14 backend using only stdlib."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.environ.get("TYPHOON_BASE_URL", "http://127.0.0.1:8765").rstrip("/")


def request_json(path: str, *, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = Request(
        f"{BASE_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> None:
    payload = json.loads(
        (REPO_ROOT / "examples" / "digital_twin_request.json").read_text(
            encoding="utf-8"
        )
    )
    try:
        health = request_json("/api/typhoon/health")
        step = request_json(
            "/api/typhoon/digital-twin/step",
            payload=payload,
        )
    except (HTTPError, URLError) as exc:
        raise SystemExit(f"Phase 14 smoke test failed: {exc}") from exc

    assert health["model_family"] == "phase14-vd-warmstart-preference-ppo"
    assert health["status"] == "ok"
    assert step["engine"] == "python-rl-multi-pareto"
    assert step["next_actions"]
    print(
        "Phase 14 smoke test passed: "
        f"{step['pareto_count']} Pareto next-action candidates"
    )


if __name__ == "__main__":
    main()
