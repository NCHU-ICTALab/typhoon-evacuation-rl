"""FastAPI service that executes the trained typhoon MaskablePPO policy."""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .baselines import select_action
from .build_scenario import DEFAULT_DB, build_scenario
from .env import TyphoonEvacuationEnv
from .evaluate_rl import PREFERENCE_PROFILES


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
MODEL_PATH = ROOT / "models" / "final.zip"
MANIFEST_PATH = ROOT / "models" / "training_manifest.json"
_MODEL_LOCK = threading.Lock()

PROFILE_LABELS = {
    "count": "艘數優先",
    "balanced": "平衡方案",
    "gt": "GT 優先",
    "risk": "風險優先",
}


class ScheduleRequest(BaseModel):
    closure_hour: float = Field(4.0, ge=2.0, le=12.0)
    tug_capacity: int = Field(8, ge=2, le=12)
    demand_compression: float = Field(3.0, ge=1.0, le=4.0)
    preference: Literal["count", "balanced", "gt", "risk"] = "balanced"


def _held_out_date() -> str:
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        dates = manifest.get("test_dates") or []
        if dates:
            return str(dates[-1])
    return "2026-07-08"


@lru_cache(maxsize=1)
def get_scenario() -> dict:
    """Build one fixed held-out public-source snapshot without copying the DB."""
    return build_scenario(
        db_path=DEFAULT_DB,
        query_date=_held_out_date(),
        closure_hour=12.0,
        tug_capacity=10,
        max_vessels=30,
        demand_compression=1.0,
    )


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Typhoon RL model not found: {MODEL_PATH}")
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(MODEL_PATH)
    probe = TyphoonEvacuationEnv(get_scenario(), randomize=False)
    if (
        model.observation_space.shape != probe.observation_space.shape
        or model.action_space.n != probe.action_space.n
    ):
        raise ValueError(
            "Typhoon model/environment mismatch: "
            f"obs {model.observation_space.shape} vs {probe.observation_space.shape}, "
            f"actions {model.action_space.n} vs {probe.action_space.n}"
        )
    return model


def _short_name(name: str, ship_id: str) -> str:
    ascii_tokens = [token for token in name.split() if token.isascii()]
    value = " ".join(ascii_tokens[:2]) if ascii_tokens else name
    return (value or ship_id)[:14]


def _vessel_payload(vessel: dict, *, index: int) -> dict:
    name = str(vessel.get("name") or vessel["ship_id"])
    return {
        "index": index,
        "ship_id": str(vessel["ship_id"]),
        "name": name,
        "short_name": _short_name(name, str(vessel["ship_id"])),
        "gross_tonnage": float(vessel["gross_tonnage"]),
        "entrance": str(vessel["entrance"]),
        "ready_hour": round(float(vessel["ready_hour"]), 3),
        "transit_hour": round(float(vessel["transit_hour"]), 3),
        "tugs": int(vessel["tugs"]),
        "risk_points": int(vessel["risk_points"]),
    }


def _preference_payload(key: str, weights: tuple[float, float, float]) -> dict:
    return {
        "key": key,
        "label": PROFILE_LABELS[key],
        "weights": [float(value) for value in weights],
    }


def _run_policy(
    *,
    scenario: dict,
    options: dict,
    policy: Literal["rl", "fcfs", "risk_aware", "value_density"],
    preference_key: str,
    model=None,
) -> dict:
    env = TyphoonEvacuationEnv(scenario, randomize=False)
    observation, _ = env.reset(seed=42, options=options)
    decisions = []
    episode_return = 0.0

    while not env._episode_done:
        mask = env.action_masks()
        feasible = np.flatnonzero(mask[:-1]).tolist()
        if policy == "rl":
            action, _ = model.predict(
                observation, deterministic=True, action_masks=mask
            )
            action = int(action)
        else:
            action = int(select_action(env, policy))

        before = float(env.now)
        if action == env.wait_action:
            decision = {
                "type": "wait",
                "action": action,
                "time": round(before, 3),
            }
        else:
            vessel = env.vessels[action]
            entrance = str(vessel["entrance"])
            entrance_candidates = [
                i
                for i in feasible
                if str(env.vessels[i]["entrance"]) == entrance
            ]
            item = _vessel_payload(vessel, index=action)
            item.update(
                {
                    "start_hour": round(before, 3),
                    "finish_hour": round(
                        before + float(vessel["transit_hour"]), 3
                    ),
                }
            )
            decision = {
                "type": "dispatch",
                "action": action,
                "time": round(before, 3),
                "candidate_count": len(feasible),
                "entrance_candidate_count": len(entrance_candidates),
                "vessel": item,
            }

        observation, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)
        if decision["type"] == "wait":
            decision["until"] = round(float(env.now), 3)
        decisions.append(decision)
        if terminated or truncated:
            break

    schedule = []
    for job in env.schedule:
        item = _vessel_payload(env.vessels[int(job["index"])], index=int(job["index"]))
        item.update(
            {
                "start_hour": round(float(job["start_hour"]), 3),
                "finish_hour": round(float(job["finish_hour"]), 3),
            }
        )
        schedule.append(item)
    remaining = [
        _vessel_payload(env.vessels[index], index=index)
        for index in sorted(env.unscheduled)
    ]
    kpi = env.kpis()
    return {
        "method": policy,
        "preference": _preference_payload(
            preference_key, tuple(float(value) for value in options["preference"])
        ),
        "episode_return": round(episode_return, 6),
        "schedule": schedule,
        "remaining": remaining,
        "decisions": decisions,
        "kpi": {
            "total": env.n_vessels,
            "count": kpi["evacuated_count"],
            "gt": kpi["evacuated_gt"],
            "risk": kpi["evacuated_risk_points"],
            "remaining_count": kpi["remaining_count"],
            "remaining_gt": kpi["remaining_gt"],
            "utility": kpi["weighted_utility"],
            "safety_violations": kpi["safety_violations"],
            "rejected_actions": kpi["rejected_actions"],
        },
    }


def _same_outcome(left: dict, right: dict) -> bool:
    return all(left["kpi"][key] == right["kpi"][key] for key in ("count", "gt", "risk"))


def _dominates(left: dict, right: dict) -> bool:
    keys = ("count", "gt", "risk")
    return all(left["kpi"][key] >= right["kpi"][key] for key in keys) and any(
        left["kpi"][key] > right["kpi"][key] for key in keys
    )


def _pareto(results: list[dict]) -> list[dict]:
    unique = []
    for result in results:
        if not any(_same_outcome(result, other) for other in unique):
            unique.append(result)
    return [
        result
        for result in unique
        if not any(other is not result and _dominates(other, result) for other in unique)
    ]


def calculate(req: ScheduleRequest, *, model, scenario: dict) -> dict:
    selected_weights = PREFERENCE_PROFILES[req.preference]
    base_options = {
        "closure_hour": req.closure_hour,
        "tug_capacity": req.tug_capacity,
        "demand_compression": req.demand_compression,
        "jitter": False,
        "deterministic": True,
    }
    rl_results = {}
    with _MODEL_LOCK:
        for key, weights in PREFERENCE_PROFILES.items():
            rl_results[key] = _run_policy(
                scenario=scenario,
                options={**base_options, "preference": weights},
                policy="rl",
                preference_key=key,
                model=model,
            )

    baselines = {}
    for policy in ("fcfs", "risk_aware", "value_density"):
        baselines[policy] = _run_policy(
            scenario=scenario,
            options={**base_options, "preference": selected_weights},
            policy=policy,
            preference_key=req.preference,
        )

    source = scenario.get("metadata", {})
    return {
        "engine": "python-maskableppo",
        "model": {
            "file": MODEL_PATH.name,
            "observation_size": int(model.observation_space.shape[0]),
            "actions": int(model.action_space.n),
            "trained_steps": 100_000,
        },
        "scenario": {
            "date": source.get("query_date"),
            "vessels": "offline_public_source_snapshot",
            "weather_and_resources": "synthetic_training_only",
            "closure_hour": req.closure_hour,
            "tug_capacity": req.tug_capacity,
            "demand_compression": req.demand_compression,
        },
        "selected_preference": req.preference,
        "baselines": baselines,
        "rl_results": rl_results,
        "pareto_rl": _pareto(list(rl_results.values())),
    }


app = FastAPI(title="颱風封港 Python RL 決策服務", version="0.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/typhoon/health")
def health():
    try:
        model = get_model()
        scenario = get_scenario()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "engine": "python-maskableppo",
        "model": MODEL_PATH.name,
        "observation_size": int(model.observation_space.shape[0]),
        "actions": int(model.action_space.n),
        "scenario_date": scenario.get("metadata", {}).get("query_date"),
    }


@app.post("/api/typhoon/schedule")
def schedule(req: ScheduleRequest):
    try:
        return calculate(req, model=get_model(), scenario=get_scenario())
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def main() -> None:
    import uvicorn

    uvicorn.run("typhoon.api:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
