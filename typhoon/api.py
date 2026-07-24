"""FastAPI service for Phase 8 experts and experimental Phase 9A Soft MoE."""

from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .baselines import select_action
from .build_scenario import DEFAULT_DB, build_scenario
from .env import TyphoonEvacuationEnv
from .evaluate_rl import PREFERENCE_PROFILES


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
MODEL_FAMILY = "phase8-preference-scalar-critic-experts"
MODEL_ROOT = (
    ROOT
    / "models"
    / "decomposed_ppo_preference_first_scalar_critic_fixed_profiles"
)
MODEL_PATHS = {
    profile: MODEL_ROOT / profile / "seed-42" / "final.pt"
    for profile in PREFERENCE_PROFILES
}
MANIFEST_PATHS = {
    profile: MODEL_ROOT / profile / "seed-42" / "training_manifest.json"
    for profile in PREFERENCE_PROFILES
}
COMPARISON_PATH = MODEL_ROOT / "comparison-seed-42.json"
MODEL_CARD_PATH = ROOT / "model_cards" / "phase8_seed42.json"
SOFT_MOE_PATH = ROOT / "models" / "soft_moe_router_distilled" / "seed-42" / "final.pt"
SOFT_MOE_COMPARISON_PATH = SOFT_MOE_PATH.parent / "comparison.json"
SOFT_MOE_CARD_PATH = ROOT / "model_cards" / "phase9a_seed42.json"
DB_PATH = Path(os.environ.get("TYPHOON_DB_PATH", DEFAULT_DB))
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
    preference_weights: list[Annotated[float, Field(ge=0.0)]] | None = Field(
        default=None, min_length=3, max_length=3
    )

    @field_validator("preference_weights")
    @classmethod
    def validate_preference_weights(cls, weights):
        if weights is not None and (
            not all(np.isfinite(weights)) or sum(weights) <= 0.0
        ):
            raise ValueError("preference_weights must have a positive finite sum")
        return weights


def _held_out_date() -> str:
    manifest_path = MANIFEST_PATHS["balanced"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dates = manifest.get("test_dates") or []
        if dates:
            return str(dates[-1])
    return "2026-07-08"


@lru_cache(maxsize=1)
def get_scenario() -> dict:
    """Build one fixed held-out public-source snapshot without copying the DB."""
    return build_scenario(
        db_path=DB_PATH,
        query_date=_held_out_date(),
        closure_hour=12.0,
        tug_capacity=10,
        max_vessels=30,
        demand_compression=1.0,
    )


def _validate_model(model, probe: TyphoonEvacuationEnv, profile: str) -> None:
    if (
        model.observation_space.shape != probe.observation_space.shape
        or model.action_space.n != probe.action_space.n
    ):
        raise ValueError(
            f"Typhoon {profile} expert/environment mismatch: "
            f"obs {model.observation_space.shape} vs {probe.observation_space.shape}, "
            f"actions {model.action_space.n} vs {probe.action_space.n}"
        )


@lru_cache(maxsize=1)
def get_models() -> dict[str, object]:
    missing = [str(path) for path in MODEL_PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Phase 8 expert checkpoint missing: " + ", ".join(missing)
        )
    from .decomposed_ppo import DecomposedActorCritic

    probe = TyphoonEvacuationEnv(get_scenario(), randomize=False)
    models = {}
    for profile, path in MODEL_PATHS.items():
        model = DecomposedActorCritic.load(path)
        _validate_model(model, probe, profile)
        models[profile] = model
    probe.close()
    return models


@lru_cache(maxsize=1)
def get_soft_moe():
    if not SOFT_MOE_PATH.exists():
        raise FileNotFoundError(f"Phase 9A Soft MoE checkpoint missing: {SOFT_MOE_PATH}")
    from .soft_moe import SoftMoEActorCritic

    model = SoftMoEActorCritic.load(SOFT_MOE_PATH)
    probe = TyphoonEvacuationEnv(get_scenario(), randomize=False)
    _validate_model(model, probe, "continuous")
    probe.close()
    return model


@lru_cache(maxsize=1)
def get_validation_record() -> dict:
    if not COMPARISON_PATH.exists():
        if MODEL_CARD_PATH.exists():
            return json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))
        return {"available": False}
    report = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    summaries = report.get("summary", {})
    return {
        "available": True,
        "seed": int(report.get("seed", 42)),
        "held_out_dates": int(report.get("held_out_dates", 0)),
        "paired_cases_per_profile": int(
            report.get("paired_cases_per_profile", 0)
        ),
        "diagonal_best": report.get("diagonal_best", {}),
        "gt_risk_gate": all(
            report.get("diagonal_best", {}).get(profile, False)
            for profile in ("gt", "risk")
        ),
        "weighted_utility": {
            profile: float(values["weighted_utility"])
            for profile, values in summaries.items()
        },
        "safety_violations": int(
            sum(values.get("safety_violations", 0) for values in summaries.values())
        ),
        "rejected_actions": int(
            sum(values.get("rejected_actions", 0) for values in summaries.values())
        ),
    }


@lru_cache(maxsize=1)
def get_soft_moe_validation_record() -> dict:
    path = (
        SOFT_MOE_COMPARISON_PATH
        if SOFT_MOE_COMPARISON_PATH.exists()
        else SOFT_MOE_CARD_PATH
    )
    if not path.exists():
        return {"available": False}
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "seed": int(report.get("seed", 42)),
        "phase": report.get("phase", "9a_frozen_experts_distilled_router"),
        "diagonal_best": report.get("diagonal_best", {}),
        "diagonal_best_count": int(report.get("diagonal_best_count", 0)),
        "gt_risk_gate": bool(report.get("gt_risk_gate", False)),
        "safety_violations": int(report.get("safety_violations", 0)),
        "paired_base_cases": int(report.get("paired_base_cases", 60)),
        "centroid_routing": report.get("centroid_routing", {}),
    }


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
        "label": PROFILE_LABELS.get(key, "自訂連續偏好"),
        "weights": [float(value) for value in weights],
    }


def _normalize_preference(weights: list[float]) -> tuple[float, float, float]:
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("preference_weights must contain three finite values")
    if np.any(values < 0.0) or float(values.sum()) <= 0.0:
        raise ValueError("preference_weights must be non-negative with a positive sum")
    values /= values.sum()
    return tuple(float(value) for value in values)


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
    initial_routing = None
    if policy == "rl" and hasattr(model, "predict_routing"):
        initial_routing = [
            round(float(value), 6)
            for value in model.predict_routing(observation)[0]
        ]
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
        "expert_profile": preference_key if policy == "rl" else None,
        "routing_weights_initial": initial_routing,
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


def calculate(
    req: ScheduleRequest,
    *,
    scenario: dict,
    models: dict[str, object] | None = None,
    model=None,
    soft_model=None,
) -> dict:
    if models is None:
        if model is None:
            raise ValueError("Phase 8 expert registry is required")
        models = {profile: model for profile in PREFERENCE_PROFILES}
    missing_profiles = set(PREFERENCE_PROFILES) - set(models)
    if missing_profiles:
        raise ValueError(
            "Phase 8 expert registry missing profiles: "
            + ", ".join(sorted(missing_profiles))
        )
    selected_weights = (
        _normalize_preference(req.preference_weights)
        if req.preference_weights is not None
        else PREFERENCE_PROFILES[req.preference]
    )
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
                model=models[key],
            )
        continuous_result = None
        if req.preference_weights is not None:
            if soft_model is None:
                raise ValueError(
                    "Phase 9A Soft MoE model is required for continuous preference"
                )
            continuous_result = _run_policy(
                scenario=scenario,
                options={**base_options, "preference": selected_weights},
                policy="rl",
                preference_key="continuous",
                model=soft_model,
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
    first_model = models["balanced"]
    return {
        "engine": (
            "python-phase8-experts+phase9a-soft-moe"
            if continuous_result is not None
            else "python-decomposed-ppo-phase8-experts"
        ),
        "model": {
            "family": MODEL_FAMILY,
            "display_name": "Phase 8 · 四偏好專家",
            "routing": "fixed-profile-hard-selector",
            "files": {
                profile: str(path.relative_to(ROOT))
                for profile, path in MODEL_PATHS.items()
            },
            "observation_size": int(first_model.observation_space.shape[0]),
            "actions": int(first_model.action_space.n),
            "expert_count": len(models),
            "trained_steps_per_expert": 100_000,
            "validation": get_validation_record(),
            "soft_moe": {
                "available": SOFT_MOE_PATH.exists(),
                "family": "phase9a-frozen-expert-soft-moe",
                "display_name": "Phase 9A · 單一 Soft MoE（實驗性）",
                "routing": "continuous-preference-soft-router",
                "file": str(SOFT_MOE_PATH.relative_to(ROOT)),
                "validation": get_soft_moe_validation_record(),
            },
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
        "selected_weights": list(selected_weights),
        "baselines": baselines,
        "rl_results": rl_results,
        "continuous_result": continuous_result,
        "pareto_rl": _pareto(list(rl_results.values())),
    }


app = FastAPI(title="颱風封港 Python RL 決策服務", version="0.3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/typhoon/health")
def health():
    try:
        models = get_models()
        scenario = get_scenario()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "engine": "python-decomposed-ppo-phase8-experts",
        "model_family": MODEL_FAMILY,
        "routing": "fixed-profile-hard-selector",
        "profiles": list(models),
        "observation_size": int(models["balanced"].observation_space.shape[0]),
        "actions": int(models["balanced"].action_space.n),
        "scenario_date": scenario.get("metadata", {}).get("query_date"),
        "validation": get_validation_record(),
        "soft_moe": {
            "available": SOFT_MOE_PATH.exists(),
            "validation": get_soft_moe_validation_record(),
        },
    }


@app.post("/api/typhoon/schedule")
def schedule(req: ScheduleRequest):
    try:
        soft_model = get_soft_moe() if req.preference_weights is not None else None
        return calculate(
            req,
            models=get_models(),
            soft_model=soft_model,
            scenario=get_scenario(),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def main() -> None:
    import uvicorn

    uvicorn.run("typhoon.api:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
