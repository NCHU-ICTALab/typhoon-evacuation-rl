"""Explainable offline baselines for pre-closure vessel evacuation."""

from __future__ import annotations

from collections import Counter

from .rules import RULE_VERSION, outbound_closure_deadlines


def _policy_key(policy: str, vessel: dict, now: float, close_hour: float) -> tuple:
    if policy == "fcfs":
        return (float(vessel["ready_hour"]), int(vessel["_source_order"]))
    if policy == "risk_aware":
        slack = close_hour - now - float(vessel["transit_hour"])
        return (
            slack,
            -int(vessel["risk_points"]),
            -float(vessel["gross_tonnage"]),
            -int(vessel["tugs"]),
            float(vessel["ready_hour"]),
            vessel["ship_id"],
        )
    raise ValueError(f"Unknown policy: {policy}")


def plan_evacuation(scenario: dict, policy: str) -> dict:
    vessels = [dict(v) for v in scenario["vessels"]]
    for source_order, vessel in enumerate(vessels):
        vessel["_source_order"] = source_order
    entrances = {str(v["entrance"]) for v in vessels}
    deadlines = outbound_closure_deadlines(
        scenario["weather"], entrances, float(scenario["horizon_hour"])
    )
    tug_capacity = int(scenario["resources"]["tug_capacity"])
    entrance_capacity = {
        str(k): int(v) for k, v in scenario["resources"]["entrance_capacity"].items()
    }
    if tug_capacity <= 0:
        raise ValueError("tug_capacity must be positive")

    unscheduled = {v["ship_id"]: v for v in vessels}
    active: list[dict] = []
    schedule: list[dict] = []
    now = min([float(v["ready_hour"]) for v in vessels] or [0.0])
    eps = 1e-9

    while unscheduled:
        active = [job for job in active if float(job["finish_hour"]) > now + eps]
        used_tugs = sum(int(job["tugs"]) for job in active)
        used_entrances = Counter(str(job["entrance"]) for job in active)

        candidates = []
        for vessel in unscheduled.values():
            entrance = str(vessel["entrance"])
            close = deadlines[entrance].close_hour
            finish = now + float(vessel["transit_hour"])
            if float(vessel["ready_hour"]) > now + eps:
                continue
            if finish > close + eps:
                continue
            if used_tugs + int(vessel["tugs"]) > tug_capacity:
                continue
            if used_entrances[entrance] >= entrance_capacity.get(entrance, 1):
                continue
            candidates.append(vessel)

        if candidates:
            selected = min(
                candidates,
                key=lambda vessel: _policy_key(
                    policy,
                    vessel,
                    now,
                    deadlines[str(vessel["entrance"])].close_hour,
                ),
            )
            finish = now + float(selected["transit_hour"])
            record = {
                **{k: v for k, v in selected.items() if not k.startswith("_")},
                "start_hour": round(now, 3),
                "finish_hour": round(finish, 3),
                "close_hour": deadlines[str(selected["entrance"])].close_hour,
            }
            schedule.append(record)
            active.append(record)
            del unscheduled[selected["ship_id"]]
            continue

        future = [
            float(job["finish_hour"])
            for job in active
            if float(job["finish_hour"]) > now + eps
        ]
        future.extend(
            float(v["ready_hour"])
            for v in unscheduled.values()
            if now + eps < float(v["ready_hour"])
            < deadlines[str(v["entrance"])].close_hour - eps
        )
        if not future:
            break
        now = min(future)

    remaining = []
    for vessel in sorted(unscheduled.values(), key=lambda v: v["ship_id"]):
        close = deadlines[str(vessel["entrance"])].close_hour
        if float(vessel["ready_hour"]) + float(vessel["transit_hour"]) > close + eps:
            reason = "not_ready_with_enough_transit_time_before_closure"
        elif int(vessel["tugs"]) > tug_capacity:
            reason = "tug_requirement_exceeds_scenario_capacity"
        else:
            reason = "resource_or_entrance_capacity_exhausted_before_closure"
        public_vessel = {
            k: v for k, v in vessel.items() if not k.startswith("_")
        }
        remaining.append({**public_vessel, "close_hour": close, "reason": reason})

    return {
        "policy": policy,
        "rule_version": RULE_VERSION,
        "closure": {
            entrance: {
                "close_hour": result.close_hour,
                "reason": result.reason,
            }
            for entrance, result in deadlines.items()
        },
        "kpi": {
            "total_vessels": len(vessels),
            "evacuated_count": len(schedule),
            "evacuated_gt": round(sum(v["gross_tonnage"] for v in schedule), 1),
            "evacuated_risk_points": sum(v["risk_points"] for v in schedule),
            "remaining_count": len(remaining),
            "remaining_gt": round(sum(v["gross_tonnage"] for v in remaining), 1),
            "remaining_risk_points": sum(v["risk_points"] for v in remaining),
        },
        "schedule": schedule,
        "remaining": remaining,
    }


def compare_policies(scenario: dict) -> dict:
    reports = {
        policy: plan_evacuation(scenario, policy)
        for policy in ("fcfs", "risk_aware")
    }
    fcfs = reports["fcfs"]["kpi"]
    risk = reports["risk_aware"]["kpi"]
    return {
        "scenario_metadata": scenario["metadata"],
        "policies": reports,
        "delta_risk_aware_minus_fcfs": {
            "evacuated_count": risk["evacuated_count"] - fcfs["evacuated_count"],
            "evacuated_gt": round(risk["evacuated_gt"] - fcfs["evacuated_gt"], 1),
            "evacuated_risk_points": (
                risk["evacuated_risk_points"] - fcfs["evacuated_risk_points"]
            ),
            "remaining_count": risk["remaining_count"] - fcfs["remaining_count"],
        },
    }
