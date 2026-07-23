"""Build a small offline evacuation scenario from the existing historical DB."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path


DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "ua1008l.sqlite"


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _tugs_for(gt: float) -> int:
    if gt >= 180_000:
        return 4
    if gt >= 100_000:
        return 3
    if gt >= 20_000:
        return 2
    return 1


def _transit_for(gt: float) -> float:
    if gt >= 100_000:
        return 1.5
    if gt >= 20_000:
        return 0.8
    return 0.4


def _synthetic_risk_profile(
    ship_no: str, voyage: str, berth: str, entrance: str
) -> tuple[int, dict[str, int]]:
    """Deterministic training proxy independent of gross tonnage.

    The components mimic fields a production system would obtain from port
    operations. They are synthetic and are never presented as observed facts.
    """
    key = "|".join((ship_no, voyage, berth, entrance)).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    components = {
        "hazardous_cargo_proxy": int(digest[0] % 5 == 0),
        "mobility_constraint_proxy": int(digest[1] % 4 == 0),
        "deep_draft_proxy": int(digest[2] % 3 == 0),
        "engine_readiness_proxy": int(digest[3] % 5 == 0),
    }
    score = (
        1
        + 2 * components["hazardous_cargo_proxy"]
        + components["mobility_constraint_proxy"]
        + components["deep_draft_proxy"]
        + components["engine_readiness_proxy"]
    )
    return min(5, score), components


def _choose_date(con: sqlite3.Connection) -> str:
    row = con.execute(
        """
        SELECT query_date, COUNT(*) AS n
        FROM movements
        WHERE direction='出港'
          AND pilot_apply_time IS NOT NULL
          AND gross_tonnage > 0
          AND port_entrance IN ('1', '2')
        GROUP BY query_date
        ORDER BY n DESC, query_date
        LIMIT 1
        """
    ).fetchone()
    if not row:
        raise ValueError("No usable outbound records in ua1008l.sqlite")
    return str(row[0])


def available_scenario_dates(
    db_path: str | Path = DEFAULT_DB,
    *,
    min_vessels: int = 30,
) -> list[str]:
    """Return offline dates with enough usable outbound vessel records."""
    path = Path(db_path).resolve()
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            """
            SELECT query_date
            FROM movements
            WHERE direction='出港'
              AND pilot_apply_time IS NOT NULL
              AND gross_tonnage > 0
              AND port_entrance IN ('1', '2')
            GROUP BY query_date
            HAVING COUNT(*) >= ?
            ORDER BY query_date
            """,
            (int(min_vessels),),
        ).fetchall()
    finally:
        con.close()
    return [str(row[0]) for row in rows]


def build_scenario_pool(
    db_path: str | Path = DEFAULT_DB,
    *,
    dates: list[str] | None = None,
    max_vessels: int = 30,
) -> list[dict]:
    """Build same-sized scenarios from multiple dates without copying the DB."""
    selected_dates = dates or available_scenario_dates(
        db_path, min_vessels=max_vessels
    )
    return [
        build_scenario(
            db_path=db_path,
            query_date=date,
            closure_hour=12.0,
            tug_capacity=10,
            max_vessels=max_vessels,
            demand_compression=1.0,
        )
        for date in selected_dates
    ]


def _synthetic_weather(closure_hour: float, horizon_h: float) -> list[dict]:
    """Generate a monotonic training-only wind ramp with an exact closure point."""
    if closure_hour < 0 or closure_hour > horizon_h:
        raise ValueError("closure_hour must be within the scenario horizon")

    end = int(math.ceil(horizon_h))
    points = []
    for hour in range(end + 1):
        if closure_hour <= 0:
            wind = 7.0
        elif hour <= closure_hour:
            wind = 4.0 + 3.0 * hour / closure_hour
        else:
            wind = min(9.0, 7.0 + 0.25 * (hour - closure_hour))
        points.append({"hour": float(hour), "wind_bft": round(wind, 2)})
    if closure_hour not in {p["hour"] for p in points}:
        points.append({"hour": float(closure_hour), "wind_bft": 7.0})
        points.sort(key=lambda p: p["hour"])
    return points


def build_scenario(
    db_path: str | Path = DEFAULT_DB,
    query_date: str | None = None,
    closure_hour: float = 8.0,
    horizon_h: float = 24.0,
    tug_capacity: int = 8,
    max_vessels: int = 30,
    demand_compression: float = 2.0,
) -> dict:
    """Create a compact scenario without copying or mutating the source DB."""

    path = Path(db_path).resolve()
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        selected_date = query_date or _choose_date(con)
        rows = con.execute(
            """
            SELECT ship_no, voyage, ship_name, gross_tonnage, berth_code,
                   port_entrance, pilot_apply_time
            FROM movements
            WHERE query_date=? AND direction='出港'
              AND pilot_apply_time IS NOT NULL
              AND gross_tonnage > 0
              AND port_entrance IN ('1', '2')
            ORDER BY pilot_apply_time, ship_no, voyage
            LIMIT ?
            """,
            (selected_date, int(max_vessels)),
        ).fetchall()
    finally:
        con.close()

    if not rows:
        raise ValueError(f"No usable outbound records for {selected_date}")
    if demand_compression <= 0:
        raise ValueError("demand_compression must be positive")

    t0 = min(_parse_time(row[6]) for row in rows)
    vessels = []
    for ship_no, voyage, name, gt_raw, berth, entrance, apply_time in rows:
        gt = float(gt_raw)
        risk_points, risk_components = _synthetic_risk_profile(
            str(ship_no), str(voyage), str(berth or ""), str(entrance)
        )
        ready_h = (_parse_time(apply_time) - t0).total_seconds() / 3600.0
        vessels.append(
            {
                "ship_id": f"{ship_no}-{voyage}",
                "name": name or f"ship-{ship_no}",
                "gross_tonnage": gt,
                "berth_code": berth or "",
                "entrance": str(entrance),
                "historical_pilot_apply_time": apply_time,
                "ready_hour": round(ready_h / demand_compression, 3),
                "transit_hour": _transit_for(gt),
                "tugs": _tugs_for(gt),
                "risk_points": risk_points,
                "risk_components": risk_components,
            }
        )

    return {
        "metadata": {
            "name": "offline typhoon evacuation training scenario",
            "query_date": selected_date,
            "scenario_start": t0.isoformat(),
            "source_db": str(path),
            "data_classes": {
                "vessels": "existing historical public-source snapshot",
                "weather": "synthetic_training_only",
                "resources": "synthetic_training_only",
                "risk_points": "synthetic_independent_proxy_not_official_priority",
                "closure_rule": "public_official_rule",
            },
            "assumptions": {
                "demand_compression": demand_compression,
                "transit_and_tug_tables": (
                    "reused track2 research knobs, not measured capacity"
                ),
            },
        },
        "horizon_hour": float(horizon_h),
        "resources": {
            "tug_capacity": int(tug_capacity),
            "entrance_capacity": {"1": 1, "2": 1},
        },
        "weather": _synthetic_weather(float(closure_hour), float(horizon_h)),
        "vessels": vessels,
    }


def write_scenario(scenario: dict, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(scenario, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out
