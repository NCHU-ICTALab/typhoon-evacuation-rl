from typhoon.planner import compare_policies, plan_evacuation
from typhoon.rules import outbound_closure_deadlines


def _scenario():
    return {
        "metadata": {"data_classes": {}},
        "horizon_hour": 8.0,
        "resources": {
            "tug_capacity": 2,
            "entrance_capacity": {"1": 1},
        },
        "weather": [
            {"hour": 0.0, "wind_bft": 4.0},
            {"hour": 3.0, "wind_bft": 7.0},
        ],
        "vessels": [
            {
                "ship_id": "small-first",
                "name": "Small",
                "gross_tonnage": 10_000.0,
                "entrance": "1",
                "ready_hour": 0.0,
                "transit_hour": 2.0,
                "tugs": 1,
                "risk_points": 1,
            },
            {
                "ship_id": "large-risk",
                "name": "Large",
                "gross_tonnage": 180_000.0,
                "entrance": "1",
                "ready_hour": 0.0,
                "transit_hour": 2.0,
                "tugs": 2,
                "risk_points": 5,
            },
        ],
    }


def test_official_outbound_wind_threshold_is_hard_deadline():
    result = outbound_closure_deadlines(
        _scenario()["weather"], {"1", "2"}, horizon_h=8.0
    )
    assert result["1"].close_hour == 3.0
    assert result["2"].close_hour == 3.0


def test_risk_policy_evacuates_more_risk_under_same_capacity():
    report = compare_policies(_scenario())
    fcfs = report["policies"]["fcfs"]["kpi"]
    risk = report["policies"]["risk_aware"]["kpi"]
    assert fcfs["evacuated_count"] == risk["evacuated_count"] == 1
    assert risk["evacuated_risk_points"] > fcfs["evacuated_risk_points"]
    assert report["policies"]["risk_aware"]["schedule"][0]["ship_id"] == "large-risk"


def test_schedule_never_finishes_after_closure_or_overlaps_entrance():
    report = plan_evacuation(_scenario(), "risk_aware")
    for job in report["schedule"]:
        assert job["finish_hour"] <= job["close_hour"]
    ordered = sorted(report["schedule"], key=lambda job: job["start_hour"])
    for left, right in zip(ordered, ordered[1:]):
        if left["entrance"] == right["entrance"]:
            assert left["finish_hour"] <= right["start_hour"]
