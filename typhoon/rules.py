"""Public hard constraints for the typhoon evacuation PoC.

The first slice only encodes the outbound rule required by the PoC. During a
typhoon, the public Kaohsiung control baseline suspends outbound traffic when
the measured 15-minute mean wind reaches Beaufort 7 at either entrance.
Wave thresholds in the same document suspend inbound traffic and therefore are
deliberately not reused as outbound rules.
"""

from __future__ import annotations

from dataclasses import dataclass


RULE_VERSION = "Kaohsiung typhoon rules 115.03 / entry-exit control baseline"
OUTBOUND_WIND_CLOSE_BFT = 7.0


@dataclass(frozen=True)
class ClosureRuleResult:
    entrance: str
    close_hour: float
    reason: str


def outbound_closure_deadlines(
    weather: list[dict], entrances: set[str], horizon_h: float
) -> dict[str, ClosureRuleResult]:
    """Return the first forecast hour at which outbound traffic must stop."""

    deadline = float(horizon_h)
    triggered = False
    for point in sorted(weather, key=lambda item: float(item["hour"])):
        if float(point["wind_bft"]) >= OUTBOUND_WIND_CLOSE_BFT:
            deadline = float(point["hour"])
            triggered = True
            break

    reason = (
        f"15-minute mean wind reaches Beaufort {OUTBOUND_WIND_CLOSE_BFT:g}; "
        "outbound traffic suspended"
        if triggered
        else "no outbound wind closure within scenario horizon"
    )
    return {
        entrance: ClosureRuleResult(entrance, deadline, reason)
        for entrance in sorted(entrances)
    }
