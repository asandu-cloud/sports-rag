"""Shared official-record presentation and bounded refresh; no Discord sends."""
from data_platform.tracking_metrics import record_summary_text


def resolve_official_outcomes():
    """Use the same prospective cutoff/budget/lease as the independent job."""
    from data_platform.services.measurement_cycle import run_measurement_cycle
    report = run_measurement_cycle(kinds=("settlement",))
    result = {"graded": 0, "hit": 0, "miss": 0, "errors": report.get("errors", 0),
              "status": report["status"], "measurement": report}
    for detail in report.get("details", []):
        if detail.get("kind") == "settlement":
            for key in ("graded", "hit", "miss"):
                result[key] += detail.get("result", {}).get(key, 0)
    return result
