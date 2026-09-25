"""Read-only Phase 2 acceptance facts. Never treats an empty sample as approval."""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import argparse
import json
from pathlib import Path
import sqlite3

from sqlalchemy import create_engine, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from Scripts.data_platform import config
from Scripts.data_platform.models import PublishedRecommendation
from Scripts.data_platform.publication_reporting import publication_metadata
from Scripts.data_platform.repositories.predictions import PredictionRepository
from Scripts.data_platform.services.measurement_runtime import CONTROL, MeasurementRuntime
from Scripts.data_platform.settlement import selection_issue, utc_datetime
from Scripts.ops.prediction_tracking_audit import audit


def build_report(repo, *, now=None):
    now = now or datetime.now(timezone.utc)
    runtime = MeasurementRuntime(repo._factory)
    control = runtime.state(CONTROL)
    rows = repo.settlement_candidates(include_graded=True, published_after_id=0)
    with repo._factory() as session:
        metadata = publication_metadata(session, [p["id"] for p in rows])
    cutoff = control.get("after_recommendation_id")
    prospective = [p for p in rows if cutoff is not None and p["recommendation_id"] > cutoff]
    identity_issues = []
    pending = []
    for p in rows:
        issue = selection_issue(p)
        if p in prospective and (issue or not metadata[p["id"]].get("system_version")):
            identity_issues.append({"prediction_id": p["id"], "issue": issue or "missing_system_version"})
        if p["outcome"] is not None:
            continue
        kickoff = utc_datetime(p.get("scheduled_kickoff") or p.get("kickoff"))
        due = kickoff is not None and kickoff + timedelta(hours=3) <= now
        settlement = p.get("settlement") or {}
        reason = settlement.get("pending_reason") or issue or "awaiting_result_or_review"
        retry = ("Recover the original recorded market identity; do not infer it" if issue else
                 "Check saved settlement policy, exact market and compatible evidence; never retrofit old rows" if "rule" in reason or "definition" in reason else
                 "Wait for kickoff/result" if not due else "Retry with authoritative fixture result/statistics")
        pending.append({"prediction_id": p["id"], "fixture_id": p["fixture_id"], "overdue": due,
                        "scheduled_kickoff": kickoff.isoformat() if kickoff else None,
                        "hours_since_kickoff": round((now - kickoff).total_seconds() / 3600, 2) if due else None,
                        "reason": reason, "next_action": retry,
                        "last_checked_at": settlement.get("last_checked_at")})
    cohorts = {scope: repo.get_track_record(published_only=True, publication_scope=scope)
               for scope in ("initial", "amendments", "all")}
    from Scripts.web_app.track_record_api import TrackRecordResponse
    mismatches = {scope: sorted(k for k, value in stats.items()
                              if TrackRecordResponse(**stats).model_dump().get(k) != value)
                  for scope, stats in cohorts.items()}
    settled = sum(p["outcome"] is not None for p in prospective)
    closed = sum((p.get("closing_capture") or {}).get("status") == "final" for p in prospective)
    gates = {
        "prospective_identity": "failed" if identity_issues else "passed" if prospective else "awaiting_live_sample",
        "public_reporting_contract": "failed" if any(mismatches.values()) else "passed",
        "prospective_settlement_observed": "observed_review_required" if settled else "awaiting_live_sample",
        "prospective_closing_observed": "observed_review_required" if closed else "awaiting_live_sample",
        "product_settlement_policy": "approved_for_new_publications_live_acceptance_pending",
        "bookmaker_equivalent_settlement": "not_claimed_product_rules_used",
    }
    return {"schema": "phase2-acceptance-audit.v1", "captured_at": now.isoformat(),
            "api_calls": 0, "production_writes": False, "full_phase2_signoff": False,
            "gates": gates, "prospective_cutoff": cutoff, "prospective_publications": len(prospective),
            "prospective_settled": settled, "prospective_closing_final": closed,
            "identity_issues": identity_issues, "reporting_mismatches": mismatches,
            "published_count": len(rows), "pending_count": len(pending),
            "overdue_pending": sum(p["overdue"] for p in pending),
            "pending_reasons": dict(Counter(p["reason"] for p in pending)),
            "pending_records": pending, "cohort_reports": cohorts,
            "runtime": runtime.status(now=now)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    url = make_url(config.SETTINGS.database_url)
    if url.get_backend_name() != "sqlite" or not url.database:
        parser.error("Read-only local audit requires SQLite")
    database = Path(url.database).resolve()
    def connect():
        conn = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        conn.execute("PRAGMA query_only=ON")
        return conn
    engine = create_engine("sqlite://", creator=connect)
    @contextmanager
    def scope():
        with Session(engine) as session:
            yield session
    report = build_report(PredictionRepository(scope))
    report["linkage_audit"] = audit(database)
    if report["linkage_audit"].get("visible_selections_missing_publication_link"):
        report["gates"]["visible_selection_links"] = "failed"
    else:
        report["gates"]["visible_selection_links"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(json.dumps({k: report[k] for k in ("gates", "published_count", "pending_count", "overdue_pending",
                                            "prospective_publications", "pending_reasons", "full_phase2_signoff")}, indent=2))
    return int("failed" in report["gates"].values())


if __name__ == "__main__":
    raise SystemExit(main())
