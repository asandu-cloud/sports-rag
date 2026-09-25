"""Offline end-to-end acceptance. Never posts, uses a live DB or calls a provider."""
from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from data_platform.models import Prediction, PublishedRecommendation, OddsSnapshot
from data_platform.services.historical_settlement import HistoricalSettlementRecovery, SavedProviderEvidence
from data_platform.services.measurement_cycle import MeasurementCycle
from data_platform.services.measurement_runtime import MeasurementRuntime
from data_platform.services.settlement import SettlementService
from data_platform.settlement import parse_fixture_result
from data_platform.tracking_metrics import build_track_record, record_summary_text
from .test_safe_settlement import store, publish, payload, fixture, team_stats, NOW, no_network
from .test_publication_lineage import services, save, visible
from .test_tracking_contract import _published_market_result
from .test_measurement_cycle import Client, KICKOFF


def archive(tmp_path):
    ledger = []
    for endpoint, body in [("/fixtures", [fixture()]), ("/fixtures/statistics", team_stats())]:
        filename = endpoint.rsplit("/", 1)[-1] + ".json"
        encoded = json.dumps({"response": body})
        (tmp_path / filename).write_text(encoded)
        ledger.append({"path": endpoint, "fixture_id": "123", "file": filename,
                       "success": True, "sha256": sha256(encoded.encode()).hexdigest()})
    (tmp_path / "requests.json").write_text(json.dumps(ledger))
    return SavedProviderEvidence(tmp_path)


def test_offline_recovery_grades_only_proven_identity_and_retains_unknown_history(store, tmp_path):
    good = publish(store)
    bad = payload(line=3.25)
    bad["decision"]["quote"].pop("period")
    unknown = publish(store, bad)
    recovery = HistoricalSettlementRecovery(store[0], evidence=archive(tmp_path))
    before = store[0].get_recent(days=3650)
    plan = recovery.plan(cutoff=2, now=NOW)
    assert plan["counts"] == {"grade": 1, "record_pending": 1}
    assert store[0].get_recent(days=3650) == before
    applied = recovery.apply(plan)
    assert applied["counts"]["graded"] == 1 and applied["counts"]["pending_recorded"] == 1
    rows = {r["id"]: r for r in store[0].get_recent(days=3650)}
    assert rows[good]["outcome"] == "hit"
    assert rows[unknown]["outcome"] is None
    assert rows[unknown]["settlement"]["pending_reason"] == "missing_or_unsupported_period"
    assert all(r["closing_odds"] is None for r in rows.values())
    assert recovery.apply(plan)["counts"] == {"unchanged": 2}
    assert {r["id"]: r for r in store[0].get_recent(days=3650)} == rows


def test_recovery_preflight_rejects_changed_plan_or_prediction_before_any_write(store, tmp_path):
    publish(store)
    second = publish(store, payload(line=3.25))
    recovery = HistoricalSettlementRecovery(store[0], evidence=archive(tmp_path))
    plan = recovery.plan(cutoff=2, now=NOW)
    altered = deepcopy(plan)
    altered["entries"][0]["assessment"]["outcome"] = "miss"
    with pytest.raises(ValueError, match="integrity"):
        recovery.apply(altered)
    with store[0]._factory() as s:
        s.get(Prediction, second).odds = 2.2
    with pytest.raises(ValueError, match="changed"):
        recovery.apply(plan)
    assert all(r["settlement"] is None for r in store[0].get_recent(days=3650))


def test_evidence_checksum_and_path_are_enforced(tmp_path):
    archive(tmp_path)
    (tmp_path / "fixtures.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        SavedProviderEvidence(tmp_path)
    ledger = json.loads((tmp_path / "requests.json").read_text())
    ledger[0]["file"] = "../outside.json"
    (tmp_path / "requests.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="inside"):
        SavedProviderEvidence(tmp_path)


def test_recovery_is_resumable_without_regrading_completed_rows(store, tmp_path):
    publish(store)
    publish(store, payload(line=3.25))
    recovery = HistoricalSettlementRecovery(store[0], evidence=archive(tmp_path))
    plan = recovery.plan(cutoff=2, now=NOW)
    calls = [0]
    def interrupted():
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError):
        recovery.apply(plan, guard=interrupted)
    resumed = recovery.apply(plan)
    assert resumed["counts"] == {"unchanged": 1, "applied": 1, "graded": 1}


def test_recovery_excludes_future_and_unpublished_rows_and_never_changes_old_grades(store):
    publish(store)
    p = payload(fid="456")
    p["fixture"]["kickoff"] = (NOW + timedelta(days=1)).isoformat()
    publish(store, p)
    store[0].log(home_team="X", away_team="Y", league="EPL", market="goals", pick="Over 1.5", odds=2.)
    recovery = HistoricalSettlementRecovery(store[0])
    plan = recovery.plan(cutoff=2, now=NOW)
    assert len(plan["entries"]) == 1 and plan["excluded"][0]["reason"] == "not_due"
    assert plan["pending_reasons"] == {"authoritative_result_not_archived": 1}
    store[0].set_outcome(plan["entries"][0]["prediction_id"], outcome="miss")
    assert recovery.plan(cutoff=2, now=NOW)["counts"]["already_graded"] == 1


def test_publish_capture_settle_restart_and_both_surface_reports_agree(services, monkeypatch):
    reads, publications, delivery, repo = services
    runtime = MeasurementRuntime(repo._factory)
    runtime.activate(now=KICKOFF - timedelta(days=2))
    a = _published_market_result()
    a["fixture"].update(event_id="123", kickoff=KICKOFF.isoformat())
    a["decision"]["quote"].update(period="regulation_time", line=3.25)
    a["provenance"]["system_version"] = "offline-acceptance-only"
    b = deepcopy(a)
    b["market"].update(group="corners", key="totals_corners_over_under")
    b["decision"]["quote"].update(market_key="totals_corners_over_under", side="under", line=10.25)
    first = save(reads, [a, b])
    for surface in ("website", "discord", "website", "discord"):
        visible(delivery, first, surface=surface, time=(KICKOFF - timedelta(days=1)).isoformat())
    assert len(repo.settlement_candidates()) == 2
    client = Client("NS")
    client.odds["bookmakers"][0]["bets"] = [
        {"id": 5, "values": [{"value": "Over 3.25", "odd": "1.80"}]},
        {"id": 45, "values": [{"value": "Under 10.25", "odd": "1.90"}]},
    ]
    clock = [KICKOFF - timedelta(minutes=4)]
    cycle = MeasurementCycle(repo, runtime, client=client, clock=lambda: clock[0])
    assert cycle.run_once()["errors"] == 0
    assert all(p["closing_capture"]["status"] == "provisional" for p in repo.settlement_candidates())
    assert all(p["closing_odds"] is None for p in repo.settlement_candidates())
    cycle.run_once()  # Same tick, no duplicate snapshot/request.
    assert len(client.calls) == 2
    clock[0] = KICKOFF + timedelta(minutes=1)
    client.fx["fixture"]["status"]["short"] = "1H"
    assert cycle.run_once()["errors"] == 0
    assert all(p["closing_capture"]["status"] == "final" for p in repo.settlement_candidates())
    clock[0] = KICKOFF + timedelta(hours=3)
    client.fx["fixture"]["status"]["short"] = "FT"
    # Simulate a process restart before grading.
    cycle = MeasurementCycle(repo, runtime, client=client, clock=lambda: clock[0])
    assert cycle.run_once()["errors"] == 0
    before_retry = repo.get_recent(days=3650)
    cycle.run_once()
    assert repo.get_recent(days=3650) == before_retry
    expected = repo.get_track_record(published_only=True)
    assert expected["half_hits"] == expected["half_misses"] == 1
    assert expected["roi_flat_stake"] == 0 and expected["hit_rate"] == .5
    assert expected["clv_sample_size"] == 2 and expected["avg_clv"] > 0
    assert expected["stake_units"] == 2 and expected["pending_count"] == 0
    from data_platform.compat import predictions_shim
    from data_platform.services.predictions import PredictionService
    monkeypatch.setattr(predictions_shim, "get_prediction_service", lambda: PredictionService(repo))
    from Scripts.web_app import track_record_api as api
    from prediction_tracker import get_track_record
    bot_record = get_track_record(published_only=True)
    http = TestClient(api.create_standalone_app())
    response = http.get("/api/track-record")
    assert response.status_code == 200
    assert response.json() == bot_record == expected
    assert http.get("/api/track-record/daily").json() == expected["daily_performance"]
    assert "1 half wins" in record_summary_text(bot_record)
    assert "Asian-weighted hit rate: 50.0%" in record_summary_text(bot_record)
    recent = http.get("/api/track-record/recent").json()
    assert {p["outcome"] for p in recent} == {"half_hit", "half_miss"}
    assert all(p["system_version"] == "offline-acceptance-only" for p in recent)
    assert all(p["closing_capture"]["status"] == "final" for p in recent)
    assert all("result" not in p["settlement"] for p in recent)  # No raw/private evidence in public responses.


def test_pending_and_voids_are_visible_but_never_count_as_losses():
    rows = [{"id": i, "outcome": outcome, "odds": 2., "prediction_date": "2026-09-20"}
            for i, outcome in enumerate(["half_hit", "half_miss", "push", "void", None])]
    stats = build_track_record(rows)
    assert stats["pushes"] == stats["voids"] == stats["pending_count"] == 1
    assert stats["total_graded"] == 4 and stats["stake_units"] == 4
    assert stats["roi_flat_stake"] == 0 and stats["hit_rate"] == .5
    assert stats["resolved_units"] == 1 and stats["win_units"] == .5
    empty = build_track_record([rows[-1]])
    assert not empty["has_settled_sample"] and empty["pending_count"] == 1
    assert "no measured ROI" in record_summary_text(empty)
    unknown = build_track_record([{"outcome": "not-a-grade", "odds": 2.}])
    assert unknown["unknown_outcomes"] == 1 and unknown["total_graded"] == 0
    assert not unknown["has_settled_sample"]


def test_unverified_clv_is_excluded_from_official_metrics():
    rows = [{"tracking_cohort": "published", "outcome": "hit", "odds": 2, "clv": .99}]
    assert build_track_record(rows)["clv_sample_size"] == 0


def test_official_daily_report_does_not_choose_best_priced_amendment(services):
    reads, publications, delivery, repo = services
    first = _published_market_result()
    visible(delivery, save(reads, [first]))
    second = deepcopy(first)
    second["decision"]["quote"]["odds"] = 3.
    visible(delivery, save(reads, [second]), time="2026-09-05T12:00:00Z")
    for row in repo.settlement_candidates():
        repo.set_outcome(row["id"], outcome="hit")
    from datetime import date
    day = repo.get_daily_breakdown(target_date=date(2026, 9, 5), published_only=True, publication_scope="all")
    summary = repo.get_track_record(published_only=True, publication_scope="all")
    assert day["total"] == summary["total_graded"] == 2
    assert day["roi_flat_stake"] == summary["roi_flat_stake"] == 1.5
    assert repo.get_track_record(published_only=True)["roi_flat_stake"] == 1.


def test_web_reporting_errors_are_not_fake_empty_success(monkeypatch):
    from Scripts.web_app import track_record_api as api
    monkeypatch.setattr(api, "get_track_record", lambda **kw: {"error": "database_unavailable"})
    http = TestClient(api.create_standalone_app())
    assert http.get("/api/track-record").status_code == 503
    assert http.get("/api/track-record/daily").status_code == 503


def test_discord_official_resolution_uses_the_bounded_prospective_worker(monkeypatch):
    from Scripts.discord_bot.tracking_report import resolve_official_outcomes
    import data_platform.services.measurement_cycle as module
    calls = []
    monkeypatch.setattr(module, "run_measurement_cycle", lambda **kw: calls.append(kw) or {
        "status": "refresh_deferred", "errors": 0, "provider_attempts": 0})
    result = resolve_official_outcomes()
    assert calls == [{"kinds": ("settlement",)}]
    assert result["graded"] == 0 and result["status"] == "refresh_deferred"


def test_settlement_compare_and_swap_rejects_changed_selection(store):
    pid = publish(store)
    stale = store[0].settlement_candidates()[0]
    with store[0]._factory() as session:
        session.get(Prediction, pid).line = 3.5
    assert store[0].record_settlement(stale, {"outcome": "hit"}, checked_at=NOW) == "conflict"
    assert store[0].get_unresolved()[0]["outcome"] is None


def test_acceptance_audit_cannot_pass_an_empty_live_sample(store):
    from Scripts.ops.phase2_acceptance_audit import build_report
    runtime = MeasurementRuntime(store[0]._factory)
    runtime.activate(now=NOW)
    report = build_report(store[0], now=NOW)
    assert not report["full_phase2_signoff"]
    assert report["gates"]["prospective_settlement_observed"] == "awaiting_live_sample"
    assert report["gates"]["prospective_identity"] == "awaiting_live_sample"


def test_pending_records_are_available_on_website_without_private_evidence(store, monkeypatch):
    pid = publish(store)
    with store[0]._factory() as session:
        row = session.get(Prediction, pid)
        row.extras = {**row.extras, "settlement": {"status": "pending", "pending_reason": "missing_market_statistics",
                     "source": {"directory": "/private/evidence"}, "result": {"sensitive": "not for public API"}}}
    from Scripts.web_app import track_record_api as api
    monkeypatch.setattr(api, "get_recent_predictions", lambda **kw: store[0].get_recent(days=3650))
    http = TestClient(api.create_standalone_app())
    row = http.get("/api/track-record/recent?graded_only=false").json()[0]
    assert row["outcome"] is None
    assert row["settlement"] == {"status": "pending", "pending_reason": "missing_market_statistics"}
