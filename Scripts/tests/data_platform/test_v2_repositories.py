"""V2 repository tests — predictions, parlays, live, odds, kb."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest


@pytest.fixture()
def pred_repo(settings, engine, session_factory):
    from data_platform.repositories.predictions import PredictionRepository
    return PredictionRepository(session_factory=session_factory)


@pytest.fixture()
def parlay_repo(settings, engine, session_factory):
    from data_platform.repositories.parlays import ParlayRepository
    return ParlayRepository(session_factory=session_factory)


@pytest.fixture()
def live_repo(settings, engine, session_factory):
    from data_platform.repositories.live import LiveRepository
    return LiveRepository(session_factory=session_factory)


@pytest.fixture()
def kb_repo(settings, engine, session_factory):
    from data_platform.repositories.kb import KBDocumentRepository
    return KBDocumentRepository(session_factory=session_factory)


def test_prediction_log_dedup(pred_repo):
    id1 = pred_repo.log(
        home_team="Arsenal", away_team="Spurs", league="EPL",
        market="goals", pick="Over 2.5", side="over", odds=1.95,
        model_prob=0.6, implied_prob=0.51, value_edge=0.09,
        confidence="medium",
    )
    id2 = pred_repo.log(
        home_team="Arsenal", away_team="Spurs", league="EPL",
        market="goals", pick="Over 2.5", side="over", odds=2.05,  # new odds, same pick
    )
    assert id1 == id2, "same (date, teams, market, pick, source) should return existing id"

    id3 = pred_repo.log(
        home_team="Arsenal", away_team="Spurs", league="EPL",
        market="goals", pick="Over 3.5", side="over",
    )
    assert id3 != id1


def test_prediction_resolve_and_track_record(pred_repo):
    ids = []
    for home, away, mkt, pick in [
        ("A", "B", "goals", "Over 2.5"),
        ("C", "D", "goals", "Over 2.5"),
        ("E", "F", "btts", "BTTS Yes"),
    ]:
        ids.append(pred_repo.log(
            home_team=home, away_team=away, league="EPL",
            market=mkt, pick=pick, side="over", odds=2.0, confidence="medium",
        ))
    assert pred_repo.set_outcome(ids[0], outcome="win", actual_result=3)
    assert pred_repo.set_outcome(ids[1], outcome="loss", actual_result=1)
    assert pred_repo.set_outcome(ids[2], outcome="win", actual_result=1)

    tr = pred_repo.get_track_record()
    assert tr["total_graded"] == 3
    assert tr["hits"] == 2
    assert tr["misses"] == 1
    assert pytest.approx(tr["hit_rate"], abs=0.01) == 2 / 3
    # ROI: 2 winning legs at odds 2.0 return 4.0; 1 loser returns 0. Stake = 3.
    assert pytest.approx(tr["roi_flat_stake"], abs=0.01) == (4.0 / 3.0 - 1.0)
    assert tr["by_market"]["goals"]["hits"] == 1

    rows = pred_repo.get_recent(days=1)
    assert {row["outcome"] for row in rows} == {"hit", "miss"}


def test_prediction_calibration_buckets(pred_repo):
    # Seed deterministic data across four buckets
    for prob, outcome in [
        (0.15, "loss"), (0.18, "loss"),
        (0.52, "win"), (0.56, "loss"),
        (0.81, "win"), (0.85, "win"),
    ]:
        pid = pred_repo.log(
            home_team="H", away_team="A", league="EPL",
            market="goals", pick=f"Over {prob}", side="over", odds=1.9,
            model_prob=prob,
        )
        pred_repo.set_outcome(pid, outcome=outcome)
    cal = pred_repo.get_calibration_data(buckets=5)
    by_bucket = {row["bucket"]: row for row in cal}
    assert by_bucket["0-20%"]["count"] == 2
    assert by_bucket["0-20%"]["actual_rate"] == 0.0
    assert by_bucket["40-60%"]["count"] == 2
    assert by_bucket["80-100%"]["actual_rate"] == 1.0


def test_parlay_session_round_trip(parlay_repo):
    session_id = parlay_repo.create(
        parent_session_id=None, discord_user_id=42, guild_id=1, channel_id=2, thread_id=None,
        source_command="parlay", action_type="build",
        request_json={"target_date": "2026-04-14", "legs": 3},
        result_json={"combined_odds": 4.2, "notes": ["ok"]},
    )
    assert session_id > 0
    parlay_repo.update_message_context(session_id, message_id=99)
    record = parlay_repo.get(session_id)
    assert record is not None
    assert record["message_id"] == 99
    assert record["request_json"]["legs"] == 3
    assert record["result_json"]["combined_odds"] == 4.2

    persistent = parlay_repo.list_persistent(limit=10)
    assert any(r["id"] == session_id for r in persistent)


def test_live_store_round_trip(live_repo):
    live_repo.ensure_fixture(
        fixture_id=111, league="EPL", home_team="A", away_team="B",
        kickoff_utc="2026-04-14T14:00:00+00:00",
    )
    live_repo.record_state(
        fixture_id=111,
        state_payload={"score": "1-0", "elapsed": 47},
        status="2H", elapsed_min=47.0,
    )
    live_repo.record_model_snapshot(
        fixture_id=111,
        snapshot_payload={"model": {"goals": 2.4}},
        snapshot_ts="2026-04-14T14:47:00+00:00",
        elapsed_min=47.0, score="1-0",
    )
    live_repo.record_alerts(
        fixture_id=111, snapshot_ts="2026-04-14T14:47:00+00:00",
        alerts=[{"type": "goals_over", "severity": "high", "message": "value"}],
    )
    active = live_repo.get_active_fixtures()
    assert any(f["fixture_id"] == 111 for f in active)

    latest = live_repo.get_latest_model_snapshot(111)
    assert latest is not None
    assert latest["snapshot"]["model"]["goals"] == 2.4

    alerts = live_repo.get_recent_alerts(111)
    assert len(alerts) == 1
    assert alerts[0]["type"] == "goals_over"

    live_repo.mark_finished(111)
    active_after = live_repo.get_active_fixtures()
    assert not any(f["fixture_id"] == 111 for f in active_after)


def test_kb_document_hash_dedup(kb_repo):
    first = kb_repo.upsert(
        doc_id="abc", entity_type="team", entity_id="42",
        league="EPL", season="2025/26", doc_type="team_profile",
        text="Arsenal — season snapshot",
        metadata={"goals_for_pm": 2.1},
    )
    assert first["changed"] is True

    second = kb_repo.upsert(
        doc_id="abc", entity_type="team", entity_id="42",
        league="EPL", season="2025/26", doc_type="team_profile",
        text="Arsenal — season snapshot",
        metadata={"goals_for_pm": 2.1},
    )
    assert second["changed"] is False

    third = kb_repo.upsert(
        doc_id="abc", entity_type="team", entity_id="42",
        league="EPL", season="2025/26", doc_type="team_profile",
        text="Arsenal — updated snapshot",
        metadata={"goals_for_pm": 2.3},
    )
    assert third["changed"] is True


def test_kb_refresh_queue_coalesces_duplicates(kb_repo):
    id1 = kb_repo.enqueue(entity_type="team", entity_key="EPL:42", reason="fixture_change")
    id2 = kb_repo.enqueue(entity_type="team", entity_key="EPL:42", reason="manual")
    assert id1 == id2, "duplicate pending enqueue should be coalesced"

    claimed = kb_repo.claim_pending(limit=10)
    assert len(claimed) == 1
    kb_repo.finish(claimed[0]["id"], status="completed")

    stats = kb_repo.queue_stats()
    assert stats.get("completed", 0) == 1
    assert stats.get("pending", 0) == 0
