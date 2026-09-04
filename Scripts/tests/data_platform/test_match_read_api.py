"""Public website API tests for persisted Match Read cards.

The app-specific router is tested in isolation so these checks do not import
the legacy live prediction engine or touch provider APIs.  Every request must
be read-only against the platform store.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _market_result(
    *,
    fixture_id: str,
    kickoff: str = "2026-09-06T15:00:00Z",
    league: str = "EPL",
    edge: float = 0.10,
    confidence: str = "high",
) -> dict:
    return {
        "schema_version": "market_result.v1",
        "fixture": {
            "event_id": fixture_id,
            "league": league,
            "home_team": f"Home {fixture_id}",
            "away_team": f"Away {fixture_id}",
            "kickoff": kickoff,
        },
        "market": {"key": "totals", "group": "goals", "unit": "goals", "participant": None},
        "projection": {"value": 2.9, "unit": "goals", "components": {}},
        "decision": {
            "status": "recommended",
            "quote": {
                "side": "over",
                "line": 2.5,
                "odds": 1.95,
                "bookmaker": "Book",
                "market_key": "totals",
            },
            "model_probability": 0.60,
            "implied_probability": 1 / 1.95,
            "value_edge": edge,
            "expected_value": edge,
            "confidence": confidence,
            "reason": "Qualified test price.",
        },
        "provenance": {
            "pipeline_version": "canonical-market-service.v1",
            "input_snapshot_id": f"snapshot:{fixture_id}",
            "generated_at": "2026-09-05T09:00:00Z",
            "model_version": "2026.09.1",
        },
        "evidence": [{"source": "test", "detail": "kept public"}],
        "context": {},
    }


def _service(session_factory):
    from data_platform.repositories.match_reads import MatchReadRepository
    from data_platform.services.match_reads import MatchReadService

    return MatchReadService(repo=MatchReadRepository(session_factory=session_factory))


def _save(
    service,
    result: dict,
    *,
    thesis: str,
    stage: str = "pre_match",
    evaluated_at: str = "2026-09-05T09:00:00Z",
) -> dict:
    return service.create(
        canonical_results=[result],
        thesis=thesis,
        status="recommended",
        selections=[{"result_index": 0, "role": "core"}],
        game_script={"tags": ["open_game"]},
        stage=stage,
        evaluated_at=evaluated_at,
    )


def _release(service, saved: dict, *, reference: str | None = None) -> dict:
    """Mark one immutable version as actually visible on the public website."""
    return service.record_delivery(
        saved["id"],
        surface="website",
        external_reference=reference or f"/match-reads/{saved['id']}",
        delivered_at="2026-09-05T11:00:00Z",
    )


def _client(monkeypatch, service) -> TestClient:
    from web_app.routers import match_reads as match_reads_router

    monkeypatch.setattr(match_reads_router, "_get_match_read_service", lambda: service)
    # The persisted-card release policy is time-sensitive.  Keep these API
    # contract tests deterministic rather than depending on the machine date.
    monkeypatch.setattr(
        match_reads_router,
        "_utc_now",
        lambda: datetime(2026, 9, 6, 12, 30, tzinfo=timezone.utc),
    )
    app = FastAPI()
    app.include_router(match_reads_router.router)
    return TestClient(app)


def test_matchday_board_prefers_latest_confirmed_lineups_read_and_is_read_only(
    settings, engine, session_factory, monkeypatch,
):
    from data_platform.models import MatchRead, MatchReadDelivery

    service = _service(session_factory)
    result = _market_result(fixture_id="fixture-effective")
    pre = _save(service, result, thesis="Initial pre-match read.")
    shadow_pre = _save(
        service,
        result,
        thesis="Unreleased pre-match shadow revision.",
        evaluated_at="2026-09-05T10:00:00Z",
    )
    confirmed_v1 = _save(
        service,
        result,
        thesis="First confirmed-lineups read.",
        stage="confirmed_lineups",
        evaluated_at="2026-09-06T12:00:00Z",
    )
    confirmed = _save(
        service,
        result,
        thesis="Latest confirmed-lineups read.",
        stage="confirmed_lineups",
        evaluated_at="2026-09-06T12:15:00Z",
    )
    discord_only = _save(
        service,
        result,
        thesis="Later Discord-only amendment.",
        stage="confirmed_lineups",
        evaluated_at="2026-09-06T12:30:00Z",
    )
    _release(service, pre)
    _release(service, confirmed_v1)
    _release(service, confirmed)
    service.record_delivery(
        discord_only["id"],
        surface="discord",
        external_reference="discord:fixture-effective",
        delivered_at="2026-09-06T12:31:00Z",
    )

    with session_factory() as session:
        before_reads = session.query(MatchRead).count()
        before_deliveries = session.query(MatchReadDelivery).count()

    response = _client(monkeypatch, service).get("/api/match-reads/EPL/2026-09-06")

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "match-read-board.v1"
    assert body["count"] == 1
    card = body["cards"][0]
    assert card["id"] == confirmed["id"]
    assert card["stage"] == "confirmed_lineups"
    assert card["thesis"] == "Latest confirmed-lineups read."
    assert card["update"] == {
        "is_updated": True,
        "label": "Updated after confirmed lineups",
        "reason": "confirmed_lineups",
        "supersedes_id": confirmed["supersedes_id"],
    }
    # The shared card is a public DTO, never the archival `read_json` blob.
    assert "data" not in card
    assert "data" not in card["selections"][0]
    # A persisted-but-undelivered shadow revision cannot leak to public API.
    assert card["id"] != shadow_pre["id"]
    # A later Discord-only revision cannot win website's effective selection.
    assert card["id"] != discord_only["id"]

    with session_factory() as session:
        assert session.query(MatchRead).count() == before_reads
        assert session.query(MatchReadDelivery).count() == before_deliveries


def test_fixture_detail_returns_effective_card_and_immutable_history(
    settings, engine, session_factory, monkeypatch,
):
    service = _service(session_factory)
    result = _market_result(fixture_id="fixture-history")
    pre = _save(service, result, thesis="Initial pre-match read.")
    shadow = _save(
        service,
        result,
        thesis="Unreleased shadow amendment.",
        evaluated_at="2026-09-05T10:00:00Z",
    )
    confirmed = _save(
        service,
        result,
        thesis="Confirmed-lineups amendment.",
        stage="confirmed_lineups",
        evaluated_at="2026-09-06T12:00:00Z",
    )
    _release(service, pre)
    _release(service, confirmed)

    response = _client(monkeypatch, service).get("/api/match-reads/fixtures/fixture-history")

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "match-read-detail.v1"
    assert body["fixture_id"] == "fixture-history"
    assert body["effective"]["id"] == confirmed["id"]
    assert body["effective"]["update"]["label"] == "Updated after confirmed lineups"
    assert body["history_count"] == 2
    assert {item["id"] for item in body["history"]} == {pre["id"], confirmed["id"]}
    assert shadow["id"] not in {item["id"] for item in body["history"]}
    assert all("data" not in item for item in body["history"])


def test_best_match_reads_is_capped_at_five_fixture_cards(
    settings, engine, session_factory, monkeypatch,
):
    service = _service(session_factory)
    for index in range(6):
        saved = _save(
            service,
            _market_result(fixture_id=f"fixture-best-{index}", edge=0.05 + index / 100),
            thesis=f"Match Read {index}.",
            evaluated_at=f"2026-09-06T12:{index:02d}:00Z",
        )
        _release(service, saved)

    response = _client(monkeypatch, service).get("/api/match-reads/best/2026-09-06")

    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 5
    assert body["available_count"] == 6
    assert len(body["cards"]) == 5
    # Equal confidence means the largest core edge leads the capped list.
    assert body["cards"][0]["fixture"]["event_id"] == "fixture-best-5"


def test_match_read_board_rejects_a_non_iso_matchday(settings, engine, session_factory, monkeypatch):
    service = _service(session_factory)
    response = _client(monkeypatch, service).get("/api/match-reads/EPL/06-09-2026")
    assert response.status_code == 400
    assert response.json()["detail"] == "target_date must use YYYY-MM-DD format."


def test_public_board_withholds_started_and_stale_cards(
    settings, engine, session_factory, monkeypatch,
):
    service = _service(session_factory)
    fresh = _save(
        service,
        _market_result(fixture_id="fixture-fresh", kickoff="2026-09-06T15:00:00Z"),
        thesis="Fresh fixture read.",
        evaluated_at="2026-09-06T12:00:00Z",
    )
    started = _save(
        service,
        _market_result(fixture_id="fixture-started", kickoff="2026-09-06T12:00:00Z"),
        thesis="Started fixture read.",
        evaluated_at="2026-09-06T11:59:00Z",
    )
    stale = _save(
        service,
        _market_result(fixture_id="fixture-stale", kickoff="2026-09-06T18:00:00Z"),
        thesis="Stale fixture read.",
        evaluated_at="2026-09-06T09:00:00Z",
    )
    for saved in (fresh, started, stale):
        _release(service, saved)

    client = _client(monkeypatch, service)
    response = client.get("/api/match-reads/EPL/2026-09-06")

    assert response.status_code == 200
    assert [card["fixture"]["event_id"] for card in response.json()["cards"]] == ["fixture-fresh"]
    expired = client.get("/api/match-reads/fixtures/fixture-started")
    assert expired.status_code == 410
    assert "no longer actionable" in expired.json()["detail"]
