"""Isolated schedule → generation → publication → HTTP board → XI amendment.

Only provider inputs and numerical predictions are synthetic. Use the actual
canonical schedule loader, dispatcher, compiler, repositories, release bridge
and public HTTP router. Nothing is inserted into the live project database.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from Scripts.tests.data_platform.test_fixture_schedule import row
from Scripts.tests.data_platform.test_match_read_cycle import _settings, _market_result


def test_four_day_card_and_verified_lineup_amendment_reach_http_board(settings, engine, session_factory, monkeypatch):
    from data_platform.services.fixture_schedule import sync_fixture_schedule
    from data_platform.repositories.match_reads import MatchReadRepository
    from data_platform.services.match_reads import MatchReadService
    from data_platform.services.match_read_cycle import MatchReadCycleService
    from Scripts.rag_ingest.rendering.match_read_dispatch import generate_match_reads_sync
    from web_app.routers import match_reads as router

    now = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
    kickoff = now + timedelta(days=3)
    fixture = row(9001)
    fixture["fixture"]["date"] = kickoff.isoformat()
    client = Mock()
    client.fixtures.return_value = [fixture]
    sync_fixture_schedule(codes=["EPL"], now=now, client=client, session_factory=session_factory)
    service = MatchReadService(repo=MatchReadRepository(session_factory))
    current = [now]
    verified = SimpleNamespace(source="lineups", is_available=True)
    event = {"id": "9001", "home_team": "Home FC", "away_team": "Away United", "commence_time": kickoff.isoformat()}

    def dispatch(league, target_date, **kwargs):
        def evaluate(*_, **__):
            result = _market_result("9001", kickoff, generated_at=current[0].isoformat())
            result["fixture"].update(home_team="Home FC", away_team="Away United")
            return [result]
        return generate_match_reads_sync(league, target_date, service=service,
            fetcher=lambda *_args, **_kwargs: ([event], []),
            enricher=lambda events, *_: (events, []), evaluator=evaluate,
            briefing_enricher=lambda draft, *_args, **_kwargs: (draft, None), **kwargs)

    worker = MatchReadCycleService(settings=_settings(), session_factory=session_factory,
        dispatcher=dispatch, lineup_provider=lambda *_: verified, lineup_preflight=lambda *_: verified,
        clock=lambda: current[0])
    monkeypatch.setattr(router, "_get_match_read_service", lambda: service)
    monkeypatch.setattr(router, "_utc_now", lambda: current[0])
    app = FastAPI()
    app.include_router(router.router)
    website = TestClient(app)
    first = worker.run_once(now=now, mode="website")
    assert not first.errors, first.errors
    first_board = website.get("/api/match-reads/EPL/2026-10-08").json()
    assert first_board["count"] == 1
    assert first_board["cards"][0]["stage"] == "pre_match"

    current[0] = kickoff - timedelta(minutes=60)
    second = worker.run_once(now=current[0], mode="website")
    assert not second.errors, second.errors
    board = website.get("/api/match-reads/EPL/2026-10-08").json()
    assert board["count"] == 1  # replace the visible card, do not add clutter
    assert board["cards"][0]["stage"] == "confirmed_lineups"
    assert board["cards"][0]["id"] != first_board["cards"][0]["id"]
    assert len(service.list_for_fixture("9001")) >= 2
