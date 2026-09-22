from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from sqlalchemy import select

from Scripts.tests.data_platform.test_sync_upserts import _api_fixture_row


NOW = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)


def row(fixture_id=10, *, days=18, status="NS"):
    value = _api_fixture_row(api_id=fixture_id, status=status)
    value["fixture"]["date"] = (NOW + timedelta(days=days)).isoformat()
    value["league"]["season"] = 2026
    return value


def test_schedule_reaches_beyond_seven_days_and_preserves_history(settings, engine, session_factory):
    from data_platform.services.fixture_schedule import sync_fixture_schedule
    from data_platform.models import Fixture
    from data_platform.sync.upserts import upsert_competition, upsert_season, upsert_fixture_from_api_row, get_watermark
    from data_platform.sync.apifootball import COMPETITIONS
    with session_factory() as session:
        comp = upsert_competition(session, COMPETITIONS["EPL"])
        season = upsert_season(session, competition=comp, year=2026)
        session.flush()
        historical, _ = upsert_fixture_from_api_row(session, api_row=row(1, days=-1, status="FT"), competition=comp, season=season)
        session.flush()
        digest = historical.payload_digest
    client = Mock()
    client.fixtures.return_value = [row(1, days=-1, status="FT"), row(2), row(3, days=25)]
    result = sync_fixture_schedule(codes=["EPL"], now=NOW, client=client, session_factory=session_factory)
    assert result["fixtures_changed"] == 2
    assert result["provider_calls"] == 1 and result["errors"] == []
    client.fixtures.assert_called_once_with(league=39, season=2026, from_date="2026-09-21", to_date="2026-10-26")
    client.fixture_players.assert_not_called()
    client.fixture_statistics.assert_not_called()
    with session_factory() as session:
        assert len(session.scalars(select(Fixture)).all()) == 3
        assert session.scalar(select(Fixture).where(Fixture.api_football_id == 1)).payload_digest == digest
        assert get_watermark(session, "fixtures:EPL:2026") is None
        assert get_watermark(session, "schedule:EPL:2026").last_cursor == "2026-10-26"
    second = sync_fixture_schedule(codes=["EPL"], now=NOW, client=client, session_factory=session_factory)
    assert second["fixtures_changed"] == 0


def test_schedule_does_not_consume_newly_finished_fixture_digest(settings, engine, session_factory):
    from data_platform.services.fixture_schedule import sync_fixture_schedule
    from data_platform.models import Fixture
    client = Mock()
    client.fixtures.return_value = [row(2, days=1)]
    sync_fixture_schedule(codes=["EPL"], now=NOW, client=client, session_factory=session_factory)
    with session_factory() as session:
        original = session.scalar(select(Fixture)).payload_digest
    client.fixtures.return_value = [row(2, days=1, status="FT")]
    sync_fixture_schedule(codes=["EPL"], now=NOW + timedelta(days=2), client=client, session_factory=session_factory)
    with session_factory() as session:
        assert session.scalar(select(Fixture)).payload_digest == original
        assert session.scalar(select(Fixture)).status == "NS"


def test_empty_calendar_is_success_and_six_hour_gate_avoids_extra_requests(settings, engine, session_factory):
    from data_platform.services.fixture_schedule import sync_fixture_schedule
    client = Mock()
    client.fixtures.return_value = []
    first = sync_fixture_schedule(codes=["EPL"], now=NOW, client=client, session_factory=session_factory)
    assert first["errors"] == [] and first["competitions"][0]["upcoming_count"] == 0
    second = sync_fixture_schedule(codes=["EPL"], now=NOW + timedelta(hours=5), if_stale_hours=6,
                                   client=client, session_factory=session_factory)
    assert second["provider_calls"] == 0
    third = sync_fixture_schedule(codes=["EPL"], now=NOW + timedelta(hours=6), if_stale_hours=6,
                                  client=client, session_factory=session_factory)
    assert third["provider_calls"] == 1


def test_failed_provider_check_does_not_advance_coverage(settings, engine, session_factory):
    from data_platform.services.fixture_schedule import sync_fixture_schedule
    from data_platform.sync.upserts import get_watermark
    from data_platform.models import SyncRun
    client = Mock()
    client.fixtures.side_effect = RuntimeError("quota exceeded")
    result = sync_fixture_schedule(codes=["EPL"], now=NOW, client=client, session_factory=session_factory)
    assert result["errors"]
    with session_factory() as session:
        assert get_watermark(session, "schedule:EPL:2026") is None
        assert session.get(SyncRun, result["run_id"]).status == "partial"


def test_july_rollover_queries_both_seasons(settings, engine, session_factory):
    from data_platform.services.fixture_schedule import sync_fixture_schedule
    client = Mock()
    client.fixtures.return_value = []
    sync_fixture_schedule(codes=["EPL"], now=datetime(2027, 6, 25, tzinfo=timezone.utc),
                          client=client, session_factory=session_factory)
    assert [call.kwargs["season"] for call in client.fixtures.call_args_list] == [2026, 2027]


def test_http_200_api_error_is_not_a_successful_empty_calendar(settings):
    from data_platform.sync.apifootball import ApiFootballClient, ApiFootballResponseError
    session = Mock()
    session.get.return_value.status_code = 200
    session.get.return_value.json.return_value = {"response": [], "errors": {"requests": "quota exceeded"}}
    client = ApiFootballClient(api_key="test", session=session, request_pause_s=0)
    with pytest.raises(ApiFootballResponseError, match="quota exceeded"):
        client.fixtures(league=39, season=2026)
    assert session.get.call_count == 1


def test_worker_ignores_cancelled_and_postponed_fixtures(settings, engine, session_factory):
    from data_platform.services.fixture_schedule import sync_fixture_schedule
    from data_platform.services.match_read_cycle import _load_upcoming_platform_fixtures
    client = Mock()
    client.fixtures.return_value = [row(1, days=1, status="NS"), row(2, days=1, status="CANC"), row(3, days=1, status="PST")]
    sync_fixture_schedule(codes=["EPL"], now=NOW, client=client, session_factory=session_factory)
    found = _load_upcoming_platform_fixtures(now=NOW, until=NOW + timedelta(days=4), leagues=["EPL"])
    assert [fixture.fixture_id for fixture in found] == ["1"]
