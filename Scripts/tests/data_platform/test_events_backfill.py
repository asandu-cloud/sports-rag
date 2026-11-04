"""Cards plan task 7 (D54 a): fill the match events of finished matches since a date."""
import json
from datetime import date

import pytest
from sqlalchemy import func, select

from Scripts.tests.data_platform.test_sync_upserts import SPEC, _api_fixture_row, _events_response

SINCE = date(2026, 9, 20)
# (API-Football id, kickoff day, status): 3 is the oldest since 20 Sep, 1 is before it, 5 is not played
FIXTURES = [(1, "2026-09-19", "FT"), (2, "2026-09-21", "FT"), (3, "2026-09-20", "AET"),
            (4, "2026-10-01", "PEN"), (5, "2026-10-02", "NS"), (6, "2026-10-03", "FT")]


class Client:
    def __init__(self, fail=(), empty=()):
        self.calls, self.fail, self.empty = [], set(fail), set(empty)

    def fixture_events(self, fixture_id):
        self.calls.append(fixture_id)
        if fixture_id in self.fail:
            raise TimeoutError("temporary failure")
        return [] if fixture_id in self.empty else _events_response()


@pytest.fixture()
def fixtures(session_factory):
    from data_platform.sync.upserts import upsert_competition, upsert_fixture_from_api_row, upsert_season
    with session_factory() as s:
        competition = upsert_competition(s, SPEC)
        season = upsert_season(s, competition=competition, year=2026)
        s.flush()
        for api_id, day, status in FIXTURES:
            row = _api_fixture_row(api_id=api_id, status=status)
            row["fixture"]["date"] = f"{day}T15:00:00+00:00"
            upsert_fixture_from_api_row(s, api_row=row, competition=competition, season=season)
        s.commit()
    return session_factory


def stored(session_factory):
    from data_platform.models import Fixture, FixtureEvent
    with session_factory() as s:
        return dict(s.execute(select(Fixture.api_football_id, func.count()).join(
            FixtureEvent, FixtureEvent.fixture_id == Fixture.id).group_by(Fixture.api_football_id)).all())


def test_a_dry_run_counts_the_requests_and_makes_none(fixtures):
    from data_platform.sync.events_backfill import backfill_events
    client = Client()
    assert backfill_events(fixtures, client, since=SINCE, dry_run=True) == {
        "candidates": 4, "requests": 4, "fetched": 0, "rows": 0, "empty": 0, "errors": 0}
    assert backfill_events(fixtures, client, since=SINCE, limit=3, dry_run=True)["requests"] == 3
    assert client.calls == [] and stored(fixtures) == {}


def test_a_limited_run_fetches_the_oldest_and_the_next_run_skips_them(fixtures):
    from data_platform.sync.events_backfill import backfill_events
    client = Client()
    assert backfill_events(fixtures, client, since=SINCE, limit=2) == {
        "candidates": 4, "requests": 2, "fetched": 2, "rows": 10, "empty": 0, "errors": 0}
    assert client.calls == [3, 2] and stored(fixtures) == {3: 5, 2: 5}
    assert backfill_events(fixtures, client, since=SINCE)["candidates"] == 2
    assert client.calls == [3, 2, 4, 6] and stored(fixtures) == {2: 5, 3: 5, 4: 5, 6: 5}


def test_a_failed_request_is_counted_and_the_run_carries_on(fixtures):
    from data_platform.sync.events_backfill import backfill_events
    client = Client(fail={3})
    assert backfill_events(fixtures, client, since=SINCE) == {
        "candidates": 4, "requests": 4, "fetched": 3, "rows": 15, "empty": 0, "errors": 1}
    assert client.calls == [3, 2, 4, 6] and 3 not in stored(fixtures)
    assert backfill_events(fixtures, Client(), since=SINCE, dry_run=True)["candidates"] == 1


def test_an_empty_reply_stores_nothing_and_stays_a_candidate(fixtures):
    from data_platform.sync.events_backfill import backfill_events
    result = backfill_events(fixtures, Client(empty={4}), since=SINCE)
    assert (result["fetched"], result["rows"], result["empty"]) == (4, 15, 1)
    assert backfill_events(fixtures, Client(), since=SINCE, dry_run=True)["candidates"] == 1


def test_before_init_db_creates_the_table_the_backfill_stops_before_any_request(engine, fixtures):
    from data_platform.sync.events_backfill import backfill_events
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE fixture_match_events")
    client = Client()
    with pytest.raises(RuntimeError, match="init-db"):
        backfill_events(fixtures, client, since=SINCE)
    assert client.calls == []


def test_the_command_runs_a_dry_run_without_an_api_key(fixtures, capsys, monkeypatch):
    from data_platform import cli, sync

    def no_client(*_, **__):
        raise AssertionError("a dry run never builds an API client")
    monkeypatch.setattr(sync, "ApiFootballClient", no_client)
    assert cli.main(["backfill-events", "--since", "2026-09-20", "--limit", "3", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["requests"] == 3
