"""The board's two reads retry a locked database like the writes do (D72): between 22 Oct and 2 Nov, 20 board and
schedule requests returned 503 after SQLite's 2-second wait while a writer held the rollback journal."""
import sqlite3
from contextlib import contextmanager
from datetime import date

from sqlalchemy.exc import OperationalError

from Scripts.tests.data_platform.test_match_read_api import _market_result, _release, _save, _service
from Scripts.tests.data_platform.test_matchday_schedule_api import _seed


def locked_once(session_factory):
    attempts = []

    @contextmanager
    def factory():
        attempts.append(1)
        if len(attempts) == 1:
            raise OperationalError("SELECT", {}, sqlite3.OperationalError("database is locked"))
        with session_factory() as session:
            yield session

    return factory, attempts


def test_the_schedule_read_retries_a_locked_database(settings, engine, session_factory):
    from data_platform.repositories.fixture_schedule import FixtureScheduleRepository

    _seed(session_factory)
    factory, attempts = locked_once(session_factory)
    rows = FixtureScheduleRepository(factory).list_for_matchday(target_date=date(2026, 10, 10), leagues=("EPL",))
    assert len(rows) == 3 and len(attempts) == 2


def test_the_board_read_retries_a_locked_database(settings, engine, session_factory):
    from data_platform.repositories.match_reads import MatchReadRepository

    service = _service(session_factory)
    _release(service, _save(service, _market_result(fixture_id="fixture-locked"), thesis="Read."))
    factory, attempts = locked_once(session_factory)
    rows = MatchReadRepository(session_factory=factory).list_delivered_for_matchday(
        league="EPL", target_date="2026-09-06", surface="website")
    assert [row["fixture"]["event_id"] for row in rows] == ["fixture-locked"] and len(attempts) == 2
