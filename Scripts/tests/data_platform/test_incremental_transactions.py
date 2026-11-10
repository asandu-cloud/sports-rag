"""Provider calls must not hold SQLite's write lock; failed details retry."""
from sqlalchemy import select
from Scripts.tests.data_platform.test_sync_upserts import (_api_fixture_row, _events_response, _players_response,
                                                          _stats_response)


def test_incremental_calls_release_write_transactions_and_remain_idempotent(settings, engine, session_factory):
    from data_platform.sync.pipeline import incremental_refresh
    from data_platform.models import Fixture
    calls = []
    with session_factory() as session:
        class Client:
            def fixtures(self, **kwargs):
                assert not session.in_transaction()
                calls.append("fixtures")
                return [_api_fixture_row()]
            def fixture_statistics(self, fixture):
                assert not session.in_transaction()
                calls.append("team")
                # Another connection can write while a provider call runs.
                with engine.begin() as connection:
                    connection.exec_driver_sql("UPDATE sync_runs SET status=status")
                return _stats_response()
            def fixture_players(self, fixture):
                assert not session.in_transaction()
                calls.append("players")
                return _players_response()
            def fixture_events(self, fixture):
                assert not session.in_transaction()
                calls.append("events")
                return _events_response()
        result = incremental_refresh(session, client=Client(), codes=["EPL"], years=[2025], short_transactions=True)
        assert result["error_count"] == 0 and result["player_stats_rows"] == 1 and result["event_rows"] == 5
        result = incremental_refresh(session, client=Client(), codes=["EPL"], years=[2025], short_transactions=True)
        assert result["fixtures_changed"] == 0
        assert calls == ["fixtures", "team", "players", "events", "fixtures"]
        assert len(session.scalars(select(Fixture)).all()) == 1


def test_failed_player_fetch_retains_checkpoint_and_retries_without_deleting_history(settings, engine, session_factory):
    from data_platform.sync.pipeline import incremental_refresh
    from data_platform.sync.upserts import get_watermark
    from data_platform.models import Fixture, SyncRun
    fail = [True]
    class Client:
        def fixtures(self, **_):
            return [_api_fixture_row()]
        def fixture_statistics(self, _):
            return _stats_response()
        def fixture_players(self, _):
            if fail[0]:
                raise TimeoutError("temporary failure")
            return _players_response()
        def fixture_events(self, _):
            return _events_response()
    with session_factory() as session:
        result = incremental_refresh(session, client=Client(), codes=["EPL"], years=[2025], short_transactions=True)
        assert result["error_count"] == 1
        assert session.scalar(select(Fixture)).payload_digest is None
        assert session.get(SyncRun, result["run_id"]).status == "partial"
        assert get_watermark(session, "fixtures:EPL:2025") is None
    fail[0] = False
    with session_factory() as session:
        result = incremental_refresh(session, client=Client(), codes=["EPL"], years=[2025], short_transactions=True)
        assert result["error_count"] == 0 and result["fixtures_changed"] == 1
        assert result["player_stats_rows"] == 1
        assert session.scalar(select(Fixture)).payload_digest is not None


def _client(events):
    class Client:
        def fixtures(self, **_):
            return [_api_fixture_row()]
        def fixture_statistics(self, _):
            return _stats_response()
        def fixture_players(self, _):
            return _players_response()
        def fixture_events(self, _):
            return events()
    return Client()


def test_a_failed_events_fetch_keeps_the_statistics_and_retries_without_duplicates(settings, engine, session_factory):
    from data_platform.sync.pipeline import incremental_refresh
    from data_platform.models import Fixture, FixtureEvent, FixturePlayerStats, FixtureTeamStats

    def failing():
        raise TimeoutError("temporary failure")
    with session_factory() as session:
        result = incremental_refresh(session, client=_client(failing), codes=["EPL"], years=[2025],
                                     short_transactions=True)
        assert [e["stage"] for e in result["errors"]] == ["events"]
        assert session.scalar(select(Fixture)).payload_digest is None  # fetched again next run
        assert len(session.scalars(select(FixtureTeamStats)).all()) == 2
        assert len(session.scalars(select(FixturePlayerStats)).all()) == 1
        assert session.scalars(select(FixtureEvent)).all() == []
    for _ in range(2):
        with session_factory() as session:
            result = incremental_refresh(session, client=_client(_events_response), codes=["EPL"], years=[2025],
                                         short_transactions=True)
            assert result["error_count"] == 0
    with session_factory() as session:
        assert len(session.scalars(select(FixtureEvent)).all()) == 5


def test_before_the_events_table_exists_the_refresh_skips_events(settings, engine, session_factory):
    from data_platform.sync.pipeline import incremental_refresh
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE fixture_match_events")

    def unexpected():
        raise AssertionError("no events request before init-db creates the table")
    with session_factory() as session:
        result = incremental_refresh(session, client=_client(unexpected), codes=["EPL"], years=[2025],
                                     short_transactions=True)
        assert result["error_count"] == 0 and result["event_rows"] == 0 and result["player_stats_rows"] == 1


def _holder(settings):
    """A second connection to the test file, as another process (the measurement heartbeat, 10 Nov) would be."""
    import sqlite3
    return sqlite3.connect(settings.database_url.removeprefix("sqlite:///"), timeout=2.0, isolation_level=None)


def test_a_fixture_update_that_meets_another_writer_lets_it_finish_and_retries(settings, engine, session_factory,
                                                                               monkeypatch):
    """D88: another process held the write lock as the sync updated a fixture. SQLite refuses at once there, without
    its busy wait, because the sync's transaction already reads; every later fixture failed the same way."""
    from data_platform.sync import pipeline
    from data_platform.models import Fixture
    with session_factory() as session:
        pipeline.incremental_refresh(session, client=_client(_events_response), codes=["EPL"], years=[2025],
                                     short_transactions=True)
        before = session.scalar(select(Fixture)).last_fetched_at
    other, waits = _holder(settings), []

    def wait(seconds):
        waits.append(seconds)
        other.execute("COMMIT")  # the other writer finishes; it can only once the sync has let go of its read lock

    class Busy:
        def fixtures(self, **_):
            other.execute("BEGIN IMMEDIATE")
            return [_api_fixture_row()]
    monkeypatch.setattr(pipeline.time, "sleep", wait)
    try:
        with session_factory() as session:
            result = pipeline.incremental_refresh(session, client=Busy(), codes=["EPL"], years=[2025],
                                                  short_transactions=True)
        assert result["error_count"] == 0 and result["fixtures_seen"] == 1
        assert waits == [0.1]
        with session_factory() as session:
            assert session.scalar(select(Fixture)).last_fetched_at > before
    finally:
        if other.in_transaction:
            other.execute("ROLLBACK")
        other.close()


def test_a_lock_that_outlasts_the_retries_fails_the_fixture_without_keeping_the_read_lock(settings, engine,
                                                                                         session_factory, monkeypatch):
    import pytest
    from sqlalchemy.exc import OperationalError
    from data_platform.sync import pipeline
    from data_platform.models import Competition, Season
    with session_factory() as session:
        pipeline.incremental_refresh(session, client=_client(_events_response), codes=["EPL"], years=[2025],
                                     short_transactions=True)
    other, waits = _holder(settings), []
    monkeypatch.setattr(pipeline.time, "sleep", waits.append)
    try:
        other.execute("BEGIN IMMEDIATE")
        with session_factory() as session:
            competition, season = session.scalar(select(Competition)), session.scalar(select(Season))
            with pytest.raises(OperationalError, match="database is locked"):
                pipeline._upsert_fixture(session, api_row=_api_fixture_row(), competition=competition,
                                         season=season, short_transactions=True)
            assert waits == pytest.approx([0.1, 0.2, 0.3])
            assert not session.in_transaction()
        other.execute("COMMIT")  # at once: no read lock is left behind (it would wait 2 s and fail otherwise)
    finally:
        if other.in_transaction:
            other.execute("ROLLBACK")
        other.close()


def test_without_short_transactions_a_lock_is_not_retried(settings, engine, session_factory, monkeypatch):
    """Ending the transaction would discard the run's earlier writes, so only the refresh's short transactions retry."""
    import pytest
    from sqlalchemy.exc import OperationalError
    from data_platform.sync import pipeline
    from data_platform.models import Competition, Season
    with session_factory() as session:
        pipeline.incremental_refresh(session, client=_client(_events_response), codes=["EPL"], years=[2025],
                                     short_transactions=True)
    other, waits = _holder(settings), []
    monkeypatch.setattr(pipeline.time, "sleep", waits.append)
    try:
        other.execute("BEGIN IMMEDIATE")
        with session_factory() as session:
            competition, season = session.scalar(select(Competition)), session.scalar(select(Season))
            with pytest.raises(OperationalError, match="database is locked"):
                pipeline._upsert_fixture(session, api_row=_api_fixture_row(), competition=competition,
                                         season=season, short_transactions=False)
            assert waits == [] and session.in_transaction()
            session.rollback()
    finally:
        if other.in_transaction:
            other.execute("ROLLBACK")
        other.close()
