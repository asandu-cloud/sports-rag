"""Provider calls must not hold SQLite's write lock; failed details retry."""
from sqlalchemy import select
from Scripts.tests.data_platform.test_sync_upserts import _api_fixture_row, _stats_response, _players_response


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
        result = incremental_refresh(session, client=Client(), codes=["EPL"], years=[2025], short_transactions=True)
        assert result["error_count"] == 0 and result["player_stats_rows"] == 1
        result = incremental_refresh(session, client=Client(), codes=["EPL"], years=[2025], short_transactions=True)
        assert result["fixtures_changed"] == 0
        assert calls == ["fixtures", "team", "players", "fixtures"]
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
