"""Offline regression checks for fair work and refresh/DB coordination."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import OperationalError

from Scripts.tests.data_platform.test_match_read_cycle import _settings, _fixture


def test_oldest_unattempted_fixture_precedes_due_failed_retry(settings, engine, session_factory, monkeypatch):
    from data_platform.repositories.match_read_cycle import MatchReadCycleRepository
    from data_platform.services.match_read_cycle import MatchReadCycleService, _dispatch_summary
    now = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
    repo = MatchReadCycleRepository(session_factory)
    repo.record_observation(fixture_api_id="1", league="EPL", stage="pre_match", status="failed",
                            checked_at=now - timedelta(hours=1))
    fixtures = [_fixture(fixture_id=str(i), kickoff=now + timedelta(hours=3)) for i in (1, 2, 3)]
    worker = MatchReadCycleService(settings=_settings(), session_factory=session_factory, fixture_loader=lambda **_: fixtures)
    order = []
    def dispatch(job, **_):
        order.append(job.fixture_ids[0])
        return _dispatch_summary(job, run=None), set()
    monkeypatch.setattr(worker, "_dispatch_job", dispatch)
    worker.run_once(now=now, mode="shadow")
    assert order == ["2", "3", "1"]


def test_does_not_start_dispatch_with_only_twenty_seconds_left(settings, engine, session_factory, monkeypatch):
    from data_platform.services.match_read_cycle import MatchReadCycleService, _dispatch_summary
    now = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
    current = [now]
    fixtures = [_fixture(fixture_id=str(i), kickoff=now + timedelta(hours=3)) for i in (1, 2)]
    worker = MatchReadCycleService(settings=replace(_settings(), max_cycle_seconds=150),
        session_factory=session_factory, fixture_loader=lambda **_: fixtures, clock=lambda: current[0])
    def dispatch(job, **_):
        current[0] += timedelta(seconds=130)
        return _dispatch_summary(job, run=None), set()
    monkeypatch.setattr(worker, "_dispatch_job", dispatch)
    result = worker.run_once(now=now, mode="shadow")
    assert len(result.dispatches) == 1
    assert result.deferred_fixture_stages == 1
    assert not result.errors


def test_persisted_backoff_grows_and_survives_new_worker(settings, engine, session_factory):
    from data_platform.services.match_read_cycle import MatchReadCycleService
    now = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
    fixture = _fixture(kickoff=now + timedelta(hours=4))
    worker = MatchReadCycleService(settings=_settings(), session_factory=session_factory, fixture_loader=lambda **_: [fixture])
    job = worker.plan(now=now).jobs[0]
    for count in range(1, 5):
        observation = worker._record_failed_observation(fixture_id="9001", job=job, sync_run_id=None,
                                                       checked_at=now, error="provider down")
        assert observation["detail"]["consecutive_failures"] == count
    assert not worker.plan(now=now + timedelta(minutes=29)).jobs
    assert worker.plan(now=now + timedelta(minutes=30)).jobs


def test_pending_lineup_preflight_never_calls_dispatcher(settings, engine, session_factory):
    from data_platform.services.match_read_cycle import MatchReadCycleService
    from data_platform.repositories.match_read_cycle import MatchReadCycleRepository
    now = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
    repo = MatchReadCycleRepository(session_factory)
    repo.record_observation(fixture_api_id="9001", league="EPL", stage="pre_match", status="verified", checked_at=now)
    def dispatch(*_, **__):
        raise AssertionError("No odds or projections should be requested")
    worker = MatchReadCycleService(settings=_settings(), session_factory=session_factory,
        fixture_loader=lambda **_: [_fixture(kickoff=now + timedelta(minutes=50))], dispatcher=dispatch,
        lineup_preflight=lambda *_: SimpleNamespace(is_available=False, source="unavailable"))
    result = worker.run_once(now=now, mode="shadow")
    assert not result.errors
    assert result.dispatches[0]["observations"][0]["status"] == "lineups_pending"
    assert not result.releases


def test_cleanup_failure_does_not_escape_or_hide_run_result(settings, engine, session_factory, monkeypatch):
    from data_platform.services.match_read_cycle import MatchReadCycleService
    worker = MatchReadCycleService(settings=_settings(), session_factory=session_factory, fixture_loader=lambda **_: [])
    def locked(**_):
        raise OperationalError("update", {}, sqlite3.OperationalError("database is locked"))
    monkeypatch.setattr(worker._repo, "release_lease", locked)
    result = worker.run_once(mode="shadow")
    assert result.run_id is not None
    assert any("cleanup failed" in error for error in result.errors)
    assert not result.succeeded


def test_start_failure_releases_claimed_lease(settings, engine, session_factory, monkeypatch):
    from data_platform.services.match_read_cycle import MatchReadCycleService, MATCH_READ_CYCLE_LEASE_KEY
    worker = MatchReadCycleService(settings=_settings(), session_factory=session_factory, fixture_loader=lambda **_: [])
    def locked(**_):
        raise OperationalError("insert", {}, sqlite3.OperationalError("database is locked"))
    monkeypatch.setattr(worker, "_start_run", locked)
    result = worker.run_once(mode="shadow")
    assert any("Could not start" in error for error in result.errors)
    assert worker._repo.active_lease(MATCH_READ_CYCLE_LEASE_KEY) is None


def test_finish_failure_is_reported_and_still_releases_lease(settings, engine, session_factory, monkeypatch):
    from data_platform.services.match_read_cycle import MatchReadCycleService, MATCH_READ_CYCLE_LEASE_KEY
    worker = MatchReadCycleService(settings=_settings(), session_factory=session_factory, fixture_loader=lambda **_: [])
    def locked(*_, **__):
        raise OperationalError("update", {}, sqlite3.OperationalError("database is locked"))
    monkeypatch.setattr(worker, "_finish_run", locked)
    result = worker.run_once(mode="shadow")
    assert any("Could not save worker run" in error for error in result.errors)
    assert worker._repo.active_lease(MATCH_READ_CYCLE_LEASE_KEY) is None


def test_failed_cli_refresh_leaves_incomplete_marker(settings, engine, monkeypatch):
    from data_platform import cli
    from data_platform.services.refresh_coordination import data_access, RefreshBusy
    args = SimpleNamespace(command="refresh-season", dry_run=False, verbose=False, func=lambda _: 1)
    monkeypatch.setattr(cli, "build_parser", lambda: SimpleNamespace(parse_args=lambda _: args))
    assert cli.main([]) == 1
    with pytest.raises(RefreshBusy):
        with data_access():
            pass
    args.func = lambda _: 0
    assert cli.main([]) == 0
    with data_access():
        pass


def test_real_season_child_command_inherits_gate(settings, engine):
    from data_platform.services.refresh_coordination import data_access
    from data_platform.season_refresh import _run_command
    script = "from Scripts.data_platform.services.refresh_coordination import data_access;\nwith data_access(writer=True): pass"
    with data_access(writer=True, full_refresh=True):
        result = _run_command([sys.executable, "-c", script], runner=subprocess.run)
        assert result.returncode == 0


def test_busy_retry_reopens_transaction_and_does_not_retry_other_errors(monkeypatch):
    from data_platform.db import retry_database_busy
    monkeypatch.setattr("data_platform.db.time.sleep", lambda _: None)
    calls = []
    @retry_database_busy
    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise OperationalError("update", {}, sqlite3.OperationalError("database is locked"))
        return 42
    assert operation() == 42 and len(calls) == 3
    calls.clear()
    @retry_database_busy
    def invalid():
        calls.append(1)
        raise OperationalError("query", {}, sqlite3.OperationalError("no such table"))
    with pytest.raises(OperationalError):
        invalid()
    assert len(calls) == 1


def test_refresh_gate_excludes_other_processes_and_inherits_to_children(settings, engine):
    from data_platform.services.refresh_coordination import data_access
    script = "from Scripts.data_platform.services.refresh_coordination import data_access;\nwith data_access(): print('acquired')"
    with data_access(writer=True, full_refresh=True):
        env = dict(os.environ)
        descriptor = int(env.pop("BETTING_REFRESH_LOCK_FD"))
        blocked = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
        assert blocked.returncode != 0 and "RefreshBusy" in blocked.stderr
        inherited = subprocess.run([sys.executable, "-c", script], pass_fds=(descriptor,), capture_output=True, text=True)
        assert inherited.returncode == 0, inherited.stderr
    with data_access():
        pass


def test_failed_full_refresh_blocks_generation_until_successful_rerun(settings, engine):
    from data_platform.services.refresh_coordination import data_access, RefreshBusy
    with pytest.raises(RuntimeError):
        with data_access(writer=True, full_refresh=True):
            raise RuntimeError("embedding interrupted")
    with pytest.raises(RefreshBusy, match="did not complete"):
        with data_access():
            pass
    # An isolated raw update must NOT clear a failed multi-stage refresh.
    with data_access(writer=True):
        pass
    with pytest.raises(RefreshBusy):
        with data_access():
            pass
    with data_access(writer=True, full_refresh=True):
        pass
    with data_access():
        pass


def test_atomic_expired_lease_takeover_has_one_winner(settings, engine, session_factory):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from data_platform.repositories.match_read_cycle import MatchReadCycleRepository
    now = datetime.now(timezone.utc)
    repo = MatchReadCycleRepository(session_factory)
    repo.acquire_lease(lease_key="test", owner_id="expired", ttl_seconds=1, now=now - timedelta(minutes=1))
    barrier = Barrier(2)
    def acquire(owner):
        barrier.wait()
        return repo.acquire_lease(lease_key="test", owner_id=owner, ttl_seconds=60, now=now)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(acquire, ["one", "two"]))
    assert sorted(results) == [False, True]


def test_refresh_between_fixtures_yields_to_new_process(settings, engine, session_factory, monkeypatch):
    from data_platform.services.match_read_cycle import MatchReadCycleService, _dispatch_summary
    from data_platform.services.refresh_coordination import _paths
    now = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
    fixtures = [_fixture(fixture_id=str(i), kickoff=now + timedelta(hours=3)) for i in (1, 2)]
    worker = MatchReadCycleService(settings=_settings(), session_factory=session_factory, fixture_loader=lambda **_: fixtures)
    def dispatch(job, **_):
        # Simulate a refresh revision appearing before the next fixture.
        Path(str(_paths()[0]) + ".generation").write_text("new inputs")
        return _dispatch_summary(job, run=None), set()
    monkeypatch.setattr(worker, "_dispatch_job", dispatch)
    result = worker.run_once(now=now, mode="shadow")
    assert len(result.dispatches) == 1 and result.deferred_fixture_stages == 1
    assert "new process" in result.yield_reason
