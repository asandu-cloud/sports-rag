"""Prospective measurement safety tests: private SQLite and mocked transport."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from data_platform.models import OddsSnapshot, SyncRun
from data_platform.services.measurement_runtime import BoundedFootballClient, BudgetExceeded, CONTROL, MeasurementRuntime
from data_platform.services.measurement_cycle import MeasurementCycle, LEASE
from .test_safe_settlement import store, publish, payload, fixture, team_stats, pred, NOW, no_network

KICKOFF = datetime(2026, 9, 22, 18, tzinfo=timezone.utc)


class Client:
    def __init__(self, status="FT"):
        self.fx, self.stats = fixture(status), team_stats()
        self.calls = []
        self.odds = odds_row()

    def fixture(self, fid):
        self.calls.append(("fixture", fid))
        return [deepcopy(self.fx)]

    def fixture_statistics(self, fid):
        self.calls.append(("stats", fid))
        return deepcopy(self.stats)

    def _get(self, path, params):
        self.calls.append((path, params))
        return {"response": [deepcopy(self.odds)]}


def odds_row():
    return {"fixture": {"id": 123, "date": KICKOFF.isoformat()}, "league": {"id": 39},
            "update": (KICKOFF - timedelta(minutes=5)).isoformat(),
            "bookmakers": [{"id": 1, "name": "Book", "bets": [
                {"id": 5, "values": [{"value": "Over 2.5", "odd": "1.8"}]}]}]}


def worker(store, now=NOW, client=None):
    repo = store[0]
    runtime = MeasurementRuntime(repo._factory)
    return MeasurementCycle(repo, runtime, client=client or Client(), clock=lambda: now)


def test_activation_is_prospective_and_reinstall_preserves_boundary(store):
    old = publish(store)
    cycle = worker(store)
    activated = cycle.runtime.activate(now=NOW)
    new = publish(store, payload(line=3.25))
    assert cycle.runtime.activate(now=NOW) == activated
    result = cycle.run_once(kinds=("settlement",))
    assert result["candidates"] == 1
    rows = {row["id"]: row for row in store[0].get_recent(days=3650)}
    assert rows[old]["outcome"] is None and rows[old]["settlement"] is None
    assert rows[new]["outcome"] == "half_miss"
    assert cycle.client.calls == [("fixture", 123)]
    assert cycle.run_once(kinds=("settlement",))["details"] == []


def test_disabled_and_offline_dry_run_never_fetch_or_write(store):
    cycle = worker(store)
    publish(store)
    assert cycle.run_once()["status"] == "disabled"
    cycle.runtime.activate(now=NOW)
    publish(store, payload(line=3.25))
    before = store[0].get_recent(days=3650)
    assert cycle.run_once(dry_run=True)["status"] == "planned"
    assert before == store[0].get_recent(days=3650)
    assert cycle.client.calls == []
    with store[0]._factory() as s:
        assert s.scalar(select(func.count()).select_from(SyncRun)) == 0


def test_overlap_skips_and_status_detects_stale_worker(store):
    cycle = worker(store)
    cycle.runtime.activate(now=NOW)
    cycle.leases.acquire_lease(lease_key=LEASE, owner_id="other", ttl_seconds=120, now=NOW)
    assert cycle.run_once()["status"] == "overlap_skipped"
    assert cycle.client.calls == []
    assert cycle.runtime.status(now=NOW)["stale"]


def test_pending_stats_are_delayed_but_can_recover(store):
    client = Client()
    client.stats = []
    cycle = worker(store, client=client)
    cycle.runtime.activate(now=NOW)
    publish(store, payload("sot", "shots_on_target_over_under", line=5.5))
    report = cycle.run_once(kinds=("settlement",))
    assert report["status"] == "warning"
    assert report["held_reasons"]
    cycle.run_once(kinds=("settlement",))
    assert len(client.calls) == 2
    client.stats = team_stats()
    cycle.clock = lambda: NOW + timedelta(minutes=6)
    assert cycle.run_once(kinds=("settlement",))["details"][0]["result"]["graded"] == 1


@pytest.mark.parametrize("missing_period", [True, False])
def test_unverified_contracts_hold_without_api_calls(store, missing_period):
    cycle = worker(store)
    cycle.runtime.activate(now=NOW)
    p = payload() if missing_period else payload("cards", "totals_cards_over_under")
    if missing_period:
        p["decision"]["quote"].pop("period")
    pid = publish(store, p)
    if not missing_period:
        # Existing card publications have no approved product-policy snapshot.
        from data_platform.models import Prediction
        with store[0]._factory() as session:
            row = session.get(Prediction, pid)
            row.extras = {k: v for k, v in row.extras.items() if k != "settlement_policy"}
    result = cycle.run_once(kinds=("settlement",))
    assert result["held_reasons"]
    assert cycle.client.calls == []
    before = store[0].get_recent(days=3650)
    cycle.clock = lambda: NOW + timedelta(hours=8)
    cycle.run_once(kinds=("settlement",))
    assert before == store[0].get_recent(days=3650)


def test_deadline_fences_pending_and_outcome_writes(store):
    cycle = worker(store)
    cycle.runtime.activate(now=NOW)
    publish(store)
    calls = [0]
    def clock():
        calls[0] += 1
        return NOW if calls[0] == 1 else NOW + timedelta(seconds=51)
    cycle.clock = clock
    assert cycle.run_once()["status"] == "partial"
    assert store[0].get_unresolved()[0]["settlement"] is None
    assert cycle.client.calls == []


def test_request_failures_consume_durable_budget_without_retries(store, settings):
    runtime = MeasurementRuntime(store[0]._factory)
    calls = []
    class Session:
        headers = {}
        def get(self, *args, **kwargs):
            calls.append(kwargs)
            raise TimeoutError("transport timeout")
    def client():
        return BoundedFootballClient(runtime, cycle_limit=2, daily_limit=2, guard=lambda: None,
                                     clock=lambda: NOW, session=Session(), settings=settings, request_pause_s=0)
    for _ in range(2):
        with pytest.raises(TimeoutError):
            client().fixture(123)
    with pytest.raises(BudgetExceeded):
        client().fixture(123)
    assert len(calls) == 2
    assert all(c["allow_redirects"] is False and c["timeout"] == 8 for c in calls)
    assert runtime.state("measurement:requests:2026-09-24")["attempts"] == 2


@pytest.mark.parametrize("response", [
    SimpleNamespace(status_code=302),
    SimpleNamespace(status_code=200, json=lambda: {"errors": {"limit": "quota"}, "response": []}),
    SimpleNamespace(status_code=200, json=lambda: {"response": [], "paging": {"total": 2}}),
])
def test_bad_transport_responses_are_not_retried(store, settings, response):
    client = BoundedFootballClient(MeasurementRuntime(store[0]._factory), cycle_limit=1, daily_limit=5,
        guard=lambda: None, settings=settings, session=SimpleNamespace(headers={}, get=lambda *a, **kw: response))
    with pytest.raises(Exception):
        client.fixture(123)
    with pytest.raises(BudgetExceeded):
        client.fixture(123)
    assert client.attempts == 1


def test_new_scheduler_render_is_independent_and_read_only(tmp_path):
    from Scripts.ops.measurement_launchd import operate, LABEL
    from Scripts.ops.match_read_launchd import build_paths
    paths = build_paths(root=tmp_path, python_executable=tmp_path / ".venv/bin/python", plist_path=tmp_path / "job.plist")
    report = operate("install", paths, dry_run=True)
    assert report["configuration"]["Label"] == LABEL
    assert report["configuration"]["StartInterval"] == 60
    assert "measurement-cycle" in report["configuration"]["ProgramArguments"]
    assert not paths.plist_path.exists()


def test_platform_closing_bridge_never_falls_back(monkeypatch):
    from Scripts.rag_ingest import prediction_tracker as tracker
    import data_platform.services.measurement_cycle as module
    monkeypatch.setattr(tracker, "_platform_on", lambda: True)
    monkeypatch.setattr(tracker, "_get_db", lambda *a: pytest.fail("Legacy database must not be opened"))
    monkeypatch.setattr(module, "run_measurement_cycle", lambda **kw: {"status": "disabled", "errors": 0})
    assert tracker.capture_closing_odds()["status"] == "disabled"
    def fail(**kwargs):
        raise RuntimeError("unavailable")
    monkeypatch.setattr(module, "run_measurement_cycle", fail)
    assert tracker.capture_closing_odds()["errors"] == 1


def test_diagnostic_contract_probes_are_labelled_and_consistent():
    from Scripts.ops.prediction_settlement_diagnostic import probes
    result = probes(fixture(), team_stats())
    assert len(result) == 15
    assert all(p["passed"] for p in result)


def test_empty_enabled_cycle_records_heartbeat_without_api(store):
    cycle = worker(store)
    cycle.runtime.activate(now=NOW)
    assert cycle.run_once()["candidates"] == 0
    assert cycle.client.calls == []
    assert cycle.runtime.status(now=NOW)["status"] == "ok"
    assert cycle.runtime.status(now=NOW + timedelta(minutes=6))["status"] == "warn"


def test_operational_ledgers_do_not_look_like_stale_ingestion(store):
    from data_platform.observability.metrics import watermark_metrics
    runtime = MeasurementRuntime(store[0]._factory)
    runtime.activate(now=NOW)
    runtime.reserve_request(day=NOW.date().isoformat(), limit=10)
    assert watermark_metrics()["stale"] == []


@pytest.mark.parametrize("success", [True, False])
def test_scheduler_install_enables_first_and_fails_closed(tmp_path, monkeypatch, success):
    from Scripts.ops import measurement_launchd as module
    paths = module.build_paths(root=tmp_path, python_executable=tmp_path / ".venv/bin/python", plist_path=tmp_path / "job.plist")
    calls = []
    monkeypatch.setattr(module, "_assert_installable", lambda p: None)
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kw: calls.append(args[-1]) or SimpleNamespace(returncode=0, stdout='{"enabled": true}'))
    def launchctl(args, **kw):
        calls.append(args[0])
        if args[0] == "bootstrap" and not success:
            raise RuntimeError("launchd unavailable")
    monkeypatch.setattr(module, "_run_launchctl", launchctl)
    if success:
        assert module.operate("install", paths)["installed"]
        assert calls == ["measurement-enable", "bootout", "bootstrap"]
    else:
        with pytest.raises(RuntimeError):
            module.operate("install", paths)
        assert calls[-1] == "measurement-disable"


def test_refresh_runs_prospective_settlement_hook_only_after_success(store, monkeypatch):
    from contextlib import nullcontext
    from data_platform import cli
    import data_platform.services.measurement_cycle as module
    import data_platform.services.refresh_coordination as coordination
    import data_platform.services.worker_runtime as worker_runtime
    calls = []
    result = [0]
    args = SimpleNamespace(command="refresh", func=lambda _: result[0], dry_run=False)
    monkeypatch.setattr(cli, "build_parser", lambda: SimpleNamespace(parse_args=lambda _: args))
    monkeypatch.setattr(coordination, "data_access", lambda **kw: nullcontext())
    monkeypatch.setattr(worker_runtime, "prevent_idle_sleep", nullcontext)
    monkeypatch.setattr(module, "run_measurement_cycle", lambda **kw: calls.append(kw) or {"errors": 0})
    assert cli.main([]) == 0
    assert calls == [{"kinds": ("settlement",)}]
    result[0] = 1
    assert cli.main([]) == 1
    assert len(calls) == 1


def test_refresh_blocker_is_visible_and_never_fetches_or_grades(store, monkeypatch):
    from data_platform.services.measurement_cycle import run_measurement_cycle
    import data_platform.services.refresh_coordination as coordination
    runtime = MeasurementRuntime(store[0]._factory)
    runtime.activate(now=NOW)
    publish(store)
    def blocked():
        raise coordination.RefreshBusy("Full refresh incomplete; rerun refresh-season")
    monkeypatch.setattr(coordination, "data_access", blocked)
    result = run_measurement_cycle()
    assert result["status"] == "refresh_deferred" and result["provider_attempts"] == 0
    assert runtime.status()["status"] == "warn"
    assert "refresh-season" in runtime.status()["recent_runs"][0]["report"]["reason"]
    assert store[0].get_unresolved()[0]["settlement"] is None


def test_open_only_candidates_skip_graded_picks_with_a_final_closing_price(store):
    from data_platform.models import Prediction
    ids = [publish(store, payload(line=line)) for line in (1.5, 2.5, 3.5, 4.5, 5.5)]
    open_pick, no_capture, null_extras, waiting, done = ids
    with store[0]._factory() as session:
        for pid in (no_capture, null_extras, waiting, done):
            session.get(Prediction, pid).outcome = "win"
        session.get(Prediction, null_extras).extras = None
        for pid, status in ((waiting, "pending"), (done, "final")):
            row = session.get(Prediction, pid)
            row.extras = {**(row.extras or {}), "closing_capture": {"status": status}}
        session.commit()
    everything = store[0].settlement_candidates(include_graded=True, published_after_id=0)
    still_open = store[0].settlement_candidates(include_graded=True, published_after_id=0, open_only=True)
    assert {p["id"] for p in everything} == set(ids)
    assert {p["id"] for p in still_open} == {open_pick, no_capture, null_extras, waiting}


def test_a_run_cut_short_by_the_deadline_before_any_request_is_marked_busy(store):
    cycle = worker(store)
    cycle.runtime.activate(now=NOW)
    publish(store)
    calls = [0]
    def clock():
        calls[0] += 1
        return NOW if calls[0] == 1 else NOW + timedelta(seconds=51)
    cycle.clock = clock
    report = cycle.run_once()
    assert report["busy"] is True and report["provider_attempts"] == 0


def test_a_run_stopped_by_the_request_budget_is_not_busy(store):
    cycle = worker(store)
    cycle.runtime.activate(now=NOW)
    publish(store)
    class Spent(Client):
        def fixture(self, fid):
            raise BudgetExceeded("Daily provider request budget exhausted")
    cycle.client = Spent()
    report = cycle.run_once(kinds=("settlement",))
    assert not report.get("busy")
