"""Tests for the one-command weekly refresh orchestration."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace


def test_current_season_year_uses_july_rollover():
    from data_platform.season_refresh import current_season_year

    assert current_season_year(date(2026, 6, 30)) == 2025
    assert current_season_year(date(2026, 7, 1)) == 2026


def test_default_plan_keeps_both_active_data_paths_in_order():
    from data_platform.season_refresh import (
        DOMESTIC_COMPETITIONS,
        build_refresh_plan,
        resolve_competitions,
    )

    codes = resolve_competitions(None)
    assert codes == DOMESTIC_COMPETITIONS

    plan = build_refresh_plan(season=2026, competitions=codes)
    assert [stage.name for stage in plan] == [
        "platform_incremental_sync",
        "legacy_model_refresh",
        "platform_feature_snapshots",
        "platform_health",
    ]

    platform_sync = plan[0].command
    assert ("--competition", *DOMESTIC_COMPETITIONS) == platform_sync[
        platform_sync.index("--competition"):platform_sync.index("--season")
    ]
    assert platform_sync[platform_sync.index("--season") + 1] == "2026"

    legacy = plan[1].command
    assert "--normalize" in legacy
    assert "--embed" in legacy
    assert "--retrain-ml" in legacy
    assert "--build-referees" in legacy
    assert plan[-1].accepted_exit_codes == (0, 1)


def test_execute_allows_health_warning_after_successful_refresh():
    from data_platform.season_refresh import RefreshStage, execute_refresh_plan

    calls = []

    def fake_runner(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if command[-1] == "health" else 0)

    ticks = iter(range(20))
    plan = [
        RefreshStage("sync", ("sync",)),
        RefreshStage("legacy", ("legacy",)),
        RefreshStage("health", ("health",), accepted_exit_codes=(0, 1)),
    ]
    report = execute_refresh_plan(
        plan,
        season=2026,
        competitions=("EPL",),
        runner=fake_runner,
        monotonic=lambda: next(ticks),
    )

    assert report.succeeded is True
    assert report.failed_stage is None
    assert calls == [["sync"], ["legacy"], ["health"]]
    assert report.stages[-1].status == "warning"
    assert report.warnings
    assert report.as_dict()["restart_long_lived_workers"] is True


def test_execute_stops_before_later_stages_after_failure():
    from data_platform.season_refresh import RefreshStage, execute_refresh_plan

    calls = []

    def fake_runner(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=9 if command[0] == "broken" else 0)

    ticks = iter(range(20))
    report = execute_refresh_plan(
        [
            RefreshStage("first", ("first",)),
            RefreshStage("broken", ("broken",)),
            RefreshStage("must_not_run", ("must_not_run",)),
        ],
        season=2026,
        competitions=("EPL",),
        runner=fake_runner,
        monotonic=lambda: next(ticks),
    )

    assert report.succeeded is False
    assert report.failed_stage == "broken"
    assert calls == [["first"], ["broken"]]
    assert len(report.stages) == 2
    assert report.stages[-1].status == "failed"


def test_dry_run_never_invokes_a_child_process():
    from data_platform.season_refresh import RefreshStage, execute_refresh_plan

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("dry run must not invoke a child command")

    report = execute_refresh_plan(
        [RefreshStage("one", ("one",))],
        season=2026,
        competitions=("EPL",),
        dry_run=True,
        runner=fail_if_called,
    )

    assert report.succeeded is True
    assert report.stages[0].status == "not_run"
    assert report.as_dict()["restart_long_lived_workers"] is False
