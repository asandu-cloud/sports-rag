"""Replay planner + executor tests using synthetic payloads."""

from __future__ import annotations

from datetime import date, datetime, timezone


def _fixture_row(api_id: int, home_id: int, away_id: int, league_id: int = 39, season: int = 2025):
    return {
        "fixture": {
            "id": api_id,
            "date": "2025-08-10T14:00:00+00:00",
            "timestamp": int(datetime(2025, 8, 10, 14, 0, tzinfo=timezone.utc).timestamp()),
            "venue": {"name": "Emirates", "city": "London"},
            "status": {"short": "FT", "elapsed": 90},
            "referee": "Ref",
        },
        "league": {"id": league_id, "season": season, "round": "Regular Season - 1"},
        "teams": {
            "home": {"id": home_id, "name": f"Team {home_id}"},
            "away": {"id": away_id, "name": f"Team {away_id}"},
        },
        "goals": {"home": 2, "away": 1},
    }


def test_planner_dedupes_targets(session_factory, monkeypatch):
    import data_platform.db as db_module
    monkeypatch.setattr(db_module, "session_scope", session_factory)
    from data_platform.replay import ReplayScope, build_plan
    plan = build_plan(ReplayScope(fixture_api_ids=(42, 42, 43), kb_entities=(("team", "EPL:1"),)))
    assert plan.fixtures == [42, 43]
    assert plan.kb_entities == [("team", "EPL:1")]


def test_executor_runs_against_stub_fetcher(session_factory, monkeypatch):
    import data_platform.db as db_module
    monkeypatch.setattr(db_module, "session_scope", session_factory)

    from data_platform.models import Fixture, SyncRun
    from data_platform.replay import ReplayExecutor, ReplayPlan

    calls = {"kb": []}

    def kb_enqueuer(etype, ekey, reason):
        calls["kb"].append((etype, ekey, reason))
        return 1

    executor = ReplayExecutor(
        fetch_fixture_row=lambda api_id: _fixture_row(api_id, 100, 200),
        fetch_team_stats=lambda api_id: [
            {"team": {"id": 100, "name": "Team 100"},
             "statistics": [{"type": "Total Shots", "value": 12},
                            {"type": "Corner Kicks", "value": 5}]},
            {"team": {"id": 200, "name": "Team 200"},
             "statistics": [{"type": "Total Shots", "value": 8},
                            {"type": "Corner Kicks", "value": 3}]},
        ],
        fetch_player_stats=lambda api_id: [],
        kb_enqueuer=kb_enqueuer,
        session_factory=session_factory,
    )

    plan = ReplayPlan(fixtures=[9001], kb_entities=[("team", "EPL:100")])
    result = executor.execute(plan)
    assert result.fixtures_replayed == 1
    assert not result.fixture_failures
    # KB enqueue called once for the fixture + once for the explicit kb_entity
    assert len(calls["kb"]) == 2

    with session_factory() as session:
        fx = session.query(Fixture).first()
        assert fx is not None
        assert fx.api_football_id == 9001
        run = session.query(SyncRun).filter_by(run_kind="replay").first()
        assert run is not None
        assert run.status == "completed"
        assert run.stats["fixtures_replayed"] == 1


def test_compare_values_tolerates_floats():
    from data_platform.replay.compare import compare_values
    diffs = compare_values(
        {"goals_for_pm": 2.100001, "label": "x", "id": 1},
        {"goals_for_pm": 2.100002, "label": "x", "id": 2},
    )
    assert diffs == []

    diffs = compare_values(
        {"goals_for_pm": 2.10, "label": "x"},
        {"goals_for_pm": 2.30, "label": "y"},
    )
    assert {d.path for d in diffs} == {"goals_for_pm", "label"}
