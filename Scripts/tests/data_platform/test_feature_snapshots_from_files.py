"""Feature snapshot importer from ``Output/*_feature_engineering`` JSON."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def _write_team_profile(base: Path, year: int, rows):
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"team_profiles_{year}.json"
    path.write_text(json.dumps(rows))
    return path


def _write_player_profile(base: Path, year: int, rows):
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"player_profiles_{year}.json"
    path.write_text(json.dumps(rows))
    return path


def _write_team_engineered(base: Path, year: int, rows):
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"team_engineered_features_{year}.json"
    path.write_text(json.dumps(rows))
    return path


def test_import_team_and_player_snapshots(tmp_path, session_factory):
    """Wire a synthetic Output tree and import snapshots from it."""
    output = tmp_path / "Output"
    prem_dir = output / "Prem_feature_engineering"

    _write_team_profile(prem_dir, 2025, [
        {
            "team": "Arsenal",
            "team_id": 42,
            "matches_played": 5,
            "goals_for_pm": 2.2,
            "corners_pm": 6.4,
            "cards_per_90_team": 1.3,
            "control_index": 0.61,
            "archetype": "possession-aggressive",
            "unused_field": "ignored",
        },
        {
            "team": "Spurs",
            "team_id": 43,
            "matches_played": 5,
            "goals_for_pm": 1.6,
            "corners_pm": 5.0,
            "cards_per_90_team": 1.7,
            "control_index": 0.49,
        },
    ])

    _write_player_profile(prem_dir, 2025, [
        {
            "name": "Some Striker",
            "player_id": 9001,
            "team": "Arsenal",
            "team_id": 42,
            "position": "F",
            "appearances_total": 5,
            "appearances_as_starter": 4,
            "start_rate": 0.8,
            "goals_per_90": 0.85,
            "starter_goals_per_90": 0.9,
            "cards_per_90_calc": 0.2,  # alias -> cards_per_90
        }
    ])

    from data_platform.models import PlayerFeatureSnapshot, TeamFeatureSnapshot
    from data_platform.sync import build_feature_snapshots_from_files

    with session_factory() as s:
        res = build_feature_snapshots_from_files(
            s, codes=["EPL"], seasons=[2025], output_dir=output,
            as_of_date=date(2025, 12, 1),
        )
    assert res["team_snapshots"] == 2
    assert res["player_snapshots"] == 1

    with session_factory() as s:
        team_snaps = s.query(TeamFeatureSnapshot).all()
        assert len(team_snaps) == 2
        arsenal = next(snap for snap in team_snaps if float(snap.goals_for_pm) == 2.2)
        assert arsenal.as_of_date == date(2025, 12, 1)
        assert arsenal.archetype == "possession-aggressive"

        player_snaps = s.query(PlayerFeatureSnapshot).all()
        assert len(player_snaps) == 1
        snap = player_snaps[0]
        assert float(snap.goals_per_90) == 0.85
        assert float(snap.cards_per_90) == 0.2, "cards_per_90_calc should have been mapped to cards_per_90"


def test_import_derives_team_snapshots_when_profiles_are_missing(tmp_path, session_factory):
    """Fixture-level team-engineered output is a database-only fallback."""
    output = tmp_path / "Output"
    prem_dir = output / "Prem_feature_engineering"
    _write_team_engineered(prem_dir, 2026, [
        {
            "fixture_id": 100,
            "fixture": "Arsenal vs Spurs",
            "home_team": "Arsenal",
            "away_team": "Spurs",
            "team": "Arsenal",
            "goals": 2,
            "assists": 1,
            "corners": 7,
            "shots_total": 15,
            "shots_on": 6,
            "yellow_cards": 1,
            "red_cards": 0,
            "cards_total": 1,
            "fouls_committed": 9,
            "expected_goals": 2.1,
            "possession": 0.62,
            "passes_total": 500,
            "accurate_passes": 440,
            "dominance_index": 0.70,
            "control_index": 0.65,
            "aggression_index_norm": 0.30,
            "form_index_team": 0.8,
            "fouls_per_90_team": 9.0,
            "cards_per_90_team": 1.0,
            "cards_per_foul_team": 1 / 9,
        },
        {
            "fixture_id": 100,
            "fixture": "Arsenal vs Spurs",
            "home_team": "Arsenal",
            "away_team": "Spurs",
            "team": "Spurs",
            "goals": 0,
            "assists": 0,
            "corners": 2,
            "shots_total": 8,
            "shots_on": 2,
            "yellow_cards": 2,
            "red_cards": 0,
            "cards_total": 2,
            "fouls_committed": 12,
            "expected_goals": 0.5,
            "possession": 0.38,
            "passes_total": 300,
            "accurate_passes": 210,
            "dominance_index": 0.32,
            "control_index": 0.35,
            "aggression_index_norm": 0.20,
            "form_index_team": 0.2,
        },
        {
            "fixture_id": 101,
            "fixture": "Spurs vs Arsenal",
            "home_team": "Spurs",
            "away_team": "Arsenal",
            "team": "Arsenal",
            "goals": 1,
            "assists": 1,
            "corners": 4,
            "shots_total": 10,
            "shots_on": 3,
            "yellow_cards": 2,
            "red_cards": 1,
            "cards_total": 3,
            "fouls_committed": 11,
            "expected_goals": 1.3,
            "possession": 0.54,
            "passes_total": 400,
            "accurate_passes": 300,
            "dominance_index": 0.50,
            "control_index": 0.55,
            "aggression_index_norm": 0.35,
            "form_index_team": 0.6,
            "fouls_per_90_team": 11.0,
            "cards_per_90_team": 3.0,
            "cards_per_foul_team": 3 / 11,
        },
        {
            "fixture_id": 101,
            "fixture": "Spurs vs Arsenal",
            "home_team": "Spurs",
            "away_team": "Arsenal",
            "team": "Spurs",
            "goals": 3,
            "assists": 2,
            "corners": 6,
            "shots_total": 14,
            "shots_on": 7,
            "yellow_cards": 1,
            "red_cards": 0,
            "cards_total": 1,
            "fouls_committed": 8,
            "expected_goals": 2.2,
            "possession": 0.46,
            "passes_total": 450,
            "accurate_passes": 360,
            "dominance_index": 0.65,
            "control_index": 0.61,
            "aggression_index_norm": 0.28,
            "form_index_team": 0.9,
        },
    ])

    from data_platform.models import Team, TeamFeatureSnapshot
    from data_platform.sync import build_feature_snapshots_from_files
    from data_platform.sync.upserts import upsert_team

    # Current engineered files omit team_id.  The platform refresh has already
    # catalogued teams, so the fallback must resolve those rows by name without
    # making an API call.
    with session_factory() as s:
        upsert_team(s, api_id=42, name="Arsenal")
        upsert_team(s, api_id=43, name="Spurs")

    with session_factory() as s:
        result = build_feature_snapshots_from_files(
            s,
            codes=["EPL"],
            seasons=[2026],
            output_dir=output,
            as_of_date=date(2026, 9, 4),
            auto_fetch_teams=False,
        )

    assert result["team_snapshots"] == 2
    assert result["team_snapshots_from_engineered"] == 2
    assert result["player_snapshots"] == 0

    # The weekly command can be safely retried: the snapshot key includes the
    # team, season, and as-of date rather than appending duplicate rows.
    with session_factory() as s:
        retry = build_feature_snapshots_from_files(
            s,
            codes=["EPL"],
            seasons=[2026],
            output_dir=output,
            as_of_date=date(2026, 9, 4),
            auto_fetch_teams=False,
        )
    assert retry["team_snapshots_from_engineered"] == 2

    with session_factory() as s:
        assert s.query(TeamFeatureSnapshot).count() == 2
        arsenal = (
            s.query(TeamFeatureSnapshot)
            .join(Team, Team.id == TeamFeatureSnapshot.team_id)
            .filter(Team.api_football_id == 42)
            .one()
        )
        assert arsenal.matches_played == 2
        assert float(arsenal.goals_for_pm) == 1.5
        assert float(arsenal.goals_against_pm) == 1.5
        assert float(arsenal.corners_pm) == 5.5
        assert float(arsenal.corners_against_pm) == 4.0
        assert float(arsenal.sot_for_pm) == 4.5
        assert float(arsenal.sot_against_pm) == 4.5
        assert float(arsenal.goals_home_pm) == 2.0
        assert float(arsenal.goals_away_pm) == 1.0
        assert float(arsenal.corners_against_home_pm) == 2.0
        assert float(arsenal.corners_against_away_pm) == 6.0
        assert float(arsenal.opp_cards_induced_pm) == 1.5
        assert float(arsenal.pass_accuracy_pct) == round(100 * 740 / 900, 4)
        assert arsenal.archetype == "possession-aggressive"
        assert arsenal.extras["source_kind"] == "derived_team_engineered_features"
        assert arsenal.extras["aggregation"] == "season_to_date_fixture_rows_v1"


def test_engineered_fallback_only_fills_teams_missing_from_profile(tmp_path, session_factory):
    """An explicit profile remains authoritative, while a partial one is recoverable."""
    output = tmp_path / "Output"
    prem_dir = output / "Prem_feature_engineering"
    _write_team_profile(prem_dir, 2026, [{
        "team": "Arsenal",
        "team_id": 42,
        "matches_played": 99,
        "goals_for_pm": 9.9,
    }])
    _write_team_engineered(prem_dir, 2026, [
        {
            "fixture_id": 200,
            "fixture": "Arsenal vs Spurs",
            "home_team": "Arsenal",
            "away_team": "Spurs",
            "team": "Arsenal",
            "team_id": 42,
            "goals": 1,
        },
        {
            "fixture_id": 200,
            "fixture": "Arsenal vs Spurs",
            "home_team": "Arsenal",
            "away_team": "Spurs",
            "team": "Spurs",
            "team_id": 43,
            "goals": 0,
        },
    ])

    from data_platform.models import Team, TeamFeatureSnapshot
    from data_platform.sync import build_feature_snapshots_from_files

    with session_factory() as s:
        result = build_feature_snapshots_from_files(
            s,
            codes=["EPL"],
            seasons=[2026],
            output_dir=output,
            as_of_date=date(2026, 9, 4),
            auto_fetch_teams=False,
        )

    assert result["team_snapshots"] == 2
    assert result["team_snapshots_from_engineered"] == 1
    with session_factory() as s:
        snapshots = {
            team.api_football_id: snapshot
            for snapshot, team in (
                s.query(TeamFeatureSnapshot, Team)
                .join(Team, Team.id == TeamFeatureSnapshot.team_id)
                .all()
            )
        }
        assert float(snapshots[42].goals_for_pm) == 9.9
        assert snapshots[42].extras["source_file"] == "team_profiles_2026.json"
        assert float(snapshots[43].goals_for_pm) == 0.0
        assert snapshots[43].extras["source_kind"] == "derived_team_engineered_features"
