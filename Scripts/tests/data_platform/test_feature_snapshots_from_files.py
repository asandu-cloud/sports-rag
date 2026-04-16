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
