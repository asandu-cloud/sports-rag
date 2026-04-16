"""Upsert idempotency + watermark progression.

We drive upserts directly with synthetic API payloads so no network is
required. The key invariants: running the same sync twice produces the
same row count, and changed payloads update in place.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from data_platform.sync.apifootball import CompetitionSpec
from data_platform.sync.upserts import (
    get_watermark,
    upsert_competition,
    upsert_fixture_from_api_row,
    upsert_fixture_player_stats,
    upsert_fixture_team_stats,
    upsert_season,
    upsert_watermark,
)


SPEC = CompetitionSpec("EPL", "Premier League", 39, "England", "domestic_league")


def _api_fixture_row(api_id: int = 42, home_goals: int = 2, away_goals: int = 1, status: str = "FT"):
    return {
        "fixture": {
            "id": api_id,
            "date": "2025-08-10T14:00:00+00:00",
            "timestamp": int(datetime(2025, 8, 10, 14, 0, tzinfo=timezone.utc).timestamp()),
            "venue": {"name": "Stadium A", "city": "London"},
            "status": {"short": status, "elapsed": 90},
            "referee": "Ref McRefFace",
        },
        "league": {"id": SPEC.api_football_id, "season": 2025, "round": "Regular Season - 1"},
        "teams": {
            "home": {"id": 100, "name": "Home FC"},
            "away": {"id": 200, "name": "Away United"},
        },
        "goals": {"home": home_goals, "away": away_goals},
    }


def _stats_response():
    return [
        {
            "team": {"id": 100, "name": "Home FC"},
            "statistics": [
                {"type": "Total Shots", "value": 12},
                {"type": "Shots on Goal", "value": 6},
                {"type": "Corner Kicks", "value": 5},
                {"type": "Ball Possession", "value": "55%"},
                {"type": "Yellow Cards", "value": 1},
                {"type": "Red Cards", "value": 0},
                {"type": "Fouls", "value": 10},
            ],
        },
        {
            "team": {"id": 200, "name": "Away United"},
            "statistics": [
                {"type": "Total Shots", "value": 8},
                {"type": "Shots on Goal", "value": 3},
                {"type": "Corner Kicks", "value": 3},
                {"type": "Ball Possession", "value": "45%"},
                {"type": "Yellow Cards", "value": 2},
                {"type": "Red Cards", "value": 0},
                {"type": "Fouls", "value": 14},
            ],
        },
    ]


def _players_response():
    return [
        {
            "team": {"id": 100, "name": "Home FC"},
            "players": [
                {
                    "player": {"id": 9001, "name": "A Scorer"},
                    "statistics": [{
                        "games": {"minutes": 90, "position": "F", "rating": "8.1"},
                        "goals": {"total": 2, "assists": 0},
                        "shots": {"total": 5, "on": 3},
                        "passes": {"total": 40, "accuracy": 32},
                        "cards": {"yellow": 0, "red": 0},
                        "fouls": {"committed": 1, "drawn": 3},
                        "tackles": {"total": 1, "interceptions": 0},
                        "duels": {"won": 6, "total": 11},
                    }],
                }
            ],
        }
    ]


def test_fixture_upsert_is_idempotent(session_factory):
    with session_factory() as s:
        comp = upsert_competition(s, SPEC)
        season = upsert_season(s, competition=comp, year=2025)
        s.flush()
        fx1, changed1 = upsert_fixture_from_api_row(
            s, api_row=_api_fixture_row(), competition=comp, season=season
        )
        assert changed1 is True
        fx2, changed2 = upsert_fixture_from_api_row(
            s, api_row=_api_fixture_row(), competition=comp, season=season
        )
        assert fx1.id == fx2.id
        assert changed2 is False, "second run with identical payload should not mark fixture as changed"


def test_fixture_upsert_detects_score_change(session_factory):
    with session_factory() as s:
        comp = upsert_competition(s, SPEC)
        season = upsert_season(s, competition=comp, year=2025)
        s.flush()
        fx, _ = upsert_fixture_from_api_row(s, api_row=_api_fixture_row(home_goals=0, away_goals=0, status="HT"),
                                            competition=comp, season=season)
        assert fx.home_goals == 0
        fx2, changed = upsert_fixture_from_api_row(
            s, api_row=_api_fixture_row(home_goals=2, away_goals=1, status="FT"),
            competition=comp, season=season,
        )
        assert fx2.id == fx.id
        assert changed is True
        assert fx2.home_goals == 2
        assert fx2.status == "FT"


def test_team_and_player_stats_upsert(session_factory):
    with session_factory() as s:
        comp = upsert_competition(s, SPEC)
        season = upsert_season(s, competition=comp, year=2025)
        s.flush()
        fx, _ = upsert_fixture_from_api_row(s, api_row=_api_fixture_row(),
                                            competition=comp, season=season)
        rows1 = upsert_fixture_team_stats(s, fixture=fx, stats_response=_stats_response())
        assert rows1 == 2
        # Re-running unchanged stats should be a no-op (0 rows changed)
        rows2 = upsert_fixture_team_stats(s, fixture=fx, stats_response=_stats_response())
        assert rows2 == 0

        p_rows1 = upsert_fixture_player_stats(s, fixture=fx, players_response=_players_response())
        assert p_rows1 == 1
        p_rows2 = upsert_fixture_player_stats(s, fixture=fx, players_response=_players_response())
        assert p_rows2 == 0


def test_watermark_round_trip(session_factory):
    with session_factory() as s:
        upsert_watermark(
            s,
            scope="fixtures:EPL:2025",
            last_successful_at=datetime(2025, 8, 11, 12, 0, tzinfo=timezone.utc),
            last_cursor="2025-08-10",
            meta={"mode": "bootstrap"},
        )
    with session_factory() as s:
        wm = get_watermark(s, "fixtures:EPL:2025")
        assert wm is not None
        assert wm.last_cursor == "2025-08-10"
        assert (wm.meta or {}).get("mode") == "bootstrap"
