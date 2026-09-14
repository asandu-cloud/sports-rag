"""Generic canonical feature builder for newly registered competitions."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace


def test_canonical_builder_creates_team_and_player_snapshots(session_factory):
    from data_platform.models import (
        Competition,
        Fixture,
        FixturePlayerStats,
        FixtureTeamStats,
        Player,
        Season,
        Team,
        TeamFeatureSnapshot,
        PlayerFeatureSnapshot,
    )
    from data_platform.sync.features_from_canonical import build_feature_snapshots_from_canonical

    with session_factory() as session:
        competition = Competition(
            code="Eredivisie", name="Eredivisie", api_football_id=88,
            competition_type="domestic_league",
        )
        season = Season(competition=competition, year=2025, label="2025/26")
        home = Team(api_football_id=88_001, name="Home")
        away = Team(api_football_id=88_002, name="Away")
        player = Player(api_football_id=88_101, name="Forward", position="F")
        session.add_all([competition, season, home, away, player])
        session.flush()
        fixture = Fixture(
            api_football_id=880_001,
            competition_id=competition.id,
            season_id=season.id,
            home_team_id=home.id,
            away_team_id=away.id,
            kickoff_utc=datetime(2026, 2, 1, 15, tzinfo=timezone.utc),
            status="FT",
            home_goals=2,
            away_goals=1,
        )
        session.add(fixture)
        session.flush()
        session.add_all([
            FixtureTeamStats(
                fixture_id=fixture.id, team_id=home.id, opponent_team_id=away.id,
                is_home=True, goals=2, shots_total=12, shots_on=6, corners=7,
                fouls_committed=9, yellow_cards=1, red_cards=0, possession=0.58,
                expected_goals=1.8, passes_total=500, passes_accurate=430,
            ),
            FixtureTeamStats(
                fixture_id=fixture.id, team_id=away.id, opponent_team_id=home.id,
                is_home=False, goals=1, shots_total=8, shots_on=3, corners=4,
                fouls_committed=12, yellow_cards=3, red_cards=0, possession=0.42,
                expected_goals=0.7, passes_total=330, passes_accurate=250,
            ),
            FixturePlayerStats(
                fixture_id=fixture.id, player_id=player.id, team_id=home.id,
                minutes=90, goals=1, assists=0, shots_total=4, shots_on=3,
                yellow_cards=0, red_cards=0,
            ),
        ])
        result = build_feature_snapshots_from_canonical(
            session, codes=["Eredivisie"], seasons=[2025], as_of_date=date(2026, 5, 31),
        )

    assert result["team_snapshots"] == 2
    assert result["player_snapshots"] == 1
    with session_factory() as session:
        home_snapshot = session.query(TeamFeatureSnapshot).filter_by(team_id=home.id).one()
        assert home_snapshot.matches_played == 1
        assert float(home_snapshot.goals_for_pm) == 2.0
        assert float(home_snapshot.corners_against_pm) == 4.0
        player_snapshot = session.query(PlayerFeatureSnapshot).filter_by(player_id=player.id).one()
        assert player_snapshot.appearances_as_starter == 1
        assert float(player_snapshot.goals_per_90) == 1.0

        from data_platform.kb.doc_builders import build_docs_for_entity
        docs = build_docs_for_entity("team", f"Eredivisie:{home.api_football_id}", session=session)
        fixture_doc = next(doc for doc in docs if doc["doc_type"] == "team_fixture")
        assert fixture_doc["metadata"]["final_score"] == "2-1"
        assert fixture_doc["metadata"]["xg_for"] == 1.8
        profile_doc = next(doc for doc in docs if doc["doc_type"] == "team_profile")
        assert float(profile_doc["metadata"]["corners_against_pm"]) == 4.0


def test_fixture_document_ids_distinguish_repeat_matchups():
    """A league fixture and a play-off can have identical team-name text."""
    from data_platform.kb.doc_builders import _team_fixture_doc

    team = SimpleNamespace(name="Middlesbrough", api_football_id=41)
    opponent = SimpleNamespace(name="Southampton", api_football_id=46)
    season = SimpleNamespace(label="2025/26")
    competition = SimpleNamespace(code="Championship")
    stats = SimpleNamespace(
        is_home=True,
        goals=2,
        expected_goals=1.2,
        possession=0.51,
        shots_total=10,
        shots_on=4,
        corners=5,
        fouls_committed=8,
        yellow_cards=2,
        red_cards=0,
    )
    january = SimpleNamespace(
        api_football_id=1386855,
        kickoff_utc=datetime(2026, 1, 4, 15, tzinfo=timezone.utc),
        home_goals=4,
        away_goals=0,
    )
    may = SimpleNamespace(
        api_football_id=1543787,
        kickoff_utc=datetime(2026, 5, 9, 11, 30, tzinfo=timezone.utc),
        home_goals=0,
        away_goals=0,
    )

    january_doc = _team_fixture_doc(team, opponent, january, season, competition, stats)
    may_doc = _team_fixture_doc(team, opponent, may, season, competition, stats)

    assert january_doc["doc_id"] != may_doc["doc_id"]
    assert january_doc["metadata"]["fixture_api_id"] == 1386855
    assert may_doc["metadata"]["fixture_api_id"] == 1543787
