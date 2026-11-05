"""Pick notes read each team's finished matches in the fixture's competition and season, before its kickoff (D75)."""
from datetime import datetime, timezone


def _at(day, hour=15):
    return datetime(2026, 11 if day < 20 else 10, day, hour, tzinfo=timezone.utc)


def _seed(session_factory):
    from data_platform.models import Competition, Fixture, FixtureTeamStats, Season, Team
    with session_factory() as session:
        a, b, c, d = (Team(api_football_id=n, name=name) for n, name in ((1, "A"), (2, "B"), (3, "C"), (4, "D")))
        session.add_all([a, b, c, d])
        league = Competition(code="EPL", name="Premier League", api_football_id=39)
        cup = Competition(code="UCL", name="Champions League", api_football_id=2)
        session.add_all([league, cup]); session.flush()
        season, cup_season = (Season(competition_id=c.id, year=2026, label="2026/27") for c in (league, cup))
        session.add_all([season, cup_season]); session.flush()

        def fixture(api_id, home, away, kickoff, status="FT", goals=(None, None), competition=league, at=season):
            row = Fixture(api_football_id=api_id, competition_id=competition.id, season_id=at.id, home_team_id=home.id,
                          away_team_id=away.id, kickoff_utc=kickoff, status=status, home_goals=goals[0],
                          away_goals=goals[1])
            session.add(row); session.flush()
            return row

        fixture(1000, a, b, _at(8), status="NS")                         # tonight
        played = fixture(1001, a, c, _at(1), goals=(2, 1))
        fixture(1002, d, a, _at(25), goals=(0, 0))                        # no team stats
        fixture(1003, a, d, _at(18), status="PST")                        # postponed
        fixture(1004, a, c, _at(9), goals=(99, 99))                        # after tonight's kickoff
        fixture(1005, a, d, _at(4), goals=(99, 99), competition=cup, at=cup_season)  # another competition
        other = fixture(1006, b, c, _at(31), goals=(1, 3))
        session.add_all([
            FixtureTeamStats(fixture_id=played.id, team_id=a.id, opponent_team_id=c.id, is_home=True, corners=7),
            FixtureTeamStats(fixture_id=played.id, team_id=c.id, opponent_team_id=a.id, is_home=False, corners=3),
            FixtureTeamStats(fixture_id=other.id, team_id=b.id, opponent_team_id=c.id, is_home=True, corners=4),
            FixtureTeamStats(fixture_id=other.id, team_id=c.id, opponent_team_id=b.id, is_home=False, corners=6),
        ])


def test_each_teams_finished_matches_in_this_competition_and_season_before_kickoff_newest_first(
    settings, engine, session_factory,
):
    from data_platform.repositories.team_form import TeamFormRepository

    _seed(session_factory)
    form = TeamFormRepository(session_factory).for_fixture(1000)
    assert form == {
        "competition": "Premier League",
        "home": [{"goals_for": 2, "goals_against": 1, "corners_for": 7, "corners_against": 3},
                 {"goals_for": 0, "goals_against": 0, "corners_for": None, "corners_against": None}],
        "away": [{"goals_for": 1, "goals_against": 3, "corners_for": 4, "corners_against": 6}],
    }


def test_an_unknown_fixture_has_no_form(settings, engine, session_factory):
    from data_platform.repositories.team_form import TeamFormRepository

    assert TeamFormRepository(session_factory).for_fixture(424242) is None
