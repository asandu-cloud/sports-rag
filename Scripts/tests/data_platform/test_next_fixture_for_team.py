"""A team's next fixture for the Research Assistant (tracker D95): the earliest unplayed one in a league Spix reads."""
from datetime import datetime, timezone

from sqlalchemy import event


def _seed(session_factory):
    from data_platform.models import Competition, Fixture, Season, Team
    with session_factory() as session:
        names = ("Brentford", "Brighton", "Everton", "Manchester United", "Arsenal", "Lonely")
        teams = {name: Team(api_football_id=500 + i, name=name) for i, name in enumerate(names)}
        session.add_all(teams.values())
        seasons = {}
        for code, api_id in (("EPL", 39), ("FACup", 45)):
            competition = Competition(code=code, name=code, api_football_id=api_id)
            session.add(competition); session.flush()
            seasons[code] = (competition, Season(competition_id=competition.id, year=2026, label="2026/27"))
            session.add(seasons[code][1]); session.flush()
        for fid, code, home, away, kickoff, status in (
                (1, "EPL", "Brighton", "Brentford", datetime(2026, 11, 8, 14), "FT"),       # played
                (2, "FACup", "Brentford", "Arsenal", datetime(2026, 11, 20, 19), "NS"),   # not a league Spix reads
                (3, "EPL", "Brentford", "Arsenal", datetime(2026, 11, 21, 15), "PST"),    # postponed
                (4, "EPL", "Manchester United", "Brentford", datetime(2026, 11, 28, 15), "NS"),
                (5, "EPL", "Brentford", "Everton", datetime(2026, 11, 23, 20), "NS")):
            competition, season = seasons[code]
            session.add(Fixture(api_football_id=1557000 + fid, competition_id=competition.id, season_id=season.id,
                                home_team_id=teams[home].id, away_team_id=teams[away].id,
                                kickoff_utc=kickoff.replace(tzinfo=timezone.utc), status=status))
        return {name: team.id for name, team in teams.items()}


def test_the_next_fixture_is_the_earliest_unplayed_one_in_a_league_spix_reads(settings, engine, session_factory):
    from data_platform.repositories.fixture_schedule import FixtureScheduleRepository
    ids = _seed(session_factory)
    repository, now = FixtureScheduleRepository(session_factory), datetime(2026, 11, 10, 12, tzinfo=timezone.utc)
    statements = []
    with session_factory() as session:
        bind = session.get_bind()
    capture = lambda conn, cursor, statement, *args: statements.append(statement)  # noqa: E731
    event.listen(bind, "before_cursor_execute", capture)
    try:
        brentford = repository.next_for_team(ids["Brentford"], leagues=("EPL",), now=now)
    finally:
        event.remove(bind, "before_cursor_execute", capture)
    assert brentford == {"event_id": "1557005", "league": "EPL", "home_team": "Brentford", "away_team": "Everton",
                         "kickoff": "2026-11-23T20:00:00+00:00", "venue": "home", "opponent": "Everton"}
    assert all(s.lstrip().upper().startswith("SELECT") for s in statements)
    everton = repository.next_for_team(ids["Everton"], leagues=("EPL",), now=now)
    assert (everton["venue"], everton["opponent"]) == ("away", "Brentford")
    assert repository.next_for_team(ids["Lonely"], leagues=("EPL",), now=now) is None
    later = datetime(2026, 11, 24, 12, tzinfo=timezone.utc)
    assert repository.next_for_team(ids["Brentford"], leagues=("EPL",), now=later)["event_id"] == "1557004"
