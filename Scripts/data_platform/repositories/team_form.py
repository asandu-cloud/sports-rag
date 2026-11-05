"""Each team's finished matches before a fixture, for the pick notes' recent form (D75). Read only."""

from typing import Optional

from sqlalchemy import or_, select

from ..db import retry_database_busy, session_scope
from ..models import Competition, Fixture, FixtureTeamStats


class TeamFormRepository:
    def __init__(self, session_factory=session_scope):
        self._factory = session_factory

    @retry_database_busy
    def for_fixture(self, api_football_id: int) -> Optional[dict]:
        """``{competition, home, away}``: each side's full-time matches in the fixture's competition and season that
        kicked off before it, newest first, as ``goals_for``, ``goals_against``, ``corners_for``,
        ``corners_against`` (corners None without team statistics). None for an unknown fixture."""
        with self._factory() as session:
            fixture = session.execute(
                select(Fixture).where(Fixture.api_football_id == int(api_football_id))).scalar_one_or_none()
            if fixture is None or fixture.kickoff_utc is None:
                return None
            competition = session.get(Competition, fixture.competition_id)
            return {"competition": competition.name if competition else None,
                    "home": _history(session, fixture, fixture.home_team_id),
                    "away": _history(session, fixture, fixture.away_team_id)}


def _history(session, fixture, team_id) -> list:
    rows = session.execute(
        select(Fixture).where(
            Fixture.competition_id == fixture.competition_id, Fixture.season_id == fixture.season_id,
            Fixture.status == "FT", Fixture.kickoff_utc < fixture.kickoff_utc,
            or_(Fixture.home_team_id == team_id, Fixture.away_team_id == team_id),
        ).order_by(Fixture.kickoff_utc.desc())
    ).scalars().all()
    corners = {(stats.fixture_id, stats.team_id): stats.corners for stats in session.execute(
        select(FixtureTeamStats).where(FixtureTeamStats.fixture_id.in_([row.id for row in rows]))).scalars()}
    history = []
    for row in rows:
        at_home = row.home_team_id == team_id
        opponent = row.away_team_id if at_home else row.home_team_id
        history.append({"goals_for": row.home_goals if at_home else row.away_goals,
                        "goals_against": row.away_goals if at_home else row.home_goals,
                        "corners_for": corners.get((row.id, team_id)),
                        "corners_against": corners.get((row.id, opponent))})
    return history
