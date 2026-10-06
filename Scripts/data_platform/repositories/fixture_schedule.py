"""Read the stored schedule without loading outcomes, prices or predictions."""

from datetime import date, datetime, timedelta, timezone
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import aliased

from ..db import session_scope
from ..models import Competition, Fixture, Team


class FixtureScheduleRepository:
    def __init__(self, session_factory=session_scope):
        self._factory = session_factory

    def list_for_matchday(self, *, target_date: date, leagues: Sequence[str]) -> list[dict]:
        start = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
        home, away = aliased(Team), aliased(Team)
        # Select only schedule metadata. In particular, never select the full
        # Fixture ORM row (which would also load goals and other outcome fields).
        statement = (
            select(
                Fixture.api_football_id, Competition.code, Fixture.kickoff_utc,
                Fixture.status, home.name, away.name, home.logo_url, away.logo_url,
            )
            .join(Competition, Competition.id == Fixture.competition_id)
            .join(home, home.id == Fixture.home_team_id)
            .join(away, away.id == Fixture.away_team_id)
            .where(
                Competition.code.in_(leagues),
                Fixture.kickoff_utc >= start,
                Fixture.kickoff_utc < start + timedelta(days=1),
            )
            .order_by(Fixture.kickoff_utc, Competition.code, Fixture.api_football_id)
        )
        with self._factory() as session:
            rows = session.execute(statement).all()
        return [
            {
                "fixture": {
                    "event_id": str(event_id), "league": league,
                    "home_team": home_name, "away_team": away_name,
                    "kickoff": (kickoff.replace(tzinfo=timezone.utc) if kickoff.tzinfo is None
                                else kickoff.astimezone(timezone.utc)).isoformat(),
                },
                "status": status,
                "visuals": {"home_team_logo": home_logo, "away_team_logo": away_logo},
            }
            for event_id, league, kickoff, status, home_name, away_name, home_logo, away_logo in rows
        ]
