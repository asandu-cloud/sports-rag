"""Read-side helpers on top of fixtures / fixture_team_stats / fixture_player_stats.

Used by higher-level services that need structured fixture access without
touching the old ``Output/*`` JSON files. Intentionally thin — business
logic (blending, projections) stays in the engine modules.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import desc, select
from sqlalchemy.orm import Session, joinedload

from ..db import session_scope
from ..models import (
    Competition,
    Fixture,
    FixturePlayerStats,
    FixtureTeamStats,
    Player,
    Season,
    Team,
)

logger = logging.getLogger(__name__)


class FixtureRepository:
    def __init__(self, session_factory=session_scope):
        self._factory = session_factory

    def list_by_date(self, *, league: Optional[str] = None, target_date: Optional[date] = None) -> List[Dict[str, Any]]:
        with self._factory() as session:
            stmt = select(Fixture).join(Competition, Competition.id == Fixture.competition_id)
            if league is not None:
                stmt = stmt.where(Competition.code == league)
            if target_date is not None:
                # Python-side date comparison via the kickoff column
                stmt = stmt.where(
                    Fixture.kickoff_utc >= datetime.combine(target_date, datetime.min.time()),
                    Fixture.kickoff_utc < datetime.combine(
                        date.fromordinal(target_date.toordinal() + 1),
                        datetime.min.time(),
                    ),
                )
            stmt = stmt.order_by(Fixture.kickoff_utc.asc())
            rows = session.scalars(stmt).all()
            return [_fixture_to_dict(r, session) for r in rows]

    def get_by_api_id(self, api_football_id: int) -> Optional[Dict[str, Any]]:
        with self._factory() as session:
            row = session.scalar(select(Fixture).where(Fixture.api_football_id == api_football_id))
            if row is None:
                return None
            return _fixture_to_dict(row, session)

    def team_stats_for_fixture(self, fixture_id: int) -> List[Dict[str, Any]]:
        with self._factory() as session:
            rows = session.scalars(
                select(FixtureTeamStats).where(FixtureTeamStats.fixture_id == fixture_id)
            ).all()
            return [_team_stats_to_dict(r) for r in rows]

    def recent_team_stats(
        self,
        *,
        team_api_id: int,
        limit: int = 6,
        before: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        with self._factory() as session:
            team = session.scalar(select(Team).where(Team.api_football_id == team_api_id))
            if team is None:
                return []
            stmt = (
                select(FixtureTeamStats, Fixture)
                .join(Fixture, Fixture.id == FixtureTeamStats.fixture_id)
                .where(FixtureTeamStats.team_id == team.id)
            )
            if before is not None:
                stmt = stmt.where(Fixture.kickoff_utc < before)
            stmt = stmt.order_by(desc(Fixture.kickoff_utc)).limit(limit)
            rows = session.execute(stmt).all()
            return [
                {
                    **_team_stats_to_dict(stats),
                    "fixture": _fixture_to_dict(fixture, session, with_teams=False),
                }
                for stats, fixture in rows
            ]


def _fixture_to_dict(fixture: Fixture, session: Session, *, with_teams: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": fixture.id,
        "api_football_id": fixture.api_football_id,
        "kickoff_utc": fixture.kickoff_utc.isoformat() if fixture.kickoff_utc else None,
        "status": fixture.status,
        "elapsed": fixture.elapsed,
        "home_goals": fixture.home_goals,
        "away_goals": fixture.away_goals,
        "venue": fixture.venue,
        "round": fixture.round,
        "referee": fixture.referee,
    }
    if with_teams:
        home = session.get(Team, fixture.home_team_id)
        away = session.get(Team, fixture.away_team_id)
        out["home_team"] = home.name if home else None
        out["away_team"] = away.name if away else None
        comp = session.get(Competition, fixture.competition_id)
        season = session.get(Season, fixture.season_id)
        out["league"] = comp.code if comp else None
        out["season"] = season.label if season else None
    return out


def _team_stats_to_dict(row: FixtureTeamStats) -> Dict[str, Any]:
    return {
        "fixture_id": row.fixture_id,
        "team_id": row.team_id,
        "opponent_team_id": row.opponent_team_id,
        "is_home": bool(row.is_home),
        "goals": row.goals,
        "shots_total": row.shots_total,
        "shots_on": row.shots_on,
        "corners": row.corners,
        "fouls_committed": row.fouls_committed,
        "yellow_cards": row.yellow_cards,
        "red_cards": row.red_cards,
        "possession": float(row.possession) if row.possession is not None else None,
        "expected_goals": float(row.expected_goals) if row.expected_goals is not None else None,
        "pass_accuracy": float(row.pass_accuracy) if row.pass_accuracy is not None else None,
    }
