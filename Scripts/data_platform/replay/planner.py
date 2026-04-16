"""Describe what to replay.

The planner is deliberately small: it translates CLI/API-level arguments
into a concrete list of (entity_type, entity_key) targets the executor
can run against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import session_scope
from ..models import Competition, Fixture


@dataclass(frozen=True)
class ReplayScope:
    """Input to the planner."""

    fixture_api_ids: Sequence[int] = ()
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    competition_codes: Sequence[str] = ()
    season_years: Sequence[int] = ()
    kb_entities: Sequence[Tuple[str, str]] = ()   # (entity_type, entity_key)
    include_team_stats: bool = True
    include_player_stats: bool = True
    include_features: bool = True
    include_kb: bool = True


@dataclass
class ReplayPlan:
    fixtures: List[int] = field(default_factory=list)           # api_football_ids
    competition_seasons: List[Tuple[str, int]] = field(default_factory=list)
    kb_entities: List[Tuple[str, str]] = field(default_factory=list)
    scope: Optional[ReplayScope] = None

    def summary(self) -> dict:
        return {
            "fixtures": len(self.fixtures),
            "competition_seasons": len(self.competition_seasons),
            "kb_entities": len(self.kb_entities),
        }


def build_plan(scope: ReplayScope) -> ReplayPlan:
    """Translate a scope into a concrete set of targets."""
    plan = ReplayPlan(scope=scope)

    with session_scope() as session:
        plan.fixtures.extend(int(x) for x in scope.fixture_api_ids)

        if scope.date_from or scope.date_to:
            stmt = select(Fixture.api_football_id).where(Fixture.api_football_id.isnot(None))
            if scope.date_from:
                stmt = stmt.where(Fixture.kickoff_utc >= datetime.combine(scope.date_from, datetime.min.time()))
            if scope.date_to:
                stmt = stmt.where(Fixture.kickoff_utc < datetime.combine(scope.date_to, datetime.max.time()))
            if scope.competition_codes:
                stmt = stmt.join(Competition, Competition.id == Fixture.competition_id) \
                           .where(Competition.code.in_(list(scope.competition_codes)))
            ids = session.scalars(stmt.order_by(Fixture.kickoff_utc.asc())).all()
            plan.fixtures.extend(int(x) for x in ids if x is not None)

        for code in scope.competition_codes:
            if scope.season_years:
                for year in scope.season_years:
                    plan.competition_seasons.append((code, int(year)))
            else:
                # Default: every season present in the DB for the code
                stmt = select(Fixture).join(Competition, Competition.id == Fixture.competition_id) \
                       .where(Competition.code == code)
                fixtures = session.scalars(stmt).all()
                years = sorted({
                    f.kickoff_utc.year if f.kickoff_utc else None
                    for f in fixtures if f.kickoff_utc
                })
                for y in years:
                    if y is not None:
                        plan.competition_seasons.append((code, int(y)))

        plan.kb_entities.extend(scope.kb_entities)

    # Dedup and keep ordering stable
    plan.fixtures = sorted(set(plan.fixtures))
    plan.competition_seasons = sorted(set(plan.competition_seasons))
    plan.kb_entities = sorted(set(plan.kb_entities))
    return plan
