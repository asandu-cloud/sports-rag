"""Data quality checks.

Each check returns a list of :class:`QualityIssue` rows. The runner
aggregates them for the CLI/API. Checks are cheap SQL probes — safe to
run in the health endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from ..db import session_scope
from ..models import (
    Competition,
    Fixture,
    FixturePlayerStats,
    FixtureTeamStats,
    OddsSnapshot,
    Player,
    Team,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualityIssue:
    """One data-quality finding."""

    check: str
    severity: str                 # 'info' | 'warn' | 'error'
    summary: str
    count: int = 0
    details: Optional[Dict[str, Any]] = None


CheckFn = Callable[[Session], List[QualityIssue]]


@dataclass
class DataQualityCheck:
    name: str
    fn: CheckFn
    description: str = ""


# ---------------------------------------------------------------------------
# Concrete checks
# ---------------------------------------------------------------------------


def _duplicate_api_ids(session: Session, model, *, label: str) -> List[QualityIssue]:
    stmt = (
        select(model.api_football_id, func.count(model.id))
        .group_by(model.api_football_id)
        .having(func.count(model.id) > 1)
    )
    rows = session.execute(stmt).all()
    if not rows:
        return []
    examples = [{"api_football_id": api_id, "count": int(count)} for api_id, count in rows[:10]]
    return [QualityIssue(
        check=f"{label}_duplicates",
        severity="error",
        summary=f"{len(rows)} {label} rows share an api_football_id",
        count=len(rows),
        details={"examples": examples},
    )]


def check_duplicate_teams(session: Session) -> List[QualityIssue]:
    return _duplicate_api_ids(session, Team, label="teams")


def check_duplicate_players(session: Session) -> List[QualityIssue]:
    return _duplicate_api_ids(session, Player, label="players")


def check_duplicate_fixtures(session: Session) -> List[QualityIssue]:
    return _duplicate_api_ids(session, Fixture, label="fixtures")


def check_orphan_team_stats(session: Session) -> List[QualityIssue]:
    missing = session.scalar(
        select(func.count(FixtureTeamStats.id))
        .outerjoin(Fixture, Fixture.id == FixtureTeamStats.fixture_id)
        .where(Fixture.id.is_(None))
    )
    if not missing:
        return []
    return [QualityIssue(
        check="orphan_team_stats", severity="error",
        summary="fixture_team_stats rows point at missing fixtures",
        count=int(missing),
    )]


def check_orphan_player_stats(session: Session) -> List[QualityIssue]:
    missing = session.scalar(
        select(func.count(FixturePlayerStats.id))
        .outerjoin(Fixture, Fixture.id == FixturePlayerStats.fixture_id)
        .where(Fixture.id.is_(None))
    )
    if not missing:
        return []
    return [QualityIssue(
        check="orphan_player_stats", severity="error",
        summary="fixture_player_stats rows point at missing fixtures",
        count=int(missing),
    )]


def check_finished_fixtures_without_stats(session: Session, *, lookback_days: int = 14) -> List[QualityIssue]:
    """Finished fixtures inside the last ``lookback_days`` missing team stats."""
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=lookback_days)
    missing = session.scalars(
        select(Fixture)
        .outerjoin(FixtureTeamStats, FixtureTeamStats.fixture_id == Fixture.id)
        .where(
            Fixture.status.in_(("FT", "AET", "PEN")),
            Fixture.kickoff_utc >= since,
            Fixture.kickoff_utc < now - timedelta(hours=4),
            FixtureTeamStats.id.is_(None),
        )
    ).all()
    if not missing:
        return []
    examples = [{
        "fixture_api_id": f.api_football_id,
        "kickoff_utc": f.kickoff_utc.isoformat() if f.kickoff_utc else None,
        "status": f.status,
    } for f in missing[:10]]
    return [QualityIssue(
        check="finished_fixtures_without_stats",
        severity="warn",
        summary=f"{len(missing)} finished fixtures within {lookback_days}d have no team stats",
        count=len(missing),
        details={"examples": examples},
    )]


def check_missing_odds_coverage(session: Session, *, lookahead_days: int = 3) -> List[QualityIssue]:
    """Upcoming fixtures with no recent odds snapshot."""
    now = datetime.now(timezone.utc)
    upcoming = session.scalars(
        select(Fixture)
        .where(
            Fixture.status.in_(("NS", "TBD")),
            Fixture.kickoff_utc >= now,
            Fixture.kickoff_utc <= now + timedelta(days=lookahead_days),
        )
    ).all()
    missing: List[Dict[str, Any]] = []
    for fixture in upcoming:
        snap = session.scalar(
            select(OddsSnapshot)
            .where(OddsSnapshot.event_id == str(fixture.api_football_id))
            .order_by(OddsSnapshot.captured_at.desc())
            .limit(1)
        )
        if snap is None:
            missing.append({
                "fixture_api_id": fixture.api_football_id,
                "kickoff_utc": fixture.kickoff_utc.isoformat() if fixture.kickoff_utc else None,
            })
    if not missing:
        return []
    return [QualityIssue(
        check="missing_odds_coverage",
        severity="warn",
        summary=f"{len(missing)} upcoming fixtures in next {lookahead_days}d have no odds snapshot",
        count=len(missing),
        details={"examples": missing[:10]},
    )]


def check_fixtures_without_competition_rows(session: Session) -> List[QualityIssue]:
    missing = session.scalar(
        select(func.count(Fixture.id))
        .outerjoin(Competition, Competition.id == Fixture.competition_id)
        .where(Competition.id.is_(None))
    )
    if not missing:
        return []
    return [QualityIssue(
        check="fixtures_without_competition",
        severity="error",
        summary="fixtures point at missing competition rows",
        count=int(missing),
    )]


DEFAULT_CHECKS: List[DataQualityCheck] = [
    DataQualityCheck("duplicate_teams", check_duplicate_teams, "Teams with shared api_football_id"),
    DataQualityCheck("duplicate_players", check_duplicate_players, "Players with shared api_football_id"),
    DataQualityCheck("duplicate_fixtures", check_duplicate_fixtures, "Fixtures with shared api_football_id"),
    DataQualityCheck("orphan_team_stats", check_orphan_team_stats, "Team stats without a fixture row"),
    DataQualityCheck("orphan_player_stats", check_orphan_player_stats, "Player stats without a fixture row"),
    DataQualityCheck(
        "finished_fixtures_without_stats", check_finished_fixtures_without_stats,
        "Finished fixtures with no team stats",
    ),
    DataQualityCheck(
        "missing_odds_coverage", check_missing_odds_coverage,
        "Upcoming fixtures with no recent odds snapshot",
    ),
    DataQualityCheck(
        "fixtures_without_competition", check_fixtures_without_competition_rows,
        "Fixtures pointing at missing competition rows",
    ),
]


def run_data_quality_checks(*, checks: Optional[Iterable[DataQualityCheck]] = None) -> List[QualityIssue]:
    issues: List[QualityIssue] = []
    check_list = list(checks) if checks is not None else DEFAULT_CHECKS
    with session_scope() as session:
        for check in check_list:
            try:
                issues.extend(check.fn(session))
            except Exception as exc:
                logger.exception("Data-quality check %s errored", check.name)
                issues.append(QualityIssue(
                    check=check.name, severity="error",
                    summary=f"check crashed: {exc}", count=0,
                ))
    return issues
