"""Re-mark who started from the stored substitution events (tracker D136).

Since August 2026 API-Football's ``games.substitute`` says False for most players who came on, so "starts" counted
appearances. The refresh now marks substitutes from each match's events (``upserts.mark_substitutes``); this fixes
the rows stored before, from the events already in the database. It makes no provider call.
"""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any, Dict

from sqlalchemy import select

from ..models import Fixture, FixtureEvent
from .upserts import mark_substitutes


def remark_substitutes(session_factory, *, since: date, dry_run: bool = False) -> Dict[str, Any]:
    """Mark substitutes from the stored events of every fixture kicking off from ``since``. A dry run reports what
    would change and rolls it back."""
    from ..services.refresh_coordination import data_access

    start = datetime.combine(since, time.min, tzinfo=timezone.utc)
    result = {"fixtures_with_events": 0, "fixtures_changed": 0, "rows_changed": 0, "dry_run": dry_run}
    with data_access() if dry_run else data_access(writer=True, wait_seconds=180):
        with session_factory() as session:
            ids = list(session.scalars(select(Fixture.id).where(
                Fixture.kickoff_utc >= start, Fixture.id.in_(select(FixtureEvent.fixture_id).distinct()))
                .order_by(Fixture.id)))
        result["fixtures_with_events"] = len(ids)
        for fixture_id in ids:
            with session_factory() as session:
                changed = mark_substitutes(session, fixture=session.get(Fixture, fixture_id))
                session.rollback() if dry_run else session.commit()
            result["fixtures_changed"] += bool(changed)
            result["rows_changed"] += changed
    return result
