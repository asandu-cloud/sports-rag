"""Fill the match events of finished fixtures fetched before the refresh collected events (cards plan task 7).

One ``/fixtures/events`` request per fixture, oldest first. The run holds the data gate throughout (a reader
gate for a dry run), so the refresh and the Match Read worker see one change, not one per fixture. Provider
calls never run inside a database transaction and each fixture's events commit on their own, so a stopped
run resumes where it left off.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from typing import Any, Dict, Optional

from sqlalchemy import exists, inspect, select

from ..models import Fixture, FixtureEvent
from .upserts import upsert_fixture_events

logger = logging.getLogger(__name__)

FINISHED = ("FT", "AET", "PEN")


def _candidates(session, since: date):
    start = datetime.combine(since, time.min, tzinfo=timezone.utc)
    has_events = exists().where(FixtureEvent.fixture_id == Fixture.id)
    return session.execute(
        select(Fixture.id, Fixture.api_football_id)
        .where(Fixture.status.in_(FINISHED), Fixture.kickoff_utc >= start, ~has_events)
        .order_by(Fixture.kickoff_utc, Fixture.id)
    ).all()


def backfill_events(session_factory, client, *, since: date, limit: Optional[int] = None,
                    dry_run: bool = False) -> Dict[str, Any]:
    """Fetch and store events for finished fixtures since ``since`` that have none; ``limit`` caps the requests."""
    from ..services.refresh_coordination import data_access

    result = {"candidates": 0, "requests": 0, "fetched": 0, "rows": 0, "empty": 0, "errors": 0}
    with data_access() if dry_run else data_access(writer=True, wait_seconds=180):
        with session_factory() as session:
            if not inspect(session.connection()).has_table(FixtureEvent.__tablename__):
                raise RuntimeError(f"{FixtureEvent.__tablename__} is missing: run init-db first")
            todo = _candidates(session, since)
        result["candidates"] = len(todo)
        todo = todo[:limit]
        result["requests"] = len(todo)
        if dry_run:
            return result
        for fixture_id, api_id in todo:
            try:
                events = client.fixture_events(api_id)
            except Exception as exc:  # the run carries on; the fixture stays a candidate
                result["errors"] += 1
                logger.warning("events failed for fixture %s: %s", api_id, exc)
                continue
            result["fetched"] += 1
            with session_factory() as session:
                rows = upsert_fixture_events(session, fixture=session.get(Fixture, fixture_id), events_response=events)
                session.commit()
            result["rows"] += rows
            result["empty"] += not rows
    return result
