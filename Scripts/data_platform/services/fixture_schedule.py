"""Small schedule-only sync: no player/statistics calls, features or embeddings.

Persist verified coverage (including valid empty responses) separately from
the historical result watermark. Never consume a finished fixture's digest:
the joined refresh must still import its team/player statistics afterwards.
"""
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select

from ..db import session_scope
from ..models import Fixture, SyncRun
from ..sync.apifootball import ApiFootballClient, iter_competitions
from ..sync.upserts import upsert_competition, upsert_season, upsert_fixture_from_api_row, upsert_watermark, get_watermark

SCHEDULE_LOOKAHEAD_DAYS = 35
FINISHED = {"FT", "AET", "PEN", "AWD", "WO"}


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)


def sync_fixture_schedule(*, codes, years=None, days_ahead=SCHEDULE_LOOKAHEAD_DAYS,
                          if_stale_hours=0, now=None, client=None, session_factory=session_scope):
    if not 4 <= days_ahead <= 90:
        raise ValueError("days_ahead must be between 4 and 90")
    if not 0 <= if_stale_hours <= 24:
        raise ValueError("if_stale_hours must be between 0 and 24")
    now = _utc(now or datetime.now(timezone.utc))
    start = now.date()
    end = start + timedelta(days=days_ahead)
    # Handle the July season boundary without hardcoding 2026 into a service.
    campaign = lambda day: day.year if day.month >= 7 else day.year - 1
    seasons = list(years) if years else sorted({campaign(start), campaign(end)})
    specs = list(iter_competitions(codes))
    report = {"from": start.isoformat(), "to": end.isoformat(), "provider_calls": 0,
              "fixtures_seen": 0, "fixtures_changed": 0, "competitions": [], "errors": []}
    with session_factory() as session:
        run = SyncRun(run_kind="fixture_schedule", scope=",".join(spec.code for spec in specs),
                      started_at=now, status="running", stats={})
        session.add(run)
        session.flush()
        run_id = run.id
    completed = False
    try:
        for spec in specs:
            for season_year in seasons:
                scope = f"schedule:{spec.code}:{season_year}"
                with session_factory() as session:
                    watermark = get_watermark(session, scope)
                    previous = watermark.last_successful_at if watermark else None
                    previous_end = watermark.last_cursor if watermark else None
                # A recent check is reusable only while it still covers the
                # next four days (calendar gaps are legitimate).
                if (if_stale_hours and previous and previous_end
                        and now - _utc(previous) < timedelta(hours=if_stale_hours)
                        and date.fromisoformat(previous_end) >= start + timedelta(days=4)):
                    report["competitions"].append({"league": spec.code, "season": season_year, "status": "fresh"})
                    continue
                try:
                    client = client or ApiFootballClient()
                    report["provider_calls"] += 1
                    rows = client.fixtures(league=spec.api_football_id, season=season_year,
                                           from_date=start.isoformat(), to_date=end.isoformat())
                    report["fixtures_seen"] += len(rows)
                    next_kickoffs = []
                    changed_count = 0
                    with session_factory() as session:
                        competition = upsert_competition(session, spec)
                        season = upsert_season(session, competition=competition, year=season_year)
                        session.flush()
                        for row in rows:
                            if (row.get("league") or {}).get("id") != spec.api_football_id:
                                raise ValueError("Provider fixture does not belong to the requested competition")
                            if (row.get("league") or {}).get("season") != season_year:
                                raise ValueError("Provider fixture does not belong to the requested season")
                            fixture = row.get("fixture") or {}
                            kickoff = _utc(fixture["date"])
                            status = (fixture.get("status") or {}).get("short")
                            # No historical/result-detail watermark is touched.
                            if kickoff <= now or status in FINISHED:
                                continue
                            existing = session.scalar(select(Fixture).where(Fixture.api_football_id == fixture["id"]))
                            if existing is not None and existing.status in FINISHED:
                                continue
                            _, changed = upsert_fixture_from_api_row(session, api_row=row,
                                competition=competition, season=season)
                            session.flush()
                            changed_count += int(changed)
                            if status in {"NS", "TBD"}:
                                next_kickoffs.append(kickoff.isoformat())
                        coverage = {"from": start.isoformat(), "to": end.isoformat(),
                                    "fixtures_seen": len(rows), "upcoming_count": len(next_kickoffs),
                                    "next_kickoff": min(next_kickoffs) if next_kickoffs else None}
                        upsert_watermark(session, scope=scope, last_successful_at=now,
                                         last_cursor=end.isoformat(), meta=coverage)
                    report["fixtures_changed"] += changed_count
                    report["competitions"].append({"league": spec.code, "season": season_year,
                                                   "status": "checked", **coverage})
                except Exception as exc:
                    report["errors"].append({"league": spec.code, "season": season_year,
                                              "error": f"{type(exc).__name__}: {exc}"})
        completed = True
        return {"run_id": run_id, **report}
    finally:
        with session_factory() as session:
            run = session.get(SyncRun, run_id)
            run.finished_at = datetime.now(timezone.utc)
            run.status = ("partial" if report["errors"] else "completed") if completed else "failed"
            run.stats = report
            run.error_text = str(report["errors"]) if report["errors"] else (None if completed else "Schedule refresh interrupted")
