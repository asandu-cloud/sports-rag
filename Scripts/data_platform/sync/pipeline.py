"""Sync pipeline orchestration.

Two modes share the same primitives:

* ``bootstrap_season(code, year)`` — full backfill of a competition/season.
  Safe to re-run; uses upserts everywhere and skips fixture-detail calls
  whose payload digest already matches.

* ``incremental_refresh(codes)`` — uses ``sync_watermarks`` to limit the
  ``/fixtures`` window and then re-fetches details only for fixtures
  whose fixture payload changed (score finalised, ref allocated, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Fixture, SyncRun
from .apifootball import ApiFootballClient, COMPETITIONS, CompetitionSpec, iter_competitions
from .upserts import (
    get_watermark,
    upsert_competition,
    upsert_fixture_from_api_row,
    upsert_fixture_player_stats,
    upsert_fixture_team_stats,
    upsert_season,
    upsert_watermark,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sync run helpers
# ---------------------------------------------------------------------------


def _start_run(session: Session, *, kind: str, scope: str) -> SyncRun:
    run = SyncRun(
        run_kind=kind,
        scope=scope,
        started_at=datetime.now(timezone.utc),
        status="running",
        stats={},
    )
    session.add(run)
    session.flush()
    return run


def _finish_run(session: Session, run: SyncRun, *, status: str, stats: Dict[str, Any], error: Optional[str] = None) -> None:
    run.finished_at = datetime.now(timezone.utc)
    run.status = status
    run.stats = stats
    run.error_text = error


@dataclass
class SyncStats:
    fixtures_seen: int = 0
    fixtures_changed: int = 0
    team_stats_rows: int = 0
    player_stats_rows: int = 0
    fixture_detail_calls: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fixtures_seen": self.fixtures_seen,
            "fixtures_changed": self.fixtures_changed,
            "team_stats_rows": self.team_stats_rows,
            "player_stats_rows": self.player_stats_rows,
            "fixture_detail_calls": self.fixture_detail_calls,
            "error_count": len(self.errors),
            "errors": self.errors[:20],
        }


# ---------------------------------------------------------------------------
# Fixture detail sync (shared between bootstrap + incremental)
# ---------------------------------------------------------------------------


def sync_fixture_details(
    session: Session,
    *,
    client: ApiFootballClient,
    fixture: Fixture,
    stats: SyncStats,
    archiver=None,
    sync_run_id: Optional[int] = None,
    fetch_players: bool = True,
) -> None:
    """Fetch + upsert team + player stats for a single fixture."""
    try:
        team_payload = client.fixture_statistics(fixture.api_football_id)
        stats.fixture_detail_calls += 1
        if archiver is not None:
            archiver.archive_json(
                provider="api_football",
                endpoint="/fixtures/statistics",
                params={"fixture": fixture.api_football_id},
                payload=team_payload,
                sync_run_id=sync_run_id,
            )
        stats.team_stats_rows += upsert_fixture_team_stats(
            session, fixture=fixture, stats_response=team_payload
        )
    except Exception as exc:  # pragma: no cover - network path
        stats.errors.append({"fixture_id": fixture.api_football_id, "stage": "team_stats", "error": str(exc)})
        logger.warning("team_stats failed for fixture %s: %s", fixture.api_football_id, exc)

    if not fetch_players:
        return

    try:
        player_payload = client.fixture_players(fixture.api_football_id)
        stats.fixture_detail_calls += 1
        if archiver is not None:
            archiver.archive_json(
                provider="api_football",
                endpoint="/fixtures/players",
                params={"fixture": fixture.api_football_id},
                payload=player_payload,
                sync_run_id=sync_run_id,
            )
        stats.player_stats_rows += upsert_fixture_player_stats(
            session, fixture=fixture, players_response=player_payload
        )
    except Exception as exc:  # pragma: no cover - network path
        stats.errors.append({"fixture_id": fixture.api_football_id, "stage": "player_stats", "error": str(exc)})
        logger.warning("player_stats failed for fixture %s: %s", fixture.api_football_id, exc)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_season(
    session: Session,
    *,
    client: ApiFootballClient,
    code: str,
    year: int,
    fetch_players: bool = True,
    archiver=None,
    finished_only: bool = True,
) -> Dict[str, Any]:
    """Backfill one competition × season end-to-end.

    Safe to re-run: every write is an upsert and fixture detail calls are
    skipped when the fixture-row digest hasn't changed since the last
    fetch.
    """
    spec = COMPETITIONS.get(code)
    if spec is None:
        raise KeyError(f"Unknown competition code: {code}")
    scope = f"bootstrap:{code}:{year}"
    run = _start_run(session, kind="bootstrap", scope=scope)
    stats = SyncStats()

    try:
        competition = upsert_competition(session, spec)
        season = upsert_season(session, competition=competition, year=year)
        session.flush()

        fixtures_payload = client.fixtures(
            league=spec.api_football_id,
            season=year,
            status="FT" if finished_only else None,
        )
        if archiver is not None:
            archiver.archive_json(
                provider="api_football",
                endpoint="/fixtures",
                params={"league": spec.api_football_id, "season": year, "status": "FT" if finished_only else None},
                payload=fixtures_payload,
                sync_run_id=run.id,
            )
        stats.fixtures_seen = len(fixtures_payload)

        for api_row in fixtures_payload:
            try:
                fixture, changed = upsert_fixture_from_api_row(
                    session, api_row=api_row, competition=competition, season=season
                )
            except Exception as exc:  # pragma: no cover - malformed row
                stats.errors.append({"stage": "fixture_upsert", "error": str(exc)})
                continue
            if changed:
                stats.fixtures_changed += 1
            # Fetch details for every fixture on bootstrap runs. The
            # upserts themselves detect unchanged payloads and no-op.
            if fixture.status in {"FT", "AET", "PEN"} or not finished_only:
                sync_fixture_details(
                    session,
                    client=client,
                    fixture=fixture,
                    stats=stats,
                    archiver=archiver,
                    sync_run_id=run.id,
                    fetch_players=fetch_players,
                )

        upsert_watermark(
            session,
            scope=f"fixtures:{code}:{year}",
            last_successful_at=datetime.now(timezone.utc),
            last_cursor=date.today().isoformat(),
            meta={"mode": "bootstrap"},
        )
        _finish_run(session, run, status="completed", stats=stats.as_dict())
        return {"run_id": run.id, **stats.as_dict()}
    except Exception as exc:
        _finish_run(session, run, status="failed", stats=stats.as_dict(), error=str(exc))
        raise


# ---------------------------------------------------------------------------
# Incremental refresh
# ---------------------------------------------------------------------------


def _watermark_from_date(session: Session, *, code: str, year: int, lookback_days: int) -> str:
    scope = f"fixtures:{code}:{year}"
    row = get_watermark(session, scope)
    if row and row.last_cursor:
        try:
            anchor = date.fromisoformat(row.last_cursor)
        except ValueError:
            anchor = date.today() - timedelta(days=lookback_days)
    else:
        anchor = date.today() - timedelta(days=lookback_days)
    # Small safety overlap so we re-fetch fixtures completed right at the boundary
    return (anchor - timedelta(days=1)).isoformat()


def incremental_refresh(
    session: Session,
    *,
    client: ApiFootballClient,
    codes: Optional[Iterable[str]] = None,
    years: Optional[Iterable[int]] = None,
    lookback_days: int = 7,
    fetch_players: bool = True,
    archiver=None,
) -> Dict[str, Any]:
    """Refresh one-or-more competitions.

    For each (competition, year) we pull fixtures in ``[watermark - 1d, today + 7d]``
    and re-fetch details for any fixture whose row digest changed. Fixtures
    with unchanged digests are skipped cheaply.
    """
    specs: List[CompetitionSpec] = list(iter_competitions(codes))
    today = date.today()
    year_list = list(years) if years else [today.year if today.month >= 7 else today.year - 1]

    scope_label = "ALL" if codes is None else ",".join(codes)
    run = _start_run(session, kind="incremental", scope=f"{scope_label}:{','.join(map(str, year_list))}")
    stats = SyncStats()

    try:
        for spec in specs:
            competition = upsert_competition(session, spec)
            for year in year_list:
                season = upsert_season(session, competition=competition, year=year)
                session.flush()
                from_date = _watermark_from_date(session, code=spec.code, year=year, lookback_days=lookback_days)
                to_date = (today + timedelta(days=7)).isoformat()
                logger.info("Incremental %s:%s window %s..%s", spec.code, year, from_date, to_date)

                fixtures_payload = client.fixtures(
                    league=spec.api_football_id,
                    season=year,
                    from_date=from_date,
                    to_date=to_date,
                )
                if archiver is not None:
                    archiver.archive_json(
                        provider="api_football",
                        endpoint="/fixtures",
                        params={"league": spec.api_football_id, "season": year, "from": from_date, "to": to_date},
                        payload=fixtures_payload,
                        sync_run_id=run.id,
                    )
                stats.fixtures_seen += len(fixtures_payload)

                for api_row in fixtures_payload:
                    try:
                        fixture, changed = upsert_fixture_from_api_row(
                            session, api_row=api_row, competition=competition, season=season
                        )
                    except Exception as exc:  # pragma: no cover - malformed row
                        stats.errors.append({"stage": "fixture_upsert", "error": str(exc)})
                        continue
                    if not changed:
                        continue
                    stats.fixtures_changed += 1
                    if fixture.status in {"FT", "AET", "PEN"}:
                        sync_fixture_details(
                            session,
                            client=client,
                            fixture=fixture,
                            stats=stats,
                            archiver=archiver,
                            sync_run_id=run.id,
                            fetch_players=fetch_players,
                        )

                upsert_watermark(
                    session,
                    scope=f"fixtures:{spec.code}:{year}",
                    last_successful_at=datetime.now(timezone.utc),
                    last_cursor=today.isoformat(),
                    meta={"mode": "incremental", "lookback_days": lookback_days},
                )

        _finish_run(session, run, status="completed", stats=stats.as_dict())
        return {"run_id": run.id, **stats.as_dict()}
    except Exception as exc:
        _finish_run(session, run, status="failed", stats=stats.as_dict(), error=str(exc))
        raise
