"""Shim for legacy ``Scripts/rag_ingest/prediction_tracker.py``.

Routes log / query calls through ``PredictionService`` when the platform
flag is on. Also provides an idempotent one-off migration from
``Index/predictions.db#predictions`` into the canonical Postgres table.
"""

from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import select

from .. import config as _config
from ..db import session_scope
from ..models import Prediction
from ..services.predictions import PredictionService

logger = logging.getLogger(__name__)


def is_platform_enabled() -> bool:
    return _config.SETTINGS.platform_enabled


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    return PredictionService()


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def migrate_sqlite_predictions(
    sqlite_path: Optional[Path] = None,
    *,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Copy every ``predictions`` row from SQLite into Postgres.

    Dedup key: ``(prediction_date, home_team, away_team, league, market, pick, source)``
    — matches the uniqueness constraint we imposed on the canonical table.
    """
    sqlite_path = Path(sqlite_path) if sqlite_path else _config.SETTINGS.project_root / "Index" / "predictions.db"
    if not sqlite_path.exists():
        logger.info("No predictions.db at %s; nothing to migrate.", sqlite_path)
        return {"rows": 0, "skipped": 0}

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute("SELECT * FROM predictions")
        rows = [dict(r) for r in cur.fetchall()]
    except sqlite3.OperationalError:
        conn.close()
        return {"rows": 0, "skipped": 0, "error": "predictions table missing"}
    finally:
        conn.close()

    if dry_run:
        return {"rows": len(rows), "skipped": 0, "dry_run": True}

    counts = {"rows": 0, "skipped": 0}
    with session_scope() as session:
        for row in rows:
            prediction_date = _parse_date(row.get("prediction_date"))
            home_team = row.get("home_team") or ""
            away_team = row.get("away_team") or ""
            league = row.get("league") or ""
            market = row.get("market") or ""
            pick = row.get("pick") or ""
            source = row.get("source") or "standalone"
            if not all([prediction_date, home_team, away_team, league, market, pick]):
                counts["skipped"] += 1
                continue

            existing = session.scalar(
                select(Prediction).where(
                    Prediction.prediction_date == prediction_date,
                    Prediction.home_team == home_team,
                    Prediction.away_team == away_team,
                    Prediction.league == league,
                    Prediction.market == market,
                    Prediction.pick == pick,
                    Prediction.source == source,
                )
            )
            fields = {
                "fixture_api_id": _str_or_none(row.get("fixture_id")),
                "home_team": home_team,
                "away_team": away_team,
                "league": league,
                "kickoff": row.get("kickoff"),
                "market": market,
                "pick": pick,
                "side": row.get("side"),
                "line": row.get("line"),
                "odds": row.get("odds"),
                "bookmaker": row.get("bookmaker"),
                "model_prob": row.get("model_prob"),
                "implied_prob": row.get("implied_prob"),
                "value_edge": row.get("value_edge"),
                "confidence": row.get("confidence"),
                "projected_total": row.get("projected_total"),
                "source": source,
                "actual_result": row.get("actual_result"),
                "outcome": row.get("outcome"),
                "graded_at": _parse_dt(row.get("graded_at")),
                "season": row.get("season"),
                "prediction_date": prediction_date,
                "closing_odds": row.get("closing_odds"),
                "closing_implied_prob": row.get("closing_implied_prob"),
                "clv": row.get("clv"),
            }
            if existing is None:
                session.add(Prediction(**fields))
                # Flush so the next iteration's existence check sees this row —
                # legacy predictions.db carries rows that share the V3 unique
                # key (the SQLite schema enforced dedup in Python, not SQL).
                session.flush()
            else:
                for key, value in fields.items():
                    setattr(existing, key, value)
            counts["rows"] += 1
    return counts


def _parse_date(value: Any):
    from datetime import date as _date
    if not value:
        return None
    if isinstance(value, _date):
        return value
    text = str(value)[:10]
    try:
        return _date.fromisoformat(text)
    except ValueError:
        return None


def _parse_dt(value: Any):
    from datetime import datetime as _dt, timezone as _tz
    if not value:
        return None
    if isinstance(value, _dt):
        return value if value.tzinfo else value.replace(tzinfo=_tz.utc)
    try:
        return _dt.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        try:
            return _dt.strptime(str(value), "%Y-%m-%d %H:%M:%S").replace(tzinfo=_tz.utc)
        except (ValueError, TypeError):
            return None


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


# ---------------------------------------------------------------------------
# Platform-mode reimplementations of the functions called by the legacy shim.
# Each returns values in the same shape the SQLite version returned, so the
# calling bot code is oblivious to which backend is in play.
# ---------------------------------------------------------------------------


def platform_log_prediction(**kwargs: Any) -> int:
    service = get_prediction_service()
    return service.log(**kwargs)


def platform_get_track_record(
    *,
    days: Optional[int] = None,
    market: Optional[str] = None,
    league: Optional[str] = None,
    confidence: Optional[str] = None,
    source: Optional[str] = None,
    published_only: bool = False,
    min_odds: Optional[float] = None,
    **_: Any,
) -> Dict[str, Any]:
    return get_prediction_service().track_record(
        days=days, market=market, league=league,
        confidence=confidence, source=source, published_only=published_only,
        min_odds=min_odds,
    )


def platform_get_recent_predictions(
    *,
    days: int = 30,
    league: Optional[str] = None,
    market: Optional[str] = None,
    published_only: bool = False,
    limit: int = 500,
    **_: Any,
) -> list:
    return get_prediction_service().recent(
        days=days, league=league, market=market,
        published_only=published_only, limit=limit,
    )


def platform_get_daily_breakdown(
    *,
    target_date=None,
    published_only: bool = False,
    **_: Any,
) -> Dict[str, Any]:
    from datetime import date as _date
    the_date = target_date or _date.today()
    if not isinstance(the_date, _date):
        try:
            the_date = _date.fromisoformat(str(the_date))
        except ValueError:
            the_date = _date.today()
    return get_prediction_service().daily_breakdown(
        target_date=the_date,
        published_only=published_only,
    )


def platform_get_calibration_data(
    *,
    buckets: int = 20,
    published_only: bool = False,
    **_: Any,
) -> list:
    return get_prediction_service().calibration(
        buckets=buckets,
        published_only=published_only,
    )


def platform_get_unresolved_for_grading(*, on_or_before=None) -> list:
    """Return predictions awaiting outcome grading. Used by resolve_outcomes."""
    from datetime import date as _date
    before = on_or_before
    if before is not None and not isinstance(before, _date):
        try:
            before = _date.fromisoformat(str(before))
        except ValueError:
            before = None
    return get_prediction_service().unresolved(before_date=before)


def platform_mark_outcome(prediction_id: int, *, outcome: str, actual_result: Optional[float] = None) -> bool:
    return get_prediction_service().mark_outcome(
        prediction_id, outcome=outcome, actual_result=actual_result,
    )


def platform_capture_closing_odds(
    prediction_id: int,
    *,
    closing_odds: float,
    closing_implied_prob: Optional[float] = None,
    clv: Optional[float] = None,
) -> bool:
    return get_prediction_service().capture_closing(
        prediction_id,
        closing_odds=closing_odds,
        closing_implied_prob=closing_implied_prob,
        clv=clv,
    )
