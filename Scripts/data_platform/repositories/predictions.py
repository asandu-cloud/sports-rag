"""Repository for the ``predictions`` table.

Mirrors the shape of legacy ``Scripts/rag_ingest/prediction_tracker.py``
so the shim in that module can forward straight through. Dictionaries
coming out of this repo look identical to the ``sqlite3.Row`` payloads
the old code was returning.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.orm import Session

from ..db import session_scope
from ..models import Prediction

logger = logging.getLogger(__name__)


def _row_to_dict(row: Optional[Prediction]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {
        "id": row.id,
        "fixture_id": row.fixture_api_id,
        "home_team": row.home_team,
        "away_team": row.away_team,
        "league": row.league,
        "kickoff": row.kickoff,
        "market": row.market,
        "pick": row.pick,
        "side": row.side,
        "line": _to_float(row.line),
        "odds": _to_float(row.odds),
        "bookmaker": row.bookmaker,
        "model_prob": _to_float(row.model_prob),
        "implied_prob": _to_float(row.implied_prob),
        "value_edge": _to_float(row.value_edge),
        "confidence": row.confidence,
        "projected_total": _to_float(row.projected_total),
        "source": row.source,
        "actual_result": _to_float(row.actual_result),
        "outcome": row.outcome,
        "graded_at": row.graded_at.isoformat() if row.graded_at else None,
        "season": row.season,
        "prediction_date": row.prediction_date.isoformat() if row.prediction_date else None,
        "closing_odds": _to_float(row.closing_odds),
        "closing_implied_prob": _to_float(row.closing_implied_prob),
        "clv": _to_float(row.clv),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class PredictionRepository:
    """All prediction CRUD + grading lives here."""

    def __init__(self, session_factory=session_scope):
        self._factory = session_factory

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log(
        self,
        *,
        home_team: str,
        away_team: str,
        league: str,
        market: str,
        pick: str,
        side: Optional[str] = None,
        line: Optional[float] = None,
        odds: Optional[float] = None,
        bookmaker: Optional[str] = None,
        model_prob: Optional[float] = None,
        implied_prob: Optional[float] = None,
        value_edge: Optional[float] = None,
        confidence: Optional[str] = None,
        projected_total: Optional[float] = None,
        source: str = "standalone",
        fixture_id: Optional[str] = None,
        kickoff: Optional[str] = None,
        season: Optional[str] = None,
        prediction_date: Optional[date] = None,
    ) -> int:
        """Insert-or-find a prediction using the legacy dedup key.

        Returns the persisted row id, or ``-1`` on failure (matches the
        legacy contract that callers never crash).
        """
        try:
            the_date = prediction_date or date.today()
            with self._factory() as session:
                existing = session.scalar(
                    select(Prediction).where(
                        Prediction.prediction_date == the_date,
                        Prediction.home_team == home_team,
                        Prediction.away_team == away_team,
                        Prediction.league == league,
                        Prediction.market == market,
                        Prediction.pick == pick,
                        Prediction.source == source,
                    ).order_by(desc(Prediction.id)).limit(1)
                )
                if existing is not None:
                    return int(existing.id)

                row = Prediction(
                    fixture_api_id=str(fixture_id) if fixture_id is not None else None,
                    home_team=home_team,
                    away_team=away_team,
                    league=league,
                    kickoff=kickoff,
                    market=market,
                    pick=pick,
                    side=side,
                    line=line,
                    odds=odds,
                    bookmaker=bookmaker,
                    model_prob=model_prob,
                    implied_prob=implied_prob,
                    value_edge=value_edge,
                    confidence=confidence,
                    projected_total=projected_total,
                    source=source,
                    season=season,
                    prediction_date=the_date,
                )
                session.add(row)
                session.flush()
                return int(row.id)
        except Exception:
            logger.exception("PredictionRepository.log failed")
            return -1

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_unresolved(
        self,
        *,
        before_date: Optional[date] = None,
        match_date: Optional[date] = None,
        league: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        with self._factory() as session:
            stmt = select(Prediction).where(Prediction.outcome.is_(None))
            if match_date is not None:
                stmt = stmt.where(Prediction.prediction_date == match_date)
            elif before_date is not None:
                stmt = stmt.where(Prediction.prediction_date <= before_date)
            if league is not None:
                stmt = stmt.where(Prediction.league == league)
            stmt = stmt.order_by(Prediction.prediction_date.asc(), Prediction.id.asc())
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = session.scalars(stmt).all()
            return [d for d in (_row_to_dict(r) for r in rows) if d]

    def get_recent(
        self,
        *,
        days: int = 30,
        league: Optional[str] = None,
        market: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        since = date.today() - timedelta(days=days)
        with self._factory() as session:
            stmt = select(Prediction).where(Prediction.prediction_date >= since)
            if league is not None:
                stmt = stmt.where(Prediction.league == league)
            if market is not None:
                stmt = stmt.where(Prediction.market == market)
            stmt = stmt.order_by(desc(Prediction.prediction_date), desc(Prediction.id)).limit(limit)
            rows = session.scalars(stmt).all()
            return [d for d in (_row_to_dict(r) for r in rows) if d]

    def get_track_record(
        self,
        *,
        market: Optional[str] = None,
        league: Optional[str] = None,
        confidence: Optional[str] = None,
        days: Optional[int] = None,
        min_odds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Hit-rate / ROI / streak summary for resolved predictions."""
        with self._factory() as session:
            stmt = select(Prediction).where(Prediction.outcome.isnot(None))
            if market is not None:
                stmt = stmt.where(Prediction.market == market)
            if league is not None:
                stmt = stmt.where(Prediction.league == league)
            if confidence is not None:
                stmt = stmt.where(Prediction.confidence == confidence)
            if days is not None:
                stmt = stmt.where(Prediction.prediction_date >= date.today() - timedelta(days=days))
            if min_odds is not None:
                stmt = stmt.where(Prediction.odds >= min_odds)
            rows = session.scalars(stmt.order_by(desc(Prediction.prediction_date))).all()
        total = len(rows)
        wins = sum(1 for r in rows if r.outcome == "win")
        losses = sum(1 for r in rows if r.outcome == "loss")
        pushes = sum(1 for r in rows if r.outcome == "push")
        roi_stake = 0.0
        roi_return = 0.0
        for r in rows:
            odds = _to_float(r.odds)
            if odds is None:
                continue
            roi_stake += 1.0
            if r.outcome == "win":
                roi_return += odds
            elif r.outcome == "push":
                roi_return += 1.0
        roi = (roi_return / roi_stake - 1.0) if roi_stake > 0 else 0.0
        hit_rate = (wins / max(1, wins + losses)) if (wins + losses) else 0.0
        streak = _compute_streak(rows)
        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "hit_rate": hit_rate,
            "roi": roi,
            "stake_units": roi_stake,
            "current_streak": streak,
        }

    def get_daily_breakdown(self, *, target_date: date) -> Dict[str, Any]:
        with self._factory() as session:
            stmt = select(Prediction).where(Prediction.prediction_date == target_date)
            rows = session.scalars(stmt).all()
        total = len(rows)
        resolved = [r for r in rows if r.outcome is not None]
        wins = sum(1 for r in resolved if r.outcome == "win")
        losses = sum(1 for r in resolved if r.outcome == "loss")
        pushes = sum(1 for r in resolved if r.outcome == "push")
        return {
            "date": target_date.isoformat(),
            "total": total,
            "resolved": len(resolved),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "predictions": [d for d in (_row_to_dict(r) for r in rows) if d],
        }

    def get_calibration_data(self, *, buckets: int = 10) -> List[Dict[str, Any]]:
        """Return predicted-vs-observed per confidence bucket."""
        with self._factory() as session:
            rows = session.scalars(
                select(Prediction).where(
                    Prediction.outcome.isnot(None),
                    Prediction.model_prob.isnot(None),
                    Prediction.outcome.in_(("win", "loss")),
                )
            ).all()
        bucket_stats: Dict[int, Dict[str, float]] = {}
        for row in rows:
            try:
                prob = float(row.model_prob)
            except (TypeError, ValueError):
                continue
            bucket = min(buckets - 1, max(0, int(prob * buckets)))
            entry = bucket_stats.setdefault(bucket, {"sum_prob": 0.0, "wins": 0, "total": 0})
            entry["sum_prob"] += prob
            entry["total"] += 1
            if row.outcome == "win":
                entry["wins"] += 1
        out = []
        for bucket in range(buckets):
            entry = bucket_stats.get(bucket, {"sum_prob": 0.0, "wins": 0, "total": 0})
            total = entry["total"]
            if total == 0:
                continue
            out.append({
                "bucket": bucket,
                "range": [bucket / buckets, (bucket + 1) / buckets],
                "avg_predicted": entry["sum_prob"] / total,
                "observed_hit_rate": entry["wins"] / total,
                "sample_size": total,
            })
        return out

    # ------------------------------------------------------------------
    # Outcome resolution / CLV
    # ------------------------------------------------------------------
    def set_outcome(
        self,
        prediction_id: int,
        *,
        outcome: str,
        actual_result: Optional[float] = None,
        graded_at: Optional[datetime] = None,
    ) -> bool:
        try:
            with self._factory() as session:
                row = session.get(Prediction, prediction_id)
                if row is None:
                    return False
                row.outcome = outcome
                if actual_result is not None:
                    row.actual_result = actual_result
                row.graded_at = graded_at or datetime.now(timezone.utc)
            return True
        except Exception:
            logger.exception("set_outcome failed for id=%s", prediction_id)
            return False

    def set_closing_odds(
        self,
        prediction_id: int,
        *,
        closing_odds: float,
        closing_implied_prob: Optional[float] = None,
        clv: Optional[float] = None,
    ) -> bool:
        try:
            with self._factory() as session:
                row = session.get(Prediction, prediction_id)
                if row is None:
                    return False
                row.closing_odds = closing_odds
                if closing_implied_prob is not None:
                    row.closing_implied_prob = closing_implied_prob
                if clv is not None:
                    row.clv = clv
            return True
        except Exception:
            logger.exception("set_closing_odds failed for id=%s", prediction_id)
            return False


def _compute_streak(rows: Iterable[Prediction]) -> int:
    """Current streak: positive = consecutive wins, negative = losses."""
    streak = 0
    for row in rows:
        if row.outcome == "win":
            if streak >= 0:
                streak += 1
            else:
                break
        elif row.outcome == "loss":
            if streak <= 0:
                streak -= 1
            else:
                break
        else:
            continue
    return streak
