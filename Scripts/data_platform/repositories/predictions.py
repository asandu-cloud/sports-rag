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

from sqlalchemy import and_, desc, func, or_, select, update
from sqlalchemy.orm import Session

from ..db import session_scope
from ..models import Fixture, Prediction, PublishedRecommendation
from ..outcomes import normalize_outcome
from ..tracking_metrics import build_calibration, build_daily_breakdown, build_track_record
from ..publication_identity import source_cohort, tracking_identity, utc_date
from ..publication_reporting import publication_metadata, select_publication_scope
from ..settlement_policy import uses_participation_cards

logger = logging.getLogger(__name__)


def _row_to_dict(row: Optional[Prediction]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    extras = row.extras if isinstance(row.extras, Mapping) else {}
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
        "tracking_cohort": source_cohort(row.source),
        "fixture_date": utc_date(row.kickoff),
        "tracking": extras.get("tracking"),
        "settlement": extras.get("settlement"),
        "settlement_policy": extras.get("settlement_policy"),
        "closing_capture": extras.get("closing_capture"),
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

    @staticmethod
    def _report_rows(session, rows, *, published_only=False, publication_scope="initial"):
        rows = [_row_to_dict(row) for row in rows]
        metadata = publication_metadata(session, [row["id"] for row in rows])
        for row in rows:
            row.update(metadata.get(row["id"], {}))
            if uses_participation_cards(row):
                row["card_definition_status"] = "product_participation_estimate"
        return select_publication_scope(rows, publication_scope) if published_only else rows

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
        published_only: bool = False,
        publication_scope: str = "initial",
        graded_only: bool = False,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        since = date.today() - timedelta(days=days)
        with self._factory() as session:
            stmt = select(Prediction).where(Prediction.prediction_date >= since)
            if league is not None:
                stmt = stmt.where(Prediction.league == league)
            if market is not None:
                stmt = stmt.where(Prediction.market == market)
            if graded_only:
                stmt = stmt.where(Prediction.outcome.isnot(None))
            if published_only:
                stmt = stmt.join(
                    PublishedRecommendation,
                    PublishedRecommendation.prediction_id == Prediction.id,
                )
            stmt = stmt.order_by(desc(Prediction.prediction_date), desc(Prediction.id))
            if not published_only:
                stmt = stmt.limit(limit)
            rows = session.scalars(stmt).all()
            return self._report_rows(session, rows, published_only=published_only,
                                     publication_scope=publication_scope)[:limit]

    def get_track_record(
        self,
        *,
        market: Optional[str] = None,
        league: Optional[str] = None,
        confidence: Optional[str] = None,
        source: Optional[str] = None,
        published_only: bool = False,
        publication_scope: str = "initial",
        days: Optional[int] = None,
        min_odds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return the shared Discord/website track-record contract."""
        with self._factory() as session:
            stmt = select(Prediction)
            if market is not None:
                stmt = stmt.where(Prediction.market == market)
            if league is not None:
                stmt = stmt.where(Prediction.league == league)
            if confidence is not None:
                stmt = stmt.where(Prediction.confidence == confidence)
            if source is not None:
                stmt = stmt.where(Prediction.source == source)
            if published_only:
                stmt = stmt.join(
                    PublishedRecommendation,
                    PublishedRecommendation.prediction_id == Prediction.id,
                )
            if days is not None:
                stmt = stmt.where(Prediction.prediction_date >= date.today() - timedelta(days=days))
            if min_odds is not None:
                stmt = stmt.where(Prediction.odds >= min_odds)
            rows = session.scalars(stmt.order_by(Prediction.prediction_date, Prediction.id)).all()
            data = self._report_rows(session, rows, published_only=published_only, publication_scope=publication_scope)
        return {**build_track_record(data), "publication_scope": publication_scope if published_only else None}

    def get_daily_breakdown(
        self,
        *,
        target_date: date,
        published_only: bool = False,
        publication_scope: str = "initial",
    ) -> Dict[str, Any]:
        with self._factory() as session:
            stmt = select(Prediction).where(
                Prediction.prediction_date == target_date,
                Prediction.outcome.isnot(None),
            )
            if published_only:
                stmt = stmt.join(
                    PublishedRecommendation,
                    PublishedRecommendation.prediction_id == Prediction.id,
                )
            rows = session.scalars(stmt).all()
            data = self._report_rows(session, rows, published_only=published_only, publication_scope=publication_scope)
        return build_daily_breakdown(
            data,
            target_date=target_date,
            deduplicate=not published_only,
        )

    def get_calibration_data(
        self,
        *,
        buckets: int = 20,
        published_only: bool = False,
        publication_scope: str = "initial",
    ) -> List[Dict[str, Any]]:
        """Return the shared calibration contract used by the Discord embed."""
        with self._factory() as session:
            stmt = select(Prediction).where(
                Prediction.outcome.isnot(None),
                Prediction.model_prob.isnot(None),
            )
            if published_only:
                stmt = stmt.join(
                    PublishedRecommendation,
                    PublishedRecommendation.prediction_id == Prediction.id,
                )
            rows = session.scalars(stmt).all()
            data = self._report_rows(session, rows, published_only=published_only, publication_scope=publication_scope)
        return build_calibration(
            data,
            buckets=buckets,
        )

    # ------------------------------------------------------------------
    # Outcome resolution / CLV
    # ------------------------------------------------------------------
    def settlement_candidates(self, *, league=None, prediction_ids=None, include_graded=False,
                              published_after_id=None):
        """Use fixture identity/schedule, never publication-day result slices.

        Older publications may supply saved quote identity but missing periods
        stay missing. Neither publication history nor prediction fields change.
        """
        with self._factory() as session:
            stmt = select(Prediction)
            if not include_graded:
                stmt = stmt.where(Prediction.outcome.is_(None))
            if league:
                stmt = stmt.where(Prediction.league == league)
            if prediction_ids is not None:
                stmt = stmt.where(Prediction.id.in_(prediction_ids))
            if published_after_id is not None:
                stmt = stmt.join(PublishedRecommendation, PublishedRecommendation.prediction_id == Prediction.id).where(
                    PublishedRecommendation.id > published_after_id)
            rows = session.scalars(stmt.order_by(Prediction.id)).all()
            data = {row.id: _row_to_dict(row) for row in rows}
            if not data:
                return []
            p = PublishedRecommendation
            for publication in session.execute(select(
                p.id, p.released_at, p.prediction_id, p.decision_json["fixture"].label("fixture"),
                p.decision_json["market"].label("market"),
                p.decision_json["decision"]["quote"].label("quote"),
            ).where(p.prediction_id.in_(data))):
                item = data[publication.prediction_id]
                item["recommendation_id"] = publication.id
                item["released_at"] = publication.released_at.replace(tzinfo=timezone.utc).isoformat() if publication.released_at.tzinfo is None else publication.released_at.isoformat()
                if not item.get("tracking"):
                    try:
                        item["tracking"] = tracking_identity({
                            "fixture": publication.fixture, "market": publication.market,
                            "decision": {"quote": publication.quote},
                        })
                    except (TypeError, ValueError, AttributeError):
                        item["tracking"] = {"error": "malformed_publication_identity"}
            from ..settlement import provider_id
            ids = {int(fid) for item in data.values() if (fid := provider_id(item["fixture_id"]))}
            schedules = {str(fid): kickoff for fid, kickoff in session.execute(select(
                Fixture.api_football_id, Fixture.kickoff_utc
            ).where(Fixture.api_football_id.in_(ids)))}
            for item in data.values():
                kickoff = schedules.get(provider_id(item["fixture_id"]))
                if kickoff is not None:
                    # SQLite discards timezone information for typed UTC columns.
                    item["scheduled_kickoff"] = kickoff.replace(tzinfo=timezone.utc).isoformat() if kickoff.tzinfo is None else kickoff.isoformat()
            return list(data.values())

    def record_settlement(self, prediction, assessment, *, checked_at, allow_correction=False):
        """Conditional write with evidence/history; a retry is not another stake.

        The caller's expected grade prevents stale workers overwriting a newer
        outcome. Corrections require an explicit targeted regrade. No migration.
        """
        with self._factory() as session:
            row = session.get(Prediction, int(prediction["id"]))
            if row is None:
                return "missing"
            current = _row_to_dict(row)
            if any(current.get(key) != prediction.get(key) for key in (
                    "fixture_id", "league", "market", "pick", "side", "line", "odds", "bookmaker", "kickoff",
                    "settlement_policy")):
                return "conflict"
            expected_time = datetime.fromisoformat(prediction["graded_at"]) if prediction.get("graded_at") else None
            if expected_time is not None and expected_time.tzinfo is None:
                expected_time = expected_time.replace(tzinfo=timezone.utc)
            stored_time = row.graded_at
            if stored_time is not None and stored_time.tzinfo is None:
                stored_time = stored_time.replace(tzinfo=timezone.utc)
            if (row.outcome != prediction.get("outcome") or stored_time != expected_time
                    or _to_float(row.actual_result) != prediction.get("actual_result")
                    or ((row.extras or {}).get("settlement") or {}).get("evidence_hash")
                    != (prediction.get("settlement") or {}).get("evidence_hash")):
                return "conflict"
            outcome = normalize_outcome(assessment.get("outcome"))
            if assessment.get("outcome") and outcome is None:
                raise ValueError("Unknown settlement outcome")
            if row.outcome is not None and not allow_correction:
                return "conflict"
            extras = dict(row.extras or {})
            previous = dict(extras.get("settlement") or {})
            if outcome and previous.get("evidence_hash") == assessment.get("evidence_hash") and row.outcome == outcome and _to_float(row.actual_result) == assessment.get("actual_result"):
                return "unchanged"
            history = list(previous.get("history") or [])
            if row.outcome is not None and (outcome is not None or previous.get("status") != "review_pending"):
                history.append({"outcome": row.outcome, "actual_result": _to_float(row.actual_result),
                                "graded_at": row.graded_at.isoformat() if row.graded_at else None,
                                "evidence": {key: value for key, value in previous.items() if key != "history"}})
            extras["settlement"] = {
                **assessment, "status": "settled" if outcome else "review_pending" if row.outcome else "pending",
                "first_checked_at": previous.get("first_checked_at") or checked_at.isoformat(),
                "last_checked_at": checked_at.isoformat(),
                "attempts": int(previous.get("attempts") or 0) + 1, "history": history,
            }
            values = {"extras": extras}
            if outcome:
                values.update(outcome=outcome, actual_result=assessment.get("actual_result"), graded_at=checked_at)
            # Compare the version read inside this transaction as well, so two
            # simultaneous writers cannot both overwrite each other's evidence.
            statement = update(Prediction).where(
                Prediction.id == row.id, Prediction.outcome == row.outcome,
                Prediction.graded_at == row.graded_at, Prediction.updated_at == row.updated_at,
            ).values(**values).execution_options(synchronize_session=False)
            return "applied" if session.execute(statement).rowcount == 1 else "conflict"

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
                canonical_outcome = normalize_outcome(outcome)
                if canonical_outcome is None:
                    logger.warning("Rejected unknown prediction outcome %r for id=%s", outcome, prediction_id)
                    return False
                row.outcome = canonical_outcome
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
