"""PredictionService — orchestrates the PredictionRepository.

Keeps the bot-facing contract (log / resolve / track-record) but drives
the writes through Postgres. Outcome grading against API-Football lives
here so callers don't need to know about the repo + fixtures pipeline.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from ..repositories.predictions import PredictionRepository

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, repo: Optional[PredictionRepository] = None):
        self._repo = repo or PredictionRepository()

    # ---- log ---------------------------------------------------------
    def log(self, **kwargs: Any) -> int:
        return self._repo.log(**kwargs)

    # ---- query -------------------------------------------------------
    def recent(self, **kwargs: Any) -> List[Dict[str, Any]]:
        return self._repo.get_recent(**kwargs)

    def unresolved(self, **kwargs: Any) -> List[Dict[str, Any]]:
        return self._repo.get_unresolved(**kwargs)

    def track_record(self, **kwargs: Any) -> Dict[str, Any]:
        return self._repo.get_track_record(**kwargs)

    def daily_breakdown(self, *, target_date: date) -> Dict[str, Any]:
        return self._repo.get_daily_breakdown(target_date=target_date)

    def calibration(self, *, buckets: int = 10) -> List[Dict[str, Any]]:
        return self._repo.get_calibration_data(buckets=buckets)

    # ---- grading -----------------------------------------------------
    def mark_outcome(
        self,
        prediction_id: int,
        *,
        outcome: str,
        actual_result: Optional[float] = None,
    ) -> bool:
        return self._repo.set_outcome(
            prediction_id,
            outcome=outcome,
            actual_result=actual_result,
            graded_at=datetime.now(timezone.utc),
        )

    def capture_closing(
        self,
        prediction_id: int,
        *,
        closing_odds: float,
        closing_implied_prob: Optional[float] = None,
        clv: Optional[float] = None,
    ) -> bool:
        return self._repo.set_closing_odds(
            prediction_id,
            closing_odds=closing_odds,
            closing_implied_prob=closing_implied_prob,
            clv=clv,
        )
