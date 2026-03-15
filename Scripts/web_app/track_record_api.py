#!/usr/bin/env python3
"""
Track Record API — FastAPI endpoints for the prediction track record page.

Endpoints:
    GET /api/track-record         — full track record stats
    GET /api/track-record/daily   — daily performance for charting
    GET /api/track-record/recent  — most recent graded predictions

Can be mounted onto the existing FastAPI app or run standalone for testing.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RAG_INGEST = os.path.join(_PROJECT_ROOT, "Scripts", "rag_ingest")
if _RAG_INGEST not in sys.path:
    sys.path.insert(0, _RAG_INGEST)

from fastapi import APIRouter, FastAPI, Query
from pydantic import BaseModel, Field

from prediction_tracker import get_track_record, get_recent_predictions

log = logging.getLogger("track_record_api")

# ---------------------------------------------------------------------------
# Router (mountable onto existing app)
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/track-record", tags=["Track Record"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ConfidenceBreakdown(BaseModel):
    total: int = 0
    hits: int = 0
    hit_rate: float = 0.0
    roi: float = 0.0


class MarketBreakdown(BaseModel):
    total: int = 0
    hits: int = 0
    hit_rate: float = 0.0


class LeagueBreakdown(BaseModel):
    total: int = 0
    hits: int = 0
    hit_rate: float = 0.0


class DailyEntry(BaseModel):
    date: str
    total: int
    hits: int
    hit_rate: float


class TrackRecordResponse(BaseModel):
    total_graded: int = 0
    hits: int = 0
    misses: int = 0
    pushes: int = 0
    hit_rate: float = 0.0
    roi_flat_stake: float = 0.0
    by_confidence: dict = Field(default_factory=dict)
    by_market: dict = Field(default_factory=dict)
    by_league: dict = Field(default_factory=dict)
    recent_streak: int = 0
    best_streak: int = 0
    daily_performance: list = Field(default_factory=list)


class PredictionEntry(BaseModel):
    id: int
    home_team: str
    away_team: str
    league: str
    market: str
    pick: str
    side: Optional[str] = None
    line: Optional[float] = None
    odds: Optional[float] = None
    bookmaker: Optional[str] = None
    model_prob: Optional[float] = None
    confidence: Optional[str] = None
    projected_total: Optional[float] = None
    source: Optional[str] = None
    actual_result: Optional[float] = None
    outcome: Optional[str] = None
    prediction_date: Optional[str] = None
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=TrackRecordResponse)
async def track_record(
    days: Optional[int] = Query(None, description="Limit to last N days"),
    league: Optional[str] = Query(None, description="Filter to one league"),
    market: Optional[str] = Query(None, description="Filter to one market"),
    confidence: Optional[str] = Query(None, description="Filter by confidence level"),
):
    """Full track record statistics."""
    stats = get_track_record(
        days=days,
        league=league,
        market=market,
        confidence=confidence,
    )
    if "error" in stats:
        return TrackRecordResponse()
    return TrackRecordResponse(**stats)


@router.get("/daily", response_model=list[DailyEntry])
async def daily_performance(
    days: int = Query(30, description="Number of days to return"),
    league: Optional[str] = Query(None, description="Filter to one league"),
):
    """Daily performance array for charting."""
    stats = get_track_record(days=days, league=league)
    if "error" in stats:
        return []
    return [DailyEntry(**d) for d in stats.get("daily_performance", [])]


@router.get("/recent", response_model=list[PredictionEntry])
async def recent_predictions(
    limit: int = Query(20, description="Number of predictions to return", ge=1, le=100),
    league: Optional[str] = Query(None, description="Filter to one league"),
):
    """Most recent graded predictions with outcomes."""
    preds = get_recent_predictions(limit=limit, league=league, graded_only=True)
    result = []
    for p in preds:
        try:
            result.append(PredictionEntry(
                id=p.get("id", 0),
                home_team=p.get("home_team", ""),
                away_team=p.get("away_team", ""),
                league=p.get("league", ""),
                market=p.get("market", ""),
                pick=p.get("pick", ""),
                side=p.get("side"),
                line=p.get("line"),
                odds=p.get("odds"),
                bookmaker=p.get("bookmaker"),
                model_prob=p.get("model_prob"),
                confidence=p.get("confidence"),
                projected_total=p.get("projected_total"),
                source=p.get("source"),
                actual_result=p.get("actual_result"),
                outcome=p.get("outcome"),
                prediction_date=p.get("prediction_date"),
                created_at=p.get("created_at"),
            ))
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# Standalone runner (for testing)
# ---------------------------------------------------------------------------

def create_standalone_app() -> FastAPI:
    """Create a standalone FastAPI app with just the track record endpoints."""
    app = FastAPI(
        title="Matchwise Track Record API",
        description="Track record statistics for the Matchwise prediction platform",
        version="1.0.0",
    )
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    app = create_standalone_app()
    uvicorn.run(app, host="0.0.0.0", port=8001)
