"""FastAPI server for live match data.

Serves real-time match projections from the LivePoller. Designed as a
standalone service (port 8001) that can be merged into the main website
API (port 8000) later.

Endpoints:
    GET /api/live/fixtures           — all currently live fixtures
    GET /api/live/fixtures/{id}      — full snapshot for one fixture
    GET /api/live/fixtures/{id}/state — raw LiveMatchState (for debugging)
    GET /api/live/fixtures/{id}/history — snapshot history for charting
    GET /api/live/stream             — SSE stream (?fixtures=123,456)
    GET /api/live/health             — health check with metrics

Run:
    cd Scripts/live
    python live_api.py                # starts on 0.0.0.0:8001
    uvicorn live_api:app --port 8001  # alternative
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup — ensure sibling modules are importable
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
_LIVE_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_LIVE_DIR))

from live.live_poller import LiveMatchState, LiveSnapshot
from live.live_runtime import get_live_poller

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("live_api")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Poller singleton
# ---------------------------------------------------------------------------
poller = get_live_poller(auto_subscribe=True)

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class FixtureSummary(BaseModel):
    fixture_id: int
    home: str
    away: str
    score: str
    elapsed: int
    status: str
    league: str
    league_id: int


class FixturesResponse(BaseModel):
    fixtures: List[FixtureSummary]
    count: int


class HealthResponse(BaseModel):
    running: bool
    started_at: Optional[str] = None
    active_fixtures: int
    subscribers: int
    archived: int
    poll_count: int
    detail_poll_count: int
    errors: int
    has_live_engine: bool
    has_rag_priors: bool
    has_alert_engine: bool
    has_live_store: bool
    has_live_odds: bool


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the poller on startup, stop on shutdown."""
    logger.info("Starting LivePoller on app startup")
    await poller.start()
    yield
    logger.info("Stopping LivePoller on app shutdown")
    await poller.stop()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Matchwise Live API",
    description="Real-time live match projections powered by the Betting RAG engine",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — permissive for local dev, tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: convert dataclass to JSON-safe dict
# ---------------------------------------------------------------------------

def _snap_to_dict(snap: LiveSnapshot) -> Dict[str, Any]:
    """Convert a LiveSnapshot to a JSON-serializable dict."""
    return snap.to_dict()


def _state_to_dict(state: LiveMatchState) -> Dict[str, Any]:
    """Convert a LiveMatchState to a JSON-serializable dict."""
    return state.to_dict()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/live/fixtures", response_model=FixturesResponse)
async def list_fixtures():
    """Return all currently live fixtures with basic info."""
    fixtures = poller.get_active_fixtures()
    return FixturesResponse(
        fixtures=[FixtureSummary(**f) for f in fixtures],
        count=len(fixtures),
    )


@app.get("/api/live/fixtures/{fixture_id}")
async def get_fixture_snapshot(fixture_id: int):
    """Return the latest LiveSnapshot for a fixture.

    If the fixture is known but not yet subscribed, this endpoint
    triggers a subscription so detail polling begins.
    """
    # Auto-subscribe if fixture is known
    state = poller.get_state(fixture_id)
    if state is not None:
        poller.subscribe(fixture_id)

    snapshot = poller.get_snapshot(fixture_id)
    if snapshot is None:
        # Check if we have a state but no snapshot yet (just subscribed)
        if state is not None:
            return {
                "fixture_id": fixture_id,
                "status": "pending",
                "message": "Fixture subscribed, waiting for first detail poll",
                "state_summary": {
                    "home": state.home_team,
                    "away": state.away_team,
                    "score": f"{state.home_stats.goals}-{state.away_stats.goals}",
                    "elapsed": state.elapsed_min,
                    "league": state.league,
                },
                "priors": {
                    "total_goals": state.prior_total_goals,
                    "total_corners": state.prior_total_corners,
                    "total_cards": state.prior_total_cards,
                    "total_sot": state.prior_total_sot,
                    "btts_prob": state.prior_btts_prob,
                    "moneyline": state.prior_moneyline,
                },
                "live_odds": state.live_odds,
            }
        raise HTTPException(
            status_code=404,
            detail=f"Fixture {fixture_id} not found or not currently live",
        )

    return _snap_to_dict(snapshot)


@app.get("/api/live/fixtures/{fixture_id}/state")
async def get_fixture_state(fixture_id: int):
    """Return the raw LiveMatchState for debugging/inspection."""
    state = poller.get_state(fixture_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Fixture {fixture_id} not found",
        )
    return _state_to_dict(state)


@app.get("/api/live/fixtures/{fixture_id}/history")
async def get_fixture_history(fixture_id: int):
    """Return array of past snapshots for charting (last 20)."""
    history = poller.get_snapshot_history(fixture_id)
    if not history:
        state = poller.get_state(fixture_id)
        if state is None:
            raise HTTPException(
                status_code=404,
                detail=f"Fixture {fixture_id} not found",
            )
        return {"fixture_id": fixture_id, "snapshots": [], "count": 0}

    return {
        "fixture_id": fixture_id,
        "snapshots": [_snap_to_dict(s) for s in history],
        "count": len(history),
    }


@app.get("/api/live/stream")
async def live_stream(
    fixtures: str = Query(
        default="",
        description="Comma-separated fixture IDs to stream. Empty = all active.",
    ),
):
    """Server-Sent Events stream for real-time snapshot updates.

    Event types:
        snapshot  — full LiveSnapshot JSON on each update
        alert     — individual alert from a snapshot
        heartbeat — keepalive with timestamp (every 5s)

    Query parameters:
        fixtures — comma-separated fixture IDs (e.g., ?fixtures=123,456)
                   If empty, streams all active fixtures.
    """
    # Parse requested fixture IDs
    if fixtures.strip():
        fixture_ids = [
            int(x) for x in fixtures.split(",")
            if x.strip().isdigit()
        ]
    else:
        fixture_ids = []  # empty = all active

    async def event_generator() -> AsyncGenerator[str, None]:
        # Subscribe to requested fixtures
        for fid in fixture_ids:
            poller.subscribe(fid)

        last_sent: Dict[int, str] = {}  # fixture_id -> last snapshot timestamp
        heartbeat_interval = 5  # seconds

        while True:
            # Determine which fixtures to stream
            if fixture_ids:
                target_ids = fixture_ids
            else:
                target_ids = [
                    f["fixture_id"]
                    for f in poller.get_active_fixtures()
                ]

            for fid in target_ids:
                snap = poller.get_snapshot(fid)
                if snap is None:
                    continue

                # Only send if snapshot has changed
                if snap.timestamp != last_sent.get(fid):
                    snap_dict = _snap_to_dict(snap)
                    data = json.dumps(snap_dict, default=str)
                    yield f"event: snapshot\ndata: {data}\n\n"
                    last_sent[fid] = snap.timestamp

                    # Send individual alerts
                    for alert in snap.alerts:
                        alert_data = json.dumps({
                            "fixture_id": fid,
                            **alert,
                        }, default=str)
                        yield f"event: alert\ndata: {alert_data}\n\n"

            # Heartbeat
            heartbeat = json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "active_fixtures": len(poller.get_active_fixtures()),
            })
            yield f"event: heartbeat\ndata: {heartbeat}\n\n"

            await asyncio.sleep(heartbeat_interval)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@app.get("/api/live/health", response_model=HealthResponse)
async def health_check():
    """Health check with operational metrics."""
    health = poller.get_health()
    return HealthResponse(**health)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("LIVE_API_PORT", "8001"))
    logger.info("Starting Matchwise Live API on port %d", port)
    uvicorn.run(
        "live_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
