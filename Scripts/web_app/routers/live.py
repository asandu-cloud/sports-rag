"""Live-match FastAPI router.

All endpoints preserve the paths previously served by
``Scripts/live/live_api.py`` so the live Discord bot and any existing
frontend subscribers keep working unchanged.

The router reads from two sources:
  * the in-memory ``LivePoller`` singleton (hot path — current
    snapshot, stream, health metrics) — unchanged from V1.
  * ``LiveService`` on top of Postgres — used as a durable fallback and
    for historical queries (snapshot history, recorded alerts).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("web_app.live")

# ---------------------------------------------------------------------------
# Path wiring so we can import the live runtime + data platform
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_WEB_APP = _HERE.parents[1]
_SCRIPTS = _WEB_APP.parent
_PROJECT = _SCRIPTS.parent
for p in (str(_SCRIPTS), str(_PROJECT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Lazy singletons so tests / environments without the live stack still work
# ---------------------------------------------------------------------------

_poller = None
_live_service = None


def _get_poller():
    global _poller
    if _poller is None:
        try:
            from live.live_runtime import get_live_poller
            _poller = get_live_poller(auto_subscribe=True)
        except Exception:
            logger.warning("LivePoller unavailable in this process", exc_info=True)
            _poller = _NoopPoller()
    return _poller


def _get_live_service():
    global _live_service
    if _live_service is None:
        try:
            from data_platform.services import LiveService
            _live_service = LiveService()
        except Exception:
            logger.warning("Platform LiveService unavailable", exc_info=True)
            _live_service = None
    return _live_service


class _NoopPoller:
    """Fallback when the live stack is offline so the API still responds."""

    started_at: Optional[str] = None

    def get_active_fixtures(self) -> List[Dict[str, Any]]:
        return []

    def get_snapshot(self, fixture_id: int):
        return None

    def get_state(self, fixture_id: int):
        return None

    def get_snapshot_history(self, fixture_id: int):
        return []

    def get_health(self) -> Dict[str, Any]:
        return {
            "running": False,
            "started_at": None,
            "active_fixtures": 0,
            "subscribers": 0,
            "archived": 0,
            "poll_count": 0,
            "detail_poll_count": 0,
            "errors": 0,
            "has_live_engine": False,
            "has_rag_priors": False,
            "has_alert_engine": False,
            "has_live_store": False,
            "has_live_odds": False,
        }

    async def start(self):
        return None

    async def stop(self):
        return None

    def subscribe(self, *args, **kwargs):
        return None


# ---------------------------------------------------------------------------
# Pydantic schemas (kept identical to live_api.py)
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
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/live", tags=["live"])


@router.get("/fixtures", response_model=FixturesResponse)
async def list_fixtures():
    poller = _get_poller()
    fixtures = poller.get_active_fixtures()
    if not fixtures:
        service = _get_live_service()
        if service is not None:
            db_rows = service.list_active()
            # Adapt DB row shape → FixtureSummary where we can.
            fixtures = [{
                "fixture_id": r["fixture_id"],
                "home": r["home_team"],
                "away": r["away_team"],
                "score": "-",
                "elapsed": 0,
                "status": "scheduled",
                "league": r["league"],
                "league_id": 0,
            } for r in db_rows]
    return FixturesResponse(fixtures=[FixtureSummary(**f) for f in fixtures], count=len(fixtures))


@router.get("/fixtures/{fixture_id}")
async def get_fixture_snapshot(fixture_id: int):
    poller = _get_poller()
    state = poller.get_state(fixture_id)
    if state is not None:
        poller.subscribe(fixture_id)

    snapshot = poller.get_snapshot(fixture_id)
    if snapshot is None:
        # Fallback: serve the most recent persisted snapshot from Postgres.
        service = _get_live_service()
        if service is not None:
            persisted = service.latest_snapshot(fixture_id)
            if persisted is not None:
                return {
                    "fixture_id": fixture_id,
                    "source": "db_snapshot",
                    "snapshot": persisted["snapshot"],
                    "snapshot_ts": persisted["snapshot_ts"],
                    "elapsed_min": persisted["elapsed_min"],
                    "score": persisted["score"],
                }
        if state is not None:
            return {
                "fixture_id": fixture_id,
                "status": "pending",
                "message": "Fixture subscribed, waiting for first detail poll",
            }
        raise HTTPException(status_code=404, detail=f"Fixture {fixture_id} not found or not currently live")
    return snapshot.to_dict() if hasattr(snapshot, "to_dict") else snapshot


@router.get("/fixtures/{fixture_id}/state")
async def get_fixture_state(fixture_id: int):
    poller = _get_poller()
    state = poller.get_state(fixture_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Fixture {fixture_id} not found")
    return state.to_dict() if hasattr(state, "to_dict") else state


@router.get("/fixtures/{fixture_id}/history")
async def get_fixture_history(fixture_id: int):
    poller = _get_poller()
    history = poller.get_snapshot_history(fixture_id)
    if not history:
        # Serve persisted history from Postgres
        service = _get_live_service()
        if service is not None:
            db_history = service.snapshot_history(fixture_id, limit=20)
            if db_history:
                return {
                    "fixture_id": fixture_id,
                    "source": "db",
                    "snapshots": [row["snapshot"] for row in db_history],
                    "count": len(db_history),
                }
        state = poller.get_state(fixture_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Fixture {fixture_id} not found")
        return {"fixture_id": fixture_id, "snapshots": [], "count": 0}
    return {
        "fixture_id": fixture_id,
        "source": "poller",
        "snapshots": [s.to_dict() if hasattr(s, "to_dict") else s for s in history],
        "count": len(history),
    }


@router.get("/fixtures/{fixture_id}/alerts")
async def get_fixture_alerts(fixture_id: int, limit: int = 20):
    """New V2 endpoint — historical alerts sourced from Postgres."""
    service = _get_live_service()
    if service is None:
        raise HTTPException(status_code=503, detail="LiveService not available")
    alerts = service.recent_alerts(fixture_id, limit=limit)
    return {"fixture_id": fixture_id, "alerts": alerts, "count": len(alerts)}


@router.get("/stream")
async def live_stream(fixtures: str = Query(default="", description="Comma-separated fixture IDs; empty = all active")):
    poller = _get_poller()
    if fixtures.strip():
        fixture_ids = [int(x) for x in fixtures.split(",") if x.strip().isdigit()]
    else:
        fixture_ids = []

    async def event_generator() -> AsyncGenerator[str, None]:
        for fid in fixture_ids:
            poller.subscribe(fid)
        last_sent: Dict[int, str] = {}
        heartbeat_interval = 5
        while True:
            target_ids = fixture_ids if fixture_ids else [
                f["fixture_id"] for f in poller.get_active_fixtures()
            ]
            for fid in target_ids:
                snap = poller.get_snapshot(fid)
                if snap is None:
                    continue
                ts = getattr(snap, "timestamp", None)
                if ts != last_sent.get(fid):
                    data = json.dumps(snap.to_dict() if hasattr(snap, "to_dict") else snap, default=str)
                    yield f"event: snapshot\ndata: {data}\n\n"
                    last_sent[fid] = ts
                    for alert in getattr(snap, "alerts", []) or []:
                        yield f"event: alert\ndata: {json.dumps({'fixture_id': fid, **alert}, default=str)}\n\n"
            heartbeat = json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "active_fixtures": len(poller.get_active_fixtures()),
            })
            yield f"event: heartbeat\ndata: {heartbeat}\n\n"
            await asyncio.sleep(heartbeat_interval)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    poller = _get_poller()
    return HealthResponse(**poller.get_health())


@asynccontextmanager
async def live_lifespan(app):
    """Start/stop the LivePoller with the host app's lifecycle.

    Import in the main ``api.py`` lifespan if the process should also
    run the live poller (e.g. single-process deployment).
    """
    poller = _get_poller()
    try:
        await poller.start()
    except Exception:
        logger.exception("Failed to start LivePoller")
    yield
    try:
        await poller.stop()
    except Exception:
        logger.exception("Failed to stop LivePoller")
