"""FastAPI server for live match data.

V2: the live endpoints are defined in ``web_app/routers/live.py`` and are
mounted by the main API (``web_app/api.py``). This module stays as a
thin entry point so running ``python live_api.py`` (or uvicorn) still
brings up a standalone live service on port 8001 — useful when the live
stack is deployed as its own process.

Endpoints (unchanged from V1):
    GET /api/live/fixtures
    GET /api/live/fixtures/{id}
    GET /api/live/fixtures/{id}/state
    GET /api/live/fixtures/{id}/history
    GET /api/live/fixtures/{id}/alerts   — new in V2 (DB-backed)
    GET /api/live/stream                   (SSE)
    GET /api/live/health

Run:
    cd Scripts/live
    python live_api.py               # 0.0.0.0:8001
    uvicorn live_api:app --port 8001
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parents[1]
_PROJECT = _SCRIPTS.parent
for p in (str(_PROJECT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from web_app.routers.live import live_lifespan, router as live_router  # noqa: E402

logger = logging.getLogger("live_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


app = FastAPI(
    title="Matchwise Live API",
    description="Real-time live match projections powered by the Betting RAG engine (V2 shared router)",
    version="0.2.0",
    lifespan=live_lifespan,
)

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

app.include_router(live_router)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("LIVE_API_PORT", "8001"))
    logger.info("Starting Matchwise Live API on port %d", port)
    uvicorn.run("live_api:app", host="0.0.0.0", port=port, reload=False, log_level="info")
