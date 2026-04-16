"""Structured platform health endpoint.

Returns the full :class:`HealthReport` JSON. A lightweight variant is
served at ``/api/health/platform?quick=1`` which skips the data-quality
sweep — useful for frequent liveness probes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Query

logger = logging.getLogger("web_app.health")

_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("/platform")
async def platform_health(quick: bool = Query(default=False, description="Skip data-quality scan if true")) -> Dict[str, Any]:
    try:
        from data_platform.observability import build_health_report
    except Exception as exc:
        return {"status": "error", "message": f"platform unavailable: {exc}"}
    report = build_health_report(include_data_quality=not quick)
    return report.as_dict()
