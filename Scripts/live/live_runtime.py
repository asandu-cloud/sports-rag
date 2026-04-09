"""Shared live runtime helpers.

This keeps the live API and the live Discord bot aligned on the same runtime
configuration inside a process and centralizes the live-store wiring.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from live.live_poller import LivePoller
from live.live_store import LiveStore

_POLLER: Optional[LivePoller] = None


def _default_store_path() -> Path:
    configured = os.environ.get("LIVE_DB_PATH")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "Index" / "live.db"


def get_live_poller(auto_subscribe: bool = True) -> LivePoller:
    global _POLLER
    if _POLLER is None:
        store = LiveStore(_default_store_path())
        _POLLER = LivePoller(auto_subscribe=auto_subscribe, store=store)
    return _POLLER
