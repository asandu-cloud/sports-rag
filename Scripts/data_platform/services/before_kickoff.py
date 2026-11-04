"""What a started match showed before kickoff (tracker D70): its Most likely and Value picks with their chances.

The entry carries no price, no "Worth it at" and nothing to add to a slip, so nothing after kickoff reads as a bet
to place now. The board lists these only when the display switch ``SPIX_BEFORE_KICKOFF`` is ``on``.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

VERSION = "before-kickoff.v1"
ITEM_KEYS = ("market", "side", "line", "label", "chance")


def enabled() -> bool:
    return os.environ.get("SPIX_BEFORE_KICKOFF", "").strip().lower() == "on"


def _items(values: Any, limit: int) -> list:
    rows = values if isinstance(values, list) else []
    return [{key: item.get(key) for key in ITEM_KEYS} for item in rows if isinstance(item, Mapping)][:limit]


def entry(card: Mapping[str, Any], *, read_at: Optional[str]) -> Optional[Dict[str, Any]]:
    """The board entry for one card shown at kickoff, or None when it had no recommendation to show."""
    block = card.get("recommendations") if isinstance(card.get("recommendations"), Mapping) else {}
    most_likely, value = _items(block.get("most_likely"), 3), _items(block.get("value"), 2)
    if not most_likely and not value:
        return None
    fixture = card.get("fixture") or {}
    visuals = card.get("visuals") or {}
    return {
        "version": VERSION,
        "fixture": {key: fixture.get(key) for key in ("event_id", "league", "home_team", "away_team", "kickoff")},
        "visuals": {"home_team_logo": visuals.get("home_team_logo"), "away_team_logo": visuals.get("away_team_logo")},
        "read_at": read_at,
        "most_likely": most_likely,
        "value": value,
        "status": None,
    }
