"""Pick notes on the website's cards (tracker D75), behind the display switch ``SPIX_PICK_NOTES``.

Each Most likely and Value item of a card's ``recommendations.v1`` block gets ``note: {for, against}`` in the fixed
wording (``pick_notes.fixed_note``; the owner chose it over GPT, D75). The teams' finished matches are read once per
fixture and kept for ``KEEP``: they change only when the daily refresh adds results. A failed read leaves the card as
it was and is tried again on the next request. Nothing else on the card changes.
"""
from __future__ import annotations

from datetime import timedelta
import logging
import os
import threading
from typing import Any, Callable, Mapping, MutableMapping, Optional

from .. import pick_notes

logger = logging.getLogger(__name__)
KEEP = timedelta(hours=1)


def enabled() -> bool:
    return os.environ.get("SPIX_PICK_NOTES", "").strip().lower() == "on"


class PickNotes:
    def __init__(self, lookup: Optional[Callable[[int], Optional[Mapping[str, Any]]]] = None, *, keep=KEEP):
        self._lookup, self._keep = lookup, keep
        self._known: dict = {}
        self._lock = threading.Lock()

    def _form(self, event_id: str, now) -> Mapping[str, Any]:
        with self._lock:
            hit = self._known.get(event_id)
            if hit and now - hit[0] < self._keep:
                return hit[1]
        form = None
        if event_id.isdigit():
            if self._lookup is None:
                from ..repositories.team_form import TeamFormRepository
                self._lookup = TeamFormRepository().for_fixture
            form = self._lookup(int(event_id))
        with self._lock:
            self._known = {key: value for key, value in self._known.items() if now - value[0] < self._keep}
            self._known[event_id] = (now, form or {})
        return form or {}

    def attach(self, card: MutableMapping[str, Any], *, now) -> None:
        """Add each pick's note to ``card`` in place; a card without picks, or whose form could not be read, is left
        as it was."""
        block = card.get("recommendations")
        if not isinstance(block, Mapping) or not (block.get("most_likely") or block.get("value")):
            return
        fixture = card.get("fixture") or {}
        try:
            form = self._form(str(fixture.get("event_id") or ""), now)
        except Exception as exc:  # the card must never fail for a note
            logger.warning("Pick notes skipped for %s (%s)", fixture.get("event_id"), type(exc).__name__)
            return
        facts = pick_notes.match_facts(block, {**fixture, "competition": form.get("competition")},
                                       {"home": form.get("home") or [], "away": form.get("away") or []})
        written = {pick["id"]: pick_notes.fixed_note(pick, facts["competition"]) for pick in facts["picks"]}
        for kind in ("most_likely", "value"):
            for index, item in enumerate(block.get(kind) or []):
                if f"{kind}-{index}" in written:
                    item["note"] = written[f"{kind}-{index}"]
