"""Shared prediction-settlement semantics.

The project has historically used both ``win``/``loss`` and ``hit``/``miss``
to mean the same thing.  The canonical store writes the latter, while some
older reports only understood the former.  This module is the one place that
defines those aliases and the financial treatment of Asian half-settlements.

Stored values are normalised at write time, but the helpers intentionally
continue to understand older values so existing SQLite/Postgres history stays
reportable without a data migration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional


HIT = "hit"
HALF_HIT = "half_hit"
MISS = "miss"
HALF_MISS = "half_miss"
PUSH = "push"
VOID = "void"

CANONICAL_OUTCOMES = frozenset({HIT, HALF_HIT, MISS, HALF_MISS, PUSH, VOID})

# Keep legacy values readable. ``won``/``lost`` are included because unit-bet
# tracking accepted them and they occasionally appear in manually corrected
# history.
OUTCOME_ALIASES = {
    "win": HIT,
    "full_win": HIT,
    "won": HIT,
    "half_win": HALF_HIT,
    "loss": MISS,
    "full_loss": MISS,
    "lost": MISS,
    "lose": MISS,
    "half_loss": HALF_MISS,
}

FULL_WIN_OUTCOMES = frozenset({HIT, "win", "full_win", "won"})
HALF_WIN_OUTCOMES = frozenset({HALF_HIT, "half_win"})
FULL_LOSS_OUTCOMES = frozenset({MISS, "loss", "full_loss", "lost", "lose"})
HALF_LOSS_OUTCOMES = frozenset({HALF_MISS, "half_loss"})
PUSH_OUTCOMES = frozenset({PUSH, VOID})


def normalize_outcome(value: Any) -> Optional[str]:
    """Return the canonical outcome label, or ``None`` for blank/unknown.

    Unknown values are deliberately not silently persisted as a valid result:
    that would contaminate hit-rate and ROI reporting.  Read paths can use
    this function to expose historical unknowns without failing.
    """
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in CANONICAL_OUTCOMES:
        return text
    return OUTCOME_ALIASES.get(text)


@dataclass(frozen=True)
class OutcomeAccounting:
    """One prediction's contribution to hit-rate and flat-stake ROI."""

    canonical: str
    stake_units: float
    returned_units: float
    resolved_units: float
    win_units: float


def outcome_accounting(outcome: Any, odds: Optional[float] = None) -> Optional[OutcomeAccounting]:
    """Return a one-unit settlement contribution for a recognised outcome.

    ``returned_units`` is zero if valid decimal odds were not recorded.  This
    keeps ROI honest: callers should exclude such rows from ROI stakes, while
    still being able to include them in hit-rate/calibration accounting.
    """
    canonical = normalize_outcome(outcome)
    if canonical is None:
        return None

    try:
        valid_odds = float(odds) if odds is not None else None
    except (TypeError, ValueError):
        valid_odds = None
    if valid_odds is not None and (not math.isfinite(valid_odds) or valid_odds <= 1.0):
        valid_odds = None

    if canonical == HIT:
        return OutcomeAccounting(HIT, 1.0, valid_odds or 0.0, 1.0, 1.0)
    if canonical == HALF_HIT:
        returned = (0.5 * valid_odds + 0.5) if valid_odds is not None else 0.0
        return OutcomeAccounting(HALF_HIT, 1.0, returned, 0.5, 0.5)
    if canonical == MISS:
        return OutcomeAccounting(MISS, 1.0, 0.0, 1.0, 0.0)
    if canonical == HALF_MISS:
        return OutcomeAccounting(HALF_MISS, 1.0, 0.5, 0.5, 0.0)
    # A void is a reported settlement, but makes no contribution to a
    # win/loss decision and returns the staked unit when odds were supplied.
    return OutcomeAccounting(canonical, 1.0, 1.0, 0.0, 0.0)


def is_valid_decimal_odds(value: Any) -> bool:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(odds) and odds > 1.0
