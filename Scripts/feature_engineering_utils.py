"""Small, dependency-light helpers shared by legacy feature pipelines."""

from __future__ import annotations

from typing import Optional

import pandas as pd


# API-Football's per-fixture player feed currently uses one-letter role codes
# (G/D/M/F), while older exports can contain longer position labels.
DEFENSIVE_POSITION_CODES = frozenset({
    "D", "DF", "CB", "LB", "RB", "LWB", "RWB", "DEFENDER",
})


def defensive_position_flags(
    positions: Optional[pd.Series],
    *,
    index: pd.Index,
) -> pd.Series:
    """Return a 0/1 defensive-role flag without trusting source dtype.

    Legacy feature steps use broad numeric ``fillna(0)`` calls, so an absent
    source position can arrive here as ``None``, ``pd.NA``, ``NaN``, or the
    integer ``0``.  Normalising through pandas' nullable string dtype keeps
    those values safely non-defensive and makes this usable for a missing
    ``position`` column too.
    """
    if positions is None:
        positions = pd.Series(pd.NA, index=index, dtype="string")
    normalized = (
        positions.astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
    )
    return normalized.isin(DEFENSIVE_POSITION_CODES).astype("int8")
