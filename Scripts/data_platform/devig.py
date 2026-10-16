"""Remove a bookmaker's margin from decimal odds (rework A3.1).

Four methods, chosen between on history by ``Scripts/ops/devig_comparison.py``:

* ``proportional``: scale every implied probability by the same factor (today's method);
* ``additive``: subtract the same amount from every outcome (None if one would go to zero);
* ``power``: raise implied probabilities to the power k that makes them sum to one;
* ``shin``: Shin's model of insider trading, which takes more margin from longshots.

Power and Shin correct the favourite–longshot bias (bookmakers load more margin
onto longshots). With no margin every method returns the implied probabilities.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

METHODS = ("proportional", "additive", "power", "shin")
TOLERANCE = 1e-12


def _implied(odds: Sequence[float]) -> List[float]:
    values = [float(o) for o in odds]
    if len(values) < 2 or any(not math.isfinite(o) or o <= 1 for o in values):
        raise ValueError(f"decimal odds above 1 required, got {odds!r}")
    return [1 / o for o in values]


def _bisect(function, low, high, iterations=200):
    """Root of a decreasing function on [low, high]."""
    for _ in range(iterations):
        middle = (low + high) / 2
        if function(middle) > 0:
            low = middle
        else:
            high = middle
        if high - low < TOLERANCE:
            break
    return (low + high) / 2


def proportional(odds):
    implied = _implied(odds)
    total = sum(implied)
    return [p / total for p in implied]


def additive(odds) -> Optional[List[float]]:
    implied = _implied(odds)
    share = (sum(implied) - 1) / len(implied)
    fair = [p - share for p in implied]
    return fair if all(p > 0 for p in fair) else None


def power(odds):
    implied = _implied(odds)
    k = _bisect(lambda k: sum(p ** k for p in implied) - 1, 1e-6, 50.0)
    fair = [p ** k for p in implied]
    total = sum(fair)
    return [p / total for p in fair]


def shin(odds, *, with_z=False):
    implied = _implied(odds)
    booksum = sum(implied)

    def probabilities(z):
        return [(math.sqrt(z * z + 4 * (1 - z) * p * p / booksum) - z) / (2 * (1 - z)) for p in implied]

    z = 0.0 if booksum <= 1 else _bisect(lambda z: sum(probabilities(z)) - 1, 0.0, 0.999)
    fair = probabilities(z) if booksum > 1 else [p / booksum for p in implied]
    total = sum(fair)
    fair = [p / total for p in fair]
    return (fair, z) if with_z else fair


def fair(odds, method="proportional"):
    if method not in METHODS:
        raise ValueError(f"unknown margin-removal method {method!r}")
    return {"proportional": proportional, "additive": additive, "power": power, "shin": shin}[method](odds)
