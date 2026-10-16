"""A5.2: selection policies replayed on the audited lines (tracker D30).

Every policy sees the same rows (``disagreement.build(..., all_sides=True)``): each
side of each evaluated line in the latest pre-kickoff read per fixture and stage,
with Spix's probability, the market consensus (power margin removal, median across
books), the best price on offer and, where saved, the closing consensus. A policy
gives each side a published probability (or rules it out); a side is a candidate
when its expected value at the best price reaches ``min_ev``. Within a read, the
best candidate per market is kept and the three best markets are published, the
compiler's cap of three per fixture.

``current`` replays the engine's own choice (its selector's pick in each market),
ranked by Spix's expected value; the compiler's confidence-then-edge order is not
replayed exactly.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .disagreement import _logit

PER_FIXTURE = 3


def tilted(c):
    """Market probability moved toward Spix by ``c`` in log-odds (c = 0: market, c = 1: Spix)."""
    def probability(row):
        x = float(_logit(row["p_ref"]) + c * (_logit(row["p_spix"]) - _logit(row["p_ref"])))
        return 1 / (1 + np.exp(-x))
    return probability


@dataclass(frozen=True)
class Policy:
    name: str
    probability: Callable[[dict], float]
    selected_only: bool = False
    max_disagreement: Optional[float] = None


def policies():
    spix = lambda r: r["p_spix"]
    return [Policy("current", spix, selected_only=True),
            Policy("spix_ev", spix),
            Policy("spix_ev_capped_10", spix, max_disagreement=0.10),
            Policy("market", tilted(0.0)),
            Policy("tilt_0.03", tilted(0.03)),
            Policy("tilt_0.10", tilted(0.10)),
            Policy("tilt_0.25", tilted(0.25))]


def pick(rows, policy, *, min_ev=0.02, per_fixture=PER_FIXTURE):
    """The published picks: (row, published probability, expected value at the best price)."""
    best = {}
    for r in rows:
        if not r.get("best_odds") or (policy.selected_only and not r["selected"]):
            continue
        if policy.max_disagreement is not None and abs(r["p_spix"] - r["p_ref"]) > policy.max_disagreement:
            continue
        p = float(policy.probability(r))
        ev = p * r["best_odds"] - 1
        if ev < min_ev:
            continue
        key = (r["fixture_id"], r["stage"], r["market"])
        if key not in best or ev > best[key][2]:
            best[key] = (r, p, ev)
    by_read = defaultdict(list)
    for (fixture, stage, _), choice in best.items():
        by_read[(fixture, stage)].append(choice)
    return [c for choices in by_read.values() for c in sorted(choices, key=lambda c: -c[2])[:per_fixture]]


def evaluate(picks):
    """Stated vs market vs actual, the value claimed, and the value left at the close."""
    if not picks:
        return {"picks": 0}
    rows = [r for r, _, _ in picks]
    w = np.array([r["w"] for r in rows])
    stated = np.array([p for _, p, _ in picks])
    closed = [(r, p) for r, p, _ in picks if r.get("p_close") is not None]
    out = {"picks": len(picks), "fixtures": len({r["fixture_id"] for r in rows}),
           "stated": float((w * stated).sum() / w.sum()),
           "market": float((w * [r["p_ref"] for r in rows]).sum() / w.sum()),
           "actual": float((w * [r["y"] for r in rows]).sum() / w.sum()),
           "claimed_ev": float(np.mean([ev for _, _, ev in picks])),
           "return": float(np.mean([r["w"] * (r["y"] * r["best_odds"] - 1) for r in rows])),
           "closing_picks": len(closed)}
    if closed:
        out["ev_at_close"] = float(np.mean([r["p_close"] * r["best_odds"] - 1 for r, _ in closed]))
        out["fair_clv"] = float(np.mean([r["p_close"] / r["p_ref"] - 1 for r, _ in closed]))
    out["best_book_at_close"] = best_book_moves(rows)
    return out


def best_book_moves(rows):
    """How the best-price bookmaker's own price stood at the close: shorter (the price was real and was
    corrected), the same (stable, or a stale quote), longer, or not quoted."""
    counts = {"shorter": 0, "same": 0, "longer": 0, "missing": 0}
    for r in rows:
        if r.get("p_close") is None:
            continue
        close = r.get("best_book_close_odds")
        counts["missing" if close is None else "shorter" if close < r["best_odds"] - 1e-9
               else "longer" if close > r["best_odds"] + 1e-9 else "same"] += 1
    return counts


def ev_at_close(picks):
    closed = [r for r, _, _ in picks if r.get("p_close") is not None]
    return float(np.mean([r["p_close"] * r["best_odds"] - 1 for r in closed])) if closed else None


def stated_gap(picks):
    """Weighted actual minus stated (negative: overconfident)."""
    if not picks:
        return None
    w = np.array([r["w"] for r, _, _ in picks])
    return float((w * ([r["y"] for r, _, _ in picks] - np.array([p for _, p, _ in picks]))).sum() / w.sum())
