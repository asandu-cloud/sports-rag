"""A5.3 checks (tracker D44): do the confidence labels separate picks that keep value at the close?

The picks are the market-price policy's (A5.2, D31: the market's consensus probability, at least 2% value at the best
price, one per market and three per fixture) on the audited reads. Two checks:

* firmness: picks whose line at least ``min_books`` bookmakers price two-sided, with fair probabilities within a
  spread cut-off (high), against the rest (low). The cut-off is chosen on the earlier weeks and confirmed on the last
  week, and each label must keep at least ``min_share`` of the picks;
* Spix, per market: picks where Spix's own probability also gives the minimum value at the price, against those where
  it does not.

Both compare value at the close (the closing consensus times the price, minus one; it does not depend on results)
and happened minus stated, with 95% intervals that resample whole fixtures. A label passes when high keeps more value
at the close (interval above zero) and is not worse calibrated (the difference's interval reaches zero or above).
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from . import selection_policies as sp
from .calibration_data import _time
from .confidence_labels import MIN_VALUE
from .disagreement import fixture_bootstrap

PERCENTILES = (20, 30, 40, 50, 60, 70, 80)


def pick_rows(rows, *, min_value=MIN_VALUE):
    """The market-price policy's picks, each with the evidence the labels use and its value at the close."""
    market = next(p for p in sp.policies() if p.name == "market")
    out = []
    for r, p, _ in sp.pick(rows, market, min_ev=min_value):
        odds = r["best_odds"]
        out.append({"fixture_id": r["fixture_id"], "stage": r["stage"], "market": r["market"], "kickoff": r["kickoff"],
                    "odds": odds, "p": float(p), "y": r["y"], "w": r["w"], "book_count": r.get("book_count"),
                    "book_spread": r.get("book_spread"), "spix_value": r["p_spix"] * odds - 1,
                    "value_at_close": None if r.get("p_close") is None else r["p_close"] * odds - 1})
    return out


def split_by_week(picks, *, holdout_days=7):
    """(earlier picks, picks kicking off in the last ``holdout_days`` before the latest kickoff)."""
    dated = [(p, _time(p["kickoff"])) for p in picks]
    dated = [(p, t) for p, t in dated if t is not None]
    if not dated:
        return [], []
    start = max(t for _, t in dated) - timedelta(days=holdout_days)
    return [p for p, t in dated if t <= start], [p for p, t in dated if t > start]


def _high(p, cutoff, min_books):
    return (p.get("book_count") or 0) >= min_books and p.get("book_spread") is not None and p["book_spread"] <= cutoff


def split_labels(picks, cutoff, *, min_books=4):
    """(high, low) at a spread cut-off."""
    return ([p for p in picks if _high(p, cutoff, min_books)], [p for p in picks if not _high(p, cutoff, min_books)])


def _value(picks):
    closed = [p["value_at_close"] for p in picks if p.get("value_at_close") is not None]
    return float(np.mean(closed)) if closed else None


def _weighted(picks, key):
    w = sum(p["w"] for p in picks)
    return sum(p["w"] * p[key] for p in picks) / w if picks and w > 0 else None


def _gap(picks):
    happened, stated = _weighted(picks, "y"), _weighted(picks, "p")
    return None if happened is None else happened - stated


def _difference(statistic):
    def between(sample):
        a = statistic([p for p in sample if p["_high"]])
        b = statistic([p for p in sample if not p["_high"]])
        return None if a is None or b is None else a - b
    return between


def _summary(picks):
    return {"picks": len(picks), "fixtures": len({p["fixture_id"] for p in picks}), "value_at_close": _value(picks),
            "stated": _weighted(picks, "p"), "happened": _weighted(picks, "y")}


def compare(high, low, *, resamples=1000):
    """High against low: value at the close and happened minus stated, with fixture-resampled 95% intervals."""
    tagged = [{**p, "_high": True} for p in high] + [{**p, "_high": False} for p in low]
    value, calibration = _difference(_value), _difference(_gap)
    return {"high": _summary(high), "low": _summary(low),
            "value_diff": value(tagged), "value_diff_interval95": fixture_bootstrap(tagged, value, resamples=resamples),
            "calibration_diff": calibration(tagged),
            "calibration_diff_interval95": fixture_bootstrap(tagged, calibration, resamples=resamples)}


def passes(comparison) -> bool:
    low_value, _ = comparison["value_diff_interval95"]
    _, high_calibration = comparison["calibration_diff_interval95"]
    return low_value is not None and high_calibration is not None and low_value > 0 and high_calibration >= 0


def choose_spread_cutoff(train, *, min_books=4, min_share=0.2):
    """The spread cut-off (a percentile of the train picks' spreads) that best separates value at the close."""
    spreads = [p["book_spread"] for p in train if (p.get("book_count") or 0) >= min_books and p.get("book_spread") is not None]
    if not spreads:
        return None
    best = None
    for q in PERCENTILES:
        cutoff = float(np.percentile(spreads, q))
        high, low = split_labels(train, cutoff, min_books=min_books)
        if min(len(high), len(low)) < min_share * len(train):
            continue
        a, b = _value(high), _value(low)
        if a is not None and b is not None and (best is None or a - b > best[0]):
            best = (a - b, cutoff)
    return None if best is None else best[1]


def spix_check(picks, *, min_value=MIN_VALUE, min_picks=30, resamples=1000):
    """Per market: do picks where Spix agrees keep more value at the close than those where it does not?"""
    out = {}
    for market in sorted({p["market"] for p in picks}):
        mine = [p for p in picks if p["market"] == market]
        agree = [p for p in mine if p["spix_value"] >= min_value]
        disagree = [p for p in mine if p["spix_value"] < min_value]
        entry = {"agree": len(agree), "disagree": len(disagree), "passes": False}
        if min(len(agree), len(disagree)) >= min_picks:
            entry.update(compare(agree, disagree, resamples=resamples))
            entry["passes"] = passes(entry)
        out[market] = entry
    return out
