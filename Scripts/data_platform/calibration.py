"""A4.2: per-market Platt calibration of Spix's own probabilities (tracker D27).

Scope is the markets where Spix publishes its own probability: corners, cards, BTTS,
goal totals other than 2.5 and Asian handicap. 1X2 and O/U 2.5 publish the market's
probability (D21) and are left out.

Each line is turned to one reference side (over, home team, yes) before fitting.
Complementary sides carry mirrored data, so a fit on both sides forces the
intercept to zero and cannot see a directional bias such as overs stated too high;
on one side the intercept can correct it. The other side is the complement.

Corners are priced with the fixed dispersion that goes live at A3.7 stage 1
(``FIXED_DISPERSIONS``) from each result's stored mean, so the calibrator is fitted
on the probabilities it will be applied to (D27 condition 2).
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from Scripts.rag_ingest.core.prediction_guardrails import FIXED_DISPERSIONS
from Scripts.rag_ingest.prob_models import asian_total_equivalent_probability, asian_total_settlement_profile

from .scorecard import _calibration_error, calibration_fit

SCOPE = ("btts", "cards", "corners", "goals", "spreads")
MARKET_ANCHORED_GOAL_LINE = 2.5
TOTALS = {"goals", "corners", "cards"}
EPS = 1e-6


def in_scope(selection):
    market = selection["market"]
    if market not in SCOPE:
        return False
    return not (market == "goals" and selection["point"] is not None
                and abs(float(selection["point"]) - MARKET_ANCHORED_GOAL_LINE) < 1e-9)


def live_probability(selection):
    """The probability the selection will carry once the stage 1 dispersion is live."""
    alpha = FIXED_DISPERSIONS.get(selection["market"])
    mean = selection.get("mean")
    if alpha is None or mean is None or float(mean) <= 0:
        return selection["model_prob"]
    mean = float(mean)
    profile = asian_total_settlement_profile(mean, float(selection["point"]), str(selection["side"]).lower(),
                                             mean + alpha * mean * mean)
    return asian_total_equivalent_probability(profile)


def reference_side(selection, probability):
    """(reference line, probability, outcome) on the over / home / yes side, or None."""
    market, side, point, y = selection["market"], str(selection["side"]).casefold(), selection["point"], selection["y"]
    if market in TOTALS:
        flip = {"over": False, "under": True}.get(side)
        line = None if point is None else float(point)
    elif market == "btts":
        flip, line = {"yes": False, "no": True}.get(side), None
    elif market == "spreads":
        home, away = str(selection.get("home") or "").casefold(), str(selection.get("away") or "").casefold()
        flip = False if side == home else True if side == away else None
        line = None if point is None or flip is None else (-float(point) if flip else float(point))
    else:
        return None
    if flip is None or (market != "btts" and line is None):
        return None
    return (line, 1.0 - probability, 1.0 - y) if flip else (line, probability, y)


def canonical(dataset, *, live_dispersion=True):
    """One row per fixture, stage, market and reference line; ``selected`` if either side was picked."""
    rows = {}
    for s in dataset:
        if not in_scope(s):
            continue
        probability = live_probability(s) if live_dispersion else s["model_prob"]
        ref = reference_side(s, probability)
        if ref is None:
            continue
        line, p, y = ref
        market_ref = None if s.get("market_prob") is None else reference_side(s, float(s["market_prob"]))[1]
        key = (s["fixture_id"], s["stage"], s["market"], line)
        if key in rows:
            rows[key]["selected"] = rows[key]["selected"] or bool(s.get("selected"))
            if rows[key]["market_p"] is None:
                rows[key]["market_p"] = market_ref
            continue
        rows[key] = {"fixture_id": s["fixture_id"], "stage": s["stage"], "kickoff": str(s["kickoff"]),
                     "market": s["market"], "line": line, "p": p, "y": y, "w": s["w"],
                     "selected": bool(s.get("selected")), "market_p": market_ref}
    return list(rows.values())


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Platt:
    """p' = sigmoid(intercept + slope * logit(p)), fitted by weighted maximum likelihood."""
    name = "platt"

    def fit(self, rows):
        self.intercept, self.slope = calibration_fit([(r["y"], r["w"], r["p"]) for r in rows])
        if self.intercept is None:
            self.intercept, self.slope = 0.0, 1.0
        return self

    def predict(self, p):
        return 1 / (1 + np.exp(-(self.intercept + self.slope * _logit(p))))


class Isotonic:
    """Weighted isotonic regression (challenger; D27: only with ~1,000 settled forecasts)."""
    name = "isotonic"

    def fit(self, rows):
        from sklearn.isotonic import IsotonicRegression
        self.model = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(
            [r["p"] for r in rows], [r["y"] for r in rows], sample_weight=[r["w"] for r in rows])
        return self

    def predict(self, p):
        return self.model.predict(np.asarray(p, dtype=float))


def metrics(y, w, p):
    y, w, p = (np.asarray(v, dtype=float) for v in (y, w, p))
    q = np.clip(p, EPS, 1 - EPS)
    total = w.sum()
    intercept, slope = calibration_fit(list(zip(y.tolist(), w.tolist(), p.tolist())))
    return {"weight": float(total),
            "log_loss": float((w * -(y * np.log(q) + (1 - y) * np.log(1 - q))).sum() / total),
            "brier": float((w * (p - y) ** 2).sum() / total),
            "stated": float((w * p).sum() / total), "actual": float((w * y).sum() / total),
            "calibration_error": _calibration_error(list(zip(y.tolist(), w.tolist(), p.tolist()))),
            "intercept": intercept, "slope": slope}


def _day(row):
    return row["kickoff"][:10]


def rolling_origin(rows, make, *, min_train_fixtures=40):
    """Out-of-sample probabilities: each kickoff day is predicted by a fit on the days before it.

    Returns (test rows, calibrated probabilities); days with fewer earlier fixtures than
    ``min_train_fixtures`` are not tested.
    """
    tested, predicted = [], []
    for day in sorted({_day(r) for r in rows}):
        train = [r for r in rows if _day(r) < day]
        if len({r["fixture_id"] for r in train}) < min_train_fixtures:
            continue
        test = [r for r in rows if _day(r) == day]
        model = make().fit(train)
        tested.extend(test)
        predicted.extend(model.predict([r["p"] for r in test]).tolist())
    return tested, predicted


def cluster_interval(rows, loss_a, loss_b, *, resamples=2000, seed=20260928):
    """95% interval of the weighted mean of (a - b), resampling whole fixtures."""
    sums, weights = defaultdict(float), defaultdict(float)
    for r, a, b in zip(rows, loss_a, loss_b):
        sums[r["fixture_id"]] += r["w"] * (a - b)
        weights[r["fixture_id"]] += r["w"]
    keys = sorted(sums)
    s, w = np.array([sums[k] for k in keys]), np.array([weights[k] for k in keys])
    draws = np.random.default_rng(seed).integers(0, len(keys), size=(resamples, len(keys)))
    return np.quantile(s[draws].sum(axis=1) / w[draws].sum(axis=1), [.025, .975]).tolist()


def row_log_loss(rows, probabilities):
    return [-(r["y"] * math.log(min(max(p, EPS), 1 - EPS)) + (1 - r["y"]) * math.log(1 - min(max(p, EPS), 1 - EPS)))
            for r, p in zip(rows, probabilities)]
