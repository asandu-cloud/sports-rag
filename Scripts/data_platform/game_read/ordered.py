"""Game-read bench (D77): an ordered logit, for outcomes that come in order (away win, draw, home win; or a goal
difference from −4 to +4).

    P(class ≤ j) = 1 / (1 + exp(−(cut_j − x·beta)))

so a larger x moves the chances towards the higher classes, and the middle classes (the draw) are likeliest where
x sits between their cutpoints. Fitted by maximum likelihood with the exact gradient; the cutpoints are kept in
order by fitting the first one and the log of each step after it.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from Scripts.data_platform import market_goals
from Scripts.data_platform.game_read.forecast import Forecast, _differences

EPS = 1e-12
MAX_DIFFERENCE = 4
MIN_FIT = 200  # matches before the logits are fitted; until then a fixed prior


def _design(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.reshape(len(x), -1)


def _cumulative(cuts, beta, X) -> np.ndarray:
    """P(class ≤ j) for j = −1 … K−1, padded with 0 and 1: shape (n, K + 1)."""
    inner = expit(cuts[None, :] - (X @ beta)[:, None])
    return np.hstack([np.zeros((len(X), 1)), inner, np.ones((len(X), 1))])


@dataclass(frozen=True)
class OrderedLogit:
    cuts: np.ndarray
    beta: np.ndarray

    def probabilities(self, x) -> np.ndarray:
        p = np.diff(_cumulative(self.cuts, self.beta, _design(x)), axis=1)
        p = np.clip(p, EPS, None)
        return p / p.sum(axis=1, keepdims=True)


def _unpack(theta, classes):
    cuts = theta[0] + np.concatenate([[0.0], np.cumsum(np.exp(theta[1:classes - 1]))])
    return cuts, theta[classes - 1:]


def fit(x, y, classes: int) -> OrderedLogit:
    X, y = _design(x), np.asarray(y, dtype=int)
    n, rows = len(y), np.arange(len(y))

    def objective(theta):
        cuts, beta = _unpack(theta, classes)
        F = _cumulative(cuts, beta, X)
        f = F * (1 - F)  # density at each cutpoint; 0 at the padding
        p = np.clip(F[rows, y + 1] - F[rows, y], EPS, None)
        upper, lower = f[rows, y + 1] / p, f[rows, y] / p
        by_cut = np.zeros((n, classes + 1))
        by_cut[rows, y + 1] -= upper
        by_cut[rows, y] += lower
        grad_cuts = by_cut[:, 1:classes].sum(axis=0)
        grad_beta = ((upper - lower)[:, None] * X).sum(axis=0)
        steps = np.exp(theta[1:classes - 1])
        grad_steps = steps * np.cumsum(grad_cuts[::-1])[::-1][1:]
        return -np.log(p).sum(), np.concatenate([[grad_cuts.sum()], grad_steps, grad_beta])

    shares = np.bincount(y, minlength=classes)[:classes - 1].cumsum() / n
    start_cuts = np.log(np.clip(shares, 1e-3, 1 - 1e-3) / (1 - np.clip(shares, 1e-3, 1 - 1e-3)))
    start = np.concatenate([[start_cuts[0]], np.log(np.clip(np.diff(start_cuts), 1e-3, None)),
                            np.zeros(X.shape[1])])
    result = minimize(objective, start, jac=True, method="L-BFGS-B")
    cuts, beta = _unpack(result.x, classes)
    return OrderedLogit(cuts=cuts, beta=np.asarray(beta, dtype=float))


def _prior() -> Forecast:
    """A finite forecast before any fit: an average match's score model (1.45 and 1.15 goals)."""
    matrix = market_goals._matrix(1.45, 1.15)
    matrix = matrix / matrix.sum()
    clipped = defaultdict(float)
    for d, p in _differences(matrix).items():
        clipped[max(-MAX_DIFFERENCE, min(MAX_DIFFERENCE, d))] += p
    i, j = np.indices(matrix.shape)
    result = (float(matrix[i > j].sum()), float(matrix[i == j].sum()), float(matrix[i < j].sum()))
    return Forecast(result=result, goal_difference=dict(sorted(clipped.items())))


class Outcomes:
    """The result and the goal difference (clipped to −4…+4) from one number per match, such as a rating gap or an
    expected goal difference, through two ordered logits refitted on every match added so far."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self._seen = []
        self._result = self._difference = None

    def add(self, x: float, home_goals: int, away_goals: int) -> None:
        result = 2 if home_goals > away_goals else 1 if home_goals == away_goals else 0
        difference = max(-MAX_DIFFERENCE, min(MAX_DIFFERENCE, home_goals - away_goals)) + MAX_DIFFERENCE
        self._seen.append((x / self.scale, result, difference))

    def refit(self) -> None:
        if len(self._seen) < MIN_FIT:
            self._result = self._difference = None
            return
        seen = np.array(self._seen)
        self._result = fit(seen[:, 0], seen[:, 1].astype(int), 3)
        self._difference = fit(seen[:, 0], seen[:, 2].astype(int), 2 * MAX_DIFFERENCE + 1)

    def forecast(self, x: float) -> Forecast:
        if self._result is None:
            return _prior()
        point = np.array([x / self.scale])
        away, draw, home = self._result.probabilities(point)[0]
        return Forecast(result=(float(home), float(draw), float(away)),
                        goal_difference={d - MAX_DIFFERENCE: float(p)
                                         for d, p in enumerate(self._difference.probabilities(point)[0])})
