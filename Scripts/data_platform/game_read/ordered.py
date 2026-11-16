"""Game-read bench (D77): an ordered logit, for outcomes that come in order (away win, draw, home win; or a goal
difference from −4 to +4).

    P(class ≤ j) = 1 / (1 + exp(−(cut_j − x·beta)))

so a larger x moves the chances towards the higher classes, and the middle classes (the draw) are likeliest where
x sits between their cutpoints. Fitted by maximum likelihood with the exact gradient; the cutpoints are kept in
order by fitting the first one and the log of each step after it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

EPS = 1e-12


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
