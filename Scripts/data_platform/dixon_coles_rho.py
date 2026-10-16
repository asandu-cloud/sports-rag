"""A3.6: fit the Dixon–Coles low-score correction (rho) on the engine's own goal means.

The engine scores goals with a Dixon–Coles matrix at a fixed rho = -0.10
(``core/projections.py``). Here rho is fitted by maximum likelihood on actual
scorelines given the engine's goal means, pooled or per league, with optional
shrinkage of league values toward the pooled one. Probabilities follow
``prob_models.dixon_coles_scoreline_prob`` exactly, and rho stays inside the
range where no low-score probability turns negative.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln


def _arrays(rows):
    h = np.array([r["home_mean"] for r in rows], dtype=float)
    a = np.array([r["away_mean"] for r in rows], dtype=float)
    x = np.array([r["home_goals"] for r in rows], dtype=int)
    y = np.array([r["away_goals"] for r in rows], dtype=int)
    return h, a, x, y


def _log_poisson(k, mean):
    return k * np.log(mean) - mean - gammaln(k + 1)


def _tau(x, y, h, a, rho):
    tau = np.ones_like(h)
    tau = np.where((x == 0) & (y == 0), 1 - h * a * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1 + a * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1 + h * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
    return tau


def nll(rows, rho):
    """Mean negative log-likelihood of the observed scorelines."""
    h, a, x, y = _arrays(rows)
    tau = np.clip(_tau(x, y, h, a, rho), 1e-300, None)
    return float(-(_log_poisson(x, h) + _log_poisson(y, a) + np.log(tau)).mean())


def valid_range(rows):
    h, a, _, _ = _arrays(rows)
    low = float(np.max(np.maximum(-1 / h, -1 / a)))
    high = float(min(1.0, np.min(1 / (h * a))))
    return low, high


def fit(rows):
    low, high = valid_range(rows)
    eps = 1e-6 * (high - low)
    result = minimize_scalar(lambda rho: nll(rows, rho), bounds=(low + eps, high - eps), method="bounded",
                             options={"xatol": 1e-7})
    return {"rho": float(result.x), "nll": float(result.fun), "n": len(rows), "bounds": [low, high]}


def shrink(league_rho, league_n, *, pooled, strength):
    """Pull a league's rho toward the pooled value; ``strength`` acts as pseudo-fixtures."""
    return pooled + (league_rho - pooled) * league_n / (league_n + strength)
