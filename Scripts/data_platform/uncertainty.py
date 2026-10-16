"""A3.4: count distributions whose width reflects how much evidence the mean rests on.

A match-total count X has mean μ estimated from roughly ``k`` matches. Its
intrinsic randomness is a negative binomial with dispersion α; the estimate of
μ itself is uncertain, modelled as gamma with relative variance 1/(κ·k).
Mixing the two gives

    Var X = μ + μ² · (α + (1 + α) / (κ·k))

so early-season totals get wider distributions and mature ones approach the
fixed-dispersion model, replacing the hand-set variance floors. α and κ are
fitted by maximum likelihood on history. The mean itself (including the
n/(n+8) early-season prior) is unchanged here; A6 replaces how means are made.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

PRIOR_STRENGTH = 8.0  # team_resolution.EARLY_SEASON_PRIOR_STRENGTH


def effective_matches(team):
    """The engine's effective sample: current matches, plus up to 8 prior ones while current < 8."""
    current, prior = float(team.get("current") or 0), float(team.get("prior") or 0)
    if current >= PRIOR_STRENGTH or prior <= 0:
        return max(current, 1.0)
    return current + min(prior, PRIOR_STRENGTH)


def total_variance(mean, alpha, matches, kappa=1.0):
    k = max(float(matches) * kappa, 1e-9)
    return mean + mean * mean * (alpha + (1 + alpha) / k)


def log_pmf(y, mean, variance):
    if variance <= mean * (1 + 1e-12):
        return float(y * math.log(mean) - mean - gammaln(y + 1))
    alpha = (variance - mean) / (mean * mean)
    r = 1 / alpha
    return float(gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * math.log(r / (r + mean)) + y * math.log(mean / (r + mean)))


def _nll(params, means, matches, ys):
    alpha, kappa = math.exp(params[0]), math.exp(params[1])
    variances = means + means ** 2 * (alpha + (1 + alpha) / (kappa * matches))
    a = (variances - means) / means ** 2
    r = 1 / a
    ll = gammaln(ys + r) - gammaln(r) - gammaln(ys + 1) + r * np.log(r / (r + means)) + ys * np.log(means / (r + means))
    return -float(ll.sum())


def fit(rows):
    means = np.array([r["mean"] for r in rows], dtype=float)
    matches = np.array([r["matches"] for r in rows], dtype=float)
    ys = np.array([r["y"] for r in rows], dtype=float)
    result = minimize(_nll, x0=np.log([0.02, 1.0]), args=(means, matches, ys), method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
    alpha, kappa = np.exp(result.x)
    return {"alpha": float(alpha), "kappa": float(kappa), "nll": float(result.fun) / len(rows), "n": len(rows),
            "converged": bool(result.success)}
