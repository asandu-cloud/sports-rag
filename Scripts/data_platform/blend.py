"""A3.3: a fitted logarithmic pool of the market and Spix.

    p_k ∝ exp(a_k) · market_k^b · spix_k^c

For a two-way market this is ``logit p = a + b·logit p_market + c·logit p_spix``,
the blend in the rework design. ``b`` rescales the market (calibration),
``c`` is the weight Spix earns once the market is known; ``use_spix=False``
fits the market-only recalibration (c = 0) that isolates Spix's contribution.
Fitted by maximum likelihood with the exact gradient; an optional L2 penalty
shrinks ``c`` toward zero for markets with little history.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

EPS = 1e-12


def _logs(probabilities):
    return np.log(np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0))


def _unpack(theta, classes, use_spix):
    intercepts = np.append(theta[:classes - 1], 0.0)
    b = theta[classes - 1]
    c = theta[classes] if use_spix else 0.0
    return intercepts, b, c


def _probabilities(intercepts, b, c, log_market, log_spix):
    logits = intercepts[None, :] + b * log_market + c * log_spix
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def fit(market, spix, outcomes, *, use_spix=True, spix_penalty=0.0):
    log_market, log_spix = _logs(market), _logs(spix)
    y = np.asarray(outcomes, dtype=int)
    n, classes = log_market.shape
    onehot = np.eye(classes)[y]

    def objective(theta):
        intercepts, b, c = _unpack(theta, classes, use_spix)
        p = _probabilities(intercepts, b, c, log_market, log_spix)
        nll = -np.log(np.clip(p[np.arange(n), y], EPS, 1)).sum() + spix_penalty * c * c
        residual = onehot - p
        grad = [-residual[:, k].sum() for k in range(classes - 1)]
        grad.append(-(residual * log_market).sum())
        if use_spix:
            grad.append(-(residual * log_spix).sum() + 2 * spix_penalty * c)
        return nll, np.array(grad)

    start = np.zeros(classes - 1 + (2 if use_spix else 1))
    start[classes - 1] = 1.0
    result = minimize(objective, start, jac=True, method="L-BFGS-B")
    intercepts, b, c = _unpack(result.x, classes, use_spix)
    return {"intercepts": intercepts.tolist(), "b": float(b), "c": float(c), "nll": float(result.fun) / n,
            "n": int(n), "use_spix": use_spix, "converged": bool(result.success)}


def predict(model, market, spix):
    return _probabilities(np.asarray(model["intercepts"], dtype=float), model["b"], model["c"],
                          _logs(market), _logs(spix))
