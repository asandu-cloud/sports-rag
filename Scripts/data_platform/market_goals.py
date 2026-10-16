"""M1: the market's expected goals per team, solved from its fair prices (rework A3.2).

Finds the home and away goal means of an independent Poisson score model whose
probabilities reproduce the market's fair probabilities for the match result,
totals and Asian handicaps (any subset that identifies both teams). Fair
two-way prices for whole and quarter lines use the zero-expected-profit
condition: q = (win + ½·half-win) / (win + ½·half-win + loss + ½·half-loss),
pushes excluded.

The Dixon–Coles low-score correction is not applied here (rho is fitted per
league in A3.6); this gives the market's view of each team's scoring rate.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import poisson

MAX_GOALS = 12


def _matrix(home, away):
    goals = np.arange(MAX_GOALS + 1)
    return np.outer(poisson.pmf(goals, home), poisson.pmf(goals, away))


def result_probabilities(home, away):
    m = _matrix(home, away)
    total = m.sum()
    return [float(np.tril(m, -1).sum() / total), float(np.trace(m) / total), float(np.triu(m, 1).sum() / total)]


def total_pmf(home, away):
    m = _matrix(home, away)
    i, j = np.indices(m.shape)
    pmf = np.bincount((i + j).ravel(), m.ravel())
    return (pmf / pmf.sum()).tolist()


def _components(line):
    return [line] if math.isclose(line * 2, round(line * 2)) else [line - 0.25, line + 0.25]


def _profile(values, probabilities, line):
    profile = {"full_win": 0.0, "half_win": 0.0, "push": 0.0, "half_loss": 0.0, "full_loss": 0.0}
    names = {2: "full_win", 1: "half_win", 0: "push", -1: "half_loss", -2: "full_loss"}
    parts = _components(line)
    for value, p in zip(values, probabilities):
        value = float(value)
        score = sum(int(value + part > 0) - int(value + part < 0) for part in parts) * (2 // len(parts))
        profile[names[score]] += p
    return profile


def _two_way(profile):
    win = profile["full_win"] + 0.5 * profile["half_win"]
    loss = profile["full_loss"] + 0.5 * profile["half_loss"]
    return win / (win + loss)


def handicap_profile(home, away, line):
    """Settlement profile for the home side at handicap ``line`` (home gets ``line``)."""
    m = _matrix(home, away)
    i, j = np.indices(m.shape)
    return _profile((i - j).ravel(), (m / m.sum()).ravel(), line)


def handicap_probability(home, away, line):
    return _two_way(handicap_profile(home, away, line))


def over_probability(home, away, line):
    pmf = total_pmf(home, away)
    return _two_way(_profile(range(len(pmf)), pmf, -line))


def solve(targets):
    """{"home", "away", "error", "inputs"} or None when the prices cannot identify both teams.

    ``targets``: ``{"1x2": [p_home, p_draw, p_away], "totals": {line: p_over},
    "handicaps": {home_line: p_home}}`` (fair probabilities, any subset).
    """
    result = targets.get("1x2")
    totals = targets.get("totals") or {}
    handicaps = targets.get("handicaps") or {}
    inputs = (3 if result else 0) + len(totals) + len(handicaps)
    if inputs < 2 or not (result or handicaps):
        return None

    def residuals(log_means):
        home, away = np.exp(log_means)
        out = []
        if result:
            out += [a - b for a, b in zip(result_probabilities(home, away), result)]
        out += [over_probability(home, away, line) - p for line, p in totals.items()]
        out += [handicap_probability(home, away, line) - p for line, p in handicaps.items()]
        return out

    fit = least_squares(residuals, x0=np.log([1.4, 1.1]), bounds=(np.log(0.05), np.log(6.0)), xtol=1e-12, ftol=1e-12)
    home, away = np.exp(fit.x)
    return {"home": float(home), "away": float(away), "error": float(np.sum(np.square(fit.fun))), "inputs": inputs}


def poisson_pmf(mean, goals):
    return float(poisson.pmf(goals, mean))
