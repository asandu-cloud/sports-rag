"""Game-read bench (D77): the average of the models, model 9 (first version).

A weighted average of the members' forecasts on the log scale (a log-linear pool): each outcome's chance is the
product of the members' chances, each raised to its weight, then rescaled to sum to 1. The weights are never
negative and sum to 1, and they are fitted by maximum likelihood on tuning forecasts, separately for three parts
of the forecast:

- ``result``: win, draw or lose;
- ``handicap``: the goal difference, clipped to −4…+4 (it gives every half-line handicap);
- ``goals``: the full score matrix (it gives every goal line and both teams to score).

A member that does not give a part (Elo and pi-ratings give no score matrix) takes no part in it.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from Scripts.data_platform import market_goals
from Scripts.data_platform.game_read.forecast import Forecast, _differences
from Scripts.data_platform.game_read.history import Match

MARKETS = ("result", "handicap", "goals")
MEMBERS = ("elo", "pi", "dc", "drift")
MAX_DIFFERENCE, SIZE = 4, market_goals.MAX_GOALS + 1
EPS = 1e-12


def distribution(forecast: Forecast, market: str) -> Optional[np.ndarray]:
    """The forecast's chances for one part, as an array, or None when it does not give that part."""
    matrix = forecast.score_matrix
    if market == "result":
        return None if forecast.result is None else np.asarray(forecast.result, dtype=float)
    if market == "goals":
        return None if matrix is None else (matrix / matrix.sum()).ravel()
    differences = forecast.goal_difference or (None if matrix is None else _differences(matrix))
    if differences is None:
        return None
    clipped = np.zeros(2 * MAX_DIFFERENCE + 1)
    for d, p in differences.items():
        clipped[max(-MAX_DIFFERENCE, min(MAX_DIFFERENCE, d)) + MAX_DIFFERENCE] += p
    return clipped / clipped.sum()


def outcome(match: Match, market: str) -> int:
    home, away = match.home_goals, match.away_goals
    if market == "result":
        return 0 if home > away else 1 if home == away else 2
    if market == "handicap":
        return max(-MAX_DIFFERENCE, min(MAX_DIFFERENCE, home - away)) + MAX_DIFFERENCE
    return min(home, SIZE - 1) * SIZE + min(away, SIZE - 1)


def fit_weights(forecasts_by_model: Mapping[str, Sequence[Forecast]], matches: Sequence[Match],
                market: str) -> dict:
    """Weights for the members that give this part, fitted by maximum likelihood on the matches where every one of
    them gives it (a lean blend or the trees give no result where the market has no price)."""
    names = [name for name, forecasts in forecasts_by_model.items()
             if any(distribution(f, market) is not None for f in forecasts)]
    if len(names) <= 1:
        return {name: 1.0 for name in names}
    kept = [i for i in range(len(matches))
            if all(distribution(forecasts_by_model[name][i], market) is not None for name in names)]
    matches = [matches[i] for i in kept]
    logs = np.stack([np.log(np.clip([distribution(forecasts_by_model[name][i], market) for i in kept], EPS, None))
                     for name in names], axis=2)  # matches × outcomes × members
    rows, happened = np.arange(len(matches)), np.array([outcome(m, market) for m in matches])
    chosen = logs[rows, happened, :]

    def objective(theta):
        weights = softmax(theta)
        pooled = logs @ weights
        scale = logsumexp(pooled, axis=1)
        chances = np.exp(pooled - scale[:, None])
        by_weight = -(chosen - np.einsum("nk,nkm->nm", chances, logs)).mean(axis=0)
        return -(pooled[rows, happened] - scale).mean(), weights * (by_weight - weights @ by_weight)

    found = minimize(objective, np.zeros(len(names)), jac=True, method="L-BFGS-B")
    return dict(zip(names, softmax(found.x).tolist()))


def _pooled(forecasts: Mapping[str, Forecast], weights: Mapping[str, float], market: str) -> Optional[np.ndarray]:
    present = [(w, distribution(forecasts[name], market)) for name, w in weights.items() if name in forecasts]
    present = [(w, d) for w, d in present if d is not None and w > 0]
    if not present:
        return None
    total = sum(w for w, _ in present)
    pooled = sum(w / total * np.log(np.clip(d, EPS, None)) for w, d in present)
    chances = np.exp(pooled - pooled.max())
    return chances / chances.sum()


def combine(forecasts: Mapping[str, Forecast], weights: Mapping[str, Mapping[str, float]]) -> Forecast:
    """One match's average forecast from its members' forecasts, with weights by part."""
    result = _pooled(forecasts, weights.get("result", {}), "result")
    difference = _pooled(forecasts, weights.get("handicap", {}), "handicap")
    matrix = _pooled(forecasts, weights.get("goals", {}), "goals")
    return Forecast(result=None if result is None else tuple(result.tolist()),
                    goal_difference=None if difference is None else {
                        d - MAX_DIFFERENCE: float(p) for d, p in enumerate(difference)},
                    score_matrix=None if matrix is None else matrix.reshape(SIZE, SIZE))


def fit_all(rows_by_member: Mapping[str, Sequence[tuple]]) -> dict:
    """Weights for every part, from the members' rows (match, forecast) on the same matches."""
    matches = [match for match, _ in next(iter(rows_by_member.values()))]
    forecasts = {name: [forecast for _, forecast in rows] for name, rows in rows_by_member.items()}
    return {market: fit_weights(forecasts, matches, market) for market in MARKETS}


def combine_rows(rows_by_member: Mapping[str, Sequence[tuple]], weights: Mapping) -> list:
    """The average's rows from its members' rows, which must cover the same matches in the same order."""
    names = list(rows_by_member)
    first = rows_by_member[names[0]]
    if any([m.id for m, _ in rows_by_member[name]] != [m.id for m, _ in first] for name in names[1:]):
        raise ValueError("the members' rows do not cover the same matches")
    return [(match, combine({name: rows_by_member[name][i][1] for name in names}, weights))
            for i, (match, _) in enumerate(first)]


class Average:
    name = "average"

    def __init__(self, members: Sequence, weights: Mapping[str, Mapping[str, float]]):
        self.members, self.weights = list(members), weights

    def predict(self, matches: Sequence[Match]) -> list:
        by_member = {member.name: member.predict(matches) for member in self.members}
        return [combine({name: forecasts[i] for name, forecasts in by_member.items()}, self.weights)
                for i in range(len(matches))]

    def learn(self, matches: Sequence[Match]) -> None:
        for member in self.members:
            member.learn(matches)
