"""Game-read bench (D77): the zero line and the control line.

The zero line copies the market: its own chances on the real lines and M1's score matrix for the rest, so it must
score zero everywhere; if it does not, the bench is wrong. The control line is M1's goal means with Dixon–Coles's
low-score correction. M1 is a plain Poisson model and is known to be off on low scores, so a model that only fixes
that would "beat" the market on derived lines such as under 1.5 that real bookmakers already price right. On the
derived lines a model must beat the control line, not zero. Its rho is fitted by maximum likelihood on every
earlier match's M1 means and score, refitted at each season start; with nothing learned it is 0, plain M1.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from Scripts.data_platform import market_goals
from Scripts.data_platform.game_read import pool
from Scripts.data_platform.game_read.forecast import Forecast
from Scripts.data_platform.game_read.history import Match, Prices, market_handicap, market_over25, market_result

RHO_BOUNDS = (-0.3, 0.3)

_means = lru_cache(maxsize=200_000)(pool.m1_means)


def _result(matrix: np.ndarray) -> tuple:
    return float(np.tril(matrix, -1).sum()), float(np.trace(matrix)), float(np.triu(matrix, 1).sum())


class MarketCopy:
    name = "market"

    def predict(self, matches: Sequence[Match]) -> list:
        return [self._forecast(match.pre) for match in matches]

    def learn(self, matches: Sequence[Match]) -> None:
        pass

    @staticmethod
    def _forecast(prices: Prices) -> Forecast:
        means = _means(prices)
        if means is None:
            return Forecast()
        lines = {}
        over = market_over25(prices)
        if over:
            lines.update({("goals", "over", 2.5): over[0], ("goals", "under", 2.5): over[1]})
        handicap = market_handicap(prices)
        if handicap and pool._half(prices.handicap_line):
            line = float(prices.handicap_line)
            lines.update({("spreads", "home", line): handicap[0], ("spreads", "away", -line): handicap[1]})
        matrix = market_goals._matrix(*means)
        return Forecast(result=market_result(prices), score_matrix=matrix / matrix.sum(), lines=lines or None)


def _tau(low: np.ndarray, rho: float) -> np.ndarray:
    home, away, x, y = low.T
    return np.select([(x == 0) & (y == 0), (x == 1) & (y == 0), (x == 0) & (y == 1)],
                     [1 - home * away * rho, 1 + away * rho, 1 + home * rho], default=1 - rho)


def corrected(home: float, away: float, rho: float) -> np.ndarray:
    """The independent Poisson score matrix with Dixon–Coles's correction to 0-0, 1-0, 0-1 and 1-1, normalised."""
    matrix = market_goals._matrix(home, away)
    matrix[0, 0] *= 1 - home * away * rho
    matrix[1, 0] *= 1 + away * rho
    matrix[0, 1] *= 1 + home * rho
    matrix[1, 1] *= 1 - rho
    matrix = np.clip(matrix, 0, None)
    return matrix / matrix.sum()


def fit_rho(low: np.ndarray) -> float:
    """Dixon–Coles's rho by maximum likelihood. Only scores of 0 or 1 a side carry it: the correction keeps each
    match's chances summing to 1, so every other score's likelihood is the same whatever rho is."""
    if not len(low):
        return 0.0
    fit = minimize_scalar(lambda rho: -np.log(np.clip(_tau(low, rho), 1e-12, None)).sum(), bounds=RHO_BOUNDS,
                          method="bounded", options={"xatol": 1e-6})
    return float(fit.x)


class MarketControl:
    name = "control"

    def __init__(self):
        self.rho = 0.0
        self._season = None
        self._low = []  # (home mean, away mean, home goals, away goals) of learned matches with 0 or 1 a side

    def predict(self, matches: Sequence[Match]) -> list:
        season = max(match.season for match in matches)
        if season != self._season:
            self.rho, self._season = fit_rho(np.array(self._low, dtype=float).reshape(-1, 4)), season
        return [self._forecast(match.pre) for match in matches]

    def learn(self, matches: Sequence[Match]) -> None:
        for match in matches:
            if match.home_goals <= 1 and match.away_goals <= 1:
                means = _means(match.pre)
                if means is not None:
                    self._low.append((*means, match.home_goals, match.away_goals))

    def _forecast(self, prices: Prices) -> Forecast:
        means = _means(prices)
        if means is None:
            return Forecast()
        matrix = corrected(*means, self.rho)
        return Forecast(result=_result(matrix), score_matrix=matrix)
