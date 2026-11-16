"""Game-read bench (D77): what a model says about one match, and its chance on any option of the pool.

A model fills what it can: the result's three chances, the chance of each goal difference (home minus away), or a
whole score matrix (13 by 13, home goals by away goals, summing to 1). ``lines`` holds chances a model gives for
named options directly; only the zero line uses it, to repeat the market's real lines exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from Scripts.data_platform.game_read.pool import _half, option_key

RESULT_SIDES = ("home", "draw", "away")


@dataclass
class Forecast:
    result: Optional[tuple] = None
    goal_difference: Optional[Mapping[int, float]] = None
    score_matrix: Optional[np.ndarray] = None
    lines: Optional[Mapping[tuple, float]] = None


def _differences(matrix: np.ndarray) -> dict:
    i, j = np.indices(matrix.shape)
    totals = np.bincount((i - j + matrix.shape[1] - 1).ravel(), matrix.ravel())
    return {d - (matrix.shape[1] - 1): float(p) for d, p in enumerate(totals)}


def chance(forecast: Forecast, option: Mapping) -> Optional[float]:
    """The forecast's chance that ``option`` wins: moneyline from ``result``; half-line handicaps from the goal
    difference or the matrix; goals and both teams to score from the matrix only. None when it says nothing."""
    key = option_key(option)
    if forecast.lines and key in forecast.lines:
        return float(forecast.lines[key])
    market, side, line = key
    if market == "moneyline":
        return None if forecast.result is None else float(forecast.result[RESULT_SIDES.index(side)])
    matrix = forecast.score_matrix
    if market == "spreads":
        differences = forecast.goal_difference or (None if matrix is None else _differences(matrix))
        if differences is None or not _half(line):
            return None
        total = sum(differences.values())
        won = sum(p for d, p in differences.items() if (d > -line if side == "home" else d < line))
        return won / total
    if matrix is None:
        return None
    total = float(matrix.sum())
    if market == "goals" and _half(line):
        i, j = np.indices(matrix.shape)
        over = float(matrix[(i + j) > line].sum()) / total
        return over if side == "over" else 1 - over
    if market == "btts":
        yes = float(matrix[1:, 1:].sum()) / total
        return yes if side == "yes" else 1 - yes
    return None
