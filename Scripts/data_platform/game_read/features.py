"""Game-read bench (D77, phase 2): the features pass, every match's out-of-sample inputs for models 5b to 7.

``Features`` keeps its rating models (``models``, by name) and ``variables``, and asks them about every day from the
season ``start`` (2013: one warm-up season after the first), each day once, before the day's results are learned.
Days the runner only lets it learn are recorded first, then learned. So every match from ``start`` has the members'
forecasts as they would have stood that morning, and ``history`` holds the recorded matches in order, results known.

``row(match)`` turns a recorded match into numbers (None where a part is missing):

- each member's result chances, its chance of over 2.5 and its expected goals (from its score matrix);
- the market's result chances, over 2.5 and main handicap (line and home chance), and the bookmakers' margin;
- the lead's inputs: the market's log chance minus each member's, on the result and over 2.5;
- each variable's values for the home and away side (``<name>_home_<i>``);
- league, month and weekday.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np

from Scripts.data_platform.game_read.forecast import RESULT_SIDES, Forecast, chance
from Scripts.data_platform.game_read.history import Match, market_handicap, market_over25, market_result

FEATURES_FROM = 2013
OVER = {"market": "goals", "side": "over", "line": 2.5}


def result_chances(forecast: Forecast) -> Optional[list]:
    found = [chance(forecast, {"market": "moneyline", "side": side, "line": None}) for side in RESULT_SIDES]
    return None if None in found else found


def over_chance(forecast: Forecast) -> Optional[float]:
    return chance(forecast, OVER)


def _log_gap(market: Optional[float], member: Optional[float]) -> Optional[float]:
    if market is None or member is None or market <= 0 or member <= 0:
        return None
    return math.log(market) - math.log(member)


class Features:
    def __init__(self, models: Mapping, variables: Sequence = (), start: int = FEATURES_FROM):
        self.models, self.variables, self.start = dict(models), tuple(variables), start
        self.forecasts = {}  # match id -> {member name: Forecast}
        self.values = {}  # match id -> [(home values, away values) per variable]
        self.history = []  # recorded matches, results learned, in order

    def predict(self, matches: Sequence[Match]) -> None:
        """Record each member's forecast and each variable's values, once per match, from ``start``."""
        todo = [m for m in matches if m.season >= self.start and m.id not in self.forecasts]
        if not todo:
            return
        said = {name: model.predict(todo) for name, model in self.models.items()}
        for i, match in enumerate(todo):
            self.forecasts[match.id] = {name: forecasts[i] for name, forecasts in said.items()}
            self.values[match.id] = [variable.features(match) for variable in self.variables]

    def learn(self, matches: Sequence[Match]) -> None:
        self.predict(matches)
        for model in self.models.values():
            model.learn(matches)
        for variable in self.variables:
            variable.learn(matches)
        self.history += [m for m in matches if m.id in self.forecasts]

    def row(self, match: Match) -> dict:
        prices = match.pre
        market = market_result(prices)
        over = market_over25(prices)
        handicap = market_handicap(prices)
        found = {"league": match.competition, "month": match.day.month, "weekday": match.day.weekday(),
                 "margin": None if prices.result_avg is None else sum(1 / p for p in prices.result_avg) - 1,
                 "market_over": None if over is None else over[0],
                 "market_line": prices.handicap_line, "market_cover": None if handicap is None else handicap[0]}
        for i, side in enumerate(RESULT_SIDES):
            found[f"market_{side}"] = None if market is None else market[i]
        for name, forecast in self.forecasts[match.id].items():
            result, member_over = result_chances(forecast), over_chance(forecast)
            matrix = forecast.score_matrix
            for i, side in enumerate(RESULT_SIDES):
                found[f"{name}_{side}"] = None if result is None else result[i]
                found[f"lead_{name}_{side}"] = _log_gap(found[f"market_{side}"], found[f"{name}_{side}"])
            found[f"{name}_over"] = member_over
            found[f"lead_{name}_over"] = _log_gap(found["market_over"], member_over)
            if matrix is None:
                found[f"{name}_home_goals"] = found[f"{name}_away_goals"] = None
            else:
                goals = np.arange(matrix.shape[0])
                total = float(matrix.sum())
                found[f"{name}_home_goals"] = float(matrix.sum(axis=1) @ goals) / total
                found[f"{name}_away_goals"] = float(matrix.sum(axis=0) @ goals) / total
        for variable, (home, away) in zip(self.variables, self.values[match.id]):
            for i, value in enumerate(home):
                found[f"{variable.name}_home_{i}"] = value
            for i, value in enumerate(away):
                found[f"{variable.name}_away_{i}"] = value
        return found
