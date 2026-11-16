"""Game-read bench, task 12: the average of the models (model 9, first version) (D77)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from Scripts.data_platform import market_goals
from Scripts.data_platform.game_read.forecast import Forecast, chance
from Scripts.data_platform.game_read.history import Match, Prices
from Scripts.data_platform.game_read.models.average import Average, combine, fit_weights


def match(id, goals):
    return Match(id=id, competition="EPL", country="X", season=2019, day=date(2019, 8, 10), home="X:A", away="X:B",
                 home_goals=goals[0], away_goals=goals[1], pre=Prices(), close=Prices())


def simulated(n=4000, seed=11):
    """Matches whose result follows a known chance per match, and three members' forecasts of it: one that knows
    the chance, one that always says a third each, and one that gets home and away the wrong way round."""
    rng = np.random.default_rng(seed)
    matches, truth = [], []
    for i in range(n):
        home = rng.uniform(0.2, 0.7)
        draw = rng.uniform(0.15, 0.3) * (1 - home) / 0.7
        p = np.array([home, draw, 1 - home - draw])
        outcome = rng.choice(3, p=p)
        matches.append(match(i, [(1, 0), (1, 1), (0, 1)][outcome]))
        truth.append(p)
    forecasts = {"perfect": [Forecast(result=tuple(p)) for p in truth],
                 "flat": [Forecast(result=(1 / 3, 1 / 3, 1 / 3)) for _ in truth],
                 "reversed": [Forecast(result=tuple(p[::-1])) for p in truth]}
    return matches, forecasts


def test_one_perfect_member_takes_all_the_weight():
    matches, forecasts = simulated()
    weights = fit_weights(forecasts, matches, "result")
    assert weights["perfect"] > 0.9


def test_weights_are_never_negative_and_sum_to_one():
    matches, forecasts = simulated(seed=12)
    forecasts.pop("perfect")  # what is left is worth little, and the reversed member is worse than nothing
    weights = fit_weights(forecasts, matches, "result")
    assert set(weights) == {"flat", "reversed"}
    assert all(w >= 0 for w in weights.values()) and sum(weights.values()) == pytest.approx(1)
    assert weights["flat"] > weights["reversed"]


class Fixed:
    def __init__(self, name, forecast):
        self.name, self.forecast, self.learned = name, forecast, 0

    def predict(self, matches):
        return [self.forecast for _ in matches]

    def learn(self, matches):
        self.learned += len(matches)


def test_goal_lines_come_only_from_members_with_a_matrix():
    matrix = market_goals._matrix(1.6, 1.1)
    matrix = matrix / matrix.sum()
    ratings = Forecast(result=(0.40, 0.30, 0.30), goal_difference={d: 1 / 9 for d in range(-4, 5)})
    scores = Forecast(result=(0.50, 0.25, 0.25), score_matrix=matrix)
    matches = [match(i, (i % 3, i % 2)) for i in range(60)]
    forecasts = {"ratings": [ratings] * 60, "scores": [scores] * 60}
    weights = {market: fit_weights(forecasts, matches, market) for market in ("result", "handicap", "goals")}
    assert weights["goals"] == {"scores": 1.0}
    assert set(weights["result"]) == set(weights["handicap"]) == {"ratings", "scores"}

    average = Average([Fixed("ratings", ratings), Fixed("scores", scores)], weights)
    found = average.predict(matches[:1])[0]
    over = {"market": "goals", "side": "over", "line": 2.5}
    assert chance(found, over) == pytest.approx(chance(scores, over))
    assert chance(found, {"market": "btts", "side": "yes", "line": None}) == pytest.approx(
        chance(scores, {"market": "btts", "side": "yes", "line": None}))
    assert sum(found.result) == pytest.approx(1) and sum(found.goal_difference.values()) == pytest.approx(1)
    average.learn(matches)
    assert [m.learned for m in average.members] == [60, 60]
    alone = combine({"ratings": ratings}, weights)  # no member with a matrix: no goal lines at all
    assert alone.score_matrix is None and chance(alone, over) is None
