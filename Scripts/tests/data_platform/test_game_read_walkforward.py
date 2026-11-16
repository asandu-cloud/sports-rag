"""Game-read bench, task 5: forecasts, the walk-forward runner, and the zero and control lines (D77)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import random

import numpy as np
import pytest

from Scripts.data_platform import market_goals
from Scripts.data_platform.game_read import pool, walkforward
from Scripts.data_platform.game_read.forecast import Forecast, chance
from Scripts.data_platform.game_read.history import Match, Prices
from Scripts.data_platform.game_read.models.market import MarketControl, MarketCopy

PRE = Prices(result_avg=(2.10, 3.40, 3.60), handicap_line=-0.5, handicap_avg=(2.08, 1.82), over25_avg=(1.95, 1.90))
MARGINS = {("goals", 1.5): 1.07, ("btts", None): 1.07}


def match(id, day, home="A", away="B", *, season=2020, goals=(1, 0), pre=PRE):
    return Match(id=id, competition="EPL", country="England", season=season, day=day, home=home, away=away,
                 home_goals=goals[0], away_goals=goals[1], pre=pre, close=pre)


class Spy:
    name = "spy"

    def __init__(self):
        self.events, self.latest = [], None

    def predict(self, matches):
        self.events.append(("predict", matches[0].day, sorted(m.id for m in matches), self.latest))
        return [Forecast(result=(0.5, 0.3, 0.2)) for _ in matches]

    def learn(self, matches):
        self.events.append(("learn", matches[0].day, sorted(m.id for m in matches), self.latest))
        self.latest = max(m.day for m in matches)


def test_no_forecast_sees_its_own_day():
    d = date(2020, 9, 12)
    history = [match(1, d, "A", "B"), match(2, d, "C", "D"), match(3, d, "E", "F"),
               match(4, d + timedelta(days=3), "A", "C"),  # A twice in three days
               match(5, d + timedelta(days=3), "B", "D"),
               match(6, d + timedelta(days=1), "E", "A"),  # a postponed game, listed last
               match(7, d + timedelta(days=7), "F", "B")]
    shuffled = history[:]
    random.Random(4).shuffle(shuffled)
    spy = Spy()
    rows = walkforward.run(spy, shuffled, score_from=2020)
    predicts = [e for e in spy.events if e[0] == "predict"]
    assert all(latest is None or latest < day for _, day, _, latest in predicts)
    assert [(day, ids) for _, day, ids, _ in predicts] == [
        (d, [1, 2, 3]), (d + timedelta(days=1), [6]), (d + timedelta(days=3), [4, 5]), (d + timedelta(days=7), [7])]
    for i, event in enumerate(spy.events):  # each day: predicted whole, then learned whole
        if event[0] == "predict":
            assert spy.events[i + 1][:3] == ("learn", event[1], event[2])
    assert [m.id for m, _ in rows] == [1, 2, 3, 6, 4, 5, 7]


def test_warm_up_seasons_are_learned_not_scored():
    history = [match(1, date(2018, 3, 1), season=2017), match(2, date(2018, 9, 1), season=2018),
               match(3, date(2019, 9, 1), season=2019), match(4, date(2023, 9, 1), season=2023)]
    spy = Spy()
    rows = walkforward.run(spy, history, score_from=2019)
    assert [m.id for m, _ in rows] == [3, 4]
    assert [e[0] for e in spy.events] == ["learn", "learn", "predict", "learn", "predict", "learn"]
    spy = Spy()
    rows = walkforward.run(spy, history, score_from=2019, until=2022)  # a tuning run never sees a locked season
    assert [m.id for m, _ in rows] == [3]
    assert all(4 not in ids for _, _, ids, _ in spy.events)


@pytest.mark.parametrize("prices", [PRE, replace(PRE, handicap_line=-0.75, handicap_avg=(2.20, 1.72)),
                                    replace(PRE, over25_avg=None)])
def test_the_market_copy_gives_the_markets_chances(prices):
    m = match(1, date(2020, 9, 12), pre=prices)
    forecast = MarketCopy().predict([m])[0]
    options = pool.build_pool(m, margins=MARGINS)
    assert len(options) == 27
    for option in options:
        if option["real"]:
            assert chance(forecast, option) == option["chance"]
        else:
            assert chance(forecast, option) == pytest.approx(option["chance"], abs=1e-9)
    assert MarketCopy().predict([match(2, date(2020, 9, 12), pre=Prices())])[0] == Forecast()


def option(market, side, line):
    return {"market": market, "side": side, "line": line, "chance": 0.5, "typical_price": 2.0}


def test_chance_reads_handicaps_from_a_goal_difference_and_nothing_else_without_a_matrix():
    forecast = Forecast(goal_difference={-1: 0.2, 0: 0.3, 1: 0.3, 2: 0.2})
    assert chance(forecast, option("spreads", "home", -0.5)) == pytest.approx(0.5)
    assert chance(forecast, option("spreads", "away", 0.5)) == pytest.approx(0.5)
    assert chance(forecast, option("spreads", "home", -1.5)) == pytest.approx(0.2)
    assert chance(forecast, option("spreads", "away", 1.5)) == pytest.approx(0.8)
    assert chance(forecast, option("spreads", "home", 0.5)) == pytest.approx(0.8)
    assert chance(forecast, option("spreads", "home", -0.75)) is None  # quarter lines are not in the pool
    for market, side, line in (("goals", "over", 2.5), ("btts", "yes", None), ("moneyline", "home", None)):
        assert chance(forecast, option(market, side, line)) is None
    matrix = market_goals._matrix(1.5, 1.1)
    from_matrix = Forecast(score_matrix=matrix / matrix.sum())
    assert chance(from_matrix, option("spreads", "home", -1.5)) == pytest.approx(
        market_goals.handicap_probability(1.5, 1.1, -1.5))
    assert chance(from_matrix, option("goals", "under", 2.5)) == pytest.approx(
        1 - market_goals.over_probability(1.5, 1.1, 2.5))
    assert chance(Forecast(), option("spreads", "home", -0.5)) is None


def test_the_control_line_moves_low_scores_only():
    start = date(2012, 8, 18)
    low = [(0, 0), (1, 1), (0, 0), (1, 1), (1, 0), (2, 1), (0, 0), (3, 1)]
    history = [match(i, start + timedelta(days=i), goals=low[i % len(low)], season=2012) for i in range(400)]
    target = match(1000, date(2019, 8, 10), season=2019)
    control = MarketControl()
    rows = walkforward.run(control, history + [target], score_from=2019)
    assert control.rho < 0
    forecast = rows[0][1]
    home, away = pool.m1_means(PRE)
    plain = market_goals._matrix(home, away)
    plain = plain / plain.sum()
    matrix = forecast.score_matrix
    assert matrix.shape == (13, 13) and matrix.sum() == pytest.approx(1)
    assert matrix[0, 0] > plain[0, 0] and matrix[1, 1] > plain[1, 1]
    assert matrix[1, 0] < plain[1, 0] and matrix[0, 1] < plain[0, 1]
    ratio = matrix[2:, :] / plain[2:, :]
    assert np.allclose(ratio, ratio.flat[0])  # every other score keeps its share
    assert forecast.result[1] > market_goals.result_probabilities(home, away)[1]  # more draws
    assert sum(forecast.result) == pytest.approx(1)


def test_the_control_line_starts_as_m1_and_refits_only_at_a_season_start():
    control = MarketControl()
    first = control.predict([match(1, date(2019, 8, 10), season=2019)])[0]
    home, away = pool.m1_means(PRE)
    plain = market_goals._matrix(home, away)
    assert control.rho == 0 and np.allclose(first.score_matrix, plain / plain.sum())
    control.learn([match(i, date(2019, 8, 10), goals=(0, 0), season=2019) for i in range(50)])
    control.predict([match(60, date(2019, 8, 17), season=2019)])
    assert control.rho == 0  # the same season: no refit
    control.predict([match(61, date(2020, 8, 15), season=2020)])
    assert control.rho < 0
    assert control.predict([match(62, date(2020, 8, 15), pre=Prices(), season=2020)])[0] == Forecast()
