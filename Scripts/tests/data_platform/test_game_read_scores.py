"""Game-read bench, task 6: the scores (D77)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import math
import random

import numpy as np
import pytest

from Scripts.data_platform import devig, market_goals
from Scripts.data_platform.game_read import pool, rules, scores, walkforward
from Scripts.data_platform.game_read.forecast import Forecast
from Scripts.data_platform.game_read.history import Match, Prices, market_result
from Scripts.data_platform.game_read.models.market import MarketCopy


def priced(fair, booksum):
    return tuple(round(p, 2) for p in pool.inverse_power(fair, booksum))


def prices(home, away, line, rng, *, booksum=1.05):
    result = market_goals.result_probabilities(home, away)
    over = market_goals.over_probability(home, away, 2.5)
    cover = market_goals.handicap_probability(home, away, line)
    best = rng.uniform(1.03, 1.10)  # stale best prices sit above the average

    def pair(fair):
        average = priced(fair, booksum - 0.01)
        return average, tuple(round(p * best, 2) for p in average)
    (result_avg, result_max), handicap, over25 = pair(result), pair([cover, 1 - cover]), pair([over, 1 - over])
    return Prices(result_avg=result_avg, result_max=result_max, handicap_line=line, handicap_avg=handicap[0],
                  handicap_max=handicap[1], over25_avg=over25[0], over25_max=over25[1])


def synthetic(n=96, seed=11):
    rng, goals = random.Random(seed), np.random.default_rng(seed)
    out = []
    for i in range(n):
        home, away = rng.uniform(0.7, 2.3), rng.uniform(0.5, 1.7)
        line = rng.choice([-1.5, -0.5, 0.5, -0.75, -0.25])
        drift = rng.uniform(0.9, 1.1)
        out.append(Match(id=i + 1, competition=("EPL", "LaLiga")[i % 2], country="X", season=2019,
                         day=date(2019, 8, 10) + timedelta(days=(i // 4) * 3), home=f"X:{i}a", away=f"X:{i}b",
                         home_goals=int(goals.poisson(home)), away_goals=int(goals.poisson(away)),
                         pre=prices(home, away, line, rng), close=prices(home * drift, away / drift, line, rng)))
    return out


@pytest.fixture(scope="module")
def zero_rows():
    return walkforward.run(MarketCopy(), synthetic(), score_from=2019)


def week_row(day, value):
    match = Match(id=0, competition="EPL", country="X", season=2019, day=day, home="X:a", away="X:b",
                  home_goals=0, away_goals=0, pre=Prices(), close=Prices())
    return match, (value, 1.0)


def test_week_interval_resamples_whole_weeks():
    monday = date(2019, 9, 2)
    rows = [week_row(monday + timedelta(days=d), 1.0) for d in range(5)]
    rows += [week_row(monday + timedelta(days=7 + d), 0.0) for d in range(5)]
    seen = []

    def mean(total):
        seen.append(total[0] / total[1])
        return seen[-1]

    low, high = scores.week_interval(rows, mean, resamples=500, seed=3)
    assert len(seen) == 500 and set(seen) <= {0.0, 0.5, 1.0} and len(set(seen)) == 3
    assert 0 <= low < high <= 1
    assert scores.week_interval(rows, mean, resamples=500, seed=3) == (low, high)  # seeded


def test_the_zero_line_scores_zero(zero_rows):
    picks = scores.picks_score(zero_rows, strength=rules.by_edge)
    assert picks["today"]["picks"] > 100
    assert picks["engine"] == picks["today"]
    assert picks["estimate"] == 0 and picks["low"] == 0 and picks["high"] == 0
    assert picks["real"]["estimate"] == 0 and picks["derived"]["estimate"] == 0
    assert all(league["estimate"] == 0 for league in picks["by_league"].values())
    for market in scores.MARKETS:
        assert scores.accuracy(zero_rows, market)["estimate"] == 0
        assert abs(scores.blend_score(zero_rows, zero_rows, market)["estimate"]) < 1e-6
    value = scores.value_score(zero_rows, price="avg")
    assert value["engine"]["picks"] == 0 and value["today"]["picks"] == 0
    assert scores.closing_check(zero_rows, strength=rules.by_edge)["estimate"] == 0


def test_the_market_copy_at_the_best_price_is_todays_value_rule(zero_rows):
    value = scores.value_score(zero_rows, price="max")
    assert value["today"]["picks"] > 10
    assert value["engine"] == value["today"]


def test_a_known_win_counts_once_against_its_chance(monkeypatch):
    one = [{"market": "goals", "side": "over", "line": 1.5, "chance": 0.6, "typical_price": 1.55, "real": False},
           {"market": "goals", "side": "under", "line": 1.5, "chance": 0.4, "typical_price": 2.30, "real": False}]
    monkeypatch.setattr(scores, "_pool", lambda match, phase="pre": tuple(one))
    match = synthetic(1)[0]
    match = replace(match, home_goals=2, away_goals=0)
    forecast = Forecast(lines={("goals", "over", 1.5): 0.6, ("goals", "under", 1.5): 0.4})
    picks = scores.picks_score([(match, forecast)], strength=rules.by_edge)
    assert picks["today"] == {"picks": 1, "wins": 1, "expected": 0.6, "per_100": pytest.approx(40),
                              "win_rate": 1.0}
    assert picks["derived"]["today"]["per_100"] == pytest.approx(40) and picks["real"]["today"]["picks"] == 0
    lost = replace(match, home_goals=1)
    assert scores.picks_score([(lost, forecast)], strength=rules.by_edge)["today"]["per_100"] == pytest.approx(-60)


def test_clv_is_price_times_closing_fair_chance_minus_one():
    pre = Prices(result_avg=(2.10, 3.40, 3.60), result_max=(2.40, 3.45, 3.65))
    close = Prices(result_avg=(2.00, 3.50, 3.80))
    match = Match(id=1, competition="EPL", country="X", season=2019, day=date(2019, 9, 1), home="X:a", away="X:b",
                  home_goals=2, away_goals=1, pre=pre, close=close)
    value = scores.value_score([(match, MarketCopy().predict([match])[0])], price="max")
    assert value["today"]["picks"] == 1
    assert value["today"]["clv"] == pytest.approx(2.40 * devig.power([2.00, 3.50, 3.80])[0] - 1)
    assert value["today"]["return"] == pytest.approx(1.40)


def test_a_moved_closing_handicap_is_priced_from_m1():
    pre = Prices(result_avg=(2.10, 3.40, 3.60), handicap_line=-0.5, handicap_avg=(2.08, 1.82), over25_avg=(1.95, 1.90))
    close = replace(pre, result_avg=(1.80, 3.70, 4.60), handicap_line=-1.0, handicap_avg=(2.30, 1.65))
    match = Match(id=1, competition="EPL", country="X", season=2019, day=date(2019, 9, 1), home="X:a", away="X:b",
                  home_goals=2, away_goals=1, pre=pre, close=close)
    home = {"market": "spreads", "side": "home", "line": -0.5}
    assert scores.closing_chance(match, home) == pytest.approx(
        market_goals.handicap_probability(*pool.m1_means(close), -0.5))
    over = {"market": "goals", "side": "over", "line": 2.5}
    assert scores.closing_chance(match, over) == pytest.approx(devig.power([1.95, 1.90])[0])
    assert scores.closing_chance(replace(match, close=Prices()), home) is None


def test_accuracy_is_log_loss_against_the_market():
    pre = Prices(result_avg=(2.10, 3.40, 3.60))
    match = Match(id=1, competition="EPL", country="X", season=2019, day=date(2019, 9, 1), home="X:a", away="X:b",
                  home_goals=2, away_goals=1, pre=pre, close=pre)
    found = scores.accuracy([(match, Forecast(result=(0.5, 0.3, 0.2)))], "result")
    assert found["model_log_loss"] == pytest.approx(-math.log(0.5))
    assert found["estimate"] == pytest.approx(-math.log(0.5) + math.log(market_result(pre)[0]))
    assert scores.accuracy([(match, Forecast())], "result")["n"] == 0
    assert scores.accuracy([(match, Forecast(result=(0.5, 0.3, 0.2)))], "over25")["n"] == 0


def test_the_reference_check_measures_the_average_against_the_named_books():
    pre = Prices(result_avg=(2.10, 3.40, 3.60))
    match = Match(id=7, competition="EPL", country="X", season=2019, day=date(2019, 9, 1), home="X:a", away="X:b",
                  home_goals=2, away_goals=1, pre=pre, close=pre)
    books = [(2.05, 3.40, 3.70), (2.10, 3.30, 3.60), (2.20, 3.40, 3.50)]
    found = scores.reference_check([match, replace(match, id=8)], {7: books, 8: books[:2]})
    median = np.median([devig.power(b) for b in books], axis=0)
    gap = max(abs(a - b) for a, b in zip(market_result(pre), median))
    assert found == {2019: {"mean": pytest.approx(gap), "p95": pytest.approx(gap), "n": 1}}  # 2 books: left out


def test_verdicts_apply_the_spec_rules():
    def value(picks, *, price="avg", low=0.01, ret=0.05):
        return {"price": price, "estimate": 0.03, "low": low, "high": 0.05, "n": picks,
                "engine": {"picks": picks, "return": ret}}

    def picks(worst_league, *, low=0.5, derived=2.0):
        return {"estimate": 3.0, "low": low, "high": 5.0, "by_league": {"EPL": {"estimate": 4.0, "n": 50},
                                                                       "LaLiga": {"estimate": worst_league, "n": 50}},
                "derived": {"estimate": derived}}

    results = {
        "control": {"picks": picks(0.0, derived=1.0)},
        "good": {"value": value(300), "picks": picks(-1.9),
                 "blend": {"result": {"c": 0.1, "high": -0.001}, "over25": {"c": 0.1, "high": 0.0001}}},
        "few": {"value": value(299, low=0.2, ret=0.5), "picks": picks(-2.5)},
        "best_price": {"value": value(5000, price="max"), "picks": picks(0.0, derived=0.9)},
        "losing": {"value": value(400, ret=-0.01), "picks": picks(0.0, low=-0.1)},
    }
    found = scores.verdicts(results)
    assert found["good"] == {"value": True, "picks": True, "blend": {"result": True, "over25": False}}
    assert found["few"] == {"value": False, "picks": False}
    assert found["best_price"] == {"value": False, "picks": False}  # derived lines no better than the control
    assert found["losing"] == {"value": False, "picks": False}
    assert found["control"] == {"picks": False}
