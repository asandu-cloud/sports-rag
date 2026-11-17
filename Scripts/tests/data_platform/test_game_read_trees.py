"""Game-read bench, phase 2 task 6: boosted trees on the market's error, model 6 (D77)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from Scripts.data_platform.game_read import pool, walkforward
from Scripts.data_platform.game_read.features import Features
from Scripts.data_platform.game_read.forecast import chance
from Scripts.data_platform.game_read.history import Match, Prices, market_handicap, market_over25, market_result
from Scripts.data_platform.game_read.models.trees import Trees

HOME = {"market": "moneyline", "side": "home", "line": None}
OVER = {"market": "goals", "side": "over", "line": 2.5}


def priced(fair, booksum=1.05):
    return tuple(round(p, 3) for p in pool.inverse_power(fair, booksum))


def simulated(seed=5, years=(2013, 2014, 2015, 2016), per_year=1600, shift=0.6):
    """Two leagues. In league B the home side wins, and the match goes over 2.5, more often than the market says
    (``shift`` on the log-odds); in league A the market is right. The handicap is home −0.5: it covers when the
    home side wins."""
    rng = np.random.default_rng(seed)
    matches = []
    for year in years:
        for i in range(per_year):
            league = "B" if i % 2 else "A"
            home, draw = rng.uniform(0.3, 0.6), rng.uniform(0.22, 0.3)
            market = np.array([home, draw, 1 - home - draw])
            over = rng.uniform(0.4, 0.6)
            truth, over_truth = market.copy(), over
            if league == "B":
                truth = market * np.exp([shift, 0, 0])
                truth /= truth.sum()
                over_truth = 1 / (1 + np.exp(-(np.log(over / (1 - over)) + shift)))
            outcome = rng.choice(3, p=truth)
            goals = [(1, 0), (0, 0), (0, 1)][outcome]
            if rng.uniform() < over_truth:
                goals = (goals[0] + 2, goals[1] + 1) if outcome != 1 else (2, 2)
            prices = Prices(result_avg=priced(list(market)), over25_avg=priced([over, 1 - over]),
                            handicap_line=-0.5, handicap_avg=priced([home, 1 - home]))
            matches.append(Match(id=len(matches), competition=league, country="X", season=year,
                                 day=date(year, 8, 10) + timedelta(days=i // 16), home=f"X:H{i % 9}",
                                 away=f"X:A{i % 7}", home_goals=goals[0], away_goals=goals[1], pre=prices,
                                 close=prices))
    return matches


def finished(matches):
    """A features pass already run over every match, as the bench shares it between models."""
    features = Features({}, start=2013)
    for day in sorted({m.day for m in matches}):
        features.learn([m for m in matches if m.day == day])
    return features


def test_with_no_rounds_it_is_the_market():
    matches = simulated(per_year=400)
    rows = walkforward.run(Trees(features=Features({}, start=2013), rounds=0), matches, score_from=2015)
    assert rows
    for match, forecast in rows:
        assert list(forecast.result) == list(market_result(match.pre))
        assert chance(forecast, OVER) == market_over25(match.pre)[0]
        cover = market_handicap(match.pre)
        assert chance(forecast, {"market": "spreads", "side": "home", "line": -0.5}) == cover[0]
        assert chance(forecast, {"market": "spreads", "side": "away", "line": 0.5}) == cover[1]
        # lines the market does not price get no chance
        assert chance(forecast, {"market": "goals", "side": "over", "line": 3.5}) is None
        assert chance(forecast, {"market": "spreads", "side": "home", "line": -1.5}) is None


def test_it_learns_a_planted_correction():
    matches = simulated()
    trees = Trees(features=Features({}, start=2013))
    rows = walkforward.run(trees, matches, score_from=2016)
    assert trees.fits[2016]["result"]["rounds"] > 0 and trees.fits[2016]["over25"]["rounds"] > 0

    def lift(league, option, market_of):
        return np.mean([chance(f, option) - market_of(m.pre)[0] for m, f in rows if m.competition == league])

    assert lift("B", HOME, market_result) > 0.04 and abs(lift("A", HOME, market_result)) < 0.02
    assert lift("B", OVER, market_over25) > 0.04 and abs(lift("A", OVER, market_over25)) < 0.02

    def loss(forecast_of):
        won = [0 if m.home_goals > m.away_goals else 1 if m.home_goals == m.away_goals else 2 for m, _ in rows]
        return np.mean([-np.log(forecast_of(m, f)[k]) for (m, f), k in zip(rows, won)])

    assert loss(lambda m, f: f.result) < loss(lambda m, f: market_result(m.pre))


def test_it_never_trains_on_the_season_it_predicts():
    matches = simulated()
    changed = [m if m.season != 2016 else Match(**{**m.__dict__, "home_goals": m.away_goals,
                                                   "away_goals": m.home_goals + 3}) for m in matches]
    found = {}
    for name, these in (("before", matches), ("after", changed)):
        trees = Trees(features=finished(these))
        found[name] = [f for m, f in walkforward.run(trees, these, score_from=2015) if m.season == 2016]
        assert trees.fits[2016]["result"]["rounds"] > 0  # the trees did learn something
        for season, fitted in trees.fits.items():
            assert max(fitted["train"]) < season and fitted["stop"] == season - 1
    assert [f.result for f in found["before"]] == [f.result for f in found["after"]]
    assert [f.lines for f in found["before"]] == [f.lines for f in found["after"]]
