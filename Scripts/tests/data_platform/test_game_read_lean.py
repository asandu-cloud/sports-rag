"""Game-read bench, phase 2 task 5: the lean-away blend, model 5b (D77)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from Scripts.data_platform.game_read import pool, walkforward
from Scripts.data_platform.game_read.forecast import Forecast, chance
from Scripts.data_platform.game_read.history import Match, Prices, market_over25, market_result
from Scripts.data_platform.game_read.models.lean import LeanAway

OVER = ("goals", "over", 2.5)
UNDER = ("goals", "under", 2.5)


def priced(fair, booksum=1.05):
    return tuple(round(p, 3) for p in pool.inverse_power(fair, booksum))


def simulated(seed=3, years=(2013, 2014, 2015, 2016), per_year=900):
    """The market's chance and a member's, independent; what happens follows market · member^−0.6 (normalised), so
    the member is informative the wrong way round and a fitted blend leans away from it (c < 0)."""
    rng = np.random.default_rng(seed)
    matches, member = [], {}
    for year in years:
        for i in range(per_year):
            market = rng.dirichlet([6, 3, 4])
            said = rng.dirichlet([5, 3, 5])
            truth = market * said ** -0.6
            truth /= truth.sum()
            over_market, over_said = rng.uniform(0.35, 0.65), rng.uniform(0.3, 0.7)
            over_truth = 1 / (1 + np.exp(-(np.log(over_market / (1 - over_market))
                                            - 0.6 * np.log(over_said / (1 - over_said)))))
            outcome = rng.choice(3, p=truth)
            goals = [(1, 0), (1, 1), (0, 1)][outcome]
            if rng.uniform() < over_truth:
                goals = (goals[0] + 2, goals[1] + 1)
            prices = Prices(result_avg=priced(list(market)), over25_avg=priced([over_market, 1 - over_market]))
            found = Match(id=len(matches), competition="EPL", country="X", season=year,
                          day=date(year, 8, 10) + timedelta(days=i // 9), home=f"X:H{i % 7}", away=f"X:A{i % 5}",
                          home_goals=goals[0], away_goals=goals[1], pre=prices, close=prices)
            matches.append(found)
            member[found.id] = Forecast(result=tuple(said), lines={OVER: over_said, UNDER: 1 - over_said})
    return matches, member


class Known:
    """A member whose forecast for each match is fixed in advance."""
    name = "known"

    def __init__(self, table):
        self.table = table

    def predict(self, matches):
        return [self.table[m.id] for m in matches]

    def learn(self, matches):
        pass


def test_a_lean_away_fit_never_sees_its_own_season():
    matches, member = simulated(per_year=300)
    lean = LeanAway("known", {"known": Known(member)}, start=2013)
    walkforward.run(lean, matches, score_from=2015)
    assert set(lean.fits) == {2015, 2016}
    for season, fitted in lean.fits.items():
        assert fitted["seasons"] and max(fitted["seasons"]) < season
    assert lean.fits[2015]["n"] == 600 and lean.fits[2016]["n"] == 900


def test_lean_away_chances_sum_to_one_with_c_below_zero():
    matches, member = simulated()
    lean = LeanAway("known", {"known": Known(member)}, start=2013)
    rows = walkforward.run(lean, matches, score_from=2016)
    assert lean.name == "lean-known"
    assert lean.fits[2016]["result"]["c"] < -0.3 and lean.fits[2016]["over25"]["c"] < -0.3
    for match, forecast in rows:
        assert sum(forecast.result) == pytest.approx(1) and all(0 < p < 1 for p in forecast.result)
        over = chance(forecast, {"market": "goals", "side": "over", "line": 2.5})
        under = chance(forecast, {"market": "goals", "side": "under", "line": 2.5})
        assert 0 < over < 1 and over + under == pytest.approx(1)
    # leaning away beats the market here, out of sample
    lean_loss = np.mean([-np.log(f.result[0 if m.home_goals > m.away_goals else 1 if m.home_goals == m.away_goals
                                          else 2]) for m, f in rows])
    market_loss = np.mean([-np.log(market_result(m.pre)[0 if m.home_goals > m.away_goals else 1
                                                       if m.home_goals == m.away_goals else 2]) for m, _ in rows])
    assert lean_loss < market_loss
    assert market_over25(rows[0][0].pre) is not None


def timid(seed=5, years=(2013, 2014, 2015), per_year=1200):
    """Matches where what happens follows the market's chances sharpened (market^1.6, normalised): the market is
    too timid, and only the market is needed to see it."""
    rng = np.random.default_rng(seed)
    matches = []
    for year in years:
        for i in range(per_year):
            market = rng.dirichlet([6, 3, 4])
            truth = market ** 1.6
            truth /= truth.sum()
            over_market = rng.uniform(0.3, 0.7)
            over_truth = over_market ** 1.6 / (over_market ** 1.6 + (1 - over_market) ** 1.6)
            goals = [(1, 0), (1, 1), (0, 1)][rng.choice(3, p=truth)]
            if rng.uniform() < over_truth:
                goals = (goals[0] + 2, goals[1] + 2)  # over 2.5, same result
            prices = Prices(result_avg=priced(list(market)), over25_avg=priced([over_market, 1 - over_market]))
            matches.append(Match(id=len(matches), competition="EPL", country="X", season=year,
                                 day=date(year, 8, 10) + timedelta(days=i // 12), home=f"X:H{i % 7}",
                                 away=f"X:A{i % 5}", home_goals=goals[0], away_goals=goals[1], pre=prices,
                                 close=prices))
    return matches


def test_market_only_lean_has_no_member_and_sharpens_the_market():
    """Model 5c (owner, 17 Nov): the lean blend with no member, refitted each season the same way, so the lean
    blends' gain can be split into the market's own sharpening and what the member adds."""
    matches = timid()
    lean = LeanAway("market", {}, start=2013)
    rows = walkforward.run(lean, matches, score_from=2015)
    assert lean.name == "lean-market" and set(lean.fits) == {2015}
    for part in ("result", "over25"):
        assert lean.fits[2015][part]["c"] == 0 and not lean.fits[2015][part]["use_spix"]
        assert lean.fits[2015][part]["b"] > 1.3
    outcome = lambda m: 0 if m.home_goals > m.away_goals else 1 if m.home_goals == m.away_goals else 2
    for match, forecast in rows:
        market = market_result(match.pre)
        assert sum(forecast.result) == pytest.approx(1)
        # sharper: the favourite's chance grows
        favourite = int(np.argmax(market))
        assert forecast.result[favourite] >= market[favourite] - 0.02
    lean_loss = np.mean([-np.log(f.result[outcome(m)]) for m, f in rows])
    market_loss = np.mean([-np.log(market_result(m.pre)[outcome(m)]) for m, _ in rows])
    assert lean_loss < market_loss
