"""Game-read bench, phase 2 task 5: the features pass (D77)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from Scripts.data_platform.game_read import pool
from Scripts.data_platform.game_read.features import Features
from Scripts.data_platform.game_read.history import Match, Prices
from Scripts.data_platform.game_read.models.dixon_coles import DixonColes
from Scripts.data_platform.game_read.models.elo import Elo
from Scripts.data_platform.game_read.models.rest import RestVariable
from Scripts.data_platform.game_read.models.shots import ShotVariable
from Scripts.data_platform.game_read.stats import MatchStats

TEAMS = [f"X:{c}" for c in "ABCDEF"]


def priced(fair, booksum):
    return tuple(round(p, 2) for p in pool.inverse_power(fair, booksum))


def history(seed=2, years=(2012, 2013, 2014), per_year=90):
    rng = np.random.default_rng(seed)
    matches, stats = [], {}
    for year in years:
        for i in range(per_year):
            home, away = rng.choice(TEAMS, size=2, replace=False)
            home_win = rng.uniform(0.3, 0.6)
            prices = Prices(result_avg=priced([home_win, 0.27, 0.73 - home_win], 1.05),
                            over25_avg=priced([0.52, 0.48], 1.05))
            found = Match(id=len(matches), competition="EPL", country="X", season=year,
                          day=date(year, 8, 10) + timedelta(days=i // 3), home=str(home), away=str(away),
                          home_goals=int(rng.poisson(1.5)), away_goals=int(rng.poisson(1.1)), pre=prices, close=prices)
            matches.append(found)
            stats[found.id] = MatchStats(home_box=int(rng.poisson(9)), away_box=int(rng.poisson(7)), home_outside=5,
                                         away_outside=4, home_rest=int(rng.integers(2, 9)), away_rest=4)
    return matches, stats


def run(matches, stats):
    features = Features({"elo": Elo(), "dc": DixonColes()}, variables=(ShotVariable("xg", stats),
                                                                       RestVariable(stats)), start=2013)
    for day in sorted({m.day for m in matches}):
        features.learn([m for m in matches if m.day == day])
    return features


def test_features_for_a_day_use_only_earlier_days():
    matches, stats = history()
    last = max(m.day for m in matches)
    changed = [m if m.day != last else Match(**{**m.__dict__, "home_goals": 9, "away_goals": 0}) for m in matches]
    # what only the match itself tells (its result and shots) changes; rest is known before kickoff and stays
    changed_stats = {**stats, **{m.id: MatchStats(home_box=40, away_box=0, home_outside=30, away_outside=0,
                                                  home_rest=stats[m.id].home_rest, away_rest=stats[m.id].away_rest)
                                 for m in matches if m.day == last}}
    before, after = run(matches, stats), run(changed, changed_stats)
    on_last = [m for m in matches if m.day == last]
    assert on_last and all(before.row(m) == after.row(m) for m in on_last)

    row = before.row(matches[-1])
    for key in ("elo_home", "elo_draw", "elo_away", "dc_home", "dc_over", "dc_home_goals", "dc_away_goals",
                "market_home", "market_over", "lead_elo_home", "lead_dc_over", "xg_home_0", "xg_away_0",
                "rest_home_0", "rest_home_1", "league", "month", "weekday", "margin"):
        assert key in row, key
    assert row["elo_over"] is None  # Elo gives no score matrix
    assert abs(row["elo_home"] + row["elo_draw"] + row["elo_away"] - 1) < 1e-9
    assert row["lead_elo_home"] == np.log(row["market_home"]) - np.log(row["elo_home"])
    assert row["margin"] > 0
    recorded = {m.season for m in before.history}
    assert recorded == {2013, 2014}  # every match from the start season has out-of-sample inputs, none before
    assert len(before.history) == 180


def test_a_finished_pass_can_be_shared():
    """A second model reading a finished pass replays its days; nothing is learned or recorded twice."""
    matches, stats = history()
    counted = []

    class Counting:
        name = "counting"

        def predict(self, found):
            return [Elo().predict([m])[0] for m in found]

        def learn(self, found):
            counted.extend(m.id for m in found)

    features = Features({"counting": Counting()}, start=2013)
    for day in sorted({m.day for m in matches}):
        features.learn([m for m in matches if m.day == day])
    before = (list(features.history), dict(features.forecasts), len(counted))
    for day in sorted({m.day for m in matches}):
        features.predict([m for m in matches if m.day == day])
        features.learn([m for m in matches if m.day == day])
    assert (list(features.history), dict(features.forecasts), len(counted)) == before
    assert len(counted) == len(matches)
