"""Game-read bench, phase 2 task 3: shot ratings and the derived shot-location xG in models 3–4 (D77)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import math

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from Scripts.data_platform.game_read import walkforward
from Scripts.data_platform.game_read.history import Match, Prices
from Scripts.data_platform.game_read.models import dixon_coles as dc
from Scripts.data_platform.game_read.models import shots
from Scripts.data_platform.game_read.models.dixon_coles import DixonColes
from Scripts.data_platform.game_read.models.drifting import Drifting
from Scripts.data_platform.game_read.models.shots import ShotVariable
from Scripts.data_platform.game_read.stats import MatchStats

TEAMS = {f"X:{c}": s for c, s in zip("ABCDEF", (0.4, 0.2, 0.1, 0.0, -0.2, -0.4))}


def match(id, day, home, away, goals=(1, 1), season=2019, league="EPL"):
    return Match(id=id, competition=league, country="X", season=season, day=day, home=home, away=away,
                 home_goals=goals[0], away_goals=goals[1], pre=Prices(), close=Prices())


def seasons(rng, years=(2019, 2020), rounds=2):
    """Each year every pair meets ``rounds`` times home and away, three matches a day; goals and shots in the box
    follow the teams' strengths."""
    matches, stats = [], {}
    for year in years:
        day, teams = date(year, 8, 10), list(TEAMS)
        for _ in range(rounds):
            for home in teams:
                for away in teams:
                    if home == away:
                        continue
                    strength = TEAMS[home] - TEAMS[away]
                    box = (int(rng.poisson(10 * math.exp(strength))), int(rng.poisson(10 * math.exp(-strength))))
                    goals = (int(rng.poisson(0.13 * box[0])), int(rng.poisson(0.11 * box[1])))
                    found = match(len(matches), day + timedelta(days=len(matches) // 3 % 300), home, away, goals,
                                  season=year)
                    matches.append(found)
                    stats[found.id] = MatchStats(home_box=box[0], away_box=box[1], home_outside=5, away_outside=5,
                                                 home_target=box[0] // 3, away_target=box[1] // 3)
    return matches, stats


def test_location_xg_recovers_known_per_shot_values():
    rng = np.random.default_rng(3)
    matches, stats = [], {}
    for i in range(6000):
        inside, outside = rng.poisson(9, 2), rng.poisson(6, 2)
        goals = rng.poisson(0.11 * inside + 0.03 * outside)
        matches.append(match(i, date(2020, 1, 1), "X:A", "X:B", (int(goals[0]), int(goals[1]))))
        stats[i] = MatchStats(home_box=int(inside[0]), away_box=int(inside[1]), home_outside=int(outside[0]),
                              away_outside=int(outside[1]))
    stats[6000] = MatchStats(home_box=8)  # a match with only some counts is left out, not an error
    matches.append(match(6000, date(2020, 1, 1), "X:A", "X:B"))
    inside, outside = shots.fit_location_xg(stats, matches)
    assert inside == pytest.approx(0.11, abs=0.01) and outside == pytest.approx(0.03, abs=0.01)


def test_location_xg_is_none_without_shot_counts():
    assert shots.location_xg(None, 4) is None and shots.location_xg(5, None) is None
    assert shots.location_xg(10, 5, weights=(0.1, 0.03)) == pytest.approx(1.15)
    assert len(shots.LOCATION_XG) == 2 and all(0 < w < 0.5 for w in shots.LOCATION_XG)
    assert shots.LOCATION_XG[0] > shots.LOCATION_XG[1]  # a shot inside the box is worth more
    stat = MatchStats(home_box=10, away_box=None, home_outside=5, away_outside=4)
    assert shots.values(stat, "xg") is None and shots.values(stat, "box") is None
    assert shots.values(MatchStats(home_target=4, away_target=2), "target") == (4, 2)


def test_beta_is_recovered_on_simulated_matches():
    rng = np.random.default_rng(7)
    base = dc.design(seasons(rng, years=(2019,), rounds=24)[0], date(2020, 6, 1), half_life_days=None,
                     window_days=1460, new=lambda m, team: False)
    n = len(base.home_goals)
    home_x, away_x = rng.normal(scale=0.5, size=(n, 1)), rng.normal(scale=0.5, size=(n, 1))
    level, beta = 0.2, 0.4
    found = replace(base, home_x=home_x, away_x=away_x,
                    home_goals=rng.poisson(np.exp(level + 0.25 + beta * home_x[:, 0])).astype(float),
                    away_goals=rng.poisson(np.exp(level + beta * away_x[:, 0])).astype(float))
    assert found.size == base.size + 1
    fitted = dc.fit(found, ridge=1e-3)
    assert fitted.beta[0] == pytest.approx(beta, abs=0.1)  # about two standard errors on 1,440 sides

    theta = rng.normal(scale=0.2, size=found.size)
    theta[-2] = -0.08  # rho sits before the variables' coefficients
    for low_scores in (True, False):
        value, grad = dc.objective(theta, found, 0.5, low_scores)
        numeric = approx_fprime(theta, lambda t: dc.objective(t, found, 0.5, low_scores)[0], 1e-6)
        assert np.allclose(grad, numeric, rtol=1e-4, atol=1e-5)


def test_a_shot_rating_uses_only_earlier_days():
    matches, stats = seasons(np.random.default_rng(5))
    last = max(m.day for m in matches)
    changed = {**stats, **{m.id: MatchStats(home_box=60, away_box=0, home_outside=40, away_outside=0)
                           for m in matches if m.day == last}}

    def last_day(stat_table, model):
        rows = walkforward.run(model(stat_table), matches, score_from=2020)
        return [forecast.result for found, forecast in rows if found.day == last]

    for model in (lambda table: DixonColes(variables=(ShotVariable("box", table),)),
                  lambda table: Drifting(variables=(ShotVariable("box", table),))):
        assert last_day(stats, model) == last_day(changed, model)
    variable = ShotVariable("box", stats)
    for m in matches[:-3]:
        variable.learn([m])
    assert variable.features(matches[-1]) != ([0.0], [0.0])  # the test above compared real signals


def test_a_team_with_no_shot_history_is_neutral():
    matches, stats = seasons(np.random.default_rng(6), years=(2019,), rounds=4)
    variable = ShotVariable("box", stats)
    variable.learn(matches)
    newcomer = match(9999, date(2020, 8, 1), "X:New", "X:A", season=2020)
    assert variable.features(newcomer) == ([0.0], [0.0])
    few = [m for m in matches if "X:F" not in (m.home, m.away)] + [m for m in matches if m.home == "X:F"][:4]
    sparse = ShotVariable("box", stats)
    sparse.learn(sorted(few, key=lambda m: (m.day, m.id)))
    assert sparse.features(match(9998, date(2020, 8, 1), "X:F", "X:A", season=2020)) == ([0.0], [0.0])
    assert sparse.features(match(9997, date(2020, 8, 1), "X:B", "X:A", season=2020)) != ([0.0], [0.0])
    strong, weak = variable.features(match(9996, date(2020, 8, 1), "X:A", "X:F", season=2020))
    assert strong[0] > 0 > weak[0]  # the stronger side expects more shots than its league's average side
    unknown = ShotVariable("box", {})
    unknown.learn(matches)
    assert unknown.features(matches[-1]) == ([0.0], [0.0])


def test_drift_takes_its_coefficients_again_at_each_season_start():
    # shots in the box start with the first scored season, when every value is still 0: a coefficient taken only at
    # the drifting ratings' start would stay 0 for good
    matches, stats = seasons(np.random.default_rng(8), years=(2019, 2020, 2021), rounds=4)
    later = {id: stat for id, stat in stats.items() if matches[id].season >= 2020}
    model = Drifting(variables=(ShotVariable("box", later),))
    walkforward.run(model, matches, score_from=2020)
    assert model.started and model.beta and model.beta[0] != 0.0
