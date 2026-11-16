"""Game-read bench, task 10: Dixon–Coles ratings with time decay (model 3) (D77)."""
from __future__ import annotations

from datetime import date, timedelta
import math

import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.linear_model import PoissonRegressor

from Scripts.data_platform import market_goals
from Scripts.data_platform.game_read.history import Match, Prices
from Scripts.data_platform.game_read.models import dixon_coles as dc
from Scripts.data_platform.game_read.models.dixon_coles import DixonColes


def season(year, strengths, rng, *, league="EPL", first_id=0, start=None, low=False):
    """Every pair home and away; goals from Poisson means set by the teams' strengths."""
    out, day, teams = [], start or date(year, 8, 10), list(strengths)
    for home in teams:
        for away in teams:
            if home == away:
                continue
            h = math.exp(0.25 + strengths[home] - strengths[away] * 0.5)
            a = math.exp(0.05 + strengths[away] - strengths[home] * 0.5)
            goals = (int(rng.poisson(h)), int(rng.poisson(a)))
            if low and len(out) % 3 == 0:
                goals = (len(out) % 2, len(out) % 2)  # extra 0-0 and 1-1
            out.append(Match(id=first_id + len(out), competition=league, country="X", season=year,
                             day=day + timedelta(days=len(out) // 3), home=home, away=away, home_goals=goals[0],
                             away_goals=goals[1], pre=Prices(), close=Prices()))
    return out


TEAMS = {f"X:{c}": s for c, s in zip("ABCDEFGH", (0.5, 0.3, 0.2, 0.0, 0.0, -0.1, -0.3, -0.4))}


def test_the_gradient_matches_a_numerical_gradient():
    rng = np.random.default_rng(3)
    matches = season(2018, TEAMS, rng) + season(2018, {"X:P": 0.2, "X:Q": 0.0, "X:R": -0.1}, rng, league="LaLiga",
                                                 first_id=500)
    design = dc.design(matches, date(2019, 6, 1), half_life_days=180, window_days=1460,
                       new=lambda m, team: team == "X:H")
    theta = rng.normal(scale=0.2, size=design.size)
    theta[-1] = -0.08
    for low_scores in (True, False):
        value, grad = dc.objective(theta, design, 0.5, low_scores)
        numeric = approx_fprime(theta, lambda t: dc.objective(t, design, 0.5, low_scores)[0], 1e-6)
        assert np.allclose(grad, numeric, rtol=1e-4, atol=1e-5)


def test_without_decay_rho_or_ridge_it_is_the_poisson_regression():
    rng = np.random.default_rng(4)
    matches = season(2018, TEAMS, rng)
    design = dc.design(matches, date(2019, 6, 1), half_life_days=None, window_days=1460, new=lambda m, team: False)
    fitted = dc.fit(design, ridge=0.0, low_scores=False)
    ours = np.concatenate(fitted.means(design))
    teams, n = design.teams, len(matches)
    X = np.zeros((2 * n, 2 + 2 * len(teams)))  # league mean, home advantage, attacks, defences
    for i, m in enumerate(matches):
        h, a = teams.index(m.home), teams.index(m.away)
        X[i, [0, 1, 2 + h]] = 1
        X[i, 2 + len(teams) + a] = -1
        X[n + i, [0, 2 + a]] = 1
        X[n + i, 2 + len(teams) + h] = -1
    y = np.array([m.home_goals for m in matches] + [m.away_goals for m in matches])
    theirs = PoissonRegressor(alpha=0, fit_intercept=False, tol=1e-12, max_iter=10_000).fit(X, y).predict(X)
    assert ours == pytest.approx(theirs, rel=1e-4)
    assert fitted.rho == 0


def test_older_matches_count_less():
    rng = np.random.default_rng(5)
    early = dict(TEAMS, **{"X:A": 1.0})
    late = dict(TEAMS, **{"X:A": -0.6})
    matches = (season(2016, early, rng, start=date(2016, 8, 1)) + season(2018, late, rng, first_id=1000,
                                                                         start=date(2018, 8, 1)))
    today = date(2019, 1, 15)
    flat = dc.fit(dc.design(matches, today, half_life_days=None, window_days=1460, new=lambda m, t: False), 0.0)
    recent = dc.fit(dc.design(matches, today, half_life_days=120, window_days=1460, new=lambda m, t: False), 0.0)
    def relative(fitted):  # without a ridge, only differences between teams are pinned down
        return fitted.attack["X:A"] - np.mean(list(fitted.attack.values()))
    assert relative(recent) < relative(flat)
    assert relative(recent) < 0  # rated weak now


def test_the_score_matrix_sums_to_one_and_rho_moves_low_scores():
    rng = np.random.default_rng(6)
    model = DixonColes(half_life_days=365)
    for game in season(2017, TEAMS, rng, low=True) + season(2018, TEAMS, rng, first_id=1000, low=True):
        model.learn([game])
    forecast = model.predict([Match(id=9999, competition="EPL", country="X", season=2018, day=date(2019, 6, 1),
                                    home="X:D", away="X:E", home_goals=0, away_goals=0, pre=Prices(),
                                    close=Prices())])[0]
    matrix = forecast.score_matrix
    assert matrix.shape == (13, 13) and matrix.sum() == pytest.approx(1)
    assert model.fitted.rho < 0
    home, away = model.fitted.means_for("EPL", "X:D", "X:E", False, False)
    plain = market_goals._matrix(home, away)
    plain = plain / plain.sum()
    assert matrix[0, 0] > plain[0, 0] and matrix[1, 1] > plain[1, 1]
    assert sum(forecast.result) == pytest.approx(1)


def test_a_promoted_team_with_no_history_gets_a_finite_forecast():
    fresh = DixonColes().predict([Match(id=1, competition="EPL", country="X", season=2018, day=date(2018, 8, 11),
                                        home="X:New", away="X:Other", home_goals=0, away_goals=0, pre=Prices(),
                                        close=Prices())])[0]
    assert fresh.score_matrix.sum() == pytest.approx(1) and all(math.isfinite(p) for p in fresh.result)
    rng = np.random.default_rng(7)
    model = DixonColes(half_life_days=365)
    for game in season(2017, TEAMS, rng) + season(2018, dict(TEAMS, **{"X:Z": -0.5}), rng, first_id=1000):
        model.learn([game])  # X:Z joins in 2018: its first season in the division
    promoted = Match(id=9999, competition="EPL", country="X", season=2019, day=date(2019, 8, 10), home="X:Y",
                     away="X:A", home_goals=0, away_goals=0, pre=Prices(), close=Prices())
    model.learn([Match(id=9998, competition="EPL", country="X", season=2019, day=date(2019, 8, 9), home="X:B",
                       away="X:C", home_goals=1, away_goals=0, pre=Prices(), close=Prices())])
    forecast = model.predict([promoted])[0]  # X:Y has never played
    assert all(math.isfinite(p) and p > 0 for p in forecast.result) and forecast.score_matrix.sum() == pytest.approx(1)
    assert model.fitted.promoted_attack < 0  # X:Z, new in 2018, was weak
