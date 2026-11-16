"""Game-read bench, task 8: Elo (model 1) and the ordered logit (D77)."""
from __future__ import annotations

from datetime import date, timedelta
import math

import numpy as np
import pytest

from Scripts.data_platform.game_read import ordered
from Scripts.data_platform.game_read.history import Match, Prices
from Scripts.data_platform.game_read.models.elo import Elo, multiplier


def match(id, day, home, away, goals, *, league="EPL", season=2018):
    return Match(id=id, competition=league, country="X", season=season, day=day, home=home, away=away,
                 home_goals=goals[0], away_goals=goals[1], pre=Prices(), close=Prices())


def test_ordered_logit_recovers_known_cutpoints():
    rng = np.random.default_rng(1)
    x = rng.normal(size=20_000)
    latent = 1.3 * x + rng.logistic(size=x.size)
    y = np.digitize(latent, [-1.0, 0.5])
    fit = ordered.fit(x, y, 3)
    assert fit.cuts == pytest.approx([-1.0, 0.5], abs=0.08) and fit.beta == pytest.approx([1.3], abs=0.06)
    p = fit.probabilities(np.array([0.0, 2.0]))
    assert p.shape == (2, 3) and np.allclose(p.sum(axis=1), 1) and p[1, 2] > p[0, 2]


def test_a_bigger_win_moves_the_rating_more():
    elo = Elo(k=20)
    day = date(2018, 8, 11)
    elo.learn([match(1, day, "X:A", "X:B", (1, 0)), match(2, day, "X:C", "X:D", (4, 0))])
    narrow, wide = elo.ratings["X:A"] - 1500, elo.ratings["X:C"] - 1500
    assert narrow > 0 and wide / narrow == pytest.approx(multiplier(4) / multiplier(1))
    assert (multiplier(0), multiplier(1), multiplier(2), multiplier(3), multiplier(5)) == (1, 1, 1.5, 14 / 8, 2)
    assert elo.ratings["X:B"] == pytest.approx(1500 - narrow)


def league_season(season, teams, first_id, *, league="EPL", strong=None):
    """Every pair once; the ``strong`` team wins all its matches 2-0, the rest draw 1-1."""
    out, day = [], date(season, 8, 10)
    for i, home in enumerate(teams):
        for away in teams[i + 1:]:
            goals = (2, 0) if home == strong else (0, 2) if away == strong else (1, 1)
            out.append(match(first_id + len(out), day + timedelta(days=len(out)), home, away, goals, league=league,
                             season=season))
    return out


def test_ratings_are_pulled_towards_the_league_mean_each_summer():
    teams = [f"X:{c}" for c in "ABCDEF"]
    elo = Elo(k=20, carry=0.7)
    elo.learn(league_season(2018, teams, 1, strong="X:A"))
    before = {t: elo.ratings[t] for t in teams}
    mean = sum(before.values()) / len(before)
    elo.predict([match(100, date(2019, 8, 10), "X:B", "X:A", (0, 0), season=2019)])
    for team in teams:
        assert elo.ratings[team] == pytest.approx(mean + 0.7 * (before[team] - mean))


def test_a_promoted_team_starts_at_the_level_of_the_teams_relegated():
    teams = [f"X:{c}" for c in "ABCDEF"]
    elo = Elo(k=20, carry=1.0)
    elo.learn(league_season(2018, teams, 1, strong="X:A"))
    elo.learn([match(50, date(2018, 12, 1), "X:F", "X:E", (0, 3)), match(51, date(2018, 12, 2), "X:D", "X:C", (0, 1))])
    # points: A 15, C 7, E 7, B 4, D 4, F 4: the bottom three by points (ties by rating) are B, D and F
    bottom = sorted(teams, key=lambda t: (elo._points[("EPL", 2018)][t], elo.ratings[t]))[:3]
    expected = sum(elo.ratings[t] for t in bottom) / 3
    elo.predict([match(100, date(2019, 8, 10), "X:G", "X:A", (0, 0), season=2019)])
    assert elo.ratings["X:G"] == pytest.approx(expected)


def test_a_team_coming_down_starts_at_the_level_of_the_teams_promoted():
    top, second = [f"X:{c}" for c in "ABCD"], [f"X:{c}" for c in "PQRS"]
    elo = Elo(k=20, carry=1.0)
    elo.learn(league_season(2018, top, 1, strong="X:A") + league_season(2018, second, 50, league="Championship",
                                                                       strong="X:P"))
    best = sorted(second, key=lambda t: (elo._points[("Championship", 2018)][t], elo.ratings[t]))[-3:]
    expected = sum(elo.ratings[t] for t in best) / 3
    elo.predict([match(100, date(2019, 8, 3), "X:D", "X:Q", (0, 0), league="Championship", season=2019)])
    assert elo.ratings["X:D"] == pytest.approx(expected)


def test_a_relegated_team_that_starts_below_first_leaves_the_top_league_as_it_ended():
    top, second = [f"X:{c}" for c in "ABCD"], [f"X:{c}" for c in "PQRS"]
    elo = Elo(k=20, carry=0.5)
    elo.learn(league_season(2018, top, 1, strong="X:A") + league_season(2018, second, 50, league="Championship",
                                                                       strong="X:P"))
    ended = {t: elo.ratings[t] for t in top}
    mean = sum(ended.values()) / len(ended)
    # the Championship starts first: D comes down; then the Premier League's summer must still read D as it ended
    elo.predict([match(100, date(2019, 8, 3), "X:D", "X:Q", (0, 0), league="Championship", season=2019)])
    in_second = elo.ratings["X:D"]
    elo.predict([match(101, date(2019, 8, 10), "X:A", "X:B", (0, 0), season=2019)])
    assert elo.ratings["X:A"] == pytest.approx(mean + 0.5 * (ended["X:A"] - mean))
    assert elo.ratings["X:D"] == in_second  # its Championship start is kept


def test_a_team_with_no_history_gets_a_finite_forecast():
    fresh = Elo().predict([match(1, date(2018, 8, 11), "X:New", "X:Other", (0, 0))])[0]
    assert sum(fresh.result) == pytest.approx(1) and all(math.isfinite(p) and p > 0 for p in fresh.result)
    assert sum(fresh.goal_difference.values()) == pytest.approx(1) and fresh.score_matrix is None
    teams = [f"X:{c}" for c in "ABCDEF"]
    elo = Elo()
    elo.learn(league_season(2018, teams, 1, strong="X:A"))
    later = elo.predict([match(100, date(2018, 12, 1), "X:Z", "X:A", (0, 0))])[0]  # a newcomer mid-season
    assert sum(later.result) == pytest.approx(1) and all(math.isfinite(p) for p in later.result)


def test_a_stronger_home_team_is_favoured_and_the_draw_peaks_at_an_even_gap():
    teams = [f"X:{c}" for c in "ABCDEFGH"]
    elo = Elo(k=30)
    for season in range(2012, 2018):
        elo.learn(league_season(season, teams, season * 100, strong="X:A"))
        elo.learn(league_season(season, teams[::-1], season * 100 + 50, strong="X:B"))
    strong = elo.predict([match(1, date(2018, 8, 11), "X:A", "X:H", (0, 0), season=2018)])[0]
    assert strong.result[0] > strong.result[2]
    draws = {gap: elo._forecast(gap).result[1] for gap in (-300, -150, 0, 150, 300)}
    assert draws[0] > draws[150] > draws[300] and draws[0] > draws[-150] > draws[-300]
