"""Game-read bench, task 9: pi-ratings (model 2, Constantinou and Fenton 2013) (D77)."""
from __future__ import annotations

from datetime import date, timedelta
import math

import pytest

from Scripts.data_platform.game_read.history import Match, Prices
from Scripts.data_platform.game_read.models.pi_ratings import PiRatings, expected


def match(id, day, home, away, goals, *, season=2018):
    return Match(id=id, competition="EPL", country="X", season=season, day=day, home=home, away=away,
                 home_goals=goals[0], away_goals=goals[1], pre=Prices(), close=Prices())


def test_the_expected_goal_difference_follows_the_paper():
    assert expected(0) == 0
    assert expected(1) == pytest.approx(10 ** (1 / 3) - 1) and expected(-1) == pytest.approx(-(10 ** (1 / 3) - 1))
    assert expected(3) == pytest.approx(9) and expected(-3) == pytest.approx(-9)


def test_a_home_win_bigger_than_expected_raises_both_home_team_ratings_and_lowers_the_away_teams():
    pi = PiRatings(learning=0.035, cross=0.7)
    pi.learn([match(1, date(2018, 8, 11), "X:A", "X:B", (3, 0))])
    change = 0.035 * 3 * math.log10(1 + 3)
    assert pi.home_rating["X:A"] == pytest.approx(change) and pi.away_rating["X:A"] == pytest.approx(0.7 * change)
    assert pi.away_rating["X:B"] == pytest.approx(-change) and pi.home_rating["X:B"] == pytest.approx(-0.7 * change)
    before = pi.home_rating["X:A"]
    pi.home_rating["X:A"] = 3.0  # expects a 9-goal win
    pi.learn([match(2, date(2018, 8, 18), "X:A", "X:C", (1, 0))])
    assert pi.home_rating["X:A"] < 3.0 and pi.away_rating["X:C"] > 0  # a narrow win falls short
    assert before > 0


def test_a_team_with_no_history_gets_a_finite_forecast():
    fresh = PiRatings().predict([match(1, date(2018, 8, 11), "X:New", "X:Other", (0, 0))])[0]
    assert sum(fresh.result) == pytest.approx(1) and all(math.isfinite(p) and p > 0 for p in fresh.result)
    assert sum(fresh.goal_difference.values()) == pytest.approx(1) and fresh.score_matrix is None


def test_a_stronger_home_team_is_favoured():
    teams = [f"X:{c}" for c in "ABCDEF"]
    pi, day, games = PiRatings(), date(2012, 8, 11), []
    for season in range(2012, 2019):  # 210 matches: past the 200 before the logits are fitted
        for home in teams:
            for away in teams:
                if home != away:
                    goals = (2, 0) if home == "X:A" else (0, 2) if away == "X:A" else (1, 1)
                    games.append(match(len(games) + 1, day + timedelta(days=len(games)), home, away, goals,
                                       season=season))
    for game in games:
        pi.learn([game])
    strong = pi.predict([match(9999, date(2019, 8, 10), "X:A", "X:F", (0, 0), season=2019)])[0]
    weak = pi.predict([match(9998, date(2019, 8, 10), "X:F", "X:A", (0, 0), season=2019)])[0]
    assert strong.result[0] > strong.result[2] and weak.result[2] > weak.result[0]
