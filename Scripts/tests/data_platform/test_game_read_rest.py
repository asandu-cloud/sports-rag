"""Game-read bench, phase 2 task 4: rest (D12's C1) in models 3–4 (D77)."""
from __future__ import annotations

from datetime import date, timedelta
import math

import numpy as np
import pytest

from Scripts.data_platform.game_read.history import Match, Prices
from Scripts.data_platform.game_read.models.dixon_coles import DixonColes
from Scripts.data_platform.game_read.models.rest import RestVariable
from Scripts.data_platform.game_read.stats import MatchStats

TEAMS = [f"X:{c}" for c in "ABCDEF"]


def match(id, day, home, away, goals=(1, 1), season=2019):
    return Match(id=id, competition="EPL", country="X", season=season, day=day, home=home, away=away,
                 home_goals=goals[0], away_goals=goals[1], pre=Prices(), close=Prices())


def scaled(rest):
    return (min(rest, 7) - 7) / 7


def test_short_rest_lowers_scoring_on_simulated_matches():
    """Equal teams; a side's goals fall with its own short rest (0.4) and rise with its opponent's (−0.25)."""
    rng = np.random.default_rng(4)
    own, opponent = 0.4, -0.25
    matches, stats = [], {}
    for i in range(1800):
        home, away = rng.choice(TEAMS, size=2, replace=False)
        rest = rng.integers(1, 11, size=2)
        means = (math.exp(0.35 + own * scaled(rest[0]) + opponent * scaled(rest[1])),
                 math.exp(0.15 + own * scaled(rest[1]) + opponent * scaled(rest[0])))
        found = match(i, date(2019, 8, 1) + timedelta(days=i // 6), str(home), str(away),
                      (int(rng.poisson(means[0])), int(rng.poisson(means[1]))))
        matches.append(found)
        stats[i] = MatchStats(home_rest=int(rest[0]), away_rest=int(rest[1]))
    model = DixonColes(half_life_days=None, variables=(RestVariable(stats),))
    for day in sorted({m.day for m in matches}):
        model.learn([m for m in matches if m.day == day])
    ahead = matches[-1].day + timedelta(days=1)
    tired, rested = match(5000, ahead, "X:A", "X:B"), match(5001, ahead, "X:A", "X:B")
    stats[5000], stats[5001] = MatchStats(home_rest=2, away_rest=7), MatchStats(home_rest=7, away_rest=7)
    model.predict([tired, rested])
    assert model.name == "dc+rest"
    assert model.fitted.beta[0] == pytest.approx(own, abs=0.15)
    assert model.fitted.beta[1] == pytest.approx(opponent, abs=0.15)
    expected = lambda found: model.fitted.means_for("EPL", "X:A", "X:B", False, False, *model.features(found))
    assert expected(tired)[0] < expected(rested)[0]  # short rest lowers the tired side's scoring
    assert expected(tired)[1] > expected(rested)[1]  # and raises its opponent's


def test_missing_rest_is_neutral():
    variable = RestVariable({1: MatchStats(home_rest=None, away_rest=3), 2: MatchStats(home_rest=12, away_rest=7)})
    day = date(2020, 1, 1)
    assert variable.name == "rest" and variable.columns == 2
    assert variable.features(match(1, day, "X:A", "X:B")) == ([0.0, scaled(3)], [scaled(3), 0.0])
    assert variable.features(match(2, day, "X:A", "X:B")) == ([0.0, 0.0], [0.0, 0.0])  # 7 days or more: rested
    assert variable.features(match(3, day, "X:A", "X:B")) == ([0.0, 0.0], [0.0, 0.0])  # no statistics at all
    variable.learn([match(1, day, "X:A", "X:B")])  # nothing to learn
