"""Game-read bench, task 11: drifting ratings (model 4) (D77)."""
from __future__ import annotations

from datetime import date, timedelta
import math

import pytest

from Scripts.data_platform.game_read.history import Match, Prices
from Scripts.data_platform.game_read.models.dixon_coles import DixonColes
from Scripts.data_platform.game_read.models.drifting import Drifting

OTHERS = ["X:B", "X:C", "X:D", "X:E", "X:F"]
START = date(2018, 8, 4)


def match(id, day, home, away, goals, season=2018):
    return Match(id=id, competition="EPL", country="X", season=season, day=day, home=home, away=away,
                 home_goals=goals[0], away_goals=goals[1], pre=Prices(), close=Prices())


def round_of(n, *, jump_from=30):
    """Week n: X:A against one of the others (home and away in turn), the other four in two matches. X:A scores 1
    a match, then 2 from week ``jump_from``; everyone else scores 1."""
    day, scored = START + timedelta(weeks=n), 2 if n >= jump_from else 1
    opponent, rest = OTHERS[n % 5], [t for t in OTHERS if t != OTHERS[n % 5]]
    first = (match(3 * n, day, "X:A", opponent, (scored, 1)) if n % 2 == 0
             else match(3 * n, day, opponent, "X:A", (1, scored)))
    return [first, match(3 * n + 1, day, rest[0], rest[1], (1, 1)), match(3 * n + 2, day, rest[2], rest[3], (1, 1))]


def test_a_team_whose_scoring_jumps_is_followed_faster_than_by_model_3():
    drifting, base = Drifting(gain=1e-3, summer_widening=0.05), DixonColes(half_life_days=365)
    for n in range(30):
        drifting.learn(round_of(n))
        base.learn(round_of(n))
    drifting.predict(round_of(30))  # starts from model 3's fit here
    for n in range(30, 40):
        drifting.learn(round_of(n))
        base.learn(round_of(n))
    ahead = match(999, START + timedelta(weeks=40), "X:A", "X:B", (0, 0))
    base.predict([ahead])
    followed = drifting.means(ahead)[0]
    smoothed = base.fitted.means_for("EPL", "X:A", "X:B", False, False)[0]
    assert abs(followed - 2.0) < abs(smoothed - 2.0)
    assert followed > 1.4


def test_uncertainty_grows_over_the_summer():
    model = Drifting(gain=1e-4, summer_widening=0.05)
    for n in range(30):
        model.learn(round_of(n, jump_from=99))
    model.predict(round_of(30))
    last = START + timedelta(weeks=29)
    now = model.uncertainty("X:A", last, 2018)
    later = model.uncertainty("X:A", last + timedelta(days=60), 2018)
    next_season = model.uncertainty("X:A", last + timedelta(days=60), 2019)
    assert now[0] < later[0] < next_season[0] and now[1] < later[1] < next_season[1]
    assert later[0] - now[0] == pytest.approx(60 * 1e-4)
    assert next_season[0] - later[0] == pytest.approx(0.05)


def test_a_team_with_no_history_gets_a_finite_forecast():
    fresh = Drifting().predict([match(1, START, "X:New", "X:Other", (0, 0))])[0]
    assert fresh.score_matrix.sum() == pytest.approx(1) and all(math.isfinite(p) and p > 0 for p in fresh.result)
    model = Drifting()
    for n in range(30):
        model.learn(round_of(n, jump_from=99))
    model.predict(round_of(30))
    newcomer = match(9999, date(2019, 8, 10), "X:Y", "X:A", (0, 0), season=2019)
    forecast = model.predict([newcomer])[0]
    assert all(math.isfinite(p) and p > 0 for p in forecast.result)
    assert forecast.score_matrix.sum() == pytest.approx(1)
    model.learn([newcomer])  # a newcomer can be learned too
    assert math.isfinite(model.means(newcomer)[0])
