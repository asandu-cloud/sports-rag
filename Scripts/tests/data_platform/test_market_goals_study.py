"""M1 study: market goal expectations against the engine's goal means (rework A3.2)."""
from datetime import date, timedelta
import math
import sqlite3

import pytest

from data_platform import market_goals as mg
from Scripts.ops import football_data_import as fd
from Scripts.ops import market_goals_study as study


def priced_history(tmp_path, fixtures):
    """fixtures: fixture_id -> (home_mean, away_mean, home_goals, away_goals); prices with a 5% proportional margin."""
    path = tmp_path / "fd.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(fd.SCHEMA)
    for i, (fid, (lh, la, hg, ag)) in enumerate(fixtures.items(), start=1):
        day = (date(2023, 1, 2) + timedelta(days=7 * (i // 5))).isoformat()
        db.execute("INSERT INTO matches (id, fixture_api_id, season, match_date, fthg, ftag, link_status) "
                   "VALUES (?, ?, 2023, ?, ?, ?, 'matched')", (i, fid, day, hg, ag))
        result = mg.result_probabilities(lh, la)
        over = mg.over_probability(lh, la, 2.5)
        home_ah = mg.handicap_probability(lh, la, -0.25)
        rows = [("1x2", s, None, p) for s, p in zip(("home", "draw", "away"), result)]
        rows += [("ou25", "over", None, over), ("ou25", "under", None, 1 - over)]
        rows += [("ah", "home", -0.25, home_ah), ("ah", "away", -0.25, 1 - home_ah)]
        for book in ("A", "B"):
            for market, side, line, p in rows:
                db.execute("INSERT INTO odds VALUES (?, ?, ?, 1, ?, ?, ?, '', 0)", (i, market, book, side, line, 1 / (p * 1.05)))
    db.commit()
    db.close()
    return path


def test_targets_rebuild_the_market_and_m1_recovers_the_means(tmp_path):
    path = priced_history(tmp_path, {7: (1.7, 1.0, 2, 1)})
    targets = study.market_targets(path)[7]
    assert set(targets) == {"1x2", "totals", "handicaps", "week", "goals"}
    fit = mg.solve(targets)
    assert fit["home"] == pytest.approx(1.7, abs=0.05) and fit["away"] == pytest.approx(1.0, abs=0.05)


def test_comparison_scores_both_means_on_the_same_fixtures(tmp_path):
    fixtures = {i: (1.5, 1.2, i % 3, (i + 1) % 2) for i in range(1, 41)}
    path = priced_history(tmp_path, fixtures)
    engine = {i: (1.5, 1.2) for i in range(1, 41)}
    engine[999] = (1.0, 1.0)  # no market price: counted as not compared
    result = study.compare(path, engine, resamples=100)
    assert result["compared"] == 40 and result["engine_only"] == 1
    expected = -sum(math.log(mg.poisson_pmf(1.5, i % 3)) + math.log(mg.poisson_pmf(1.2, (i + 1) % 2)) for i in range(1, 41)) / 40
    assert result["engine_nll"] == pytest.approx(expected)
    assert result["mean_difference"] == pytest.approx(0, abs=0.01)  # same means, same score
