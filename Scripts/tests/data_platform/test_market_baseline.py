"""Market baseline for research comparisons (rework A1 step 6)."""
from datetime import date, timedelta
import math
import sqlite3

import pytest

from Scripts.data_platform.features import market_baseline as mb
from Scripts.ops import football_data_import as fd


def history(tmp_path, matches):
    """matches: fixture_id -> list of (market, bookmaker, closing, selection, price, suspect)."""
    path = tmp_path / "fd.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(fd.SCHEMA)
    for i, (fid, odds) in enumerate(matches.items(), start=1):
        db.execute("INSERT INTO matches (id, fixture_api_id, season, link_status) VALUES (?, ?, 2023, 'matched')", (i, fid))
        db.executemany("INSERT INTO odds VALUES (?, ?, ?, ?, ?, NULL, ?, '', ?)",
                       [(i, m, b, c, s, p, sus) for m, b, c, s, p, sus in odds])
    # A fixture whose football-data score disagreed with ours (an awarded result).
    db.execute("INSERT INTO matches (id, fixture_api_id, season, link_status) VALUES (999, 999, 2023, 'score_mismatch')")
    db.executemany("INSERT INTO odds VALUES (999, '1x2', 'Bet365', 1, ?, NULL, 2.5, '', 0)", [("home",), ("draw",), ("away",)])
    db.commit()
    db.close()
    return path


def x12(book, h, d, a, closing=1, suspect=0):
    return [("1x2", book, closing, "home", h, suspect), ("1x2", book, closing, "draw", d, suspect),
            ("1x2", book, closing, "away", a, suspect)]


def ou(book, over, under, closing=1):
    return [("ou25", book, closing, "over", over, 0), ("ou25", book, closing, "under", under, 0)]


def fair(prices):
    inverse = sum(1 / p for p in prices)
    return [(1 / p) / inverse for p in prices]


def test_market_probability_is_the_median_of_individual_books_closing_prices(tmp_path):
    path = history(tmp_path, {7: x12("Pinnacle", 2.0, 3.5, 4.0) + x12("Bet365", 2.1, 3.4, 3.8)
                              + x12("Market maximum", 2.2, 3.7, 4.3) + x12("Market average", 2.05, 3.45, 3.9)
                              + x12("Betano", 1.5, 4.0, 6.0, closing=0)})
    p = mb.market_probabilities(path)[7]["winner"]
    expected = [(a + b) / 2 for a, b in zip(fair([2.0, 3.5, 4.0]), fair([2.1, 3.4, 3.8]))]
    assert p["p"] == pytest.approx(expected)
    assert (p["books"], p["basis"]) == (2, "closing")


def test_averages_stand_in_only_when_no_single_bookmaker_priced_it(tmp_path):
    path = history(tmp_path, {7: x12("BetBrain average", 2.0, 3.5, 4.0, closing=0)
                              + x12("BetBrain maximum", 2.3, 3.8, 4.4, closing=0)})
    assert mb.market_probabilities(path, timing="closing") == {}
    p = mb.market_probabilities(path, timing="best_available")[7]["winner"]
    assert p["p"] == pytest.approx(fair([2.0, 3.5, 4.0])) and p["basis"] == "pre_closing_average"


def test_suspect_pinnacle_prices_are_left_out(tmp_path):
    path = history(tmp_path, {7: x12("Pinnacle", 1.5, 4.0, 6.0, suspect=1) + x12("Bet365", 2.1, 3.4, 3.8)})
    p = mb.market_probabilities(path)[7]["winner"]
    assert p["p"] == pytest.approx(fair([2.1, 3.4, 3.8])) and p["books"] == 1


def test_over_under_market(tmp_path):
    path = history(tmp_path, {7: ou("Bet365", 1.8, 2.0) + ou("Pinnacle", 1.9, 1.9)})
    p = mb.market_probabilities(path)[7]["over2.5"]
    assert p["p"] == pytest.approx((fair([1.8, 2.0])[0] + 0.5) / 2)


def forecast(fid, week, winner_p, winner_y, over_p=0.5, over_y=1):
    return {"fixture_id": fid, "week": week, "scores": {
        "derived": {"winner": {"p": winner_p, "y": winner_y}},
        "totals": {"2.5": {"binary": {"p": over_p, "y": over_y}}}}}


def weeks(n):
    start = date(2023, 1, 2)
    return [f"{(start + timedelta(weeks=i)).isocalendar()[0]}-W{(start + timedelta(weeks=i)).isocalendar()[1]:02d}"
            for i in range(n)]


def test_forecast_equal_to_the_market_has_zero_gap():
    market = {i: {"winner": {"p": [0.5, 0.3, 0.2]}} for i in range(40)}
    rows = [forecast(i, w, [0.5, 0.3, 0.2], i % 3) for i, w in enumerate(weeks(40))]
    result = mb.compare_with_market(rows, market, "winner")
    assert result["mean_difference"] == pytest.approx(0)
    assert not result["beats_market"] and not result["worse_than_market"]


def test_a_sharper_forecast_beats_the_market_and_a_worse_one_does_not():
    market = {i: {"winner": {"p": [0.4, 0.3, 0.3]}} for i in range(60)}
    good = [forecast(i, w, [0.7, 0.15, 0.15], 0) for i, w in enumerate(weeks(60))]
    bad = [forecast(i, w, [0.1, 0.45, 0.45], 0) for i, w in enumerate(weeks(60))]
    better = mb.compare_with_market(good, market, "winner")
    assert better["beats_market"] and better["paired_week_delta95"][1] < 0
    assert better["forecast_logloss"] == pytest.approx(-math.log(0.7))
    assert mb.compare_with_market(bad, market, "winner")["worse_than_market"]


def test_fixtures_without_a_market_price_are_counted_not_dropped_silently():
    market = {0: {"over2.5": {"p": 0.55}}}
    rows = [forecast(0, "2023-W01", [1, 0, 0], 0, over_p=0.6, over_y=1), forecast(1, "2023-W01", [1, 0, 0], 0)]
    result = mb.compare_with_market(rows, market, "over2.5")
    assert (result["forecasts"], result["paired"]) == (2, 1)
    assert result["market_logloss"] == pytest.approx(-math.log(0.55))


def test_fixtures_whose_scores_disagreed_with_ours_are_left_out(tmp_path):
    path = history(tmp_path, {7: x12("Bet365", 2.1, 3.4, 3.8)})
    assert set(mb.market_probabilities(path)) == {7}


def test_report_reads_saved_forecasts_by_candidate_and_scores_each_event(tmp_path):
    import gzip, json
    from Scripts.ops import market_baseline_report as report
    path = history(tmp_path, {i: x12("Bet365", 2.0, 3.5, 4.0) + ou("Bet365", 1.9, 1.9) for i in range(30)})
    lines = [dict(forecast(i, w, [0.5, 0.3, 0.2], 0, over_p=0.6, over_y=1), candidate_id=c, market=m)
             for c in ("control", "challenger") for m in ("goals", "corners") for i, w in enumerate(weeks(30))]
    with gzip.open(tmp_path / "forecasts.jsonl.gz", "wt") as handle:
        handle.write("\n".join(json.dumps(line) for line in lines))
    forecasts = report.load_forecasts([tmp_path / "forecasts.jsonl.gz"])
    assert {k: len(v) for k, v in forecasts.items()} == {"control": 30, "challenger": 30}
    result = report.build(forecasts, db_path=path)
    assert set(result["candidates"]) == {"control", "challenger"}
    assert result["candidates"]["control"]["winner"]["paired"] == 30
    assert "| control |" in report.markdown(result)
