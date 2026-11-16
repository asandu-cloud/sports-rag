"""Game-read bench, task 1: the history loader (D77)."""
from __future__ import annotations

from contextlib import closing
import os
import sqlite3

import pytest

from Scripts.data_platform import devig
from Scripts.data_platform.game_read import history
from Scripts.ops.football_data_import import SCHEMA

MATCHES = [
    # id, div, competition, season, day, home, away, home goals, away goals
    (1, "E0", "EPL", 2020, "2020-09-12", "Arsenal", "Chelsea", 2, 1),
    (2, "E0", "EPL", 2018, "2018-08-11", "Everton", "Burnley", 0, 0),
    (3, "E0", "EPL", 2020, "2020-09-19", "Fulham", "Leeds", 1, 1),
    (4, "E1", "Championship", 2019, "2019-08-10", "Fulham", "Brentford", 3, 2),
    (5, "E0", "EPL", 2020, "2020-09-20", "Tottenham", "Wolves", None, None),
    (6, "SP1", "LaLiga", 2025, "2025-08-16", "Getafe", "Elche", 1, 0),
]
SIDES = {"1x2": ("home", "draw", "away"), "ah": ("home", "away"), "ou25": ("over", "under")}
ODDS = [
    # match, market, bookmaker, closing, line, prices (in SIDES order), suspect
    (1, "1x2", "Market average", 0, None, (2.10, 3.40, 3.60), 0),
    (1, "1x2", "Market maximum", 0, None, (2.20, 3.60, 3.80), 0),
    (1, "1x2", "Market average", 1, None, (2.00, 3.50, 3.80), 0),
    (1, "ah", "Market average", 0, -0.5, (1.95, 1.90), 0),
    (1, "ah", "Market maximum", 0, -0.5, (2.00, 1.96), 0),
    (1, "ah", "Market average", 1, -0.75, (2.02, 1.86), 0),
    (1, "ou25", "Market average", 0, None, (1.80, 2.05), 0),
    (1, "ou25", "Market maximum", 0, None, (1.85, 2.12), 0),
    (1, "ou25", "Market average", 1, None, (1.75, 2.10), 0),
    (1, "1x2", "Bet365", 0, None, (2.05, 3.40, 3.50), 0),
    (2, "1x2", "BetBrain average", 0, None, (1.80, 3.60, 4.50), 0),
    (2, "1x2", "BetBrain maximum", 0, None, (1.90, 3.80, 4.80), 0),
    (2, "1x2", "Pinnacle", 0, None, (1.85, 3.70, 4.70), 0),
    (2, "1x2", "Bet365", 0, None, (1.80, 3.60, 4.60), 0),
    (3, "1x2", "Market average", 0, None, (2.40, 3.30, 3.00), 0),
    (3, "ou25", "Market average", 0, None, (1.90, 1.95), 0),
    (4, "1x2", "Market average", 0, None, (1.70, 3.80, 5.00), 0),
    (5, "1x2", "Market average", 0, None, (1.60, 4.00, 5.50), 0),
    (6, "1x2", "Pinnacle", 0, None, (1.95, 3.30, 4.30), 1),
    (6, "1x2", "Bet365", 0, None, (1.90, 3.25, 4.20), 0),
    (6, "1x2", "Market average", 0, None, (1.92, 3.20, 4.10), 0),
]


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "football_data.sqlite3"
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO matches (id, file, div, competition, season, match_date, home_name, away_name, fthg, ftag)"
            " VALUES (?, 'test.csv', ?, ?, ?, ?, ?, ?, ?, ?)", MATCHES)
        conn.executemany(
            "INSERT INTO odds (match_id, market, bookmaker, closing, selection, line, price, source_column, suspect)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 'test', ?)",
            [(match, market, book, closing_price, side, line, price, suspect)
             for match, market, book, closing_price, line, prices, suspect in ODDS
             for side, price in zip(SIDES[market], prices)])
        conn.commit()
    return path


def by_id(path, **kwargs):
    return {m.id: m for m in history.load_matches(path, **kwargs)}


def test_load_matches_reads_the_average_and_best_prices_before_kickoff_and_at_the_close(db):
    match = by_id(db)[1]
    assert (match.competition, match.country, match.season, match.day.isoformat()) == (
        "EPL", "England", 2020, "2020-09-12")
    assert (match.home, match.away, match.home_goals, match.away_goals) == ("England:Arsenal", "England:Chelsea", 2, 1)
    assert match.pre.result_avg == (2.10, 3.40, 3.60)
    assert match.pre.result_max == (2.20, 3.60, 3.80)
    assert match.close.result_avg == (2.00, 3.50, 3.80)
    assert (match.pre.handicap_line, match.pre.handicap_avg, match.pre.handicap_max) == (-0.5, (1.95, 1.90),
                                                                                       (2.00, 1.96))
    assert (match.close.handicap_line, match.close.handicap_avg) == (-0.75, (2.02, 1.86))
    assert (match.pre.over25_avg, match.pre.over25_max, match.close.over25_avg) == ((1.80, 2.05), (1.85, 2.12),
                                                                                  (1.75, 2.10))
    assert match.close.result_max is None


def test_before_2019_the_average_is_betbrains(db):
    match = by_id(db)[2]
    assert match.pre.result_avg == (1.80, 3.60, 4.50)
    assert match.pre.result_max == (1.90, 3.80, 4.80)


def test_a_missing_market_stays_missing_and_the_others_load(db):
    match = by_id(db)[3]
    assert match.pre.handicap_avg is None and match.pre.handicap_line is None
    assert history.market_handicap(match.pre) is None
    assert sum(history.market_result(match.pre)) == pytest.approx(1.0)
    assert history.market_over25(match.pre) == pytest.approx(tuple(devig.power([1.90, 1.95])))
    assert history.market_result(match.close) is None


def test_market_chances_are_the_power_method_on_the_average_prices(db):
    match = by_id(db)[1]
    assert history.market_result(match.pre) == pytest.approx(tuple(devig.power([2.10, 3.40, 3.60])))
    assert history.market_handicap(match.pre) == pytest.approx(tuple(devig.power([1.95, 1.90])))


def test_a_team_keeps_its_key_across_divisions(db):
    matches = by_id(db)
    assert matches[3].home == matches[4].home == "England:Fulham"
    assert matches[4].competition == "Championship"


def test_matches_without_a_score_are_left_out(db):
    assert 5 not in by_id(db)


def test_matches_come_ordered_by_day_then_id(db):
    days = [(m.day, m.id) for m in history.load_matches(db)]
    assert days == sorted(days)


def test_competitions_limit_the_matches(db):
    assert set(by_id(db, competitions=["Championship", "LaLiga"])) == {4, 6}


def test_block_splits_the_seasons():
    assert history.block(2012) == "warmup"
    assert history.block(2018) == "warmup"
    assert history.block(2019) == "tuning"
    assert history.block(2022) == "tuning"
    assert history.block(2023) == "locked"
    assert history.block(2025) == "locked"


def test_the_bench_reads_its_own_history_file_not_the_shared_one():
    # the shared file feeds the live pick evidence (D82), which must not see the 2026/27 CSVs
    import inspect
    from Scripts.ops.football_data_import import OUT_DB
    assert history.BENCH_DB != OUT_DB and history.BENCH_DB.parent == OUT_DB.parent
    for function in (history.connect, history.load_matches, history.named_result_prices):
        assert inspect.signature(function).parameters["path"].default == history.BENCH_DB


def test_2026_is_fresh():
    assert history.FRESH_FROM == 2026
    assert history.block(2026) == "fresh"
    assert history.block(2027) == "fresh"


def test_the_database_is_opened_read_only(db):
    before = os.stat(db).st_mtime_ns
    history.load_matches(db)
    history.named_result_prices(db)
    assert os.stat(db).st_mtime_ns == before
    with closing(history.connect(db)) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM matches")


def test_named_prices_leave_out_averages_and_suspect_rows(db):
    named = history.named_result_prices(db)
    assert sorted(named[2]) == [(1.80, 3.60, 4.60), (1.85, 3.70, 4.70)]
    assert named[6] == [(1.90, 3.25, 4.20)]
    assert named[1] == [(2.05, 3.40, 3.50)]
    assert 3 not in named
