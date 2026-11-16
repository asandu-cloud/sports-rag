"""Game-read bench, phase 2 task 2: match statistics, export and loader (D77)."""
from __future__ import annotations

from contextlib import closing, contextmanager
import sqlite3

import pytest

from Scripts.data_platform.game_read import stats
from Scripts.data_platform.services.refresh_coordination import RefreshBusy
from Scripts.ops.football_data_import import SCHEMA

# football-data matches: (id, season, date, home, away, fixture api id, home shots on target, away shots on target)
HISTORY = [
    (1, 2018, "2019-05-12", "Arsenal", "Burnley", None, 7, 2),
    (2, 2019, "2019-08-10", "Arsenal", "Burnley", 9001, 6, 3),
    (3, 2019, "2019-08-17", "Chelsea", "Arsenal", 9003, None, None),
    (4, 2019, "2019-08-24", "Burnley", "Chelsea", None, 4, 4),
]
# our fixtures: (id, api id, home team id, away team id, kickoff, status)
FIXTURES = [
    (1, 9001, 10, 20, "2019-08-10 14:00:00.000000", "FT"),
    (2, 9002, 10, 40, "2019-08-14 19:00:00.000000", "FT"),  # Arsenal's European midweek match
    (3, 9003, 30, 10, "2019-08-17 16:30:00.000000", "FT"),
    (4, 9004, 30, 50, "2019-08-20 19:00:00.000000", "PST"),  # postponed: never counts as a match played
]
# per-team statistics of our fixtures: (fixture id, team id, is home, inside box, outside box, xG)
TEAM_STATS = [
    (1, 10, 1, 9, 5, 1.7), (1, 20, 0, 4, 6, None),
    (3, 30, 1, 11, 3, 2.1), (3, 10, 0, 6, 8, 0.9),
]


@pytest.fixture
def files(tmp_path):
    history_db, platform_db = tmp_path / "history.sqlite3", tmp_path / "platform.db"
    with closing(sqlite3.connect(history_db)) as db:
        db.executescript(SCHEMA)
        for id, season, day, home, away, fixture, home_target, away_target in HISTORY:
            db.execute("INSERT INTO matches (id, file, div, competition, season, match_date, home_name, away_name, "
                       "fixture_api_id, link_status, fthg, ftag, hst, ast) VALUES (?, 'E0_1920.csv', 'E0', 'EPL', ?, "
                       "?, ?, ?, ?, 'linked', 1, 0, ?, ?)",
                       (id, season, day, home, away, fixture, home_target, away_target))
        db.commit()
    with closing(sqlite3.connect(platform_db)) as db:
        db.executescript("""
            CREATE TABLE fixtures (id INTEGER PRIMARY KEY, api_football_id INTEGER, home_team_id INTEGER,
                                   away_team_id INTEGER, kickoff_utc TEXT, status TEXT);
            CREATE TABLE fixture_team_stats (fixture_id INTEGER, team_id INTEGER, is_home INTEGER,
                                             shots_inside_box INTEGER, shots_outside_box INTEGER, expected_goals REAL);
        """)
        db.executemany("INSERT INTO fixtures VALUES (?, ?, ?, ?, ?, ?)", FIXTURES)
        db.executemany("INSERT INTO fixture_team_stats VALUES (?, ?, ?, ?, ?, ?)", TEAM_STATS)
        db.commit()
    return history_db, platform_db, tmp_path / "stats.sqlite3"


@contextmanager
def open_gate():
    yield


def exported(files):
    history_db, platform_db, out = files
    stats.export(out, history_db=history_db, platform_db=platform_db, gate=open_gate)
    return stats.load(out)


def test_rest_counts_a_european_midweek_match(files):
    found = exported(files)
    # Arsenal played in Europe on the 14th (19:00), then away at Chelsea on the 17th (16:30): 2 whole days
    assert found[3].away_rest == 2
    assert found[3].home_rest is None  # Chelsea's postponed match on the 20th is later anyway; none before


def test_rest_is_none_before_2019_and_never_negative(files):
    found = exported(files)
    assert found[1].home_rest is None and found[1].away_rest is None  # 2018/19: no fixtures of ours
    assert found[2].home_rest is None  # Arsenal's first fixture we hold
    assert all(r is None or r >= 0 for s in found.values() for r in (s.home_rest, s.away_rest))


def test_an_unlinked_match_has_no_box_or_xg(files):
    found = exported(files)
    unlinked = found[4]
    assert (unlinked.home_box, unlinked.away_box, unlinked.home_outside, unlinked.away_outside) == (None,) * 4
    assert (unlinked.home_xg, unlinked.away_xg, unlinked.home_rest, unlinked.away_rest) == (None,) * 4
    assert (unlinked.home_target, unlinked.away_target) == (4, 4)  # football-data's own columns
    linked = found[2]
    assert (linked.home_box, linked.home_outside, linked.away_box, linked.away_outside) == (9, 5, 4, 6)
    assert (linked.home_xg, linked.away_xg) == (1.7, None)
    assert (linked.home_target, linked.away_target) == (6, 3)
    assert (found[3].home_box, found[3].away_box) == (11, 6)  # sides follow the fixture, not the row order
    assert set(found) == {1, 2, 3, 4}


def test_export_refuses_while_the_refresh_runs(files):
    history_db, platform_db, out = files

    @contextmanager
    def busy():
        raise RefreshBusy("refresh running")
        yield

    with pytest.raises(RefreshBusy):
        stats.export(out, history_db=history_db, platform_db=platform_db, gate=busy)
    assert not out.exists()


def test_the_export_command_stops_cleanly_while_the_refresh_runs(monkeypatch, capsys):
    from Scripts.ops import game_read_bench as bench

    def busy(out):
        raise RefreshBusy("refresh running")

    monkeypatch.setattr(bench.stats, "export", busy)
    assert bench.main(["export-stats"]) == 2
    assert "refresh" in capsys.readouterr().err
