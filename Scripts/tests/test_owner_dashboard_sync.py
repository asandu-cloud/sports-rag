"""Owner dashboard (D67): the Most likely log and the Value picks, read from platform.db, kept in its own file."""
import json
import sqlite3
from contextlib import nullcontext
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import create_engine

from Scripts.data_platform.models import Base
from Scripts.data_platform.services.refresh_coordination import RefreshBusy
from Scripts.ops.owner_dashboard import store, sync

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
HOME, AWAY = "Lyon", "Lens"


def option(key, side, point, odds, book, team=None):
    return {"market_key": key, "side": side, "team": team, "point": point, "odds": odds, "bookmaker": book}


def result(group, options):
    return {"market": {"group": group, "key": group}, "fixture": {"home_team": HOME, "away_team": AWAY},
            "projection": {"value": 2.9, "components": {}}, "decision": {"status": "no_bet"},
            "context": {"decision_audit": {"input_options": options, "candidates": []}}}


def priced_read():
    """Two books price Over/Under 2.5 goals: Over 2.5 at about 64% is the read's one Most likely pick."""
    goals = [option("totals", side, 2.5, odds, book) for book, pair in {"A": (1.50, 2.60), "B": (1.48, 2.65)}.items()
             for side, odds in zip(("over", "under"), pair)]
    return {"canonical_results": [result("goals", goals)], "selections": []}


def unpriced_read():
    return {"canonical_results": [result("goals", [option("totals", "over", 2.5, 1.5, "A")])], "selections": []}


@pytest.fixture()
def platform(tmp_path):
    path = tmp_path / "platform.db"
    Base.metadata.create_all(create_engine(f"sqlite:///{path}"))
    return path


def add_fixture(path, api_id, kickoff, status="FT", home_goals=2, away_goals=1):
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO fixtures (api_football_id, competition_id, season_id, home_team_id, away_team_id, "
                   "kickoff_utc, status, home_goals, away_goals, created_at, updated_at) "
                   "VALUES (?, 1, 1, 1, 2, ?, ?, ?, ?, ?, ?)",
                   (api_id, kickoff, status, home_goals, away_goals, kickoff, kickoff))


def add_read(path, api_id, evaluated_at, kickoff, read, *, stage="pre_match", status="recommended"):
    with sqlite3.connect(path) as db:
        version = db.execute("SELECT count(*) + 1 FROM match_reads WHERE fixture_api_id = ? AND stage = ?",
                             (str(api_id), stage)).fetchone()[0]
        cursor = db.execute(
            "INSERT INTO match_reads (read_key, fixture_api_id, league, home_team, away_team, kickoff, stage, version, "
            "status, thesis, input_snapshot_id, pipeline_version, evaluated_at, read_json, created_at, updated_at) "
            "VALUES (?, ?, 'Ligue1', ?, ?, ?, ?, ?, ?, '', 's', 'p', ?, ?, ?, ?)",
            (f"{api_id}-{evaluated_at}", str(api_id), HOME, AWAY, kickoff, stage, version, status, evaluated_at,
             json.dumps(read), evaluated_at, evaluated_at))
        return cursor.lastrowid


@pytest.fixture()
def conn(tmp_path):
    return store.connect(tmp_path / "owner_dashboard.db")


def test_a_finished_match_logs_the_most_likely_of_its_last_read_before_kickoff_once(platform, conn):
    add_fixture(platform, 101, "2026-11-03 20:00:00.000000")
    add_read(platform, 101, "2026-11-03 10:00:00.000000", "2026-11-03T20:00:00Z", unpriced_read())
    last = add_read(platform, 101, "2026-11-03 18:00:00.000000", "2026-11-03T20:00:00Z", priced_read(),
                    stage="confirmed_lineups")
    add_read(platform, 101, "2026-11-03 21:00:00.000000", "2026-11-03T20:00:00Z", unpriced_read())  # after kickoff
    assert sync.log_most_likely(conn, platform, now=NOW, gate=nullcontext) == {
        "logged": 1, "picks": 1, "skipped_busy": False}
    match = conn.execute("SELECT * FROM matches").fetchone()
    assert (match["fixture_api_id"], match["read_id"], match["home"], match["league"]) == ("101", last, HOME, "Ligue1")
    pick = conn.execute("SELECT * FROM most_likely").fetchone()
    assert (pick["market"], pick["side"], pick["line"], pick["label"]) == ("goals", "over", 2.5, "Over 2.5 goals")
    assert 0.6 < pick["chance"] < 0.7 and pick["version"] == "recommendations.v1" and pick["outcome"] is None
    assert sync.log_most_likely(conn, platform, now=NOW, gate=nullcontext)["logged"] == 0  # logged once


def test_a_match_not_over_waits_and_a_postponed_one_logs_its_new_read(platform, conn):
    add_fixture(platform, 102, "2026-11-04 20:00:00.000000", status="NS")
    add_read(platform, 102, "2026-11-04 09:00:00.000000", "2026-11-04T20:00:00Z", priced_read())
    add_fixture(platform, 103, "2026-11-02 18:00:00.000000", status="PST")
    add_read(platform, 103, "2026-10-25 09:00:00.000000", "2026-10-25T18:00:00Z", priced_read())  # first date
    assert sync.log_most_likely(conn, platform, now=NOW, gate=nullcontext)["logged"] == 0
    with sqlite3.connect(platform) as db:  # played on the new date
        db.execute("UPDATE fixtures SET status = 'FT' WHERE api_football_id = 103")
    new = add_read(platform, 103, "2026-11-02 12:00:00.000000", "2026-11-02T18:00:00Z", priced_read())
    assert sync.log_most_likely(conn, platform, now=NOW, gate=nullcontext)["logged"] == 1
    assert conn.execute("SELECT read_id FROM matches WHERE fixture_api_id = '103'").fetchone()[0] == new


def test_an_unavailable_read_is_not_the_one_logged(platform, conn):
    add_fixture(platform, 104, "2026-11-03 20:00:00.000000")
    kept = add_read(platform, 104, "2026-11-03 12:00:00.000000", "2026-11-03T20:00:00Z", priced_read())
    add_read(platform, 104, "2026-11-03 19:00:00.000000", "2026-11-03T20:00:00Z", {"canonical_results": []},
             status="unavailable")
    sync.log_most_likely(conn, platform, now=NOW, gate=nullcontext)
    assert conn.execute("SELECT read_id FROM matches").fetchone()[0] == kept


def test_a_read_with_no_priced_market_marks_the_match_seen_with_no_picks(platform, conn, monkeypatch):
    add_fixture(platform, 105, "2026-11-03 20:00:00.000000")
    add_read(platform, 105, "2026-11-03 12:00:00.000000", "2026-11-03T20:00:00Z", unpriced_read())
    assert sync.log_most_likely(conn, platform, now=NOW, gate=nullcontext) == {
        "logged": 1, "picks": 0, "skipped_busy": False}
    fetched = []
    real = sync._read_json
    monkeypatch.setattr(sync, "_read_json", lambda db, ids: fetched.append(ids) or real(db, ids))
    assert sync.log_most_likely(conn, platform, now=NOW, gate=nullcontext)["logged"] == 0
    assert fetched == []  # its read is never fetched again


def test_a_busy_gate_stops_the_update_and_keeps_what_it_wrote(platform, conn):
    for n in range(25):
        add_fixture(platform, 200 + n, "2026-11-03 20:00:00.000000")
        add_read(platform, 200 + n, "2026-11-03 12:00:00.000000", "2026-11-03T20:00:00Z", priced_read())
    entries = []

    def gate():  # the scan, then the first batch of 20, then the refresh takes the gate
        entries.append(1)
        if len(entries) == 3:
            raise RefreshBusy("refresh running")
        return nullcontext()
    assert sync.log_most_likely(conn, platform, now=NOW, gate=gate) == {
        "logged": 20, "picks": 20, "skipped_busy": True}
    assert sync.log_most_likely(conn, platform, now=NOW, gate=nullcontext)["logged"] == 5


def add_team_stats(path, api_id, *, is_home, corners, shots_on=4, yellow=2, red=0):
    with sqlite3.connect(path) as db:
        fixture = db.execute("SELECT id FROM fixtures WHERE api_football_id = ?", (api_id,)).fetchone()[0]
        db.execute("INSERT INTO fixture_team_stats (fixture_id, team_id, opponent_team_id, is_home, corners, shots_on, "
                   "yellow_cards, red_cards, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'x', 'x')",
                   (fixture, 1 if is_home else 2, 2 if is_home else 1, is_home, corners, shots_on, yellow, red))


def log_pick(conn, api_id, market, side, line, position=0):
    with conn:
        conn.execute("INSERT OR IGNORE INTO matches (fixture_api_id, league, home, away, read_id) "
                     "VALUES (?, 'Ligue1', ?, ?, 1)", (str(api_id), HOME, AWAY))
        conn.execute("INSERT INTO most_likely (fixture_api_id, position, market, side, line, label, chance, version) "
                     "VALUES (?, ?, ?, ?, ?, 'x', 0.6, 'recommendations.v1')", (str(api_id), position, market, side, line))


def outcomes(conn, api_id):
    return {(r["market"], r["side"], r["line"]): (r["outcome"], r["pending_reason"]) for r in conn.execute(
        "SELECT * FROM most_likely WHERE fixture_api_id = ?", (str(api_id),))}


def test_goal_markets_grade_from_the_score_and_corners_from_both_teams(platform, conn):
    add_fixture(platform, 301, "2026-11-03 20:00:00.000000", home_goals=2, away_goals=1)
    add_team_stats(platform, 301, is_home=True, corners=6, shots_on=5, yellow=1, red=1)
    add_team_stats(platform, 301, is_home=False, corners=5, shots_on=3, yellow=3)
    picks = [("goals", "over", 2.5), ("btts", "no", None), ("moneyline", "home", None), ("spreads", "home", -0.5),
             ("corners", "over", 9.5), ("corners", "under", 9.5)]
    for position, (market, side, line) in enumerate(picks):
        log_pick(conn, 301, market, side, line, position)
    assert sync.grade(conn, platform, now=NOW, gate=nullcontext) == {"graded": 6, "pending": 0, "skipped_busy": False}
    assert outcomes(conn, 301) == {
        ("goals", "over", 2.5): ("hit", None), ("btts", "no", None): ("miss", None),
        ("moneyline", "home", None): ("hit", None), ("spreads", "home", -0.5): ("hit", None),
        ("corners", "over", 9.5): ("hit", None), ("corners", "under", 9.5): ("miss", None)}
    facts = conn.execute("SELECT * FROM matches WHERE fixture_api_id = '301'").fetchone()
    assert (facts["status"], facts["home_goals"], facts["away_goals"]) == ("FT", 2, 1)
    assert (facts["corners_home"], facts["corners_away"], facts["sot_home"], facts["sot_away"]) == (6, 5, 5, 3)
    assert (facts["yellow_home"], facts["yellow_away"], facts["red_home"], facts["red_away"]) == (1, 3, 1, 0)


def test_extra_time_and_missing_corners_stay_pending_and_grade_when_the_corners_arrive(platform, conn):
    add_fixture(platform, 302, "2026-11-03 20:00:00.000000", status="AET")
    log_pick(conn, 302, "goals", "over", 2.5)
    add_fixture(platform, 303, "2026-11-03 20:00:00.000000")
    add_team_stats(platform, 303, is_home=True, corners=6)
    log_pick(conn, 303, "corners", "over", 9.5)
    assert sync.grade(conn, platform, now=NOW, gate=nullcontext)["pending"] == 2
    assert outcomes(conn, 302) == {("goals", "over", 2.5): (None, "extra time")}
    assert outcomes(conn, 303) == {("corners", "over", 9.5): (None, "corners not in yet")}
    add_team_stats(platform, 303, is_home=False, corners=3)
    assert sync.grade(conn, platform, now=NOW, gate=nullcontext)["graded"] == 1
    assert outcomes(conn, 303) == {("corners", "over", 9.5): ("miss", None)}


def test_a_value_picks_match_gets_its_facts_without_a_most_likely_log(platform, conn):
    add_fixture(platform, 304, "2026-11-03 20:00:00.000000", home_goals=0, away_goals=0)
    add_team_stats(platform, 304, is_home=True, corners=4)
    add_team_stats(platform, 304, is_home=False, corners=7)
    with conn:
        conn.execute("INSERT INTO value_picks (selection_key, fixture_api_id, kickoff, league, home, away, market) "
                     "VALUES ('k', '304', '2026-11-03T20:00:00Z', 'EPL', 'Everton', 'Coventry', 'corners')")
    sync.grade(conn, platform, now=NOW, gate=nullcontext)
    facts = conn.execute("SELECT * FROM matches WHERE fixture_api_id = '304'").fetchone()
    assert (facts["home"], facts["league"], facts["corners_home"], facts["corners_away"], facts["read_id"]) == (
        "Everton", "EPL", 4, 7, None)


def value_row(key, *, rec_id=1, role="initial", policy="market_value", outcome=None, actual=None, market="corners",
              side="under", line=10.0, pick="Under 10", home="Everton", away="Coventry", published="2026-11-03T09:00:00Z"):
    return {"selection_key": key, "recommendation_id": rec_id, "publication_role": role, "fixture_id": "401",
            "kickoff": "2026-11-06T20:00:00Z", "league": "EPL", "home_team": home, "away_team": away,
            "market": market, "side": side, "line": line, "pick": pick, "odds": 1.7, "bookmaker": "Pinnacle",
            "model_prob": 0.618, "outcome": outcome, "actual_result": actual, "published_at": published,
            "tracking": {"engine_switches": {"SPIX_SELECTION_POLICY": policy}}}


def test_only_stage_4_picks_each_selection_once_and_outcomes_update(conn):
    asked = []
    rows = [value_row("a", rec_id=1), value_row("a", rec_id=2, role="amendment"),
            value_row("legacy", policy="legacy"), value_row("old", policy=None)]
    load = lambda days: asked.append(days) or rows
    assert sync.log_value(conn, load_rows=load, today=date(2026, 11, 4)) == {"picks": 1}
    stored = conn.execute("SELECT * FROM value_picks").fetchall()
    assert [(r["selection_key"], r["label"], r["said"], r["outcome"]) for r in stored] == [
        ("a", "Under 10 corners", 0.618, None)]
    rows[0] = value_row("a", rec_id=1, outcome="hit", actual=8.0)
    sync.log_value(conn, load_rows=load, today=date(2026, 11, 7))
    assert [tuple(r) for r in conn.execute("SELECT outcome, actual_result FROM value_picks")] == [("hit", 8.0)]
    assert asked == [3, 5]  # from 1 Nov, then from the unsettled pick published 3 Nov, a day early


def test_once_every_pick_is_settled_the_window_starts_at_the_newest_pick(conn):
    asked = []
    rows = [value_row("a", outcome="miss", published="2026-11-05T09:00:00Z")]
    sync.log_value(conn, load_rows=lambda days: asked.append(days) or rows, today=date(2026, 11, 20))
    sync.log_value(conn, load_rows=lambda days: asked.append(days) or rows, today=date(2026, 11, 20))
    assert asked == [19, 16]  # 1 Nov, then 4 Nov (a day before the newest pick), never the whole history


def test_value_labels_use_the_layers_words():
    assert sync.value_label("spreads", "As Roma", -1.5, "AS Roma", "Sassuolo", "As Roma -1.5") == "AS Roma −1.5"
    assert sync.value_label("corners", "under", 10.0, "Everton", "Coventry", "Under 10") == "Under 10 corners"
    assert sync.value_label("moneyline", "away", None, "Lyon", "Lens", "Away") == "Lens to win"
    assert sync.value_label("cards", "over", 4.5, "Lyon", "Lens", "Over 4.5") == "Over 4.5 cards"
    assert sync.value_label("sot", "under", 8.5, "Lyon", "Lens", "Under 8.5") == "Under 8.5 shots on target"
    assert sync.value_label("spreads", "Lens B", 0.5, "Lyon", "Lens", "Lens B 0.5") == "Lens B 0.5"
