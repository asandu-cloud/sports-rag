"""Owner dashboard (D67): the Most likely log and the Value picks, read from platform.db, kept in its own file."""
import json
import sqlite3
from contextlib import nullcontext
from datetime import datetime, timezone

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
