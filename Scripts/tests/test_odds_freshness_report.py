from datetime import datetime, timedelta, timezone
import json
import plistlib
import sqlite3

import pytest

from Scripts.ops import odds_archive as archive
from Scripts.ops import odds_archive_launchd as schedule
from Scripts.ops import odds_freshness_report as report
from Scripts.ops.match_read_launchd import build_paths
from Scripts.tests.test_odds_archive import Response, Transport

KICKOFF = datetime(2026, 10, 17, 14, tzinfo=timezone.utc)
BOOKS = ["Pinnacle", "Bet365", "Betano", "1xBet", "Marathonbet"]


def body(when, prices, *, fixture=900, league=39):
    """One odds page: prices maps book -> (home, draw, away, over 2.5, under 2.5)."""
    params = dict(league=league, season=2026, date=KICKOFF.date().isoformat(), page=1)
    books = [dict(id=i + 1, name=name, bets=[
        dict(id=1, name="Match Winner", values=[dict(value="Home", odd=str(h)), dict(value="Draw", odd=str(d)),
                                                dict(value="Away", odd=str(a))]),
        dict(id=5, name="Goals Over/Under", values=[dict(value="Over 2.5", odd=str(o)), dict(value="Under 2.5", odd=str(u))]),
    ]) for i, (name, (h, d, a, o, u)) in enumerate(prices.items())]
    return params, dict(get="odds", parameters={k: str(v) for k, v in params.items()}, errors=[], results=1,
                        paging=dict(current=1, total=1),
                        response=[dict(league=dict(id=league, season=2026),
                                       fixture=dict(id=fixture, date=KICKOFF.isoformat()),
                                       update=(when - timedelta(minutes=1)).isoformat(), bookmakers=books)])


@pytest.fixture
def db():
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys=ON")
    archive.initialize(connection)
    yield connection
    connection.close()


def capture(db, when, prices):
    _params, data = body(when, prices)
    archive.collect(db, "test", archive.config(["EPL"], days_ahead=0), session=Transport([Response(data)]),
                    clock=lambda: when, sleep=lambda _: None)


def market_moves(db, lagging_book=None):
    """Three captures; the market shortens the home side at capture two."""
    start = KICKOFF - timedelta(hours=2)
    before = {b: (2.0, 3.4, 3.8, 1.9, 1.95) for b in BOOKS}
    after = {b: (1.7, 3.8, 4.8, 1.9, 1.95) for b in BOOKS}
    capture(db, start, before)
    second = dict(after)
    if lagging_book:
        second[lagging_book] = before[lagging_book]
    capture(db, start + timedelta(minutes=10), second)
    capture(db, start + timedelta(minutes=20), after)


def test_pairs_lines_and_removes_margin():
    lines = report.selections(5, [dict(value="Over 2.5", odd="1.9"), dict(value="Under 2.5", odd="1.95"),
                                  dict(value="Over 3.5", odd="3.1")])
    assert list(lines) == ["2.5"]
    p, margin = report.fair_first(lines["2.5"])
    assert p == pytest.approx((1 / 1.9) / (1 / 1.9 + 1 / 1.95))
    assert margin == pytest.approx(1 / 1.9 + 1 / 1.95 - 1)
    assert report.selections(1, [dict(value="Home", odd="2")]) == {}


def test_book_moving_with_market_counts_as_following(db):
    market_moves(db)
    series, kickoffs, _ = report.load_captures(db)
    stats = report.freshness(series, kickoffs)
    s = stats[("near_kickoff", "match_result", "Pinnacle")]
    assert (s["moves"], s["with"], s["next"], s["missed"]) == (1, 1, 0, 0)
    assert s["changed"] == 1 and s["pairs"] == 2


def test_lagging_book_is_caught_one_capture_later(db):
    market_moves(db, lagging_book="Pinnacle")
    series, kickoffs, _ = report.load_captures(db)
    stats = report.freshness(series, kickoffs)
    s = stats[("near_kickoff", "match_result", "Pinnacle")]
    assert (s["moves"], s["with"], s["next"]) == (1, 0, 1)


def test_post_kickoff_captures_are_ignored(db):
    capture(db, KICKOFF + timedelta(minutes=5), {b: (2.0, 3.4, 3.8, 1.9, 1.95) for b in BOOKS})
    series, kickoffs, _ = report.load_captures(db)
    assert not series and not kickoffs


def test_accuracy_compares_last_price_with_consensus(db):
    market_moves(db)
    series, kickoffs, _ = report.load_captures(db)
    acc = report.accuracy(series, kickoffs, {900: (2, 0)}, books=BOOKS)
    row = acc["match_result|Pinnacle"]
    assert row["fixtures"] == 1 and row["difference"] == 0
    assert "goals_over_under|Pinnacle" in acc
    verdict = report.verdict(report.freshness(series, kickoffs), acc, "Pinnacle")
    assert verdict["lag_test"] == "insufficient" and verdict["accuracy_test"] == "insufficient"


def test_near_kickoff_mode_only_requests_imminent_league_dates(db):
    capture(db, KICKOFF - timedelta(hours=10), {b: (2.0, 3.4, 3.8, 1.9, 1.95) for b in BOOKS})
    settings = archive.config(["EPL", "LaLiga"], near_kickoff_hours=3)
    idle = archive.collect(db, "test", settings, session=Transport([]),
                           clock=lambda: KICKOFF - timedelta(hours=5), sleep=lambda _: None)
    assert idle["status"] == "idle"
    runs = db.execute("SELECT count(*) FROM runs").fetchone()[0]
    _params, data = body(KICKOFF - timedelta(hours=2), {b: (2.0, 3.4, 3.8, 1.9, 1.95) for b in BOOKS})
    client = Transport([Response(data)])
    result = archive.collect(db, "test", settings, session=client,
                             clock=lambda: KICKOFF - timedelta(hours=2), sleep=lambda _: None)
    assert result["status"] == "complete" and result["scopes"] == 1 and len(client.calls) == 1
    assert client.calls[0]["params"]["league"] == 39
    assert db.execute("SELECT count(*) FROM runs").fetchone()[0] == runs + 1
    stored = json.loads(db.execute("SELECT config_json FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()[0])
    assert stored["near_kickoff_hours"] == 3


@pytest.mark.parametrize("hours", [0, -1, 7])
def test_near_kickoff_window_is_bounded(hours):
    with pytest.raises(ValueError):
        archive.config(near_kickoff_hours=hours)


def test_near_kickoff_schedule_is_a_separate_ten_minute_job(tmp_path):
    paths = build_paths(root=tmp_path, plist_path=tmp_path / "near.plist")
    settings = plistlib.loads(schedule.render(paths, near=True).encode())
    assert settings["Label"] == "com.bettingrag.odds-archive-near-kickoff"
    assert settings["StartInterval"] == 600
    assert settings["ProgramArguments"][-4:] == ["Scripts.ops.odds_archive", "--execute", "--near-kickoff-hours", "3"]
    assert plistlib.loads(schedule.render(paths).encode())["StartInterval"] == 10800
    assert schedule.operate("install", paths, dry_run=True, near=True)["configuration"]["Label"] == schedule.NEAR_LABEL
    assert not paths.plist_path.exists()
