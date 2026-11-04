"""The owner dashboard's report and history (tracker D67), from its own file only."""
import re

import pytest

from Scripts.ops.owner_dashboard import pages, report, store


@pytest.fixture
def conn(tmp_path):
    connection = store.connect(tmp_path / "owner_dashboard.db")
    yield connection
    connection.close()


def add_match(conn, fixture, kickoff, *, league="EPL", home="Everton", away="Coventry", status="FT", score=(2, 1),
              corners=(6, 4), cards=((2, 0), (1, 1)), sot=(5, 4)):
    conn.execute(
        "INSERT INTO matches (fixture_api_id, league, home, away, kickoff, status, home_goals, away_goals, "
        "corners_home, corners_away, yellow_home, red_home, yellow_away, red_away, sot_home, sot_away, read_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
        (fixture, league, home, away, kickoff, status, *score, *corners, *cards[0], *cards[1], *sot))


def add_likely(conn, fixture, market, side, line, chance, outcome, *, label=None, position=0, reason=None):
    conn.execute(
        "INSERT INTO most_likely (fixture_api_id, position, market, side, line, label, chance, typical_price, "
        "version, outcome, pending_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'recommendations.v1', ?, ?)",
        (fixture, position, market, side, line, label or f"{side} {line}", chance, round(1 / chance, 2), outcome,
         reason))


def add_value(conn, key, fixture, kickoff, market, said, outcome, *, label="Under 10 corners", side="under",
              line=10.0, actual=None, home="Everton", away="Coventry"):
    conn.execute(
        "INSERT INTO value_picks (selection_key, fixture_api_id, kickoff, league, home, away, market, side, line, "
        "label, pick, odds, bookmaker, said, outcome, actual_result, published_at) "
        "VALUES (?, ?, ?, 'EPL', ?, ?, ?, ?, ?, ?, ?, 2.1, 'Pinnacle', ?, ?, ?, '2026-11-03T09:00:00Z')",
        (key, fixture, kickoff, home, away, market, side, line, label, label, said, outcome, actual))


def test_said_and_hit_are_taken_over_settled_picks_with_half_results_weighted(conn):
    for key, said, outcome in (("a", 0.60, "hit"), ("b", 0.50, "half_hit"), ("c", 0.55, "push"), ("d", 0.70, None)):
        add_value(conn, key, "1", "2026-11-04T19:00:00Z", "corners", said, outcome)
    overall = report.summary(conn, kind="value")["overall"]
    assert overall["settled"] == pytest.approx(1.5)
    assert overall["hit"] == pytest.approx(1.0)
    assert overall["said"] == pytest.approx((0.60 * 1 + 0.50 * 0.5) / 1.5)
    assert overall["pending"] == 1
    assert overall["picks"] == 3  # settled picks, the push included


def test_weeks_follow_kickoff_and_a_market_can_be_opened_alone(conn):
    add_match(conn, "1", "2026-10-28T12:00:00Z")  # week of Mon 26 Oct
    add_match(conn, "2", "2026-11-11T12:00:00Z")  # week of Mon 9 Nov; the week of 2 Nov has none
    add_likely(conn, "1", "goals", "over", 1.5, 0.80, "hit")
    add_likely(conn, "1", "corners", "under", 11.5, 0.60, "miss", position=1)
    add_likely(conn, "2", "goals", "over", 1.5, 0.70, "miss")
    add_likely(conn, "2", "corners", "under", 11.5, 0.65, None, position=1, reason="corners not in yet")
    whole = report.summary(conn, kind="most_likely")
    assert [(w["week"], w["settled"], w["pending"]) for w in whole["weeks"]] == [
        ("2026-10-26", 2, 0), ("2026-11-02", 0, 0), ("2026-11-09", 1, 1)]
    assert whole["weeks"][0]["hit"] == pytest.approx(0.5) and whole["weeks"][0]["said"] == pytest.approx(0.70)
    assert whole["weeks"][1]["hit"] is None and whole["weeks"][1]["said"] is None
    assert [m["market"] for m in whole["markets"]] == ["goals", "corners"]
    corners = report.summary(conn, kind="most_likely", market="corners")
    assert [(w["week"], w["settled"], w["pending"]) for w in corners["weeks"]] == [
        ("2026-10-26", 1, 0), ("2026-11-02", 0, 0), ("2026-11-09", 0, 1)]
    assert corners["overall"]["hit"] == 0.0 and corners["overall"]["said"] == pytest.approx(0.60)
    assert [m["market"] for m in corners["markets"]] == ["goals", "corners"]  # the tiles stay


def test_the_report_page_has_the_tiles_the_legend_and_the_weekly_table(conn):
    add_match(conn, "1", "2026-11-04T19:00:00Z")
    add_likely(conn, "1", "goals", "over", 1.5, 0.80, "hit")
    add_likely(conn, "1", "spreads", "home", 0.5, 0.62, "hit", position=1)
    page = pages.report_page(conn, kind="most_likely", updated="2026-11-04T15:20:00Z")
    assert page.startswith("<!doctype html>")
    assert 'href="/?kind=most_likely&amp;market=goals"' in page and 'href="/?kind=most_likely&amp;market=spreads"' in page
    assert ">Handicap<" in page and ">Goals<" in page
    assert 'class="legend"' in page and ">Hit<" in page and ">Said<" in page
    assert "<table" in page and "Week of 2 Nov" in page
    assert 'href="/?kind=value"' in page and 'href="/history"' in page
    assert "Update now" in page
    text = re.sub(r"<[^>]+>", " ", re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", page, flags=re.S)).lower()
    for word in ("profit", "roi", "return", "stake", "lock", "banker", "ai"):
        assert not re.search(rf"\b{word}\b", text), word
    assert "€" not in text


def test_an_empty_dashboard_says_so(conn):
    page = pages.report_page(conn, kind="value")
    assert "No settled Value picks yet" in page
    assert "<svg" not in page
    assert "Not updated yet" in page
