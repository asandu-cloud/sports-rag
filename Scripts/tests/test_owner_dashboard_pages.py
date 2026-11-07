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
              line=10.0, actual=None, home="Everton", away="Coventry", league="EPL"):
    conn.execute(
        "INSERT INTO value_picks (selection_key, fixture_api_id, kickoff, league, home, away, market, side, line, "
        "label, pick, odds, bookmaker, said, outcome, actual_result, published_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 2.1, 'Pinnacle', ?, ?, ?, '2026-11-03T09:00:00Z')",
        (key, fixture, kickoff, league, home, away, market, side, line, label, label, said, outcome, actual))


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


def three_leagues(conn):
    add_match(conn, "1", "2026-11-04T19:00:00Z", league="UCL")
    add_match(conn, "2", "2026-11-07T15:00:00Z", league="EPL")
    add_match(conn, "3", "2026-11-08T15:00:00Z", league="LaLiga")
    add_likely(conn, "1", "goals", "over", 1.5, 0.70, "miss")
    add_likely(conn, "2", "goals", "over", 1.5, 0.80, "hit")
    add_likely(conn, "2", "corners", "under", 11.5, 0.60, "miss", position=1)
    add_likely(conn, "3", "corners", "under", 11.5, 0.65, "hit")


def test_leagues_come_in_the_live_sites_order_and_follow_the_market(conn):
    three_leagues(conn)
    whole = report.summary(conn, kind="most_likely")
    assert [lg["league"] for lg in whole["leagues"]] == ["EPL", "LaLiga", "UCL"]
    epl = whole["leagues"][0]
    assert (epl["picks"], epl["hit"]) == (2, 0.5) and epl["said"] == pytest.approx(0.70)
    corners = report.summary(conn, kind="most_likely", market="corners")
    assert [(lg["league"], lg["hit"]) for lg in corners["leagues"]] == [("EPL", 0.0), ("LaLiga", 1.0)]


def test_a_league_can_be_opened_alone_and_its_markets_follow_it(conn):
    three_leagues(conn)
    epl = report.summary(conn, kind="most_likely", league="EPL")
    assert epl["overall"]["picks"] == 2 and epl["overall"]["hit"] == 0.5
    assert [m["market"] for m in epl["markets"]] == ["goals", "corners"]
    assert [lg["league"] for lg in epl["leagues"]] == ["EPL", "LaLiga", "UCL"]  # the table stays
    ucl = report.summary(conn, kind="most_likely", league="UCL")
    assert [m["market"] for m in ucl["markets"]] == ["goals"]
    both = report.summary(conn, kind="most_likely", market="goals", league="UCL")
    assert (both["overall"]["picks"], both["overall"]["hit"]) == (1, 0.0)
    assert [(w["week"], w["settled"]) for w in both["weeks"]] == [("2026-11-02", 1)]


def test_value_picks_are_split_by_their_own_league(conn):
    add_value(conn, "a", "1", "2026-11-04T19:00:00Z", "corners", 0.55, "hit", league="UCL")
    add_value(conn, "b", "2", "2026-11-07T15:00:00Z", "corners", 0.52, "miss")
    assert [lg["league"] for lg in report.summary(conn, kind="value")["leagues"]] == ["EPL", "UCL"]


def test_the_report_page_has_a_league_table_that_keeps_the_market(conn):
    three_leagues(conn)
    page = pages.report_page(conn, kind="most_likely", market="goals")
    assert "<h2>Leagues</h2>" in page
    assert 'href="/?kind=most_likely&amp;market=goals&amp;league=EPL"' in page
    assert ">Premier League<" in page and ">Champions League<" in page and ">All leagues<" in page
    assert ">La Liga<" not in page.split("<h2>Leagues</h2>")[1]  # no goals pick there
    assert "<td>+20</td>" in page and "<td>−70</td>" in page  # hit minus said, in points
    opened = pages.report_page(conn, kind="most_likely", league="EPL")
    assert "· All markets · Premier League</span>" in opened
    assert 'href="/?kind=most_likely&amp;market=corners&amp;league=EPL"' in opened  # market tiles keep the league
    assert 'href="/?kind=value&amp;league=EPL"' in opened
    assert 'class="on"' in opened and 'href="/?kind=most_likely" aria-current' not in opened
    unknown = pages.report_page(conn, kind="most_likely", league="Narnia")
    assert "· All markets</span>" in unknown and "Narnia" not in unknown


def test_the_league_table_counts_settled_picks_as_the_tiles_do(conn):
    add_value(conn, "a", "1", "2026-11-04T19:00:00Z", "corners", 0.55, "hit")
    add_value(conn, "b", "2", "2026-11-04T19:00:00Z", "corners", 0.52, "push")
    table = pages.report_page(conn, kind="value").split("<h2>Leagues</h2>")[1].split("</table>")[0]
    assert ">All leagues</a></td><td>2</td>" in table and ">Premier League</a></td><td>2</td>" in table


def test_an_empty_dashboard_says_so(conn):
    page = pages.report_page(conn, kind="value")
    assert "No settled Value picks yet" in page
    assert "<svg" not in page
    assert "Not updated yet" in page


FACTS = {"status": "FT", "home": "Lyon", "away": "Lens", "home_goals": 2, "away_goals": 1, "corners_home": 6,
         "corners_away": 4, "yellow_home": 2, "red_home": 0, "yellow_away": 2, "red_away": 1, "sot_home": 5,
         "sot_away": 4}


def test_what_happened_names_the_number_the_pick_settles_on():
    assert pages.what_happened("goals", FACTS, {"side": "over", "line": 1.5}) == "3 goals"
    assert pages.what_happened("goals", dict(FACTS, home_goals=1, away_goals=0), {"side": "over", "line": 0.5}) == "1 goal"
    assert pages.what_happened("corners", FACTS, {"side": "under", "line": 11.5}) == "10 corners"
    assert pages.what_happened("corners", dict(FACTS, corners_away=None), {"side": "under", "line": 11.5}) == (
        "corners not in yet")
    assert pages.what_happened("moneyline", FACTS, {"side": "home"}) == "2–1"
    assert pages.what_happened("btts", FACTS, {"side": "yes"}) == "2–1"
    assert pages.what_happened("spreads", FACTS, {"side": "away", "line": 0.5}) == "Lens lost by 1"
    assert pages.what_happened("spreads", FACTS, {"side": "LENS", "line": 0.5}) == "Lens lost by 1"  # a Value side
    assert pages.what_happened("spreads", FACTS, {"side": "home", "line": -1.5}) == "Lyon won by 1"
    assert pages.what_happened("spreads", dict(FACTS, away_goals=2), {"side": "home", "line": -0.5}) == "Lyon drew"
    assert pages.what_happened("cards", FACTS, {"side": "over", "line": 4.5, "actual_result": 6.0}) == "6 cards"
    assert pages.what_happened("cards", FACTS, {"side": "over", "line": 4.5}) == "5 cards"
    assert pages.what_happened("sot", FACTS, {"side": "under", "line": 9.5}) == "9 shots on target"
    assert pages.what_happened("goals", dict(FACTS, status="AET"), {"side": "over", "line": 2.5}) == (
        "2–1 after extra time")


def test_history_groups_a_day_by_league_with_one_line_per_pick(conn):
    add_match(conn, "1", "2026-11-03T12:00:00Z", league="UCL", home="Lyon", away="Lens")
    add_match(conn, "2", "2026-11-03T12:30:00Z", league="EPL", corners=(6, None))
    add_likely(conn, "1", "goals", "over", 1.5, 0.80, "hit", label="Over 1.5 goals")
    add_likely(conn, "1", "spreads", "away", 0.5, 0.55, "miss", label="Lens +0.5", position=1)
    add_value(conn, "v1", "1", "2026-11-03T12:00:00Z", "spreads", 0.49, "half_miss", label="Lens +0.25",
              side="Lens", line=0.25, home="Lyon", away="Lens")
    add_likely(conn, "2", "corners", "under", 11.5, 0.63, None, label="Under 11.5 corners", reason="corners not in yet")
    page = pages.history_page(conn, day="2026-11-03")
    assert page.index("Premier League") < page.index("Champions League")  # the live site's league order
    assert page.count('class="pick"') == 4
    assert "Lyon v Lens" in page and "2–1" in page and "Corners 10 (6–4)" in page and "Corners not in yet" in page
    assert "Over 1.5 goals" in page and "said 80%" in page and "1.25" in page and "3 goals" in page
    assert "Lens +0.25" in page and "said 49%" in page and "2.10 at Pinnacle" in page and "Lens lost by 1" in page
    for result in (">Hit<", ">Miss<", ">Half miss<", ">Pending<"):
        assert result in page
    assert "waiting for the corners" in page
    assert page.index("Over 1.5 goals") < page.index("Lens +0.5") < page.index("Lens +0.25")  # Most likely first


def test_history_opens_on_the_latest_day_with_a_settled_pick_and_skips_days_without_picks(conn):
    add_match(conn, "1", "2026-11-02T12:00:00Z")
    add_likely(conn, "1", "goals", "over", 1.5, 0.80, "hit")
    add_match(conn, "2", "2026-11-04T12:00:00Z", home="Lyon", away="Lens")
    add_likely(conn, "2", "corners", "under", 11.5, 0.63, None, reason="corners not in yet")
    add_match(conn, "3", "2026-11-05T12:00:00Z", home="Ajax", away="PSV")  # seen, no picks
    add_match(conn, "4", "2026-10-28T12:00:00Z", home="Roma", away="Lazio")
    add_likely(conn, "4", "goals", "under", 3.5, 0.70, "hit")
    page = pages.history_page(conn)
    assert 'href="/history?day=2026-11-02" aria-current="page"' in page
    assert 'href="/history?day=2026-11-04"' in page and "2026-11-05" not in page and "Ajax" not in page
    assert "Lyon v Lens" not in page  # that day is in the strip, not on the page
    assert 'href="/history?day=2026-10-28"' in page  # the earlier week's latest day with picks
    later = pages.history_page(conn, day="2026-11-04")
    assert "Lyon v Lens" in later and 'href="/history?day=2026-11-04" aria-current="page"' in later


def test_an_empty_history_says_so(conn):
    assert "No played matches with picks yet" in pages.history_page(conn)
