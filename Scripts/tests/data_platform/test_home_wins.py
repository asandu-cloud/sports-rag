"""The home page's wins reel (D99): last week's best wins as the board showed them at kickoff, beside the week's
whole record, losses included."""
import json
from datetime import date, datetime, timezone

import pytest

from data_platform.services import home_wins

NOW = datetime(2026, 11, 12, 6, 0, tzinfo=timezone.utc)  # Thursday; the last full week is 2-8 Nov


def option(market, side, line, chance, price, label=None):
    return {"market": market, "side": side, "line": line, "chance": chance, "typical_price": price,
            "label": label or f"{side} {line} {market}"}


def card(event_id, kickoff, *, most_likely=(), value=(), markets=None):
    """A match card as the board showed it at kickoff; ``markets`` holds the options a Value pick is priced from."""
    return {"fixture": {"event_id": event_id, "league": "EPL", "home_team": f"Home {event_id}",
                        "away_team": f"Away {event_id}", "kickoff": kickoff},
            "visuals": {"home_team_logo": f"{event_id}-h.png", "away_team_logo": None},
            "recommendations": {"most_likely": list(most_likely),
                                "value": [{k: v for k, v in o.items() if k != "typical_price"} for o in value],
                                "markets": markets or [{"market": o["market"], "options": [o]} for o in value]}}


def ft(home, away, corners=(5, 5), status="FT"):
    return {"status": status, "home_goals": home, "away_goals": away,
            "corners_home": corners[0], "corners_away": corners[1]}


@pytest.mark.parametrize("pick, facts, outcome", [
    (option("goals", "over", 5.5, 0.23, 4.1), ft(2, 5), "hit"),
    (option("goals", "under", 3.5, 0.58, 1.73), ft(1, 0), "hit"),
    (option("btts", "yes", None, 0.54, 1.73), ft(2, 1), "hit"),
    (option("btts", "no", None, 0.55, 1.72), ft(2, 1), "miss"),
    (option("moneyline", "away", None, 0.4, 2.4), ft(1, 1), "miss"),
    (option("spreads", "home", 0.5, 0.53, 1.82), ft(1, 1), "hit"),
    (option("spreads", "away", 2.5, 0.61, 1.56), ft(3, 1), "hit"),
    (option("spreads", "away", -1.75, 0.45, 2.1), ft(1, 3), "half_hit"),  # wins by two: half won, half back
    (option("corners", "over", 9.5, 0.53, 1.76), ft(2, 0, (8, 5)), "hit"),
    (option("corners", "under", 10, 0.5, None), ft(0, 0, (4, 6)), "push"),
    (option("goals", "over", 2.5, 0.6, 1.6), ft(2, 1, status="AET"), None),  # settles on 90 minutes: waits
    (option("goals", "over", 2.5, 0.6, 1.6), ft(None, None, status="NS"), None),
    (option("corners", "over", 9.5, 0.5, 1.9), ft(2, 0, (None, 5)), None),  # corners not in yet
    (option("goals", None, None, 0.5, 2.0), ft(1, 0), None),  # a Value pick the layer could not name
])
def test_a_pick_settles_on_the_full_time_score_as_the_scorecard_grades_it(pick, facts, outcome):
    assert home_wins.grade(pick, facts) == outcome


def test_the_reel_keeps_each_matchs_highest_priced_win_highest_first_and_at_most_twelve():
    cards, facts = [], {}
    for i in range(14):
        event = str(1000 + i)
        cards.append(card(event, f"2026-11-0{5 + i % 3}T15:00:00Z", most_likely=[
            option("goals", "over", 1.5, 0.8, 1.20 + i / 100), option("btts", "yes", None, 0.55, 1.50 + i / 100)]))
        facts[event] = ft(2, 1)
    reel = home_wins.block(cards, facts, monday=date(2026, 11, 2))
    assert len(reel["wins"]) == home_wins.MAX_WINS == 12
    assert [w["fixture"]["event_id"] for w in reel["wins"]] == [str(1013 - i) for i in range(12)]
    assert {w["pick"]["market"] for w in reel["wins"]} == {"btts"}  # the dearer of each match's two wins
    first = reel["wins"][0]
    assert first == {
        "fixture": {"event_id": "1013", "league": "EPL", "home_team": "Home 1013", "away_team": "Away 1013",
                    "kickoff": "2026-11-06T15:00:00Z"},
        "visuals": {"home_team_logo": "1013-h.png", "away_team_logo": None},
        "score": {"home": 2, "away": 1}, "list": "most_likely",
        "pick": {"label": "yes None btts", "market": "btts", "side": "yes", "line": None, "chance": 0.55,
                 "typical_price": 1.50 + 13 / 100},
        "result": None}


def test_a_value_win_takes_its_options_typical_price_and_says_how_many_goals_or_corners():
    value = option("goals", "over", 5.5, 0.2265, 4.1, "Over 5.5 goals")
    corners = option("corners", "over", 9.5, 0.53, 1.76, "Over 9.5 corners")
    under = option("goals", "under", 3.5, 0.58, 1.73, "Under 3.5 goals")
    cards = [card("1", "2026-11-08T15:45:00Z", value=[value]), card("2", "2026-11-07T20:00:00Z", value=[corners]),
             card("3", "2026-11-06T19:45:00Z", value=[under])]
    facts = {"1": ft(2, 5), "2": ft(2, 0, (9, 4)), "3": ft(1, 0)}
    wins = home_wins.block(cards, facts, monday=date(2026, 11, 2))["wins"]
    assert [(w["list"], w["pick"]["label"], w["pick"]["typical_price"], w["result"]) for w in wins] == [
        ("value", "Over 5.5 goals", 4.1, "7 goals"), ("value", "Over 9.5 corners", 1.76, "13 corners"),
        ("value", "Under 3.5 goals", 1.73, "1 goal")]


def test_the_record_counts_every_pick_shown_at_kickoff_losses_and_voids_included():
    cards = [
        card("1", "2026-11-05T20:00:00Z", most_likely=[option("goals", "over", 0.5, 0.8, 1.25),
                                                        option("btts", "yes", None, 0.6, 1.6)],
             value=[option("corners", "under", 10, 0.5, None)]),
        card("2", "2026-11-07T15:00:00Z", most_likely=[option("goals", "under", 3.5, 0.7, 1.4)],
             value=[option("spreads", "away", -1.75, 0.45, 2.1), option("goals", "over", 2.5, 0.42, 2.3)]),
        card("3", "2026-11-08T17:30:00Z", most_likely=[option("goals", "over", 2.5, 0.65, 1.5)]),  # AET: waits
    ]
    facts = {"1": ft(1, 0, (4, 6)), "2": ft(1, 3), "3": ft(2, 2, status="AET")}
    reel = home_wins.block(cards, facts, monday=date(2026, 11, 2))
    most_likely, value = reel["record"]
    assert most_likely == {"list": "most_likely", "picks": 3, "won": 1, "lost": 2, "void": 0, "said": 0.7,
                           "expected_wins": 2.1, "happened": 0.3333}
    # A half won counts with the wins and a push is void, as the owner dashboard shows them.
    assert value == {"list": "value", "picks": 3, "won": 2, "lost": 0, "void": 1, "said": 0.4567,
                     "expected_wins": 1.4, "happened": 0.6667}
    assert reel["week"] == {"from": "2026-11-05", "to": "2026-11-07"}  # the days with a graded pick
    # Only a full win with a price makes a card: the half win and the void stay in the record only.
    assert [(w["fixture"]["event_id"], w["pick"]["label"]) for w in reel["wins"]] == [
        ("2", "over 2.5 goals"), ("1", "over 0.5 goals")]


def test_only_picks_from_this_week_and_after_the_lists_went_live_count():
    shown = option("goals", "over", 1.5, 0.8, 1.3)
    cards = [card("early", "2026-11-04T19:00:00Z", most_likely=[shown]),  # before 19:50 UTC on 4 Nov (D68)
             card("live", "2026-11-04T20:00:00Z", most_likely=[shown]),
             card("next", "2026-11-09T19:45:00Z", most_likely=[shown])]  # the following Monday
    facts = {event: ft(2, 1) for event in ("early", "live", "next")}
    reel = home_wins.block(cards, facts, monday=date(2026, 11, 2))
    assert [w["fixture"]["event_id"] for w in reel["wins"]] == ["live"]
    assert reel["record"][0]["picks"] == 1 and reel["record"][1]["picks"] == 0
    assert reel["record"][1] == {"list": "value", "picks": 0, "won": 0, "lost": 0, "void": 0, "said": None,
                                 "expected_wins": 0.0, "happened": None}


def test_the_week_is_londons_and_a_late_sunday_kickoff_stays_in_it():
    late = card("late", "2027-03-28T22:30:00Z", most_likely=[option("goals", "over", 1.5, 0.8, 1.3)])  # 23:30 BST
    monday = card("mon", "2027-03-28T23:30:00Z", most_likely=[option("goals", "over", 1.5, 0.8, 1.3)])  # 00:30 Mon
    reel = home_wins.block([late, monday], {"late": ft(2, 1), "mon": ft(2, 1)}, monday=date(2027, 3, 22))
    assert [w["fixture"]["event_id"] for w in reel["wins"]] == ["late"]


class Loader:
    """The board's before-kickoff record and the fixtures' scores, by day, without a database."""

    def __init__(self, cards, facts):
        self.cards, self.facts, self.days = cards, facts, []

    def load_cards(self, days, *, now):
        self.days.append((days[0], days[-1]))
        wanted = {d.isoformat() for d in days}
        return [c for c in self.cards if c["fixture"]["kickoff"][:10] in wanted]

    def load_facts(self, event_ids):
        return {e: self.facts[e] for e in event_ids if e in self.facts}


def test_build_shows_the_last_full_week_and_keeps_it_through_a_break(tmp_path):
    loader = Loader([card("1", "2026-11-08T15:45:00Z", most_likely=[option("goals", "over", 1.5, 0.8, 1.3)]),
                     card("now", "2026-11-17T19:45:00Z", most_likely=[option("goals", "over", 1.5, 0.8, 1.3)])],
                    {"1": ft(2, 5), "now": ft(1, 1)})
    target = tmp_path / "home_wins.json"
    # Thursday 19 Nov: this week has a graded match, but it is not over; the week before had none (a break).
    reel = home_wins.build(target, now=datetime(2026, 11, 19, 6, 0, tzinfo=timezone.utc),
                           load_cards=loader.load_cards, load_facts=loader.load_facts)
    assert reel["week"] == {"from": "2026-11-08", "to": "2026-11-08"}
    assert [w["fixture"]["event_id"] for w in reel["wins"]] == ["1"]
    # Each week is read from the Sunday before it (a kickoff after midnight UTC is still Sunday in London).
    assert loader.days == [(date(2026, 11, 8), date(2026, 11, 15)), (date(2026, 11, 1), date(2026, 11, 8))]
    assert json.loads(target.read_text()) == reel
    assert home_wins.load(target) == reel


def test_build_with_no_week_of_picks_writes_an_empty_reel_and_stops_at_the_go_live_week(tmp_path):
    loader = Loader([], {})
    target = tmp_path / "home_wins.json"
    reel = home_wins.build(target, now=NOW, load_cards=loader.load_cards, load_facts=loader.load_facts)
    assert reel == {"version": "home_wins.v1", "week": None, "record": [], "wins": []}
    assert loader.days == [(date(2026, 11, 1), date(2026, 11, 8))]  # 2-8 Nov holds the go-live; nothing before
    assert home_wins.load(target) == reel


def test_load_gives_an_empty_reel_when_the_file_is_missing_or_unreadable(tmp_path):
    empty = {"version": "home_wins.v1", "week": None, "record": [], "wins": []}
    assert home_wins.load(tmp_path / "missing.json") == empty
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert home_wins.load(broken) == empty
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"version": "home_wins.v0", "wins": [1]}))
    assert home_wins.load(other) == empty


def test_shown_at_kickoff_gives_the_full_card_the_board_showed_at_kickoff(
    settings, engine, session_factory, monkeypatch,
):
    from contextlib import nullcontext

    from web_app.routers import match_reads as board
    from Scripts.tests.data_platform.test_before_kickoff_api import started_read
    from Scripts.tests.data_platform.test_match_read_api import _service

    service = _service(session_factory)
    started_read(service, "fixture-upcoming", kickoff="2026-09-06T15:00:00Z", evaluated_at="2026-09-06T12:00:00Z")
    started_read(service, "fixture-stale", evaluated_at="2026-09-06T08:00:00Z")  # four hours old at kickoff
    started_read(service, "fixture-late", evaluated_at="2026-09-06T12:10:00Z")  # made after kickoff
    started_read(service, "fixture-shown")
    monkeypatch.setattr(board, "_get_match_read_service", lambda: service)
    monkeypatch.setenv("SPIX_RECOMMENDATIONS", "on")
    monkeypatch.setenv("SPIX_PICK_EVIDENCE", "on")
    monkeypatch.setattr(board, "_get_pick_evidence", lambda: pytest.fail("the reel needs no pick evidence"))
    [shown] = home_wins.shown_at_kickoff([date(2026, 9, 6)], now=datetime(2026, 9, 6, 12, 30, tzinfo=timezone.utc),
                                         gate=nullcontext)
    assert shown["fixture"]["event_id"] == "fixture-shown"
    block = shown["recommendations"]
    assert block["value"][0]["label"] == "Over 2.5 goals"
    [goals] = [m for m in block["markets"] if m["market"] == "goals"]
    assert {(o["side"], o["line"]) for o in goals["options"] if o["typical_price"]} == {("over", 2.5), ("under", 2.5)}


def test_stored_facts_reads_each_matchs_status_score_and_both_teams_corners(session_factory):
    from contextlib import nullcontext

    from data_platform.models import Competition, Fixture, FixtureTeamStats, Season, Team

    with session_factory() as s:
        league = Competition(code="EPL", name="Premier League", api_football_id=39,
                             competition_type="domestic_league", country="England")
        s.add(league); s.flush()
        season = Season(competition_id=league.id, year=2026, label="2026/27")
        alpha, beta = Team(name="Alpha", api_football_id=1), Team(name="Beta", api_football_id=2)
        s.add_all([season, alpha, beta]); s.flush()
        for api_id, status, score in ((501, "FT", (2, 5)), (502, "NS", (None, None))):
            f = Fixture(api_football_id=api_id, competition_id=league.id, season_id=season.id,
                        home_team_id=alpha.id, away_team_id=beta.id, status=status, home_goals=score[0],
                        away_goals=score[1], kickoff_utc=datetime(2026, 11, 8, 15, tzinfo=timezone.utc))
            s.add(f); s.flush()
            if status == "FT":
                s.add_all([FixtureTeamStats(fixture_id=f.id, team_id=alpha.id, opponent_team_id=beta.id,
                                            is_home=True, corners=4),
                           FixtureTeamStats(fixture_id=f.id, team_id=beta.id, opponent_team_id=alpha.id,
                                            is_home=False, corners=9)])
        s.commit()
    facts = home_wins.stored_facts(["501", "502", "fixture-x", "999"], session_factory=session_factory,
                                   gate=nullcontext)
    assert facts == {
        "501": {"status": "FT", "home_goals": 2, "away_goals": 5, "corners_home": 4, "corners_away": 9},
        "502": {"status": "NS", "home_goals": None, "away_goals": None, "corners_home": None, "corners_away": None}}
    assert home_wins.stored_facts(["fixture-x"], session_factory=session_factory, gate=nullcontext) == {}


def test_the_home_wins_command_writes_the_file_and_a_busy_database_is_a_warning(tmp_path, monkeypatch, capsys):
    from data_platform import cli
    from data_platform.services.refresh_coordination import RefreshBusy

    built = []
    reel = {"version": "home_wins.v1", "week": {"from": "2026-11-04", "to": "2026-11-08"},
            "record": [{"list": "most_likely", "picks": 3, "won": 2}, {"list": "value", "picks": 1, "won": 1}],
            "wins": [{}] * 3}
    monkeypatch.setattr(home_wins, "build", lambda target: built.append(target) or reel)
    assert cli.main(["home-wins", "--out", str(tmp_path / "reel.json")]) == 0
    assert built == [tmp_path / "reel.json"]
    assert "4 to 8 Nov: 3 wins on the reel" in capsys.readouterr().out
    monkeypatch.setattr(home_wins, "build", lambda target: (_ for _ in ()).throw(RefreshBusy("busy")))
    assert cli.main(["home-wins"]) == 1


def test_the_public_endpoint_serves_the_last_reel_and_an_empty_one_without_a_file(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from web_app.routers import home

    app = FastAPI()
    app.include_router(home.router)
    client = TestClient(app)
    monkeypatch.setattr(home, "WINS_PATH", tmp_path / "missing.json")
    response = client.get("/api/home/wins")
    assert response.status_code == 200
    assert response.json() == {"version": "home_wins.v1", "week": None, "record": [], "wins": []}
    reel = {"version": "home_wins.v1", "week": {"from": "2026-11-04", "to": "2026-11-08"}, "record": [], "wins": []}
    (tmp_path / "home_wins.json").write_text(json.dumps(reel))
    monkeypatch.setattr(home, "WINS_PATH", tmp_path / "home_wins.json")
    assert client.get("/api/home/wins").json() == reel
