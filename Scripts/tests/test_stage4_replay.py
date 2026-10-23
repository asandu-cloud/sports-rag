"""Stage 4 replay (A3.7 stage 4 bundle, D33 (b), D35): the engine's own selectors on stored reads' prices."""
import os

import pytest

from Scripts.ops import stage4_replay as s4
from Scripts.tests.test_market_goal_model import market_options

HOME, AWAY = "Lyon", "Lens"


def corners_options(books=("A", "B", "C")):
    return [{"side": side, "point": 9.5, "odds": odds, "bookmaker": book, "market_key": "totals_corners_over_under"}
            for book in books for side, odds in (("over", 1.90), ("under", 1.90))]


def result(group, options, *, eligible=True, projection=None):
    return {"market": {"group": group, "key": group},
            "fixture": {"event_id": "7", "home_team": HOME, "away_team": AWAY},
            "projection": projection or {"value": 2.7, "variance": 3.0},
            "decision": {"status": "no_bet"},
            "context": {"decision_audit": {"input_options": options},
                        "data_quality": {"recommendation_guardrail": {"eligible": eligible}},
                        "release_policy": {"mode": "live"}}}


def stored_read(*, value_book="Betano", extra=(), blocked=(), identifiable=True):
    """A read whose prices are fair from known goal means, plus a two-sided over 3.5 at ``value_book`` worth ~6%."""
    h2h_options, totals, spreads = market_options()
    totals = totals + [{"side": "over", "point": 3.5, "odds": 3.71, "bookmaker": value_book, "market_key": "totals"},
                       {"side": "under", "point": 3.5, "odds": 1.33, "bookmaker": value_book, "market_key": "totals"},
                       *extra]
    btts = [{"side": side, "odds": odds, "bookmaker": book, "market_key": "btts"}
            for book in ("A", "B") for side, odds in (("yes", 1.80), ("no", 1.95))]
    if not identifiable:  # totals alone cannot identify both teams' goal means
        h2h_options, spreads = [], []
    groups = {"goals": totals, "btts": btts, "spreads": spreads, "corners": corners_options(),
              "moneyline": h2h_options}
    projections = {"btts": {"value": 0.55, "components": {"yes_probability": 0.55}},
                   "spreads": {"value": 0.3},
                   "moneyline": {"value": 0.45, "components": {"home_probability": 0.45, "draw_probability": 0.27,
                                                               "away_probability": 0.28}},
                   "corners": {"value": 9.8, "variance": 12.0}}
    return {"canonical_results": [result(g, o, eligible=g not in blocked, projection=projections.get(g))
                                  for g, o in groups.items()]}


def goals_picks(replayed):
    return [p for p in replayed["picks"] if p["market"] == "goals"]


def test_a_value_price_is_picked_at_m1s_probability():
    replayed = s4.replay_read(stored_read(), threshold=0.02)
    [pick] = goals_picks(replayed)
    assert (pick["side"], pick["point"], pick["bookmaker"], pick["odds"]) == ("over", 3.5, "Betano", 3.71)
    assert pick["ev"] == pytest.approx(pick["p"] * 3.71 - 1) and 0.05 < pick["ev"] < 0.07
    assert not replayed["unfitted"]


def test_a_higher_threshold_drops_the_pick():
    assert goals_picks(s4.replay_read(stored_read(), threshold=0.05))
    assert not goals_picks(s4.replay_read(stored_read(), threshold=0.07))


def test_the_romanian_set_takes_only_betano_and_superbet_prices():
    elsewhere = stored_read(value_book="B")
    assert goals_picks(s4.replay_read(elsewhere, threshold=0.02, books="all"))[0]["bookmaker"] == "B"
    assert not goals_picks(s4.replay_read(elsewhere, threshold=0.02, books="romania"))
    assert goals_picks(s4.replay_read(stored_read(), threshold=0.02, books="romania"))


def test_prices_never_taken_at_stage_4_are_not_picked():
    assert not goals_picks(s4.replay_read(stored_read(value_book="SBO"), threshold=0.02))  # D39: unconfirmed


def test_a_market_the_guardrail_blocks_publishes_nothing():
    assert not goals_picks(s4.replay_read(stored_read(blocked=("goals",)), threshold=0.02))


def test_a_read_m1_cannot_fit_is_counted_and_its_goal_markets_left_out():
    replayed = s4.replay_read(stored_read(identifiable=False), threshold=0.02)
    assert replayed["unfitted"]
    assert not [p for p in replayed["picks"] if p["market"] in ("goals", "btts", "spreads")]


def test_the_switches_and_threshold_are_restored_after_a_replay():
    names = [*s4.BUNDLE, "SPIX_BOOKMAKER_SET"]
    before = {name: os.environ.get(name) for name in names}
    s4.replay_read(stored_read(), threshold=0.05, books="romania")
    assert {name: os.environ.get(name) for name in names} == before
    assert s4._engine()[0].MARKET_VALUE_MIN_EV == 0.02  # the module the replay's selectors read


def test_a_pick_is_graded_on_the_result():
    pick = {"market": "goals", "side": "over", "point": 3.5, "odds": 3.71, "p": 0.29, "ev": 0.076,
            "bookmaker": "Betano"}
    graded = s4.grade(pick, {"home_goals": 3, "away_goals": 1}, None, HOME, AWAY)
    assert (graded["y"], graded["w"], graded["p_close"]) == (1.0, 1.0, None)
    assert s4.grade(pick, {"home_goals": None, "away_goals": None}, None, HOME, AWAY) is None


def test_the_summary_counts_picks_and_fixtures_per_market():
    rows = [{"fixture_id": str(i), "market": "goals", "odds": 2.0, "p": 0.53, "ev": 0.06, "y": i % 2, "w": 1.0,
             "p_close": 0.52} for i in range(10)]
    summary = s4.summarize(rows, resamples=200)
    assert summary["goals"]["picks"] == 10 and summary["goals"]["fixtures"] == 10
    assert summary["goals"]["value_at_close"] == pytest.approx(0.04)
    assert summary["all"]["picks"] == 10


def pick(market, side, ev, point=None):
    return {"market": market, "side": side, "point": point, "odds": 2.0, "bookmaker": "Betano", "p": 0.5, "ev": ev}


def test_a_match_read_publishes_at_most_three_compatible_picks_by_value():
    picks = [pick("goals", "over", 0.06, 2.5), pick("btts", "no", 0.05), pick("corners", "over", 0.04, 9.5),
             pick("spreads", "lyon", 0.03, -0.5), pick("moneyline", "away", 0.02)]
    published = s4.publish(picks, HOME, AWAY)
    assert [p["market"] for p in published] == ["goals", "corners", "spreads"]  # BTTS no and Lens contradict


def test_spix_priced_picks_come_after_market_priced_ones():
    picks = [pick("sot", "over", 0.30, 8.5), pick("goals", "over", 0.03, 2.5), pick("corners", "under", 0.02, 9.5),
             pick("moneyline", "home", 0.025)]
    assert [p["market"] for p in s4.publish(picks, HOME, AWAY)] == ["goals", "moneyline", "corners"]


def stored_sot(bookmaker):
    sot = result("sot", [{"side": "over", "point": 8.5, "odds": 1.9, "bookmaker": bookmaker}])
    sot["decision"] = {"status": "recommended", "model_probability": 0.6, "expected_value": 0.14,
                       "quote": {"side": "over", "line": 8.5, "odds": 1.9, "bookmaker": bookmaker}}
    return sot


def test_shots_on_target_keep_todays_pick_at_an_allowed_bookmaker():
    read = stored_read()
    read["canonical_results"].append(stored_sot("Bet365"))
    assert [p for p in s4.replay_read(read, books="all")["picks"] if p["market"] == "sot"]
    assert not [p for p in s4.replay_read(read, books="romania")["picks"] if p["market"] == "sot"]


def test_todays_published_picks_are_the_reads_own_selections():
    read = stored_read()
    read["canonical_results"].append(stored_sot("Bet365"))
    read["published"] = [len(read["canonical_results"]) - 1]
    assert [p["market"] for p in s4.published_today(read)] == ["sot"]


# Price confirmation (D42) at each read, from the bookmaker's history as it stood then (cached histories only)
from datetime import timedelta  # noqa: E402

from Scripts.tests.data_platform.test_price_confirmation import INDEX, KICKOFF, NOW, entry, history  # noqa: E402

FIXTURE = {"event_id": "1635001", "league": "EPL", "home_team": "Arsenal", "away_team": "Chelsea",
           "kickoff": KICKOFF.isoformat()}
OVER = {"market": "goals", "side": "over", "point": 2.5, "odds": 2.10, "bookmaker": "Betano", "p": 0.52, "ev": 0.092,
        "profile": None}


def betano_at(over, *, seen=4):
    """Betano's over 2.5 entries as given; its under 2.5 last updated ``seen`` minutes before the read."""
    return history("betano", {("1010", "1011"): over, ("1010", "1010"): [entry(seen, 1.80)]})


def verdict(body, *, max_age=30, threshold=0.02, pick=OVER):
    return s4.confirm_pick(pick, FIXTURE, at=NOW, index=INDEX, odds_id="op1", history=body, threshold=threshold,
                           max_age=max_age)


def test_a_history_is_read_as_it_stood_at_the_read():
    body = betano_at([entry(30, 2.10), entry(-20, 1.90)])  # shortened 20 minutes after the read
    entries = s4.history_at(body, NOW)["bookmakers"]["betano"]["markets"]["1010"]["outcomes"]["1011"]["players"]["0"]
    assert [e["price"] for e in entries] == [2.10]
    assert verdict(body)["status"] == "confirmed"  # the later shortening is not seen at the read


def test_a_bookmaker_not_seen_within_the_limit_is_not_confirmed():
    body = betano_at([entry(50, 2.10)], seen=45)
    assert (verdict(body)["status"], verdict(body)["reason"]) == ("not_confirmed", "not_recent")
    assert verdict(body, max_age=None)["status"] == "confirmed"


def test_a_shorter_price_is_repriced_while_its_value_clears_the_threshold():
    body = betano_at([entry(10, 2.00)])
    assert (verdict(body)["status"], verdict(body)["own_odds"]) == ("repriced", 2.00)  # 0.52 x 2.00 - 1 = 4%
    assert verdict(body, threshold=0.05)["reason"] == "price_gone"


class CacheOnly:
    def __init__(self, cached):
        self.cached_slugs, self.asked = set(cached), []

    def cached(self, path, **params):
        return params["bookmakers"] in self.cached_slugs

    def get(self, path, **params):
        self.asked.append(params["bookmakers"])
        return betano_at([entry(30, 2.10)])


def test_a_pick_without_a_downloaded_history_or_a_listed_match_is_unknown():
    row = {**OVER, "fixture": FIXTURE, "evaluated_at": NOW.isoformat(), "threshold": 0.02}
    unmapped = s4.confirmations(row, CacheOnly({"betano"}), INDEX, {}, max_ages=(30,))
    assert unmapped["30"] == {"status": "unknown", "reason": "fixture_not_in_cached_lists"}
    client = CacheOnly(set())
    missing = s4.confirmations(row, client, INDEX, {"1635001": "op1"}, max_ages=(30,))
    assert missing["30"] == {"status": "unknown", "reason": "history_not_cached"} and client.asked == []
    found = s4.confirmations(row, CacheOnly({"betano"}), INDEX, {"1635001": "op1"}, max_ages=(30, None))
    assert found["30"]["status"] == found["none"]["status"] == "confirmed"


def test_a_bookmaker_oddspapi_does_not_carry_is_not_confirmed_without_a_history():
    row = {**OVER, "bookmaker": "Unibet FR", "fixture": FIXTURE, "evaluated_at": NOW.isoformat(), "threshold": 0.02}
    record = s4.confirmations(row, CacheOnly(set()), INDEX, {"1635001": "op1"}, max_ages=(30,))["30"]
    assert (record["status"], record["reason"]) == ("not_confirmed", "bookmaker_not_carried")


def test_the_confirmed_summary_counts_by_status_and_values_repriced_picks_at_their_own_price():
    def row(i, status, own=None):
        record = {"status": status, **({"own_odds": own} if own else {}),
                  **({"reason": "price_gone"} if status == "not_confirmed" else {})}
        return {"fixture_id": str(i), "market": "goals", "odds": 2.0, "p": 0.55, "ev": 0.1, "y": 1.0, "w": 1.0,
                "p_close": 0.52, "confirmation": {"30": record}}
    rows = [row(1, "confirmed"), row(2, "repriced", 1.95), row(3, "not_confirmed"),
            {**row(4, "confirmed"), "confirmation": {"30": {"status": "unknown", "reason": "history_not_cached"}}}]
    out = s4.confirmed_summary(rows, "30", resamples=100)
    assert (out["published"], out["confirmed"], out["repriced"], out["not_confirmed"], out["unknown"]) == (2, 1, 1, 1, 1)
    assert out["reasons"] == {"price_gone": 1}
    assert out["value_at_close"] == pytest.approx(((0.52 * 2.0 - 1) + (0.52 * 1.95 - 1)) / 2)


def test_the_read_time_is_read_in_the_databases_format():
    row = {**OVER, "fixture": FIXTURE, "evaluated_at": NOW.strftime("%Y-%m-%d %H:%M:%S.000000"), "threshold": 0.02}
    record = s4.confirmations(row, CacheOnly({"betano"}), INDEX, {"1635001": "op1"}, max_ages=(30,))["30"]
    assert record["status"] == "confirmed" and record["checked_at"] == NOW.isoformat()
