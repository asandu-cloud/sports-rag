"""A5.4 replay (D58, D61): cards priced from Pinnacle's fair price, on stored reads' prices."""
from datetime import datetime, timedelta, timezone

import pytest

from Scripts.data_platform import devig, oddspapi
from Scripts.ops import cards_reference_replay as cr

HOME, AWAY = "Lyon", "Lens"
REFEREE = "Cards recommendations require a profiled referee assignment."
PRICES = {"Pinnacle": {4.5: (2.10, 1.75)}, "Bet365": {4.5: (1.85, 1.95)}, "Betano": {4.5: (1.80, 2.05)}}


def options(prices=PRICES):
    return [{"side": side, "point": point, "odds": odds, "bookmaker": book, "market_key": cr.MARKET_KEY}
            for book, lines in prices.items() for point, pair in lines.items()
            for side, odds in zip(("over", "under"), pair) if odds]


def stored_read(prices=PRICES, *, reasons=(REFEREE,)):
    cards = {"market": {"group": "cards", "key": "totals"},
             "fixture": {"event_id": "7", "home_team": HOME, "away_team": AWAY},
             "projection": {"value": 4.0, "variance": 6.0}, "decision": {"status": "no_bet"},
             "context": {"decision_audit": {"input_options": options(prices)},
                         "data_quality": {"recommendation_guardrail": {"eligible": not reasons,
                                                                       "reasons": list(reasons)}},
                         "release_policy": {"mode": "live"}}}
    return {"canonical_results": [cards]}


def fair_under(over, under):
    return devig.fair([over, under], "power")[1]


def closing(pinnacle=(2.00, 1.83)):
    books = [{"name": "Betano", "bets": [{"id": 80, "values": [{"value": "Over 4.5", "odd": "1.80"},
                                                                {"value": "Under 4.5", "odd": "2.05"}]}]}]
    if pinnacle:
        books.append({"name": "Pinnacle", "bets": [{"id": 80, "values": [
            {"value": "Over 4.5", "odd": str(pinnacle[0])}, {"value": "Under 4.5", "odd": str(pinnacle[1])}]}]})
    return {"response": [{"bookmakers": books}]}


def test_a_book_beating_pinnacles_fair_price_is_the_pick_graded_on_the_bookmakers_count():
    [pick] = cr.replay_read(stored_read())
    assert (pick["side"], pick["point"], pick["bookmaker"], pick["odds"]) == ("under", 4.5, "Betano", 2.05)
    assert pick["p"] == pytest.approx(fair_under(2.10, 1.75)) and pick["ev"] == pytest.approx(pick["p"] * 2.05 - 1)
    row = cr.grade(pick, {"home_goals": 1, "away_goals": 0, "cards": 3}, closing(), HOME, AWAY)
    assert (row["y"], row["w"]) == (1.0, 1.0)
    assert row["p_close"] == pytest.approx(fair_under(2.00, 1.83))  # Pinnacle's close alone, not the median


def test_no_two_sided_pinnacle_price_means_no_pick():
    one_sided = {**PRICES, "Pinnacle": {4.5: (2.10, None)}}
    assert cr.replay_read(stored_read(one_sided)) == []
    assert cr.replay_read(stored_read({k: v for k, v in PRICES.items() if k != "Pinnacle"})) == []


def test_any_guardrail_but_the_referee_still_blocks_the_pick():
    thin = "Fewer than 2 completed current-season matches for one or both teams."
    assert cr.replay_read(stored_read(reasons=(REFEREE, thin))) == []
    assert len(cr.replay_read(stored_read(reasons=()))) == 1


def test_one_pick_per_match_the_best_value_line():
    prices = {"Pinnacle": {3.5: (1.60, 2.40), 4.5: (2.10, 1.75)}, "Betano": {3.5: (1.55, 2.55), 4.5: (1.80, 2.05)}}
    [pick] = cr.replay_read(stored_read(prices))
    values = {p: fair_under(*prices["Pinnacle"][p]) * prices["Betano"][p][1] - 1 for p in (3.5, 4.5)}
    assert pick["point"] == max(values, key=values.get)


def test_without_pinnacles_close_a_pick_counts_in_results_not_in_value_at_the_close():
    [pick] = cr.replay_read(stored_read())
    rows = [{**cr.grade(pick, {"home_goals": 1, "away_goals": 0, "cards": 6}, closing(pinnacle=None), HOME, AWAY), "fixture_id": "7"}]
    summary = cr.summarize(rows, resamples=10)
    assert summary["all"]["graded"] == 1 and summary["all"]["won"] == 0.0
    assert summary["all"]["closing_picks"] == 0 and summary["all"]["value_at_close"] is None
    assert set(summary) == {"all", "under"}


AT = datetime(2026, 10, 24, 12, tzinfo=timezone.utc)
INDEX = oddspapi.market_index([
    {"marketId": 1200, "marketType": "totals-bookings", "handicap": 4.5, "period": "fulltime", "playerProp": False,
     "sportId": 10, "outcomes": [{"outcomeId": 1200, "outcomeName": "Under"}, {"outcomeId": 1201, "outcomeName": "Over"}]}])


def pinnacle_history(over, under, minutes_before=5):
    when = (AT - timedelta(minutes=minutes_before)).isoformat().replace("+00:00", "Z")
    later = (AT + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    return {"bookmakers": {"pinnacle": {"markets": {"1200": {"outcomes": {
        "1201": {"players": {"0": [{"createdAt": when, "price": over}, {"createdAt": later, "price": 1.20}]}},
        "1200": {"players": {"0": [{"createdAt": when, "price": under}, {"createdAt": later, "price": 4.00}]}}}}}}}}


def test_pinnacles_own_price_at_the_read_decides_whether_the_check_keeps_the_pick():
    pick = {"market": "cards", "side": "under", "point": 4.5, "odds": 2.05, "bookmaker": "Betano"}
    fixture = {"home_team": HOME, "away_team": AWAY}
    kept = cr.pinnacle_then(pick, fixture, pinnacle_history(2.05, 1.78), AT, INDEX)
    assert kept["status"] == "kept"  # entries after the read are not seen
    assert kept["fair_probability"] == pytest.approx(fair_under(2.05, 1.78))
    assert cr.pinnacle_then(pick, fixture, pinnacle_history(1.85, 2.00), AT, INDEX) == {
        "status": "dropped", "reason": "reference_moved", "fair_probability": pytest.approx(fair_under(1.85, 2.00))}
    stale = cr.pinnacle_then(pick, fixture, pinnacle_history(2.05, 1.78, minutes_before=90), AT, INDEX)
    assert stale["status"] == "dropped" and stale["reason"] == "reference_not_recent"
    assert cr.pinnacle_then(pick, fixture, None, AT, INDEX) == {"status": "unknown", "reason": "history_not_cached"}


def test_pinnacles_true_close_is_its_last_price_before_kickoff():
    pick = {"market": "cards", "side": "under", "point": 4.5, "odds": 2.05, "bookmaker": "Betano"}
    fixture = {"home_team": HOME, "away_team": AWAY}
    record = cr.pinnacle_then(pick, fixture, pinnacle_history(2.05, 1.78), AT, INDEX, kickoff=AT + timedelta(hours=1))
    assert record["true_close_probability"] == pytest.approx(fair_under(1.20, 4.00))  # the entry 30 min after the read
    rows = [{**pick, "p": 0.55, "ev": 0.1, "y": 1.0, "w": 1.0, "p_close": None, "fixture_id": "7",
             "p_true_close": record["true_close_probability"]}]
    assert cr.summarize(rows, resamples=10)["all"]["value_at_true_close"] == pytest.approx(
        fair_under(1.20, 4.00) * 2.05 - 1)
