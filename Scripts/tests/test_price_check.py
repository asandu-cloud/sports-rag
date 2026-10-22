"""A5.3: best-price quotes checked against the bookmaker's own price history."""
from datetime import datetime, timedelta, timezone
import gzip
import json

import pytest

from Scripts.ops import pinnacle_check as pc
from Scripts.ops import price_check as prc

KO = datetime(2026, 10, 10, 14, tzinfo=timezone.utc)
MARKETS = [
    {"marketId": 101, "marketType": "1x2", "handicap": 0, "period": "fulltime", "playerProp": False, "sportId": 10,
     "outcomes": [{"outcomeId": 101, "outcomeName": "1"}, {"outcomeId": 102, "outcomeName": "X"},
                  {"outcomeId": 103, "outcomeName": "2"}]},
    {"marketId": 1010, "marketType": "totals", "handicap": 2.5, "period": "fulltime", "playerProp": False, "sportId": 10,
     "outcomes": [{"outcomeId": 1010, "outcomeName": "Under"}, {"outcomeId": 1011, "outcomeName": "Over"}]},
    {"marketId": 1050, "marketType": "spreads", "handicap": -0.75, "period": "fulltime", "playerProp": False, "sportId": 10,
     "outcomes": [{"outcomeId": 1050, "outcomeName": "1"}, {"outcomeId": 1051, "outcomeName": "2"}]},
    {"marketId": 1010, "marketType": "totals", "handicap": 2.5, "period": "p1", "playerProp": False, "sportId": 10,
     "outcomes": [{"outcomeId": 9, "outcomeName": "Over"}]},
]


def line(market, side, point, home="Arsenal", away="Leeds"):
    return {"market": market, "side": side, "point": point, "home": home, "away": away}


def at(minutes_before):
    return KO - timedelta(minutes=minutes_before)


def entry(minutes_before, price, active=True):
    return {"createdAt": at(minutes_before).isoformat().replace("+00:00", "Z"), "price": price, "active": active}


def test_lines_map_to_full_time_markets_with_the_home_teams_handicap():
    index = prc.market_index(MARKETS)
    assert prc.locate(index, line("goals", "over", 2.5)) == ("1010", "1011")
    assert prc.locate(index, line("moneyline", "draw", None)) == ("101", "102")
    assert prc.locate(index, line("spreads", "arsenal", -0.75)) == ("1050", "1050")
    assert prc.locate(index, line("spreads", "leeds", 0.75)) == ("1050", "1051")  # away at minus the home line
    assert prc.locate(index, line("spreads", "leeds", -0.75)) is None
    assert prc.locate(index, line("corners", "over", 9.5)) is None


def test_a_quote_is_real_stale_never_or_suspended_at_the_read():
    points = [(at(600), 2.10, True), (at(240), 2.00, True), (at(90), 1.90, False), (at(60), 1.95, True), (at(-20), 3.0, True)]
    assert prc.check(2.00, points, at(200), KO)["status"] == "real"
    assert prc.check(2.01, points, at(200), KO)["status"] == "real"  # within 0.5%
    stale = prc.check(2.10, points, at(200), KO)
    assert stale["status"] == "stale" and stale["lag_minutes"] == pytest.approx(40)  # moved away 240 min out
    assert prc.check(2.50, points, at(200), KO)["status"] == "never"
    assert prc.check(1.90, points, at(80), KO)["status"] == "suspended"
    assert prc.check(2.00, points, at(700), KO)["status"] == "not_listed"
    assert prc.check(2.00, [], at(200), KO)["status"] == "no_history"
    assert prc.check(2.00, points, at(200), KO)["own_close"] == 1.95  # in-play prices ignored


def test_series_reads_one_outcome_in_time_order():
    body = {"bookmakers": {"betano": {"markets": {"1010": {"outcomes": {"1011": {"players": {"0": [
        entry(60, 1.9), entry(300, 2.0, active=False)]}}}}}}}}
    assert prc.series(body, "betano", "1010", "1011") == [(at(300), 2.0, False), (at(60), 1.9, True)]
    assert prc.series(body, "pinnacle", "1010", "1011") == []
    assert prc.series({"error": "x"}, "betano", "1010", "1011") == []


def test_value_at_the_bookmakers_own_price_counts_a_withdrawn_quote_as_no_bet():
    rows = [{"status": "real", "own_ev": 0.05, "claimed_ev": 0.05, "lag_minutes": None, "own_close": 1.9, "quoted": 2.0},
            {"status": "stale", "own_ev": -0.01, "claimed_ev": 0.04, "lag_minutes": 30, "own_close": 2.0, "quoted": 2.0},
            {"status": "suspended", "own_ev": None, "claimed_ev": 0.03, "lag_minutes": None, "own_close": None, "quoted": 2.0},
            {"status": "bookmaker_not_carried", "own_ev": None, "claimed_ev": 0.10, "lag_minutes": None, "own_close": None,
             "quoted": 2.0}]
    block = prc._block(rows)
    assert block["checked"] == 3 and block["real_share"] == pytest.approx(1 / 3)
    assert block["own_ev_mean"] == pytest.approx((0.05 - 0.01 + 0) / 3)
    assert block["still_value_share"] == pytest.approx(1 / 3)
    assert block["own_close_vs_quoted"] == {"shorter": 1, "same": 1, "longer": 0}


def test_compressed_cache_is_written_and_read_back(tmp_path):
    calls = []

    class Response:
        status_code = 200

        def json(self):
            return {"ok": True}

    class Session:
        def get(self, url, params, timeout):
            calls.append(url)
            return Response()

    client = pc.OddsPapiClient("k", tmp_path, budget=5, session=Session(), sleep=lambda s: None, compress=True)
    assert client.get("/v4/historical-odds", fixtureId="x") == {"ok": True}
    packed = list(tmp_path.glob("*.json.gz"))
    assert len(packed) == 1 and json.loads(gzip.decompress(packed[0].read_bytes()))["body"] == {"ok": True}
    assert client.get("/v4/historical-odds", fixtureId="x") == {"ok": True} and len(calls) == 1


def test_each_pick_is_recorded_once_even_when_its_bookmaker_is_not_carried():
    index = prc.market_index(MARKETS)
    rows = []
    for book, odds in (("Pinnacle", 2.0), ("BetVictor", 2.2)):
        for side, other in (("over", "under"),):
            rows.append({"fixture_id": "1", "stage": "pre_match", "kickoff": KO.isoformat(), "evaluated_at": at(60).isoformat(),
                         "market": "goals" if book == "Pinnacle" else "corners", "market_key": "totals", "side": side,
                         "point": 2.5, "home": "Arsenal", "away": "Leeds", "p_spix": 0.6, "p_ref": 0.55,
                         "book_count": 3, "best_odds": odds, "best_book": book, "selected": True})

    class Client:
        def get(self, path, **params):
            if path == "/v4/fixtures":
                return [{"fixtureId": "op1", "startTime": KO.isoformat().replace("+00:00", "Z"),
                         "participant1Name": "Arsenal", "participant2Name": "Leeds"}]
            return {"bookmakers": {"pinnacle": {"markets": {"1010": {"outcomes": {"1011": {"players": {"0": [entry(300, 2.0)]}}}}}}}}

    fixtures = [{"fixture_id": 1, "kickoff": KO, "league": "EPL", "home": "Arsenal", "away": "Leeds"}]
    report = prc.run(Client(), rows, fixtures, index, policies=("market",))
    statuses = sorted((p["book"], p["status"]) for p in report["picks"]["market"])
    assert statuses == [("BetVictor", "bookmaker_not_carried"), ("Pinnacle", "real")]


def _picks(*books):
    return [{"fixture_id": "1", "stage": "pre_match", "kickoff": KO.isoformat(), "evaluated_at": at(60).isoformat(),
             "market": "goals", "market_key": "totals", "side": "over", "point": 2.5, "home": "Arsenal", "away": "Leeds",
             "p_spix": 0.6, "p_ref": 0.55, "book_count": 3, "best_odds": 2.0, "best_book": book, "selected": True}
            for book in books]


FIXTURES = [{"fixture_id": 1, "kickoff": KO, "league": "EPL", "home": "Arsenal", "away": "Leeds"}]
LISTED = [{"fixtureId": "op1", "startTime": KO.isoformat().replace("+00:00", "Z"),
           "participant1Name": "Arsenal", "participant2Name": "Leeds"}]


def test_betfair_exchange_prices_are_read_alone_one_outcome_at_a_time():
    calls = []

    class Client:  # OddsPapi: 'betfair-ex' only alone and with one outcomeId, else 400 INVALID_PARAMETER
        def get(self, path, **params):
            if path == "/v4/fixtures":
                return LISTED
            calls.append(params)
            if "betfair-ex" in params["bookmakers"] and (params["bookmakers"] != "betfair-ex" or "outcomeId" not in params):
                raise prc.OddsPapiError(path, 400, {"error": {"code": "INVALID_PARAMETER"}})
            slug = params["bookmakers"].split(",")[0]
            return {"bookmakers": {slug: {"markets": {"1010": {"outcomes": {"1011": {"players": {"0": [entry(300, 2.0)]}}}}}}}}

    report = prc.run(Client(), _picks("Betfair"), FIXTURES, prc.market_index(MARKETS), policies=("market",))
    assert [p["status"] for p in report["picks"]["market"]] == ["real"]
    assert {"fixtureId": "op1", "bookmakers": "betfair-ex", "outcomeId": "1011"} in calls


def _answering(status, code):
    class Client:
        def get(self, path, **params):
            if path == "/v4/fixtures":
                return LISTED
            raise prc.OddsPapiError(path, status, {"error": {"code": code}})
    return Client()


def test_an_oddspapi_error_is_its_own_status_not_a_missing_history():
    report = prc.run(_answering(503, "UNAVAILABLE"), _picks("Bet365"), FIXTURES, prc.market_index(MARKETS),
                     policies=("market",))
    assert [p["status"] for p in report["picks"]["market"]] == ["history_error"]
    assert report["summary"]["market"]["all"]["checked"] == 0


def test_no_historical_odds_found_is_a_missing_history():
    report = prc.run(_answering(404, "NOT_FOUND"), _picks("Bet365"), FIXTURES, prc.market_index(MARKETS),
                     policies=("market",))
    assert [p["status"] for p in report["picks"]["market"]] == ["no_history"]
