"""Price confirmation before publishing (rework A5.3, tracker D40 (a), D42)."""
from datetime import datetime, timedelta, timezone

import pytest

from data_platform import oddspapi
from data_platform.services import price_confirmation as pcf
from .test_true_close import Response, Session

KICKOFF = datetime(2026, 10, 24, 14, tzinfo=timezone.utc)
NOW = KICKOFF - timedelta(minutes=60)
MARKETS = [
    {"marketId": 1010, "marketType": "totals", "handicap": 2.5, "period": "fulltime", "playerProp": False, "sportId": 10,
     "outcomes": [{"outcomeId": 1010, "outcomeName": "Under"}, {"outcomeId": 1011, "outcomeName": "Over"}]},
    {"marketId": 1008, "marketType": "totals", "handicap": 2.25, "period": "fulltime", "playerProp": False, "sportId": 10,
     "outcomes": [{"outcomeId": 1008, "outcomeName": "Under"}, {"outcomeId": 1009, "outcomeName": "Over"}]},
]
INDEX = oddspapi.market_index(MARKETS)
FIXTURES = [{"fixtureId": "op1", "startTime": "2026-10-24T14:00:00.000Z", "participant1Name": "Arsenal FC",
             "participant2Name": "Chelsea FC"}]
MARKET_VALUE = {"engine_switches": {"SPIX_SELECTION_POLICY": "market_value", "SPIX_DEVIG_METHOD": "proportional"}}


def entry(minutes_before_now, price, active=True):
    return {"createdAt": (NOW - timedelta(minutes=minutes_before_now)).isoformat().replace("+00:00", "Z"),
            "price": price, "active": active}


def history(slug, outcomes):
    markets = {}
    for (market_id, outcome_id), entries in outcomes.items():
        markets.setdefault(market_id, {"outcomes": {}})["outcomes"][outcome_id] = {"players": {"0": entries}}
    return {"bookmakers": {slug: {"markets": markets}}}


def betano(over, *, seen=4, under=1.80):
    """Betano's history: the over 2.5 entries given, under 2.5 at ``under``, last seen ``seen`` minutes ago."""
    return history("betano", {("1010", "1011"): over, ("1010", "1010"): [entry(seen, under)]})


def result(*, bookmaker="Betano", odds=2.10, line=2.5, probability=0.52, status="recommended", context=MARKET_VALUE,
           profile=None):
    decision = {"status": status, "quote": {"side": "over", "line": line, "odds": odds, "bookmaker": bookmaker,
                                            "market_key": "totals"},
                "model_probability": probability, "implied_probability": 0.47, "value_edge": probability - 0.47,
                "expected_value": probability * odds - 1, "confidence": "medium",
                "reason": None if status == "recommended" else "Current price did not qualify.",
                "probability_basis": "asian_equivalent_non_push" if profile else "outcome_probability",
                "probability_version": "market-probability.v2"}
    if profile:
        decision["settlement_profile"] = profile
    return {"schema_version": "market_result.v1",
            "fixture": {"event_id": "1635001", "league": "EPL", "home_team": "Arsenal", "away_team": "Chelsea",
                        "kickoff": KICKOFF.isoformat()},
            "market": {"group": "goals", "key": "totals", "unit": "goals", "participant": None},
            "projection": {"value": 2.9, "unit": "goals", "components": {}},
            "decision": decision,
            "provenance": {"pipeline_version": "canonical-market-service.v1", "input_snapshot_id": "s1",
                           "generated_at": NOW.isoformat(), "model_version": "m"},
            "evidence": [], "context": dict(context)}


class Client:
    """Cached fixture lists and markets through ``get``; price histories only through ``fresh``."""

    def __init__(self, bodies, fixtures=FIXTURES, failing=()):
        self.bodies, self.fixtures, self.failing, self.calls = bodies, fixtures, set(failing), []

    def get(self, path, **params):
        self.calls.append(("get", path, params))
        if path in self.failing:
            raise oddspapi.OddsPapiError(path, 503, {"error": {"code": "UNAVAILABLE"}})
        assert path == "/v4/fixtures", "price histories must never come from the cache"
        return self.fixtures

    def fresh(self, path, **params):
        self.calls.append(("fresh", path, params))
        if path in self.failing:
            raise oddspapi.OddsPapiError(path, 503, {"error": {"code": "UNAVAILABLE"}})
        return {"bookmakers": {k: v for slug in params["bookmakers"].split(",")
                               for k, v in (self.bodies.get(slug) or {}).get("bookmakers", {}).items()}}


def confirm(results, client, **kwargs):
    return pcf.PriceConfirmation(client, INDEX, min_ev=0.02, clock=lambda: NOW, **kwargs)(results)


def record(confirmed):
    return confirmed["context"]["price_confirmation"]


def test_a_price_still_on_offer_is_confirmed_with_when_the_bookmaker_was_last_seen():
    client = Client({"betano": betano([entry(300, 2.00), entry(40, 2.12)])})
    [out] = confirm([result()], client)
    assert out["decision"]["status"] == "recommended" and out["decision"]["quote"]["odds"] == 2.10
    assert record(out) == {
        "version": pcf.VERSION, "status": "confirmed", "bookmaker": "Betano", "site": "betano",
        "quoted_odds": 2.10, "own_odds": 2.12, "own_odds_since": (NOW - timedelta(minutes=40)).isoformat(),
        "bookmaker_seen_at": (NOW - timedelta(minutes=4)).isoformat(), "checked_at": NOW.isoformat(),
        "oddspapi_fixture_id": "op1"}
    assert [c[:2] for c in client.calls] == [("get", "/v4/fixtures"), ("fresh", "/v4/historical-odds")]
    assert client.calls[1][2] == {"fixtureId": "op1", "bookmakers": "betano"}


def test_a_price_within_half_a_percent_of_the_quote_is_confirmed():
    [out] = confirm([result()], Client({"betano": betano([entry(40, 2.095)])}))
    assert record(out)["status"] == "confirmed" and out["decision"]["quote"]["odds"] == 2.10


def test_a_shorter_price_that_still_has_value_is_published_at_the_bookmakers_own_price():
    [out] = confirm([result()], Client({"betano": betano([entry(300, 2.10), entry(20, 2.02)])}))
    decision = out["decision"]
    assert decision["status"] == "recommended" and decision["quote"]["odds"] == 2.02
    assert decision["expected_value"] == pytest.approx(0.52 * 2.02 - 1)
    fair_over = (1 / 2.02) / (1 / 2.02 + 1 / 1.80)  # Betano's own two prices now, margin removed
    assert decision["implied_probability"] == pytest.approx(fair_over)
    assert decision["value_edge"] == pytest.approx(0.52 - fair_over)
    assert record(out)["status"] == "repriced" and record(out)["quoted_odds"] == 2.10
    assert record(out)["own_odds"] == 2.02


def test_an_asian_line_is_repriced_on_its_settlement_profile():
    profile = {"full_win": 0.45, "half_win": 0.10, "push": 0.0, "half_loss": 0.0, "full_loss": 0.45}
    probability = 0.5 / 0.95  # win share without the half stakes returned
    body = history("betano", {("1008", "1009"): [entry(10, 2.00)], ("1008", "1008"): [entry(10, 1.85)]})
    [out] = confirm([result(line=2.25, odds=2.20, probability=probability, profile=profile)], Client({"betano": body}))
    assert out["decision"]["quote"]["odds"] == 2.00
    assert out["decision"]["expected_value"] == pytest.approx(0.45 * 1.0 + 0.10 * 0.5 - 0.45)


def test_a_shorter_price_without_value_is_no_bet():
    [out] = confirm([result()], Client({"betano": betano([entry(20, 1.90)])}))
    assert out["decision"]["status"] == "no_bet" and out["decision"]["quote"]["odds"] == 2.10
    assert out["decision"]["reason"].startswith("Price not confirmed")
    assert record(out)["status"] == "not_confirmed" and record(out)["reason"] == "price_gone"
    assert record(out)["own_odds"] == 1.90


@pytest.mark.parametrize("body, reason", [
    (betano([entry(50, 2.12)], seen=45), "not_recent"),  # OddsPapi has not checked Betano for 45 minutes
    (betano([entry(50, 2.12), entry(10, 2.12, active=False)]), "suspended"),
    (history("betano", {("1010", "1010"): [entry(4, 1.80)]}), "line_not_listed"),
    ({"bookmakers": {}}, "no_history"),
])
def test_a_price_oddspapi_cannot_vouch_for_is_no_bet(body, reason):
    [out] = confirm([result()], Client({"betano": body}))
    assert out["decision"]["status"] == "no_bet" and record(out)["reason"] == reason


def test_how_recent_the_bookmaker_must_have_been_seen_is_set_by_max_age():
    body = betano([entry(50, 2.12)], seen=45)
    [out] = confirm([result()], Client({"betano": body}), max_age=timedelta(minutes=60))
    assert record(out)["status"] == "confirmed"
    [out] = confirm([result()], Client({"betano": betano([entry(500, 2.12)], seen=500)}), max_age=None)
    assert record(out)["status"] == "confirmed"  # no limit: the newest price, with when the bookmaker was seen
    assert record(out)["bookmaker_seen_at"] == (NOW - timedelta(minutes=500)).isoformat()


@pytest.mark.parametrize("bookmaker, reason", [("BetVictor", "bookmaker_not_carried"), ("Betfair", "bookmaker_not_carried"),
                                             ("Betfair Exchange", "exchange")])
def test_bookmakers_oddspapi_cannot_check_fail_closed(bookmaker, reason, monkeypatch):
    monkeypatch.setitem(oddspapi.BOOKS, "Betfair Exchange", "betfair-ex")  # no feed quotes the exchange now (D43)
    client = Client({})
    [out] = confirm([result(bookmaker=bookmaker)], client)
    assert out["decision"]["status"] == "no_bet" and record(out)["reason"] == reason
    assert not [c for c in client.calls if c[0] == "fresh"]


@pytest.mark.parametrize("client, reason", [
    (Client({}, failing={"/v4/historical-odds"}), "history_error"),
    (Client({}, failing={"/v4/fixtures"}), "fixture_list_error"),
    (Client({}, fixtures=[]), "fixture_not_matched"),
])
def test_oddspapi_failures_fail_closed(client, reason):
    out = confirm([result(), result(odds=2.3)], client)
    assert [o["decision"]["status"] for o in out] == ["no_bet", "no_bet"]
    assert {record(o)["reason"] for o in out} == {reason}


def test_a_competition_oddspapi_does_not_list_fails_closed():
    pick = result()
    pick["fixture"]["league"] = "Allsvenskan"
    [out] = confirm([pick], Client({}))
    assert record(out)["reason"] == "fixture_not_matched"


def test_only_market_value_reads_are_checked_and_other_results_are_untouched():
    client = Client({"betano": betano([entry(20, 1.90)])})
    legacy = [result(context={"engine_switches": {"SPIX_SELECTION_POLICY": "legacy"}})]
    assert confirm(legacy, client) == legacy and client.calls == []
    no_bet = result(status="no_bet")
    out = confirm([no_bet, result()], client)
    assert out[0] is no_bet and record(out[1])["reason"] == "price_gone"


def test_one_history_request_per_three_bookmakers_per_fixture():
    picks = [result(bookmaker=b) for b in ("Betano", "Bet365", "1xBet", "Pinnacle")]
    client = Client({})
    confirm(picks, client)
    assert [c[2]["bookmakers"] for c in client.calls if c[0] == "fresh"] == ["1xbet,bet365.es,betano", "pinnacle"]


def test_market_result_objects_come_back_as_validated_market_results():
    from rag_ingest.core.market_result import MarketResult
    pick = MarketResult.from_dict(result())
    [out] = confirm([pick], Client({"betano": betano([entry(20, 1.90)])}))
    assert isinstance(out, MarketResult) and out.decision.status.value == "no_bet"
    assert out.context["price_confirmation"]["reason"] == "price_gone"
    [out] = confirm([pick], Client({"betano": betano([entry(20, 2.02)])}))
    assert isinstance(out, MarketResult) and out.decision.is_recommended and out.decision.quote.odds == 2.02


def test_without_an_oddspapi_key_market_value_picks_fail_closed(monkeypatch):
    monkeypatch.delenv("ODDSPAPI_API_KEY", raising=False)
    [out] = pcf.confirm_prices([result()], min_ev=0.02)
    assert out["decision"]["status"] == "no_bet" and record(out)["reason"] == "no_oddspapi_key"
    legacy = [result(context={})]
    assert pcf.confirm_prices(legacy, min_ev=0.02) == legacy


def test_the_worker_keeps_one_client_so_the_request_cooldown_holds_across_fixtures(monkeypatch, tmp_path):
    made = []

    class Shared(Client):
        def __init__(self, key, cache_dir, **kwargs):
            super().__init__({"betano": betano([entry(20, 2.12)], seen=4)})
            self.budget = None
            made.append(cache_dir)

        def _usage(self):
            return {"billable": 7}

        def get(self, path, **params):
            return MARKETS if path == "/v4/markets" else super().get(path, **params)

    monkeypatch.setenv("ODDSPAPI_API_KEY", "k")
    monkeypatch.setattr(pcf.oddspapi, "OddsPapiClient", Shared)
    monkeypatch.setattr(pcf, "_shared", {})
    for _ in range(2):
        [out] = pcf.confirm_prices([result()], min_ev=0.02, clock=lambda: NOW)
        assert record(out)["status"] == "confirmed"
    assert made == [pcf.CACHE]  # one client, apart from the true close's cache
    assert pcf._shared["k"][0].budget == 9  # at most the market list and one fixture list per call


def test_a_fresh_request_never_reads_or_writes_the_cache(tmp_path):
    session = Session(Response(200, {"bookmakers": {"a": 1}}), Response(200, {"bookmakers": {"a": 2}}))
    client = oddspapi.OddsPapiClient("k", tmp_path, session=session, sleep=lambda s: None)
    assert client.fresh("/v4/historical-odds", fixtureId="x", bookmakers="betano") == {"bookmakers": {"a": 1}}
    assert client.fresh("/v4/historical-odds", fixtureId="x", bookmakers="betano") == {"bookmakers": {"a": 2}}
    assert session.calls == 2 and not list(tmp_path.glob("*.json*"))


def test_a_fresh_request_raises_on_errors_and_retries_when_rate_limited(tmp_path):
    waits = []
    session = Session(Response(429, {"error": {"code": "RATE_LIMITED", "retryMs": 4295}}),
                      Response(200, {"bookmakers": {}}), Response(404, {"error": {"code": "NOT_FOUND"}}))
    client = oddspapi.OddsPapiClient("k", tmp_path, session=session, sleep=waits.append)
    assert client.fresh("/v4/historical-odds", fixtureId="x", bookmakers="betano") == {"bookmakers": {}}
    assert any(w >= 4.295 for w in waits)
    with pytest.raises(oddspapi.OddsPapiError) as info:
        client.fresh("/v4/historical-odds", fixtureId="x", bookmakers="betano")
    assert info.value.code == "NOT_FOUND" and not list(tmp_path.glob("*.json*"))


def test_last_seen_is_the_bookmakers_newest_record_on_any_line():
    body = betano([entry(40, 2.12)], seen=4)
    assert oddspapi.last_seen(body, "betano") == NOW - timedelta(minutes=4)
    assert oddspapi.last_seen(body, "pinnacle") is None
