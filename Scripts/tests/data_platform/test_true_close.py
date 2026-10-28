"""True closing prices from OddsPapi's history (rework A1 step 4, D18)."""
from datetime import datetime, timedelta, timezone

import pytest

from data_platform import oddspapi
from data_platform.services import true_close as tc
from .test_safe_settlement import store, publish, payload, NOW, no_network  # noqa: F401  (fixtures)

KICKOFF = datetime(2026, 9, 22, 18, tzinfo=timezone.utc)
MARKETS = [
    {"marketId": 1010, "marketType": "totals", "handicap": 2.5, "period": "fulltime", "playerProp": False, "sportId": 10,
     "outcomes": [{"outcomeId": 1010, "outcomeName": "Under"}, {"outcomeId": 1011, "outcomeName": "Over"}]},
    {"marketId": 1050, "marketType": "spreads", "handicap": -0.5, "period": "fulltime", "playerProp": False, "sportId": 10,
     "outcomes": [{"outcomeId": 1050, "outcomeName": "1"}, {"outcomeId": 1051, "outcomeName": "2"}]},
]
INDEX = oddspapi.market_index(MARKETS)


def entry(minutes_before, price, active=True):
    return {"createdAt": (KICKOFF - timedelta(minutes=minutes_before)).isoformat().replace("+00:00", "Z"),
            "price": price, "active": active}


def history(slug, outcomes):
    markets = {}
    for (market_id, outcome_id), entries in outcomes.items():
        markets.setdefault(market_id, {"outcomes": {}})["outcomes"][outcome_id] = {"players": {"0": entries}}
    return {"bookmakers": {slug: {"markets": markets}}}


PINNACLE = history("pinnacle", {
    ("1010", "1011"): [entry(300, 2.10), entry(3, 1.95), entry(-10, 1.40)],  # in-play change ignored
    ("1010", "1010"): [entry(300, 1.80), entry(3, 1.95), entry(-10, 3.0)],
    ("1050", "1050"): [entry(60, 1.90)], ("1050", "1051"): [entry(60, 2.00)],
})


class Client:
    def __init__(self, bodies, fixtures):
        self.bodies, self.fixtures, self.calls = bodies, fixtures, []

    def get(self, path, **params):
        self.calls.append((path, params))
        if path == "/v4/fixtures":
            return self.fixtures
        return {"bookmakers": {k: v for slug in params["bookmakers"].split(",")
                               for k, v in (self.bodies.get(slug) or {}).get("bookmakers", {}).items()}}


def test_true_close_is_pinnacles_last_pre_kickoff_price_with_its_margin_removed():
    pick = {"market": "goals", "home_team": "A", "away_team": "B", "bookmaker": "Book",
            "tracking": {"selection": {"side": "over", "line": 2.5, "bookmaker": "Book"}}}
    record = tc.true_close(pick, {"pinnacle": PINNACLE}, INDEX, KICKOFF)
    assert record["status"] == "final" and "same_book" not in record  # "Book" is not in OddsPapi
    consensus = record["consensus"]
    assert consensus["prices"] == [["over", 2.5, 1.95], ["under", 2.5, 1.95]]
    assert consensus["fair_probability_proportional"] == pytest.approx(0.5)
    assert consensus["minutes_before_kickoff"] == 3


def test_handicap_picks_use_the_home_line_and_their_own_bookmakers_close():
    pick = {"market": "spreads", "home_team": "A", "away_team": "B", "bookmaker": "Pinnacle",
            "tracking": {"selection": {"side": "B", "line": 0.5, "bookmaker": "Pinnacle"}}}
    record = tc.true_close(pick, {"pinnacle": PINNACLE}, INDEX, KICKOFF)
    assert record["consensus"]["prices"] == [["a", -0.5, 1.90], ["b", 0.5, 2.00]]
    assert record["consensus"]["fair_probability_proportional"] == pytest.approx((1 / 2.0) / (1 / 1.9 + 1 / 2.0))
    assert record["same_book"]["odds"] == 2.00 and record["same_book"]["site"] == "pinnacle"


def test_a_line_missing_from_the_history_is_unavailable():
    pick = {"market": "goals", "home_team": "A", "away_team": "B", "bookmaker": "Book",
            "tracking": {"selection": {"side": "over", "line": 3.5, "bookmaker": "Book"}}}
    assert tc.true_close(pick, {"pinnacle": PINNACLE}, INDEX, KICKOFF) == {
        "source": tc.SOURCE, "kickoff": KICKOFF.isoformat(), "status": "unavailable", "reason": "line_not_in_history"}


def test_fixture_lists_are_requested_per_competition_and_week():
    assert tc.week_window(KICKOFF) == ("2026-09-20T00:00:00Z", "2026-09-29T00:00:00Z")  # Tue 22 Sep: week of Mon 21


FIXTURES = [{"fixtureId": "op1", "startTime": "2026-09-22T18:00:00.000Z", "participant1Name": "A", "participant2Name": "B"}]


class Failing(Client):
    """OddsPapi answering an error for some bookmakers' histories (21 Oct: 8 fixtures 2.5 h after kickoff)."""

    def __init__(self, bodies, fixtures, failing, status=503, code="UNAVAILABLE"):
        super().__init__(bodies, fixtures)
        self.failing, self.status, self.code = failing, status, code

    def get(self, path, **params):
        if path == "/v4/historical-odds" and params["bookmakers"] in self.failing:
            self.calls.append((path, params))
            raise oddspapi.OddsPapiError(path, self.status, {"error": {"code": self.code}})
        return super().get(path, **params)


def not_found(bodies, failing):
    """OddsPapi's "No historical odds found" (404 NOT_FOUND) for the ``failing`` bookmaker groups."""
    return Failing(bodies, FIXTURES, failing, 404, "NOT_FOUND")


def rows(store):
    return store[0].get_recent(days=3650)


def test_service_writes_each_finished_pick_once(store):
    publish(store)
    client = Client({"pinnacle": PINNACLE}, FIXTURES)
    service = tc.TrueCloseService(store[0], client, INDEX, clock=lambda: NOW)
    rows = store[0].get_recent(days=3650)
    assert service.run(rows) == {"due": 1, "fixtures": 1, "matched": 1, "final": 1, "unavailable": 0,
                                 "unmatched": 0, "errors": 0, "conflicts": 0}
    saved = store[0].get_recent(days=3650)[0]["true_close"]
    assert saved["status"] == "final" and saved["oddspapi_fixture_id"] == "op1"
    assert saved["consensus"]["fair_probability_power"] == pytest.approx(0.5)
    assert service.run(store[0].get_recent(days=3650))["due"] == 0
    assert [c[0] for c in client.calls] == ["/v4/fixtures", "/v4/historical-odds"]  # nothing due: no request


def test_an_oddspapi_error_leaves_the_pick_due_for_the_next_run(store):
    publish(store)
    failing = tc.TrueCloseService(store[0], Failing({"pinnacle": PINNACLE}, FIXTURES, {"pinnacle"}), INDEX,
                                  clock=lambda: KICKOFF + timedelta(hours=3))
    assert failing.run(rows(store))["errors"] == 1
    assert rows(store)[0]["true_close"] is None  # not "unavailable": nothing was learned about the prices
    working = tc.TrueCloseService(store[0], Client({"pinnacle": PINNACLE}, FIXTURES), INDEX,
                                  clock=lambda: KICKOFF + timedelta(hours=4))
    assert working.run(rows(store))["final"] == 1


def test_an_error_for_the_picks_own_bookmaker_defers_the_whole_fixture(store):
    bet365 = payload()
    bet365["decision"]["quote"]["bookmaker"] = "Bet365"
    publish(store, bet365)
    client = Failing({"pinnacle": PINNACLE}, FIXTURES, {"bet365.es"})
    service = tc.TrueCloseService(store[0], client, INDEX, clock=lambda: NOW)
    assert service.run(rows(store))["errors"] == 1 and rows(store)[0]["true_close"] is None


def test_after_three_days_an_error_or_a_missing_fixture_is_recorded_as_unavailable(store):
    publish(store)
    late = KICKOFF + tc.GIVE_UP + timedelta(minutes=1)
    tc.TrueCloseService(store[0], Failing({}, FIXTURES, {"pinnacle"}), INDEX, clock=lambda: late).run(rows(store))
    saved = rows(store)[0]["true_close"]
    assert (saved["status"], saved["reason"], saved["error"]) == ("unavailable", "oddspapi_error", "UNAVAILABLE")


def test_a_bookmaker_with_no_history_has_no_price_once_the_match_has_settled(store):
    bet365 = payload()
    bet365["decision"]["quote"]["bookmaker"] = "Bet365"
    publish(store, bet365)  # 14 Oct, QPR v Derby: William Hill had no history; it failed the whole fixture
    soon = tc.TrueCloseService(store[0], not_found({"pinnacle": PINNACLE}, {"bet365.es"}), INDEX,
                               clock=lambda: KICKOFF + timedelta(hours=3))
    assert soon.run(rows(store))["errors"] == 1 and rows(store)[0]["true_close"] is None  # may still be filling in
    settled = tc.TrueCloseService(store[0], not_found({"pinnacle": PINNACLE}, {"bet365.es"}), INDEX,
                                  clock=lambda: KICKOFF + tc.SETTLED)
    assert settled.run(rows(store))["final"] == 1
    saved = rows(store)[0]["true_close"]
    assert "consensus" in saved and "same_book" not in saved


def test_a_fixture_not_found_for_three_days_is_recorded_as_unavailable(store):
    publish(store)
    early = tc.TrueCloseService(store[0], Client({}, []), INDEX, clock=lambda: NOW)
    assert early.run(rows(store))["unmatched"] == 1 and rows(store)[0]["true_close"] is None
    late = KICKOFF + tc.GIVE_UP + timedelta(minutes=1)
    assert tc.TrueCloseService(store[0], Client({}, []), INDEX, clock=lambda: late).run(rows(store))["unavailable"] == 1
    assert rows(store)[0]["true_close"]["reason"] == "fixture_not_matched"


def test_a_fixture_list_error_is_not_read_as_a_missing_fixture(store):
    publish(store)

    class ListDown(Client):
        def get(self, path, **params):
            if path == "/v4/fixtures":
                raise oddspapi.OddsPapiError(path, 503, None)
            return super().get(path, **params)

    late = KICKOFF + tc.GIVE_UP + timedelta(minutes=1)
    counts = tc.TrueCloseService(store[0], ListDown({}, FIXTURES), INDEX, clock=lambda: late).run(rows(store))
    assert counts["errors"] == 1 and rows(store)[0]["true_close"] is None


class Exchange(Client):
    """OddsPapi's rule for exchanges: 'betfair-ex' alone and with one outcomeId, else 400 INVALID_PARAMETER."""

    def get(self, path, **params):
        if path == "/v4/historical-odds" and "betfair-ex" in params["bookmakers"]:
            if params["bookmakers"] != "betfair-ex" or "outcomeId" not in params:
                raise oddspapi.OddsPapiError(path, 400, {"error": {"code": "INVALID_PARAMETER"}})
            self.calls.append((path, params))
            markets = self.bodies["betfair-ex"]["bookmakers"]["betfair-ex"]["markets"]
            kept = {m: {"outcomes": {o: v for o, v in market["outcomes"].items() if o == params["outcomeId"]}}
                    for m, market in markets.items()}
            return {"bookmakers": {"betfair-ex": {"markets": {m: v for m, v in kept.items() if v["outcomes"]}}}}
        return super().get(path, **params)


def test_betfair_exchange_prices_are_requested_alone_one_outcome_at_a_time(store, monkeypatch):
    monkeypatch.setitem(oddspapi.BOOKS, "Betfair Exchange", "betfair-ex")  # no feed quotes the exchange now (D43)
    betfair = payload()
    betfair["decision"]["quote"]["bookmaker"] = "Betfair Exchange"
    publish(store, betfair)
    exchange = history("betfair-ex", {("1010", "1011"): [entry(5, 2.02)], ("1010", "1010"): [entry(5, 1.98)]})
    client = Exchange({"pinnacle": PINNACLE, "betfair-ex": exchange}, FIXTURES)
    assert tc.TrueCloseService(store[0], client, INDEX, clock=lambda: NOW).run(rows(store))["final"] == 1
    saved = rows(store)[0]["true_close"]
    assert (saved["same_book"]["site"], saved["same_book"]["odds"]) == ("betfair-ex", 2.02)
    assert ("/v4/historical-odds", {"fixtureId": "op1", "bookmakers": "betfair-ex", "outcomeId": "1011"}) in client.calls


def test_betfair_sportsbook_picks_get_pinnacles_close_but_no_own_close(store):
    """API-Football's Betfair is the sportsbook (D43): OddsPapi has no history for it on our plan."""
    betfair = payload()
    betfair["decision"]["quote"]["bookmaker"] = "Betfair"
    publish(store, betfair)
    client = Client({"pinnacle": PINNACLE}, FIXTURES)
    assert tc.TrueCloseService(store[0], client, INDEX, clock=lambda: NOW).run(rows(store))["final"] == 1
    saved = rows(store)[0]["true_close"]
    assert "consensus" in saved and "same_book" not in saved
    assert [c[1]["bookmakers"] for c in client.calls if c[0] == "/v4/historical-odds"] == ["pinnacle"]


def test_a_match_moved_after_publication_uses_its_settled_kickoff(store):
    publish(store)  # 10 Oct: Casa Pia v Santa Clara was published for 14:30 UTC and played at 19:30
    moved = [{**row, "settlement": {"result": {"kickoff": "2026-09-22T23:00:00+00:00"}}} for row in rows(store)]
    listed_late = [{**FIXTURES[0], "startTime": "2026-09-22T23:00:00.000Z"}]
    service = tc.TrueCloseService(store[0], Client({"pinnacle": PINNACLE}, listed_late), INDEX, clock=lambda: NOW)
    assert service.run(moved)["final"] == 1
    saved = rows(store)[0]["true_close"]
    assert saved["kickoff"] == "2026-09-22T23:00:00+00:00"
    assert saved["consensus"]["prices"] == [["over", 2.5, 1.40], ["under", 2.5, 3.0]]  # last before 23:00


def test_recapture_replaces_a_fixtures_records(store):
    publish(store)
    tc.TrueCloseService(store[0], Client({}, FIXTURES), INDEX, clock=lambda: NOW).run(rows(store))
    assert rows(store)[0]["true_close"]["status"] == "unavailable"
    fixed = tc.TrueCloseService(store[0], Client({"pinnacle": PINNACLE}, FIXTURES), INDEX, clock=lambda: NOW)
    assert fixed.run(rows(store))["due"] == 0
    assert fixed.run(rows(store), recapture={str(rows(store)[0]["fixture_id"])})["final"] == 1
    assert rows(store)[0]["true_close"]["status"] == "final"


class Response:
    def __init__(self, status, body):
        self.status_code, self.body = status, body

    def json(self):
        if self.body is None:
            raise ValueError("not JSON")
        return self.body


class Session:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def get(self, url, params, timeout):
        self.calls += 1
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_an_error_answer_raises_and_is_never_cached(tmp_path):
    session = Session(Response(500, {"error": {"code": "INTERNAL", "message": "boom"}}), Response(502, None))
    client = oddspapi.OddsPapiClient("k", tmp_path, session=session, sleep=lambda s: None)
    with pytest.raises(oddspapi.OddsPapiError) as info:
        client.get("/v4/historical-odds", fixtureId="x", bookmakers="pinnacle")
    assert (info.value.status, info.value.code) == (500, "INTERNAL")
    with pytest.raises(oddspapi.OddsPapiError) as info:  # an HTML error page
        client.get("/v4/historical-odds", fixtureId="x", bookmakers="pinnacle")
    assert info.value.status == 502 and session.calls == 2
    assert not list(tmp_path.glob("*.json*"))


def test_a_rate_limited_request_waits_as_told_and_retries(tmp_path):
    waits = []
    limited = Response(429, {"error": {"code": "RATE_LIMITED", "retryMs": 4295}})
    session = Session(limited, Response(200, {"bookmakers": {}}))
    client = oddspapi.OddsPapiClient("k", tmp_path, session=session, sleep=waits.append)
    assert client.get("/v4/historical-odds", fixtureId="x", bookmakers="pinnacle") == {"bookmakers": {}}
    assert session.calls == 2 and any(w >= 4.295 for w in waits)


def test_a_network_failure_is_an_oddspapi_error_that_never_shows_the_key(tmp_path):
    import requests
    session = Session(requests.ConnectionError("Max retries exceeded with url: /v4/account?apiKey=SECRET"))
    client = oddspapi.OddsPapiClient("SECRET", tmp_path, session=session, sleep=lambda s: None)
    with pytest.raises(oddspapi.OddsPapiError) as info:
        client.get("/v4/historical-odds", fixtureId="x", bookmakers="pinnacle")
    assert "SECRET" not in str(info.value) and info.value.__cause__ is None and info.value.__suppress_context__
    assert info.value.code == "ConnectionError"


def test_an_unreachable_account_check_is_reported_without_the_key(monkeypatch):
    import requests
    from Scripts.ops import true_close as job

    def down(*args, **kwargs):
        raise requests.ConnectionError("Max retries exceeded with url: /v4/account?apiKey=SECRET")

    monkeypatch.setattr(requests, "get", down)
    with pytest.raises(job.AccountUnavailable) as info:
        job.quota_budget(None, "SECRET", 40)
    assert "SECRET" not in str(info.value) and info.value.__suppress_context__


def fixture_at(home, away, when=datetime(2026, 10, 13, 18, 45, tzinfo=timezone.utc)):
    return [{"fixture_id": "1", "kickoff": when, "home": home, "away": away}]


def listed(*pairs, start="2026-10-13T18:45:00.000Z"):
    return [{"fixtureId": f"op{n}", "startTime": start, "participant1Name": home, "participant2Name": away}
            for n, (home, away) in enumerate(pairs)]


def test_turkish_and_german_letters_match_their_plain_spelling():
    assert oddspapi.match_fixtures(fixture_at("Kasımpaşa", "Gençlerbirliği S.K."),
                                   listed(("Kasimpasa Istanbul", "Genclerbirligi SK"))) == {"1": "op0"}


def test_a_short_name_matches_when_only_one_fixture_at_that_kickoff_has_the_other_team():
    same_time = listed(("Bristol City", "Blackburn Rovers"), ("Wolverhampton Wanderers", "Bolton Wanderers"))
    assert oddspapi.match_fixtures(fixture_at("Wolves", "Bolton"), same_time) == {"1": "op1"}
    assert oddspapi.match_fixtures(fixture_at("1. FC Köln", "Borussia Mönchengladbach"),
                                   listed(("1. FC Cologne", "Borussia Monchengladbach"))) == {"1": "op0"}


def test_one_team_is_not_enough_when_it_is_only_a_shared_word_or_two_fixtures_fit():
    assert oddspapi.match_fixtures(fixture_at("Wolves", "Swansea City"),
                                   listed(("Wolverhampton Wanderers", "Bristol City"))) == {}
    assert oddspapi.match_fixtures(fixture_at("Paris FC", "Rennes"),
                                   listed(("Paris FC", "Stade Rennais FC"), ("Paris Saint-Germain", "FC Lorient"))) == {}


def test_nothing_is_due_until_two_hours_after_kickoff(store):
    publish(store)
    early = tc.TrueCloseService(store[0], Client({}, []), INDEX, clock=lambda: KICKOFF + timedelta(minutes=90))
    assert early.due(store[0].get_recent(days=3650)) == []


def test_a_line_withdrawn_before_kickoff_has_no_close():
    withdrawn = history("pinnacle", {("1010", "1011"): [entry(40000, 2.0), entry(30000, 2.0, active=False)],
                                     ("1010", "1010"): [entry(40000, 1.8), entry(30000, 1.8, active=False)]})
    pick = {"market": "goals", "home_team": "A", "away_team": "B", "bookmaker": "Pinnacle",
            "tracking": {"selection": {"side": "over", "line": 2.5, "bookmaker": "Pinnacle"}}}
    record = tc.true_close(pick, {"pinnacle": withdrawn}, INDEX, KICKOFF)
    assert record["status"] == "unavailable"  # not the price from three weeks before kickoff
    assert oddspapi.last_before([(KICKOFF - timedelta(days=21), 2.0, True), (KICKOFF - timedelta(days=20), 2.0, False)],
                                KICKOFF) is None


def test_true_close_runs_daily_as_its_own_job(tmp_path):
    import plistlib
    from Scripts.ops import true_close_launchd as schedule
    from Scripts.ops.match_read_launchd import build_paths
    paths = build_paths(root=tmp_path, plist_path=tmp_path / "job.plist")
    config = plistlib.loads(schedule.render(paths).encode())
    assert config["Label"] == "com.bettingrag.true-close"
    assert config["StartCalendarInterval"] == {"Minute": 30}
    assert config["ProgramArguments"][-2:] == ["-m", "Scripts.ops.true_close"]
    assert schedule.operate("install", paths, dry_run=True)["configuration"]["Label"] == schedule.LABEL
    assert not paths.plist_path.exists()


def true_close_job(monkeypatch, *, budget=None, run=None):
    """``ops.true_close.main`` with OddsPapi, the database and the chat replaced; returns the chat hooks' calls."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    from Scripts.ops import true_close as job
    from Scripts.data_platform import oddspapi as client_module
    from Scripts.data_platform.repositories import predictions
    from Scripts.data_platform.services import measurement_runtime, ops_alerts, refresh_coordination
    from Scripts.data_platform.services import true_close as service
    calls = []
    monkeypatch.setenv("ODDSPAPI_API_KEY", "test-key")
    monkeypatch.setattr(client_module, "OddsPapiClient", lambda key, compress: SimpleNamespace(get=lambda path: {}))
    monkeypatch.setattr(client_module, "market_index", lambda body: {})
    monkeypatch.setattr(job, "quota_budget", budget or (lambda client, key, reserve: (66, 45)))
    monkeypatch.setattr(predictions, "PredictionRepository",
                        lambda: SimpleNamespace(_factory=None, get_recent=lambda **_: []))
    monkeypatch.setattr(measurement_runtime, "MeasurementRuntime", lambda factory: SimpleNamespace(state=lambda c: {}))
    monkeypatch.setattr(refresh_coordination, "data_access", nullcontext)
    monkeypatch.setattr(service, "TrueCloseService", lambda repo, client, index: SimpleNamespace(
        run=run or (lambda rows, **_: {"fixtures": 0, "errors": 0})))
    monkeypatch.setattr(ops_alerts, "oddspapi_left", lambda left, *, reserve: calls.append(("left", left, reserve)))
    monkeypatch.setattr(ops_alerts, "true_close_run",
                        lambda counts=None, *, problem=None: calls.append(("run", counts, problem)))
    return job, calls


def test_the_true_close_job_reports_the_allowance_and_each_run(monkeypatch):
    job, calls = true_close_job(monkeypatch)
    assert job.main([]) == 0
    assert calls == [("left", 45, 40), ("run", {"fixtures": 0, "errors": 0}, None)]


def test_the_true_close_job_reports_when_it_cannot_run(monkeypatch):
    from Scripts.ops import true_close as job_module

    def unreachable(client, key, reserve):
        raise job_module.AccountUnavailable("ConnectionError")
    job, calls = true_close_job(monkeypatch, budget=unreachable)
    assert job.main([]) == 0 and calls == [("run", None, "unreachable")]

    def broken(rows, **_):
        raise KeyError("odds")
    job, calls = true_close_job(monkeypatch, run=broken)
    with pytest.raises(KeyError):
        job.main([])
    assert calls == [("left", 45, 40), ("run", None, "crashed")]
