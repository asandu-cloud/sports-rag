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


def test_service_writes_each_finished_pick_once(store):
    publish(store)
    client = Client({"pinnacle": PINNACLE}, [{"fixtureId": "op1", "startTime": "2026-09-22T18:00:00.000Z",
                                              "participant1Name": "A", "participant2Name": "B"}])
    service = tc.TrueCloseService(store[0], client, INDEX, clock=lambda: NOW)
    rows = store[0].get_recent(days=3650)
    assert service.run(rows) == {"due": 1, "fixtures": 1, "matched": 1, "final": 1, "unavailable": 0,
                                 "unmatched": 0, "conflicts": 0}
    saved = store[0].get_recent(days=3650)[0]["true_close"]
    assert saved["status"] == "final" and saved["oddspapi_fixture_id"] == "op1"
    assert saved["consensus"]["fair_probability_power"] == pytest.approx(0.5)
    assert service.run(store[0].get_recent(days=3650))["due"] == 0
    assert [c[0] for c in client.calls] == ["/v4/fixtures", "/v4/historical-odds"]  # nothing due: no request


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
    assert config["StartCalendarInterval"] == {"Hour": 10, "Minute": 30}
    assert config["ProgramArguments"][-2:] == ["-m", "Scripts.ops.true_close"]
    assert schedule.operate("install", paths, dry_run=True)["configuration"]["Label"] == schedule.LABEL
    assert not paths.plist_path.exists()
