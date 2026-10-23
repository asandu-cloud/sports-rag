"""D42 open item: does OddsPapi's post-match rewrite change bookmakers' pre-kickoff prices?"""
from datetime import datetime, timedelta, timezone
import gzip
import json

from Scripts.data_platform import oddspapi
from Scripts.ops import history_rewrite_check as hrc
from Scripts.tests.test_price_check import MARKETS

KO = datetime(2026, 10, 24, 14, tzinfo=timezone.utc)
FIXTURE = {"fixtureId": "op1", "startTime": "2026-10-24T14:00:00Z", "participant1Name": "Arsenal",
           "participant2Name": "Leeds", "tournament": "EPL"}


def at(minutes_after):
    return KO + timedelta(minutes=minutes_after)


def entry(minutes_after, price, active=True):
    return {"createdAt": at(minutes_after).isoformat().replace("+00:00", "Z"), "price": price, "active": active}


def body(slug, lines):
    """{(market id, outcome id): [entries]} -> an OddsPapi history body for one bookmaker."""
    markets = {}
    for (market_id, outcome_id), entries in lines.items():
        markets.setdefault(market_id, {"outcomes": {}})["outcomes"][outcome_id] = {"players": {"0": entries}}
    return {"bookmakers": {slug: {"markets": markets}}}


def test_the_kickoff_capture_is_due_just_after_kickoff_and_before_the_rewrite():
    assert hrc.due_phase(KO, at(1), done=set()) is None
    assert hrc.due_phase(KO, at(5), done=set()) == "at_kickoff"
    assert hrc.due_phase(KO, at(84), done=set()) == "at_kickoff"
    assert hrc.due_phase(KO, at(100), done=set()) is None  # too close to the rewrite (~2 h after kickoff)
    assert hrc.due_phase(KO, at(5), done={"at_kickoff"}) is None


def test_the_second_capture_is_due_three_to_twelve_hours_after_kickoff():
    assert hrc.due_phase(KO, at(179), done={"at_kickoff"}) is None
    assert hrc.due_phase(KO, at(185), done={"at_kickoff"}) == "after_rewrite"
    assert hrc.due_phase(KO, at(185), done=set()) == "after_rewrite"  # kept even when the first was missed
    assert hrc.due_phase(KO, at(12 * 60), done={"at_kickoff"}) is None
    assert hrc.due_phase(KO, at(185), done={"at_kickoff", "after_rewrite"}) is None


def test_nothing_is_due_while_the_hourly_true_close_runs():
    assert hrc.due_phase(KO, at(25), done=set()) is None  # 14:25 UTC
    assert hrc.due_phase(KO, at(39), done=set()) is None
    assert hrc.due_phase(KO, at(40), done=set()) == "at_kickoff"


def test_last_pre_kickoff_prices_are_compared_line_by_line():
    before = body("betano", {("1010", "1011"): [entry(-90, 2.0), entry(-20, 1.95)],   # same in both
                             ("1010", "1010"): [entry(-90, 1.8), entry(-20, 1.85)],   # last record changed
                             ("1050", "1050"): [entry(-60, 1.9)],                     # dropped
                             ("1060", "1060"): [entry(-60, 1.7, active=False)]})      # suspended: no close in either
    after = body("betano", {("1010", "1011"): [entry(-90, 2.0), entry(-20, 1.95), entry(10, 3.0)],  # in-play ignored
                            ("1010", "1010"): [entry(-90, 1.8)],
                            ("1070", "1070"): [entry(-30, 2.2)]})                     # added
    diff = hrc.compare_closes(before, after, "betano", KO)
    assert diff["counts"] == {"same": 1, "changed": 1, "dropped": 1, "added": 1}
    assert {(d["market_id"], d["outcome_id"], d["kind"]) for d in diff["lines"]} == {
        ("1010", "1010", "changed"), ("1050", "1050", "dropped"), ("1070", "1070", "added")}
    assert diff["markets_with_close"] == {"at_kickoff": 2, "after_rewrite": 2}


class Client:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def fresh(self, path, **params):
        self.calls.append(params["bookmakers"])
        answer = self.answers.get(params["bookmakers"])
        if isinstance(answer, Exception):
            raise answer
        return answer

    def get(self, path, **params):
        raise AssertionError("captures never use the cache")


def test_a_capture_saves_each_request_and_reads_no_history_found_as_none(tmp_path):
    missing = oddspapi.OddsPapiError("/v4/historical-odds", 404, {"error": {"code": "NOT_FOUND"}})
    client = Client({"pinnacle": body("pinnacle", {}), "betano,1xbet,bet365.es": body("betano", {}),
                     "marathonbet": body("marathonbet", {}), "williamhill": missing})
    record = hrc.capture(client, FIXTURE, "at_kickoff", tmp_path, now=at(5))
    assert record["status"] == "saved" and record["groups"]["williamhill"] == "NOT_FOUND"
    assert client.calls == ["pinnacle", "betano,1xbet,bet365.es", "marathonbet", "williamhill"]
    saved = json.loads(gzip.decompress((tmp_path / "op1" / "at_kickoff.json.gz").read_bytes()))
    assert saved["requested_at"] == at(5).isoformat() and saved["bodies"]["williamhill"] is None
    assert saved["fixture"]["fixtureId"] == "op1"


def test_a_failed_request_saves_nothing_so_the_next_pass_retries(tmp_path):
    failed = oddspapi.OddsPapiError("/v4/historical-odds", 503, None, code="UNAVAILABLE")
    client = Client({"pinnacle": body("pinnacle", {}), "betano,1xbet,bet365.es": failed})
    record = hrc.capture(client, FIXTURE, "at_kickoff", tmp_path, now=at(5))
    assert record["status"] == "error" and record["error"] == "UNAVAILABLE"
    assert not (tmp_path / "op1").exists()


def test_bodies_by_bookmaker_come_from_the_request_that_asked_for_it():
    saved = {"bodies": {"pinnacle": body("pinnacle", {}), "betano,1xbet,bet365.es": body("1xbet", {}),
                        "williamhill": None}}
    bodies = hrc.bodies_by_slug(saved)
    assert bodies["1xbet"] is bodies["betano"] is saved["bodies"]["betano,1xbet,bet365.es"]
    assert bodies["williamhill"] == {} and "marathonbet" not in bodies


def test_a_picks_close_is_compared_before_and_after_the_rewrite():
    pick = {"market": "goals", "home_team": "Arsenal", "away_team": "Leeds",
            "tracking": {"selection": {"side": "over", "line": 2.5, "bookmaker": "Betano"}}}
    pinnacle = body("pinnacle", {("1010", "1011"): [entry(-10, 1.9)], ("1010", "1010"): [entry(-10, 1.95)]})
    before = {"pinnacle": pinnacle, "betano": body("betano", {("1010", "1011"): [entry(-15, 1.88)]})}
    after = {"pinnacle": pinnacle, "betano": body("betano", {("1010", "1011"): [entry(-40, 1.92)]})}
    change = hrc.pick_change(pick, before, after, oddspapi.market_index(MARKETS), KO)
    assert change == {"consensus": "same", "same_book": "changed"}
    assert hrc.pick_change(pick, before, {"pinnacle": pinnacle}, oddspapi.market_index(MARKETS), KO)["same_book"] == "dropped"


def _all_groups():
    return Client({",".join(g): body(g[0], {}) for g in hrc.GROUPS})


def test_captures_run_in_their_windows_and_stop_when_none_is_left(tmp_path):
    now, logs = [KO], []

    def sleep(seconds):
        now[0] += timedelta(seconds=seconds)
    hrc.run_captures(_all_groups(), [FIXTURE], tmp_path, clock=lambda: now[0], sleep=sleep, log=logs.append,
                     busy=lambda: False)
    assert [json.loads(line)["phase"] for line in logs] == ["at_kickoff", "after_rewrite"]
    assert now[0] < KO + timedelta(hours=4)  # stopped once both were saved


def test_no_capture_starts_while_the_true_close_is_running(tmp_path):
    now = [KO]

    def sleep(seconds):
        now[0] += timedelta(seconds=seconds)
    hrc.run_captures(_all_groups(), [FIXTURE], tmp_path, clock=lambda: now[0], sleep=sleep, log=lambda line: None,
                     busy=lambda: now[0] < at(50))
    saved = json.loads(gzip.decompress((tmp_path / "op1" / "at_kickoff.json.gz").read_bytes()))
    assert datetime.fromisoformat(saved["requested_at"]) >= at(50)
