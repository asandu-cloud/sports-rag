"""User-approved simplified policy: isolated stores and no provider network."""
from copy import deepcopy
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from data_platform.models import Prediction
from data_platform.participation_cards import parse_participation_cards
from data_platform.settlement import grade_selection, parse_fixture_result
from data_platform.settlement_policy import POLICY_NOTE, POLICY_VERSION, new_publication_policy, product_policy
from data_platform.services.settlement import FixtureResultLoader, SettlementService
from data_platform.services.measurement_runtime import BoundedFootballClient, BudgetExceeded, MeasurementRuntime
from data_platform.tracking_metrics import record_summary_text
from .test_safe_settlement import store, publish, payload, pred, fixture, team_stats, NOW, no_network
from .test_measurement_cycle import Client, worker


def players():
    return [{"team": {"id": tid}, "players": [
        {"player": {"id": tid * 100 + i}, "statistics": [
            {"games": {"minutes": 90, "substitute": False}, "cards": {"yellow": 0, "red": 0}}]}
        for i in range(11)]} for tid in (10, 20)]


def card_player(rows, team=0, player=0, *, minutes=90, yellow=0, red=0):
    rows[team]["players"][player]["statistics"][0].update(
        games={"minutes": minutes}, cards={"yellow": yellow, "red": red})


def selection(**kwargs):
    return {**pred("cards", "totals_cards_over_under", **kwargs),
            "settlement_policy": new_publication_policy()}


def card_result(rows=None, status="FT"):
    result = parse_fixture_result(fixture(status))
    result["participation_cards"] = parse_participation_cards(result, rows if rows is not None else players())
    return result


@pytest.mark.parametrize("yellow,red,weighted", [(0, 0, 0), (1, 0, 1), (0, 1, 2), (1, 1, 3), (2, 1, 3)])
def test_user_card_weights(yellow, red, weighted):
    rows = players()
    card_player(rows, yellow=yellow, red=red)
    result = card_result(rows)
    grade = grade_selection(selection(line=weighted + .5, side="under"), result)
    assert (grade.outcome, grade.actual_result) == ("hit", weighted)
    assert result["participation_cards"]["players"][0]["weighted_cards"] == weighted


def test_zero_minutes_excluded_but_participation_does_not_require_starter_status():
    rows = players()
    card_player(rows, minutes=0, yellow=1, red=1)
    card_player(rows, player=1, minutes=1, yellow=1)
    rows[0]["players"][1]["statistics"][0]["games"]["substitute"] = True
    assert grade_selection(selection(), card_result(rows)).actual_result == 1


@pytest.mark.parametrize("minutes", [None, -1, True, "bad", float("nan"), float("inf"), .5])
def test_missing_or_invalid_carded_minutes_remain_pending(minutes):
    rows = players()
    card_player(rows, minutes=minutes, yellow=1)
    grade = grade_selection(selection(), card_result(rows))
    assert grade.outcome is None
    assert grade.pending_reason == "missing_carded_player_minutes"


def test_missing_minutes_are_not_zero_but_explicit_zero_cards_cannot_change_total():
    rows = players()
    card_player(rows, minutes=None)
    grade = grade_selection(selection(side="under"), card_result(rows))
    assert (grade.outcome, grade.actual_result) == ("hit", 0)
    card_player(rows, minutes=None, red=None)
    assert grade_selection(selection(), card_result(rows)).pending_reason == "missing_player_card_counts"


@pytest.mark.parametrize("yellow,red", [(None, 0), (0, None), (True, 0), (0, float("nan")),
                                          (-1, 0), (3, 0), (0, 2), (2, 0), (1.5, 0)])
def test_missing_or_contradictory_cards_never_guess(yellow, red):
    rows = players()
    card_player(rows, yellow=yellow, red=red)
    assert grade_selection(selection(), card_result(rows)).outcome is None


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop(),
    lambda r: r[0]["team"].update(id=999),
    lambda r: r[0].update(players=[]),
    lambda r: r[0]["players"].pop(),
    lambda r: r[0]["players"].append(deepcopy(r[0]["players"][0])),
    lambda r: r[0]["players"][0]["player"].update(id=None),
    lambda r: r[0]["players"][0].update(statistics=[]),
])
def test_partial_or_conflicting_player_payloads_are_not_zero_totals(mutate):
    rows = players()
    mutate(rows)
    result = card_result(rows)
    assert result["participation_cards"]["totals"] == {}
    assert grade_selection(selection(), result).outcome is None


def test_placeholder_roster_without_participation_is_not_a_zero_card_match():
    rows = players()
    for i in range(11):
        card_player(rows, player=i, minutes=None)
    assert grade_selection(selection(), card_result(rows)).pending_reason == "incomplete_player_participation"


def test_stored_policy_is_detached_and_card_failure_does_not_break_other_stats(store):
    client = PlayerClient()
    client.players[0]["team"] = "malformed"
    publish(store, payload("sot", "shots_on_target_over_under", line=5.5))
    publish(store, payload("cards", "totals_cards_over_under"))
    report = SettlementService(store[0], result_loader=FixtureResultLoader(client)).resolve(now=NOW)
    assert report["graded"] == 1
    assert report["pending_reasons"] == {"malformed_player_statistics": 1}
    changed = new_publication_policy()
    changed["void_statuses"].append("PST")
    assert new_publication_policy()["void_statuses"] == ["ABD"]


def test_abandoned_card_fixture_worker_voids_without_downloading_players(store):
    client = PlayerClient("ABD")
    cycle = worker(store, client=client)
    cycle.runtime.activate(now=NOW)
    publish(store, payload("cards", "totals_cards_over_under"))
    assert cycle.run_once(kinds=("settlement",))["details"][0]["result"]["void"] == 1
    assert client.calls == [("fixture", 123)]


def test_yellow_only_market_needs_no_player_or_team_download_before_definition_review(store):
    client = PlayerClient()
    cycle = worker(store, client=client)
    cycle.runtime.activate(now=NOW)
    publish(store, payload("cards", "totals_yellow_cards"))
    result = cycle.run_once(kinds=("settlement",))
    assert result["held_reasons"] == {"card_definition_unverified": 1}
    assert client.calls == [("fixture", 123)]
    cycle.clock = lambda: NOW + timedelta(hours=8)
    assert cycle.run_once(kinds=("settlement",))["details"] == []
    assert client.calls == [("fixture", 123)]


def test_explicit_reviewed_bookmaker_override_keeps_its_original_data_adapter(store):
    from .test_safe_settlement import policy
    publish(store, payload("cards", "totals_cards_over_under", line=3.5))
    client = PlayerClient()
    report = SettlementService(store[0], result_loader=FixtureResultLoader(client), rules={
        ("Book", "totals_cards_over_under", "regulation_time"): policy(),
    }).resolve(now=NOW)
    assert report["hit"] == 1
    assert client.calls == [("fixture", 123), ("stats", 123)]
    assert store[0].get_track_record(publication_scope="all")["product_rule_settlements"] == 0


@pytest.mark.parametrize("status", ["AET", "PEN", "ET"])
def test_extra_time_not_graded_from_full_match_player_statistics(status):
    assert grade_selection(selection(), card_result(status=status)).outcome is None


def test_product_policy_voids_only_abandonment_not_other_exceptional_statuses():
    for market, key in [("cards", "totals_cards_over_under"), ("goals", "totals")]:
        pick = {**pred(market, key), "settlement_policy": new_publication_policy()}
        assert grade_selection(pick, parse_fixture_result(fixture("ABD"))).outcome == "void"
        for status in ("PST", "CANC", "AWD", "WO", "SUSP"):
            assert grade_selection(pick, parse_fixture_result(fixture(status))).outcome is None
        pick.pop("settlement_policy")
        assert grade_selection(pick, parse_fixture_result(fixture("ABD"))).outcome is None


def test_unknown_policy_and_missing_period_cannot_be_approved_by_default():
    pick = selection()
    pick["settlement_policy"]["version"] = "future-unapproved"
    assert product_policy(pick) is None
    assert grade_selection(pick, card_result()).outcome is None
    pick = selection()
    pick["tracking"]["selection"]["period"] = None
    assert grade_selection(pick, card_result()).pending_reason == "missing_or_unsupported_period"


def test_weighted_definition_cannot_grade_a_yellow_only_market():
    pick = {**pred("cards", "totals_yellow_cards"), "settlement_policy": new_publication_policy()}
    assert grade_selection(pick, parse_fixture_result(fixture(), team_stats())).pending_reason == "card_definition_unverified"


class PlayerClient(Client):
    def __init__(self, status="FT"):
        super().__init__(status)
        self.players = players()

    def fixture_players(self, fid):
        self.calls.append(("players", fid))
        return deepcopy(self.players)


def test_loader_only_fetches_players_when_requested_and_only_for_ft():
    client = PlayerClient()
    loader = FixtureResultLoader(client)
    loader("123", need_statistics=False)
    assert client.calls == [("fixture", 123)]
    loader("123", need_statistics=False, need_players=True)
    assert client.calls[-2:] == [("fixture", 123), ("players", 123)]
    client.fx = fixture("AET")
    result = loader("123", need_statistics=False, need_players=True)
    assert client.calls[-1] == ("fixture", 123)
    assert result["participation_cards"]["pending_reason"] == "regulation_player_statistics_unavailable"


def test_worker_grades_card_half_result_once_shares_fetch_and_preserves_old_rows(store):
    old = publish(store, payload("cards", "totals_cards_over_under", line=1.5))
    # Emulate an existing publication created before this policy was installed.
    with store[0]._factory() as session:
        row = session.get(Prediction, old)
        row.extras = {k: v for k, v in row.extras.items() if k != "settlement_policy"}
    before = store[0].get_unresolved()[0]
    client = PlayerClient()
    card_player(client.players, yellow=2, red=1)
    cycle = worker(store, client=client)
    cycle.runtime.activate(now=NOW)
    new = publish(store, payload("cards", "totals_cards_over_under", line=2.75))
    publish(store, payload("cards", "totals_cards_over_under", side="under", line=3.25))
    first = cycle.run_once(kinds=("settlement",))
    assert first["details"][0]["result"]["half_hit"] == 2
    assert client.calls == [("fixture", 123), ("players", 123)]
    rows = {p["id"]: p for p in store[0].get_recent(days=3650)}
    assert rows[old]["settlement"] == before["settlement"] is None
    assert rows[old]["settlement_policy"] is None
    assert rows[new]["settlement"]["policy_version"] == POLICY_VERSION
    assert rows[new]["settlement"]["basis"] == "product_rules_estimate"
    assert cycle.run_once(kinds=("settlement",))["details"] == []


def test_publication_retry_does_not_retrofit_old_policy_even_on_another_surface(store):
    original = payload()
    pid = publish(store, original)
    with store[0]._factory() as session:
        row = session.get(Prediction, pid)
        row.extras = {k: v for k, v in row.extras.items() if k != "settlement_policy"}
    retry = store[1].publish(original, surface="discord", published_at=NOW)
    assert retry["prediction_id"] == pid and not retry["created"]
    assert store[0].get_unresolved()[0]["settlement_policy"] is None


def test_player_failure_does_not_prevent_goal_grading_and_can_retry(store):
    client = PlayerClient()
    client.players = []
    cycle = worker(store, client=client)
    cycle.runtime.activate(now=NOW)
    publish(store)
    publish(store, payload("cards", "totals_cards_over_under"))
    first = cycle.run_once(kinds=("settlement",))
    assert first["details"][0]["result"]["graded"] == 1
    assert first["held_reasons"] == {"missing_player_statistics": 1}
    client.players = players()
    cycle.clock = lambda: NOW + timedelta(minutes=6)
    assert cycle.run_once(kinds=("settlement",))["details"][0]["result"]["graded"] == 1


def test_player_transport_failure_leaves_card_pending_but_grades_goals(store):
    client = PlayerClient()
    def fail(fid):
        raise RuntimeError("quota unavailable")
    client.fixture_players = fail
    publish(store)
    publish(store, payload("cards", "totals_cards_over_under"))
    result = SettlementService(store[0], result_loader=FixtureResultLoader(client)).resolve(now=NOW)
    assert result["graded"] == 1
    assert result["pending_reasons"] == {"player_statistics_fetch_failed": 1}


def test_player_requests_obey_same_daily_and_cycle_budget(store, settings):
    class Session:
        headers = {}
        def get(self, *args, **kwargs):
            return type("Response", (), {"status_code": 200, "json": lambda self: {"response": players()}})()
    runtime = MeasurementRuntime(store[0]._factory)
    client = BoundedFootballClient(runtime, cycle_limit=1, daily_limit=1, guard=lambda: None,
        clock=lambda: NOW, session=Session(), settings=settings, request_pause_s=0)
    assert client.fixture_players(123) == players()
    with pytest.raises(BudgetExceeded):
        client.fixture_players(123)
    assert runtime.state("measurement:requests:2026-09-24")["attempts"] == 1
    with pytest.raises(ValueError, match="scope"):
        client._get("/fixtures/events", {"fixture": 123})


def test_policy_is_visible_in_api_and_discord_reports_without_raw_player_data(store, monkeypatch):
    publish(store, payload("cards", "totals_cards_over_under", side="under"))
    SettlementService(store[0], result_loader=FixtureResultLoader(PlayerClient())).resolve(now=NOW)
    from Scripts.web_app import track_record_api as api
    monkeypatch.setattr(api, "get_track_record", lambda **kw: store[0].get_track_record(published_only=True, publication_scope="all"))
    monkeypatch.setattr(api, "get_recent_predictions", lambda **kw: store[0].get_recent(days=3650, published_only=True, publication_scope="all"))
    http = TestClient(api.create_standalone_app())
    summary = http.get("/api/track-record").json()
    assert summary["participation_card_settlements"] == summary["product_rule_settlements"] == 1
    assert summary["settlement_policy_note"] == POLICY_NOTE
    assert POLICY_NOTE in record_summary_text(summary)
    recent = http.get("/api/track-record/recent").json()[0]
    assert recent["card_definition_status"] == "product_participation_estimate"
    assert recent["settlement_policy"]["version"] == POLICY_VERSION
    assert recent["settlement"]["basis"] == "product_rules_estimate"
    assert "players" not in recent["settlement"] and "result" not in recent["settlement"]


def test_concurrent_policy_change_prevents_stale_settlement(store):
    pid = publish(store)
    stale = store[0].settlement_candidates()[0]
    with store[0]._factory() as session:
        row = session.get(Prediction, pid)
        row.extras = {**row.extras, "settlement_policy": None}
    assert store[0].record_settlement(stale, {"outcome": "hit"}, checked_at=NOW) == "conflict"
