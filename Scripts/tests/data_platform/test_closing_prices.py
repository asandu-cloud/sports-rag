"""Closing prices from API-Football's final pre-kickoff odds snapshot, fetched after the match."""
from datetime import timedelta
import gzip
import json

import pytest
from sqlalchemy import func, select

from data_platform.models import OddsSnapshot
from data_platform.services import closing_prices as closing
from data_platform.services.closing_prices import ClosingPriceService, SavedOddsClient
from data_platform.services.measurement_runtime import BudgetExceeded
from .test_measurement_cycle import Client, KICKOFF, worker
from .test_safe_settlement import store, publish, payload, NOW, no_network  # noqa: F401  (fixtures)


def book(name, *bets):
    return {"id": hash(name) % 1000, "name": name, "bets": [{"id": bid, "values": [
        {"value": value, "odd": str(odd)} for value, odd in values]} for bid, values in bets]}


GOALS = lambda over, under: (5, [("Over 2.5", over), ("Under 2.5", under)])


def client_with(*books, status="FT", minutes_before=5):
    client = Client(status)
    client.odds["bookmakers"] = list(books)
    client.odds["update"] = (KICKOFF - timedelta(minutes=minutes_before)).isoformat()
    return client


def service(store, client, now=NOW, guard=lambda: None):
    return ClosingPriceService(store[0], client=client, guard=guard, clock=lambda: now)


def only(store):
    return store[0].settlement_candidates()[0]


def odds_calls(client):
    return [c for c in client.calls if c[0] == "/odds"]


def test_final_snapshot_becomes_the_closing_price_with_clv_and_consensus(store):
    publish(store)  # goals over 2.5 at 2.0 with "Book"
    client = client_with(book("Book", GOALS(1.8, 2.0)), book("Other", GOALS(1.9, 1.9)))
    result = service(store, client).process(store[0].settlement_candidates())
    assert result["captured"] == 1 and result["finished"]
    row = only(store)
    assert row["closing_odds"] == 1.8 and row["clv"] == pytest.approx(2 / 1.8 - 1, abs=1e-6)
    capture = row["closing_capture"]
    assert capture["status"] == "final" and capture["source"] == closing.SOURCE
    assert capture["minutes_before_kickoff"] == pytest.approx(5)
    book_fair = (1 / 1.8) / (1 / 1.8 + 1 / 2.0)
    assert capture["consensus"]["fair_probability"] == pytest.approx((book_fair + 0.5) / 2)
    assert capture["consensus"]["book_count"] == 2
    with store[0]._factory() as s:
        snap = s.scalar(select(OddsSnapshot))
        assert snap.kind == "final_prematch" and snap.id == capture["snapshot_id"]


def test_nothing_is_fetched_until_two_hours_after_kickoff(store):
    publish(store)
    client = client_with(book("Book", GOALS(1.8, 2.0)), status="2H")
    result = service(store, client, now=KICKOFF + timedelta(hours=1)).process(store[0].settlement_candidates())
    assert result["captured"] == 0 and not result.get("finished")
    assert odds_calls(client) == []


def test_a_snapshot_stamped_after_kickoff_is_never_a_closing_price(store):
    publish(store)
    client = client_with(book("Book", GOALS(1.8, 2.0)), minutes_before=-1)
    result = service(store, client).process(store[0].settlement_candidates())
    assert (result["captured"], result["pending_reason"], result["finished"]) == (0, "snapshot_after_kickoff", True)
    assert only(store)["closing_odds"] is None


def test_an_unplayed_fixture_needs_no_closing_price(store):
    publish(store)
    client = client_with(book("Book", GOALS(1.8, 2.0)), status="PST")
    result = service(store, client).process(store[0].settlement_candidates())
    assert (result["pending_reason"], result["finished"]) == ("fixture_not_played", True)
    assert odds_calls(client) == []


def test_handicap_pick_stored_as_a_team_name_is_matched(store):
    p = payload("spreads", "spreads", side="A", line=-0.5)
    publish(store, p)
    client = client_with(book("Book", (4, [("Home -0.5", 1.9), ("Away -0.5", 1.95)])))
    assert service(store, client).process(store[0].settlement_candidates())["captured"] == 1
    capture = only(store)["closing_capture"]
    assert only(store)["closing_odds"] == 1.9
    assert capture["consensus"]["fair_probability"] == pytest.approx((1 / 1.9) / (1 / 1.9 + 1 / 1.95))


def test_without_the_same_bookmaker_the_consensus_is_still_kept(store):
    publish(store)
    client = client_with(book("Other", GOALS(1.9, 1.9)))
    result = service(store, client).process(store[0].settlement_candidates())
    row = only(store)
    assert row["closing_odds"] is None and row["clv"] is None
    assert row["closing_capture"]["status"] == "final"
    assert row["closing_capture"]["consensus"]["fair_probability"] == pytest.approx(0.5)
    assert result["pending_reasons"] == {"same_bookmaker_unavailable": 1}


@pytest.mark.parametrize("mutate,reason", [
    (lambda c: c.odds["league"].update(id=140), "odds_competition_mismatch"),
    (lambda c: c.odds["fixture"].update(id=999), "fixture_identity_mismatch"),
    (lambda c: c.odds["fixture"].update(date=(KICKOFF + timedelta(days=1)).isoformat()), "odds_kickoff_mismatch"),
])
def test_wrong_or_rescheduled_odds_rows_are_rejected(store, mutate, reason):
    publish(store)
    client = client_with(book("Book", GOALS(1.8, 2.0)))
    mutate(client)
    assert service(store, client).process(store[0].settlement_candidates())["pending_reason"] == reason
    assert only(store)["closing_capture"] is None


def test_missing_odds_are_retried_until_they_have_expired(store):
    publish(store)
    client = client_with()
    client.odds = None
    original = client._get
    client._get = lambda path, params: (original(path, params), {"response": []})[1]
    result = service(store, client).process(store[0].settlement_candidates())
    assert (result["pending_reason"], result["finished"]) == ("odds_unavailable", False)
    later = service(store, client, now=KICKOFF + timedelta(days=8)).process(store[0].settlement_candidates())
    assert later["finished"]


def test_expired_lease_writes_nothing(store):
    publish(store)
    count = [0]

    def guard():
        count[0] += 1
        if count[0] >= 3:
            raise BudgetExceeded("expired")
    with pytest.raises(BudgetExceeded):
        service(store, client_with(book("Book", GOALS(1.8, 2.0))), guard=guard).process(store[0].settlement_candidates())
    assert only(store)["closing_capture"] is None
    with store[0]._factory() as s:
        assert s.scalar(select(func.count()).select_from(OddsSnapshot)) == 0


def test_track_record_counts_clv_from_the_final_snapshot(store):
    pid = publish(store)
    service(store, client_with(book("Book", GOALS(1.8, 2.0)))).process(store[0].settlement_candidates())
    store[0].set_outcome(pid, outcome="hit")  # the track record reports CLV on graded picks
    from data_platform.tracking_metrics import build_track_record
    record = build_track_record(store[0].get_recent(days=3650))
    assert record["clv_sample_size"] == 1 and record["avg_clv"] == pytest.approx(2 / 1.8 - 1, abs=1e-6)


def test_saved_odds_are_used_before_spending_a_request(tmp_path):
    saved = {"response": [{"fixture": {"id": 123}}]}
    with gzip.open(tmp_path / "123.json.gz", "wt") as handle:
        json.dump({"fixture_id": 123, "body": saved}, handle)
    live = Client()
    cached = SavedOddsClient(live, tmp_path)
    assert cached._get("/odds", {"fixture": 123}) == saved
    assert live.calls == []
    cached._get("/odds", {"fixture": 456})
    assert live.calls == [("/odds", {"fixture": 456})]
    cached.fixture(123)
    assert live.calls[-1] == ("fixture", 123)


def test_measurement_cycle_captures_once_after_the_match(store):
    client = client_with(book("Book", GOALS(1.8, 2.0)))
    cycle = worker(store, client=client)
    cycle.runtime.activate(now=NOW)  # measurement only covers picks published after activation
    publish(store)
    report = cycle.run_once(kinds=("closing",))
    assert report["errors"] == 0 and only(store)["closing_capture"]["status"] == "final"
    cycle.run_once(kinds=("closing",))
    assert len(odds_calls(client)) == 1


def test_card_quotes_do_not_cross_match_other_card_markets(store):
    p = payload("cards", "totals_cards_over_under", side="over", line=3.5)
    publish(store, p)
    yellow_only = (153, [("Over 3.5", 1.8), ("Under 3.5", 2.0)])  # not the published cards market
    result = service(store, client_with(book("Book", yellow_only))).process(store[0].settlement_candidates())
    assert result["captured"] == 0 and only(store)["closing_capture"] is None
