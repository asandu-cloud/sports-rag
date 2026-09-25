"""Exact-fixture settlement on synthetic inputs and isolated SQLite only."""
from copy import deepcopy
from datetime import date, datetime, timezone
import sys
from pathlib import Path

import pytest

from data_platform.settlement import asian_outcome, grade_selection, parse_fixture_result
from data_platform.services.settlement import FixtureResultLoader, SettlementService
from data_platform.repositories.predictions import PredictionRepository
from data_platform.repositories.publications import PublicationRepository
from data_platform.services.publications import PublicationService

NOW = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests
    monkeypatch.setattr(requests.Session, "request", lambda *args, **kwargs: pytest.fail("Unexpected network request in settlement test"))


def payload(market="goals", key="totals", side="over", line=2.5, fid="123"):
    return {
        "schema_version": "market_result.v1",
        "fixture": {"event_id": fid, "league": "EPL", "home_team": "A", "away_team": "B", "kickoff": "2026-09-22T18:00:00Z"},
        "market": {"key": key, "group": market, "participant": None},
        "projection": {"value": 3.1},
        "decision": {"status": "recommended", "model_probability": .6,
                     "quote": {"side": side, "line": line, "odds": 2., "bookmaker": "Book", "market_key": key, "period": "regulation_time"}},
        "provenance": {"pipeline_version": "test", "input_snapshot_id": "fixture-test", "system_version": "test-v1"},
    }


def fixture(status="FT"):
    return {"fixture": {"id": 123, "date": "2026-09-22T18:00:00Z", "status": {"short": status}},
            "league": {"id": 39}, "teams": {"home": {"id": 10, "name": "Renamed A"}, "away": {"id": 20, "name": "Renamed B"}},
            "goals": {"home": 2, "away": 1}, "score": {"fulltime": {"home": 2, "away": 1}}}


def team_stats(red=0):
    return [{"team": {"id": tid, "name": "deliberately misleading name"}, "statistics": [
        {"type": "Corner Kicks", "value": 5}, {"type": "Shots on Goal", "value": 3},
        {"type": "Yellow Cards", "value": 2}, {"type": "Red Cards", "value": red},
    ]} for tid in [20, 10]]


def pred(market="goals", key="totals", side="over", line=2.5):
    from data_platform.publication_identity import tracking_identity
    p = payload(market, key, side, line)
    return {"fixture_id": "123", "market": market, "side": side, "line": line,
            "bookmaker": "Book", "pick": f"{side} {line}", "tracking": tracking_identity(p)}


def policy(key="totals_cards_over_under", **extra):
    return {"bookmaker": "Book", "market_key": key, "period": "regulation_time", "version": "test-only",
            "evidence_reference": "synthetic reviewed policy, NOT a real bookmaker rule",
            "provider_stats_compatible": True,
            "card_count": "yellow_only" if key == "totals_yellow_cards" else "yellow_plus_red", **extra}


@pytest.mark.parametrize("value,line,over,outcome", [
    (3, 2.5, True, "hit"), (3, 3, True, "push"), (3, 3.5, True, "miss"),
    (3, 2.75, True, "half_hit"), (3, 3.25, True, "half_miss"),
    (3, 2.75, False, "half_miss"), (3, 3.25, False, "half_hit"),
    (-1, -1.25, True, "half_hit"), (-1, -.75, True, "half_miss"),
])
def test_asian_lines(value, line, over, outcome):
    assert asian_outcome(value, line, over=over) == outcome


@pytest.mark.parametrize("line", [None, True, float("nan"), float("inf"), "oops", 2.3])
def test_malformed_lines_are_pending(line):
    prediction = pred()
    prediction["line"] = prediction["tracking"]["selection"]["line"] = line
    assert grade_selection(prediction, parse_fixture_result(fixture())).outcome is None


@pytest.mark.parametrize("market,key,side,line,outcome,actual", [
    ("goals", "totals", "over", 2.75, "half_hit", 3),
    ("corners", "totals_corners_over_under", "under", 10.25, "half_hit", 10),
    ("sot", "shots_on_target_over_under", "over", 6.25, "half_miss", 6),
    ("btts", "btts", "yes", None, "hit", 1),
    ("btts", "btts", "no", None, "miss", 1),
    ("moneyline", "h2h", "home", None, "hit", 1),
    ("moneyline", "h2h", "draw", None, "miss", 1),
    ("spreads", "spreads", "home", -1.25, "half_miss", 1),
    ("spreads", "spreads", "home", -.75, "half_hit", 1),
    ("spreads", "spreads", "away", 1.25, "half_hit", 1),
    ("spreads", "spreads", "away", .75, "half_miss", 1),
])
def test_supported_markets(market, key, side, line, outcome, actual):
    grade = grade_selection(pred(market, key, side, line), parse_fixture_result(fixture(), team_stats()))
    assert (grade.outcome, grade.actual_result) == (outcome, actual)


@pytest.mark.parametrize("status", ["AET", "PEN"])
def test_extended_matches_use_only_fulltime_score_and_withhold_unsegmented_stats(status):
    fx = fixture(status)
    fx["goals"] = {"home": 4, "away": 3}
    fx["score"]["penalty"] = {"home": 5, "away": 4}
    result = parse_fixture_result(fx, team_stats())
    assert grade_selection(pred(line=3.5), result).outcome == "miss"
    assert grade_selection(pred("corners", "totals_corners_over_under"), result).pending_reason == "regulation_statistics_unavailable"
    fx["score"]["fulltime"]["home"] = None
    assert grade_selection(pred(), parse_fixture_result(fx)).pending_reason == "missing_regulation_score"


def test_zero_is_real_but_null_nonfinite_fractional_and_negative_are_unknown():
    for value in [None, float("nan"), float("inf"), -1, True, "bad", 1.5]:
        fx = fixture()
        fx["goals"]["home"] = fx["score"]["fulltime"]["home"] = value
        assert grade_selection(pred(), parse_fixture_result(fx)).pending_reason == "missing_regulation_score"
    fx = fixture()
    fx["goals"] = fx["score"]["fulltime"] = {"home": 0, "away": 0}
    assert grade_selection(pred(side="under"), parse_fixture_result(fx)).outcome == "hit"


def test_statistics_pairing_is_exact_and_missing_red_never_becomes_zero():
    result = parse_fixture_result(fixture(), team_stats(red=None))
    assert result["cards_home"] is None
    assert result["corners_home"] == 5
    rows = team_stats()
    rows[0]["team"]["id"] = 999
    assert parse_fixture_result(fixture(), rows)["statistics_error"] == "statistics_team_identity_mismatch"
    rows[0]["team"]["id"] = 10
    assert parse_fixture_result(fixture(), rows)["statistics_error"] == "statistics_team_identity_mismatch"


def test_card_definitions_must_be_reviewed_and_do_not_mix_yellow_with_total():
    result = parse_fixture_result(fixture(), team_stats(red=1))
    total = pred("cards", "totals_cards_over_under", line=4.5)
    yellow = pred("cards", "totals_yellow_cards", line=4.5)
    assert grade_selection(total, result).pending_reason == "card_definition_unverified"
    assert grade_selection(total, result, rules=policy()).outcome == "hit"
    assert grade_selection(yellow, result, rules=policy("totals_yellow_cards")).outcome == "miss"
    assert grade_selection(yellow, result, rules=policy()).outcome is None
    missing_red = parse_fixture_result(fixture(), team_stats(red=None))
    assert grade_selection(total, missing_red, rules=policy()).pending_reason == "missing_market_statistics"
    assert grade_selection(yellow, missing_red, rules=policy("totals_yellow_cards")).outcome == "miss"


@pytest.mark.parametrize("status", ["NS", "1H", "HT", "2H", "ET", "P", "CANC", "PST", "ABD", "AWD", "WO"])
def test_unfinished_or_exceptional_status_does_not_invent_loss_or_void(status):
    assert grade_selection(pred(), parse_fixture_result(fixture(status))).outcome is None


def test_void_is_distinct_from_push_and_requires_exact_reviewed_policy():
    result = parse_fixture_result(fixture("CANC"))
    rule = policy("totals", void_statuses=["CANC"])
    assert grade_selection(pred(), result, rules=rule).outcome == "void"
    assert grade_selection(pred(), result, rules={**rule, "bookmaker": "Other"}).outcome is None


@pytest.mark.parametrize("field,value", [("period", None), ("period", "first_half"), ("market_key", "totals_yellow_cards"), ("fixture_id", "999"), ("participant", "A"), ("side", "under")])
def test_incomplete_or_conflicting_selection_identity_is_pending(field, value):
    prediction = pred()
    prediction["tracking"]["selection"][field] = value
    assert grade_selection(prediction, parse_fixture_result(fixture())).outcome is None


@pytest.fixture()
def store(settings, engine, session_factory):
    return (PredictionRepository(session_factory=session_factory),
            PublicationService(PublicationRepository(session_factory=session_factory)))


def publish(store, p=None):
    return store[1].publish(p or payload(), surface="website", published_at="2026-09-18T10:00:00Z")["prediction_id"]


def test_publication_before_matchday_grades_by_id_once_and_keeps_original_decision(store):
    pid = publish(store)
    amended = payload(line=3.25)
    second = publish(store, amended)
    calls = []
    def loader(fid, **kwargs):
        calls.append((fid, kwargs))
        return parse_fixture_result(fixture())
    service = SettlementService(store[0], result_loader=loader)
    result = service.resolve(on_or_before=date(2026, 9, 23), now=NOW)
    assert result["graded"] == 2
    assert (result["hit"], result["half_miss"]) == (1, 1)
    assert calls == [("123", {"need_statistics": False})]
    assert service.resolve(now=NOW)["graded"] == 0
    rows = store[0].get_recent(days=3650)
    assert {row["id"] for row in rows} == {pid, second}
    assert {row["prediction_date"] for row in rows} == {"2026-09-18"}
    assert {row["odds"] for row in rows} == {2.}
    assert all(row["settlement"]["result"]["fixture_id"] == "123" for row in rows)


def test_dry_run_reads_same_backend_and_writes_nothing(store):
    publish(store)
    before = store[0].get_recent(days=3650)
    result = SettlementService(store[0], result_loader=lambda *_, **kw: parse_fixture_result(fixture())).resolve(now=NOW, dry_run=True)
    assert result["graded"] == 0 and result["would_grade"] == 1
    assert store[0].get_recent(days=3650) == before


def test_pending_is_persisted_with_reason_and_retry_can_later_settle(store):
    publish(store, payload("sot", "shots_on_target_over_under", line=5.5))
    service = SettlementService(store[0], result_loader=lambda *_, **kw: parse_fixture_result(fixture()))
    report = service.resolve(now=NOW)
    assert report["pending_reasons"] == {"missing_statistics": 1}
    row = store[0].get_unresolved()[0]
    assert row["outcome"] is None and row["settlement"]["attempts"] == 1
    service.result_loader = lambda *_, **kw: parse_fixture_result(fixture(), team_stats())
    assert service.resolve(now=NOW)["hit"] == 1
    assert store[0].get_recent(days=3650)[0]["settlement"]["attempts"] == 2


def test_old_missing_period_remains_unknown_even_when_fixture_is_final(store):
    p = payload()
    p["decision"]["quote"].pop("period")
    publish(store, p)
    result = SettlementService(store[0], result_loader=lambda *_, **kw: parse_fixture_result(fixture())).resolve(now=NOW)
    assert result["pending_reasons"] == {"missing_or_unsupported_period": 1}


def test_targeted_correction_keeps_audit_history_and_stale_writer_cannot_overwrite(store):
    pid = publish(store)
    service = SettlementService(store[0], result_loader=lambda *_, **kw: parse_fixture_result(fixture()))
    stale = store[0].settlement_candidates()[0]
    assert service.resolve(now=NOW)["hit"] == 1
    with pytest.raises(ValueError, match="explicit prediction_ids"):
        service.resolve(now=NOW, regrade=True)
    corrected = fixture()
    corrected["goals"] = corrected["score"]["fulltime"] = {"home": 1, "away": 1}
    service.result_loader = lambda *_, **kw: parse_fixture_result(corrected)
    assert service.resolve(now=NOW, regrade=True, prediction_ids=[pid])["miss"] == 1
    row = store[0].get_recent(days=3650)[0]
    assert row["outcome"] == "miss"
    assert row["settlement"]["history"][0]["outcome"] == "hit"
    assert service.resolve(now=NOW, regrade=True, prediction_ids=[pid])["graded"] == 0
    assert store[0].record_settlement(stale, {"outcome": "hit"}, checked_at=NOW) == "conflict"


def test_one_fixture_failure_does_not_block_the_rest(store):
    publish(store)
    publish(store, payload(fid="456"))
    def loader(fid, **kw):
        if fid == "123":
            raise RuntimeError("unavailable")
        fx = fixture()
        fx["fixture"]["id"] = 456
        return parse_fixture_result(fx)
    report = SettlementService(store[0], result_loader=loader).resolve(now=NOW)
    assert report["graded"] == 1 and report["pending"] == 1 and report["errors"] == 1


def test_future_fixture_not_fetched_and_rescheduled_provider_result_not_graded(store):
    future = payload()
    future["fixture"]["kickoff"] = "2026-10-01T18:00:00Z"
    publish(store, future)
    service = SettlementService(store[0], result_loader=lambda *_, **kw: pytest.fail("future API request"))
    assert service.resolve(now=NOW)["skipped"] == 1
    publish(store, payload(fid="456"))
    rescheduled = fixture()
    rescheduled["fixture"].update(id=456, date="2026-10-01T18:00:00Z")
    service.result_loader = lambda *_, **kw: parse_fixture_result(rescheduled)
    assert service.resolve(now=NOW)["pending_reasons"] == {"fixture_not_due": 1}


def test_result_loader_requests_by_id_and_does_not_download_players():
    calls = []
    class Client:
        def fixture(self, fid):
            calls.append(("fixture", fid))
            return [fixture()]
        def fixture_statistics(self, fid):
            calls.append(("statistics", fid))
            raise RuntimeError("quota")
    result = FixtureResultLoader(Client())("123", need_statistics=True)
    assert calls == [("fixture", 123), ("statistics", 123)]
    assert result["statistics_error"] == "statistics_fetch_failed"
    assert grade_selection(pred(), result).outcome == "hit"


def test_canonical_failure_never_falls_back_to_legacy_even_for_dry_run(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag_ingest"))
    import prediction_tracker as tracker
    import data_platform.compat as compat
    monkeypatch.setattr(tracker, "_platform_on", lambda: True)
    def fail():
        raise RuntimeError("canonical unavailable")
    monkeypatch.setattr(compat, "get_prediction_service", fail)
    monkeypatch.setattr(tracker, "_get_db", lambda *_: pytest.fail("legacy database opened"))
    report = tracker.resolve_outcomes("2026-09-23", dry_run=True)
    assert report["error"] == "canonical_settlement_failed"


def test_api_client_exact_fixture_endpoint(monkeypatch):
    from data_platform.sync.apifootball import ApiFootballClient
    client = ApiFootballClient(api_key="test", request_pause_s=0)
    calls = []
    monkeypatch.setattr(client, "_get", lambda path, params: calls.append((path, params)) or {"response": [fixture()]})
    assert client.fixture(123) == [fixture()]
    assert calls == [("/fixtures", {"id": 123})]


def test_correct_score_and_invalid_side_are_not_implicitly_losses():
    prediction = pred("correct_score", "correctscore", "2-1", None)
    prediction["pick"] = "2-1"
    assert grade_selection(prediction, parse_fixture_result(fixture())).outcome == "hit"
    prediction["pick"] = "not a score"
    assert grade_selection(prediction, parse_fixture_result(fixture())).pending_reason == "invalid_score_selection"
    assert grade_selection(pred("moneyline", "h2h", "oops", None), parse_fixture_result(fixture())).pending_reason == "invalid_side"
    fx = fixture()
    fx["goals"]["home"] = 5
    assert grade_selection(pred(), parse_fixture_result(fx)).pending_reason == "conflicting_fulltime_score"


def test_invalid_fixture_ids_pending_without_network_or_name_fallback(store):
    publish(store, payload(fid="not-an-id"))
    publish(store, payload(fid="999999999999999999999999999999999999999"))
    report = SettlementService(store[0], result_loader=lambda *_, **kw: pytest.fail("invalid fixture API request")).resolve(now=NOW)
    assert report["pending_reasons"] == {"missing_fixture_id": 2}


def test_changed_competition_or_timezone_boundary_cannot_grade_wrong_day(store):
    publish(store)
    result = parse_fixture_result(fixture())
    result["league_id"] = 140
    service = SettlementService(store[0], result_loader=lambda *_, **kw: result)
    assert service.resolve(now=NOW)["pending_reasons"] == {"competition_identity_mismatch": 1}
    result["league_id"] = 39
    # A local Sep 23 kickoff can be Sep 22 UTC; publication day is irrelevant.
    result["kickoff"] = "2026-09-23T01:00:00+03:00"
    assert service.resolve(on_or_before="2026-09-22", now=NOW)["hit"] == 1


def test_malformed_tracking_does_not_abort_other_selections(store, session_factory):
    from data_platform.models import Prediction
    first = publish(store)
    publish(store, payload(line=3.25))
    with session_factory() as session:
        row = session.get(Prediction, first)
        row.extras = {**row.extras, "tracking": ["bad historical payload"]}
    result = SettlementService(store[0], result_loader=lambda *_, **kw: parse_fixture_result(fixture())).resolve(now=NOW)
    assert result["graded"] == 1
    assert result["pending_reasons"] == {"malformed_settlement_input": 1}


def test_concurrent_retries_do_not_count_two_grades(store):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    publish(store)
    barrier = Barrier(2)
    def loader(*_, **kw):
        barrier.wait(timeout=5)
        return parse_fixture_result(fixture())
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(lambda _: SettlementService(store[0], result_loader=loader).resolve(now=NOW), range(2)))
    assert sum(report["graded"] for report in reports) == 1
    row = store[0].get_recent(days=3650)[0]
    assert row["outcome"] == "hit"
    assert row["settlement"]["history"] == []


def test_corrected_missing_evidence_preserves_grade_for_explicit_review(store):
    pid = publish(store)
    service = SettlementService(store[0], result_loader=lambda *_, **kw: parse_fixture_result(fixture()))
    service.resolve(now=NOW)
    service.result_loader = lambda *_, **kw: parse_fixture_result(fixture("PST"))
    report = service.resolve(now=NOW, regrade=True, prediction_ids=[pid])
    assert report["pending"] == 1
    row = store[0].get_recent(days=3650)[0]
    assert row["outcome"] == "hit"
    assert row["settlement"]["status"] == "review_pending"
    assert row["settlement"]["history"][0]["evidence"]["result"]["goals_home"] == 2


def test_saved_canonical_schedule_supersedes_old_publication_kickoff(store, session_factory):
    from data_platform.models import Competition, Season, Team, Fixture
    publish(store)
    with session_factory() as session:
        comp = Competition(code="EPL", name="Premier League", api_football_id=39, competition_type="domestic_league")
        session.add(comp)
        session.flush()
        season = Season(competition_id=comp.id, year=2026, label="2026/27")
        home = Team(api_football_id=10, name="A")
        away = Team(api_football_id=20, name="B")
        session.add_all([season, home, away])
        session.flush()
        session.add(Fixture(api_football_id=123, competition_id=comp.id, season_id=season.id,
                            home_team_id=home.id, away_team_id=away.id,
                            kickoff_utc=datetime(2026, 10, 1, tzinfo=timezone.utc)))
    report = SettlementService(store[0], result_loader=lambda *_, **kw: pytest.fail("rescheduled future fixture fetched")).resolve(now=NOW)
    assert report["skipped"] == 1
