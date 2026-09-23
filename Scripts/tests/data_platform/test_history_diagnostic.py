from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock
import hashlib
import json

import pytest

from Scripts.data_platform.features.history_diagnostic import (
    audit_snapshot, choose_sample, compare_fixture, fixture_issues, snapshot, state,
)
from Scripts.ops.prediction_history_diagnostic import EvidenceClient, RequestBudgetExceeded, analyze, prepare, verify_prepared
from Scripts.tests.data_platform.test_model_dataset import _seed, _path


def fixture(fid=1, league="EPL", season=2025):
    values = dict(goals=0, corners=2, shots_on=1, shots_total=3, yellow_cards=1,
                  red_cards=None, fouls_committed=5, expected_goals=0.5, possession=0.5, pass_accuracy=0.8)
    return dict(fixture_id=fid, internal_id=fid, league=league, league_id=39,
                competition_id=1, season_competition_id=1, season=season,
                home_team_id=1, away_team_id=2, home_id=10, away_id=20, home="H", away="A",
                kickoff="2025-08-01T12:00:00Z", status="FT", home_goals=0, away_goals=0,
                team_rows=[dict(team_id=tid, opponent_team_id=3-tid, api_team_id=tid*10, is_home=tid == 1,
                                values=deepcopy(values), stats_json=deepcopy(values)) for tid in (1, 2)])


def test_local_snapshot_is_read_only(settings, session_factory):
    _seed(session_factory)
    path = _path(settings)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    data = snapshot(path, [2026])
    assert data["database_checks"]["quick_check"] == ["ok"]
    assert data["database_checks"]["foreign_key_violations"] == []
    assert len(data["fixtures"]) == 2
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_missing_file_is_not_created(tmp_path):
    path = tmp_path / "absent.db"
    with pytest.raises(Exception):
        snapshot(path, [2025])
    assert not path.exists()


def test_audit_distinguishes_null_lossy_and_real_zero():
    f1, f2 = fixture(1), fixture(2)
    f2["team_rows"][0]["stats_json"]["_history_recovery"] = {"zero_policy": "legacy_zero_is_unknown"}
    data = dict(seasons=[2025], database_checks={}, fixtures=[f1, f2])
    cohort = audit_snapshot(data, ["EPL"])["cohorts"]["EPL:2025"]
    assert cohort["fields"]["red_cards"] == {"provider_null": 2, "legacy_ambiguous_or_missing": 2}
    assert cohort["fields"]["goals"]["numeric_zero"] == 4
    assert cohort["unknown_labels"]["cards"] == 2
    assert cohort["known_labels"]["goals"] == 2


def test_fixture_errors_and_anomalies_are_visible():
    bad = fixture()
    bad["team_rows"][0]["opponent_team_id"] = 1
    assert "invalid_home_opponent" in fixture_issues(bad)
    bad["season_competition_id"] = 9
    assert "season_competition_mismatch" in fixture_issues(bad)
    f = fixture(2)
    f["team_rows"][0]["values"]["shots_on"] = 10
    report = audit_snapshot(dict(seasons=[2025], database_checks={}, fixtures=[bad, f]), ["EPL"])
    assert len(report["quarantined"]) == 1
    assert report["anomalies"][0]["issue"] == "sot_exceeds_total_shots"


@pytest.mark.parametrize("value,field", [(True, "goals"), (-1, "corners"), (1.2, "red_cards"), (50, "possession"), ("NaN", "shots_on")])
def test_invalid_numbers(value, field):
    assert state(value, field) == "invalid"


def test_sample_is_reproducible_and_spans_populated_cohorts():
    data = {"fixtures": [fixture(fid + year * 100, league, year) for fid in range(1, 6)
                         for league in ["EPL"] for year in [2025, 2026]]}
    sample = choose_sample(data, ["EPL"], size=6)
    assert sample == choose_sample(data, ["EPL"], size=6)
    assert len({r["fixture_id"] for r in sample["fixtures"]}) == 6
    assert {r["season"] for r in sample["fixtures"]} == {2025, 2026}
    assert sum(r["selection"] == "random_control" for r in sample["fixtures"]) == 2
    with pytest.raises(ValueError, match="too small"):
        choose_sample(data, ["EPL"], size=2)


def test_comparison_preserves_raw_null_and_second_yellow_separately():
    stats = [{"team": {"id": tid}, "statistics": [{"type": "Red Cards", "value": None},
              {"type": "Ball Possession", "value": "50%"}]} for tid in (10, 20)]
    events = [{"type": "Card", "detail": "Second Yellow card", "team": {"id": 10},
               "player": {"id": 42}, "time": {"elapsed": 80}}]
    report = compare_fixture(fixture(), stats, events)
    red = [r for r in report["comparisons"] if r["field"] == "red_cards"]
    assert all(r["provider_numeric"] is None and r["comparison"] == "still_unknown" for r in red)
    assert report["card_events"]["10"] == {"Second Yellow card": 1}
    assert all(r["comparison"] == "agrees" for r in report["comparisons"] if r["field"] == "possession")
    assert "NOT prove zero" in report["event_caveat"]


def test_comparison_flags_bad_pairs_and_detects_numeric_recovery():
    assert compare_fixture(fixture(), [], [])["issue"] == "empty_provider_statistics"
    stats = [{"team": {"id": tid}, "statistics": [{"type": "Red Cards", "value": 0},
              {"type": "Corner Kicks", "value": 3}]} for tid in (10, 20)]
    report = compare_fixture(fixture(), stats, [])
    assert sum(r["comparison"] == "recoverable_numeric_zero" for r in report["comparisons"]) == 2
    assert sum(r["comparison"] == "numeric_disagreement" for r in report["comparisons"]) == 2


def test_empty_or_mismatched_statistics_retain_events_and_provenance():
    events = [{"type": "Card", "detail": "Yellow Card", "team": {"id": 10},
               "player": {"id": None, "name": "Unknown role"}, "time": {"elapsed": 40}}]
    for stats, issue in [([], "empty_provider_statistics"),
                         ([{"team": {"id": 10}}, {"team": {"id": 30}}], "conflicting_provider_team_pair")]:
        result = compare_fixture(fixture(), stats, events)
        assert result["issue"] == issue
        assert not result["valid_statistics_pair"]
        assert result["comparisons"] == []
        assert result["league"] == "EPL" and result["source"] == "canonical_provider"
        assert result["event_count"] == 1
        assert result["card_events"]["10"] == {"Yellow Card": 1}
        assert len(result["ambiguous_card_events"]) == 1


def response(payload=None, status=200):
    return Mock(status_code=status, content=json.dumps(payload or {"response": [], "errors": [], "paging": {"total": 1}}).encode(), headers={})


def test_budget_survives_failure_and_restart_and_never_retries(tmp_path):
    session = Mock()
    session.get.side_effect = OSError("network failed")
    request = {"endpoint": "/fixtures/statistics", "params": {"fixture": 1}}
    client = EvidenceClient(tmp_path / "provider", session, limit=2, pause=0)
    with pytest.raises(OSError):
        client.fetch(request)
    assert client.attempts == 1
    session.get.assert_called_once()
    session.get.side_effect = None
    session.get.return_value = response()
    resumed = EvidenceClient(tmp_path / "provider", session, limit=2, pause=0)
    resumed.fetch(request)
    assert len(resumed.successes()) == 1
    with pytest.raises(RequestBudgetExceeded):
        resumed.fetch(request)
    assert session.get.call_count == 2
    assert session.get.call_args.kwargs["allow_redirects"] is False
    assert "headers" not in (tmp_path / "provider/002.request.json").read_text()


def test_http200_errors_not_success_and_tampering_rejected(tmp_path):
    session = Mock()
    session.get.return_value = response({"response": [], "errors": {"rateLimit": "quota exhausted"}})
    client = EvidenceClient(tmp_path / "provider", session, pause=0)
    request = {"endpoint": "/fixtures/events", "params": {"fixture": 1}}
    with pytest.raises(RuntimeError, match="error/invalid"):
        client.fetch(request)
    assert client.successes() == {}
    session.get.return_value = response()
    client.fetch(request)
    (tmp_path / "provider/002.body").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        client.successes()


def test_preparation_respects_budget_and_refuses_overwrite(settings, session_factory, tmp_path):
    _seed(session_factory)
    root = tmp_path / "project"
    with pytest.raises(ValueError, match="cap"):
        prepare(root, _path(settings), "bad", [2026], sample_size=60)
    target = prepare(root, _path(settings), "good", [2026], sample_size=2)
    plan = verify_prepared(target)
    assert len(plan["requests"]) == 17
    with pytest.raises(FileExistsError):
        prepare(root, _path(settings), "good", [2026], sample_size=2)


def test_offline_analysis_does_not_count_empty_responses_as_compared(settings, session_factory, tmp_path):
    _seed(session_factory)
    target = prepare(tmp_path / "project", _path(settings), "analysis", [2026], sample_size=2)
    session = Mock()
    session.get.return_value = response()
    client = EvidenceClient(target / "provider", session, pause=0)
    for request in verify_prepared(target)["requests"]:
        client.fetch(request)
    report = analyze(target)
    assert report["processed_fixtures"] == 2
    assert report["compared_fixtures"] == report["compared_team_sides"] == 0
    assert report["valid_statistics_pairs"] == 0
    assert report["unavailable_statistics"] == {"empty_provider_statistics": 2}
    assert report["pending_requests"] == []
    first = (target / "analysis-001.json").read_bytes()
    analyze(target)
    assert (target / "analysis-002.json").exists()
    assert (target / "analysis-001.json").read_bytes() == first
    assert session.get.call_count == 17
