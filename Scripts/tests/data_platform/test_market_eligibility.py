from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import socket
import sqlite3

import pytest

from Scripts.data_platform.features.history_evidence import LocalEvidence, load_eligible_inputs
from Scripts.data_platform.features.market_eligibility import count, market_decisions, round_scope
from Scripts.tests.data_platform.test_model_dataset import _seed, _path


AS_OF = datetime(2026, 9, 26, tzinfo=timezone.utc)


def raw_fixture(fid=100, *, round_name="Regular Season - 1"):
    return {"fixture": {"id": fid, "date": "2026-08-01T12:00:00+00:00", "referee": "Official",
                        "status": {"short": "FT", "elapsed": 90}},
            "league": {"id": 39, "season": 2026, "round": round_name},
            "teams": {"home": {"id": 1}, "away": {"id": 2}}, "goals": {"home": 0, "away": 0},
            "score": {"fulltime": {"home": 0, "away": 0}, "extratime": {"home": None, "away": None},
                      "penalty": {"home": None, "away": None}}}


def raw_stats(corners=0, sot=0):
    return [{"team": {"id": team}, "statistics": [
        {"type": "Corner Kicks", "value": corners}, {"type": "Shots on Goal", "value": sot}
    ]} for team in (1, 2)]


def archive(root, db, endpoint, payload, *, params, archive_id, fetched="2026-08-01T17:00:00+00:00"):
    path = root / "Index/raw_archive" / f"{archive_id}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, sort_keys=True, default=str).encode()
    path.write_bytes(gzip.compress(body))
    digest = hashlib.sha256(body).hexdigest()
    params_digest = hashlib.sha256(json.dumps(params, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with sqlite3.connect(db) as connection:
        connection.execute("""INSERT INTO raw_payload_archive
            (id,provider,endpoint,params_digest,params,storage_backend,storage_uri,payload_digest,fetched_at,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (archive_id, "api_football", endpoint, params_digest, json.dumps(params), "local", path.as_uri(), digest, fetched, fetched, fetched))
    return path


def seeded(settings, session_factory, tmp_path):
    _seed(session_factory)
    db = _path(settings)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE fixtures SET round='Regular Season - 1'")
    archive(tmp_path, db, "/fixtures", [raw_fixture()], params={"league": 39, "season": 2026}, archive_id=1)
    archive(tmp_path, db, "/fixtures/statistics", raw_stats(), params={"fixture": 100}, archive_id=2)
    return db


@pytest.mark.parametrize("value,expected", [(0, 0), (3.0, 3), (None, None), (True, None), (-1, None), (0.5, None), (float("inf"), None)])
def test_count_validity(value, expected):
    assert count(value) == expected


@pytest.mark.parametrize("league,season,label,group,target,context", [
    ("EPL", 2025, "Regular Season - 12", "domestic_regular", True, True),
    ("BelgianProLeague", 2025, "Championship Group - 31", "domestic_integral_split", True, True),
    ("BelgianProLeague", 2024, "Relegation Round - 1", "domestic_integral_split", True, True),
    ("BelgianProLeague", 2024, "Relegation Round", "domestic_separate_playoff", False, True),
    ("Championship", 2025, "Promotion Play-offs - Final", "domestic_separate_playoff", False, True),
    ("UCL", 2023, "Group A - 1", "european_group", True, True),
    ("UCL", 2024, "League Stage - 1", "european_league", True, True),
    ("UECL", 2026, "Playoff round", "european_qualifying", False, True),
    ("UEL", 2022, "Knockout Round Play-offs", "european_knockout", True, True),
    ("UCL", 2022, "League Stage - 1", "unknown", False, False),
    ("EPL", 2025, "Final", "unknown", False, False),
    ("UECL", 2020, "Final", "unknown", False, False),
])
def test_explicit_round_scope(league, season, label, group, target, context):
    result = round_scope(league, season, label)
    assert (result["group"], result["target"], result["context"]) == (group, target, context)


def test_cards_always_excluded_and_zero_is_known():
    labels = {side: {m: 0 for m in ("goals", "corners", "sot", "cards")} for side in ("home", "away")}
    decisions = market_decisions(labels)
    assert decisions["corners"]["eligible"]
    assert decisions["cards"] == {"eligible": False, "reasons": ["cards_target_not_qualified"]}
    labels["away"]["corners"] = None
    assert market_decisions(labels)["corners"]["reasons"] == ["missing_away_corners"]


def test_readonly_real_zero_provenance_and_unresolved_period(settings, engine, session_factory, tmp_path, monkeypatch):
    db = seeded(settings, session_factory, tmp_path)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("No network allowed"))
    targets, history, evidence, report = load_eligible_inputs(db, tmp_path, AS_OF)
    assert len(targets) == 2
    assert [row["fixture_id"] for row in history] == [100]
    e = evidence[100]
    assert e["labels"]["corners"] == 0
    assert e["market_eligibility"]["corners"]["eligible"]
    assert e["recorded_team_labels"]["home"]["cards"] == 0
    assert not e["market_eligibility"]["cards"]["eligible"]
    assert e["label_available_at"] == "2026-08-01T15:00:00+00:00"
    assert e["actual_observed_at"] == "2026-08-01T17:00:00+00:00"
    assert e["statistics_source"]["archive_id"] == 2
    assert evidence[101]["common_reasons"] == ["period_evidence_unavailable"]
    assert evidence[101]["labels"]["goals"] is None
    assert evidence[101]["recorded_team_labels"]["home"]["goals"] == 0
    assert e["statistic_presence"]["home"]["corners"]["state"] == "explicit_zero"
    assert e["statistic_presence"]["home"]["xg"]["state"] == "omitted"
    assert len(report["source_files"]) == 2
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    assert load_eligible_inputs(db, tmp_path, AS_OF) == (targets, history, evidence, report)


@pytest.mark.parametrize("change,reason", [
    ("extra_time", "nonnull_extratime_score"),
    ("score", "goals_fulltime_mismatch"),
    ("identity", "evidence_identity_mismatch"),
    ("kickoff", "evidence_kickoff_mismatch"),
])
def test_unsafe_result_cannot_enter_context(settings, engine, session_factory, tmp_path, change, reason):
    _seed(session_factory)
    db = _path(settings)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE fixtures SET round='Regular Season - 1'")
    raw = raw_fixture()
    if change == "extra_time": raw["score"]["extratime"]["home"] = 0
    if change == "score": raw["score"]["fulltime"]["home"] = 1
    if change == "identity": raw["teams"]["home"]["id"] = 3
    if change == "kickoff": raw["fixture"]["date"] = "2026-08-02T12:00:00+00:00"
    archive(tmp_path, db, "/fixtures", [raw], params={"league": 39, "season": 2026}, archive_id=1)
    _, history, evidence, _ = load_eligible_inputs(db, tmp_path, AS_OF)
    assert history == []
    assert reason in evidence[100]["common_reasons"]
    assert evidence[100]["labels"]["goals"] is None
    assert all(not d["eligible"] for d in evidence[100]["market_eligibility"].values())


def test_statistic_checksum_failure_masks_event_context_only(settings, engine, session_factory, tmp_path):
    db = seeded(settings, session_factory, tmp_path)
    path = tmp_path / "Index/raw_archive/2.json.gz"
    path.write_bytes(gzip.compress(b"[]"))
    _, history, evidence, _ = load_eligible_inputs(db, tmp_path, AS_OF)
    assert history[0]["home"]["corners"] is None
    assert evidence[100]["market_eligibility"]["goals"]["eligible"]
    assert not evidence[100]["market_eligibility"]["corners"]["eligible"]
    assert evidence[100]["statistics_source_reason"] == "archive_payload_checksum_mismatch"


def test_native_provider_revision_does_not_replace_canonical_value(settings, engine, session_factory, tmp_path):
    db = seeded(settings, session_factory, tmp_path)
    archive(tmp_path, db, "/fixtures/statistics", raw_stats(corners=3), params={"fixture": 100}, archive_id=3,
            fetched="2026-08-02T17:00:00+00:00")
    _, history, evidence, _ = load_eligible_inputs(db, tmp_path, AS_OF)
    assert history[0]["home"]["corners"] == 0
    assert evidence[100]["statistics_source"]["archive_id"] == 2


def test_suspected_unplayed_result_excludes_history(settings, engine, session_factory, tmp_path):
    _seed(session_factory)
    db = _path(settings)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE fixtures SET round='Regular Season - 1', home_goals=3, referee=NULL WHERE api_football_id=100")
    raw = raw_fixture(); raw["goals"]["home"] = raw["score"]["fulltime"]["home"] = 3; raw["fixture"]["referee"] = None
    archive(tmp_path, db, "/fixtures", [raw], params={"league": 39, "season": 2026}, archive_id=1)
    archive(tmp_path, db, "/fixtures/statistics", [], params={"fixture": 100}, archive_id=2)
    _, history, evidence, _ = load_eligible_inputs(db, tmp_path, AS_OF)
    assert history == []
    assert "suspected_administrative_result_participation_unverified" in evidence[100]["common_reasons"]


def test_verified_legacy_plan_preserves_unknown_zero(settings, engine, session_factory, tmp_path):
    _seed(session_factory)
    db = _path(settings)
    source = {"path": "Output/example.json", "sha256": "a" * 64, "metadata_sha256": "b" * 64}
    provenance = {"source": source, "prepared_at": "2026-09-22T10:00:00Z", "zero_policy": "legacy_zero_is_unknown",
                  "source_kind": "legacy_team_export_verified_fixture"}
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE fixtures SET round='Regular Season - 1'")
        connection.execute("UPDATE fixture_team_stats SET stats_json=? WHERE fixture_id=(SELECT id FROM fixtures WHERE api_football_id=100)",
                           (json.dumps({"_history_recovery": provenance}),))
    plan = {"schema": "historical-recovery.v1", "policy": "legacy_zero_is_unknown", "prepared_at": provenance["prepared_at"],
            "entries": [{"api_row": raw_fixture(), "source": source, "stats": [{"corners": 0, "shots_on": 0}] * 2}]}
    path = tmp_path / "Index/prediction_experiments/recovery/plan.json"; path.parent.mkdir(parents=True)
    body = json.dumps(plan).encode(); path.write_bytes(body)
    (path.parent / "COMPLETE.json").write_text(json.dumps({path.name: hashlib.sha256(body).hexdigest()}))
    _, history, evidence, _ = load_eligible_inputs(db, tmp_path, AS_OF, recovery_plans=[path])
    assert evidence[100]["source_class"] == "verified_local_reconstruction"
    assert history[0]["home"]["corners"] is None
    assert "home_corners_ambiguous_legacy_zero" in evidence[100]["market_eligibility"]["corners"]["reasons"]
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_eligible_inputs(db, tmp_path, AS_OF, recovery_plans=[path])


def test_source_symlink_and_external_uri_rejected(tmp_path):
    root = tmp_path / "project"; root.mkdir()
    resolver = LocalEvidence(root, [], as_of=AS_OF)
    with pytest.raises(ValueError, match="outside_allowed"):
        resolver._local_path(tmp_path / "other.json", root / "Index/raw_archive")
    with pytest.raises(ValueError, match="not_local"):
        resolver.read_archive({"provider": "api_football", "storage_backend": "local", "storage_uri": "https://example.com/data",
                               "fetched_at": "2026-01-01T00:00:00Z"})


def test_null_and_omitted_statistics_are_not_explicit_zero(settings, engine, session_factory, tmp_path):
    _seed(session_factory)
    db = _path(settings)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE fixtures SET round='Regular Season - 1'")
        connection.execute("UPDATE fixture_team_stats SET corners=NULL, shots_on=NULL")
    archive(tmp_path, db, "/fixtures", [raw_fixture()], params={"league": 39, "season": 2026}, archive_id=1)
    blocks = raw_stats(corners=None, sot=None)
    blocks[1]["statistics"] = blocks[1]["statistics"][:1]
    archive(tmp_path, db, "/fixtures/statistics", blocks, params={"fixture": 100}, archive_id=2)
    _, history, evidence, _ = load_eligible_inputs(db, tmp_path, AS_OF)
    assert history[0]["home"]["corners"] is None
    e = evidence[100]
    assert e["statistic_presence"]["home"]["sot"]["state"] == "null"
    assert e["statistic_presence"]["away"]["sot"]["state"] == "omitted"
    assert not e["market_eligibility"]["sot"]["eligible"]


def test_duplicate_fixture_archive_is_rejected_atomically(settings, engine, session_factory, tmp_path):
    _seed(session_factory)
    db = _path(settings)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE fixtures SET round='Regular Season - 1'")
    archive(tmp_path, db, "/fixtures", [raw_fixture(), raw_fixture()], params={"league": 39, "season": 2026}, archive_id=1)
    _, history, evidence, report = load_eligible_inputs(db, tmp_path, AS_OF)
    assert history == []
    assert report["archive_errors"] == [{"archive_id": 1, "reason": "duplicate_or_invalid_archived_fixture_id"}]
    assert not evidence[100]["period_verified"]


def test_staged_source_requires_consistent_linked_archives(settings, engine, session_factory, tmp_path):
    db = seeded(settings, session_factory, tmp_path)
    with sqlite3.connect(db) as connection:
        hashes = dict(connection.execute("SELECT id, payload_digest FROM raw_payload_archive"))
        provenance = {"schema": "verified-staged-history.v1", "fixture_archive_id": 1, "statistics_archive_id": 2,
                      "fixture_payload_sha256": hashes[1], "statistics_payload_sha256": hashes[2],
                      "statistics_response_present": True, "zero_policy": "explicit_provider_zero_or_unknown"}
        connection.execute("UPDATE fixture_team_stats SET stats_json=? WHERE fixture_id=(SELECT id FROM fixtures WHERE api_football_id=100)",
                           (json.dumps({"_staged_history": provenance}),))
    _, history, evidence, _ = load_eligible_inputs(db, tmp_path, AS_OF)
    assert len(history) == 1
    assert evidence[100]["source_provenance"] == provenance
    provenance["fixture_archive_id"] = 999
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE fixture_team_stats SET stats_json=? WHERE fixture_id=(SELECT id FROM fixtures WHERE api_football_id=100) AND is_home=1",
                           (json.dumps({"_staged_history": provenance}),))
    _, history, evidence, _ = load_eligible_inputs(db, tmp_path, AS_OF)
    assert history == []
    assert evidence[100]["common_reasons"] == ["staged_provenance_pair_mismatch"]


def test_future_observed_archive_not_used_for_export(settings, engine, session_factory, tmp_path):
    _seed(session_factory)
    db = _path(settings)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE fixtures SET round='Regular Season - 1'")
    archive(tmp_path, db, "/fixtures", [raw_fixture()], params={"league": 39, "season": 2026}, archive_id=1,
            fetched="2027-01-01T00:00:00Z")
    _, history, evidence, _ = load_eligible_inputs(db, tmp_path, AS_OF)
    assert history == []
    assert evidence[100]["common_reasons"] == ["period_evidence_unavailable"]
