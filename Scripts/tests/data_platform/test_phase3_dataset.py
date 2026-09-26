"""Batch A exporter integration against isolated snapshots and synthetic evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import socket
import sqlite3

import pytest

from Scripts.data_platform.features import history_evidence
from Scripts.ops import prediction_benchmark as benchmark


AS_OF = datetime(2026, 9, 26, tzinfo=timezone.utc)
COMPETITIONS = {"EPL": "domestic_league", "UCL": "continental_cup"}
MARKETS = ("goals", "corners", "sot", "cards")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path):
    return json.loads(path.read_text())


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _synthetic_inputs():
    fixtures, history, evidence = [], [], {}
    for season in range(2019, 2027):
        starts = [datetime(season, 8, 1, 12, tzinfo=timezone.utc)]
        if season < 2026:
            starts.append(datetime(season + 1, 2, 1, 12, tzinfo=timezone.utc))
        for start in starts:
            for day in range(7):
                fid = len(fixtures) + 1
                kickoff = start + timedelta(days=day)
                actual_observed = (kickoff + timedelta(days=2)).isoformat()
                fixture = {
                    "fixture_id": fid, "competition": "EPL", "season": season,
                    "home_team_id": 1, "away_team_id": 2,
                    "kickoff": kickoff.isoformat(), "status": "FT", "referee": None,
                    "referee_observed_at": None,
                    "observed_at": (kickoff + timedelta(hours=4)).isoformat(),
                }
                home = {"goals": fid % 3, "corners": fid % 7, "sot": fid % 5,
                        "cards": None, "shots": 9, "fouls": 8, "xg": None, "possession": 50}
                away = {**home, "goals": (fid + 1) % 3}
                fixtures.append(fixture)
                history.append({**fixture, "observed_at": actual_observed, "home": home, "away": away})
                evidence[fid] = {
                    "fixture_id": fid, "source_class": "raw_provider_archive",
                    "round_group": "domestic_regular", "actual_observed_at": actual_observed,
                    "market_eligibility": {
                        market: {"eligible": market != "cards",
                                 "reasons": [] if market != "cards" else ["cards_target_not_qualified"]}
                        for market in MARKETS
                    },
                    "labels": {market: home[market] + away[market] if market != "cards" else None
                               for market in MARKETS},
                    "team_labels": {side: {market: values[market] for market in MARKETS}
                                    for side, values in (("home", home), ("away", away))},
                }
    return fixtures, history, evidence, {"ft_targets": len(fixtures), "synthetic_evidence": True}


@pytest.fixture
def dataset_factory(tmp_path, monkeypatch):
    root = tmp_path / "project"
    database = root / "Index/platform.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE preserved (id INTEGER PRIMARY KEY, value TEXT)")
        db.execute("INSERT INTO preserved VALUES (1, 'unchanged canonical data')")
    active = root / "Index/ml_models/model_r2.json"
    active.parent.mkdir()
    active.write_text('{"goals": 0.0181}')
    original_database, original_model = _sha(database), _sha(active)
    source = _synthetic_inputs()

    def no_network(*args, **kwargs):
        raise AssertionError("Dataset preparation must not request providers or publish")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(benchmark, "supported_competitions", lambda: dict(COMPETITIONS))

    def make(name="dataset", *, reverse=False, exclude_all=False):
        payload = deepcopy(source)
        if reverse:
            payload[0].reverse()
            payload[1].reverse()
            payload = (payload[0], payload[1], dict(reversed(list(payload[2].items()))), payload[3])
        if exclude_all:
            for item in payload[2].values():
                for decision in item["market_eligibility"].values():
                    decision.update(eligible=False, reasons=["period_unverified"])

        def evidence_loader(snapshot, source_root, *, as_of):
            assert snapshot != database
            assert source_root == root
            assert as_of == AS_OF
            with sqlite3.connect(snapshot.resolve().as_uri() + "?mode=ro", uri=True) as db:
                assert db.execute("SELECT value FROM preserved WHERE id=1").fetchone()[0] == "unchanged canonical data"
            return deepcopy(payload)

        monkeypatch.setattr(history_evidence, "load_eligible_inputs", evidence_loader)
        result = benchmark.prepare_dataset(root=root, name=name, database=database, as_of=AS_OF)
        assert _sha(database) == original_database
        assert _sha(active) == original_model
        assert list(active.parent.iterdir()) == [active]
        return result

    return make


def _reseal(path, relative):
    """Recompute a checksum to exercise semantic validation beyond corruption."""
    completion = _read(path / "COMPLETE.json")
    completion[relative] = _sha(path / relative)
    (path / "COMPLETE.json").write_text(json.dumps(completion))


def test_prepare_preserves_canonical_data_and_separates_every_reserve(dataset_factory):
    path = dataset_factory()
    manifest = benchmark.verify_dataset(path)
    assert manifest["training_performed"] is False
    assert manifest["publication_enabled"] is False
    assert manifest["promotion_allowed"] is False
    assert _read(path / "snapshot-capture.json")["consistency"] == "sqlite_online_backup"
    with sqlite3.connect((path / "platform-snapshot.db").as_uri() + "?mode=ro", uri=True) as db:
        assert db.execute("PRAGMA quick_check").fetchone() == ("ok",)
    splits = _read(path / "splits.json")
    assert set(item["partition"] for item in splits["memberships"]) == {
        "initial_training", "development", "phase3_confirmation", "calibration",
        "final_system_test", "prospective_reserve",
    }
    development = benchmark.load_development(path, market="goals")
    assert development
    assert all(row["partition"] in benchmark.DEVELOPMENT for row in development)
    assert all(datetime.fromisoformat(row["fixture"]["kickoff"]).year < 2024 for row in development)
    assert all(row["support"]["goals"][side]["count"] >= 5
               for row in development for side in ("home", "away"))
    audit = _rows(path / "audit/feature-rows.jsonl")
    for row in audit:
        kickoff = datetime.fromisoformat(row["fixture"]["kickoff"])
        assert row["actual_observed_at"] == (kickoff + timedelta(days=2)).isoformat()
        assert row["observed_at"] == row["actual_observed_at"]
        assert row["label_available_at"] == (kickoff + timedelta(hours=3)).isoformat()
        assert "labels" not in row and "target" not in row
    assert _read(path / "lockbox-policy.json")["inspection_events"] == []
    assert all("card" not in name for name in _read(path / "feature-schema.json")["names"])
    with pytest.raises(FileExistsError):
        dataset_factory()


def test_development_reader_never_opens_audit_lockboxes_or_snapshot(dataset_factory, monkeypatch):
    path = dataset_factory()
    original = Path.open
    opened = []

    def guarded_open(candidate, *args, **kwargs):
        try:
            relative = candidate.relative_to(path)
        except ValueError:
            return original(candidate, *args, **kwargs)
        assert relative.parts[0] not in {"audit", "lockbox"}, f"Forbidden outcome store: {relative}"
        assert relative.name != "platform-snapshot.db", "Development accessed the complete source database"
        opened.append(relative.as_posix())
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    result = benchmark.load_development(path, market="goals", fit_cutoff="2023-01-01T00:00:00+00:00")
    assert result["rows"]
    assert "development/labels.jsonl" in opened
    assert all(datetime.fromisoformat(row["label_available_at"]) < datetime(2023, 1, 1, tzinfo=timezone.utc)
               for row in result["rows"])


def test_development_cutoff_obeys_strict_label_time_and_confirmation_boundary(dataset_factory):
    path = dataset_factory()
    eligible = benchmark.load_development(path, market="goals")
    chosen = eligible[-1]
    at_label_time = benchmark.load_development(path, market="goals", fit_cutoff=chosen["label_available_at"])
    assert chosen["fixture"]["fixture_id"] not in {row["fixture"]["fixture_id"] for row in at_label_time["rows"]}
    assert any(item["fixture_id"] == chosen["fixture"]["fixture_id"]
               and item["reasons"] == ["label_available_after_or_at_fit"] for item in at_label_time["excluded"])
    boundary = _read(path / "splits.json")["boundaries"]["phase3_confirmation_start"]
    assert benchmark.load_development(path, market="goals", fit_cutoff=boundary)["rows"]
    with pytest.raises(ValueError, match="held-out boundary"):
        benchmark.load_development(path, market="goals", fit_cutoff=datetime.fromisoformat(boundary) + timedelta(seconds=1))
    with pytest.raises(ValueError, match="not qualified"):
        benchmark.load_development(path, market="cards")


def test_replay_uses_frozen_builder_when_current_implementation_changes(dataset_factory, monkeypatch):
    path = dataset_factory()

    def current_builder_must_not_run(*args, **kwargs):
        raise AssertionError("Replay used the current feature implementation")

    monkeypatch.setattr(benchmark.core, "build_features", current_builder_must_not_run)
    monkeypatch.setattr(benchmark.core, "capture_snapshot", current_builder_must_not_run)
    replay = benchmark.replay_dataset(path)
    assert replay["full_replay"] is True
    assert replay["verified_rows"] == _read(path / "manifest.json")["row_count"]
    assert replay["training_performed"] is False


def test_reordered_inputs_preserve_membership_and_feature_hashes(dataset_factory):
    first = dataset_factory("first")
    second = dataset_factory("second", reverse=True)
    first_splits, second_splits = _read(first / "splits.json"), _read(second / "splits.json")
    assert first_splits["membership_sha256"] == second_splits["membership_sha256"]
    assert first_splits["sha256"] == second_splits["sha256"]
    first_complete, second_complete = _read(first / "COMPLETE.json"), _read(second / "COMPLETE.json")
    stable = {"audit/inputs.json", "audit/evidence.jsonl", "audit/feature-rows.jsonl", "splits.json", "feature-schema.json"}
    stable.update(name for name in first_complete if name.startswith(("development/", "lockbox/")))
    assert {name: first_complete[name] for name in stable} == {name: second_complete[name] for name in stable}
    assert _read(first / "manifest.json")["dataset_id"] == _read(second / "manifest.json")["dataset_id"]


def test_all_excluded_cohort_is_reported_without_creating_a_training_sample(dataset_factory):
    path = dataset_factory(exclude_all=True)
    benchmark.verify_dataset(path)
    splits = _read(path / "splits.json")
    assert splits["status"] == "insufficient"
    assert splits["memberships"] == []
    assert "no_eligible_completed_fixtures" in splits["reasons"]
    assert not (path / "development").exists()
    assert (path / "lockbox/quarantine/features.jsonl").is_file()
    assert all(item["eligible"] == 0 for item in _read(path / "coverage.json")["groups"]["all"]["markets"].values())
    with pytest.raises(ValueError, match="Incomplete development dataset|Insufficient eligible coverage"):
        benchmark.load_development(path, market="goals")


@pytest.mark.parametrize("relative", ["development/labels.jsonl", "lockbox/final_system_test/labels.jsonl", "source/model_features.py"])
def test_explicit_verification_rejects_corrupted_artifacts(dataset_factory, relative):
    path = dataset_factory()
    with (path / relative).open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        benchmark.verify_dataset(path)


def test_feature_contract_mismatch_is_rejected_after_hash_validation(dataset_factory):
    path = dataset_factory()
    manifest = _read(path / "manifest.json")
    manifest["feature_contract_id"] = "wrong-contract"
    (path / "manifest.json").write_text(json.dumps(manifest))
    _reseal(path, "manifest.json")
    with pytest.raises(ValueError, match="Feature contract mismatch"):
        benchmark.load_development(path, market="goals")


def test_development_reader_rejects_heldout_identity_even_if_row_is_relabelled(dataset_factory):
    path = dataset_factory()
    rows = _rows(path / "development/features.jsonl")
    heldout = _rows(path / "lockbox/phase3_confirmation/features.jsonl")[-1]
    heldout["partition"] = "development"
    rows.append(heldout)
    relative = "development/features.jsonl"
    (path / relative).write_text("".join(json.dumps(row) + "\n" for row in rows))
    _reseal(path, relative)
    with pytest.raises(ValueError, match="Held-out fixture"):
        benchmark.load_development(path, market="goals")
