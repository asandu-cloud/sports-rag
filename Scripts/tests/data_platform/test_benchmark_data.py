from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from Scripts.data_platform.features.benchmarks import data
from Scripts.rag_ingest.core import model_features as core


def sha(body):
    return hashlib.sha256(body).hexdigest()


def dumps(value):
    return json.dumps(value, sort_keys=True).encode()


def synthetic():
    schema = data.expected_schema()
    contract = core.digest(schema)
    dates = [f"2021-01-{day:02d}T12:00:00+00:00" for day in range(1, 6)] + [
        "2021-12-10T12:00:00+00:00", "2021-12-31T21:00:00+00:00",
        "2022-03-01T12:00:00+00:00", "2022-07-01T00:00:00+00:00", "2024-02-01T00:00:00+00:00"]
    rows, labels, memberships = [], [], []
    for fid, kickoff in enumerate(dates, 1):
        partition = "initial_training" if fid <= 7 else "development" if fid <= 9 else "phase3_confirmation"
        eligible = fid >= 6
        membership = {"fixture_id": fid, "kickoff": kickoff, "competition": "EPL", "season": int(kickoff[:4]),
                      "completed": True, "eligible_markets": list(data.MARKETS) if eligible else [],
                      "partition": partition, "row_count": 1}
        memberships.append(membership)
        if fid == 10:
            continue
        values = {name: None for name in schema["names"] if not name.endswith("__missing")}
        for league in data.COMPETITIONS:
            values["competition_" + league] = float(league == "EPL")
        support, decisions = {}, {}
        for market in data.MARKETS:
            reasons = []
            support[market] = {}
            for side in ("home", "away"):
                if eligible:
                    values[side + "_" + core.RATE_FIELDS[market]] = 0.0 if market == "goals" else 1.0
                else:
                    reasons += [f"insufficient_{side}_history", f"missing_{side}_production_rate"]
                support[market][side] = {"count": 5 if eligible else 0,
                    "fixture_ids": list(range(1, 6)) if eligible else [],
                    "primary_fixture_ids": list(range(1, 6)) if eligible else [],
                    "band": "5-9" if eligible else "0", "rate_available": eligible}
            decisions[market] = {"eligible": eligible, "reasons": reasons, "primary_reason": reasons[0] if reasons else None}
        decisions["cards"] = {"eligible": False, "reasons": ["cards_target_not_qualified"], "primary_reason": "cards_target_not_qualified"}
        for name in schema["names"]:
            if name.endswith("__missing"):
                values[name] = float(values[name[:-9]] is None)
        rows.append({"fixture": {"fixture_id": fid, "kickoff": kickoff, "season": membership["season"],
                    "competition": "EPL", "home_team_id": 1, "away_team_id": 2, "status": "FT"},
            "as_of": kickoff, "label_available_at": (datetime.fromisoformat(kickoff) + timedelta(hours=3)).isoformat(),
            "observed_at": "2026-09-26T00:00:00+00:00", "actual_observed_at": "2026-09-26T00:00:00+00:00",
            "partition": partition, "availability": "assumed_final", "feature_contract_id": contract,
            "forecast_stage": schema["forecast_stage"], "snapshot_id": sha(str(fid).encode()),
            "source_class": "verified_local_reconstruction", "round_group": "domestic_regular",
            "values": [values[name] for name in schema["names"]], "support": support, "market_eligibility": decisions})
        labels.append({"fixture_id": fid, "labels": {"goals": 0, "corners": 2, "sot": 2, "cards": None},
                       "team_labels": {side: {"goals": 0, "corners": 1, "sot": 1, "cards": None} for side in ("home", "away")}})
    splits = {"version": "phase3-chronological-splits.v1", "status": "defined", "memberships": memberships,
        "partition_by_fixture": {str(m["fixture_id"]): m["partition"] for m in memberships},
        "boundaries": {"initial_training_start": "2021-01-01T00:00:00Z", "development_start": "2022-01-01T00:00:00Z",
            "phase3_confirmation_start": "2024-01-01T00:00:00Z", "calibration_start": "2025-01-01T00:00:00Z",
            "final_system_start": "2025-07-01T00:00:00Z", "final_system_end": "2026-07-01T00:00:00Z"},
        "development_folds": [{"fold_id": "development-01", "fit_cutoff": "2022-01-01T00:00:00Z",
                               "validation_start": "2022-01-01T00:00:00Z", "validation_end": "2022-07-01T00:00:00Z"}]}
    manifest = {"version": "phase3-dataset.v1", "availability": "assumed_final", "feature_contract_id": contract,
        "row_count": len(memberships), "publication_enabled": False, "promotion_allowed": False,
        "league_weighting": "equal_fixture_before_recency", "snapshot_sha256": "a" * 64,
        "source_hashes": {name: sha((data.ROOT / name).read_bytes()) for name in data.COMPATIBLE_SOURCES}}
    return {"manifest.json": manifest, "feature-schema.json": schema, "splits.json": splits,
            "lockbox-policy.json": {"version": "phase3-lockbox.v1", "development_partitions": sorted(data.DEVELOPMENT),
                                    "status": "unopened_for_model_evaluation", "inspection_events": []},
            "development/features.jsonl": rows, "development/labels.jsonl": labels}


def save(path, docs):
    path.mkdir(exist_ok=True)
    splits, manifest = docs["splits.json"], docs["manifest.json"]
    splits["membership_sha256"] = core.digest(splits["memberships"])
    splits["sha256"] = core.digest({key: value for key, value in splits.items() if key != "sha256"})
    complete = {"audit/inputs.json": "b" * 64, "audit/evidence.jsonl": "c" * 64,
                "platform-snapshot.db": "d" * 64, "lockbox/labels.jsonl": "e" * 64}
    manifest["dataset_id"] = core.digest({"snapshot": manifest["snapshot_sha256"], "features": manifest["feature_contract_id"],
        "inputs": complete["audit/inputs.json"], "splits": core.digest(splits), "evidence": complete["audit/evidence.jsonl"],
        "sources": manifest["source_hashes"]})
    for name, value in docs.items():
        item = path / name
        item.parent.mkdir(exist_ok=True)
        body = b"\n".join(dumps(row) for row in value) + b"\n" if name.endswith("jsonl") else dumps(value)
        item.write_bytes(body)
        complete[name] = sha(body)
    (path / "COMPLETE.json").write_bytes(dumps(complete))


@pytest.fixture
def dataset(tmp_path):
    docs = synthetic()
    save(tmp_path, docs)
    return tmp_path, docs


def test_exact_development_reader_never_opens_heldout_audit_database_or_source_archive(dataset, monkeypatch):
    path, _ = dataset
    original, opened = Path.read_bytes, []
    def guarded(item):
        if item.is_relative_to(path):
            relative = item.relative_to(path).as_posix()
            assert relative in data.ALLOWED_FILES | {"COMPLETE.json"}
            opened.append(relative)
        return original(item)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    loaded = data.DevelopmentDataset(path)
    assert set(opened) == data.ALLOWED_FILES | {"COMPLETE.json"}
    assert len(loaded.rows) == 9
    with pytest.raises(ValueError, match="refuses"):
        loaded._read("lockbox/labels.jsonl")


@pytest.mark.parametrize("market", data.MARKETS)
def test_fold_strict_label_availability_boundaries_zero_and_reproducibility(dataset, market):
    loaded = data.DevelopmentDataset(dataset[0])
    train, validation, weights, report = loaded.select_fold(market, "development-01")
    assert [row["fixture"]["fixture_id"] for row in train] == [6]
    assert [row["fixture"]["fixture_id"] for row in validation] == [8]
    assert report["exclusions"]["label_available_after_fit"] == 1  # exactly cutoff is unavailable
    assert report["effective_sample_size"] == 1
    assert weights.tolist() == [1]
    assert report == data.DevelopmentDataset(dataset[0]).select_fold(market, "development-01")[3]
    assert train[0]["target"] == (0 if market == "goals" else 2)
    assert loaded.select_fold(market, "development-01", lookback_days=10)[0] == []


def test_recency_uses_elapsed_time_not_row_order_and_equal_fixture_league_weights():
    cutoff = datetime(2022, 1, 1, tzinfo=timezone.utc)
    rows = [{"fixture": {"fixture_id": fid, "kickoff": (cutoff - timedelta(days=days)).isoformat(), "competition": league}}
            for fid, days, league in [(1, 10, "EPL"), (2, 20, "EPL"), (3, 30, "UCL")]]
    weights, report = data.recency_weights(rows, cutoff, 10)
    assert weights == pytest.approx([12/7, 6/7, 3/7])
    assert report["effective_sample_size"] == pytest.approx(7/3)
    assert report["weight_sum"] == pytest.approx(3)
    assert report["league_contributions"]["EPL"]["raw_share"] == pytest.approx(2/3)
    assert report["league_contributions"]["EPL"]["weighted_share"] == pytest.approx(6/7)
    assert data.recency_weights(list(reversed(rows)), cutoff, 10)[0] == pytest.approx(weights[::-1])
    assert data.recency_weights(rows, cutoff)[0].tolist() == [1, 1, 1]
    with pytest.raises(ValueError, match="Duplicate"):
        data.recency_weights(rows + rows[:1], cutoff)
    with pytest.raises(ValueError, match="underflow"):
        data.recency_weights(rows, cutoff, 1e-10)


@pytest.mark.parametrize("recipe", [0, -1, float("nan"), float("inf"), True, "365"])
def test_invalid_recency_and_lookback_rejected(dataset, recipe):
    loaded = data.DevelopmentDataset(dataset[0])
    for name in ("lookback_days", "half_life_days"):
        with pytest.raises(ValueError):
            loaded.select_fold("goals", "development-01", **{name: recipe})


def test_predeclared_family_gates_and_validation_minimum():
    report = {"training_unique_fixtures": 5000, "effective_sample_size": 2000, "validation_unique_fixtures": 500}
    for family in ("ridge", "poisson", "lightgbm", "xgboost", "catboost", "league_average", "statistical"):
        assert data.family_gate(report, family)["qualified"]
    for key in report:
        assert not data.family_gate({**report, key: 0}, "ridge")["qualified"]
    assert not data.family_gate({**report, "effective_sample_size": 1999}, "lightgbm")["qualified"]
    assert data.family_gate({**report, "effective_sample_size": 1999}, "ridge")["qualified"]
    with pytest.raises(ValueError):
        data.family_gate({**report, "effective_sample_size": float("nan")}, "ridge")


def test_tampered_checksums_and_path_traversal_rejected(dataset):
    path, docs = dataset
    (path / "development/features.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        data.DevelopmentDataset(path)
    save(path, docs)
    complete = json.loads((path / "COMPLETE.json").read_text())
    for unsafe in ("../secret", "/secret", "development/../lockbox/labels.jsonl", "development\\features.jsonl"):
        (path / "COMPLETE.json").write_bytes(dumps({**complete, unsafe: "a" * 64}))
        with pytest.raises(ValueError, match="Unsafe"):
            data.DevelopmentDataset(path)


def test_symlink_rejected_even_if_content_checksum_matches(dataset, tmp_path):
    path, _ = dataset
    item = path / "development/features.jsonl"
    destination = path / "saved-features.jsonl"
    item.rename(destination)
    item.symlink_to(destination)
    with pytest.raises(ValueError, match="symlink"):
        data.DevelopmentDataset(path)


@pytest.mark.parametrize("change", ["wrong_schema", "source_hash", "duplicate", "snapshot_duplicate", "target_negative",
    "target_fraction", "target_bool", "target_missing", "team_total", "future_asof", "naive_timestamp", "label_available",
    "missingness", "vector_length", "nonfinite", "support_count", "support_future", "support_reason", "eligibility",
    "split_metadata", "split_partition", "fold_heldout", "fixture_identity", "cards", "conflicting_observed", "manifest_count"])
def test_resigned_invalid_contracts_still_rejected(dataset, change):
    path, docs = dataset
    row = docs["development/features.jsonl"][5]
    label = docs["development/labels.jsonl"][5]
    splits = docs["splits.json"]
    if change == "wrong_schema":
        docs["feature-schema.json"]["names"][0] = "wrong"
    elif change == "source_hash":
        docs["manifest.json"]["source_hashes"][data.COMPATIBLE_SOURCES[0]] = "f" * 64
    elif change == "duplicate":
        docs["development/features.jsonl"].append(deepcopy(row))
    elif change == "snapshot_duplicate":
        row["snapshot_id"] = docs["development/features.jsonl"][0]["snapshot_id"]
    elif change.startswith("target_"):
        label["labels"]["goals"] = {"target_negative": -1, "target_fraction": .5, "target_bool": True, "target_missing": None}[change]
    elif change == "team_total":
        label["team_labels"]["home"]["goals"] = 2
    elif change == "future_asof":
        row["as_of"] = "2026-01-01T00:00:00Z"
    elif change == "naive_timestamp":
        row["as_of"] = row["as_of"][:19]
    elif change == "label_available":
        row["label_available_at"] = row["as_of"]
    elif change == "missingness":
        row["values"][docs["feature-schema.json"]["names"].index("home_" + core.RATE_FIELDS["goals"] + "__missing")] = 1
    elif change == "vector_length":
        row["values"].pop()
    elif change == "nonfinite":
        row["values"][0] = float("nan")
    elif change == "support_count":
        row["support"]["goals"]["home"]["count"] = 4
    elif change == "support_future":
        row["support"]["goals"]["home"]["fixture_ids"][-1] = 8
    elif change == "support_reason":
        docs["development/features.jsonl"][0]["market_eligibility"]["goals"]["reasons"].pop()
    elif change == "eligibility":
        row["market_eligibility"]["goals"]["eligible"] = False
    elif change == "split_metadata":
        splits["memberships"][5]["season"] += 1
    elif change == "split_partition":
        splits["memberships"][5]["partition"] = "development"
    elif change == "fold_heldout":
        splits["development_folds"][0]["validation_end"] = "2025-01-01T00:00:00Z"
    elif change == "fixture_identity":
        row["fixture"]["home_team_id"] = True
    elif change == "cards":
        row["market_eligibility"]["cards"] = {"eligible": True, "reasons": [], "primary_reason": None}
    elif change == "conflicting_observed":
        row["actual_observed_at"] = "2026-09-25T00:00:00Z"
    elif change == "manifest_count":
        docs["manifest.json"]["row_count"] += 1
    save(path, docs)
    with pytest.raises(ValueError):
        data.DevelopmentDataset(path)


def test_changed_nonfeature_runner_hash_does_not_require_reexport(dataset):
    path, docs = dataset
    docs["manifest.json"]["source_hashes"]["Scripts/ops/prediction_benchmark.py"] = "a" * 64
    save(path, docs)
    assert data.DevelopmentDataset(path).rows


@pytest.mark.parametrize("market", (*data.MARKETS, "cards"))
@pytest.mark.parametrize("bad", [-1, .5, True, float("inf")])
def test_every_target_is_validated_including_ineligible_rows(dataset, market, bad):
    path, docs = dataset
    label = docs["development/labels.jsonl"][0]  # no eligible markets
    label["labels"][market] = bad
    label["team_labels"]["home"][market] = bad
    label["team_labels"]["away"][market] = 0
    save(path, docs)
    with pytest.raises(ValueError):
        data.DevelopmentDataset(path)


@pytest.mark.parametrize("market", data.MARKETS)
def test_eligible_missing_target_rejected_but_zero_is_preserved(dataset, market):
    path, docs = dataset
    label = docs["development/labels.jsonl"][5]
    label["labels"][market] = None
    for side in ("home", "away"):
        label["team_labels"][side][market] = None
    save(path, docs)
    with pytest.raises(ValueError, match="missing target"):
        data.DevelopmentDataset(path)


@pytest.mark.parametrize("field,value", [("source_class", "unresolved"), ("round_group", "european_qualifying"),
    ("round_group", "domestic_separate_playoff"), ("source_class", "invented")])
def test_source_and_round_scope_cannot_be_overridden_by_eligibility_flag(dataset, field, value):
    path, docs = dataset
    docs["development/features.jsonl"][5][field] = value
    save(path, docs)
    with pytest.raises(ValueError):
        data.DevelopmentDataset(path)
