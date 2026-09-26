"""Confirmation temporal boundaries, permanent consumption, gates and lifecycle."""
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Scripts.data_platform.features.benchmarks import confirmation as c, confirmation_data as cd
from Scripts.data_platform.features.benchmarks import data, baselines as b
from Scripts.data_platform.features.benchmarks.artifacts import read_json, sha, verify_complete
from Scripts.data_platform.features.benchmarks.checkpoints import complete_atomic
from Scripts.data_platform.features.phase3_features import build_reference, HistoryIndex
from Scripts.rag_ingest.core import model_features as core
from Scripts.tests.data_platform.test_benchmark_data import synthetic, save
from Scripts.tests.data_platform.test_benchmark_baselines import fixture


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def confirmation_fixture(tmp_path):
    docs = synthetic()
    save(tmp_path, docs)
    loaded = data.DevelopmentDataset(tmp_path)
    row = deepcopy(loaded.rows[-1])
    row.pop("labels"); row.pop("team_labels")
    m = loaded.memberships[10]
    row["fixture"].update(fixture_id=10, kickoff=m["kickoff"], season=m["season"])
    row.update(partition=cd.PARTITION, as_of=m["kickoff"], snapshot_id=core.digest(10),
               label_available_at=(core.utc(m["kickoff"]) + timedelta(hours=3)).isoformat())
    return loaded, row


def test_confirmation_has_separate_reader_development_still_refuses_it(tmp_path):
    loaded, row = confirmation_fixture(tmp_path)
    assert cd.feature_rows(encoded(row), loaded) == [row]
    with pytest.raises(ValueError):
        loaded._validate_row(row)
    for name in (cd.FEATURES, cd.LABELS, cd.HISTORY, "lockbox/calibration/labels.jsonl", "lockbox/final_system_test/labels.jsonl"):
        with pytest.raises(ValueError, match="refuses"):
            loaded._read(name)


def test_2024_european_league_phase_is_recognized_without_changing_eligibility(tmp_path):
    loaded, row = confirmation_fixture(tmp_path)
    row["round_group"] = "european_league"
    assert cd.feature_rows(encoded(row), loaded)[0]["market_eligibility"] == row["market_eligibility"]
    row["market_eligibility"]["goals"] = {"eligible": False, "primary_reason": "missing_home_goals", "reasons": ["missing_home_goals"]}
    loaded.memberships[10]["eligible_markets"].remove("goals")
    assert not cd.feature_rows(encoded(row), loaded)[0]["market_eligibility"]["goals"]["eligible"]


def test_source_repair_keeps_original_specification_and_rejects_model_or_gate_changes(tmp_path):
    source = "Scripts/repair.py"
    directory = tmp_path / "correctness-amendment-0001"
    path = directory / "source" / source
    path.parent.mkdir(parents=True)
    path.write_text("fixed validator")
    saved = {"source_hashes": {source: "old"}, "weight": .5, "rules": {"minimum": .01}, "dataset": "same"}
    current = {**saved, "source_hashes": {source: sha(path)}}
    record = {"original_specification_id": core.digest(saved), "original_source_hashes": saved["source_hashes"],
        "replacement_source_hashes": current["source_hashes"], "changed_sources": [source], "reason": "validation omission",
        "predictions_generated_before_repair": False, "labels_scored_before_repair": False,
        "unchanged_candidates_cohorts_and_rules": True}
    (directory / "amendment.json").write_bytes(encoded(record))
    complete_atomic(directory)
    assert c.verify_source_amendment(tmp_path, saved, current)["original_specification_id"] == core.digest(saved)
    for changed in ({**current, "weight": .9}, {**current, "rules": {"minimum": 0}}, {**current, "dataset": "changed"}):
        with pytest.raises(ValueError, match="specification changed"): c.verify_source_amendment(tmp_path, saved, changed)
    with pytest.raises(ValueError, match="inconsistent"):
        c.verify_source_amendment(tmp_path, saved, {**current, "source_hashes": {source: "unrecorded"}})


@pytest.mark.parametrize("change", ["partition", "later", "early", "as_of", "label_time", "duplicate", "drop", "target",
    "embedded", "missingness", "support_future", "support_missing", "competition", "cards", "source", "nan"])
def test_confirmation_contract_failures(tmp_path, change):
    loaded, row = confirmation_fixture(tmp_path)
    if change == "partition": row["partition"] = "development"
    if change == "later": row["fixture"]["kickoff"] = "2025-01-01T00:00:00Z"
    if change == "early": row["fixture"]["kickoff"] = "2023-12-31T00:00:00Z"
    if change == "as_of": row["as_of"] = "2024-02-02T00:00:00Z"
    if change == "label_time": row["label_available_at"] = row["as_of"]
    if change == "target": row["target"] = 0
    if change == "embedded": row["fixture"]["home"] = {"goals": 0}
    if change == "missingness": row["values"][loaded.schema["names"].index("home_goals_for_pm__missing")] = 1
    if change == "support_future": row["support"]["goals"]["home"]["fixture_ids"][0] = 10
    if change == "support_missing": row["support"]["goals"]["home"]["fixture_ids"][0] = 999
    if change == "competition": row["fixture"]["competition"] = "LaLiga"
    if change == "cards": row["market_eligibility"]["cards"] = {"eligible": True, "reasons": [], "primary_reason": None}
    if change == "source": row["source_class"] = "unresolved"
    if change == "nan": row["values"][0] = float("nan")
    body = b"" if change == "drop" else encoded(row) + (b"\n" + encoded(row) if change == "duplicate" else b"")
    with pytest.raises(ValueError): cd.feature_rows(body, loaded)


def test_skips_later_outcome_payload_and_fixture_duplicates_before_decoding(monkeypatch):
    before = fixture(1, season=2024)
    later = fixture(2, season=2025)
    later["home"]["goals"] = "POISON_LATER_OUTCOME"
    archive = json.dumps({"competitions": data.COMPETITIONS, "fixtures": ["POISON_DUPLICATE_OUTCOME"],
                          "history": [before, later]})
    decode = cd._decode
    def watched(body):
        assert "POISON" not in body
        return decode(body)
    monkeypatch.setattr(cd, "_decode", watched)
    rows, competitions, audit = cd.history_before(archive, "2025-01-01T00:00:00Z")
    assert rows == [before] and competitions == data.COMPETITIONS
    assert audit["opaque_later_history_rows"] == 1 and audit["later_outcomes_decoded"] is False


def test_history_strict_label_boundary_nested_strings_and_chronology():
    end = core.utc("2025-01-01T00:00:00Z")
    rows = [fixture(1, kickoff=(end - timedelta(hours=3, seconds=1)).isoformat()),
            fixture(2, kickoff=(end - timedelta(hours=3)).isoformat())]
    rows[1]["home"]["unparsed"] = 'escaped " braces } [ \\ and unicode é'
    archive = {"competitions": data.COMPETITIONS, "fixtures": [], "history": rows}
    assert cd.history_before(json.dumps(archive), end)[0] == rows[:1]
    archive["history"].reverse()
    with pytest.raises(ValueError, match="chronology"): cd.history_before(json.dumps(archive), end)


@pytest.mark.parametrize("body", ['{"history": [], "history": []}', '{"history":[', '{"x": [}', '{} trailing'])
def test_corrupt_archive_structure_rejected(body):
    with pytest.raises((ValueError, IndexError)): cd.history_before(body, "2025-01-01T00:00:00Z")


def test_confirmation_replay_matches_existing_arithmetic_and_ignores_future_results():
    history = [fixture(i, day=i * 4, season=2023) for i in range(1, 9)]
    future = fixture(90, season=2024, day=20)
    target = fixture(89, season=2024, day=10)
    identity = {k: v for k, v in target.items() if k not in {"home", "away"}}
    history += [target, future]
    snapshot, _, view = build_reference(identity, HistoryIndex(history), data.COMPETITIONS)
    row = {"fixture": identity, "as_of": identity["kickoff"], "snapshot_id": snapshot["snapshot_id"],
        **view, "market_eligibility": {m: {"eligible": True} for m in data.MARKETS}}
    scoring = b.baseline_metadata(c.ROOT)["scoring"]
    initial = cd.prepare_rows([row], history, data.COMPETITIONS, scoring=scoring, end="2025-01-01T00:00:00Z")
    inputs = b.reconstruct_snapshot(snapshot, scoring=scoring)
    for baseline in initial:
        assert baseline["statistical"] == b.statistical_projection(inputs["profiles"]["home"], inputs["profiles"]["away"],
            inputs["recent"]["home"], inputs["recent"]["away"], baseline["market"], scoring=scoring)["value"]
    for item in history[-2:]:
        item["home"] = {m: 9999 for m in data.MARKETS}
        item["away"] = {m: 9999 for m in data.MARKETS}
    assert cd.prepare_rows([row], history, data.COMPETITIONS, scoring=scoring, end="2025-01-01T00:00:00Z") == initial
    row["values"][0] = 999
    with pytest.raises(ValueError, match="replay"):
        cd.prepare_rows([row], history, data.COMPETITIONS, scoring=scoring, end="2025-01-01T00:00:00Z")


@pytest.mark.parametrize("change", ["missing", "duplicate", "sum", "negative", "fraction", "cards", "unknown", "boolean"])
def test_labels_require_exact_membership_and_verified_team_totals(tmp_path, change):
    _, row = confirmation_fixture(tmp_path)
    label = {"fixture_id": 10, "labels": {m: 0 for m in data.MARKETS} | {"cards": None},
        "team_labels": {s: {m: 0 for m in data.MARKETS} | {"cards": None} for s in ("home", "away")}}
    assert cd.labels_by_fixture(encoded(label), [row])[10]["goals"] == 0
    if change == "sum": label["labels"]["goals"] = 1
    if change == "negative": label["labels"]["goals"] = -1
    if change == "fraction": label["labels"]["goals"] = .5
    if change == "cards": label["team_labels"]["home"]["cards"] = 1
    if change == "unknown": label["fixture_id"] = 999
    if change == "boolean": label["labels"]["goals"] = False
    body = b"" if change == "missing" else encoded(label) + (b"\n" + encoded(label) if change == "duplicate" else b"")
    with pytest.raises(ValueError): cd.labels_by_fixture(body, [row])


def test_read_allowlist_hash_and_symlink(tmp_path):
    path = tmp_path / cd.FEATURES
    path.parent.mkdir(parents=True)
    path.write_text("original")
    assert cd.checked_read(tmp_path, cd.FEATURES, sha(path)) == b"original"
    with pytest.raises(ValueError, match="checksum"): cd.checked_read(tmp_path, cd.FEATURES, "a" * 64)
    for name in ("lockbox/calibration/labels.jsonl", "lockbox/final_system_test/features.jsonl", "../secret"):
        with pytest.raises(ValueError, match="refuses"): cd.checked_read(tmp_path, name, "a" * 64)
    original = tmp_path / "original"
    path.rename(original); path.symlink_to(original)
    with pytest.raises(ValueError, match="path"): cd.checked_read(tmp_path, cd.FEATURES, sha(original))


def test_permanent_claim_cannot_change_recipe_experiment_or_dataset(tmp_path):
    registry = tmp_path / "registry"
    spec = {"dataset_id": "dataset", "recipe": "frozen", "seed": 42}
    first = c.inspection_claim(registry, spec, tmp_path / "experiment")
    before = sha(registry / "OPENED.json")
    assert c.inspection_claim(registry, spec, tmp_path / "experiment") == first
    for other, experiment in [({**spec, "recipe": "retuned"}, "experiment"), ({**spec, "seed": 43}, "experiment"),
                              ({**spec, "dataset_id": "other"}, "experiment"), (spec, "another")]:
        with pytest.raises(ValueError, match="already consumed"):
            c.inspection_claim(registry, other, tmp_path / experiment)
    assert sha(registry / "OPENED.json") == before


def test_stage_resume_retains_failures_and_reuses_complete_without_recomputing(tmp_path):
    def fail(path):
        (path / "partial").write_text("evidence")
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError): c._stage(tmp_path, "inputs", fail)
    path = c._stage(tmp_path, "inputs", lambda p: (p / "success").write_text("finished"))
    assert path.name == "inputs-0002" and (tmp_path / "inputs-0001/partial").exists()
    assert c._stage(tmp_path, "inputs", lambda p: pytest.fail("must reuse")) == path
    (path / "success").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"): c._stage(tmp_path, "inputs", lambda p: None)


def uncertainty():
    support = {"unique_fixtures": 1500, "calendar_weeks": 40, "match_dates": 200}
    overall = {"support": support, "point_improvement_at_least_one_percent": True,
        "interval_status": "available", "improvement_interval_excludes_zero": True,
        "improvement_interval": {"lower": .005, "upper": .025}, "relative_rmse_improvement": .02}
    league = {"support": {**support, "unique_fixtures": 300}, "interval_status": "available",
              "slice_deterioration_below_five_percent": True, "deterioration_upper_95": .03}
    return {base + f"_{w}w": {"by_market": {m: {"overall": deepcopy(overall), "slices": {
        "competition": {"EPL": deepcopy(league)}, "missingness_band": {"0": deepcopy(league)}}} for m in data.MARKETS}}
        for base in ("statistical", "league_average") for w in (1, 2)}


def test_gates_require_both_baselines_pooled_before_league_and_report_absent_leagues():
    u = uncertainty()
    decision = c.decision(u)["by_market"]["goals"]
    assert decision["status"] == "qualified_pooled_numerical"
    assert decision["league_qualification"]["EPL"]["qualified"]
    assert not decision["league_qualification"]["SuperLig"]["qualified"]
    u["league_average_1w"]["by_market"]["goals"]["overall"]["point_improvement_at_least_one_percent"] = False
    decision = c.decision(u)["by_market"]["goals"]
    assert decision["status"] == "no_qualified_improvement"
    assert not decision["league_qualification"]["EPL"]["qualified"]


@pytest.mark.parametrize("change", ["count", "weeks", "precision", "negative", "league_count", "league_dates", "league_bound"])
def test_insufficient_harmful_and_slice_gate_outcomes(change):
    u = uncertainty()
    for base in ("statistical", "league_average"):
        overall = u[base + "_1w"]["by_market"]["goals"]["overall"]
        if change == "count": overall["support"]["unique_fixtures"] = 999
        if change == "weeks": overall["support"]["calendar_weeks"] = 19
        if change == "precision": overall["interval_status"] = "unavailable"
        if change == "negative":
            overall.update(improvement_interval_excludes_zero=False, improvement_interval={"lower": -.03, "upper": -.01})
    league = u["statistical_1w"]["by_market"]["goals"]["slices"]["competition"]["EPL"]
    if change == "league_count": league["support"]["unique_fixtures"] = 199
    if change == "league_dates": league["support"]["match_dates"] = 19
    if change == "league_bound": league["slice_deterioration_below_five_percent"] = False
    result = c.decision(u)["by_market"]["goals"]
    if change in {"count", "weeks", "precision"}: assert result["status"] == "inconclusive"
    if change == "negative": assert result["status"] == "harmful_against_at_least_one_baseline"
    assert not result["league_qualification"]["EPL"]["qualified"]


def test_proposal_choice_is_development_only_deterministic_and_exact_tie_retains_original():
    a = {m: {"rmse": 2., "outer_support": {"fixture_membership_sha256": "same"},
             "baseline_contract_id": "same", "origin": "batch_c"} for m in data.MARKETS}
    b = {m: {**v, "origin": "extension"} for m, v in a.items()}
    assert c.choose_proposals(a, b) == a
    b["goals"]["rmse"] = 1.9
    assert c.choose_proposals(a, b)["goals"] == b["goals"]
    b["goals"]["outer_support"] = {"fixture_membership_sha256": "different"}
    with pytest.raises(ValueError, match="identical"): c.choose_proposals(a, b)


def test_full_synthetic_lifecycle_claim_predictions_before_labels_then_completed_resume(tmp_path, monkeypatch):
    """Real filesystem/guards/stages; numerical replay is tested independently."""
    root = tmp_path
    experiments = root / "Index/prediction_experiments"
    dataset = experiments / "dataset"
    dataset.mkdir(parents=True)
    loaded, row = confirmation_fixture(dataset)
    directory = dataset / cd.FEATURES
    directory.parent.mkdir(parents=True)
    directory.write_bytes(encoded(row) + b"\n")
    label = {"fixture_id": 10, "labels": dict(goals=0, corners=2, sot=2, cards=None),
        "team_labels": {s: dict(goals=0, corners=1, sot=1, cards=None) for s in ("home", "away")}}
    (dataset / cd.LABELS).write_bytes(encoded(label) + b"\n")
    (dataset / "audit").mkdir()
    (dataset / cd.HISTORY).write_bytes(encoded({"competitions": data.COMPETITIONS, "history": [], "fixtures": []}))
    complete = read_json(dataset / "COMPLETE.json")
    complete.update({n: sha(dataset / n) for n in (cd.FEATURES, cd.LABELS, cd.HISTORY)})
    manifest = read_json(dataset / "manifest.json")
    manifest["dataset_id"] = core.digest({"snapshot": manifest["snapshot_sha256"], "features": manifest["feature_contract_id"],
        "inputs": complete[cd.HISTORY], "splits": core.digest(loaded.splits), "evidence": complete["audit/evidence.jsonl"],
        "sources": manifest["source_hashes"]})
    (dataset / "manifest.json").write_bytes(encoded(manifest))
    complete["manifest.json"] = sha(dataset / "manifest.json")
    (dataset / "COMPLETE.json").write_bytes(encoded(complete))
    baseline = b.baseline_metadata(c.ROOT)
    directories = {}
    for name in ("baselines", "batch_c", "extension"):
        path = experiments / name; path.mkdir()
        (path / "manifest.json").write_bytes(encoded({"baseline": baseline, "dataset_id": manifest["dataset_id"]}))
        complete_atomic(path); directories[name] = path
    bundle = directories["batch_c"] / "bundle"; bundle.mkdir()
    (bundle / "model").write_text("synthetic no-fit predictor")
    complete_atomic(bundle)
    # Add the candidate under the verified development input allowlist.
    completed = read_json(directories["batch_c"] / "COMPLETE.json")
    completed.update({"bundle/" + n: sha(bundle / n) for n in ("model", "COMPLETE.json")})
    (directories["batch_c"] / "COMPLETE.json").write_bytes(encoded(completed))
    proposals = {m: {"origin": "batch_c", "source_bundle": str(bundle), "source_complete_sha256": sha(bundle / "COMPLETE.json"),
        "weight_ml": .5, "strategy": "selected_blend", "rmse": 1., "outer_support": {"n": 1},
        "baseline_contract_id": core.digest(baseline)} for m in data.MARKETS}
    monkeypatch.setattr(c, "_development_proposals", lambda *a, **k: proposals)
    monkeypatch.setattr(c, "environment_report", lambda: {"synthetic": True})
    def make_baselines(rows, *a, **k):
        return [{"fixture_id": 10, "snapshot_id": row["snapshot_id"], "market": m,
                 "feature_contract_id": row["feature_contract_id"], "statistical": 2., "league_average": 3.} for m in data.MARKETS]
    monkeypatch.setattr(cd, "prepare_rows", make_baselines)
    calls = []
    def infer(path, rows, **kwargs):
        assert all("target" not in r and "labels" not in r for r in rows)
        calls.append(path.name)
        return np.full(len(rows), 1.5)
    monkeypatch.setattr(c, "predict_strategy", infer)
    real_read = cd.checked_read
    def watch(dataset, name, expected):
        claim = list((experiments / "phase3-confirmation-inspections").glob("*/OPENED.json"))
        assert len(claim) == 1, "protected data read before permanent inspection"
        if name == cd.LABELS:
            assert (experiments / "confirmation/predictions-0001/COMPLETE.json").exists()
        return real_read(dataset, name, expected)
    monkeypatch.setattr(cd, "checked_read", watch)
    args = dict(root=root, dataset=dataset, name="confirmation", **directories)
    result = c.run_confirmation(**args)
    verified = verify_complete(result)
    assert len(calls) == 3
    assert read_json(result / "evaluation-0001/decision.json")["by_market"]["goals"]["status"] == "inconclusive"
    monkeypatch.setattr(cd, "checked_read", lambda *a: pytest.fail("completed resume must not reopen protected stores"))
    assert c.run_confirmation(**args, resume=True) == result
    assert verify_complete(result) == verified and len(calls) == 3
    with pytest.raises(ValueError, match="already consumed"):
        c.run_confirmation(**{**args, "name": "another"})
