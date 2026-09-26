"""Controlled prior-policy rebuilds, temporal isolation and variant contracts."""
from __future__ import annotations

import builtins
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from Scripts.data_platform.features.benchmarks import variants as v
from Scripts.data_platform.features.benchmarks.estimators import INTERACTION_BASES
from Scripts.data_platform.features import phase3_features as view
from Scripts.rag_ingest.core import model_features as core

COMPETITIONS = {"EPL": "domestic_league", "LaLiga": "domestic_league", "UCL": "continental_cup"}
BOUNDARY = "2024-01-01T00:00:00+00:00"


@pytest.fixture(autouse=True)
def no_network_or_live_profiles(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("chromadb", "requests", "httpx")) or name in {
            "rag_cli_v2", "core.projections", "Scripts.rag_ingest.core.projections",
            "core.team_resolution", "Scripts.rag_ingest.core.team_resolution",
        }:
            raise AssertionError("Live dependency requested by offline variant")
        return original(name, *args, **kwargs)

    def blocked(*args, **kwargs):
        raise AssertionError("Network requested by offline variant")

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(socket.socket, "connect", blocked)


def fixture(fid, season=2023, day=1, *, competition="EPL", home=1, away=2, goals=2):
    kickoff = datetime(season, 8, 1, 12, tzinfo=timezone.utc) + timedelta(days=day)
    return {"fixture_id": fid, "competition": competition, "season": season,
            "home_team_id": home, "away_team_id": away, "kickoff": kickoff.isoformat(), "status": "FT",
            "referee": "Historic official", "observed_at": (kickoff + timedelta(hours=4)).isoformat(),
            "home": {"goals": goals, "corners": 6., "sot": 5., "xg": 1.7, "fouls": 10., "shots": 12., "possession": 55., "cards": None},
            "away": {"goals": 1., "corners": 4., "sot": 3., "xg": .9, "fouls": 11., "shots": 8., "possession": 45., "cards": None}}


def request(target, history):
    identity = {k: value for k, value in target.items() if k not in {"home", "away"}}
    snapshot, features, masked = view.build_reference(identity, history, COMPETITIONS)
    decisions = {market: {"eligible": True, "reasons": [], "primary_reason": None} for market in v.MARKETS}
    decisions["cards"] = {"eligible": False, "reasons": ["cards_target_not_qualified"], "primary_reason": "cards_target_not_qualified"}
    return {"fixture": identity, "as_of": snapshot["as_of"], "snapshot_id": snapshot["snapshot_id"],
            "availability": "assumed_final", "partition": "development", **masked,
            "market_eligibility": view.apply_support(decisions, masked["support"])}, view.feature_contract(features["names"])


def build(target, history, variants=v.PROFILE_VARIANTS):
    row, schema = request(target, history)
    result = v.build_variant_rows([row], history, COMPETITIONS, confirmation_start=BOUNDARY,
                                  schema=schema, variants=variants)
    return row, schema, {r["variant"]: r for r in result["rows"]}, result


def values(row, schema):
    return dict(zip(schema["names"], row["values"]))


def test_longer_prior_uses_equal_fixture_weights_not_equal_season_weights():
    history = [fixture(i, 2021, i, goals=8) for i in range(1, 4)]
    history += [fixture(4, 2022, goals=2), fixture(5, goals=0)]
    original, schema, rows, _ = build(fixture(100, day=20), history)
    reference, three = values(original, schema), values(rows["profile_3"], schema)
    assert reference["home_goals_for_pm"] == pytest.approx(8 / 9 * 2)
    assert three["home_goals_for_pm"] == pytest.approx(8 / 9 * ((3 * 8 + 2) / 4))
    assert three["home_prior_matches"] == 4
    assert three["home_prior_weight"] == pytest.approx(8 / 9)
    assert rows["profile_3"]["evidence"]["profile_audit"]["home"]["primary"]["additional_prior_fixture_ids"] == [1, 2, 3]
    assert rows["profile_3"]["support"]["goals"]["home"]["fixture_ids"] == [1, 2, 3, 4, 5]


def test_three_and_five_windows_do_not_use_still_older_seasons():
    history = [fixture(1, 2018, goals=100), fixture(2, 2019, goals=20), fixture(3, 2020, goals=10),
               fixture(4, 2021, goals=8), fixture(5, 2022, goals=2), fixture(6, goals=0)]
    _, schema, rows, _ = build(fixture(100, day=20), history)
    assert values(rows["profile_3"], schema)["home_goals_for_pm"] == pytest.approx(8 / 9 * 5)
    assert values(rows["profile_5"], schema)["home_goals_for_pm"] == pytest.approx(8 / 9 * 10)
    assert 1 not in rows["profile_5"]["evidence"]["profile_source_fixture_ids"]
    assert rows["profile_5"]["evidence"]["profile_source_fixture_ids"] == [2, 3, 4, 5, 6]


def test_longer_variants_preserve_reference_recent_elo_referee_rest_and_missingness():
    history = [fixture(1, 2021, goals=20), fixture(2, 2022, goals=1), fixture(3, goals=0)]
    original, schema, rows, result = build(fixture(100, day=20), history)
    base = values(original, schema)
    for variant in ("profile_3", "profile_5"):
        changed = values(rows[variant], schema)
        for name in schema["names"]:
            if ("_last_5" in name or "_elo_" in name or "rest_days" in name or name.startswith("referee_")
                    or name.startswith("competition_") or "continental_matches" in name):
                assert changed[name] == base[name], name
        assert result["contracts"][variant]["policy"]["recent_features"].startswith("unchanged_reference")
        assert rows[variant]["source_snapshot_id"] == original["snapshot_id"]
        assert rows[variant]["snapshot_id"] != original["snapshot_id"]
        assert rows[variant]["feature_contract_id"] != original["feature_contract_id"]


def test_mature_longer_profiles_replay_exact_reference_vector_including_prior_count():
    history = [fixture(1, 2020, goals=80), fixture(2, 2021, goals=50), fixture(3, 2022, goals=20)]
    history += [fixture(i, day=i, goals=1) for i in range(4, 12)]
    original, schema, rows, _ = build(fixture(100, day=25), history)
    for variant in ("profile_3", "profile_5"):
        assert rows[variant]["values"] == original["values"]
        assert rows[variant]["evidence"]["features_changed"] is False
        assert values(rows[variant], schema)["home_prior_matches"] == 1
        assert not rows[variant]["evidence"]["profile_audit"]["home"]["primary"]["prior_fixture_ids"]


def test_reference_equality_when_no_additional_profile_history_exists():
    history = [fixture(i, 2022, i) for i in range(1, 7)] + [fixture(7, goals=0)]
    original, _, rows, _ = build(fixture(100, day=20), history)
    assert rows["profile_3"]["values"] == original["values"]
    assert rows["profile_5"]["values"] == original["values"]
    assert rows["profile_3"]["support"] == original["support"]


def test_prior_off_rebuilds_all_profile_rates_recent_and_summaries_from_current_only():
    history = [fixture(i, 2022, i, goals=10) for i in range(1, 7)]
    history.append(fixture(7, goals=0))
    original, schema, rows, _ = build(fixture(100, day=20), history)
    off = values(rows["prior_off"], schema)
    assert values(original, schema)["home_goals_for_pm"] > 0
    assert off["home_goals_for_pm"] == 0
    assert off["home_goals_last_5"] == 0
    assert off["home_prior_weight"] == 0
    assert off["home_prior_matches"] == 0
    assert off["home_goals_for_pm__missing"] == 0
    assert rows["prior_off"]["support"]["goals"]["home"]["fixture_ids"] == [7]
    assert rows["prior_off"]["evidence"]["profile_source_fixture_ids"] == [7]
    assert rows["prior_off"]["common_eligible"]["goals"] is False
    assert "variant:insufficient_home_history" in rows["prior_off"]["common_exclusion_reasons"]["goals"]


def test_prior_off_without_current_history_keeps_null_distinct_from_zero():
    history = [fixture(i, 2022, i, goals=0) for i in range(1, 7)]
    original, schema, rows, _ = build(fixture(100, day=20), history)
    assert values(original, schema)["home_goals_for_pm"] == 0
    off = values(rows["prior_off"], schema)
    assert off["home_goals_for_pm"] is None
    assert off["home_goals_for_pm__missing"] == 1
    assert off["home_goals_last_5"] is None
    assert off["home_goals_last_5__missing"] == 1
    assert "missing_home_production_rate" in rows["prior_off"]["market_eligibility"]["goals"]["reasons"]


def test_missing_prior_observations_do_not_count_as_zero_or_support():
    history = [fixture(i, 2021, i, goals=None) for i in range(1, 5)]
    history += [fixture(5, 2022, goals=2), fixture(6, goals=0)]
    _, schema, rows, _ = build(fixture(100, day=20), history)
    three = rows["profile_3"]
    assert values(three, schema)["home_goals_for_pm"] == pytest.approx(8 / 9 * 2)
    assert three["support"]["goals"]["home"]["count"] == 2
    assert three["support"]["goals"]["home"]["fixture_ids"] == [5, 6]
    assert not three["market_eligibility"]["goals"]["eligible"]


def test_new_variant_support_does_not_add_reference_ineligible_fixtures_to_comparison():
    history = [fixture(i, 2021, i) for i in range(1, 5)] + [fixture(5, 2022), fixture(6)]
    original, _, rows, result = build(fixture(100, day=20), history)
    assert original["market_eligibility"]["goals"]["eligible"] is False
    assert rows["profile_3"]["market_eligibility"]["goals"]["eligible"] is True
    assert rows["profile_3"]["common_eligible"]["goals"] is False
    entry = next(r for r in result["coverage"]["slices"] if
                 (r["variant"], r["market"], r["competition"], r["season_stage"]) == ("profile_3", "goals", "all", "all"))
    assert entry["variant_only"] == 1
    assert entry["common_eligible"] == 0
    assert any(r["competition"] == "LaLiga" and r["rows"] == 0 for r in result["coverage"]["slices"])


def test_structural_evidence_exclusions_survive_new_support():
    history = [fixture(i, 2022, i) for i in range(1, 7)]
    row, schema = request(fixture(100, day=20), history)
    row["market_eligibility"]["goals"] = {"eligible": False, "reasons": ["unverified_period"], "primary_reason": "unverified_period"}
    result = v.build_variant_rows([row], history, COMPETITIONS, confirmation_start=BOUNDARY, schema=schema)
    assert all("unverified_period" in r["market_eligibility"]["goals"]["reasons"] for r in result["rows"])
    assert all(not r["common_eligible"]["goals"] for r in result["rows"])


def test_european_profile_keeps_exact_current_domestic_identity_and_blend():
    history = [fixture(1, 2021, competition="LaLiga", goals=100), fixture(2, 2021, goals=20),
               fixture(3, 2022, goals=10), fixture(4, goals=0)]
    history += [fixture(i, day=i, competition="UCL", goals=2) for i in range(5, 8)]
    original, schema, rows, _ = build(fixture(100, day=20, competition="UCL"), history)
    audit = rows["profile_3"]["evidence"]["profile_audit"]["home"]
    assert audit["mode"] == "domestic_continental_blend"
    assert audit["primary"]["competition"] == "EPL"
    assert 1 not in audit["primary"]["source_fixture_ids"]
    assert values(rows["profile_3"], schema)["home_goals_for_pm"] == pytest.approx(.8 * (8 / 9 * 15) + .2 * 2)
    assert values(rows["profile_3"], schema)["home_goals_last_5"] == values(original, schema)["home_goals_last_5"]


def test_future_results_exact_three_hour_boundary_and_target_result_never_enter():
    history = [fixture(1, 2021), fixture(2, 2022), fixture(3)]
    target = fixture(100, day=20)
    row, schema = request(target, history)
    expected = v.build_variant_rows([row], history, COMPETITIONS, confirmation_start=BOUNDARY, schema=schema)
    exact = fixture(200, day=20)
    exact["kickoff"] = (core.utc(target["kickoff"]) - timedelta(hours=3)).isoformat()
    future = fixture(201, day=21)
    locked = fixture(202, season=2024)
    locked["home"] = object()  # Exclude before any statistics are inspected.
    actual = v.build_variant_rows([row], history + [exact, future, locked, target], COMPETITIONS,
                                   confirmation_start=BOUNDARY, schema=schema)
    assert actual == expected


def test_rows_hashes_and_membership_are_order_deterministic_without_input_mutation():
    history = [fixture(1, 2021), fixture(2, 2022), fixture(3)]
    first, schema = request(fixture(100, day=20), history)
    second, _ = request(fixture(101, day=21), history)
    before = deepcopy((history, first, second, schema))
    result = v.build_variant_rows([first, second], history, COMPETITIONS, confirmation_start=BOUNDARY, schema=schema)
    reversed_result = v.build_variant_rows([second, first], history[::-1], COMPETITIONS, confirmation_start=BOUNDARY, schema=schema)
    assert result == reversed_result
    assert core.digest(result) == core.digest(reversed_result)
    assert (history, first, second, schema) == before


@pytest.mark.parametrize("kind", ["snapshot", "values", "support"])
def test_reference_replay_disagreement_fails_closed(kind):
    history = [fixture(1, 2022)]
    row, schema = request(fixture(100, day=20), history)
    if kind == "snapshot":
        row["snapshot_id"] = "0" * 64
    elif kind == "values":
        row["values"][0] += 1
    else:
        row["support"]["goals"]["home"]["count"] += 1
    with pytest.raises(ValueError, match="replay mismatch"):
        v.build_variant_rows([row], history, COMPETITIONS, confirmation_start=BOUNDARY, schema=schema)


@pytest.mark.parametrize("partition", ["phase3_confirmation", "calibration", "final_system_test", "prospective_reserve"])
def test_heldout_requests_fail_before_history_is_read(partition):
    row, schema = request(fixture(100, day=20), [])
    row["partition"] = partition

    class Forbidden:
        def __iter__(self):
            raise AssertionError("Read history for held-out request")

    with pytest.raises(ValueError, match="held-out"):
        v.build_variant_rows([row], Forbidden(), COMPETITIONS, confirmation_start=BOUNDARY, schema=schema)


def test_post_confirmation_fixture_cannot_be_relabelled_development():
    row, schema = request(fixture(100, season=2024), [])
    with pytest.raises(ValueError, match="held-out"):
        v.build_variant_rows([row], [], COMPETITIONS, confirmation_start=BOUNDARY, schema=schema)


@pytest.mark.parametrize("variant", v.DROP_VARIANTS)
def test_drop_groups_preserve_pairs_unrelated_nulls_and_linear_interaction_bases(variant):
    row, schema = request(fixture(100, day=20), [])
    before = deepcopy(row)
    transformed, names, contract = v.transform_features([row], schema["names"], variant)
    assert len(names) < len(schema["names"])
    assert row == before
    by_name = values(transformed[0], {"names": names})
    for name in names:
        if name.endswith("__missing"):
            assert name[:-9] in names
            assert by_name[name] == float(by_name[name[:-9]] is None)
    assert {n for fields in INTERACTION_BASES.values() for n in fields}.issubset(names)
    assert transformed[0]["feature_contract_id"] == core.digest(contract)
    assert contract["candidate_selection_allowed"] is False
    assert transformed[0]["snapshot_id"] != row["snapshot_id"]


def test_drop_masks_do_not_fit_on_other_rows_and_missingness_mismatch_is_rejected():
    row, schema = request(fixture(100, day=20), [])
    later = deepcopy(row)
    later["snapshot_id"] = "1" * 64
    i = schema["names"].index("home_goals_for_pm")
    j = schema["names"].index("home_goals_for_pm__missing")
    later["values"][i], later["values"][j] = 100000., 0.
    single = v.transform_features([row], schema["names"], "drop_elo")
    together = v.transform_features([row, later], schema["names"], "drop_elo")
    assert single[0][0] == together[0][0]
    assert single[2] == together[2]
    row["values"][j] = 0
    with pytest.raises(ValueError, match="missingness"):
        v.transform_features([row], schema["names"], "drop_elo")


def test_stage_uses_less_experienced_team_and_keeps_individual_counts():
    history = [fixture(i, day=i, home=1, away=10 + i) for i in range(1, 10)]
    history += [fixture(30, day=10, home=2, away=99)]
    _, _, rows, _ = build(fixture(100, day=20), history)
    stage = rows["profile_3"]["season_stage"]
    assert stage["home_current_matches"] == 9
    assert stage["away_current_matches"] == 1
    assert stage["home_band"] == "8-19"
    assert stage["fixture_band"] == "0-7"


def test_prepare_and_load_sidecar_are_immutable_and_reader_never_opens_audit(tmp_path, monkeypatch):
    from Scripts.data_platform.features.benchmarks import artifacts, data as data_module

    history = [fixture(1, 2021), fixture(2, 2022), fixture(3)]
    target = fixture(100, day=20)
    row, schema = request(target, history)
    dataset = tmp_path / "Index/prediction_experiments/dataset"
    (dataset / "audit").mkdir(parents=True)
    audit = dataset / "audit/inputs.json"
    audit.write_text(json.dumps({"fixtures": [target, fixture(101, season=2024)], "history": history,
                                 "competitions": COMPETITIONS}))
    artifacts.write_json(dataset / "COMPLETE.json", {"audit/inputs.json": artifacts.sha(audit)})
    fake = SimpleNamespace(rows=[row], schema=schema,
                           manifest={"dataset_id": "dataset-test", "feature_contract_id": core.digest(schema)},
                           splits={"boundaries": {"phase3_confirmation_start": BOUNDARY}},
                           memberships={r["fixture_id"]: r for r in (*history, target)})
    monkeypatch.setattr(data_module, "DevelopmentDataset", lambda path: fake)
    monkeypatch.setattr(artifacts, "source_hashes", lambda: {"frozen": "hash"})
    output = v.prepare_variants(root=tmp_path, dataset=dataset, name="variants-test")
    assert (output / "COMPLETE.json").is_file()
    with pytest.raises(FileExistsError):
        v.prepare_variants(root=tmp_path, dataset=dataset, name="variants-test")
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if "audit" in path.parts:
            raise AssertionError("Ordinary variant reader opened audit")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    manifest, mapping = v.load_variants(output, fake)
    assert set(mapping) == {(100, variant) for variant in v.PROFILE_VARIANTS}
    assert manifest["training_performed"] is False
    assert manifest["candidate_selection_allowed"] is False
    assert set(artifacts.read_json(output / "COMPLETE.json")) == {
        "preparation.json", "manifest.json", "features.jsonl", "coverage.json"}


@pytest.mark.parametrize("corruption", [
    "count", "duplicate_support_ids", "duplicate_primary_ids", "boolean_primary_id",
    "rate_flag", "numeric_rate_flag", "missing_feature", "unbound_support_id",
    "future_source", "exact_availability_boundary", "source_outside_window", "common_eligibility",
])
def test_reader_rejects_semantic_corruption_even_after_row_and_file_hashes_are_resealed(tmp_path, corruption):
    from Scripts.data_platform.features.benchmarks import artifacts

    history = [fixture(i, day=i) for i in range(1, 7)] + [fixture(7, 2022), fixture(8, 2021)]
    target = fixture(100, day=30)
    original, schema, _, result = build(target, history)
    future = fixture(101, day=40)
    exact = fixture(102, day=30)
    exact["kickoff"] = (core.utc(target["kickoff"]) - timedelta(hours=3)).isoformat()
    too_old = fixture(103, 2018)
    fake = SimpleNamespace(rows=[original], schema=schema,
                           manifest={"dataset_id": "dataset-test", "feature_contract_id": core.digest(schema)},
                           splits={"boundaries": {"phase3_confirmation_start": BOUNDARY}},
                           memberships={r["fixture_id"]: r for r in (*history, target, future, exact, too_old)})
    changed = result["rows"][0]
    support = changed["support"]["goals"]["home"]
    if corruption == "count":
        support["count"] += 1
    elif corruption == "duplicate_support_ids":
        support["fixture_ids"].append(support["fixture_ids"][0])
        support["count"] = len(support["fixture_ids"])
    elif corruption == "duplicate_primary_ids":
        support["primary_fixture_ids"].append(support["primary_fixture_ids"][0])
    elif corruption == "boolean_primary_id":
        support["primary_fixture_ids"][0] = True
    elif corruption == "rate_flag":
        support["rate_available"] = False
    elif corruption == "numeric_rate_flag":
        support["rate_available"] = 1
    elif corruption == "missing_feature":
        changed["values"][schema["names"].index("home_goals_for_pm")] = None
        changed["values"][schema["names"].index("home_goals_for_pm__missing")] = 1.0
    elif corruption == "unbound_support_id":
        changed["evidence"]["profile_source_fixture_ids"].remove(support["fixture_ids"][0])
    elif corruption in {"future_source", "exact_availability_boundary", "source_outside_window"}:
        new_id = {"future_source": 101, "exact_availability_boundary": 102, "source_outside_window": 103}[corruption]
        changed["evidence"]["profile_source_fixture_ids"].append(new_id)
        changed["evidence"]["profile_source_fixture_ids"].sort()
        support["fixture_ids"][0] = new_id
    else:
        changed["common_eligible"]["goals"] = False
    changed["row_sha256"] = core.digest({k: item for k, item in changed.items() if k != "row_sha256"})
    metadata = {"version": v.VERSION, "scope": "development_only", "dataset_id": "dataset-test",
                "source_feature_contract_id": core.digest(schema), "confirmation_start": BOUNDARY,
                "variants": list(v.PROFILE_VARIANTS), "contracts": result["contracts"],
                "diagnostic_only": True, "candidate_selection_allowed": False, "training_performed": False,
                "publication_enabled": False, "promotion_allowed": False}
    artifacts.write_json(tmp_path / "preparation.json", metadata)
    artifacts.write_json(tmp_path / "manifest.json", {**metadata, "rows": len(result["rows"])})
    artifacts.write_json(tmp_path / "coverage.json", result["coverage"])
    with (tmp_path / "features.jsonl").open("x") as handle:
        for row in result["rows"]:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    artifacts.complete(tmp_path)
    assert artifacts.verify_complete(tmp_path)  # All cryptographic checks pass.
    with pytest.raises(ValueError, match="support/feature/source|source fixture|common cohort"):
        v.load_variants(tmp_path, fake)
