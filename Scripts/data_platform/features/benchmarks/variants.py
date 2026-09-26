"""Versioned, development-only feature diagnostics for Phase 3 Batch C.

Longer profiles widen only the early-season prior pool, not the live feature
policy or the main selected candidate. Their unchanged eight-match cutoff
means that they are not continuous three-/five-year production averages.
The separately invoked preparation operation may read frozen audit history;
``load_variants`` and the ordinary trainer read only its immutable sidecar.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

from Scripts.rag_ingest.core import model_features as core
from Scripts.data_platform.features import phase3_features as view

VERSION = "phase3-feature-variants.v1"
SNAPSHOT_VERSION = "phase3-profile-variant-input.v1"
PROFILE_VARIANTS = ("profile_3", "profile_5", "prior_off")
DROP_VARIANTS = ("drop_elo", "drop_referee", "drop_venue", "drop_recent")
VARIANTS = (*PROFILE_VARIANTS, *DROP_VARIANTS)
MARKETS = view.MARKETS
STAGES = ("0-7", "8-19", "20+")
_SUPPORT_REASONS = frozenset(f"{prefix}_{side}_{suffix}" for side in ("home", "away")
                           for prefix, suffix in (("missing", "production_rate"), ("insufficient", "history")))


def _validate_names(names):
    if not names or len(set(names)) != len(names):
        raise ValueError("Variant schema requires unique ordered feature names")
    positions = set(names)
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid variant feature name")
        if name.endswith("__missing"):
            if name[:-9] not in positions:
                raise ValueError("Variant schema has orphan missingness indicator")
        elif name + "__missing" not in positions:
            raise ValueError("Variant schema lacks explicit missingness indicator")


def _validate_values(values, names):
    if not isinstance(values, list) or len(values) != len(names):
        raise ValueError("Variant feature vector length mismatch")
    mapping = dict(zip(names, values))
    for name, value in mapping.items():
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError("Variant feature vector contains invalid numeric value")
        if name.endswith("__missing") and value != float(mapping[name[:-9]] is None):
            raise ValueError("Variant missingness indicator disagrees with its value")


def variant_contract(schema, variant):
    """Freeze profile weighting and controlled features before any fitting."""
    if variant not in PROFILE_VARIANTS:
        raise ValueError("Unknown rebuilt profile variant")
    _validate_names(schema["names"])
    if schema["policy"] != vars(core.FeaturePolicy()):
        raise ValueError("Unsupported reference feature policy")
    window = {"profile_3": 3, "profile_5": 5, "prior_off": 1}[variant]
    return {
        "version": VERSION, "variant": variant,
        "source_feature_contract_id": core.digest(schema), "names": list(schema["names"]),
        "source_policy": deepcopy(schema["policy"]),
        "policy": {
            "provider_seasons": window,
            "prior_provider_seasons": window - 1,
            "prior_pool": "equal_observed_fixture_weight_per_statistic_ignoring_nulls",
            "prior_strength": 0.0 if variant == "prior_off" else 8.0,
            "prior_rule": "disabled" if variant == "prior_off" else
                          "one_if_no_current_else_8_over_n_plus_8_until_n_reaches_8",
            "current_profile": "current_provider_season_only",
            "recent_features": "current_only_last_five_rebuilt" if variant == "prior_off" else
                               "unchanged_reference_current_previous_last_five_blend",
            "prior_summary": "zero" if variant == "prior_off" else
                             "pooled_count_if_prior_weight_positive_else_reference_count",
            "unrelated_features": "reference_Elo_rest_referee_competition_and_domestic_identity_unchanged",
            "continental_blend": "unchanged_reference_domestic_identity_and_80_20_three_current_euro_match_gate",
            "availability": "assumed_final_strictly_kickoff_plus_3h_before_forecast",
            "support": "at_least_five_known_positive_weight_own_market_observations_per_team",
            "comparison": "paired_reference_eligible_intersection_only",
        },
        "season_stage": "minimum_of_home_and_away_primary_current_appearances_bands_0_7_8_19_20_plus",
        "limitation": "longer_profiles_change_early_season_priors_only_not_continuous_long_term_averages",
        "diagnostic_only": True, "candidate_selection_allowed": False,
        "publication_enabled": False, "promotion_allowed": False,
    }


def _drop_name(name, variant):
    base = name[:-9] if name.endswith("__missing") else name
    if variant == "drop_elo":
        return "_elo_" in base
    if variant == "drop_referee":
        return base.startswith("referee_")
    if variant == "drop_venue":
        return base.endswith("_venue_pm")
    if variant == "drop_recent":
        return base.endswith("_last_5")
    raise ValueError("Unknown drop-group variant")


def transform_features(rows, names, variant):
    """Drop a declared group and its indicators without fitting or imputation.

    Returns transformed rows, ordered names and a versioned contract. Production
    rates used by declared linear interactions are retained by all four masks.
    """
    if variant not in DROP_VARIANTS:
        raise ValueError("Mask transformation requires a declared drop variant")
    names, rows = list(names), list(rows)
    _validate_names(names)
    source_ids = {row.get("feature_contract_id") for row in rows}
    if not rows or len(source_ids) != 1 or None in source_ids:
        raise ValueError("Mask transformation requires one nonempty source feature contract")
    retained = [i for i, name in enumerate(names) if not _drop_name(name, variant)]
    selected = [names[i] for i in retained]
    removed = [name for name in names if _drop_name(name, variant)]
    if not removed:
        raise ValueError("Declared ablation group has no features")
    _validate_names(selected)
    contract = {"version": VERSION, "variant": variant, "source_feature_contract_id": next(iter(source_ids)),
                "source_names": names, "names": selected, "removed_names": removed,
                "transformation": "drop_columns_and_paired_missing_indicators_no_fit",
                "diagnostic_only": True, "candidate_selection_allowed": False,
                "publication_enabled": False, "promotion_allowed": False}
    contract_id = core.digest(contract)
    transformed = []
    for row in rows:
        _validate_values(row["values"], names)
        if not row.get("snapshot_id"):
            raise ValueError("Mask transformation requires source snapshot identity")
        values = [row["values"][i] for i in retained]
        _validate_values(values, selected)
        transformed.append({**row, "source_snapshot_id": row["snapshot_id"],
                            "source_feature_contract_id": row["feature_contract_id"],
                            "feature_contract_id": contract_id, "variant": variant,
                            "snapshot_id": core.digest({"reference_snapshot_id": row["snapshot_id"],
                                                        "feature_contract_id": contract_id}),
                            "values": values})
    return transformed, selected, contract


def _available(row):
    return core.utc(row["kickoff"]) + timedelta(hours=3)


def _older_rows(index, fixture, cutoff, competitions):
    """Capture only the target teams' additional three/five-season inputs."""
    found = {}
    for season in range(fixture["season"] - 4, fixture["season"] - 1):
        for side in ("home", "away"):
            rows = index.by_team.get((fixture[f"{side}_team_id"], season), [])
            end = bisect_left(index.times.get(id(rows), []), cutoff)
            for raw in rows[:end]:
                if raw["fixture_id"] == fixture["fixture_id"] or _available(raw) >= cutoff or raw.get("status") != "FT":
                    continue
                row = core.fixture_identity(raw, competitions)
                row.update(status="FT", observed_at=core.utc(raw["observed_at"]).isoformat() if raw.get("observed_at") else None,
                           referee=raw.get("referee") or None)
                for venue in ("home", "away"):
                    row[venue] = {key: core.number((raw.get(venue) or {}).get(key))
                                  for key in (*core.MARKETS, "shots", "fouls", "xg", "possession")}
                found[row["fixture_id"]] = row
    return sorted(found.values(), key=lambda r: (r["kickoff"], r["fixture_id"]))


def _component(history, team, competition, season, venue, reference_audit, variant, policy):
    rows = [row for row in history if row["competition"] == competition
            and team in (row["home_team_id"], row["away_team_id"])]
    current = [row for row in rows if row["season"] == season]
    window = {"profile_3": 3, "profile_5": 5, "prior_off": 1}[variant]
    # Mature profiles never inspect/use an older prior statistic. This keeps
    # their numerical vectors identical to the reference (including summaries).
    prior = ([row for row in rows if season - window < row["season"] < season]
             if variant != "prior_off" and len(current) < policy.prior_strength else [])
    current_values = core._aggregates(current, team, venue, policy)
    prior_values = core._aggregates(prior, team, venue, policy)
    n = len(current)
    weight = ((1.0 if n == 0 else policy.prior_strength / (n + policy.prior_strength)) if prior else 0.0)
    values = {}
    for key, value in current_values.items():
        old = prior_values[key]
        values[key] = ((1 - weight) * value + weight * old if value is not None and old is not None and weight
                       else old if value is None and weight else value)
    summary_count = (0 if variant == "prior_off" else len(prior) if weight else reference_audit["prior_matches"])
    audit = {"competition": competition, "current_matches": n, "prior_matches": summary_count,
             "prior_weight": weight, "prior_pool_matches_used": len(prior),
             "current_fixture_ids": [r["fixture_id"] for r in current],
             "prior_fixture_ids": [r["fixture_id"] for r in prior],
             "source_fixture_ids": [r["fixture_id"] for r in (*current, *prior)],
             "additional_prior_fixture_ids": [r["fixture_id"] for r in prior if r["season"] < season - 1]}
    return values, audit


def _profiles(snapshot, reference, extended_history, variant):
    target, policy = snapshot["fixture"], core.FeaturePolicy(**snapshot["policy"])
    profiles, audits = {}, {}
    for side in ("home", "away"):
        team, original = target[f"{side}_team_id"], reference["profile_audit"][side]
        primary, primary_audit = _component(extended_history, team, original["primary"]["competition"],
                                            target["season"], side, original["primary"], variant, policy)
        continental_audit = None
        if original["continental"] is not None:
            continental, continental_audit = _component(extended_history, team, original["continental"]["competition"],
                                                        target["season"], side, original["continental"], variant, policy)
            if original["mode"] == "domestic_continental_blend":
                for key, value in primary.items():
                    other = continental[key]
                    if value is not None and other is not None:
                        primary[key] = policy.domestic_weight * value + (1 - policy.domestic_weight) * other
        if variant != "prior_off":
            for key, value in reference["profiles"][side].items():
                if key.endswith(f"_last_{policy.recent_matches}"):
                    primary[key] = value
        profiles[side] = primary
        audits[side] = {"mode": original["mode"], "primary": primary_audit, "continental": continental_audit,
                        "domestic_candidates": original["domestic_candidates"]}
    return profiles, audits


def _support(history, fixture, profiles, audits, policy):
    lookup = {r["fixture_id"]: r for r in history}

    def known_ids(component, team, market):
        if component is None:
            return set()
        ids = list(component["current_fixture_ids"])
        if component["prior_weight"] > 0:
            ids += component["prior_fixture_ids"]
        return {fid for fid in ids if core.number(lookup[fid]["home" if lookup[fid]["home_team_id"] == team else "away"][market]) is not None}

    result = {}
    for market in MARKETS:
        result[market] = {}
        for side in ("home", "away"):
            team, audit = fixture[f"{side}_team_id"], audits[side]
            ids = known_ids(audit["primary"], team, market)
            primary_ids = set(ids)
            if audit["mode"] == "domestic_continental_blend":
                continental = known_ids(audit["continental"], team, market)
                if ids and continental:
                    ids = (ids if policy.domestic_weight > 0 else set()) | (continental if policy.domestic_weight < 1 else set())
            rate = profiles[side][core.RATE_FIELDS[market]]
            count = len(ids) if rate is not None else 0
            result[market][side] = {"count": count, "fixture_ids": sorted(ids) if count else [],
                                    "primary_fixture_ids": sorted(primary_ids), "rate_available": rate is not None,
                                    "band": "0" if count == 0 else "1-4" if count < 5 else "5-9" if count < 10 else "10+"}
    return result


def _band(n):
    return "0-7" if n < 8 else "8-19" if n < 20 else "20+"


def _stage(audits):
    counts = {side: audits[side]["primary"]["current_matches"] for side in ("home", "away")}
    return {"home_current_matches": counts["home"], "away_current_matches": counts["away"],
            "home_band": _band(counts["home"]), "away_band": _band(counts["away"]),
            "fixture_band": _band(min(counts.values())), "rule": "minimum_primary_current_appearances"}


def _decisions(reference_decisions, support):
    raw = {market: {**reference_decisions[market],
                    "reasons": [reason for reason in reference_decisions[market]["reasons"] if reason not in _SUPPORT_REASONS]}
           for market in MARKETS}
    return view.apply_support(raw, support)


def _coverage(rows, originals, variants, competitions):
    buckets = {}
    for variant in variants:
        for market in MARKETS:
            for league in ("all", *sorted(competitions)):
                for stage in ("all", *STAGES):
                    buckets[(variant, market, league, stage)] = Counter(
                        rows=0, reference_eligible=0, variant_eligible=0, common_eligible=0,
                        excluded_from_common=0, reference_only=0, variant_only=0,
                        changed_vectors=0, extra_history_contributes=0)
    exclusions = Counter()
    for row in rows:
        original = originals[row["fixture_id"]]
        for market in MARKETS:
            ref = original["market_eligibility"][market]["eligible"]
            eligible = row["market_eligibility"][market]["eligible"]
            common = ref and eligible
            extra = any(set(row["support"][market][side]["fixture_ids"]) - set(original["support"][market][side]["fixture_ids"])
                        for side in ("home", "away"))
            for league in ("all", original["fixture"]["competition"]):
                for stage in ("all", row["season_stage"]["fixture_band"]):
                    buckets[(row["variant"], market, league, stage)].update(
                        rows=1, reference_eligible=int(ref), variant_eligible=int(eligible), common_eligible=int(common),
                        excluded_from_common=int(not common), reference_only=int(ref and not eligible),
                        variant_only=int(eligible and not ref), changed_vectors=int(row["evidence"]["features_changed"]),
                        extra_history_contributes=int(extra))
            for reason in row["common_exclusion_reasons"][market]:
                exclusions[(row["variant"], market, reason)] += 1
    return {"version": VERSION, "comparison": "each_variant_versus_reference_on_its_paired_intersection",
            "season_stage_rule": "minimum_primary_current_appearances",
            "slices": [{"variant": v, "market": m, "competition": c, "season_stage": s, **dict(counts)}
                       for (v, m, c, s), counts in sorted(buckets.items())],
            "exclusions": [{"variant": v, "market": m, "reason": r, "count": n}
                           for (v, m, r), n in sorted(exclusions.items())]}


def build_variant_rows(development_rows, history, competitions, *, confirmation_start, schema,
                       variants=PROFILE_VARIANTS):
    """Rebuild development variants from evidence-filtered, frozen input history.

    Reference snapshots/vectors/support replay first. Held-out forecasts are
    rejected before history access; later source rows are removed before their
    statistics are inspected. No target value is read to construct a feature.
    """
    variants = tuple(variants)
    if not variants or len(set(variants)) != len(variants) or not set(variants).issubset(PROFILE_VARIANTS):
        raise ValueError("Choose distinct rebuilt profile variants")
    contracts = {v: variant_contract(schema, v) for v in variants}
    contract_ids = {v: core.digest(contract) for v, contract in contracts.items()}
    source_contract = core.digest(schema)
    boundary = core.utc(confirmation_start)
    requested, originals = list(development_rows), {}
    for row in requested:
        fixture = row["fixture"]
        core.fixture_identity(fixture, competitions)
        if (row.get("partition") not in {"initial_training", "development"}
                or core.utc(fixture["kickoff"]) >= boundary or core.utc(row["as_of"]) >= boundary):
            raise ValueError("Variant preparation refuses held-out forecasts")
        if row.get("availability") != "assumed_final" or core.utc(row["as_of"]) > core.utc(fixture["kickoff"]):
            raise ValueError("Unsupported variant availability or prediction cutoff")
        if row.get("feature_contract_id") != source_contract or not row.get("snapshot_id"):
            raise ValueError("Variant source feature contract/snapshot mismatch")
        _validate_values(row["values"], schema["names"])
        fid = fixture["fixture_id"]
        if fid in originals:
            raise ValueError("Duplicate fixture version in variant preparation")
        originals[fid] = row
    safe = [r for r in history if r.get("status") == "FT" and _available(r) < boundary]
    index = view.HistoryIndex(safe)
    output = []
    for original in sorted(requested, key=lambda r: (core.utc(r["as_of"]), r["fixture"]["fixture_id"])):
        fixture, cutoff = original["fixture"], core.utc(original["as_of"])
        snapshot = core.capture_snapshot(fixture, index.candidates(fixture), as_of=cutoff,
                                         competitions=competitions, availability="assumed_final")
        if snapshot["snapshot_id"] != original["snapshot_id"]:
            raise ValueError(f"Variant reference snapshot replay mismatch: {fixture['fixture_id']}")
        reference = core.build_features(snapshot)
        actual_schema = view.feature_contract(reference["names"])
        source_values = dict(zip(reference["names"], reference["values"]))
        if actual_schema != schema or [source_values[n] for n in schema["names"]] != original["values"]:
            raise ValueError("Variant reference feature replay mismatch")
        if view.contributing_support(snapshot, reference) != original["support"]:
            raise ValueError("Variant reference support replay mismatch")
        reference_history_hash = core.digest(snapshot["history"])
        # No older rows are needed when every relevant component is mature.
        sparse = any(component and component["current_matches"] < 8
                     for audit in reference["profile_audit"].values()
                     for component in (audit["primary"], audit["continental"]))
        older = _older_rows(index, fixture, cutoff, competitions) if sparse and any(v != "prior_off" for v in variants) else []
        extended = sorted([*snapshot["history"], *older], key=lambda r: (r["kickoff"], r["fixture_id"]))
        for variant in variants:
            profiles, audits = _profiles(snapshot, reference, extended, variant)
            mapping = dict(source_values)
            for side in ("home", "away"):
                mapping.update({f"{side}_{key}": value for key, value in profiles[side].items()})
                for key in ("current_matches", "prior_matches", "prior_weight"):
                    mapping[f"{side}_{key}"] = audits[side]["primary"][key]
            for name in schema["names"]:
                if name.endswith("__missing"):
                    mapping[name] = float(mapping[name[:-9]] is None)
            values = [mapping[name] for name in schema["names"]]
            _validate_values(values, schema["names"])
            support = _support(extended, fixture, profiles, audits, core.FeaturePolicy(**snapshot["policy"]))
            decisions = _decisions(original["market_eligibility"], support)
            common = {m: original["market_eligibility"][m]["eligible"] and decisions[m]["eligible"] for m in MARKETS}
            exclusions = {m: [f"reference:{r}" for r in original["market_eligibility"][m]["reasons"]]
                              + [f"variant:{r}" for r in decisions[m]["reasons"]] for m in MARKETS}
            source_ids = sorted({fid for audit in audits.values() for component in (audit["primary"], audit["continental"])
                                 if component for fid in component["source_fixture_ids"]})
            source_set = set(source_ids)
            selected = [r for r in extended if r["fixture_id"] in source_set]
            history_hash = core.digest(selected)
            variant_snapshot = {"version": SNAPSHOT_VERSION, "reference_snapshot_id": original["snapshot_id"],
                                "feature_contract_id": contract_ids[variant], "profile_history_sha256": history_hash}
            result = {"fixture_id": fixture["fixture_id"], "variant": variant,
                      "source_snapshot_id": original["snapshot_id"], "source_feature_contract_id": source_contract,
                      "snapshot_id": core.digest(variant_snapshot), "snapshot_contract": variant_snapshot,
                      "feature_contract_id": contract_ids[variant], "values": values,
                      "support": support, "market_eligibility": decisions,
                      "reference_eligible": {m: original["market_eligibility"][m]["eligible"] for m in MARKETS},
                      "common_eligible": common,
                      "common_exclusion_reasons": exclusions, "season_stage": _stage(audits),
                      "evidence": {"as_of": cutoff.isoformat(), "availability": "assumed_final", "reference_replayed": True,
                                   "features_changed": values != original["values"], "profile_audit": audits,
                                   "profile_source_fixture_ids": source_ids, "profile_history_sha256": history_hash,
                                   "reference_history_sha256": reference_history_hash}}
            result["row_sha256"] = core.digest(result)
            output.append(result)
    return {"rows": output, "contracts": contracts,
            "coverage": _coverage(output, originals, variants, competitions)}


def prepare_variants(*, root, dataset, name, variants=PROFILE_VARIANTS):
    """Explicit audit-read preparation; never invoked by the ordinary trainer."""
    from .artifacts import (ROOT, ALLOWED_DATASET_FILES, complete, new_experiment, read_json,
                            sha, source_hashes, write_json)
    from .data import DevelopmentDataset
    from .isolation import offline_guard

    dataset = Path(dataset).resolve()
    data = DevelopmentDataset(dataset)
    checked = read_json(dataset / "COMPLETE.json")
    history_path = dataset / "audit/inputs.json"
    if history_path.resolve() != history_path or sha(history_path) != checked.get("audit/inputs.json"):
        raise ValueError("Frozen variant history checksum mismatch")
    end = data.splits["boundaries"]["phase3_confirmation_start"]
    contracts = {v: variant_contract(data.schema, v) for v in variants}
    hashes = source_hashes()
    target = new_experiment(root, name)
    metadata = {"version": VERSION, "dataset_id": data.manifest["dataset_id"],
                "source_feature_contract_id": data.manifest["feature_contract_id"],
                "scope": "development_only", "variants": list(variants), "contracts": contracts,
                "confirmation_start": end, "source_hashes": hashes,
                "source_audit_sha256": checked["audit/inputs.json"],
                "diagnostic_only": True, "candidate_selection_allowed": False, "training_performed": False,
                "publication_enabled": False, "promotion_allowed": False}
    write_json(target / "preparation.json", metadata)
    try:
        readable = [dataset / file for file in ALLOWED_DATASET_FILES] + [history_path]
        with offline_guard(root=root, output=target, readable_files=readable):
            inputs = read_json(history_path)
            history = [r for r in inputs["history"] if _available(r) < core.utc(end)]
            competitions = inputs["competitions"]
            del inputs
            result = build_variant_rows(data.rows, history, competitions, confirmation_start=end,
                                        schema=data.schema, variants=variants)
            with (target / "features.jsonl").open("x") as handle:
                for row in result["rows"]:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            write_json(target / "coverage.json", result["coverage"])
            write_json(target / "manifest.json", {**metadata, "rows": len(result["rows"]),
                                                   "created_at": datetime.now(timezone.utc).isoformat()})
        if source_hashes() != hashes:
            raise RuntimeError("Variant source changed during preparation; completion withheld")
        complete(target)
    except Exception as exc:
        write_json(target / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    return target


def load_variants(path, data):
    """Read and validate a development sidecar without opening audit history."""
    from .artifacts import read_json, verify_complete

    path = Path(path)
    files = verify_complete(path, required={"manifest.json", "preparation.json", "features.jsonl", "coverage.json"})
    if set(files) != {"manifest.json", "preparation.json", "features.jsonl", "coverage.json"}:
        raise ValueError("Unexpected variant sidecar files")
    manifest = read_json(path / "manifest.json")
    if (manifest.get("version") != VERSION or manifest.get("scope") != "development_only"
            or manifest.get("dataset_id") != data.manifest["dataset_id"]
            or manifest.get("source_feature_contract_id") != data.manifest["feature_contract_id"]
            or manifest.get("publication_enabled") is not False or manifest.get("promotion_allowed") is not False
            or manifest.get("candidate_selection_allowed") is not False or manifest.get("diagnostic_only") is not True
            or manifest.get("training_performed") is not False
            or manifest.get("confirmation_start") != data.splits["boundaries"]["phase3_confirmation_start"]):
        raise ValueError("Variant sidecar contract mismatch")
    variants = manifest.get("variants", [])
    if not variants or len(set(variants)) != len(variants) or not set(variants).issubset(PROFILE_VARIANTS):
        raise ValueError("Invalid prepared variant roster")
    contracts = {v: variant_contract(data.schema, v) for v in variants}
    if manifest.get("contracts") != contracts:
        raise ValueError("Variant policy/schema mismatch")
    originals = {r["fixture"]["fixture_id"]: r for r in data.rows}
    result = {}
    with (path / "features.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["fixture_id"], row["variant"])
            original = originals.get(row["fixture_id"])
            if (key in result or original is None or row["variant"] not in contracts
                    or row["source_snapshot_id"] != original["snapshot_id"]
                    or row["source_feature_contract_id"] != data.manifest["feature_contract_id"]
                    or row["feature_contract_id"] != core.digest(contracts[row["variant"]])
                    or row["row_sha256"] != core.digest({k: v for k, v in row.items() if k != "row_sha256"})
                    or row["snapshot_id"] != core.digest(row["snapshot_contract"])):
                raise ValueError("Variant row identity/hash mismatch")
            _validate_values(row["values"], data.schema["names"])
            if (row["snapshot_contract"]["reference_snapshot_id"] != original["snapshot_id"]
                    or row["snapshot_contract"].get("version") != SNAPSHOT_VERSION
                    or row["snapshot_contract"]["feature_contract_id"] != row["feature_contract_id"]
                    or row["snapshot_contract"]["profile_history_sha256"] != row["evidence"]["profile_history_sha256"]
                    or row["evidence"]["as_of"] != core.utc(original["as_of"]).isoformat()
                    or row["evidence"]["availability"] != "assumed_final"):
                raise ValueError("Variant snapshot/availability mismatch")
            source_ids = row["evidence"]["profile_source_fixture_ids"]
            if (not isinstance(source_ids, list) or any(type(fid) is not int or fid <= 0 for fid in source_ids)
                    or source_ids != sorted(set(source_ids))):
                raise ValueError("Invalid variant source fixture identities")
            window = contracts[row["variant"]]["policy"]["provider_seasons"]
            for fid in source_ids:
                source = data.memberships.get(fid)
                if (source is None or fid == row["fixture_id"] or _available(source) >= core.utc(original["as_of"])
                        or not original["fixture"]["season"] - window < source["season"] <= original["fixture"]["season"]):
                    raise ValueError("Variant source fixture is missing/future/outside profile window")
            mapping = dict(zip(data.schema["names"], row["values"]))
            for market in MARKETS:
                for side in ("home", "away"):
                    item = row["support"][market][side]
                    count, ids = item["count"], item["fixture_ids"]
                    primary_ids = item["primary_fixture_ids"]
                    rate = mapping[f"{side}_{core.RATE_FIELDS[market]}"]
                    if (type(count) is not int or count < 0 or not isinstance(ids, list)
                            or any(type(fid) is not int or fid <= 0 for fid in ids)
                            or len(set(ids)) != len(ids) or count != len(ids)
                            or not set(ids).issubset(source_ids)
                            or not isinstance(primary_ids, list)
                            or any(type(fid) is not int or fid <= 0 for fid in primary_ids)
                            or len(set(primary_ids)) != len(primary_ids)
                            or not set(primary_ids).issubset(source_ids)
                            or type(item["rate_available"]) is not bool
                            or item["rate_available"] != (rate is not None)
                            or (rate is None and count != 0) or (rate is not None and (rate < 0 or count == 0))
                            or item["band"] != ("0" if count == 0 else "1-4" if count < 5 else "5-9" if count < 10 else "10+")):
                        raise ValueError("Variant support/feature/source mismatch")
            # Recompute eligibility from immutable original evidence and the
            # new support. A prepared variant never expands a paired cohort.
            expected = _decisions(original["market_eligibility"], row["support"])
            common = {m: original["market_eligibility"][m]["eligible"] and expected[m]["eligible"] for m in MARKETS}
            reasons = {m: [f"reference:{r}" for r in original["market_eligibility"][m]["reasons"]]
                          + [f"variant:{r}" for r in expected[m]["reasons"]] for m in MARKETS}
            if (row["market_eligibility"] != expected or row["common_eligible"] != common
                    or row["reference_eligible"] != {m: original["market_eligibility"][m]["eligible"] for m in MARKETS}
                    or row["common_exclusion_reasons"] != reasons):
                raise ValueError("Variant common cohort/eligibility mismatch")
            result[key] = row
    expected_keys = {(fid, variant) for fid in originals for variant in variants}
    if set(result) != expected_keys or len(result) != manifest.get("rows"):
        raise ValueError("Variant sidecar fixture coverage mismatch")
    return manifest, result
