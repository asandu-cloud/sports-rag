"""Strict development-only inputs and elapsed-time fitting weights.

The allowlist is an auditable access boundary: this module never opens the
snapshot database, audit histories, frozen source code or held-out stores.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

import numpy as np

from Scripts.data_platform.features.phase3_features import feature_contract
from Scripts.rag_ingest.core import model_features as core

ROOT = Path(__file__).resolve().parents[4]
MARKETS = ("goals", "corners", "sot")
DEVELOPMENT = {"initial_training", "development"}
ALLOWED_FILES = frozenset({"manifest.json", "feature-schema.json", "splits.json", "lockbox-policy.json",
                           "development/features.jsonl", "development/labels.jsonl"})
COMPATIBLE_SOURCES = ("Scripts/rag_ingest/core/model_features.py",
                      "Scripts/data_platform/features/phase3_features.py",
                      "Scripts/data_platform/registry/competitions.yaml")
COMPETITIONS = {name: "continental_cup" if name in {"UCL", "UEL", "UECL"} else "domestic_league"
                for name in ("EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1", "Championship", "SuperLig",
                             "Eredivisie", "PrimeiraLiga", "BelgianProLeague", "UCL", "UEL", "UECL")}
_HEX = re.compile(r"[0-9a-f]{64}")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _utc(value):
    if not isinstance(value, (str, datetime)):
        raise ValueError("Timestamp must be an explicit timezone-aware string or datetime")
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _decode(body):
    def invalid(value):
        raise ValueError("Non-finite JSON number: " + value)
    return json.loads(body, object_pairs_hook=_object, parse_constant=invalid)


def _integer(value, *, positive=False):
    return type(value) is int and value >= int(positive)


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _count(value):
    return _finite(value) and value >= 0 and float(value).is_integer()


def _ids(values):
    return isinstance(values, list) and all(_integer(x, positive=True) for x in values) and len(set(values)) == len(values)


def _positive_parameter(value, name):
    if value is not None and (not _finite(value) or value <= 0):
        raise ValueError(name + " must be finite and positive or None")


def expected_schema():
    """Generate ordered names from the compatible pure builder, without data IO."""
    fixture = {"fixture_id": 1, "competition": "EPL", "season": 2026, "home_team_id": 1,
               "away_team_id": 2, "kickoff": "2026-01-01T00:00:00+00:00"}
    snapshot = core.capture_snapshot(fixture, [], as_of=fixture["kickoff"], competitions=COMPETITIONS,
                                     availability="assumed_final")
    return feature_contract(core.build_features(snapshot)["names"])


class DevelopmentDataset:
    def __init__(self, path, *, source_root=ROOT):
        path = Path(path)
        if path.is_symlink():
            raise ValueError("Dataset directory cannot be a symlink")
        self.path = path.resolve()
        complete = _decode(self._read("COMPLETE.json"))
        if not isinstance(complete, dict) or not ALLOWED_FILES.issubset(complete):
            raise ValueError("Incomplete development artifact manifest")
        # Validate every reference syntactically, but read only the fixed allowlist.
        for name, checksum in complete.items():
            self._validate_name(name)
            if not isinstance(checksum, str) or not _HEX.fullmatch(checksum):
                raise ValueError("Invalid artifact checksum")
        documents = {}
        for name in sorted(ALLOWED_FILES):
            body = self._read(name)
            if _sha(body) != complete[name]:
                raise ValueError("Dataset checksum mismatch: " + name)
            documents[name] = body
        self.artifact_hashes = {name: complete[name] for name in sorted(ALLOWED_FILES)}
        self.manifest = _decode(documents["manifest.json"])
        self.schema = _decode(documents["feature-schema.json"])
        self.splits = _decode(documents["splits.json"])
        policy = _decode(documents["lockbox-policy.json"])
        if self.manifest.get("version") != "phase3-dataset.v1" or self.manifest.get("availability") != "assumed_final":
            raise ValueError("Unsupported dataset version/availability")
        if (self.manifest.get("publication_enabled") is not False or self.manifest.get("promotion_allowed") is not False
                or self.manifest.get("league_weighting") != "equal_fixture_before_recency"):
            raise ValueError("Not an isolated equal-fixture research dataset")
        if (policy.get("version") != "phase3-lockbox.v1" or set(policy.get("development_partitions", [])) != DEVELOPMENT
                or policy.get("status") != "unopened_for_model_evaluation" or policy.get("inspection_events") != []):
            raise ValueError("Invalid development access policy")
        if self.schema != expected_schema() or len(self.schema["names"]) != 126 or len(self.schema["source_names"]) != 148:
            raise ValueError("Unsupported ordered feature schema/mask/policy")
        if self.manifest.get("feature_contract_id") != core.digest(self.schema):
            raise ValueError("Feature contract digest mismatch")
        for name in COMPATIBLE_SOURCES:
            expected = self.manifest.get("source_hashes", {}).get(name)
            if not isinstance(expected, str) or _sha((Path(source_root) / name).read_bytes()) != expected:
                raise ValueError("Re-export required: incompatible source " + name)
        self._validate_splits()
        dataset_id = core.digest({"snapshot": self.manifest.get("snapshot_sha256"),
            "features": self.manifest["feature_contract_id"], "inputs": complete.get("audit/inputs.json"),
            "splits": core.digest(self.splits), "evidence": complete.get("audit/evidence.jsonl"),
            "sources": self.manifest["source_hashes"]})
        if self.manifest.get("dataset_id") != dataset_id:
            raise ValueError("Dataset identity mismatch")
        rows = [_decode(line) for line in documents["development/features.jsonl"].splitlines() if line.strip()]
        labels = [_decode(line) for line in documents["development/labels.jsonl"].splitlines() if line.strip()]
        label_map = {}
        for label in labels:
            fid = label.get("fixture_id")
            if not _integer(fid, positive=True) or fid in label_map:
                raise ValueError("Duplicate or invalid label fixture")
            if set(label.get("labels", {})) != {*MARKETS, "cards"} or set(label.get("team_labels", {})) != {"home", "away"}:
                raise ValueError("Invalid target schema")
            for side in ("home", "away"):
                if set(label["team_labels"][side]) != {*MARKETS, "cards"}:
                    raise ValueError("Invalid team target schema")
            for market in (*MARKETS, "cards"):
                parts = [label["team_labels"][side][market] for side in ("home", "away")]
                total = label["labels"][market]
                if any(x is not None and not _count(x) for x in [total, *parts]):
                    raise ValueError("Invalid nonnegative integer target")
                expected = sum(parts) if all(x is not None for x in parts) else None
                if total != expected:
                    raise ValueError("Team targets disagree with total")
            if label["labels"]["cards"] is not None or any(label["team_labels"][side]["cards"] is not None for side in ("home", "away")):
                raise ValueError("Unqualified card target in development dataset")
            label_map[fid] = label
        seen, snapshots = set(), set()
        self.rows = []
        for row in rows:
            self._validate_row(row)
            fid = row["fixture"]["fixture_id"]
            if fid in seen or row["snapshot_id"] in snapshots:
                raise ValueError("Duplicate fixture version or snapshot")
            seen.add(fid); snapshots.add(row["snapshot_id"])
            if fid not in label_map:
                raise ValueError("Missing development label record")
            label = label_map[fid]
            for market in MARKETS:
                if row["market_eligibility"][market]["eligible"] and label["labels"][market] is None:
                    raise ValueError("Eligible row has missing target")
            self.rows.append({**row, "labels": label["labels"], "team_labels": label["team_labels"]})
        expected_ids = {fid for fid, m in self.memberships.items() if m["partition"] in DEVELOPMENT}
        if seen != set(label_map) or seen != expected_ids:
            raise ValueError("Development feature/label/membership fixture sets differ")
        self.rows.sort(key=lambda row: (_utc(row["fixture"]["kickoff"]), row["fixture"]["fixture_id"]))

    @staticmethod
    def _validate_name(name):
        if (not isinstance(name, str) or not name or "\\" in name or PurePosixPath(name).is_absolute()
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            raise ValueError("Unsafe artifact path")

    def _read(self, name):
        if name not in ALLOWED_FILES | {"COMPLETE.json"}:
            raise ValueError("Reader refuses non-development artifact")
        self._validate_name(name)
        item = self.path / name
        if any(p.is_symlink() for p in [item, *item.parents] if p != self.path and p.is_relative_to(self.path)):
            raise ValueError("Artifact symlink is forbidden")
        if not item.resolve().is_relative_to(self.path):
            raise ValueError("Artifact escapes dataset")
        return item.read_bytes()

    def _validate_splits(self):
        s = self.splits
        if s.get("version") != "phase3-chronological-splits.v1" or s.get("status") != "defined":
            raise ValueError("No supported chronological splits")
        if s.get("sha256") != core.digest({k: v for k, v in s.items() if k != "sha256"}):
            raise ValueError("Split digest mismatch")
        if s.get("membership_sha256") != core.digest(s.get("memberships")):
            raise ValueError("Membership digest mismatch")
        keys = ("initial_training_start", "development_start", "phase3_confirmation_start", "calibration_start", "final_system_start", "final_system_end")
        boundaries = s.get("boundaries", {})
        self.boundaries = {name: _utc(boundaries[name]) for name in keys}
        if any(self.boundaries[a] >= self.boundaries[b] for a, b in zip(keys, keys[1:])):
            raise ValueError("Nonchronological partition boundaries")
        self.memberships = {}
        times = {}
        for m in s["memberships"]:
            fid = m.get("fixture_id")
            if not _integer(fid, positive=True) or fid in self.memberships:
                raise ValueError("Duplicate/invalid fixture membership")
            kickoff = _utc(m.get("kickoff"))
            partition = m.get("partition")
            boundaries_order = [self.boundaries[name] for name in keys[1:]]
            expected = ("initial_training", "development", "phase3_confirmation", "calibration", "final_system_test", "prospective_reserve")[sum(kickoff >= b for b in boundaries_order)]
            if partition != expected or s.get("partition_by_fixture", {}).get(str(fid)) != partition:
                raise ValueError("Fixture partition contradicts chronology")
            if kickoff in times and times[kickoff] != partition:
                raise ValueError("Simultaneous fixtures split across partitions")
            times[kickoff] = partition
            self.memberships[fid] = m
        if set(s.get("partition_by_fixture", {})) != {str(fid) for fid in self.memberships}:
            raise ValueError("Partition lookup differs from membership")
        if self.manifest.get("row_count") != len(self.memberships):
            raise ValueError("Manifest fixture count differs from membership")
        self.folds = s.get("development_folds", [])
        seen = set(); previous_end = None
        for fold in self.folds:
            fid = fold.get("fold_id")
            if not isinstance(fid, str) or fid in seen:
                raise ValueError("Duplicate/invalid development fold")
            seen.add(fid)
            cutoff, start, end = (_utc(fold[k]) for k in ("fit_cutoff", "validation_start", "validation_end"))
            if not (self.boundaries["development_start"] <= cutoff == start < end <= self.boundaries["phase3_confirmation_start"]):
                raise ValueError("Fold crosses allowed development interval")
            if previous_end is not None and start < previous_end:
                raise ValueError("Overlapping/nonchronological development folds")
            previous_end = end
        if not self.folds:
            raise ValueError("No development folds")

    def _validate_row(self, row):
        # Keep the ordinary reader's access scope fixed; the confirmation
        # operation uses the shared pure validator with its own fixed interval.
        _validate_feature_row(self, row, partitions=DEVELOPMENT, start=None,
                              end=self.boundaries["phase3_confirmation_start"])

    def select_fold(self, market, fold_id, lookback_days=None, half_life_days=None):
        if market not in MARKETS:
            raise ValueError("Unsupported benchmark market")
        _positive_parameter(lookback_days, "lookback_days")
        _positive_parameter(half_life_days, "half_life_days")
        matching = [f for f in self.folds if f["fold_id"] == fold_id]
        if len(matching) != 1:
            raise ValueError("Unknown development fold")
        fold = matching[0]
        cutoff, start, end = (_utc(fold[k]) for k in ("fit_cutoff", "validation_start", "validation_end"))
        earliest = cutoff - timedelta(days=lookback_days) if lookback_days is not None else None
        train, validation = [], []
        excluded = Counter()
        for row in self.rows:
            if not row["market_eligibility"][market]["eligible"]:
                excluded["ineligible_market"] += 1
                continue
            kickoff = _utc(row["fixture"]["kickoff"])
            joined = {**row, "target": row["labels"][market]}
            if kickoff < cutoff:
                if _utc(row["label_available_at"]) >= cutoff:
                    excluded["label_available_after_fit"] += 1
                elif earliest is not None and kickoff < earliest:
                    excluded["outside_lookback"] += 1
                else:
                    train.append(joined)
            elif start <= kickoff < end:
                if row["partition"] != "development":
                    raise ValueError("Validation row outside development")
                validation.append(joined)
        weights, weighting = recency_weights(train, cutoff, half_life_days)
        report = {"dataset_id": self.manifest["dataset_id"], "market": market, "fold_id": fold_id,
                  "fit_cutoff": cutoff.isoformat(), "validation_start": start.isoformat(), "validation_end": end.isoformat(),
                  "lookback_days": lookback_days, "training_unique_fixtures": len(train),
                  "validation_unique_fixtures": len(validation), "exclusions": dict(sorted(excluded.items())), **weighting,
                  "training_membership_sha256": core.digest([r["fixture"]["fixture_id"] for r in train]),
                  "validation_membership_sha256": core.digest([r["fixture"]["fixture_id"] for r in validation])}
        return train, validation, weights, report


def recency_weights(rows, fit_cutoff, half_life_days=None):
    """Equal fixtures before decay; normalized weights sum to training row count."""
    _positive_parameter(half_life_days, "half_life_days")
    cutoff = _utc(fit_cutoff)
    ids = [row["fixture"]["fixture_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate fitting fixture")
    ages = np.asarray([(cutoff - _utc(row["fixture"]["kickoff"])).total_seconds() / 86400 for row in rows], dtype=float)
    if np.any(ages <= 0) or not np.isfinite(ages).all():
        raise ValueError("Fitting fixture is not strictly before cutoff")
    if not len(rows):
        return np.asarray([], dtype=float), {"half_life_days": half_life_days, "effective_sample_size": 0.0,
            "weight_sum": 0.0, "weight_mean": None, "age_days": None, "league_contributions": {}, "training_date_range": None}
    logs = np.zeros(len(rows)) if half_life_days is None else -np.log(2.0) * ages / half_life_days
    weights = np.exp(logs - logs.max())
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("Recency weights underflow; recipe is numerically unsupported")
    weights *= len(rows) / weights.sum()
    ess = float(weights.sum() ** 2 / np.dot(weights, weights))
    contributions = {}
    for league in sorted({row["fixture"]["competition"] for row in rows}):
        selected = np.asarray([row["fixture"]["competition"] == league for row in rows])
        w = weights[selected]
        contributions[league] = {"unique_fixtures": int(selected.sum()), "raw_share": float(selected.mean()),
                                "weight_sum": float(w.sum()), "weighted_share": float(w.sum() / weights.sum()),
                                "effective_sample_size": float(w.sum() ** 2 / np.dot(w, w))}
    dates = [_utc(row["fixture"]["kickoff"]) for row in rows]
    return weights, {"half_life_days": half_life_days, "effective_sample_size": ess, "weight_sum": float(weights.sum()),
        "weight_mean": float(weights.mean()), "age_days": {"min": float(ages.min()), "median": float(np.median(ages)), "max": float(ages.max())},
        "league_contributions": contributions, "training_date_range": {"first": min(dates).isoformat(), "last": max(dates).isoformat()}}


def family_gate(report, family):
    if family not in {"ridge", "poisson", "lightgbm", "xgboost", "catboost", "league_average", "statistical"}:
        raise ValueError("Unknown benchmark family")
    trees = family in {"lightgbm", "xgboost", "catboost"}
    thresholds = {"training_unique_fixtures": 5000 if trees else 2000,
                  "effective_sample_size": 2000 if trees else 1000, "validation_unique_fixtures": 500}
    if any(not _finite(report.get(key, 0)) or report.get(key, 0) < 0 for key in thresholds):
        raise ValueError("Invalid family-gate sample support")
    reasons = ["insufficient_" + key for key, minimum in thresholds.items() if report.get(key, 0) < minimum]
    return {"qualified": not reasons, "reasons": reasons, "thresholds": thresholds}


def _validate_feature_row(data, row, *, partitions, start, end):
    """Pure schema/identity/temporal validation; does not read any data store."""
    fixture = row.get("fixture", {})
    fid = fixture.get("fixture_id")
    if not _integer(fid, positive=True) or fid not in data.memberships:
        raise ValueError("Invalid development fixture ID")
    if (any(not _integer(fixture.get(k), positive=True) for k in ("home_team_id", "away_team_id", "season"))
            or fixture["home_team_id"] == fixture["away_team_id"] or fixture.get("competition") not in COMPETITIONS
            or fixture.get("status") != "FT"):
        raise ValueError("Invalid fixture identity/status")
    m = data.memberships[fid]
    if (row.get("partition") not in partitions or m["partition"] != row["partition"]
            or m.get("competition") != fixture["competition"] or m.get("season") != fixture["season"]
            or m.get("completed") is not True or m.get("row_count") != 1):
        raise ValueError("Fixture metadata differs from split membership")
    kickoff = _utc(fixture.get("kickoff")); as_of = _utc(row.get("as_of"))
    if (kickoff != _utc(m["kickoff"]) or as_of != kickoff or kickoff >= end or (start is not None and kickoff < start)
            or _utc(row.get("label_available_at")) != kickoff + timedelta(hours=3)):
        raise ValueError("Invalid feature/label availability or held-out chronology")
    for key in ("observed_at", "actual_observed_at"):
        if row.get(key) is not None:
            _utc(row[key])
    if row.get("observed_at") != row.get("actual_observed_at"):
        raise ValueError("Conflicting actual observation timestamps")
    if (row.get("availability") != "assumed_final" or row.get("forecast_stage") != data.schema["forecast_stage"]
            or row.get("feature_contract_id") != data.manifest["feature_contract_id"]
            or not isinstance(row.get("snapshot_id"), str) or not _HEX.fullmatch(row["snapshot_id"])):
        raise ValueError("Row feature contract/snapshot mismatch")
    if (row.get("source_class") not in {"raw_provider_archive", "verified_local_reconstruction", "unresolved"}
            or row.get("round_group") not in {"domestic_regular", "domestic_integral_split", "domestic_separate_playoff",
                "european_group", "european_league", "european_knockout", "european_qualifying", "unknown"}
            or "observed_at" not in row or "actual_observed_at" not in row):
        raise ValueError("Missing/unsupported source or round evidence metadata")
    values = row.get("values")
    if not isinstance(values, list) or len(values) != len(data.schema["names"]) or any(v is not None and not _finite(v) for v in values):
        raise ValueError("Invalid feature vector shape/numbers")
    by_name = dict(zip(data.schema["names"], values))
    for name, value in by_name.items():
        if name.endswith("__missing") and value != float(by_name[name[:-9]] is None):
            raise ValueError("Missingness indicator disagrees with value")
    for competition in COMPETITIONS:
        if by_name["competition_" + competition] != float(fixture["competition"] == competition):
            raise ValueError("Competition feature disagrees with identity")
    decisions = row.get("market_eligibility", {})
    if set(decisions) != {*MARKETS, "cards"} or set(row.get("support", {})) != set(MARKETS):
        raise ValueError("Incomplete eligibility/support contract")
    eligible = []
    for market, decision in decisions.items():
        reasons = decision.get("reasons")
        if (type(decision.get("eligible")) is not bool or not isinstance(reasons, list)
                or any(not isinstance(r, str) or not r for r in reasons) or len(set(reasons)) != len(reasons)
                or decision["eligible"] != (not reasons) or decision.get("primary_reason") != (reasons[0] if reasons else None)):
            raise ValueError("Contradictory eligibility reasons")
        if market == "cards":
            if decision["eligible"] or "cards_target_not_qualified" not in reasons:
                raise ValueError("Unqualified cards cannot enter benchmarks")
            continue
        if decision["eligible"]:
            if (row["source_class"] == "unresolved" or row["round_group"] in {"domestic_separate_playoff", "european_qualifying", "unknown"}
                    or kickoff < data.boundaries["initial_training_start"]):
                raise ValueError("Eligible fixture contradicts source/round/cohort scope")
            eligible.append(market)
        if set(row["support"][market]) != {"home", "away"}:
            raise ValueError("Invalid support sides")
        for side in ("home", "away"):
            support = row["support"][market][side]
            n, ids = support.get("count"), support.get("fixture_ids")
            if not _integer(n) or not _ids(ids) or n != len(ids) or not _ids(support.get("primary_fixture_ids")):
                raise ValueError("Support count/fixture identities disagree")
            if support.get("band") != ("0" if n == 0 else "1-4" if n < 5 else "5-9" if n < 10 else "10+"):
                raise ValueError("Invalid support band")
            rate = by_name[side + "_" + core.RATE_FIELDS[market]]
            if type(support.get("rate_available")) is not bool or support["rate_available"] != (rate is not None):
                raise ValueError("Support production-rate mismatch")
            if (n < 5 and f"insufficient_{side}_history" not in reasons) or (rate is None and f"missing_{side}_production_rate" not in reasons):
                raise ValueError("Support exclusion reason missing")
            if decision["eligible"] and (n < 5 or rate is None or rate < 0):
                raise ValueError("Eligible row fails support gate")
            for source_id in set(ids) | set(support["primary_fixture_ids"]):
                source = data.memberships.get(source_id)
                if not source or source_id == fid or _utc(source["kickoff"]) + timedelta(hours=3) >= as_of:
                    raise ValueError("Future/missing fixture in support history")
    if sorted(eligible) != sorted(m.get("eligible_markets", [])):
        raise ValueError("Eligibility differs from split membership")
