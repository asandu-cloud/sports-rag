"""Read-only local evidence resolution for the Phase 3 research dataset.

No provider calls, database changes, archived-code execution or guessed periods.
Verified reconstruction plans and raw responses remain distinct source classes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from urllib.parse import unquote, urlparse

from .market_eligibility import CONTRACT, MARKETS, count, market_decisions, round_scope
from .model_dataset import load_canonical_inputs
from .staged_history import normalize_team_statistics, regulation_issues

EVIDENCE_VERSION = "phase3-history-evidence.v1"
_RECOVERY_PLANS = (
    ("phase1-history-2025-2026-09-22-verified", "plan.json"),
    ("local-history-import-2026-09-25", "reviewed-plans.json"),
)
_COLUMNS = {"corners": "corners", "sot": "shots_on", "shots": "shots_total", "fouls": "fouls_committed",
            "xg": "expected_goals", "possession": "possession"}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _time(value):
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _json(value):
    return json.loads(value) if isinstance(value, str) else (value or {})


def _equal(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    try:
        return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10)
    except (TypeError, ValueError, OverflowError):
        return False


class LocalEvidence:
    """Small fixture index and lazy, bounded raw-statistics reads; never writes."""

    def __init__(self, root, archives, *, as_of, recovery_plans=None):
        self.root = Path(root).resolve()
        self.as_of = _time(as_of)
        self.archives = {row["id"]: row for row in archives}
        self.fixtures = defaultdict(list)
        self.stat_archives = defaultdict(list)
        self.recoveries = defaultdict(list)
        self.source_files = {}
        self.errors = []
        for row in sorted(archives, key=lambda item: item["id"]):
            if row.get("provider") != "api_football" or row.get("endpoint") not in {"/fixtures", "/fixtures/statistics"}:
                continue
            try:
                if _time(row["fetched_at"]) > self.as_of:
                    continue
                params = _json(row["params"])
                if row["endpoint"] == "/fixtures/statistics":
                    if type(params.get("fixture")) is int:
                        self.stat_archives[params["fixture"]].append(row)
                    continue
                response, ref = self.read_archive(row)
                if not isinstance(response, list):
                    raise ValueError("fixture_response_not_list")
                seen = set()
                entries = []
                for raw in response:
                    fid = raw.get("fixture", {}).get("id") if isinstance(raw, dict) else None
                    if type(fid) is not int or fid in seen:
                        raise ValueError("duplicate_or_invalid_archived_fixture_id")
                    league = raw.get("league", {})
                    if (("league" in params and params["league"] != league.get("id"))
                            or ("season" in params and params["season"] != league.get("season"))
                            or ("id" in params and params["id"] != fid)):
                        raise ValueError("fixture_archive_parameter_scope_mismatch")
                    seen.add(fid)
                    entries.append((fid, raw))
                for fid, raw in entries:
                    self.fixtures[fid].append({"raw": raw, "reference": ref})
            except (ValueError, OSError, TypeError, KeyError) as exc:
                self.errors.append({"archive_id": row["id"], "reason": str(exc)})
        plans = recovery_plans if recovery_plans is not None else [
            self.root / "Index/prediction_experiments" / directory / filename
            for directory, filename in _RECOVERY_PLANS
        ]
        for plan in plans:
            if Path(plan).exists():
                self.read_recovery_plan(Path(plan))

    def _local_path(self, path, base):
        path = Path(path)
        if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
            raise ValueError("evidence_path_outside_allowed_root")
        return path

    def read_archive(self, row):
        if row.get("provider") != "api_football" or row.get("storage_backend") != "local":
            raise ValueError("unsupported_archive_provider_or_storage")
        if _time(row["fetched_at"]) > self.as_of:
            raise ValueError("evidence_observed_after_export_cutoff")
        uri = urlparse(row["storage_uri"])
        if uri.scheme != "file" or uri.netloc not in {"", "localhost"}:
            raise ValueError("archive_uri_not_local")
        path = self._local_path(unquote(uri.path), self.root / "Index/raw_archive")
        compressed = path.read_bytes()
        payload = gzip.decompress(compressed) if path.suffix == ".gz" else compressed
        if _sha(payload) != row["payload_digest"]:
            raise ValueError("archive_payload_checksum_mismatch")
        params = _json(row["params"])
        if _sha(json.dumps(params, sort_keys=True, separators=(",", ":"), default=str).encode()) != row["params_digest"]:
            raise ValueError("archive_parameters_checksum_mismatch")
        relative = str(path.relative_to(self.root))
        self.source_files[relative] = _sha(compressed)
        reference = {"kind": "raw_provider_archive", "archive_id": row["id"], "path": relative,
                     "payload_sha256": row["payload_digest"], "file_sha256": _sha(compressed),
                     "observed_at": _time(row["fetched_at"]).isoformat(), "params": params}
        return json.loads(payload), reference

    def read_recovery_plan(self, path):
        path = self._local_path(path, self.root / "Index/prediction_experiments")
        complete_path = path.parent / "COMPLETE.json"
        complete_bytes = complete_path.read_bytes()
        manifest = json.loads(complete_bytes)
        contents = path.read_bytes()
        if manifest.get(path.name) != _sha(contents):
            raise ValueError("recovery_plan_checksum_mismatch")
        self.source_files[str(path.relative_to(self.root))] = _sha(contents)
        self.source_files[str(complete_path.relative_to(self.root))] = _sha(complete_bytes)
        plans = json.loads(contents)
        for plan in plans if isinstance(plans, list) else [plans]:
            if plan.get("schema") != "historical-recovery.v1" or "legacy_zero_is_unknown" not in plan.get("policy", ""):
                raise ValueError("unsupported_recovery_plan")
            if _time(plan["prepared_at"]) > self.as_of:
                continue
            reference = {"kind": "verified_local_reconstruction", "path": str(path.relative_to(self.root)),
                         "file_sha256": _sha(contents), "observed_at": _time(plan["prepared_at"]).isoformat()}
            for entry in plan["entries"]:
                raw = entry["api_row"]
                fid = raw["fixture"]["id"]
                self.recoveries[fid].append({"raw": raw, "reference": reference, "entry": entry})

    def fixture_evidence(self, fixture, paired):
        """Use exact import lineage when present; otherwise latest archived row."""
        fid = fixture["fixture_id"]
        payloads = [_json(row.get("stats_json")) for row in paired.values()]
        staged = [row.get("_staged_history") for row in payloads if row.get("_staged_history")]
        legacy = [row.get("_history_recovery") for row in payloads if row.get("_history_recovery")]
        if staged:
            if len(staged) != 2 or staged[0] != staged[1]:
                return None, "staged_provenance_pair_mismatch"
            provenance = staged[0]
            candidates = [item for item in self.fixtures[fid]
                          if item["reference"].get("archive_id") == provenance.get("fixture_archive_id")
                          and item["reference"]["payload_sha256"] == provenance.get("fixture_payload_sha256")]
            if len(candidates) != 1:
                return None, "staged_fixture_archive_unavailable"
            return {**candidates[0], "source_class": "raw_provider_archive", "provenance": provenance}, None
        if legacy:
            if len(legacy) != 2 or legacy[0] != legacy[1]:
                return None, "legacy_provenance_pair_mismatch"
            provenance = legacy[0]
            candidates = [item for item in self.recoveries[fid]
                          if item["entry"].get("source") == provenance.get("source")]
            if len(candidates) != 1:
                return None, "verified_reconstruction_unavailable"
            return {**candidates[0], "source_class": "verified_local_reconstruction", "provenance": provenance}, None
        candidates = self.fixtures[fid]
        if not candidates:
            return None, "period_evidence_unavailable"
        selected = max(candidates, key=lambda item: (item["reference"]["observed_at"], item["reference"]["archive_id"]))
        return {**selected, "source_class": "raw_provider_archive", "provenance": {}}, None

    def statistics_evidence(self, fixture, paired, resolved):
        """Return both provider team maps; missing raw data stays unavailable."""
        if resolved["source_class"] == "verified_local_reconstruction":
            if resolved["provenance"].get("zero_policy") != "legacy_zero_is_unknown":
                raise ValueError("unrecognized_legacy_zero_policy")
            values = resolved["entry"]["stats"]
            return {fixture[f"{side}_team_id"]: values[index] for index, side in enumerate(("home", "away"))}, resolved["reference"], "legacy_zero_is_unknown"
        provenance = resolved["provenance"]
        if provenance:
            row = self.archives.get(provenance.get("statistics_archive_id"))
            if not row or row["payload_digest"] != provenance.get("statistics_payload_sha256"):
                raise ValueError("staged_statistics_archive_unavailable")
            candidates = [row]
        else:
            candidates = sorted(self.stat_archives[fixture["fixture_id"]], key=lambda item: (_time(item["fetched_at"]), item["id"]), reverse=True)
        # Native canonical rows may retain an older evidenced provider revision.
        # Prefer an exactly matching response, never silently replace values.
        latest = None
        for row in candidates:
            response, reference = self.read_archive(row)
            if row["endpoint"] != "/fixtures/statistics" or _json(row["params"]).get("fixture") != fixture["fixture_id"]:
                raise ValueError("statistics_archive_fixture_mismatch")
            normalized = normalize_team_statistics(resolved["raw"], response)
            if provenance and provenance.get("statistics_response_present") != bool(response):
                raise ValueError("statistics_response_presence_mismatch")
            candidate = (normalized, reference, "explicit_provider_zero_or_unknown")
            if latest is None:
                latest = candidate
            if all(all(_equal(source.get(column), paired.get(side, {}).get(column)) for column in _COLUMNS.values())
                   for side in ("home", "away")
                   for source in [normalized.get(fixture[f"{side}_team_id"], {})]):
                return candidate
        if latest is not None:
            return latest
        raise ValueError("statistics_source_unavailable")


def _identity_issues(fixture, canonical, raw):
    issues = []
    expected = {"id": fixture["fixture_id"], "league": canonical["provider_competition_id"], "season": fixture["season"],
                "home": fixture["home_team_id"], "away": fixture["away_team_id"]}
    actual = {"id": raw.get("fixture", {}).get("id"), "league": raw.get("league", {}).get("id"),
              "season": raw.get("league", {}).get("season"), "home": raw.get("teams", {}).get("home", {}).get("id"),
              "away": raw.get("teams", {}).get("away", {}).get("id")}
    if expected != actual or any(type(value) is not int for value in actual.values()):
        issues.append("evidence_identity_mismatch")
    try:
        if _time(raw["fixture"]["date"]) != _time(fixture["kickoff"]):
            issues.append("evidence_kickoff_mismatch")
    except (TypeError, ValueError, KeyError):
        issues.append("evidence_kickoff_mismatch")
    if any(count(raw.get("goals", {}).get(side)) != count(canonical.get(f"{side}_goals")) for side in ("home", "away")):
        issues.append("canonical_provider_score_conflict")
    if canonical.get("round") != raw.get("league", {}).get("round"):
        issues.append("canonical_provider_round_conflict")
    return issues


def load_eligible_inputs(database: Path, root: Path, as_of: datetime, *, recovery_plans=None):
    """Return all FT targets, safe context, per-fixture evidence and coverage.

    The database must be the caller's consistent research snapshot. Actual
    observations are retained; retrospective +3h label timestamps are separately
    marked assumed. No labels or optional statistics are filled from a new source.
    """
    if not isinstance(as_of, datetime) or as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    targets, original_history, source_report = load_canonical_inputs(database)
    targets = [row for row in targets if row["status"] == "FT"]
    histories = {row["fixture_id"]: row for row in original_history}
    with sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        fixtures = {row["api_football_id"]: dict(row) for row in db.execute(
            "SELECT f.*, c.api_football_id AS provider_competition_id FROM fixtures f JOIN competitions c ON c.id=f.competition_id")}
        stats = defaultdict(dict)
        by_internal = {row["id"]: row for row in fixtures.values()}
        for row in db.execute("SELECT * FROM fixture_team_stats"):
            fixture = by_internal[row["fixture_id"]]
            side = "home" if row["team_id"] == fixture["home_team_id"] else "away"
            stats[fixture["api_football_id"]][side] = dict(row)
        archives = [dict(row) for row in db.execute("SELECT * FROM raw_payload_archive WHERE endpoint IN ('/fixtures','/fixtures/statistics')")]
    local = LocalEvidence(root, archives, as_of=as_of, recovery_plans=recovery_plans)
    evidence, clean_history = {}, []
    excluded = Counter()
    sources = Counter()
    rounds = Counter()
    for fixture in targets:
        fid = fixture["fixture_id"]
        canonical, paired = fixtures[fid], stats[fid]
        original = histories[fid]
        context = deepcopy(original)
        resolved, error = local.fixture_evidence(fixture, paired)
        common = [error] if error else []
        ref, stat_ref = None, None
        source_class = "unresolved"
        zero_policy = "unknown"
        normalized = {}
        stat_error = "statistics_source_unavailable"
        period_issues = []
        statistic_presence = {side: {} for side in ("home", "away")}
        scope = round_scope(fixture["competition"], fixture["season"], canonical.get("round"))
        if resolved:
            raw, ref = resolved["raw"], resolved["reference"]
            source_class = resolved["source_class"]
            period_issues = regulation_issues(raw, as_of=as_of) + _identity_issues(fixture, canonical, raw)
            common.extend(period_issues)
            try:
                normalized, stat_ref, zero_policy = local.statistics_evidence(fixture, paired, resolved)
                stat_error = None
            except (ValueError, TypeError, KeyError, OSError) as exc:
                stat_error = str(exc)
            # A source's result status is not exhaustive participation proof.
            # Preserve a conservative concern, not a claimed official award.
            if ({canonical.get("home_goals"), canonical.get("away_goals")} == {0, 3}
                    and not raw.get("fixture", {}).get("referee")
                    and not any(normalized.get(fixture[f"{side}_team_id"], {}) for side in ("home", "away"))):
                common.append("suspected_administrative_result_participation_unverified")
        if not scope["context"]:
            common.append("unknown_round")
        common = list(dict.fromkeys(common))
        stat_reasons = {side: {market: [] for market in MARKETS} for side in ("home", "away")}
        for side in ("home", "away"):
            source = normalized.get(fixture[f"{side}_team_id"], {})
            for field, column in _COLUMNS.items():
                value = context[side][field]
                reason = stat_error
                presence = ("source_unavailable" if stat_error else "omitted" if column not in source
                            else "null" if source[column] is None
                            else "ambiguous_legacy_zero" if zero_policy == "legacy_zero_is_unknown" and source[column] == 0
                            else "explicit_zero" if source[column] == 0 else "observed")
                statistic_presence[side][field] = {"state": presence, "canonical_matches_source":
                    stat_error is None and _equal(value, source.get(column))}
                if reason is None and not _equal(value, source.get(column)):
                    reason = f"{side}_{field}_source_conflict"
                if reason is None and zero_policy == "legacy_zero_is_unknown" and source.get(column) == 0:
                    reason = f"{side}_{field}_ambiguous_legacy_zero"
                if field in {"corners", "sot", "shots", "fouls"} and value is not None and count(value) is None:
                    reason = f"invalid_{side}_{field}_count"
                if reason:
                    context[side][field] = None
                    if field in stat_reasons[side]:
                        stat_reasons[side][field].append(reason)
            if count(context[side]["goals"]) is None:
                context[side]["goals"] = None
            # Card targets and features are disqualified in this research batch.
            context[side]["cards"] = None
        labels_by_side = {side: {market: context[side][market] for market in MARKETS} for side in ("home", "away")}
        # Keep uncertified recorded outcomes only in the explicit audit field.
        # Unknown round scope alone does not invalidate a verified period/count.
        if any(reason != "unknown_round" for reason in common):
            labels_by_side = {side: {market: None for market in MARKETS} for side in ("home", "away")}
        labels = {market: sum(labels_by_side[side][market] for side in ("home", "away"))
                  if all(labels_by_side[side][market] is not None for side in ("home", "away")) else None for market in MARKETS}
        decisions = market_decisions(labels_by_side, common, target_scope=scope["target"], round_group=scope["group"], statistic_reasons=stat_reasons)
        item = {"evidence_version": EVIDENCE_VERSION, "eligibility_version": CONTRACT["version"],
                "fixture_id": fid, "competition": fixture["competition"], "season": fixture["season"],
                "round": canonical.get("round"), "round_group": scope["group"], "status": canonical["status"],
                "elapsed": canonical.get("elapsed"), "source_class": source_class, "zero_policy": zero_policy,
                "fixture_source": ref, "statistics_source": stat_ref, "statistics_source_reason": stat_error,
                "period_verified": bool(resolved and not period_issues), "participation_status": "concern" if any("participation" in r for r in common) else "no_identified_concern_not_universal_certification",
                "period_evidence": resolved["raw"].get("score") if resolved else None,
                "statistic_presence": statistic_presence,
                "history_eligible": not common, "target_scope_eligible": scope["target"], "common_reasons": common,
                "market_eligibility": decisions, "team_labels": labels_by_side, "labels": labels,
                "recorded_team_labels": {side: {market: original[side][market] for market in MARKETS} for side in ("home", "away")},
                "availability": "assumed_final", "actual_observed_at": original.get("observed_at"),
                "label_available_at": (_time(fixture["kickoff"]) + timedelta(hours=3)).isoformat(),
                "label_availability_is_assumed": True, "forecast_stage": "kickoff_reconstruction",
                "source_provenance": resolved.get("provenance", {}) if resolved else {}}
        evidence[fid] = item
        if not common:
            clean_history.append(context)
        excluded.update(common)
        sources[source_class] += 1
        rounds[scope["group"]] += 1
    report = {"evidence_version": EVIDENCE_VERSION, "contract": CONTRACT, "canonical": source_report,
              "ft_targets": len(targets), "safe_history_rows": len(clean_history), "common_exclusions": dict(sorted(excluded.items())),
              "source_classes": dict(sorted(sources.items())), "round_groups": dict(sorted(rounds.items())),
              "target_eligible_before_support": {market: sum(item["market_eligibility"][market]["eligible"] for item in evidence.values()) for market in MARKETS},
              "source_files": dict(sorted(local.source_files.items())), "archive_errors": local.errors,
              "availability_caveat": "Retrospective final statistics; +3h is assumed availability, not original observed vintage."}
    return targets, clean_history, evidence, report
