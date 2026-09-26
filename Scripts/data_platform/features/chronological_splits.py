"""Pure chronological research splits; no targets, databases or providers are read.

Rows contain flat fixture metadata or ``fixture`` metadata, and either
``eligible_markets`` or ``market_eligibility[market]['eligible']``. Eligibility
is an upstream decision, never inferred here from a numerical target. Calendar
membership is shared by every version, stage and market of a provider fixture.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Iterable, Mapping


SPLIT_VERSION = "phase3-chronological-splits.v1"
MARKETS = ("goals", "corners", "sot")
PARTITIONS = ("initial_training", "development", "phase3_confirmation", "calibration",
              "final_system_test", "prospective_reserve")
HELD_OUT_PARTITIONS = frozenset(PARTITIONS[2:])
BOUNDARY_KEYS = ("initial_training_start", "development_start", "phase3_confirmation_start",
                 "calibration_start", "final_system_start", "final_system_end")


def _utc(value) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _metadata(row: Mapping) -> Mapping:
    fixture = row.get("fixture", row)
    if not isinstance(fixture, Mapping):
        raise ValueError("invalid fixture metadata")
    return fixture


def _value(row: Mapping, key: str):
    return row[key] if key in row else _metadata(row).get(key)


def _eligible(row: Mapping) -> set[str]:
    explicit = row.get("eligible_markets")
    if explicit is not None:
        if not isinstance(explicit, (list, tuple, set, frozenset)):
            raise ValueError("eligible_markets must be a collection")
        return {market for market in MARKETS if market in explicit}
    decisions = row.get("market_eligibility", {})
    if not isinstance(decisions, Mapping):
        raise ValueError("market_eligibility must be a mapping")
    return {market for market in MARKETS
            if decisions.get(market) is True or
            (isinstance(decisions.get(market), Mapping) and decisions[market].get("eligible") is True)}


def _completed(row: Mapping) -> bool:
    return _value(row, "status") == "FT" or (_value(row, "status") is None and _value(row, "completed") is True)


def _groups(rows: Iterable[Mapping]) -> tuple[list[dict], list[dict]]:
    """Reject ambiguous fixture metadata before unioning market eligibility."""
    grouped, invalid = defaultdict(list), []
    for row in rows:
        try:
            fixture = _metadata(row)
            fixture_id = fixture.get("fixture_id")
            if type(fixture_id) is not int or fixture_id <= 0:
                raise ValueError("invalid_fixture_id")
            grouped[fixture_id].append(row)
        except (ValueError, TypeError, AttributeError):
            invalid.append({"fixture_id": None, "reasons": ["invalid_fixture_id"]})
    accepted = []
    for fixture_id, versions in sorted(grouped.items()):
        reasons, kickoffs, competitions, seasons, markets = set(), set(), set(), set(), set()
        teams = {"home_team_id": set(), "away_team_id": set()}
        completed = False
        for row in versions:
            try:
                kickoffs.add(_iso(_utc(_metadata(row).get("kickoff"))))
            except (ValueError, TypeError, OverflowError):
                reasons.add("invalid_kickoff")
            competition, season = _value(row, "competition"), _value(row, "season")
            if competition is not None:
                if not isinstance(competition, str):
                    reasons.add("invalid_competition")
                else:
                    competitions.add(competition)
            if season is not None:
                if type(season) is not int:
                    reasons.add("invalid_season")
                else:
                    seasons.add(season)
            for side, identities in teams.items():
                identity = _value(row, side)
                if identity is not None:
                    if type(identity) is not int or identity <= 0:
                        reasons.add("invalid_team_identity")
                    else:
                        identities.add(identity)
            try:
                # A not-yet-completed revision cannot supply eligible labels.
                if _completed(row):
                    completed = True
                    markets.update(_eligible(row))
            except (ValueError, TypeError):
                reasons.add("invalid_market_eligibility")
        if len(kickoffs) > 1:
            reasons.add("conflicting_kickoff_revisions")
        if len(competitions) > 1 or len(seasons) > 1:
            reasons.add("conflicting_fixture_identity")
        if any(len(identities) > 1 for identities in teams.values()):
            reasons.add("conflicting_fixture_team_identities")
        if teams["home_team_id"] & teams["away_team_id"]:
            reasons.add("conflicting_fixture_team_identities")
        if reasons:
            invalid.append({"fixture_id": fixture_id, "reasons": sorted(reasons),
                            "kickoffs": sorted(kickoffs), "row_count": len(versions)})
            continue
        accepted.append({"fixture_id": fixture_id, "kickoff": next(iter(kickoffs)),
                         "competition": next(iter(competitions), "unknown"),
                         "season": next(iter(seasons), None), "eligible_markets": sorted(markets),
                         "completed": completed, "row_count": len(versions)})
    accepted.sort(key=lambda group: (group["kickoff"], group["fixture_id"]))
    invalid.sort(key=lambda item: (-1 if item["fixture_id"] is None else item["fixture_id"], _hash(item)))
    return accepted, invalid


def _months(value: datetime, count: int) -> datetime:
    """Only first-of-month calendar anchors are shifted by this contract."""
    month_index = value.year * 12 + value.month - 1 + count
    return value.replace(year=month_index // 12, month=month_index % 12 + 1, day=1)


def _semester_at_or_after(value: datetime) -> datetime:
    anchor = datetime(value.year, 1 if value.month <= 6 else 7, 1, tzinfo=timezone.utc)
    return anchor if value == anchor else _months(anchor, 6)


def choose_boundaries(rows: Iterable[Mapping]) -> dict:
    """Choose July-anchored 12/6/12-month reserves from eligible FT metadata.

    No sufficiently late data means an explicit insufficient report, never a
    shortened holdout. Initial training seeks 24 elapsed calendar months.
    """
    groups, quarantined = _groups(rows)
    eligible = [group for group in groups if group["completed"] and group["eligible_markets"]]
    if not eligible:
        return {"version": SPLIT_VERSION, "status": "insufficient", "reasons": ["no_eligible_completed_fixtures"],
                "development_folds": [], "quarantined": quarantined}
    first, latest = _utc(eligible[0]["kickoff"]), _utc(eligible[-1]["kickoff"])
    end = datetime(latest.year, 7, 1, tzinfo=timezone.utc)
    if latest < end:
        end = end.replace(year=end.year - 1)
    final_start = _months(end, -12)
    calibration_start = _months(final_start, -6)
    confirmation_start = _months(calibration_start, -12)
    # Keep the exact first kickoff and conservatively round 24 months upward.
    minimum_end = first.replace(year=first.year + 2) if (first.month, first.day) != (2, 29) else first.replace(year=first.year + 2, day=28)
    development_start = _semester_at_or_after(minimum_end)
    reasons = []
    if development_start >= confirmation_start:
        reasons.append("insufficient_initial_training_and_development_history")
    folds, start = [], development_start
    while start < confirmation_start:
        finish = _months(start, 6)
        folds.append({"fold_id": f"development-{len(folds) + 1:02d}", "fit_cutoff": _iso(start),
                      "validation_start": _iso(start), "validation_end": _iso(finish),
                      "earlier_inner_blocks": len(folds), "has_two_earlier_inner_blocks": len(folds) >= 2})
        start = finish
    if not any(fold["has_two_earlier_inner_blocks"] for fold in folds):
        reasons.append("insufficient_nested_development_blocks")
    result = {"version": SPLIT_VERSION, "status": "insufficient" if reasons else "defined",
              "reasons": reasons, "initial_training_start": _iso(first),
              "development_start": _iso(development_start), "phase3_confirmation_start": _iso(confirmation_start),
              "calibration_start": _iso(calibration_start), "final_system_start": _iso(final_start),
              "final_system_end": _iso(end), "latest_eligible_kickoff": _iso(latest),
              "development_folds": folds, "quarantined": quarantined}
    result["sha256"] = _hash(result)
    return result


def _boundaries(boundaries: Mapping) -> dict[str, datetime]:
    if boundaries.get("version") != SPLIT_VERSION:
        raise ValueError("unsupported split contract version")
    if boundaries.get("sha256") and boundaries["sha256"] != _hash({key: value for key, value in boundaries.items() if key != "sha256"}):
        raise ValueError("split boundary manifest hash mismatch")
    parsed = {key: _utc(boundaries[key]) for key in BOUNDARY_KEYS}
    reserves = [parsed[key] for key in BOUNDARY_KEYS[2:]]
    if reserves != sorted(reserves) or len(set(reserves)) != 4:
        raise ValueError("reserve boundaries must be strictly increasing")
    if parsed["development_start"] < parsed["initial_training_start"]:
        raise ValueError("development cannot precede initial training")
    end = parsed["final_system_end"]
    if end != datetime(end.year, 7, 1, tzinfo=timezone.utc):
        raise ValueError("final system end must be the July 1 UTC calendar anchor")
    if (parsed["final_system_start"] != _months(end, -12)
            or parsed["calibration_start"] != _months(end, -18)
            or parsed["phase3_confirmation_start"] != _months(end, -30)):
        raise ValueError("split contract requires twelve/six/twelve-month reserves")
    return parsed


def _partition(kickoff: datetime, dates: Mapping[str, datetime]) -> str:
    # Reserve boundaries take precedence even if early development is too short.
    if kickoff >= dates["final_system_end"]:
        return "prospective_reserve"
    if kickoff >= dates["final_system_start"]:
        return "final_system_test"
    if kickoff >= dates["calibration_start"]:
        return "calibration"
    if kickoff >= dates["phase3_confirmation_start"]:
        return "phase3_confirmation"
    return "development" if kickoff >= dates["development_start"] else "initial_training"


def _coverage(membership: list[dict], *, competitions: Iterable[str] = ()) -> dict:
    buckets = defaultdict(list)
    competitions = set(competitions) | {group["competition"] for group in membership}
    for partition in PARTITIONS:
        for market in MARKETS:
            buckets[(partition, market, "all", "all")] = []
            for competition in sorted(competitions):
                buckets[(partition, market, competition, "all")] = []
    # A season present in a partition still needs visible zero-market coverage.
    for group in membership:
        for market in MARKETS:
            buckets[(group["partition"], market, group["competition"], str(group["season"]))] = []
    for group in membership:
        for market in group["eligible_markets"]:
            for competition, season in (("all", "all"), (group["competition"], "all"),
                                         (group["competition"], str(group["season"]))):
                buckets[(group["partition"], market, competition, season)].append(group)
    output = []
    for (partition, market, competition, season), groups in sorted(buckets.items()):
        dates = {_utc(group["kickoff"]).date() for group in groups}
        weeks = {(date.isocalendar().year, date.isocalendar().week) for date in dates}
        n = len(groups)
        record = {"partition": partition, "market": market, "competition": competition, "season": season,
                  "unique_fixtures": n, "match_dates": len(dates), "calendar_weeks": len(weeks)}
        if competition == "all":
            record["minimum_sample_gate"] = (n >= 1000 and len(weeks) >= 20) if partition == "phase3_confirmation" else None
        else:
            record["minimum_slice_sample_gate"] = n >= 200 and len(dates) >= 20
        output.append(record)
    return {"slices": output, "note": "Counts are unique fixtures; sample gates do not establish accuracy or qualification."}


def _fold_coverage(rows: list[Mapping], membership: list[dict], boundaries: Mapping) -> list[dict]:
    """Raw fitting support; weighted ESS remains a later recipe-specific gate."""
    valid = {group["fixture_id"] for group in membership}
    observations = defaultdict(list)
    modes = {row.get("availability") for row in rows if isinstance(row, Mapping) and row.get("availability") is not None}
    mixed = len(modes) > 1
    for row in rows:
        try:
            fixture_id = _metadata(row).get("fixture_id")
            if type(fixture_id) is not int or fixture_id not in valid or not _completed(row):
                continue
            available = label_available_at(row, availability=row.get("availability"))
            for market in _eligible(row):
                observations[market].append((fixture_id, available))
        except (ValueError, TypeError, OverflowError, AttributeError):
            continue  # No invented vintage: unsupported fitting counts stay low.
    output = []
    for fold in boundaries.get("development_folds", []):
        start, end = _utc(fold["validation_start"]), _utc(fold["validation_end"])
        if (_utc(fold["fit_cutoff"]) != start or end != _months(start, 6)
                or start < _utc(boundaries["development_start"])
                or end > _utc(boundaries["phase3_confirmation_start"])):
            raise ValueError("invalid development fold boundaries")
        record = dict(fold, markets={})
        for market in MARKETS:
            fitting = {fixture_id for fixture_id, available in observations[market] if available < start} if not mixed else set()
            evaluation = [group for group in membership if group["partition"] == "development"
                          and market in group["eligible_markets"] and start <= _utc(group["kickoff"]) < end]
            record["markets"][market] = {
                "training_unique_fixtures": len(fitting), "validation_unique_fixtures": len(evaluation),
                "minimum_pooled_training_count": len(fitting) >= 2000,
                "minimum_tree_training_count": len(fitting) >= 5000,
                "minimum_validation_count": len(evaluation) >= 500,
                "weighted_ess_gate": "not_evaluated_until_recipe_fitting",
                "availability_issue": "mixed_availability_cohorts" if mixed else None,
            }
        output.append(record)
    return output


def assign_partitions(rows: Iterable[Mapping], boundaries: Mapping) -> dict:
    """Return deterministic fixture memberships, quarantines and count reports."""
    frozen = list(rows)
    groups, quarantined = _groups(frozen)
    if "no_eligible_completed_fixtures" in boundaries.get("reasons", []):
        if boundaries.get("version") != SPLIT_VERSION:
            raise ValueError("unsupported split contract version")
        if any(group["eligible_markets"] for group in groups):
            raise ValueError("empty boundary manifest conflicts with eligible rows")
        result = {"version": SPLIT_VERSION, "status": "insufficient",
                  "reasons": ["no_eligible_completed_fixtures"], "boundaries": {},
                  "memberships": [], "partition_by_fixture": {}, "quarantined": quarantined,
                  "unassigned_fixture_ids": [group["fixture_id"] for group in groups],
                  "coverage": _coverage([], competitions={group["competition"] for group in groups}),
                  "development_folds": []}
        result["membership_sha256"] = _hash([])
        result["sha256"] = _hash(result)
        return result
    dates = _boundaries(boundaries)
    membership = [dict(group, partition=_partition(_utc(group["kickoff"]), dates)) for group in groups]
    result = {"version": SPLIT_VERSION, "status": boundaries.get("status", "defined"),
              "reasons": list(boundaries.get("reasons", [])),
              "boundaries": {key: _iso(value) for key, value in dates.items()},
              "memberships": membership, "partition_by_fixture": {str(group["fixture_id"]): group["partition"] for group in membership},
              "quarantined": quarantined, "coverage": _coverage(membership),
              "development_folds": _fold_coverage(frozen, membership, boundaries)}
    result["membership_sha256"] = _hash(membership)
    result["sha256"] = _hash(result)
    return result


def label_available_at(row: Mapping, *, availability: str) -> datetime:
    """Return the declared conservative label time, rejecting vintage mixing."""
    if availability not in {"assumed_final", "observed"}:
        raise ValueError("unsupported availability mode")
    declared = row.get("availability")
    if declared is not None and declared != availability:
        raise ValueError("availability_mode_mismatch")
    available = _utc(_metadata(row).get("kickoff")) + timedelta(hours=3)
    if availability == "observed":
        observed = [mapping[key] for mapping in (row, _metadata(row))
                    for key in ("observed_at", "label_observed_at") if mapping.get(key) is not None]
        if not observed:
            raise ValueError("missing_observed_label_time")
        available = max(available, *(_utc(value) for value in observed))
    if row.get("label_available_at") is not None and _utc(row["label_available_at"]) != available:
        raise ValueError("label_availability_contract_mismatch")
    return available


def training_rows(rows: Iterable[Mapping], cutoff, *, availability: str, market: str | None = None) -> dict:
    """Filter by explicit market eligibility and strictly earlier label time.

    This is a temporal filter, not permission to load a held-out target store.
    The caller must first enforce ``assert_development_access`` on its store.
    Original row objects are returned without reading or modifying targets.
    """
    if availability not in {"assumed_final", "observed"}:
        raise ValueError("unsupported availability mode")
    if market is not None and market not in MARKETS:
        raise ValueError("unsupported market")
    frozen = list(rows)
    cutoff = _utc(cutoff)
    groups, quarantined = _groups(frozen)
    valid = {group["fixture_id"] for group in groups}
    kept, excluded = [], list(quarantined)
    for row in frozen:
        try:
            fixture_id = _metadata(row).get("fixture_id")
        except (ValueError, TypeError, AttributeError):
            continue
        if type(fixture_id) is not int or fixture_id not in valid:
            continue
        reason = None
        markets = _eligible(row)
        if not _completed(row):
            reason = "not_completed"
        elif not markets or (market is not None and market not in markets):
            reason = "market_not_eligible"
        else:
            try:
                if label_available_at(row, availability=availability) >= cutoff:
                    reason = "label_available_after_or_at_fit"
            except (ValueError, TypeError, OverflowError) as exc:
                reason = str(exc)
        if reason:
            excluded.append({"fixture_id": fixture_id, "reasons": [reason]})
        else:
            kept.append(row)
    def key(row):
        return (_iso(_utc(_metadata(row)["kickoff"])), _metadata(row)["fixture_id"],
                tuple(sorted(_eligible(row))), str(row.get("forecast_stage", "")),
                str(row.get("revision", "")), str(row.get("as_of", "")),
                _iso(label_available_at(row, availability=availability)))
    kept.sort(key=key)
    excluded.sort(key=lambda item: (-1 if item["fixture_id"] is None else item["fixture_id"], _hash(item)))
    metadata = [{"fixture_id": _metadata(row)["fixture_id"], "kickoff": _iso(_utc(_metadata(row)["kickoff"])),
                 "eligible_markets": sorted(_eligible(row)), "available_at": _iso(label_available_at(row, availability=availability))}
                for row in kept]
    return {"version": SPLIT_VERSION, "availability": availability, "cutoff": _iso(cutoff),
            "market": market, "rows": kept, "excluded": excluded, "membership_sha256": _hash(metadata)}


def assert_development_access(partition: str) -> None:
    """Fail closed before opening any confirmation/calibration/later target store."""
    if partition not in {"initial_training", "development"}:
        raise PermissionError(f"Development cannot access target partition {partition!r}")
