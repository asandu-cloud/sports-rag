#!/usr/bin/env python3
"""Chronological performance and calibration reporting for recorded picks.

This tool evaluates decisions that were actually captured before kick-off. It
does *not* reconstruct old odds or profiles from today's database and call
that a backtest. Until immutable historical input snapshots exist, a recorded
shadow/live decision log is the trustworthy way to evaluate the selector's
probabilities, pricing and release readiness.

Examples:
  python Scripts/rag_ingest/walk_forward.py \
      --predictions-db Index/predictions.db --source canonical_shadow
  python Scripts/rag_ingest/walk_forward.py \
      --records Index/shadow_runs/canonical_shadow.jsonl --strict
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from data_platform.outcomes import outcome_accounting


@dataclass(frozen=True)
class RecordedDecision:
    """Minimum immutable fields required for chronological evaluation."""

    record_id: str
    source: str
    league: str
    market: str
    model_probability: Optional[float]
    odds: Optional[float]
    outcome: Optional[str]
    generated_at: Optional[str] = None
    kickoff: Optional[str] = None
    prediction_date: Optional[str] = None


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp(value: Any) -> Optional[datetime]:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text[:10])
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _outcome(value: Any) -> Optional[str]:
    text = str(value or "").lower().strip()
    return text or None


def _candidate_from_snapshot(snapshot: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return the candidate decision and its containing MarketResult payload."""
    result = snapshot.get("result")
    result = result if isinstance(result, Mapping) else snapshot
    context = result.get("context") if isinstance(result, Mapping) else {}
    context = context if isinstance(context, Mapping) else {}
    candidate = snapshot.get("candidate") or context.get("release_candidate")
    if not isinstance(candidate, Mapping):
        candidate = result.get("decision") if isinstance(result, Mapping) else {}
    return candidate if isinstance(candidate, Mapping) else {}, result if isinstance(result, Mapping) else {}


def record_from_mapping(payload: Mapping[str, Any], *, fallback_id: str = "") -> Optional[RecordedDecision]:
    """Normalise a shadow JSONL record or a flat tracker/database row."""
    candidate, result = _candidate_from_snapshot(payload)
    fixture = result.get("fixture") if isinstance(result.get("fixture"), Mapping) else {}
    market_ref = result.get("market") if isinstance(result.get("market"), Mapping) else {}
    provenance = result.get("provenance") if isinstance(result.get("provenance"), Mapping) else {}
    quote = candidate.get("quote") if isinstance(candidate.get("quote"), Mapping) else {}

    outcome = _outcome(payload.get("outcome"))
    if outcome is None:
        outcome = _outcome(candidate.get("outcome"))
    model_probability = _number(
        payload.get("model_prob", candidate.get("model_probability"))
    )
    odds = _number(payload.get("odds", quote.get("odds")))
    market = str(payload.get("market") or market_ref.get("group") or "").strip().lower()
    source = str(payload.get("source") or "canonical_shadow").strip()
    record_id = str(
        payload.get("record_key")
        or payload.get("id")
        or provenance.get("input_snapshot_id")
        or fallback_id
    ).strip()
    if not record_id or not market:
        return None
    return RecordedDecision(
        record_id=record_id,
        source=source,
        league=str(payload.get("league") or fixture.get("league") or "").strip(),
        market=market,
        model_probability=model_probability,
        odds=odds,
        outcome=outcome,
        generated_at=str(
            payload.get("captured_at")
            or payload.get("generated_at")
            or provenance.get("generated_at")
            or ""
        ).strip() or None,
        kickoff=str(payload.get("kickoff") or fixture.get("kickoff") or "").strip() or None,
        prediction_date=str(payload.get("prediction_date") or "").strip() or None,
    )


def load_jsonl_records(path: Path) -> List[RecordedDecision]:
    """Load a newline-delimited canonical shadow archive."""
    records: List[RecordedDecision] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw_line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"Expected an object on {path}:{line_number}")
        record = record_from_mapping(payload, fallback_id=f"{path.name}:{line_number}")
        if record is not None:
            records.append(record)
    return records


def load_tracker_records(db_path: Path, *, sources: Sequence[str] = ()) -> List[RecordedDecision]:
    """Load recorded picks from the existing SQLite prediction tracker."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        query = "SELECT * FROM predictions"
        params: List[Any] = []
        if sources:
            query += " WHERE source IN (" + ",".join("?" for _ in sources) + ")"
            params.extend(sources)
        query += " ORDER BY prediction_date ASC, id ASC"
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()
    return [
        record
        for row in rows
        if (record := record_from_mapping(dict(row), fallback_id=f"tracker:{row['id']}")) is not None
    ]


def _chronological_key(record: RecordedDecision) -> Tuple[datetime, str]:
    value = _timestamp(record.generated_at) or _timestamp(record.prediction_date)
    return value or datetime.min.replace(tzinfo=timezone.utc), record.record_id


def _settlement(outcome: Optional[str], odds: Optional[float]) -> Optional[Dict[str, float]]:
    """Return the financial and calibration contribution of one settled pick."""
    contribution = outcome_accounting(outcome, odds)
    if contribution is None:
        return None
    return {
        "stake": contribution.stake_units,
        "return": contribution.returned_units,
        "resolved": contribution.resolved_units,
        "wins": contribution.win_units,
    }


def _bucket_label(index: int, width: float) -> str:
    low = index * width
    high = min(1.0, low + width)
    return f"{low:.0%}-{high:.0%}"


def _summary(records: Iterable[RecordedDecision], *, bucket_width: float = 0.05) -> Dict[str, Any]:
    if bucket_width <= 0 or bucket_width > 1:
        raise ValueError("bucket_width must be between 0 and 1.")
    rows = list(records)
    graded_rows: List[Tuple[RecordedDecision, Dict[str, float]]] = []
    pending = 0
    unknown_outcomes = 0
    for record in rows:
        settled = _settlement(record.outcome, record.odds)
        if settled is None:
            if record.outcome is None:
                pending += 1
            else:
                unknown_outcomes += 1
            continue
        graded_rows.append((record, settled))

    stake = sum(item["stake"] for _, item in graded_rows)
    returned = sum(item["return"] for _, item in graded_rows)
    resolved = sum(item["resolved"] for _, item in graded_rows)
    wins = sum(item["wins"] for _, item in graded_rows)
    profit = returned - stake
    outcome_counts: Dict[str, int] = defaultdict(int)
    for record, _ in graded_rows:
        if record.outcome:
            outcome_counts[record.outcome] += 1

    bucket_count = max(1, int(math.ceil(1.0 / bucket_width)))
    buckets: Dict[int, Dict[str, float]] = {}
    brier_weight = 0.0
    brier_total = 0.0
    for record, settlement in graded_rows:
        probability = record.model_probability
        if probability is None or not 0.0 <= probability <= 1.0 or settlement["resolved"] <= 0:
            continue
        index = min(bucket_count - 1, int(probability / bucket_width))
        bucket = buckets.setdefault(index, {
            "probability_weight": 0.0,
            "resolved_weight": 0.0,
            "win_weight": 0.0,
            "brier_weight": 0.0,
        })
        weight = settlement["resolved"]
        observed = settlement["wins"] / weight
        bucket["probability_weight"] += probability * weight
        bucket["resolved_weight"] += weight
        bucket["win_weight"] += settlement["wins"]
        bucket["brier_weight"] += (probability - observed) ** 2 * weight
        brier_weight += weight
        brier_total += (probability - observed) ** 2 * weight

    calibration = []
    for index in sorted(buckets):
        bucket = buckets[index]
        weight = bucket["resolved_weight"]
        predicted = bucket["probability_weight"] / weight
        observed = bucket["win_weight"] / weight
        calibration.append({
            "bucket": _bucket_label(index, bucket_width),
            "model_probability": predicted,
            "observed_rate": observed,
            "calibration_gap": observed - predicted,
            "sample_size": weight,
        })

    return {
        "total_records": len(rows),
        "graded_records": len(graded_rows),
        "pending_records": pending,
        "unknown_outcomes": unknown_outcomes,
        "outcomes": dict(sorted(outcome_counts.items())),
        "stake_units": stake,
        "profit_units": profit,
        "roi": profit / stake if stake else None,
        "resolved_stake_units": resolved,
        "win_stake_units": wins,
        "hit_rate": wins / resolved if resolved else None,
        "brier_score": brier_total / brier_weight if brier_weight else None,
        "calibration": calibration,
    }


def pre_match_violations(records: Iterable[RecordedDecision]) -> List[Dict[str, str]]:
    """List snapshots whose capture time was not strictly before kick-off."""
    violations = []
    for record in records:
        generated = _timestamp(record.generated_at)
        kickoff = _timestamp(record.kickoff)
        if generated is not None and kickoff is not None and generated >= kickoff:
            violations.append({
                "record_id": record.record_id,
                "generated_at": record.generated_at or "",
                "kickoff": record.kickoff or "",
            })
    return violations


def build_walk_forward_report(records: Iterable[RecordedDecision], *, bucket_width: float = 0.05) -> Dict[str, Any]:
    """Build an auditable chronological report without changing any weights."""
    ordered = sorted(records, key=_chronological_key)
    by_market: Dict[str, List[RecordedDecision]] = defaultdict(list)
    by_league: Dict[str, List[RecordedDecision]] = defaultdict(list)
    by_source: Dict[str, List[RecordedDecision]] = defaultdict(list)
    for record in ordered:
        by_market[record.market or "unknown"].append(record)
        by_league[record.league or "unknown"].append(record)
        by_source[record.source or "unknown"].append(record)
    return {
        "schema_version": "canonical-walk-forward-report.v1",
        "chronological": True,
        "first_recorded_at": ordered[0].generated_at if ordered else None,
        "last_recorded_at": ordered[-1].generated_at if ordered else None,
        "pre_match_violations": pre_match_violations(ordered),
        "overall": _summary(ordered, bucket_width=bucket_width),
        "by_market": {key: _summary(value, bucket_width=bucket_width) for key, value in sorted(by_market.items())},
        "by_league": {key: _summary(value, bucket_width=bucket_width) for key, value in sorted(by_league.items())},
        "by_source": {key: _summary(value, bucket_width=bucket_width) for key, value in sorted(by_source.items())},
    }


def release_readiness(
    report: Mapping[str, Any],
    *,
    minimum_resolved_stake: float = 50.0,
    minimum_bucket_sample: float = 20.0,
    maximum_calibration_gap: float = 0.10,
) -> Dict[str, Any]:
    """Turn an analysis report into a deliberately conservative promotion gate."""
    overall = report.get("overall") if isinstance(report.get("overall"), Mapping) else {}
    reasons = []
    if report.get("pre_match_violations"):
        reasons.append("One or more records were captured at or after kick-off.")
    resolved = _number(overall.get("resolved_stake_units")) or 0.0
    if resolved < minimum_resolved_stake:
        reasons.append(
            f"Only {resolved:.1f} resolved stake units; need at least {minimum_resolved_stake:.1f}."
        )
    calibration = overall.get("calibration") if isinstance(overall.get("calibration"), list) else []
    qualifying = [
        row for row in calibration
        if isinstance(row, Mapping) and (_number(row.get("sample_size")) or 0.0) >= minimum_bucket_sample
    ]
    if not qualifying:
        reasons.append(
            f"No probability bucket has at least {minimum_bucket_sample:.1f} resolved stake units."
        )
    for bucket in qualifying:
        gap = abs(_number(bucket.get("calibration_gap")) or 0.0)
        if gap > maximum_calibration_gap:
            reasons.append(
                f"Calibration gap {gap:.1%} exceeds {maximum_calibration_gap:.1%} in {bucket.get('bucket', 'a bucket')}."
            )
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "minimum_resolved_stake": minimum_resolved_stake,
        "minimum_bucket_sample": minimum_bucket_sample,
        "maximum_calibration_gap": maximum_calibration_gap,
    }


def _print_report(report: Mapping[str, Any], readiness: Mapping[str, Any]) -> None:
    overall = report["overall"]
    print("Canonical recorded-decision walk-forward report")
    print(
        f"Records {overall['total_records']} | graded {overall['graded_records']} | "
        f"pending {overall['pending_records']} | ROI "
        f"{overall['roi'] if overall['roi'] is not None else 'n/a'} | "
        f"hit rate {overall['hit_rate'] if overall['hit_rate'] is not None else 'n/a'}"
    )
    if overall["calibration"]:
        print("Calibration (model / observed / gap / sample):")
        for bucket in overall["calibration"]:
            print(
                f"  {bucket['bucket']}: {bucket['model_probability']:.1%} / "
                f"{bucket['observed_rate']:.1%} / {bucket['calibration_gap']:+.1%} / "
                f"{bucket['sample_size']:.1f}"
            )
    if report["pre_match_violations"]:
        print(f"Pre-match violations: {len(report['pre_match_violations'])}")
    print("Release gate: " + ("PASS" if readiness["eligible"] else "HOLD"))
    for reason in readiness["reasons"]:
        print(f"  - {reason}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Report chronological selector performance from recorded picks.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--records", type=Path, help="JSONL archive written by shadow_release.py")
    source.add_argument("--predictions-db", type=Path, help="SQLite predictions.db path")
    parser.add_argument("--source", action="append", default=[], help="Filter tracker rows by source; may be repeated")
    parser.add_argument("--bucket-width", type=float, default=0.05)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when the release gate is not met")
    parser.add_argument("--minimum-resolved-stake", type=float, default=50.0)
    parser.add_argument("--minimum-bucket-sample", type=float, default=20.0)
    parser.add_argument("--maximum-calibration-gap", type=float, default=0.10)
    args = parser.parse_args(argv)

    records = (
        load_jsonl_records(args.records)
        if args.records is not None
        else load_tracker_records(args.predictions_db, sources=args.source)
    )
    report = build_walk_forward_report(records, bucket_width=args.bucket_width)
    readiness = release_readiness(
        report,
        minimum_resolved_stake=args.minimum_resolved_stake,
        minimum_bucket_sample=args.minimum_bucket_sample,
        maximum_calibration_gap=args.maximum_calibration_gap,
    )
    report["release_readiness"] = readiness
    _print_report(report, readiness)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if readiness["eligible"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
