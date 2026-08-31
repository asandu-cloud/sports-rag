#!/usr/bin/env python3
"""Capture canonical recommendations in shadow mode without publishing them.

The command evaluates the same delivery boundary used by Discord and the
website, forces ``shadow`` mode for this process, records every evaluation as
an immutable JSONL snapshot, and logs recommended candidates to the existing
prediction tracker as ``canonical_shadow`` for later settlement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from core.release_control import temporary_release_mode
from prediction_tracker import DB_PATH, log_prediction
from rendering.market_dispatch import evaluate_market_results_sync


SHADOW_SOURCE = "canonical_shadow"
SHADOW_RECORD_SCHEMA = "canonical-shadow-decision.v1"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR.parents[1] / "Index" / "shadow_runs" / "canonical_shadow.jsonl"
DEFAULT_MARKETS = ("totals", "corners", "cards", "sot", "btts", "moneyline", "spreads")


def _season(target_date: date) -> str:
    first_year = target_date.year if target_date.month >= 7 else target_date.year - 1
    return f"{first_year}/{str(first_year + 1)[2:]}"


def _pick_label(quote: Mapping[str, Any]) -> str:
    side = str(quote.get("side") or "Pick").strip()
    line = quote.get("line")
    if line is None:
        return side.title()
    try:
        return f"{side.title()} {float(line):g}"
    except (TypeError, ValueError):
        return f"{side.title()} {line}"


def _record_key(result: Mapping[str, Any], candidate: Optional[Mapping[str, Any]]) -> str:
    fixture = result.get("fixture") if isinstance(result.get("fixture"), Mapping) else {}
    market = result.get("market") if isinstance(result.get("market"), Mapping) else {}
    provenance = result.get("provenance") if isinstance(result.get("provenance"), Mapping) else {}
    payload = {
        "event_id": fixture.get("event_id"),
        "market": market.get("group"),
        "snapshot": provenance.get("input_snapshot_id"),
        "candidate": dict(candidate or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def _existing_record_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Shadow archive has invalid JSON on {path}:{line_number}") from exc
        key = str(payload.get("record_key") or "").strip()
        if key:
            keys.add(key)
    return keys


def append_shadow_records(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Append new immutable records, deduplicated by fixture/market/snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_record_keys(path)
    additions = []
    for record in records:
        key = str(record.get("record_key") or "")
        if not key or key in existing:
            continue
        additions.append(json.dumps(dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False))
        existing.add(key)
    if additions:
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(additions) + "\n")
    return len(additions)


def _candidate_from_result(result: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    context = result.get("context") if isinstance(result.get("context"), Mapping) else {}
    candidate = context.get("release_candidate") if isinstance(context, Mapping) else None
    return candidate if isinstance(candidate, Mapping) else None


def _log_candidate(
    result: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    target_date: date,
    db_path: Path,
) -> int:
    fixture = result.get("fixture") if isinstance(result.get("fixture"), Mapping) else {}
    market = result.get("market") if isinstance(result.get("market"), Mapping) else {}
    projection = result.get("projection") if isinstance(result.get("projection"), Mapping) else {}
    quote = candidate.get("quote") if isinstance(candidate.get("quote"), Mapping) else {}
    return log_prediction(
        home_team=str(fixture.get("home_team") or ""),
        away_team=str(fixture.get("away_team") or ""),
        league=str(fixture.get("league") or ""),
        market=str(market.get("group") or ""),
        pick=_pick_label(quote),
        side=str(quote.get("side") or ""),
        line=quote.get("line"),
        odds=quote.get("odds"),
        bookmaker=quote.get("bookmaker"),
        model_prob=candidate.get("model_probability"),
        implied_prob=candidate.get("implied_probability"),
        value_edge=candidate.get("value_edge"),
        confidence=candidate.get("confidence"),
        projected_total=projection.get("value"),
        source=SHADOW_SOURCE,
        fixture_id=str(fixture.get("event_id") or ""),
        kickoff=fixture.get("kickoff"),
        season=_season(target_date),
        db_path=db_path,
    )


def capture_shadow_run(
    *,
    leagues: Sequence[str],
    markets: Sequence[str],
    target_date: date,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    db_path: Path = DB_PATH,
    evaluator: Callable[[str, str, date], Tuple[List[Any], List[str]]] = evaluate_market_results_sync,
) -> Dict[str, Any]:
    """Evaluate, capture, and tracker-log one shadow run without publishing picks."""
    captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    records: List[Dict[str, Any]] = []
    tracker_logged = 0
    notes: List[str] = []
    with temporary_release_mode("shadow"):
        for league in leagues:
            for market in markets:
                results, evaluation_notes = evaluator(league, market, target_date)
                notes.extend(str(note) for note in evaluation_notes)
                for market_result in results:
                    result = market_result.to_dict()
                    candidate = _candidate_from_result(result)
                    record = {
                        "schema_version": SHADOW_RECORD_SCHEMA,
                        "record_key": _record_key(result, candidate),
                        "captured_at": captured_at,
                        "target_date": target_date.isoformat(),
                        "source": SHADOW_SOURCE,
                        "outcome": None,
                        "candidate": dict(candidate) if candidate else None,
                        "result": result,
                    }
                    records.append(record)
                    if candidate is not None and _log_candidate(
                        result, candidate, target_date=target_date, db_path=db_path,
                    ) > 0:
                        tracker_logged += 1
    written = append_shadow_records(output_path, records)
    return {
        "evaluated": len(records),
        "candidates": sum(1 for record in records if record["candidate"] is not None),
        "snapshots_written": written,
        "tracker_logged": tracker_logged,
        "notes": notes,
        "output_path": str(output_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Capture canonical candidates in non-publishing shadow mode.")
    parser.add_argument("--league", action="append", dest="leagues", required=True,
                        help="League code to capture; may be repeated")
    parser.add_argument("--market", action="append", dest="markets",
                        choices=DEFAULT_MARKETS, help="Canonical market; may be repeated")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(), help="Fixture date (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--predictions-db", type=Path, default=DB_PATH)
    args = parser.parse_args(argv)

    result = capture_shadow_run(
        leagues=args.leagues,
        markets=args.markets or DEFAULT_MARKETS,
        target_date=args.date,
        output_path=args.output,
        db_path=args.predictions_db,
    )
    print(
        f"Shadow run: evaluated {result['evaluated']} market results; "
        f"captured {result['candidates']} candidates; wrote {result['snapshots_written']} snapshots; "
        f"logged {result['tracker_logged']} tracker rows."
    )
    if result["notes"]:
        print("Notes: " + "; ".join(result["notes"][:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
