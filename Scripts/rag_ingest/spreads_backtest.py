#!/usr/bin/env python3
"""Quick spread-performance report from the predictions tracker.

This is not a historical odds replay harness. It summarizes logged spread picks
from `Index/predictions.db`, dedupes same-day identical picks, and reports hit
rate / ROI slices so spread regressions are easier to spot.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "Index" / "predictions.db"


def _load_rows(db_path: Path, league: str | None = None) -> List[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    query = """
        SELECT *
        FROM predictions
        WHERE market = 'spreads'
          AND outcome IS NOT NULL
    """
    params: List[str] = []
    if league:
        query += " AND league = ?"
        params.append(league)
    query += " ORDER BY prediction_date, id"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def _dedupe(rows: Iterable[sqlite3.Row]) -> List[sqlite3.Row]:
    deduped: Dict[Tuple[str, str, str, str, str], sqlite3.Row] = {}
    for row in rows:
        key = (
            row["prediction_date"] or "",
            row["home_team"] or "",
            row["away_team"] or "",
            row["pick"] or "",
            row["source"] or "",
        )
        deduped.setdefault(key, row)
    return list(deduped.values())


def _roi(rows: Iterable[sqlite3.Row]) -> float:
    total = 0.0
    count = 0
    for row in rows:
        odds = row["odds"]
        outcome = (row["outcome"] or "").lower()
        if odds is None or outcome not in {"hit", "miss", "push"}:
            continue
        count += 1
        if outcome == "hit":
            total += float(odds) - 1.0
        elif outcome == "miss":
            total -= 1.0
    return (total / count) if count else 0.0


def _bucket(rows: Iterable[sqlite3.Row], key_fn) -> Dict[str, List[sqlite3.Row]]:
    out: Dict[str, List[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        out[key_fn(row)].append(row)
    return dict(out)


def _confidence_bucket(row: sqlite3.Row) -> str:
    return (row["confidence"] or "unknown").lower()


def _line_shape(row: sqlite3.Row) -> str:
    side = (row["side"] or "").lower()
    line = row["line"]
    if line is None:
        return "unknown"
    line = float(line)
    if side == "home":
        return "favorite_minus" if line < 0 else "dog_plus"
    if side == "away":
        return "favorite_minus" if line < 0 else "dog_plus"
    return "unknown"


def _summarize(rows: List[sqlite3.Row], label: str) -> None:
    hits = sum(1 for row in rows if (row["outcome"] or "").lower() == "hit")
    misses = sum(1 for row in rows if (row["outcome"] or "").lower() == "miss")
    pushes = sum(1 for row in rows if (row["outcome"] or "").lower() == "push")
    total = len(rows)
    hit_rate = (hits / (hits + misses)) if (hits + misses) else 0.0
    print(f"{label}: n={total} | hit={hits} miss={misses} push={pushes} | hit_rate={hit_rate:.1%} | roi={_roi(rows):+.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize logged spread predictions.")
    parser.add_argument("--league", help="Optional league code filter, e.g. EPL")
    parser.add_argument("--db-path", default=str(DB_PATH), help="Path to predictions.db")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    rows = _load_rows(db_path, args.league)
    deduped = _dedupe(rows)

    print("Spread tracker summary")
    print(f"DB: {db_path}")
    print(f"League filter: {args.league or 'ALL'}")
    print()
    _summarize(rows, "Raw graded")
    _summarize(deduped, "Deduped")
    print()

    for label, bucket_fn in (
        ("Confidence", _confidence_bucket),
        ("Side shape", _line_shape),
        ("Source", lambda row: (row["source"] or "unknown").lower()),
    ):
        print(f"{label} slices")
        buckets = _bucket(deduped, bucket_fn)
        for bucket_name in sorted(buckets):
            _summarize(buckets[bucket_name], f"  {bucket_name}")
        print()


if __name__ == "__main__":
    main()
