"""Backend-independent reporting contract for prediction tracking.

Both the legacy SQLite tracker and the canonical platform use these pure
functions.  They accept dictionaries rather than ORM rows so reporting stays
identical while the storage backend changes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .outcomes import (
    HALF_HIT,
    HALF_MISS,
    HIT,
    MISS,
    PUSH,
    VOID,
    is_valid_decimal_odds,
    normalize_outcome,
    outcome_accounting,
)


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _date_text(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value or "")[:10]


def _normalise_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Copy generic rows and make date/order fields safe for reporting."""
    normalised: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["prediction_date"] = _date_text(item.get("prediction_date"))
        item["_tracking_index"] = index
        normalised.append(item)
    return normalised


def _ordered(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        _normalise_rows(rows),
        key=lambda item: (item.get("prediction_date", ""), item.get("id") or item["_tracking_index"]),
    )


def outcome_totals(rows: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    """Counts plus weighted resolved/win units for a set of predictions."""
    totals: Dict[str, float] = {
        "hits": 0,
        "half_hits": 0,
        "misses": 0,
        "half_misses": 0,
        "pushes": 0,
        "unknown_outcomes": 0,
        "win_units": 0.0,
        "resolved_units": 0.0,
    }
    for row in rows:
        canonical = normalize_outcome(row.get("outcome"))
        if canonical == HIT:
            totals["hits"] += 1
        elif canonical == HALF_HIT:
            totals["half_hits"] += 1
        elif canonical == MISS:
            totals["misses"] += 1
        elif canonical == HALF_MISS:
            totals["half_misses"] += 1
        elif canonical in {PUSH, VOID}:
            totals["pushes"] += 1
        else:
            totals["unknown_outcomes"] += 1
            continue
        contribution = outcome_accounting(canonical)
        if contribution is not None:
            totals["win_units"] += contribution.win_units
            totals["resolved_units"] += contribution.resolved_units
    return totals


def flat_stake_roi(rows: Iterable[Mapping[str, Any]]) -> Tuple[float, float, float]:
    """Return ``(roi, stake_units, returned_units)`` for recorded odds only."""
    stake = 0.0
    returned = 0.0
    for row in rows:
        if not is_valid_decimal_odds(row.get("odds")):
            continue
        contribution = outcome_accounting(row.get("outcome"), _number(row.get("odds")))
        if contribution is None:
            continue
        stake += contribution.stake_units
        returned += contribution.returned_units
    return ((returned - stake) / stake if stake else 0.0), stake, returned


def _clv_summary(rows: Iterable[Mapping[str, Any]]) -> Tuple[Optional[float], Optional[float], int]:
    values = [_number(row.get("clv")) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None, None, 0
    return sum(values) / len(values), sum(1 for value in values if value > 0) / len(values), len(values)


def _summary(rows: Sequence[Mapping[str, Any]], *, include_roi: bool = False) -> Dict[str, Any]:
    totals = outcome_totals(rows)
    result: Dict[str, Any] = {
        "total": len(rows),
        "hits": int(totals["hits"]),
        "half_hits": int(totals["half_hits"]),
        "misses": int(totals["misses"]),
        "half_misses": int(totals["half_misses"]),
        "pushes": int(totals["pushes"]),
        "hit_rate": totals["win_units"] / totals["resolved_units"] if totals["resolved_units"] else 0.0,
    }
    if include_roi:
        roi, _, _ = flat_stake_roi(rows)
        result["roi"] = roi
    return result


def _streaks(rows: Sequence[Mapping[str, Any]]) -> Tuple[int, int]:
    """Current decided full-result streak and historical best win streak."""
    decided = []
    for row in _ordered(rows):
        canonical = normalize_outcome(row.get("outcome"))
        if canonical in {HIT, MISS}:
            decided.append(canonical)
    if not decided:
        return 0, 0

    current = 0
    last = decided[-1]
    for outcome in reversed(decided):
        if outcome != last:
            break
        current += 1
    if last == MISS:
        current = -current

    best = 0
    run = 0
    for outcome in decided:
        if outcome == HIT:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return current, best


def _daily_performance(rows: Sequence[Mapping[str, Any]], *, days: int, today: date) -> List[Dict[str, Any]]:
    cutoff = (today - timedelta(days=days)).isoformat()
    by_date: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        day = _date_text(row.get("prediction_date"))
        if day and day >= cutoff:
            by_date[day].append(row)
    return [
        {
            "date": day,
            "total": summary["total"],
            "hits": summary["hits"],
            "hit_rate": summary["hit_rate"],
        }
        for day, summary in (
            (day, _summary(day_rows)) for day, day_rows in sorted(by_date.items())
        )
    ]


def build_calibration(
    rows: Iterable[Mapping[str, Any]],
    *,
    buckets: int = 20,
) -> List[Dict[str, Any]]:
    """Return the legacy calibration contract with Asian-line weighting.

    ``buckets=20`` preserves the original five-percentage-point reporting
    bands (for example ``"50-55%"``).  Pushes and voids are excluded because
    they contain no binary win/loss observation.
    """
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    width = 1.0 / buckets
    grouped: Dict[int, Dict[str, float]] = {}
    for row in rows:
        probability = _number(row.get("model_prob"))
        contribution = outcome_accounting(row.get("outcome"))
        if probability is None or not 0.0 <= probability <= 1.0:
            continue
        if contribution is None or contribution.resolved_units <= 0:
            continue
        index = min(buckets - 1, int(probability / width))
        entry = grouped.setdefault(index, {"probability": 0.0, "wins": 0.0, "resolved": 0.0})
        entry["probability"] += probability * contribution.resolved_units
        entry["wins"] += contribution.win_units
        entry["resolved"] += contribution.resolved_units

    result: List[Dict[str, Any]] = []
    for index in sorted(grouped):
        entry = grouped[index]
        low = int(round(index * width * 100))
        high = int(round(min(1.0, (index + 1) * width) * 100))
        result.append({
            "bucket": f"{low}-{high}%",
            "model_avg": entry["probability"] / entry["resolved"],
            "actual_rate": entry["wins"] / entry["resolved"],
            "count": entry["resolved"],
        })
    return result


def build_track_record(
    rows: Iterable[Mapping[str, Any]],
    *,
    today: Optional[date] = None,
    daily_days: int = 30,
) -> Dict[str, Any]:
    """Return the stable contract consumed by Discord and the website.

    The field names intentionally match the original ``prediction_tracker``
    interface.  Existing consumers therefore get correct platform metrics
    without a second API migration.
    """
    ordered = _ordered(rows)
    overall = _summary(ordered, include_roi=True)
    avg_clv, clv_positive_rate, clv_sample_size = _clv_summary(ordered)
    recent_streak, best_streak = _streaks(ordered)

    by_confidence: Dict[str, Any] = {}
    for level in ("high", "medium", "low"):
        subset = [row for row in ordered if str(row.get("confidence") or "").lower() == level]
        if subset:
            by_confidence[level] = _summary(subset, include_roi=True)

    by_market: Dict[str, Any] = {}
    for market in sorted({str(row.get("market") or "") for row in ordered} - {""}):
        subset = [row for row in ordered if str(row.get("market") or "") == market]
        summary = _summary(subset)
        summary["avg_clv"] = _clv_summary(subset)[0]
        by_market[market] = summary

    by_league: Dict[str, Any] = {}
    for league in sorted({str(row.get("league") or "") for row in ordered} - {""}):
        subset = [row for row in ordered if str(row.get("league") or "") == league]
        by_league[league] = _summary(subset)

    return {
        "total_graded": len(ordered),
        "hits": overall["hits"],
        "half_hits": overall["half_hits"],
        "misses": overall["misses"],
        "half_misses": overall["half_misses"],
        "pushes": overall["pushes"],
        "unknown_outcomes": outcome_totals(ordered)["unknown_outcomes"],
        "hit_rate": overall["hit_rate"],
        "roi_flat_stake": overall["roi"],
        "stake_units": flat_stake_roi(ordered)[1],
        "avg_clv": avg_clv,
        "clv_positive_rate": clv_positive_rate,
        "clv_sample_size": clv_sample_size,
        "by_confidence": by_confidence,
        "by_market": by_market,
        "by_league": by_league,
        "recent_streak": recent_streak,
        "best_streak": best_streak,
        "daily_performance": _daily_performance(ordered, days=daily_days, today=today or date.today()),
    }


def _deduplicate_daily(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Match the old recap rule: retain the best logged price per exact leg."""
    selected: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in _normalise_rows(rows):
        key = (
            str(row.get("home_team") or ""),
            str(row.get("away_team") or ""),
            str(row.get("market") or ""),
            str(row.get("pick") or ""),
        )
        current = selected.get(key)
        current_odds = _number(current.get("odds")) if current else None
        candidate_odds = _number(row.get("odds"))
        if current is None or (candidate_odds or 0.0) > (current_odds or 0.0):
            selected[key] = row
    return list(selected.values())


def build_daily_breakdown(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_date: Any,
) -> Dict[str, Any]:
    """Return the recap contract used by scheduled Discord result posts."""
    prediction_date = _date_text(target_date)
    deduped = _deduplicate_daily(rows)
    if not deduped:
        return {
            "date": prediction_date,
            "total": 0,
            "hits": 0,
            "half_hits": 0,
            "misses": 0,
            "half_misses": 0,
            "pushes": 0,
            "hit_rate": 0.0,
            "avg_clv": None,
            "clv_positive_rate": None,
            "clv_sample_size": 0,
            "by_market": {},
            "by_league": {},
            "by_confidence": {},
            "notable_hits": [],
            "notable_misses": [],
        }

    overall = _summary(deduped)
    avg_clv, clv_positive_rate, clv_sample_size = _clv_summary(deduped)

    def grouped(field: str, *, confidence: bool = False) -> Dict[str, Any]:
        groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in deduped:
            key = str(row.get(field) or ("unknown" if confidence else "other")).lower() if confidence else str(row.get(field) or "other")
            groups[key].append(row)
        result: Dict[str, Any] = {}
        for key, subset in groups.items():
            summary = _summary(subset, include_roi=confidence)
            if field == "market":
                summary["avg_clv"] = _clv_summary(subset)[0]
            result[key] = summary
        return result

    hit_rows = sorted(
        (row for row in deduped if normalize_outcome(row.get("outcome")) == HIT and is_valid_decimal_odds(row.get("odds"))),
        key=lambda row: _number(row.get("odds")) or 0.0,
        reverse=True,
    )[:5]
    miss_rows = sorted(
        (row for row in deduped if normalize_outcome(row.get("outcome")) == MISS and _number(row.get("model_prob")) is not None),
        key=lambda row: _number(row.get("model_prob")) or 0.0,
        reverse=True,
    )[:5]
    return {
        "date": prediction_date,
        "total": overall["total"],
        "hits": overall["hits"],
        "half_hits": overall["half_hits"],
        "misses": overall["misses"],
        "half_misses": overall["half_misses"],
        "pushes": overall["pushes"],
        "hit_rate": overall["hit_rate"],
        "avg_clv": avg_clv,
        "clv_positive_rate": clv_positive_rate,
        "clv_sample_size": clv_sample_size,
        "by_market": grouped("market"),
        "by_league": grouped("league"),
        "by_confidence": grouped("confidence", confidence=True),
        "notable_hits": [
            {
                "fixture": f"{row.get('home_team', '')} vs {row.get('away_team', '')}",
                "market": row.get("market"),
                "pick": row.get("pick"),
                "odds": _number(row.get("odds")),
                "league": row.get("league"),
            }
            for row in hit_rows
        ],
        "notable_misses": [
            {
                "fixture": f"{row.get('home_team', '')} vs {row.get('away_team', '')}",
                "market": row.get("market"),
                "pick": row.get("pick"),
                "model_prob": _number(row.get("model_prob")),
                "league": row.get("league"),
            }
            for row in miss_rows
        ],
    }
