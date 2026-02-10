from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import log
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class OddsLeg:
    event_id: str
    fixture: str
    market_key: str
    outcome: str
    odds: float
    point: Optional[float] = None
    bookmaker: Optional[str] = None


def extract_candidates(
    events: List[Dict],
    allowed_market_keys: Optional[Set[str]] = None,
) -> List[OddsLeg]:
    legs: List[OddsLeg] = []
    for ev in events:
        event_id = str(ev.get("id") or "")
        home = ev.get("home_team") or "Home"
        away = ev.get("away_team") or "Away"
        fixture = f"{home} vs {away}"

        market_prices: Iterable[Dict] = ev.get("market_prices") or []
        if market_prices:
            for row in market_prices:
                market_key = str(row.get("market_key") or "").strip()
                if not market_key:
                    continue
                if allowed_market_keys and market_key not in allowed_market_keys:
                    continue
                try:
                    odds = float(row.get("price"))
                except Exception:
                    continue
                if odds <= 1.0:
                    continue
                point = row.get("point")
                try:
                    point = float(point) if point is not None else None
                except Exception:
                    point = None
                legs.append(
                    OddsLeg(
                        event_id=event_id,
                        fixture=fixture,
                        market_key=market_key,
                        outcome=str(row.get("name")),
                        odds=odds,
                        point=point,
                        bookmaker=row.get("bookmaker"),
                    )
                )
            continue

        # Backward-compatible fallback if only h2h map exists.
        best = ev.get("best_prices") or {}
        if allowed_market_keys and "h2h" not in allowed_market_keys:
            continue
        for outcome, payload in best.items():
            try:
                odds = float(payload.get("price"))
            except Exception:
                continue
            if odds <= 1.0:
                continue
            legs.append(
                OddsLeg(
                    event_id=event_id,
                    fixture=fixture,
                    market_key="h2h",
                    outcome=str(outcome),
                    odds=odds,
                    bookmaker=payload.get("bookmaker"),
                )
            )
    return legs


def extract_h2h_candidates(events: List[Dict]) -> List[OddsLeg]:
    return extract_candidates(events, allowed_market_keys={"h2h"})


def combined_odds(legs: List[OddsLeg]) -> float:
    v = 1.0
    for leg in legs:
        v *= leg.odds
    return v


def _unique_events(legs: List[OddsLeg]) -> bool:
    ids = [x.event_id for x in legs]
    return len(ids) == len(set(ids))


def _has_internal_conflicts(legs: List[OddsLeg]) -> bool:
    # Prevent contradictory legs from the same event.
    for i in range(len(legs)):
        a = legs[i]
        for j in range(i + 1, len(legs)):
            b = legs[j]
            if a.event_id != b.event_id:
                continue

            # Two H2H outcomes in same fixture are mutually exclusive.
            if a.market_key == "h2h" and b.market_key == "h2h":
                return True

            # Opposite sides of same totals/spreads line conflict.
            if (
                a.market_key == b.market_key
                and a.market_key in {"totals", "spreads"}
                and (a.point == b.point)
                and a.outcome != b.outcome
            ):
                return True
    return False


def select_best_parlay(
    candidates: List[OddsLeg],
    leg_count: int,
    target_multiplier: Optional[float] = None,
    target_mode: str = "none",  # none | around | max | min
    require_unique_events: bool = False,
) -> Tuple[List[OddsLeg], Optional[str]]:
    if leg_count <= 0:
        return [], "Leg count must be > 0."
    if not candidates:
        return [], "No odds candidates available."

    if require_unique_events:
        unique_events = {c.event_id for c in candidates}
        if len(unique_events) < leg_count:
            return [], (
                f"Only {len(unique_events)} fixtures are available. "
                f"Cannot build a valid {leg_count}-leg parlay with one leg per match."
            )

    combos: List[Tuple[List[OddsLeg], float]] = []
    for combo in combinations(candidates, leg_count):
        combo_list = list(combo)
        if require_unique_events and not _unique_events(combo_list):
            continue
        if _has_internal_conflicts(combo_list):
            continue
        combo_odds = combined_odds(combo_list)
        if combo_odds <= 0:
            continue
        combos.append((combo_list, combo_odds))

    if not combos:
        return [], "No valid parlay combination found under current constraints."

    if target_multiplier is None or target_mode == "none":
        # No user target provided: choose the safer combo (highest implied probability).
        best_combo, _ = min(combos, key=lambda x: x[1])
        return best_combo, None

    if target_multiplier <= 0:
        best_combo, _ = min(combos, key=lambda x: x[1])
        return best_combo, None

    if target_mode == "max":
        feasible = [x for x in combos if x[1] <= target_multiplier]
        if feasible:
            best_combo, _ = max(feasible, key=lambda x: x[1])  # closest below upper bound
            return best_combo, None
        # fallback if nothing is under max: pick closest overall.
        best_combo, _ = min(combos, key=lambda x: abs(x[1] - target_multiplier))
        return best_combo, None

    if target_mode == "min":
        feasible = [x for x in combos if x[1] >= target_multiplier]
        if feasible:
            best_combo, _ = min(feasible, key=lambda x: x[1])  # closest above lower bound
            return best_combo, None
        best_combo, _ = min(combos, key=lambda x: abs(x[1] - target_multiplier))
        return best_combo, None

    # around target
    best_combo, _ = min(combos, key=lambda x: abs(log(x[1]) - log(target_multiplier)))
    return best_combo, None
