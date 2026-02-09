from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import log
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class OddsLeg:
    event_id: str
    fixture: str
    outcome: str
    odds: float
    bookmaker: Optional[str] = None


def extract_h2h_candidates(events: List[Dict]) -> List[OddsLeg]:
    legs: List[OddsLeg] = []
    for ev in events:
        event_id = str(ev.get("id") or "")
        home = ev.get("home_team") or "Home"
        away = ev.get("away_team") or "Away"
        fixture = f"{home} vs {away}"
        best = ev.get("best_prices") or {}
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
                    outcome=str(outcome),
                    odds=odds,
                    bookmaker=payload.get("bookmaker"),
                )
            )
    return legs


def combined_odds(legs: List[OddsLeg]) -> float:
    v = 1.0
    for leg in legs:
        v *= leg.odds
    return v


def _unique_events(legs: List[OddsLeg]) -> bool:
    ids = [x.event_id for x in legs]
    return len(ids) == len(set(ids))


def select_best_parlay(
    candidates: List[OddsLeg],
    leg_count: int,
    target_multiplier: float,
    require_unique_events: bool = True,
) -> Tuple[List[OddsLeg], Optional[str]]:
    if leg_count <= 0:
        return [], "Leg count must be > 0."
    if not candidates:
        return [], "No odds candidates available."

    if require_unique_events:
        unique_events = {c.event_id for c in candidates}
        if len(unique_events) < leg_count:
            return [], (
                f"Only {len(unique_events)} fixtures are available for H2H markets. "
                f"Cannot build a valid {leg_count}-leg parlay with one leg per match."
            )

    # Optimize closeness to target in log-space to balance over/under deviations.
    best_combo: Optional[List[OddsLeg]] = None
    best_score = float("inf")

    for combo in combinations(candidates, leg_count):
        combo_list = list(combo)
        if require_unique_events and not _unique_events(combo_list):
            continue
        combo_odds = combined_odds(combo_list)
        if combo_odds <= 0:
            continue
        score = abs(log(combo_odds) - log(target_multiplier))
        # Tie-breaker: prefer slightly above target vs below if same score.
        if best_combo is None:
            best_combo = combo_list
            best_score = score
            continue
        if score < best_score:
            best_combo = combo_list
            best_score = score
        elif abs(score - best_score) < 1e-9:
            if combined_odds(combo_list) >= target_multiplier and combined_odds(best_combo) < target_multiplier:
                best_combo = combo_list
                best_score = score

    if not best_combo:
        return [], "No valid parlay combination found under current constraints."

    return best_combo, None
