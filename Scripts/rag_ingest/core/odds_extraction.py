"""
core/odds_extraction.py — Market classification and odds extraction from
bookmaker event payloads.

Extracted verbatim from rag_cli_v2.py.  Do NOT modify logic here without
updating the original file (until the migration is complete).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

# --- team resolution helpers ---
try:
    from core.team_resolution import (
        safe_float,
        canonical_team_name,
        team_name_aliases,
    )
except ImportError:
    try:
        from team_resolution import (
            safe_float,
            canonical_team_name,
            team_name_aliases,
        )
    except ImportError:
        # Minimal stubs so file is independently parseable
        def safe_float(x) -> Optional[float]:
            try:
                return float(x)
            except Exception:
                return None

        def canonical_team_name(name: str) -> str:
            return name

        def team_name_aliases(name: str) -> Set[str]:
            return {name.lower().strip()}


# ---------------------------------------------------------------------------
# Half-time / player-prop filter
# ---------------------------------------------------------------------------

HALF_MARKET_PATTERN = re.compile(
    r"(?:^|_)(h1|h2|1st_half|2nd_half|first_half|second_half)(?:_|$)"
)


# ---------------------------------------------------------------------------
# Market classification
# ---------------------------------------------------------------------------

def market_group_from_key(market_key: str) -> str:
    k = (market_key or "").lower()
    if "corner" in k:
        return "corners"
    if "card" in k or "booking" in k:
        return "cards"
    # SoT must be checked BEFORE generic "total" catch-all
    if ("shot" in k and "target" in k) or k == "sot":
        return "sot"
    if "btts" in k or "both_teams" in k:
        return "btts"
    if "h2h" in k or "moneyline" in k or "1x2" in k:
        return "moneyline"
    if "spread" in k or "handicap" in k:
        return "spreads"
    if "total" in k:
        return "totals"
    return k


def is_full_game_market(market_key: str) -> bool:
    k = (market_key or "").lower().strip()
    if not k:
        return False
    # Exchange lay markets are semantically inverted and produce duplicate/confusing exposure.
    if "lay" in k:
        return False
    if "player" in k:
        return False
    if HALF_MARKET_PATTERN.search(k):
        return False
    return True


# ---------------------------------------------------------------------------
# Totals extraction (goals / corners / cards / SoT)
# ---------------------------------------------------------------------------

def extract_total_line_options(event: Dict, stat_group: str) -> List[Dict]:
    out: List[Dict] = []
    for bm in event.get("bookmakers", []) or []:
        bm_name = str(bm.get("title") or "")
        for mk in bm.get("markets", []) or []:
            key = str(mk.get("key") or "")
            k = key.lower()
            if not is_full_game_market(key):
                continue
            if stat_group == "corners":
                if ("corner" not in k) or ("total" not in k):
                    continue
                # Exclude per-team corner lines -- we want match totals only
                if "team_total" in k or "home_corner" in k or "away_corner" in k:
                    continue
            elif stat_group == "cards":
                if (("card" not in k) and ("booking" not in k) and ("yellow" not in k)) or ("total" not in k):
                    continue
                # Exclude per-team card lines -- we want match totals only
                if "team_total" in k or "home_card" in k or "away_card" in k or "home_team_total" in k or "away_team_total" in k:
                    continue
            elif stat_group == "goals":
                # Match totals goals markets. Avoid corners/cards/teams/goalscorer props.
                if "total" not in k:
                    continue
                if "corner" in k or "card" in k or "booking" in k:
                    continue
                if "team_total" in k or "player" in k or "goalscorer" in k:
                    continue
                if "shot" in k or "yellow" in k:
                    continue
            elif stat_group == "sot":
                if "shot" not in k:
                    continue
                # Must be a totals-style market (over/under), not 1x2 or handicap
                if "total" not in k and "over" not in k and "under" not in k:
                    continue
                # Exclude per-team SoT and 1x2/handicap shot markets
                if "team_total" in k or "home" in k or "away" in k:
                    continue
                if "1x2" in k or "handicap" in k or "double" in k:
                    continue
            else:
                continue
            for outcome in mk.get("outcomes", []) or []:
                name = str(outcome.get("name") or "").lower().strip()
                if name not in {"over", "under"}:
                    continue
                point = safe_float(outcome.get("point"))
                price = safe_float(outcome.get("price"))
                if point is None or price is None or price <= 1.0:
                    continue
                out.append(
                    {
                        "market_key": key,
                        "bookmaker": bm_name,
                        "side": name,
                        "point": point,
                        "odds": price,
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Per-team totals extraction (corners / cards / goals)
# ---------------------------------------------------------------------------

def extract_team_total_line_options(
    event: Dict, team_name: str, stat_group: str,
) -> List[Dict]:
    """Extract over/under lines for a specific team from team_totals markets.

    The Odds API only offers team_totals for goals -- corners/cards team totals
    don't exist, so this returns empty for those stat groups (triggering model-only
    fallback in the renderer).
    """
    out: List[Dict] = []
    aliases = team_name_aliases(team_name)
    for bm in event.get("bookmakers", []) or []:
        bm_name = str(bm.get("title") or "")
        for mk in bm.get("markets", []) or []:
            key = str(mk.get("key") or "")
            k = key.lower()
            if not is_full_game_market(key):
                continue
            if "team_total" not in k:
                continue
            if stat_group == "corners" and "corner" not in k:
                continue
            if stat_group == "cards" and "card" not in k and "booking" not in k:
                continue
            # Determine if this market is for the requested team.
            # API-Football keys: team_totals_home_corners, team_totals_away_cards, etc.
            # Match "home" in key to the event's home team, "away" to away team.
            home_in_event = str(event.get("home_team") or "").lower().strip()
            away_in_event = str(event.get("away_team") or "").lower().strip()
            team_lower = team_name.lower().strip()

            key_is_home = "home" in k
            key_is_away = "away" in k
            team_is_home = (team_lower == home_in_event
                            or any(a in home_in_event for a in aliases))
            team_is_away = (team_lower == away_in_event
                            or any(a in away_in_event for a in aliases))

            # If key says "home" but team is away (or vice versa), skip
            if key_is_home and not team_is_home:
                continue
            if key_is_away and not team_is_away:
                continue
            # If key doesn't specify home/away, fall back to old text matching
            if not key_is_home and not key_is_away:
                # Legacy path for Odds API format
                pass

            for outcome in mk.get("outcomes", []) or []:
                name = str(outcome.get("name") or "").strip()
                side = None
                if name.lower() in ("over", "under"):
                    side = name.lower()
                if side is None:
                    desc = str(outcome.get("description") or "").strip()
                    combined_text = f"{name} {desc}".lower()
                    if "over" in combined_text:
                        side = "over"
                    elif "under" in combined_text:
                        side = "under"
                if side is None:
                    continue
                point = safe_float(outcome.get("point"))
                price = safe_float(outcome.get("price"))
                if point is None or price is None or price <= 1.0:
                    continue
                out.append({
                    "market_key": key, "bookmaker": bm_name,
                    "side": side, "point": point, "odds": price,
                    "team": team_name,
                })
    return out


# ---------------------------------------------------------------------------
# Spread / Handicap extraction
# ---------------------------------------------------------------------------

def extract_spread_line_options(event: Dict) -> List[Dict]:
    """Extract available handicap/spread lines from bookmaker data for one event."""
    out: List[Dict] = []
    for bm in event.get("bookmakers", []) or []:
        bm_name = str(bm.get("title") or "")
        for mk in bm.get("markets", []) or []:
            key = str(mk.get("key") or "")
            k = key.lower()
            if not is_full_game_market(key):
                continue
            if "spread" not in k and "handicap" not in k:
                continue
            for oc in mk.get("outcomes", []) or []:
                name = str(oc.get("name") or "")
                point = oc.get("point")
                price = oc.get("price")
                try:
                    point = float(point) if point is not None else None
                    price = float(price) if price is not None else None
                except (TypeError, ValueError):
                    continue
                if point is None or price is None or price < 1.10:
                    continue
                out.append({
                    "market_key": key,
                    "bookmaker": bm_name,
                    "team": name,
                    "point": point,
                    "odds": price,
                })
    return out


# ---------------------------------------------------------------------------
# Moneyline (1X2) extraction
# ---------------------------------------------------------------------------

def extract_moneyline_odds(event: Dict) -> List[Dict]:
    """Extract home/draw/away moneyline odds from event bookmaker data."""
    home = str(event.get("home_team") or "")
    away = str(event.get("away_team") or "")
    results: List[Dict] = []
    for bm in event.get("bookmakers", []) or []:
        bm_name = str(bm.get("title") or "")
        for mk in bm.get("markets", []) or []:
            key = str(mk.get("key") or "").lower()
            if market_group_from_key(key) != "moneyline":
                continue
            if not is_full_game_market(key):
                continue
            for oc in mk.get("outcomes", []) or []:
                name = str(oc.get("name") or "")
                odds = safe_float(oc.get("price"))
                if odds is None or odds <= 1.0:
                    continue
                name_l = name.lower().strip()
                if name_l == "draw" or name_l == "x":
                    side, team = "draw", "Draw"
                elif name_l == home.lower().strip() or name_l.startswith("home") or name_l == "1":
                    side, team = "home", home
                elif name_l == away.lower().strip() or name_l.startswith("away") or name_l == "2":
                    side, team = "away", away
                else:
                    continue
                results.append({"side": side, "odds": odds, "bookmaker": bm_name, "team": team})
    return results


# ---------------------------------------------------------------------------
# BTTS extraction
# ---------------------------------------------------------------------------

def extract_btts_odds(event: Dict) -> List[Dict]:
    """Extract BTTS Yes/No odds from bookmaker data for one event."""
    out: List[Dict] = []
    for bm in event.get("bookmakers", []) or []:
        bm_name = str(bm.get("title") or "")
        for mk in bm.get("markets", []) or []:
            key = str(mk.get("key") or "")
            if market_group_from_key(key) != "btts":
                continue
            if not is_full_game_market(key):
                continue
            for outcome in mk.get("outcomes", []) or []:
                name = str(outcome.get("name") or "").strip()
                price = safe_float(outcome.get("price"))
                if not name or price is None or price <= 1.0:
                    continue
                out.append({
                    "market_key": key,
                    "bookmaker": bm_name,
                    "side": name,
                    "odds": price,
                })
    return out


# ---------------------------------------------------------------------------
# Correct Score extraction
# ---------------------------------------------------------------------------

def extract_correct_score_odds(event: Dict) -> Dict[Tuple[int, int], List[Dict]]:
    """Extract correct score odds from event bookmaker data.

    Returns {(home_goals, away_goals): [{"odds": float, "bookmaker": str, "key": str}, ...]}.
    """
    result: Dict[Tuple[int, int], List[Dict]] = {}
    for bm in event.get("bookmakers", []) or []:
        bm_name = str(bm.get("title") or "")
        for mk in bm.get("markets", []) or []:
            key = str(mk.get("key") or "").lower()
            if "correctscore" not in key and "correct_score" not in key and "exactscore" not in key:
                continue
            for oc in mk.get("outcomes", []) or []:
                name = str(oc.get("name") or "")
                odds = safe_float(oc.get("price"))
                if odds is None or odds <= 1.0:
                    continue
                m = re.match(r"(\d+)\s*[-:]\s*(\d+)", name.strip())
                if not m:
                    continue
                score = (int(m.group(1)), int(m.group(2)))
                result.setdefault(score, []).append({
                    "odds": odds,
                    "bookmaker": bm_name,
                    "key": key,
                })
    return result


# ---------------------------------------------------------------------------
# Goal Interval extraction
# ---------------------------------------------------------------------------

GOAL_INTERVAL_BANDS = [
    (0, 1, "0-1 goals"),
    (0, 2, "0-2 goals"),
    (1, 2, "1-2 goals"),
    (1, 3, "1-3 goals"),
    (2, 3, "2-3 goals"),
    (2, 4, "2-4 goals"),
    (2, 5, "2-5 goals"),
    (3, 4, "3-4 goals"),
    (3, 5, "3-5 goals"),
    (4, 5, "4-5 goals"),
    (6, 30, "6+ goals"),
]


def _match_interval_label(outcome_name: str) -> Optional[str]:
    """Map bookmaker outcome name like '0-1', '2-3', '4-5', '6+' to our band labels."""
    n = outcome_name.lower().strip()
    for low, high, label in GOAL_INTERVAL_BANDS:
        if high >= 30:
            # Open-ended: match "6+", "6 or more", "6-7", etc.
            if f"{low}+" in n or f"{low} or more" in n or f"{low}-" in n:
                return label
        else:
            if f"{low}-{high}" in n or f"{low} - {high}" in n:
                return label
    return None


def extract_goal_interval_odds(event: Dict) -> Dict[str, Dict]:
    """Extract goal range / goal band odds from bookmaker data.

    Returns {label: {"odds": float, "bookmaker": str, "market_key": str}} for
    each interval band found.  Empty dict when no goal-range market exists
    (graceful model-only fallback).
    """
    out: Dict[str, Dict] = {}
    for bm in event.get("bookmakers", []) or []:
        bm_name = str(bm.get("title") or "")
        for mk in bm.get("markets", []) or []:
            key = str(mk.get("key") or "")
            k = key.lower()
            if not is_full_game_market(key):
                continue
            # Match goal range / goal band market keys
            if not (("goal" in k and ("range" in k or "band" in k or "interval" in k))
                    or k in {"goals_range", "goal_range", "goal_band", "goal_interval"}):
                continue
            for outcome in mk.get("outcomes", []) or []:
                name = str(outcome.get("name") or "").strip()
                price = safe_float(outcome.get("price"))
                if price is None or price <= 1.0:
                    continue
                # Map outcome name to our standard bands
                label = _match_interval_label(name)
                if label and (label not in out or price > out[label]["odds"]):
                    out[label] = {"odds": price, "bookmaker": bm_name, "market_key": key}
    return out
