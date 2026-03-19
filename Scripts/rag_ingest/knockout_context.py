"""Knockout round context for European competitions.

Detects two-legged knockout ties, fetches first-leg results from API-Football,
and computes projection modifiers based on aggregate context.

Key insight: If team A won the first leg 6-1, the second leg will likely have
fewer goals, corners, and cards because team A has no urgency and team B may
be demoralized or tactically conservative.

Graceful degradation: every failure returns neutral modifiers (1.0).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None  # type: ignore

log = logging.getLogger("knockout_context")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEAGUE_TO_API_ID: Dict[str, int] = {
    "UCL": 2, "UEL": 3, "UECL": 848,
}

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
SEASON = 2025

# Knockout round keywords — if the round string contains any of these,
# it's a knockout tie (two-legged or single-leg final).
KNOCKOUT_ROUND_KEYWORDS = [
    "round of 16", "round of 32",
    "quarter-final", "quarterfinal", "quarter final",
    "semi-final", "semifinal", "semi final",
    "final",  # includes "final" but NOT "group"
    "playoff", "play-off",
    "qualifying",  # some qualifying rounds are two-legged
]

GROUP_KEYWORDS = ["group", "matchday", "league phase"]

# Cache: {(league, season): [(fixture_dict, ...)]}
_fixtures_cache: Dict[Tuple[str, int], Tuple[float, List[Dict]]] = {}
_CACHE_TTL = 900  # 15 minutes


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KnockoutContext:
    """Aggregate context for a knockout-round fixture."""
    is_knockout: bool = False
    round_name: str = ""
    is_second_leg: bool = False
    first_leg_home: str = ""       # who was home in leg 1
    first_leg_away: str = ""
    first_leg_home_goals: int = 0
    first_leg_away_goals: int = 0
    aggregate_lead: int = 0        # positive = home team in THIS fixture leads on agg
    aggregate_total_goals: int = 0
    is_blowout: bool = False       # aggregate margin >= 3
    is_tight: bool = False         # aggregate is tied or 1-goal difference
    # Modifiers (1.0 = neutral)
    goals_modifier: float = 1.0
    corners_modifier: float = 1.0
    cards_modifier: float = 1.0
    sot_modifier: float = 1.0
    source: str = "none"           # "api" or "none"


# ---------------------------------------------------------------------------
# API-Football fetching
# ---------------------------------------------------------------------------

def _get_api_key() -> Optional[str]:
    # Ensure .env is loaded (same approach as referee_data.py)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    for k in ("API-FOOTBALL-KEY", "API_FOOTBALL_KEY"):
        val = os.environ.get(k)
        if val:
            return val
    return None


def _fetch_season_fixtures(league: str) -> List[Dict]:
    """Fetch all fixtures for a European competition season. Cached 15 min."""
    api_id = LEAGUE_TO_API_ID.get(league)
    if not api_id:
        return []

    cache_key = (league, SEASON)
    cached = _fixtures_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL:
        return cached[1]

    api_key = _get_api_key()
    if not api_key or requests is None:
        return []

    try:
        resp = requests.get(
            f"{API_FOOTBALL_BASE}/fixtures",
            headers={"x-apisports-key": api_key},
            params={"league": api_id, "season": SEASON},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        fixtures = data.get("response", [])
        _fixtures_cache[cache_key] = (time.time(), fixtures)
        log.info("Fetched %d %s fixtures from API-Football", len(fixtures), league)
        return fixtures
    except Exception as exc:
        log.warning("Failed to fetch %s fixtures: %s", league, exc)
        return []


# ---------------------------------------------------------------------------
# Knockout detection and first-leg lookup
# ---------------------------------------------------------------------------

def _normalize_team(name: str) -> str:
    """Normalize team name for matching across API sources."""
    n = name.strip().lower()
    for strip in (".", "'", "-", " fc", "fc "):
        n = n.replace(strip, "")
    return n.strip()


def _teams_match_fuzzy(name1: str, name2: str) -> bool:
    """Check if two team names refer to the same team.

    Handles mismatches like 'Newcastle United' (Odds API) vs 'Newcastle' (API-Football).
    """
    n1 = _normalize_team(name1)
    n2 = _normalize_team(name2)
    if n1 == n2:
        return True
    # Substring containment (longer contains shorter)
    if len(n1) >= 4 and len(n2) >= 4:
        if n1 in n2 or n2 in n1:
            return True
    return False


def _is_knockout_round(round_str: str) -> bool:
    """Check if a round string indicates a knockout stage."""
    r = round_str.lower()
    if any(kw in r for kw in GROUP_KEYWORDS):
        return False
    return any(kw in r for kw in KNOCKOUT_ROUND_KEYWORDS)


def _teams_match(t1a: str, t1b: str, t2a: str, t2b: str) -> bool:
    """Check if two fixtures involve the same two teams (in any order)."""
    # Try both orderings
    return ((_teams_match_fuzzy(t1a, t2a) and _teams_match_fuzzy(t1b, t2b)) or
            (_teams_match_fuzzy(t1a, t2b) and _teams_match_fuzzy(t1b, t2a)))


def find_first_leg(
    home_team: str,
    away_team: str,
    fixture_date: str,
    league: str,
) -> Optional[Dict]:
    """Find the first leg of a two-legged knockout tie.

    Logic: same round + same two teams + earlier date + status FT = first leg.
    """
    fixtures = _fetch_season_fixtures(league)
    if not fixtures:
        return None

    # Find the upcoming fixture's round
    target_round = None
    target_dt = None

    # Parse target date
    try:
        if "T" in fixture_date:
            target_dt = datetime.fromisoformat(fixture_date.replace("Z", "+00:00"))
        else:
            target_dt = datetime.fromisoformat(fixture_date + "T23:59:59+00:00")
    except Exception:
        return None

    # Find the round for our target fixture
    for f in fixtures:
        teams = f.get("teams", {})
        h = teams.get("home", {}).get("name", "")
        a = teams.get("away", {}).get("name", "")
        fx = f.get("fixture", {})
        fx_date = fx.get("date", "")

        if _teams_match(home_team, away_team, h, a):
            try:
                fx_dt = datetime.fromisoformat(fx_date.replace("Z", "+00:00"))
                # This fixture is close to our target date (within 2 days)
                diff = abs((fx_dt - target_dt).total_seconds())
                if diff < 172800:  # 48 hours
                    target_round = f.get("league", {}).get("round", "")
                    break
            except Exception:
                continue

    if not target_round or not _is_knockout_round(target_round):
        return None

    # Now find the OTHER fixture in the same round with the same teams
    # that was played BEFORE our target date and is finished
    candidates = []
    for f in fixtures:
        teams = f.get("teams", {})
        h = teams.get("home", {}).get("name", "")
        a = teams.get("away", {}).get("name", "")
        fx = f.get("fixture", {})
        round_str = f.get("league", {}).get("round", "")
        status = fx.get("status", {}).get("short", "")

        if round_str != target_round:
            continue
        if not _teams_match(home_team, away_team, h, a):
            continue
        if status not in ("FT", "AET", "PEN"):
            continue

        try:
            fx_dt = datetime.fromisoformat(fx.get("date", "").replace("Z", "+00:00"))
            # Must be at least 3 days before target (not the same fixture)
            gap_days = (target_dt - fx_dt).total_seconds() / 86400
            if gap_days >= 3:
                candidates.append((fx_dt, f))
        except Exception:
            continue

    if not candidates:
        return None  # this is leg 1 (no earlier completed fixture)

    # Return the most recent completed fixture before target = first leg
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Modifier computation
# ---------------------------------------------------------------------------

def _compute_modifiers(
    aggregate_lead: int,
    aggregate_total: int,
    is_second_leg: bool,
) -> Tuple[float, float, float, float]:
    """Compute projection modifiers based on aggregate context.

    Returns (goals_mod, corners_mod, cards_mod, sot_mod).

    Key principles:
    - Large aggregate lead (3+) → the tie is dead → less intensity →
      fewer goals, corners, cards, shots
    - Tight aggregate (0-1 goal difference) → high intensity →
      more cards (fouls), similar or higher corners/goals
    - Comfortable lead (2 goals) → moderate reduction
    - Leading team plays conservatively; trailing team may go all-out
      (more corners, more shots) but also more desperate fouls (more cards)
    """
    if not is_second_leg:
        return 1.0, 1.0, 1.0, 1.0

    abs_lead = abs(aggregate_lead)

    if abs_lead >= 4:
        # Blowout — tie is essentially over
        return 0.80, 0.82, 0.75, 0.82
    elif abs_lead == 3:
        # Comfortable — very likely going through, reduced intensity
        return 0.85, 0.87, 0.80, 0.85
    elif abs_lead == 2:
        # Solid lead — still some tension but leading team relaxes
        return 0.92, 0.93, 0.90, 0.92
    elif abs_lead == 1:
        # Tight — trailing team pushes hard, leading team defends deep
        # More fouls/cards, slightly more corners (crosses), similar goals
        return 1.0, 1.03, 1.08, 1.0
    else:
        # Level on aggregate — maximum intensity
        # Both teams go for it → more corners from attacks, more cards
        return 1.03, 1.05, 1.10, 1.02

    return 1.0, 1.0, 1.0, 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_knockout_context(
    home_team: str,
    away_team: str,
    league: str,
    fixture_date: Optional[str] = None,
) -> KnockoutContext:
    """Get knockout round context for a European competition fixture.

    Returns KnockoutContext with aggregate info and projection modifiers.
    Falls back to neutral context (all modifiers = 1.0) on any failure.
    """
    if league not in LEAGUE_TO_API_ID:
        return KnockoutContext()

    if fixture_date is None:
        fixture_date = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        first_leg = find_first_leg(home_team, away_team, fixture_date, league)
    except Exception as exc:
        log.warning("Knockout context lookup failed: %s", exc)
        return KnockoutContext()

    if first_leg is None:
        # Either this is leg 1, or it's a group stage match, or lookup failed
        # Check if the round is at least a knockout round
        fixtures = _fetch_season_fixtures(league)
        for f in fixtures:
            teams = f.get("teams", {})
            h = teams.get("home", {}).get("name", "")
            a = teams.get("away", {}).get("name", "")
            if _teams_match(home_team, away_team, h, a):
                round_str = f.get("league", {}).get("round", "")
                if _is_knockout_round(round_str):
                    return KnockoutContext(
                        is_knockout=True,
                        round_name=round_str,
                        is_second_leg=False,
                        source="api",
                    )
                break
        return KnockoutContext()

    # We have a first leg result — this means current fixture IS leg 2
    teams = first_leg.get("teams", {})
    goals = first_leg.get("goals", {})
    round_str = first_leg.get("league", {}).get("round", "")

    leg1_home = teams.get("home", {}).get("name", "")
    leg1_away = teams.get("away", {}).get("name", "")
    leg1_home_goals = goals.get("home") or 0
    leg1_away_goals = goals.get("away") or 0
    agg_total = leg1_home_goals + leg1_away_goals

    # Determine aggregate lead from perspective of the HOME team in THIS fixture
    # In leg 2, home/away are swapped from leg 1
    # If leg1 was A(home) 3-1 B(away), and leg2 is B(home) vs A(away):
    #   A scored 3 away goals in leg1, B scored 1 home goal
    #   From leg2 home (B) perspective: B has 1, A has 3 → B trails by 2
    if _teams_match_fuzzy(home_team, leg1_home):
        # Same home team both legs (unusual but possible in some formats)
        home_agg = leg1_home_goals
        away_agg = leg1_away_goals
    else:
        # Normal: home/away swapped between legs
        # home_team in leg2 was away_team in leg1
        home_agg = leg1_away_goals  # goals they scored as away in leg1
        away_agg = leg1_home_goals  # goals opponent scored as home in leg1

    aggregate_lead = home_agg - away_agg  # positive = home leads on aggregate

    abs_lead = abs(aggregate_lead)
    is_blowout = abs_lead >= 3
    is_tight = abs_lead <= 1

    goals_mod, corners_mod, cards_mod, sot_mod = _compute_modifiers(
        aggregate_lead, agg_total, is_second_leg=True,
    )

    ctx = KnockoutContext(
        is_knockout=True,
        round_name=round_str,
        is_second_leg=True,
        first_leg_home=leg1_home,
        first_leg_away=leg1_away,
        first_leg_home_goals=leg1_home_goals,
        first_leg_away_goals=leg1_away_goals,
        aggregate_lead=aggregate_lead,
        aggregate_total_goals=agg_total,
        is_blowout=is_blowout,
        is_tight=is_tight,
        goals_modifier=goals_mod,
        corners_modifier=corners_mod,
        cards_modifier=cards_mod,
        sot_modifier=sot_mod,
        source="api",
    )

    log.info(
        "Knockout context: %s vs %s (%s) — Leg 2, first leg %s %d-%d %s, "
        "agg lead %+d, modifiers: G=%.2f C=%.2f K=%.2f S=%.2f",
        home_team, away_team, round_str,
        leg1_home, leg1_home_goals, leg1_away_goals, leg1_away,
        aggregate_lead, goals_mod, corners_mod, cards_mod, sot_mod,
    )

    return ctx


def knockout_evidence_line(ctx: KnockoutContext) -> Optional[str]:
    """Generate a human-readable evidence line for knockout context."""
    if not ctx.is_knockout or ctx.source == "none":
        return None

    if not ctx.is_second_leg:
        return f"Knockout {ctx.round_name} (Leg 1 — no aggregate context yet)."

    score = f"{ctx.first_leg_home} {ctx.first_leg_home_goals}-{ctx.first_leg_away_goals} {ctx.first_leg_away}"

    if ctx.is_blowout:
        intensity = "low intensity expected (blowout aggregate)"
    elif ctx.is_tight:
        intensity = "high intensity expected (tight aggregate)"
    else:
        intensity = "moderate adjustment (comfortable lead)"

    mods = []
    for label, val in [("goals", ctx.goals_modifier), ("corners", ctx.corners_modifier),
                       ("cards", ctx.cards_modifier), ("SoT", ctx.sot_modifier)]:
        if val != 1.0:
            pct = (val - 1.0) * 100
            mods.append(f"{label} {pct:+.0f}%")

    mod_str = ", ".join(mods) if mods else "no adjustment"

    return (
        f"Knockout {ctx.round_name} Leg 2 — First leg: {score} — "
        f"{intensity} — Modifiers: {mod_str}."
    )
