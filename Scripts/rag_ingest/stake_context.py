"""Stake intensity model — adjusts projections based on what teams are playing for.

Teams fighting for the title, European spots, or survival play with more intensity
(more fouls, more corners from desperate attacks, more goals from open play).
Dead rubbers between mid-table teams with nothing to play for are duller.

Activates after gameweek 25 — early season stakes are too uncertain.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None  # type: ignore

log = logging.getLogger("stake_context")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEAGUE_TO_API_ID: Dict[str, int] = {
    "EPL": 39, "LaLiga": 140, "SerieA": 135,
    "Bundesliga": 78, "Ligue1": 61,
    "UCL": 2, "UEL": 3, "UECL": 848,
}

# Total matches per season (used to determine "late season")
LEAGUE_TOTAL_MATCHES: Dict[str, int] = {
    "EPL": 38, "LaLiga": 38, "SerieA": 38,
    "Bundesliga": 34, "Ligue1": 34,
}

SEASON = 2025
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

# Cache: {league: (timestamp, standings_list)}
_standings_cache: Dict[str, Tuple[float, list]] = {}
_CACHE_TTL = 3600  # 1 hour — standings don't change mid-day

# Minimum gameweek before stakes matter
MIN_GAMEWEEK = 25


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TeamStake:
    """What a team is playing for."""
    team: str
    rank: int
    points: int
    gameweek: int
    zone: str              # "title", "ucl", "uel", "uecl", "mid", "relegation", "relegated", "champion"
    points_to_target: int  # points gap to the zone boundary (negative = inside, positive = chasing)
    intensity: float       # 0.0 (nothing to play for) to 1.0 (maximum stakes)


@dataclass
class StakeContext:
    """Combined stake context for a fixture."""
    home_stake: Optional[TeamStake] = None
    away_stake: Optional[TeamStake] = None
    combined_intensity: float = 0.5   # 0.0 (dead rubber) to 1.0 (both fighting)
    # Modifiers (1.0 = neutral)
    goals_modifier: float = 1.0
    corners_modifier: float = 1.0
    cards_modifier: float = 1.0
    sot_modifier: float = 1.0
    description: str = ""
    source: str = "none"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def _get_api_key() -> Optional[str]:
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


def _fetch_standings(league: str) -> list:
    """Fetch league standings from API-Football. Cached 1 hour."""
    cached = _standings_cache.get(league)
    if cached and (time.time() - cached[0]) < _CACHE_TTL:
        return cached[1]

    api_id = LEAGUE_TO_API_ID.get(league)
    if not api_id or requests is None:
        return []

    api_key = _get_api_key()
    if not api_key:
        return []

    try:
        resp = requests.get(
            f"{API_FOOTBALL_BASE}/standings",
            headers={"x-apisports-key": api_key},
            params={"league": api_id, "season": SEASON},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("response", [])
        if not data:
            return []
        standings = data[0].get("league", {}).get("standings", [[]])[0]
        _standings_cache[league] = (time.time(), standings)
        log.info("Fetched standings for %s: %d teams", league, len(standings))
        return standings
    except Exception as exc:
        log.warning("Failed to fetch standings for %s: %s", league, exc)
        return []


# ---------------------------------------------------------------------------
# Stake classification
# ---------------------------------------------------------------------------

def _classify_team(entry: dict, standings: list, league: str) -> TeamStake:
    """Classify what a team is playing for based on their position and points."""
    team_name = entry.get("team", {}).get("name", "Unknown")
    rank = entry.get("rank", 0)
    points = entry.get("points", 0)
    played = entry.get("all", {}).get("played", 0)
    desc = (entry.get("description") or "").lower()
    total_teams = len(standings)
    total_matches = LEAGUE_TOTAL_MATCHES.get(league, 38)
    remaining = total_matches - played

    # Find zone boundaries from standings descriptions
    ucl_cutoff_rank = 0
    uel_cutoff_rank = 0
    relegation_start_rank = total_teams + 1

    for s in standings:
        s_desc = (s.get("description") or "").lower()
        r = s.get("rank", 0)
        if "champions league" in s_desc:
            ucl_cutoff_rank = max(ucl_cutoff_rank, r)
        if "europa league" in s_desc and "conference" not in s_desc:
            uel_cutoff_rank = max(uel_cutoff_rank, r)
        if "conference" in s_desc:
            uel_cutoff_rank = max(uel_cutoff_rank, r)
        if "relegation" in s_desc:
            relegation_start_rank = min(relegation_start_rank, r)

    # League-specific defaults if API descriptions are sparse
    if ucl_cutoff_rank == 0:
        ucl_cutoff_rank = 4
    if uel_cutoff_rank == 0:
        uel_cutoff_rank = max(ucl_cutoff_rank + 2, 6)
    if relegation_start_rank > total_teams:
        relegation_start_rank = total_teams - 2

    # Points at specific positions
    def _points_at_rank(r: int) -> int:
        for s in standings:
            if s.get("rank") == r:
                return s.get("points", 0)
        return 0

    # Classify based on POSITION and POINTS GAPS, not API description
    intensity = 0.5
    points_to_target = 0
    zone = "mid"

    if rank == 1:
        gap_to_second = points - _points_at_rank(2)
        max_remaining_pts = remaining * 3
        if gap_to_second > max_remaining_pts:
            zone = "champion"
            intensity = 0.3
        elif gap_to_second <= 6:
            zone = "title"
            intensity = 1.0
        else:
            zone = "title"
            intensity = 0.75
        points_to_target = -gap_to_second
    elif rank == 2 and (points + remaining * 3) >= _points_at_rank(1):
        gap = _points_at_rank(1) - points
        zone = "title"
        intensity = 0.95 if gap <= 6 else 0.75
        points_to_target = gap
    elif rank <= ucl_cutoff_rank:
        gap_below = points - _points_at_rank(ucl_cutoff_rank + 1)
        zone = "ucl"
        intensity = 0.85 if gap_below <= 6 else 0.65
        points_to_target = -gap_below
    elif rank <= uel_cutoff_rank:
        gap_to_ucl = _points_at_rank(ucl_cutoff_rank) - points
        zone = "uel"
        intensity = 0.80 if gap_to_ucl <= 6 else 0.65
        points_to_target = gap_to_ucl
    elif rank <= uel_cutoff_rank + 3:
        # Chasing European spots
        gap_to_europe = _points_at_rank(uel_cutoff_rank) - points
        if gap_to_europe <= remaining * 3:
            zone = "europe_chase"
            intensity = 0.75 if gap_to_europe <= 6 else 0.60
        else:
            zone = "mid"
            intensity = 0.30
        points_to_target = gap_to_europe
    elif rank >= relegation_start_rank:
        gap_to_safety = _points_at_rank(relegation_start_rank - 1) - points
        zone = "relegation"
        intensity = min(1.0, 0.85 + gap_to_safety * 0.02)
        points_to_target = gap_to_safety
    elif rank >= relegation_start_rank - 3:
        gap_above_drop = points - _points_at_rank(relegation_start_rank)
        if gap_above_drop <= 6:
            zone = "relegation_risk"
            intensity = 0.75
        else:
            zone = "mid"
            intensity = 0.30
        points_to_target = -gap_above_drop
    else:
        # True mid-table — safe from relegation, can't reach Europe
        gap_to_europe = _points_at_rank(uel_cutoff_rank) - points
        gap_above_drop = points - _points_at_rank(relegation_start_rank)
        if gap_to_europe > remaining * 3 and gap_above_drop > 6:
            zone = "mid"
            intensity = 0.25
        elif gap_to_europe <= remaining * 3:
            zone = "europe_chase"
            intensity = 0.55
        else:
            zone = "mid"
            intensity = 0.35

    # Scale intensity by season progress — stakes matter more late
    if played < MIN_GAMEWEEK:
        intensity = 0.5  # neutral before GW25

    season_factor = min(1.0, max(0.0, (played - MIN_GAMEWEEK) / (total_matches - MIN_GAMEWEEK)))
    intensity = 0.5 + (intensity - 0.5) * season_factor

    return TeamStake(
        team=team_name,
        rank=rank,
        points=points,
        gameweek=played,
        zone=zone,
        points_to_target=points_to_target,
        intensity=round(intensity, 3),
    )


# ---------------------------------------------------------------------------
# Modifier computation
# ---------------------------------------------------------------------------

def _compute_modifiers(home_intensity: float, away_intensity: float) -> Tuple[float, float, float, float]:
    """Compute projection modifiers from combined stake intensity.

    Returns (goals_mod, corners_mod, cards_mod, sot_mod).
    """
    # Combined intensity: both teams' stakes matter
    # Use the higher intensity as base, boosted if both are high
    combined = max(home_intensity, away_intensity)
    if home_intensity >= 0.7 and away_intensity >= 0.7:
        combined = min(1.0, combined + 0.1)  # both fighting = extra intensity

    # Convert intensity (0-1) to modifiers
    # 0.5 = neutral (1.0 modifier), below = dampened, above = boosted
    delta = combined - 0.5

    goals_mod = 1.0 + delta * 0.16      # ±8% max
    corners_mod = 1.0 + delta * 0.20    # ±10% max
    cards_mod = 1.0 + delta * 0.30      # ±15% max (biggest swing)
    sot_mod = 1.0 + delta * 0.16        # ±8% max

    return (
        round(goals_mod, 3),
        round(corners_mod, 3),
        round(cards_mod, 3),
        round(sot_mod, 3),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _normalize_team(name: str) -> str:
    return name.strip().lower().replace("fc ", "").replace(" fc", "")


def get_stake_context(
    home_team: str,
    away_team: str,
    league: str,
) -> StakeContext:
    """Get stake context for a domestic league fixture.

    Returns StakeContext with intensity scores and projection modifiers.
    European competitions return neutral (stakes are always high).
    """
    # European comps always have high stakes — knockout context handles that
    if league not in LEAGUE_TO_API_ID or league in ("UCL", "UEL", "UECL"):
        return StakeContext()

    standings = _fetch_standings(league)
    if not standings:
        return StakeContext()

    # Match teams to standings entries
    home_norm = _normalize_team(home_team)
    away_norm = _normalize_team(away_team)

    home_entry = None
    away_entry = None
    for s in standings:
        s_name = _normalize_team(s.get("team", {}).get("name", ""))
        if not home_entry and (home_norm in s_name or s_name in home_norm):
            home_entry = s
        if not away_entry and (away_norm in s_name or s_name in away_norm):
            away_entry = s

    if not home_entry and not away_entry:
        return StakeContext()

    home_stake = _classify_team(home_entry, standings, league) if home_entry else None
    away_stake = _classify_team(away_entry, standings, league) if away_entry else None

    home_int = home_stake.intensity if home_stake else 0.5
    away_int = away_stake.intensity if away_stake else 0.5
    combined = max(home_int, away_int)
    if home_int >= 0.7 and away_int >= 0.7:
        combined = min(1.0, combined + 0.1)

    goals_mod, corners_mod, cards_mod, sot_mod = _compute_modifiers(home_int, away_int)

    # Build description
    parts = []
    if home_stake:
        parts.append(f"{home_stake.team}: {home_stake.zone} (#{home_stake.rank}, {home_stake.intensity:.0%} intensity)")
    if away_stake:
        parts.append(f"{away_stake.team}: {away_stake.zone} (#{away_stake.rank}, {away_stake.intensity:.0%} intensity)")

    ctx = StakeContext(
        home_stake=home_stake,
        away_stake=away_stake,
        combined_intensity=round(combined, 3),
        goals_modifier=goals_mod,
        corners_modifier=corners_mod,
        cards_modifier=cards_mod,
        sot_modifier=sot_mod,
        description=" | ".join(parts),
        source="api",
    )

    if abs(goals_mod - 1.0) >= 0.02:
        log.info("Stake context: %s vs %s — %s — mods: G=%.2f C=%.2f K=%.2f",
                 home_team, away_team, ctx.description,
                 goals_mod, corners_mod, cards_mod)

    return ctx


def stake_evidence_line(ctx: StakeContext) -> Optional[str]:
    """Generate a human-readable evidence line for stake context."""
    if ctx.source == "none" or not ctx.description:
        return None

    mods = []
    for label, val in [("goals", ctx.goals_modifier), ("corners", ctx.corners_modifier),
                       ("cards", ctx.cards_modifier), ("SoT", ctx.sot_modifier)]:
        if abs(val - 1.0) >= 0.02:
            pct = (val - 1.0) * 100
            mods.append(f"{label} {pct:+.0f}%")

    if not mods:
        return None

    mod_str = ", ".join(mods)
    return f"Stakes: {ctx.description} \u2014 Modifiers: {mod_str}"
