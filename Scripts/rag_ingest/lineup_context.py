"""Lineup / injury awareness module for match-level projection adjustments.

When key players are missing from a confirmed starting XI, team output in goals,
SoT, and cards can shift meaningfully.  This module:

1. Fetches confirmed lineups from API-Football (``/fixtures/lineups``).
2. Queries Chroma for ``player_profile`` docs to compute each player's share
   of the team's statistical output.
3. Identifies missing key contributors (>10% share in any stat).
4. Produces per-stat multipliers (0.75--1.0) that downstream projection
   functions apply *after* the base projection (same pattern as
   ``KnockoutContext``).

Graceful degradation: every failure returns neutral multipliers (1.0).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("lineup_context")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEAGUE_TO_API_ID: Dict[str, int] = {
    "EPL": 39, "LaLiga": 140, "SerieA": 135,
    "Bundesliga": 78, "Ligue1": 61,
    "UCL": 2, "UEL": 3, "UECL": 848,
}

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
SEASON = 2025

# Player contribution thresholds
KEY_PLAYER_SHARE_THRESHOLD = 0.10   # >10% of any stat = key player
DAMPENING_FACTOR = 0.50             # teams partially compensate for absences
CORNERS_DAMPENING_SCALE = 0.30      # corners are team-tactical, not player-driven
MULTIPLIER_FLOOR = 0.75             # max 25% reduction even with many absences

# Stats tracked for contribution shares
TRACKED_STATS = ("goals", "sot", "cards")

# Cache: {cache_key: (timestamp, LineupContext)}
_lineup_cache: Dict[str, Tuple[float, "LineupContext"]] = {}
_CACHE_TTL = 300  # 5 minutes

# Player contribution cache: {(team_lower, league): Dict[player, Dict[stat, share]]}
_contribution_cache: Dict[Tuple[str, str], Tuple[float, Dict[str, Dict[str, float]]]] = {}
_CONTRIBUTION_CACHE_TTL = 900  # 15 minutes


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CardRiskProfile:
    """Expected-XI card-risk summary for one team.

    Built from player_profile metadata (starter_cards_per_90, starter_fouls_per_90,
    minutes_risk, recent_role_trend, position) so it works before confirmed lineups.
    """
    team_starter_cards_per_90: float = 0.0    # sum of starter cards_per_90 (expected XI)
    team_starter_fouls_per_90: float = 0.0    # sum of starter fouls_per_90
    high_card_risk_players: int = 0           # starters with cards_per_90 >= 0.25
    high_foul_players: int = 0                # starters with fouls_per_90 >= 1.5
    position_risk_score: float = 0.0          # positional card-risk score (DM/CB/FB weighted)
    minutes_risk_factor: float = 1.0          # 1.0 = normal, <1 = rotation risk
    evidence_lines: List[str] = field(default_factory=list)
    n_players: int = 0                        # number of player profiles evaluated


@dataclass
class LineupContext:
    """Lineup-aware adjustment context for a fixture."""
    home_starters: List[str] = field(default_factory=list)
    away_starters: List[str] = field(default_factory=list)
    home_adjustment: Dict[str, float] = field(default_factory=lambda: {
        "goals": 1.0, "corners": 1.0, "cards": 1.0, "sot": 1.0,
    })
    away_adjustment: Dict[str, float] = field(default_factory=lambda: {
        "goals": 1.0, "corners": 1.0, "cards": 1.0, "sot": 1.0,
    })
    home_missing_key: List[str] = field(default_factory=list)
    away_missing_key: List[str] = field(default_factory=list)
    source: str = "unavailable"    # "lineups" or "unavailable"
    is_available: bool = False
    # Card-risk profiles (available even when lineups are not confirmed)
    home_card_risk: Optional["CardRiskProfile"] = None
    away_card_risk: Optional["CardRiskProfile"] = None


# ---------------------------------------------------------------------------
# API helpers (lazy imports)
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


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents-naive, collapse whitespace."""
    return " ".join(name.strip().lower().split())


def _names_match(a: str, b: str) -> bool:
    """Fuzzy player name matching.

    Handles API-Football short names (e.g. ``M. Salah``) vs Chroma full
    names (e.g. ``Mohamed Salah``).  Also handles reversed order and
    substring containment for surnames.
    """
    na = _normalize_name(a)
    nb = _normalize_name(b)
    if na == nb:
        return True

    # Exact surname match (last token)
    parts_a = na.split()
    parts_b = nb.split()
    if parts_a and parts_b and parts_a[-1] == parts_b[-1]:
        # If surnames match, check first initial consistency
        if len(parts_a) >= 1 and len(parts_b) >= 1:
            # "m. salah" vs "mohamed salah" — first char of first token
            fa = parts_a[0].rstrip(".")
            fb = parts_b[0].rstrip(".")
            if fa and fb and fa[0] == fb[0]:
                return True

    # Substring containment for short names
    if len(na) >= 4 and len(nb) >= 4:
        if na in nb or nb in na:
            return True

    return False


def _team_matches_fuzzy(name1: str, name2: str) -> bool:
    """Check if two team names refer to the same team."""
    n1 = name1.strip().lower().replace(".", "").replace("'", "").replace("-", "")
    n2 = name2.strip().lower().replace(".", "").replace("'", "").replace("-", "")
    # Remove common suffixes
    for s in (" fc", " cf", " afc"):
        n1 = n1.replace(s, "")
        n2 = n2.replace(s, "")
    n1 = n1.strip()
    n2 = n2.strip()
    if n1 == n2:
        return True
    if len(n1) >= 4 and len(n2) >= 4:
        if n1 in n2 or n2 in n1:
            return True
    return False


# ---------------------------------------------------------------------------
# Lineup fetching
# ---------------------------------------------------------------------------

def _find_fixture_id(
    home: str,
    away: str,
    league: str,
    fixture_date: str,
    api_key: str,
) -> Optional[int]:
    """Find the API-Football fixture ID for a match."""
    try:
        import requests as _req
    except ImportError:
        return None

    league_id = LEAGUE_TO_API_ID.get(league)
    if not league_id:
        return None

    try:
        resp = _req.get(
            f"{API_FOOTBALL_BASE}/fixtures",
            headers={"x-apisports-key": api_key},
            params={"league": league_id, "season": SEASON, "date": fixture_date},
            timeout=15,
        )
        resp.raise_for_status()
        fixtures = resp.json().get("response", [])
    except Exception as exc:
        log.warning("Fixture lookup failed for %s vs %s: %s", home, away, exc)
        return None

    for f in fixtures:
        teams = f.get("teams", {})
        h = teams.get("home", {}).get("name", "")
        a = teams.get("away", {}).get("name", "")
        if _team_matches_fuzzy(home, h) and _team_matches_fuzzy(away, a):
            return f.get("fixture", {}).get("id")

    return None


def _fetch_lineups_raw(fixture_id: int, api_key: str) -> Optional[List[Dict]]:
    """Fetch raw lineup data from API-Football."""
    try:
        import requests as _req
    except ImportError:
        return None

    try:
        resp = _req.get(
            f"{API_FOOTBALL_BASE}/fixtures/lineups",
            headers={"x-apisports-key": api_key},
            params={"fixture": fixture_id},
            timeout=15,
        )
        resp.raise_for_status()
        lineups = resp.json().get("response", [])
        return lineups if lineups else None
    except Exception as exc:
        log.warning("Lineup fetch failed for fixture %d: %s", fixture_id, exc)
        return None


def fetch_match_lineups(
    home: str,
    away: str,
    league: str,
    fixture_date: str,
) -> Tuple[List[str], List[str], bool]:
    """Fetch confirmed starting XIs from API-Football.

    Returns (home_starters, away_starters, is_available).
    Names are returned in their original casing (lowercase-normalized
    internally for matching).
    """
    api_key = _get_api_key()
    if not api_key:
        log.debug("No API-Football key — lineup fetch skipped")
        return [], [], False

    fixture_id = _find_fixture_id(home, away, league, fixture_date, api_key)
    if fixture_id is None:
        log.debug("No fixture found for %s vs %s on %s", home, away, fixture_date)
        return [], [], False

    lineups = _fetch_lineups_raw(fixture_id, api_key)
    if not lineups:
        log.debug("Lineups not yet published for fixture %d", fixture_id)
        return [], [], False

    home_starters: List[str] = []
    away_starters: List[str] = []

    for team_lineup in lineups:
        team_name = team_lineup.get("team", {}).get("name", "")
        starters = []
        for p in team_lineup.get("startXI", []):
            name = p.get("player", {}).get("name", "")
            if name:
                starters.append(name)

        if _team_matches_fuzzy(home, team_name):
            home_starters = starters
        elif _team_matches_fuzzy(away, team_name):
            away_starters = starters

    is_available = bool(home_starters or away_starters)
    if is_available:
        log.info(
            "Lineups fetched: %s (%d starters) vs %s (%d starters)",
            home, len(home_starters), away, len(away_starters),
        )

    return home_starters, away_starters, is_available


# ---------------------------------------------------------------------------
# Player contribution shares (Chroma)
# ---------------------------------------------------------------------------

def _get_chroma_collection():
    """Lazy import and return the Chroma collection handle."""
    try:
        from chroma_backend import get_chroma_client
    except ImportError:
        from Scripts.rag_ingest.chroma_backend import get_chroma_client

    import pathlib
    chroma_dir = str(pathlib.Path(__file__).resolve().parent.parent.parent / "Index" / "chroma")
    client = get_chroma_client(chroma_dir)
    return client.get_or_create_collection("football_top5")


def _build_where(league: str, extra_filters: List[Dict]) -> Dict:
    """Minimal where-clause builder matching rag_cli_v2.build_where."""
    clauses: List[Dict] = []
    if league:
        clauses.append({"league": league})
    clauses.extend(extra_filters)
    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _team_variants(team_name: str) -> List[str]:
    """Generate team name variants for Chroma lookup."""
    base = team_name.strip()
    variants = list({base, base.lower(), base.title()})
    for suffix in [" united", " hotspur", " fc", " afc", " cf", " city"]:
        for b in [base.lower()]:
            if b.endswith(suffix):
                stripped = b[: -len(suffix)].strip().title()
                if stripped and stripped not in variants:
                    variants.append(stripped)
    for prefix in ["afc ", "fc "]:
        for b in [base.lower()]:
            if b.startswith(prefix):
                stripped = b[len(prefix):].strip().title()
                if stripped and stripped not in variants:
                    variants.append(stripped)
    return variants


def get_player_contribution_shares(
    team: str,
    league: str,
) -> Dict[str, Dict[str, float]]:
    """Query Chroma for player_profile docs and compute stat contribution shares.

    Returns ``{player_name: {goals: share, sot: share, cards: share}}``.
    Shares sum to ~1.0 per stat across all squad players with data.

    Corners are intentionally excluded — they are a team-tactical event,
    not driven by individual players.
    """
    cache_key = (team.strip().lower(), league)
    cached = _contribution_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _CONTRIBUTION_CACHE_TTL:
        return cached[1]

    try:
        col = _get_chroma_collection()
    except Exception as exc:
        log.warning("Chroma unavailable for contribution shares: %s", exc)
        return {}

    # Fetch player profiles for this team
    variants = _team_variants(team)
    player_rows: List[Dict] = []

    for tv in variants:
        where = _build_where(league, [
            {"doc_type": "player_profile"},
            {"team": tv},
        ])
        try:
            res = col.get(where=where, include=["metadatas"], limit=40)
        except Exception:
            continue
        for m in (res.get("metadatas") or []):
            if m:
                player_rows.append(m)
        if player_rows:
            break  # found data with this variant

    if not player_rows:
        log.debug("No player profiles found for %s (%s)", team, league)
        _contribution_cache[cache_key] = (time.time(), {})
        return {}

    # Deduplicate by player_id
    seen_ids = set()
    unique_rows: List[Dict] = []
    for row in player_rows:
        pid = row.get("player_id")
        if pid and pid in seen_ids:
            continue
        if pid:
            seen_ids.add(pid)
        unique_rows.append(row)

    # Compute raw per-90 totals.
    # Use starter-specific rates when available (more representative of
    # what the XI actually produces); fall back to overall per-90.
    stat_field_map = {
        "goals": ("starter_goals_per_90", "goals_per_90"),
        "sot":   ("starter_sot_per_90",   "sot_per_90"),
        "cards": ("starter_cards_per_90",  "cards_per_90"),
    }

    # Accumulate raw rates per player
    raw: Dict[str, Dict[str, float]] = {}  # player_name -> {stat: rate}
    totals: Dict[str, float] = {s: 0.0 for s in TRACKED_STATS}

    for row in unique_rows:
        pname = row.get("player_name") or ""
        if not pname:
            continue

        player_rates: Dict[str, float] = {}
        for stat, (starter_key, fallback_key) in stat_field_map.items():
            rate = None
            for k in (starter_key, fallback_key):
                try:
                    v = row.get(k)
                    if v is not None:
                        rate = float(v)
                        break
                except (TypeError, ValueError):
                    continue
            if rate is not None and rate >= 0:
                player_rates[stat] = rate
                totals[stat] += rate

        if player_rates:
            raw[pname] = player_rates

    # Convert to shares
    shares: Dict[str, Dict[str, float]] = {}
    for pname, rates in raw.items():
        player_shares: Dict[str, float] = {}
        for stat, rate in rates.items():
            total = totals.get(stat, 0.0)
            if total > 0:
                player_shares[stat] = rate / total
            else:
                player_shares[stat] = 0.0
        shares[pname] = player_shares

    _contribution_cache[cache_key] = (time.time(), shares)
    log.debug(
        "Computed contribution shares for %s (%s): %d players",
        team, league, len(shares),
    )
    return shares


# ---------------------------------------------------------------------------
# Adjustment computation
# ---------------------------------------------------------------------------

def compute_lineup_adjustments(
    team: str,
    league: str,
    confirmed_starters: List[str],
    contribution_shares: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """Identify missing key contributors and compute stat multipliers.

    A "key player" has >10% contribution share in any stat.  For each
    missing key player, the relevant stat multiplier is reduced by
    ``share * DAMPENING_FACTOR``.  The multiplier is floored at
    ``MULTIPLIER_FLOOR`` (0.75).

    Returns ``(adjustments, missing_key_names)``.
    """
    neutral = {"goals": 1.0, "corners": 1.0, "cards": 1.0, "sot": 1.0}

    if not confirmed_starters:
        return neutral, []

    if contribution_shares is None:
        contribution_shares = get_player_contribution_shares(team, league)

    if not contribution_shares:
        return neutral, []

    # Build set of confirmed starter names (normalized)
    starter_set_norm = [_normalize_name(s) for s in confirmed_starters]

    # Identify which known players are NOT in the starting XI
    missing_key: List[str] = []
    adjustments = dict(neutral)

    for player_name, shares in contribution_shares.items():
        # Check if this player is in the confirmed lineup
        is_starting = False
        for sn in starter_set_norm:
            if _names_match(player_name, sn):
                is_starting = True
                break

        if is_starting:
            continue

        # Player is NOT starting — check if they are a key contributor
        is_key = any(
            s >= KEY_PLAYER_SHARE_THRESHOLD
            for s in shares.values()
        )
        if not is_key:
            continue

        missing_key.append(player_name)

        # Apply dampened reduction per stat
        for stat, share in shares.items():
            if share < 0.01:
                continue  # negligible contribution

            dampening = DAMPENING_FACTOR
            if stat == "corners":
                dampening *= CORNERS_DAMPENING_SCALE

            reduction = share * dampening
            adjustments[stat] = max(
                MULTIPLIER_FLOOR,
                adjustments[stat] - reduction,
            )

    # Corners always gets minimal treatment even for non-corner stats
    # (corners are team-tactical; individual absences matter less)
    # Already handled via CORNERS_DAMPENING_SCALE above.

    if missing_key:
        log.info(
            "Missing key players for %s: %s — adjustments: %s",
            team, ", ".join(missing_key),
            {k: f"{v:.3f}" for k, v in adjustments.items() if v != 1.0},
        )

    return adjustments, missing_key


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_lineup_context(
    home: str,
    away: str,
    league: str,
    fixture_date: str,
) -> LineupContext:
    """Orchestrate lineup fetch, contribution share lookup, and adjustment.

    This is the main entry point.  Returns a fully populated
    ``LineupContext`` with per-side multipliers.

    Falls back to neutral ``LineupContext`` (all multipliers 1.0,
    ``is_available=False``) on any failure.
    """
    cache_key = f"{home.lower()}|{away.lower()}|{league}|{fixture_date}"
    cached = _lineup_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL:
        return cached[1]

    ctx = LineupContext()

    try:
        home_starters, away_starters, available = fetch_match_lineups(
            home, away, league, fixture_date,
        )
    except Exception as exc:
        log.warning("Lineup fetch failed for %s vs %s: %s", home, away, exc)
        _lineup_cache[cache_key] = (time.time(), ctx)
        return ctx

    if not available:
        _lineup_cache[cache_key] = (time.time(), ctx)
        return ctx

    ctx.home_starters = home_starters
    ctx.away_starters = away_starters
    ctx.source = "lineups"
    ctx.is_available = True

    # Compute contribution shares (may use Chroma)
    try:
        home_shares = get_player_contribution_shares(home, league)
        away_shares = get_player_contribution_shares(away, league)
    except Exception as exc:
        log.warning("Contribution share computation failed: %s", exc)
        _lineup_cache[cache_key] = (time.time(), ctx)
        return ctx

    # Compute adjustments
    try:
        home_adj, home_missing = compute_lineup_adjustments(
            home, league, home_starters, home_shares,
        )
        away_adj, away_missing = compute_lineup_adjustments(
            away, league, away_starters, away_shares,
        )
    except Exception as exc:
        log.warning("Lineup adjustment computation failed: %s", exc)
        _lineup_cache[cache_key] = (time.time(), ctx)
        return ctx

    ctx.home_adjustment = home_adj
    ctx.away_adjustment = away_adj
    ctx.home_missing_key = home_missing
    ctx.away_missing_key = away_missing

    # Compute card risk profiles (works even without confirmed lineups)
    try:
        ctx.home_card_risk = compute_card_risk_profile(home, league, home_shares)
        ctx.away_card_risk = compute_card_risk_profile(away, league, away_shares)
    except Exception as exc:
        log.warning("Card risk computation failed: %s", exc)

    _lineup_cache[cache_key] = (time.time(), ctx)

    log.info(
        "Lineup context ready: %s vs %s — home missing %d key, away missing %d key",
        home, away, len(home_missing), len(away_missing),
    )
    return ctx


# ---------------------------------------------------------------------------
# Evidence rendering
# ---------------------------------------------------------------------------

def lineup_evidence_line(ctx: LineupContext) -> Optional[str]:
    """Generate a human-readable evidence line describing lineup adjustments.

    Returns ``None`` if lineups are not available or there are no
    adjustments to report.
    """
    if not ctx.is_available or ctx.source == "unavailable":
        return None

    parts: List[str] = []

    for side, missing, adj in [
        ("Home", ctx.home_missing_key, ctx.home_adjustment),
        ("Away", ctx.away_missing_key, ctx.away_adjustment),
    ]:
        if not missing:
            continue

        # Build modification summary
        mods = []
        for stat_label, stat_key in [
            ("goals", "goals"), ("SoT", "sot"),
            ("cards", "cards"), ("corners", "corners"),
        ]:
            val = adj.get(stat_key, 1.0)
            if val != 1.0:
                pct = (val - 1.0) * 100
                mods.append(f"{stat_label} {pct:+.0f}%")

        mod_str = ", ".join(mods) if mods else "minimal impact"
        names_str = ", ".join(missing[:3])
        if len(missing) > 3:
            names_str += f" +{len(missing) - 3} more"

        parts.append(f"{side} missing: {names_str} -> {mod_str}")

    if not parts:
        return None

    return "Lineup adj: " + " | ".join(parts) + "."


# ---------------------------------------------------------------------------
# Card-risk profiling (works before confirmed lineups)
# ---------------------------------------------------------------------------

# Position weights for card risk: defensive midfielders, centre-backs, full-backs
# get more cards on average than attackers or goalkeepers
_POSITION_CARD_WEIGHTS = {
    "Goalkeeper": 0.02,
    "Defender": 0.18,       # CB/FB blended
    "Midfielder": 0.22,     # DM/CM/AM blended
    "Attacker": 0.10,
}


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def compute_card_risk_profile(
    team: str,
    league: str,
    contribution_shares: Optional[Dict[str, Dict[str, float]]] = None,
) -> CardRiskProfile:
    """Build an expected-XI card-risk profile from player_profile metadata.

    Uses starter_cards_per_90, starter_fouls_per_90, minutes_risk,
    recent_role_trend, and position — all available in player_profile docs.
    Works before confirmed lineups (uses likely starters based on minutes).
    """
    profile = CardRiskProfile()

    try:
        col = _get_chroma_collection()
    except Exception as exc:
        log.warning("Chroma unavailable for card risk profile: %s", exc)
        return profile

    # Fetch player profiles for this team
    variants = _team_variants(team)
    player_rows: List[Dict] = []

    for tv in variants:
        where = _build_where(league, [
            {"doc_type": "player_profile"},
            {"team": tv},
        ])
        try:
            res = col.get(where=where, include=["metadatas"], limit=40)
        except Exception:
            continue
        for m in (res.get("metadatas") or []):
            if m:
                player_rows.append(m)
        if player_rows:
            break

    if not player_rows:
        return profile

    # Deduplicate by player_id
    seen_ids = set()
    unique_rows: List[Dict] = []
    for row in player_rows:
        pid = row.get("player_id")
        if pid and pid in seen_ids:
            continue
        if pid:
            seen_ids.add(pid)
        unique_rows.append(row)

    # Identify likely starters: sort by minutes_risk (or total_minutes) descending,
    # take top 11 outfield + GK
    def _sort_key(row):
        mr = _safe_float(row.get("minutes_risk"), 0.0)
        tm = _safe_float(row.get("total_minutes"), 0.0)
        return (mr, tm)

    unique_rows.sort(key=_sort_key, reverse=True)
    likely_starters = unique_rows[:11]  # top 11 by playing time

    total_cards_90 = 0.0
    total_fouls_90 = 0.0
    high_card_risk = 0
    high_foul = 0
    position_risk = 0.0
    minutes_risk_sum = 0.0
    minutes_risk_count = 0
    evidence_parts: List[str] = []

    for row in likely_starters:
        pname = row.get("player_name", "Unknown")
        pos = row.get("position", "")
        cards_90 = _safe_float(row.get("starter_cards_per_90") or row.get("cards_per_90"), 0.0)
        fouls_90 = _safe_float(row.get("starter_fouls_per_90"), 0.0)
        mr = _safe_float(row.get("minutes_risk"), 1.0)

        total_cards_90 += cards_90
        total_fouls_90 += fouls_90

        if cards_90 >= 0.25:
            high_card_risk += 1
            evidence_parts.append(f"{pname} ({pos}): {cards_90:.2f} cards/90")

        if fouls_90 >= 1.5:
            high_foul += 1

        # Position risk score
        pos_w = _POSITION_CARD_WEIGHTS.get(pos, 0.12)
        position_risk += pos_w * cards_90

        if mr > 0:
            minutes_risk_sum += mr
            minutes_risk_count += 1

    profile.team_starter_cards_per_90 = round(total_cards_90, 3)
    profile.team_starter_fouls_per_90 = round(total_fouls_90, 3)
    profile.high_card_risk_players = high_card_risk
    profile.high_foul_players = high_foul
    profile.position_risk_score = round(position_risk, 3)
    profile.minutes_risk_factor = round(
        minutes_risk_sum / minutes_risk_count if minutes_risk_count > 0 else 1.0, 3
    )
    profile.n_players = len(likely_starters)

    if evidence_parts:
        profile.evidence_lines = [f"High-card starters: {'; '.join(evidence_parts[:3])}"]
    if high_card_risk >= 3:
        profile.evidence_lines.append(f"{high_card_risk} starters with cards/90 >= 0.25")

    return profile


def get_card_risk_profiles(
    home: str,
    away: str,
    league: str,
) -> Tuple[CardRiskProfile, CardRiskProfile]:
    """Get card-risk profiles for both teams (no API call needed).

    Works before confirmed lineups by using player_profile metadata
    to estimate expected-XI card risk.
    """
    try:
        home_risk = compute_card_risk_profile(home, league)
    except Exception:
        home_risk = CardRiskProfile()
    try:
        away_risk = compute_card_risk_profile(away, league)
    except Exception:
        away_risk = CardRiskProfile()
    return home_risk, away_risk


def card_risk_evidence_line(ctx: "LineupContext") -> Optional[str]:
    """Generate card-risk evidence line from lineup context."""
    parts: List[str] = []

    for side, risk in [("Home", ctx.home_card_risk), ("Away", ctx.away_card_risk)]:
        if risk is None or risk.n_players == 0:
            continue
        if risk.high_card_risk_players >= 2:
            parts.append(
                f"{side}: {risk.high_card_risk_players} high-card-risk starters "
                f"({risk.team_starter_cards_per_90:.2f} cards/90 total)"
            )

    if not parts:
        return None
    return "Card risk: " + " | ".join(parts) + "."
