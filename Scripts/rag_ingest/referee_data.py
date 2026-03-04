"""Referee data layer for yellow-card prediction.

Two-layer architecture:
  Static  – historical referee profiles rebuilt weekly from API-Football
            fixtures cross-referenced with team_engineered_features card data.
            Stored as Index/referee_profiles.json (~100-200 entries).
  Dynamic – live referee assignments fetched at query time from API-Football,
            cached in-memory for 15 min per league.

Modifier formula:
  confidence = clamp((sample - MIN_SAMPLE) / (FULL_CONFIDENCE - MIN_SAMPLE), 0, 1)
  multiplier = 1.0 + (strictness - 1.0) * confidence * weight

Graceful degradation: every failure path returns multiplier=1.0 (no effect).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "Output"
PROFILES_PATH = ROOT / "Index" / "referee_profiles.json"

LEAGUE_TO_API_ID: Dict[str, int] = {
    "EPL": 39,
    "LaLiga": 140,
    "SerieA": 135,
    "Bundesliga": 78,
    "Ligue1": 61,
}

LEAGUE_TO_FE_DIR: Dict[str, str] = {
    "EPL": "Prem_feature_engineering",
    "LaLiga": "LaLiga_feature_engineering",
    "SerieA": "SeriaA_feature_engineering",
    "Bundesliga": "Bundesliga_feature_engineering",
    "Ligue1": "Ligue1_feature_engineering",
}

SEASON = 2025
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
MIN_SAMPLE = 5
FULL_CONFIDENCE = 20
DEFAULT_WEIGHT = 0.25
ASSIGNMENT_CACHE_TTL = 900  # seconds (15 min)

_api_key_warned = False

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RefereeProfile:
    name: str
    country: Optional[str] = None
    leagues: List[str] = field(default_factory=list)
    total_matches: int = 0
    avg_cards_per_match: float = 0.0
    avg_fouls_per_match: float = 0.0
    avg_yellows_per_match: float = 0.0
    avg_reds_per_match: float = 0.0
    strictness_ratio: float = 1.0
    cards_per_foul: float = 0.0
    last_updated: str = ""


@dataclass
class RefereeModifier:
    multiplier: float = 1.0
    referee_name: Optional[str] = None
    sample_size: int = 0
    confidence: float = 0.0
    strictness_ratio: float = 1.0
    avg_cards_per_match: float = 0.0
    source: str = "unavailable"  # "profile" | "fallback" | "unavailable"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _api_football_key() -> Optional[str]:
    for k in ("API-FOOTBALL-KEY", "API_FOOTBALL_KEY"):
        v = os.environ.get(k)
        if v and str(v).strip():
            return str(v).strip().strip("'\"")
    return None


def _api_get(endpoint: str, params: dict) -> Optional[list]:
    """Call API-Football. Returns the 'response' list or None on failure."""
    key = _api_football_key()
    if not key:
        global _api_key_warned
        if not _api_key_warned:
            print("[referee_data] Warning: API-FOOTBALL-KEY not set. Referee features disabled.")
            _api_key_warned = True
        return None
    url = f"{API_FOOTBALL_BASE}{endpoint}"
    headers = {"x-apisports-key": key}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=25)
        if r.status_code != 200:
            print(f"[referee_data] API-Football HTTP {r.status_code}: {(r.text or '')[:200]}")
            return None
        data = r.json()
        return data.get("response", [])
    except Exception as exc:
        print(f"[referee_data] API-Football request failed: {exc}")
        return None


def _normalize_ref_name(ref_str: Optional[str]) -> Optional[str]:
    """'Michael Oliver, England' → 'michael oliver'. Returns None if empty."""
    if not ref_str or not ref_str.strip():
        return None
    name = ref_str.split(",")[0].strip()
    return name.lower() if name else None


def _parse_ref_country(ref_str: Optional[str]) -> Optional[str]:
    """'Michael Oliver, England' → 'England'."""
    if not ref_str or "," not in ref_str:
        return None
    return ref_str.split(",", 1)[1].strip() or None


# ---------------------------------------------------------------------------
# Static layer: build referee profiles
# ---------------------------------------------------------------------------

def _load_team_fixture_stats(league: str) -> List[dict]:
    """Load team_engineered_features JSON for a league (all available seasons)."""
    fe_dir = LEAGUE_TO_FE_DIR.get(league)
    if not fe_dir:
        return []
    base = OUTPUT_DIR / fe_dir
    rows: List[dict] = []
    for path in sorted(base.glob("team_engineered_features_*.json")):
        rows.extend(json.loads(path.read_text()))
    if not rows:
        print(f"[referee_data] No team_engineered_features_*.json found in {base}")
    return rows


def _sum_card_entry(entry: dict, r: dict) -> None:
    """Accumulate card/foul totals from a team fixture row into an entry dict."""
    yc = r.get("yellow_cards") or 0
    rc = r.get("red_cards") or 0
    cards = r.get("cards_total") or 0
    fouls = r.get("fouls_committed") or 0
    try:
        entry["total_yellows"] += int(float(yc))
        entry["total_reds"] += int(float(rc))
        entry["total_cards"] += int(float(cards))
        entry["total_fouls"] += int(float(fouls))
    except (TypeError, ValueError):
        pass


def _new_card_entry() -> dict:
    return {"total_cards": 0, "total_fouls": 0, "total_yellows": 0, "total_reds": 0}


def _build_fixture_card_map(team_rows: List[dict]) -> Dict[int, dict]:
    """Group team_fixture rows by fixture_id, summing both teams' card/foul data.

    Returns { fixture_id: { total_cards, total_fouls, total_yellows, total_reds } }.
    """
    fixtures: Dict[int, dict] = {}
    for r in team_rows:
        fid = r.get("fixture_id")
        if fid is None:
            continue
        fid = int(fid)
        if fid not in fixtures:
            fixtures[fid] = _new_card_entry()
        _sum_card_entry(fixtures[fid], r)
    return fixtures


def _normalize_team(name: str) -> str:
    """Lowercase, strip whitespace, remove common suffixes for fuzzy matching."""
    return name.lower().strip()


def _build_fixture_card_map_by_name(team_rows: List[dict]) -> Dict[frozenset, dict]:
    """Group team_fixture rows by fixture name (pair of teams).

    Fallback for leagues without fixture_id. Groups rows by the 'fixture' field
    (e.g. 'Alaves vs Atletico Madrid'), summing card/foul data.

    Returns { frozenset({team_a_lower, team_b_lower}): card_data }.
    """
    fixtures: Dict[frozenset, dict] = {}
    for r in team_rows:
        fixture_str = r.get("fixture", "")
        team = _normalize_team(r.get("team", ""))
        if not fixture_str or not team:
            continue
        # Parse "TeamA vs TeamB" from fixture string
        parts = fixture_str.split(" vs ")
        if len(parts) != 2:
            continue
        key = frozenset(_normalize_team(p) for p in parts)
        if key not in fixtures:
            fixtures[key] = _new_card_entry()
        _sum_card_entry(fixtures[key], r)
    return fixtures


def fetch_completed_fixtures(league_id: int, season: int = SEASON) -> List[dict]:
    """Fetch all finished fixtures for a league/season from API-Football."""
    result = _api_get("/fixtures", {
        "league": league_id,
        "season": season,
        "status": "FT",
    })
    return result or []


def build_referee_profiles_for_league(
    league: str, season: int = SEASON,
) -> Tuple[Dict[str, dict], float, int]:
    """Build referee profiles for one league.

    Returns (per_referee_accum, league_avg_cards, matched_count).
    per_referee_accum: { ref_name_lower: { name, country, cards, fouls, yellows, reds, matches } }
    """
    league_id = LEAGUE_TO_API_ID.get(league)
    if league_id is None:
        return {}, 0.0, 0

    # Step 1: fetch fixtures from API-Football → get referee per fixture
    fixtures = fetch_completed_fixtures(league_id, season)
    if not fixtures:
        print(f"[referee_data] No completed fixtures from API-Football for {league}")
        return {}, 0.0, 0

    # Build two maps from API data: by fixture_id and by team-name pair
    fixture_ref_by_id: Dict[int, Tuple[str, Optional[str]]] = {}
    fixture_ref_by_name: Dict[frozenset, List[Tuple[str, Optional[str]]]] = {}
    for f in fixtures:
        fid = f.get("fixture", {}).get("id")
        ref_str = f.get("fixture", {}).get("referee")
        norm = _normalize_ref_name(ref_str)
        if not norm:
            continue
        country = _parse_ref_country(ref_str)
        if fid is not None:
            fixture_ref_by_id[int(fid)] = (norm, country)
        home = _normalize_team(f.get("teams", {}).get("home", {}).get("name", ""))
        away = _normalize_team(f.get("teams", {}).get("away", {}).get("name", ""))
        if home and away:
            name_key = frozenset({home, away})
            # Multiple matches between same teams → keep all via list
            fixture_ref_by_name.setdefault(name_key, []).append((norm, country))

    print(f"  [{league}] {len(fixtures)} fixtures from API, "
          f"{len(fixture_ref_by_id)} with fixture_id referee data, "
          f"{len(fixture_ref_by_name)} with name-based referee data")

    # Step 2: load our team fixture stats → build card maps
    team_rows = _load_team_fixture_stats(league)
    if not team_rows:
        return {}, 0.0, 0
    card_map_by_id = _build_fixture_card_map(team_rows)
    card_map_by_name = _build_fixture_card_map_by_name(team_rows) if not card_map_by_id else {}

    # Step 3: cross-reference (prefer fixture_id, fall back to name matching)
    ref_accum: Dict[str, dict] = {}
    all_match_cards = 0
    matched = 0

    def _accum_match(ref_name: str, ref_country: Optional[str], card_data: dict) -> None:
        nonlocal matched, all_match_cards
        matched += 1
        all_match_cards += card_data["total_cards"]
        if ref_name not in ref_accum:
            ref_accum[ref_name] = {
                "name": ref_name,
                "country": ref_country,
                "leagues": set(),
                "cards": 0, "fouls": 0, "yellows": 0, "reds": 0,
                "matches": 0,
            }
        acc = ref_accum[ref_name]
        acc["leagues"].add(league)
        acc["cards"] += card_data["total_cards"]
        acc["fouls"] += card_data["total_fouls"]
        acc["yellows"] += card_data["total_yellows"]
        acc["reds"] += card_data["total_reds"]
        acc["matches"] += 1

    if card_map_by_id:
        # Primary path: match by fixture_id (EPL and leagues with fixture_id)
        for fid, (ref_name, ref_country) in fixture_ref_by_id.items():
            card_data = card_map_by_id.get(fid)
            if card_data is not None:
                _accum_match(ref_name, ref_country, card_data)
    else:
        # Fallback path: match by team names (LaLiga, Bundesliga, Ligue1, etc.)
        for name_key, ref_list in fixture_ref_by_name.items():
            card_data = card_map_by_name.get(name_key)
            if card_data is None:
                continue
            # Divide card totals evenly across the number of matches between same teams
            n_matches = len(ref_list)
            per_match = {k: v / n_matches for k, v in card_data.items()} if n_matches > 1 else card_data
            for ref_name, ref_country in ref_list:
                _accum_match(ref_name, ref_country, per_match)

    league_avg = all_match_cards / matched if matched > 0 else 0.0
    print(f"  [{league}] Matched {matched}/{len(fixture_ref_by_id) or len(fixture_ref_by_name)} fixtures, "
          f"{len(ref_accum)} referees, league avg {league_avg:.2f} cards/match")

    return ref_accum, league_avg, matched


def build_all_referee_profiles(
    leagues: Optional[List[str]] = None,
    season: int = SEASON,
) -> Dict[str, RefereeProfile]:
    """Build and merge referee profiles across all requested leagues."""
    if leagues is None:
        leagues = list(LEAGUE_TO_API_ID.keys())

    # Accumulate per-referee data across all leagues
    merged: Dict[str, dict] = {}
    all_cards_total = 0
    all_matches_total = 0

    for lg in leagues:
        print(f"Building referee profiles for {lg}...")
        accum, _lg_avg, _matched = build_referee_profiles_for_league(lg, season)
        for ref_name, data in accum.items():
            if ref_name not in merged:
                merged[ref_name] = {
                    "name": data["name"],
                    "country": data["country"],
                    "leagues": set(),
                    "cards": 0, "fouls": 0, "yellows": 0, "reds": 0,
                    "matches": 0,
                }
            m = merged[ref_name]
            m["leagues"] |= data["leagues"]
            m["cards"] += data["cards"]
            m["fouls"] += data["fouls"]
            m["yellows"] += data["yellows"]
            m["reds"] += data["reds"]
            m["matches"] += data["matches"]
            if data["country"] and not m["country"]:
                m["country"] = data["country"]

        all_cards_total += sum(d["cards"] for d in accum.values())
        all_matches_total += sum(d["matches"] for d in accum.values())

    global_avg = all_cards_total / all_matches_total if all_matches_total > 0 else 4.0
    today_str = date.today().isoformat()

    profiles: Dict[str, RefereeProfile] = {}
    for ref_name, m in merged.items():
        matches = m["matches"]
        if matches == 0:
            continue
        avg_cards = m["cards"] / matches
        avg_fouls = m["fouls"] / matches
        avg_yellows = m["yellows"] / matches
        avg_reds = m["reds"] / matches
        cpf = avg_cards / avg_fouls if avg_fouls > 0 else 0.0

        # Compute strictness per-league weighted (use global avg)
        strictness = avg_cards / global_avg if global_avg > 0 else 1.0

        profiles[ref_name] = RefereeProfile(
            name=ref_name,
            country=m["country"],
            leagues=sorted(m["leagues"]),
            total_matches=matches,
            avg_cards_per_match=round(avg_cards, 3),
            avg_fouls_per_match=round(avg_fouls, 3),
            avg_yellows_per_match=round(avg_yellows, 3),
            avg_reds_per_match=round(avg_reds, 3),
            strictness_ratio=round(strictness, 4),
            cards_per_foul=round(cpf, 4),
            last_updated=today_str,
        )

    return profiles


# ---------------------------------------------------------------------------
# Save / load profiles
# ---------------------------------------------------------------------------

def save_profiles(profiles: Dict[str, RefereeProfile], path: Path = PROFILES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: asdict(v) for k, v in profiles.items()}
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
    print(f"Saved {len(profiles)} referee profiles to {path}")


def load_profiles(path: Path = PROFILES_PATH) -> Dict[str, RefereeProfile]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        out: Dict[str, RefereeProfile] = {}
        for k, v in raw.items():
            out[k] = RefereeProfile(**v)
        return out
    except Exception as exc:
        print(f"[referee_data] Failed to load profiles: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Dynamic layer: live referee assignments
# ---------------------------------------------------------------------------

_assignment_cache: Dict[str, Dict[Tuple[str, str], str]] = {}  # league -> assignments
_assignment_cache_ts: Dict[str, float] = {}  # league -> timestamp


def fetch_referee_assignments(
    league: str, target_date: Optional[str] = None,
) -> Dict[Tuple[str, str], str]:
    """Fetch referee assignments for upcoming fixtures in a league.

    Returns { (home_lower, away_lower): referee_display_name }.
    Cached in-memory for ASSIGNMENT_CACHE_TTL seconds.
    """
    cache_key = f"{league}:{target_date or 'next'}"
    now = time.time()
    if cache_key in _assignment_cache:
        ts = _assignment_cache_ts.get(cache_key, 0)
        if now - ts < ASSIGNMENT_CACHE_TTL:
            return _assignment_cache[cache_key]

    league_id = LEAGUE_TO_API_ID.get(league)
    if league_id is None:
        return {}

    params: dict = {"league": league_id, "season": SEASON}
    if target_date:
        params["date"] = target_date
    else:
        params["next"] = 10

    result = _api_get("/fixtures", params)
    if result is None:
        return _assignment_cache.get(cache_key, {})

    assignments: Dict[Tuple[str, str], str] = {}
    for f in result:
        ref_str = (f.get("fixture") or {}).get("referee")
        home = (f.get("teams", {}).get("home", {}).get("name") or "").lower().strip()
        away = (f.get("teams", {}).get("away", {}).get("name") or "").lower().strip()
        if home and away and ref_str and ref_str.strip():
            assignments[(home, away)] = ref_str.strip()

    _assignment_cache[cache_key] = assignments
    _assignment_cache_ts[cache_key] = now
    return assignments


def _match_event_to_ref(
    home_team: str, away_team: str, league: str,
    assignments: Dict[Tuple[str, str], str],
) -> Optional[str]:
    """Match an odds-API event to a referee assignment. Returns display name or None."""
    h = home_team.lower().strip()
    a = away_team.lower().strip()

    # Exact match first (both orderings — Odds API and API-Football may swap home/away)
    ref = assignments.get((h, a)) or assignments.get((a, h))
    if ref:
        return ref

    # Substring containment fallback (both orderings)
    for (ah, aa), ref_name in assignments.items():
        if (ah in h or h in ah) and (aa in a or a in aa):
            return ref_name
        if (ah in a or a in ah) and (aa in h or h in aa):
            return ref_name

    return None


# ---------------------------------------------------------------------------
# Modifier computation
# ---------------------------------------------------------------------------

_profiles_cache: Optional[Dict[str, RefereeProfile]] = None


def _ensure_profiles_loaded() -> Dict[str, RefereeProfile]:
    global _profiles_cache
    if _profiles_cache is None:
        _profiles_cache = load_profiles()
    return _profiles_cache


def _fuzzy_profile_lookup(
    norm: str, profiles: Dict[str, "RefereeProfile"]
) -> Optional["RefereeProfile"]:
    """Fallback match when exact key misses.

    API-Football often returns abbreviated names like 'D. England' while
    profile keys use full names like 'darren england'.  Strategy:
    1. Extract the last name and optional first initial from the query.
    2. Find profiles whose last name matches AND first initial matches (if present).
    3. Return the match only if exactly one candidate is found (avoids ambiguity).
    4. For multi-word names, also try matching all non-initial words as a set.
    """
    parts = norm.split()
    if len(parts) < 2:
        return None
    initial = parts[0].rstrip(".")[:1]   # e.g. "d" from "d. england"
    last = parts[-1]                       # e.g. "england"

    candidates = []
    for key, prof in profiles.items():
        key_parts = key.split()
        if not key_parts:
            continue
        if key_parts[-1] == last:
            if initial and key_parts[0][:1] == initial:
                candidates.append(prof)
            elif not initial:
                candidates.append(prof)

    if len(candidates) == 1:
        return candidates[0]

    # Multi-word fallback: match by shared non-initial words + initial
    if len(parts) >= 3 and initial:
        query_words = set(p.rstrip(".") for p in parts[1:])  # all words except initial
        candidates = []
        for key, prof in profiles.items():
            key_parts = key.split()
            if len(key_parts) < 2:
                continue
            if key_parts[0][:1] != initial:
                continue
            key_words = set(key_parts[1:])
            # All query words must appear in the profile key (order-independent)
            if query_words <= key_words or key_words <= query_words:
                candidates.append(prof)
        if len(candidates) == 1:
            return candidates[0]

    return None


def compute_referee_modifier(
    referee_name: Optional[str],
    weight: float = DEFAULT_WEIGHT,
) -> RefereeModifier:
    """Compute the card projection modifier for a given referee."""
    if not referee_name:
        return RefereeModifier(source="unavailable")

    profiles = _ensure_profiles_loaded()
    norm = _normalize_ref_name(referee_name)
    if not norm:
        return RefereeModifier(source="unavailable")

    profile = profiles.get(norm)
    if profile is None:
        profile = _fuzzy_profile_lookup(norm, profiles)
    if profile is None:
        return RefereeModifier(
            referee_name=referee_name,
            source="unavailable",
        )

    n = profile.total_matches
    if n < MIN_SAMPLE:
        return RefereeModifier(
            referee_name=profile.name,
            sample_size=n,
            strictness_ratio=profile.strictness_ratio,
            avg_cards_per_match=profile.avg_cards_per_match,
            source="profile",
        )

    confidence = min(1.0, max(0.0,
        (n - MIN_SAMPLE) / (FULL_CONFIDENCE - MIN_SAMPLE)
    ))
    multiplier = 1.0 + (profile.strictness_ratio - 1.0) * confidence * weight

    return RefereeModifier(
        multiplier=multiplier,
        referee_name=profile.name,
        sample_size=n,
        confidence=confidence,
        strictness_ratio=profile.strictness_ratio,
        avg_cards_per_match=profile.avg_cards_per_match,
        source="profile",
    )


def get_referee_modifier(
    home_team: str, away_team: str, league: str,
    target_date: Optional[str] = None,
    weight: Optional[float] = None,
) -> RefereeModifier:
    """Top-level entry point called by rag_cli_v2.py.

    1. Fetch referee assignment for this fixture (dynamic layer)
    2. Look up profile and compute modifier
    3. Always returns a valid RefereeModifier (never raises)
    """
    if weight is None:
        weight = DEFAULT_WEIGHT

    try:
        assignments = fetch_referee_assignments(league, target_date)
        ref_display = _match_event_to_ref(home_team, away_team, league, assignments)
        if not ref_display:
            return RefereeModifier(source="unavailable")
        return compute_referee_modifier(ref_display, weight=weight)
    except Exception:
        return RefereeModifier(source="unavailable")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Referee data: build profiles, lookup assignments, show stats."
    )
    parser.add_argument("--build", action="store_true",
                        help="Build referee profiles from API-Football + team stats.")
    parser.add_argument("--league", type=str, default=None,
                        help="Restrict to a single league (e.g. EPL, LaLiga).")
    parser.add_argument("--lookup", type=str, default=None, metavar="LEAGUE",
                        help="Show upcoming referee assignments for a league.")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date for --lookup (YYYY-MM-DD). Default: today.")
    parser.add_argument("--stats", action="store_true",
                        help="Print referee profile summary statistics.")
    args = parser.parse_args()

    if args.build:
        leagues = [args.league] if args.league else None
        profiles = build_all_referee_profiles(leagues=leagues)
        save_profiles(profiles)
        return

    if args.lookup:
        target = args.date or date.today().isoformat()
        print(f"Referee assignments for {args.lookup} on {target}:")
        assignments = fetch_referee_assignments(args.lookup, target)
        if not assignments:
            print("  No fixtures found or referee data unavailable.")
        else:
            profiles = load_profiles()
            for (h, a), ref_display in assignments.items():
                norm = _normalize_ref_name(ref_display)
                prof = profiles.get(norm) if norm else None
                if prof:
                    print(f"  {h.title()} vs {a.title()}: {ref_display} "
                          f"(strictness {prof.strictness_ratio:.2f}x, "
                          f"{prof.total_matches} matches)")
                else:
                    print(f"  {h.title()} vs {a.title()}: {ref_display} (no profile)")
        return

    if args.stats:
        profiles = load_profiles()
        if not profiles:
            print("No profiles found. Run --build first.")
            return
        print(f"Referee profiles: {len(profiles)} total")
        print()

        sorted_strict = sorted(
            profiles.values(),
            key=lambda p: p.strictness_ratio,
            reverse=True,
        )
        print("Top 10 strictest:")
        for i, p in enumerate(sorted_strict[:10], 1):
            print(f"  {i:2d}. {p.name.title()} ({', '.join(p.leagues)}) - "
                  f"{p.strictness_ratio:.2f}x, {p.total_matches} matches, "
                  f"{p.avg_cards_per_match:.2f} cards/match")
        print()
        print("Top 10 most lenient:")
        for i, p in enumerate(sorted_strict[-10:][::-1], 1):
            print(f"  {i:2d}. {p.name.title()} ({', '.join(p.leagues)}) - "
                  f"{p.strictness_ratio:.2f}x, {p.total_matches} matches, "
                  f"{p.avg_cards_per_match:.2f} cards/match")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
