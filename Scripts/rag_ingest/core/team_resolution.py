"""
core/team_resolution.py — Team name resolution, Chroma profile retrieval,
recent stats, variance computation, and European blending.

Extracted verbatim from rag_cli_v2.py.  Do NOT modify logic here without
updating the original file (until the migration is complete).
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure sibling modules are importable when running from Scripts/rag_ingest/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- weights ---
try:
    from core.weights import SCORING_WEIGHTS, DOMESTIC_LEAGUES, EUROPEAN_COMPETITIONS
except ImportError:
    from weights import SCORING_WEIGHTS, DOMESTIC_LEAGUES, EUROPEAN_COMPETITIONS

# --- chroma backend ---
try:
    from chroma_backend import get_chroma_client, env_first
except ImportError:
    try:
        from Scripts.rag_ingest.chroma_backend import get_chroma_client, env_first
    except ImportError:
        # Absolute last resort — allow import even without chroma (for linting / testing).
        def get_chroma_client(*a, **kw):
            raise RuntimeError("chroma_backend not available")
        def env_first(*a, default=None, **kw):
            import os
            for a_ in a:
                v = os.environ.get(a_)
                if v:
                    return v
            return default

# ---------------------------------------------------------------------------
# Config derived from env (mirrors rag_cli_v2.py)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
CHROMA_DIR = str(ROOT / "Index" / "chroma")
COLLECTION = env_first("CHROMA_COLLECTION", default="football_top5")

# ---------------------------------------------------------------------------
# Caches (module-level singletons, same as rag_cli_v2.py)
# ---------------------------------------------------------------------------

_team_meta_cache: Dict[Tuple[str, str], Dict] = {}
_team_profile_doc_cache: Dict[Tuple[str, str], Dict] = {}
_team_recent_stats_cache: Dict[Tuple, Dict] = {}
_team_recent_var_cache: Dict[Tuple, Dict] = {}
_team_fixture_meta_cache: Dict[Tuple[str, str, str, str, str], Optional[Dict]] = {}
_domestic_league_cache: Dict[str, Optional[str]] = {}
_chroma_client = None
_chroma_collection = None

# ---------------------------------------------------------------------------
# TEAM_ALIAS_MAP — canonical team name lookup
# ---------------------------------------------------------------------------

TEAM_ALIAS_MAP = {
    # Arsenal
    "arsenal": "Arsenal",
    "arsenal fc": "Arsenal",
    "the gunners": "Arsenal",
    # Aston Villa
    "aston villa": "Aston Villa",
    "aston villa fc": "Aston Villa",
    "villa": "Aston Villa",
    # Bournemouth
    "bournemouth": "Bournemouth",
    "afc bournemouth": "Bournemouth",
    "bournemouth afc": "Bournemouth",
    "the cherries": "Bournemouth",
    # Brentford
    "brentford": "Brentford",
    "brentford fc": "Brentford",
    "the bees": "Brentford",
    # Brighton
    "brighton": "Brighton",
    "brighton and hove albion": "Brighton",
    "brighton and hove albion fc": "Brighton",
    "the seagulls": "Brighton",
    # Burnley
    "burnley": "Burnley",
    "burnley fc": "Burnley",
    "the clarets": "Burnley",
    # Chelsea
    "chelsea": "Chelsea",
    "chelsea fc": "Chelsea",
    "the blues": "Chelsea",
    # Crystal Palace
    "crystal palace": "Crystal Palace",
    "crystal palace fc": "Crystal Palace",
    "palace": "Crystal Palace",
    "cpfc": "Crystal Palace",
    # Everton
    "everton": "Everton",
    "everton fc": "Everton",
    "the toffees": "Everton",
    # Fulham
    "fulham": "Fulham",
    "fulham fc": "Fulham",
    # Leeds
    "leeds": "Leeds",
    "leeds united": "Leeds",
    "leeds united fc": "Leeds",
    "lufc": "Leeds",
    # Liverpool
    "liverpool": "Liverpool",
    "liverpool fc": "Liverpool",
    "the reds": "Liverpool",
    # Manchester City
    "manchester city": "Manchester City",
    "manchester city fc": "Manchester City",
    "man city": "Manchester City",
    "city": "Manchester City",
    "mcfc": "Manchester City",
    # Manchester United
    "manchester united": "Manchester United",
    "manchester united fc": "Manchester United",
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "mufc": "Manchester United",
    # Newcastle
    "newcastle": "Newcastle",
    "newcastle united": "Newcastle",
    "newcastle united fc": "Newcastle",
    "nufc": "Newcastle",
    # Nottingham Forest
    "nottingham forest": "Nottingham Forest",
    "nottingham forest fc": "Nottingham Forest",
    "nottm forest": "Nottingham Forest",
    "forest": "Nottingham Forest",
    # Sunderland
    "sunderland": "Sunderland",
    "sunderland afc": "Sunderland",
    "sunderland fc": "Sunderland",
    "safc": "Sunderland",
    # Tottenham
    "tottenham": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "tottenham hotspur fc": "Tottenham",
    "spurs": "Tottenham",
    # West Ham
    "west ham": "West Ham",
    "west ham united": "West Ham",
    "west ham united fc": "West Ham",
    "whufc": "West Ham",
    "hammers": "West Ham",
    # Wolves
    "wolves": "Wolves",
    "wolverhampton": "Wolves",
    "wolverhampton wanderers": "Wolves",
    "wolverhampton wanderers fc": "Wolves",
    "wwfc": "Wolves",
    # --- LaLiga ---
    # Alaves
    "alaves": "Alaves",
    "deportivo alaves": "Alaves",
    "alav\u00e9s": "Alaves",
    # Athletic Club
    "athletic club": "Athletic Club",
    "athletic bilbao": "Athletic Club",
    "bilbao": "Athletic Club",
    # Atletico Madrid
    "atletico madrid": "Atletico Madrid",
    "atl\u00e9tico madrid": "Atletico Madrid",
    "atletico": "Atletico Madrid",
    "atleti": "Atletico Madrid",
    # Barcelona
    "barcelona": "Barcelona",
    "fc barcelona": "Barcelona",
    "barca": "Barcelona",
    "bar\u00e7a": "Barcelona",
    # Celta Vigo
    "celta vigo": "Celta Vigo",
    "celta": "Celta Vigo",
    "rc celta": "Celta Vigo",
    # Elche
    "elche": "Elche",
    "elche cf": "Elche",
    # Espanyol
    "espanyol": "Espanyol",
    "rcd espanyol": "Espanyol",
    # Getafe
    "getafe": "Getafe",
    "getafe cf": "Getafe",
    # Girona
    "girona": "Girona",
    "girona fc": "Girona",
    # Levante
    "levante": "Levante",
    "levante ud": "Levante",
    # Mallorca
    "mallorca": "Mallorca",
    "rcd mallorca": "Mallorca",
    # Osasuna
    "osasuna": "Osasuna",
    "ca osasuna": "Osasuna",
    # Oviedo
    "oviedo": "Oviedo",
    "real oviedo": "Oviedo",
    # Rayo Vallecano
    "rayo vallecano": "Rayo Vallecano",
    "rayo": "Rayo Vallecano",
    # Real Betis
    "real betis": "Real Betis",
    "betis": "Real Betis",
    # Real Madrid
    "real madrid": "Real Madrid",
    "real madrid cf": "Real Madrid",
    "madrid": "Real Madrid",
    # Real Sociedad
    "real sociedad": "Real Sociedad",
    "sociedad": "Real Sociedad",
    "la real": "Real Sociedad",
    # Sevilla
    "sevilla": "Sevilla",
    "sevilla fc": "Sevilla",
    # Valencia
    "valencia": "Valencia",
    "valencia cf": "Valencia",
    # Villarreal
    "villarreal": "Villarreal",
    "villarreal cf": "Villarreal",
    # --- Bundesliga ---
    # 1. FC Heidenheim
    "1. fc heidenheim": "1. FC Heidenheim",
    "heidenheim": "1. FC Heidenheim",
    # 1. FC K\u00f6ln
    "1. fc k\u00f6ln": "1. FC K\u00f6ln",
    "1. fc koln": "1. FC K\u00f6ln",
    "koln": "1. FC K\u00f6ln",
    "k\u00f6ln": "1. FC K\u00f6ln",
    "fc koln": "1. FC K\u00f6ln",
    # 1899 Hoffenheim
    "1899 hoffenheim": "1899 Hoffenheim",
    "hoffenheim": "1899 Hoffenheim",
    "tsg hoffenheim": "1899 Hoffenheim",
    # Bayer Leverkusen
    "bayer leverkusen": "Bayer Leverkusen",
    "leverkusen": "Bayer Leverkusen",
    "bayer 04": "Bayer Leverkusen",
    # Bayern M\u00fcnchen
    "bayern m\u00fcnchen": "Bayern M\u00fcnchen",
    "bayern munchen": "Bayern M\u00fcnchen",
    "bayern munich": "Bayern M\u00fcnchen",
    "bayern": "Bayern M\u00fcnchen",
    "fcb": "Bayern M\u00fcnchen",
    # Borussia Dortmund
    "borussia dortmund": "Borussia Dortmund",
    "dortmund": "Borussia Dortmund",
    "bvb": "Borussia Dortmund",
    # Borussia M\u00f6nchengladbach
    "borussia m\u00f6nchengladbach": "Borussia M\u00f6nchengladbach",
    "borussia monchengladbach": "Borussia M\u00f6nchengladbach",
    "monchengladbach": "Borussia M\u00f6nchengladbach",
    "m\u00f6nchengladbach": "Borussia M\u00f6nchengladbach",
    "gladbach": "Borussia M\u00f6nchengladbach",
    # Eintracht Frankfurt
    "eintracht frankfurt": "Eintracht Frankfurt",
    "frankfurt": "Eintracht Frankfurt",
    "sge": "Eintracht Frankfurt",
    # FC Augsburg
    "fc augsburg": "FC Augsburg",
    "augsburg": "FC Augsburg",
    # FC St. Pauli
    "fc st. pauli": "FC St. Pauli",
    "fc st pauli": "FC St. Pauli",
    "st. pauli": "FC St. Pauli",
    "st pauli": "FC St. Pauli",
    # FSV Mainz 05
    "fsv mainz 05": "FSV Mainz 05",
    "mainz": "FSV Mainz 05",
    "mainz 05": "FSV Mainz 05",
    # Hamburger SV
    "hamburger sv": "Hamburger SV",
    "hamburg": "Hamburger SV",
    "hsv": "Hamburger SV",
    # RB Leipzig
    "rb leipzig": "RB Leipzig",
    "leipzig": "RB Leipzig",
    "rasenballsport leipzig": "RB Leipzig",
    # SC Freiburg
    "sc freiburg": "SC Freiburg",
    "freiburg": "SC Freiburg",
    # Union Berlin
    "union berlin": "Union Berlin",
    "1. fc union berlin": "Union Berlin",
    # VfB Stuttgart
    "vfb stuttgart": "VfB Stuttgart",
    "stuttgart": "VfB Stuttgart",
    # VfL Wolfsburg
    "vfl wolfsburg": "VfL Wolfsburg",
    "wolfsburg": "VfL Wolfsburg",
    # Werder Bremen
    "werder bremen": "Werder Bremen",
    "bremen": "Werder Bremen",
    "sv werder bremen": "Werder Bremen",
    # --- Ligue 1 ---
    # Angers
    "angers": "Angers",
    "angers sco": "Angers",
    # Auxerre
    "auxerre": "Auxerre",
    "aj auxerre": "Auxerre",
    # Le Havre
    "le havre": "Le Havre",
    "le havre ac": "Le Havre",
    # Lens
    "lens": "Lens",
    "rc lens": "Lens",
    # Lille
    "lille": "Lille",
    "losc lille": "Lille",
    "losc": "Lille",
    # Lorient
    "lorient": "Lorient",
    "fc lorient": "Lorient",
    # Lyon
    "lyon": "Lyon",
    "olympique lyonnais": "Lyon",
    "ol": "Lyon",
    # Marseille
    "marseille": "Marseille",
    "olympique de marseille": "Marseille",
    "olympique marseille": "Marseille",
    "om": "Marseille",
    # Metz
    "metz": "Metz",
    "fc metz": "Metz",
    # Monaco
    "monaco": "Monaco",
    "as monaco": "Monaco",
    # Nantes
    "nantes": "Nantes",
    "fc nantes": "Nantes",
    # Nice
    "nice": "Nice",
    "ogc nice": "Nice",
    # Paris FC
    "paris fc": "Paris FC",
    # Paris Saint Germain
    "paris saint germain": "Paris Saint Germain",
    "paris saint-germain": "Paris Saint Germain",
    "paris sg": "Paris Saint Germain",
    "psg": "Paris Saint Germain",
    # Rennes
    "rennes": "Rennes",
    "stade rennais": "Rennes",
    # Stade Brestois 29
    "stade brestois 29": "Stade Brestois 29",
    "stade brestois": "Stade Brestois 29",
    "brest": "Stade Brestois 29",
    "brestois": "Stade Brestois 29",
    # Strasbourg
    "strasbourg": "Strasbourg",
    "rc strasbourg": "Strasbourg",
    "rc strasbourg alsace": "Strasbourg",
    # Toulouse
    "toulouse": "Toulouse",
    "toulouse fc": "Toulouse",
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _numeric(value) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", (s or "").lower()).strip()


def parse_season_rank(season_value: Optional[str]) -> int:
    if not season_value:
        return -1
    s = str(season_value)
    m = re.search(r"(20\d{2})", s)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except ValueError:
        return -1


def _weighted_avg(values: List[Optional[float]], alpha: float = 0.85) -> Optional[float]:
    """Exponentially-weighted average.  Index 0 = most recent (highest weight)."""
    cleaned = [(i, x) for i, x in enumerate(values) if x is not None]
    if not cleaned:
        return None
    weights = [alpha ** i for i, _ in cleaned]
    total_w = sum(weights)
    return sum(w * x for w, (_, x) in zip(weights, cleaned)) / total_w


def _linear_slope(values: List[Optional[float]]) -> Optional[float]:
    """Compute linear slope of values (index 0 = most recent).
    Positive slope = improving (recent values higher than older values).
    Returns slope per match, or None if fewer than 4 valid points."""
    cleaned = [(i, x) for i, x in enumerate(values) if x is not None]
    if len(cleaned) < 4:
        return None
    n = len(cleaned)
    # Reverse so older = lower index (conventional regression direction)
    xs = [float(n - 1 - i) for i, _ in cleaned]
    ys = [x for _, x in cleaned]
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# Chroma helpers
# ---------------------------------------------------------------------------

def get_collection_handle(create_if_missing: bool = True):
    global _chroma_client, _chroma_collection
    if _chroma_client is None:
        _chroma_client = get_chroma_client(CHROMA_DIR)
    if _chroma_collection is not None:
        return _chroma_collection
    if create_if_missing:
        _chroma_collection = _chroma_client.get_or_create_collection(COLLECTION)
    else:
        _chroma_collection = _chroma_client.get_collection(COLLECTION)
    return _chroma_collection


def build_where(league: Optional[str], extra_filters: Optional[List[Dict]] = None) -> Dict:
    clauses: List[Dict] = []
    if league:
        clauses.append({"league": league})
    if extra_filters:
        clauses.extend(extra_filters)
    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ---------------------------------------------------------------------------
# Team name resolution
# ---------------------------------------------------------------------------

def canonical_team_name(name: str) -> str:
    n = normalize_name(name)
    if not n:
        return name
    # Exact alias match
    if n in TEAM_ALIAS_MAP:
        return TEAM_ALIAS_MAP[n]
    # Fuzzy fallback: find closest alias key (cutoff 0.65 to catch e.g. "Brighton Hove Albion" -> "Brighton and Hove Albion")
    matches = difflib.get_close_matches(n, TEAM_ALIAS_MAP.keys(), n=1, cutoff=0.65)
    if matches:
        return TEAM_ALIAS_MAP[matches[0]]
    return name


def team_name_aliases(name: str) -> Set[str]:
    n = normalize_name(name)
    aliases = {n}
    # drop common suffix tokens for matching in casual queries
    for suffix in [" united", " hotspur", " fc", " afc", " cf"]:
        if n.endswith(suffix):
            aliases.add(n[: -len(suffix)].strip())
    # add known dictionary aliases
    canon = canonical_team_name(name)
    aliases.add(normalize_name(canon))
    for k, v in TEAM_ALIAS_MAP.items():
        if normalize_name(v) == normalize_name(canon):
            aliases.add(k)
    return {x for x in aliases if x}


def _kb_team_variants(team_name: str) -> List[str]:
    """Generate team name variants for KB lookup: exact -> alias -> fuzzy."""
    canonical = canonical_team_name(team_name)
    variants = list({team_name, canonical, team_name.lower(), canonical.lower(), team_name.title(), canonical.title()})
    # Add token-stripped variants (e.g., "AFC Bournemouth" -> "Bournemouth")
    for suffix in [" united", " hotspur", " fc", " afc", " cf", " city"]:
        for base in [team_name.lower(), canonical.lower()]:
            if base.endswith(suffix):
                stripped = base[: -len(suffix)].strip().title()
                if stripped and stripped not in variants:
                    variants.append(stripped)
    for prefix in ["afc ", "fc "]:
        for base in [team_name.lower(), canonical.lower()]:
            if base.startswith(prefix):
                stripped = base[len(prefix):].strip().title()
                if stripped and stripped not in variants:
                    variants.append(stripped)
    return variants


# ---------------------------------------------------------------------------
# Chroma profile retrieval
# ---------------------------------------------------------------------------

def get_team_profile_doc(team_name: str, league: str) -> Dict:
    cache_key = (league, canonical_team_name(team_name).lower())
    cached = _team_profile_doc_cache.get(cache_key)
    if cached is not None:
        return cached

    col = get_collection_handle(create_if_missing=True)
    variants = _kb_team_variants(team_name)
    rows: List[Dict] = []
    for v in variants:
        where = build_where(league, extra_filters=[{"doc_type": "team_profile"}, {"team": v}])
        res = col.get(where=where, include=["documents", "metadatas"], limit=6)
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        for d, m in zip(docs, metas):
            if not m:
                continue
            rows.append({"text": d or "", "meta": m})
    if not rows:
        _team_profile_doc_cache[cache_key] = {}
        return {}

    rows.sort(key=lambda x: parse_season_rank((x.get("meta") or {}).get("season")), reverse=True)
    best = rows[0]
    _team_profile_doc_cache[cache_key] = best
    return best


def get_team_profile_meta(team_name: str, league: str) -> Dict:
    cache_key = (league, canonical_team_name(team_name).lower())
    cached = _team_meta_cache.get(cache_key)
    if cached is not None:
        return cached
    doc = get_team_profile_doc(team_name, league)
    meta = (doc.get("meta") or {}) if doc else {}
    _team_meta_cache[cache_key] = meta
    return meta


# ---------------------------------------------------------------------------
# European competition: domestic-anchored projection helpers
# ---------------------------------------------------------------------------

def resolve_domestic_league(team_name: str) -> Optional[str]:
    """Find which domestic league a team belongs to by searching Chroma profiles."""
    canon = canonical_team_name(team_name).lower()
    if canon in _domestic_league_cache:
        return _domestic_league_cache[canon]
    for lg in sorted(DOMESTIC_LEAGUES):
        meta = get_team_profile_meta(team_name, lg)
        if meta:
            _domestic_league_cache[canon] = lg
            return lg
    _domestic_league_cache[canon] = None
    return None


def get_blended_profile_meta(team_name: str, competition: str) -> Dict:
    """For European competitions, blend domestic (80%) + European (20%) profiles.
    For domestic leagues, returns the standard profile unchanged."""
    if competition not in EUROPEAN_COMPETITIONS:
        return get_team_profile_meta(team_name, competition)

    ew = SCORING_WEIGHTS["european"]
    euro_meta = get_team_profile_meta(team_name, competition)
    domestic_lg = resolve_domestic_league(team_name)

    if not domestic_lg:
        return euro_meta  # Non-top-5 team -- European data only

    domestic_meta = get_team_profile_meta(team_name, domestic_lg)
    if not domestic_meta:
        return euro_meta
    if not euro_meta:
        return domestic_meta

    # Need enough European fixtures before blending is meaningful
    euro_n = _numeric(euro_meta.get("matches_played")) or 0
    if euro_n < ew["min_euro_fixtures"]:
        return domestic_meta  # Not enough European data yet -- domestic only

    # Weighted blend of all numeric fields
    blended = dict(domestic_meta)  # start with domestic as base
    d_w, e_w = ew["domestic_weight"], ew["euro_weight"]
    for key, d_val in domestic_meta.items():
        e_val = euro_meta.get(key)
        if isinstance(d_val, (int, float)) and isinstance(e_val, (int, float)):
            blended[key] = d_w * d_val + e_w * e_val
    blended["_domestic_league"] = domestic_lg
    blended["_euro_competition"] = competition
    blended["_blend_mode"] = "domestic_anchored"
    return blended


def get_blended_recent_stats(team_name: str, competition: str, last_n: int = 6,
                             venue: Optional[str] = None,
                             target_date: Optional[str] = None) -> Dict:
    """For European competitions, blend domestic + European recent stats."""
    if competition not in EUROPEAN_COMPETITIONS:
        return get_team_recent_stats(team_name, competition, last_n, venue=venue, target_date=target_date)

    ew = SCORING_WEIGHTS["european"]
    euro_stats = get_team_recent_stats(team_name, competition, last_n, venue=venue, target_date=target_date)
    domestic_lg = resolve_domestic_league(team_name)

    if not domestic_lg:
        return euro_stats  # Non-top-5 team

    domestic_stats = get_team_recent_stats(team_name, domestic_lg, last_n, venue=venue, target_date=target_date)
    if not domestic_stats or domestic_stats.get("n", 0) == 0:
        return euro_stats
    if not euro_stats or euro_stats.get("n", 0) == 0:
        return domestic_stats

    # Weighted blend
    blended: Dict = {}
    d_w, e_w = ew["domestic_weight"], ew["euro_weight"]
    all_keys = set(list(domestic_stats.keys()) + list(euro_stats.keys()))
    for key in all_keys:
        d_val = domestic_stats.get(key)
        e_val = euro_stats.get(key)
        if isinstance(d_val, (int, float)) and isinstance(e_val, (int, float)):
            blended[key] = d_w * d_val + e_w * e_val
        elif d_val is not None:
            blended[key] = d_val
        else:
            blended[key] = e_val
    blended["n"] = domestic_stats.get("n", 0) + euro_stats.get("n", 0)
    return blended


# ---------------------------------------------------------------------------
# Thin dispatch wrappers (auto-blend for European competitions)
# ---------------------------------------------------------------------------

def _profile_meta(team: str, league: str) -> Dict:
    """Profile metadata -- auto-blends for European competitions."""
    if league in EUROPEAN_COMPETITIONS:
        return get_blended_profile_meta(team, league)
    return get_team_profile_meta(team, league)


def _recent_stats(team: str, league: str, last_n: int = 6,
                  venue: Optional[str] = None,
                  target_date: Optional[str] = None) -> Dict:
    """Recent stats -- auto-blends for European competitions."""
    if league in EUROPEAN_COMPETITIONS:
        return get_blended_recent_stats(team, league, last_n, venue=venue, target_date=target_date)
    return get_team_recent_stats(team, league, last_n, venue=venue, target_date=target_date)


# ---------------------------------------------------------------------------
# Recent fixture rows and stats
# ---------------------------------------------------------------------------

def get_recent_team_fixture_rows(team_name: str, league: str, limit: int = 8,
                                  season: Optional[str] = None,
                                  target_date: Optional[str] = None) -> List[Dict]:
    col = get_collection_handle(create_if_missing=True)
    variants = _kb_team_variants(team_name)

    # Derive current season from team profile to avoid cross-season contamination.
    # Try profile season first, then check if fixtures actually exist for that season.
    target_season = season
    if target_season is None:
        profile = get_team_profile_doc(team_name, league)
        target_season = (profile.get("meta") or {}).get("season")

    def _fetch(season_filter: Optional[str]) -> Dict[str, Dict]:
        dedup: Dict[str, Dict] = {}
        for v in variants:
            filters: List[Dict] = [{"doc_type": "team_fixture"}, {"team": v}]
            if season_filter:
                filters.append({"season": season_filter})
            where = build_where(league, extra_filters=filters)
            res = col.get(where=where, include=["metadatas", "documents"], limit=50)
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            for d, m in zip(docs, metas):
                if not m:
                    continue
                fixture = str(m.get("fixture") or "")
                fixture_date = str(m.get("fixture_date") or "")
                key = f"{fixture_date}|{fixture}"
                dedup[key] = {"meta": m, "text": d or ""}
        return dedup

    dedup = _fetch(target_season)

    # Fallback: if profile season yields no fixtures (season mismatch), retry without
    if not dedup and target_season:
        dedup = _fetch(None)

    rows = list(dedup.values())
    if target_date:
        cutoff = str(target_date)[:10]
        rows = [
            row for row in rows
            if str((row.get("meta") or {}).get("fixture_date") or "")[:10] <= cutoff
        ]
    rows.sort(key=lambda x: ((x.get("meta") or {}).get("fixture_date") or ""), reverse=True)
    return rows[:limit]


def _get_team_fixture_meta(team_name: str, league: str, fixture: str,
                           fixture_date: str = "", season: Optional[str] = None) -> Optional[Dict]:
    """Fetch one team_fixture metadata row for a specific team+fixture."""
    cache_key = (
        league,
        canonical_team_name(team_name).lower(),
        str(fixture or ""),
        str(fixture_date or ""),
        str(season or ""),
    )
    if cache_key in _team_fixture_meta_cache:
        return _team_fixture_meta_cache[cache_key]

    col = get_collection_handle(create_if_missing=True)
    variants = _kb_team_variants(team_name)
    for v in variants:
        filters: List[Dict] = [{"doc_type": "team_fixture"}, {"team": v}, {"fixture": fixture}]
        if fixture_date:
            filters.append({"fixture_date": fixture_date})
        if season:
            filters.append({"season": season})
        where = build_where(league, extra_filters=filters)
        res = col.get(where=where, include=["metadatas"], limit=4)
        metas = res.get("metadatas") or []
        if metas:
            meta = metas[0] or None
            _team_fixture_meta_cache[cache_key] = meta
            return meta

    _team_fixture_meta_cache[cache_key] = None
    return None


def get_team_recent_stats(team_name: str, league: str, last_n: int = 6,
                          venue: Optional[str] = None,
                          target_date: Optional[str] = None) -> Dict:
    cache_key = (
        league,
        canonical_team_name(team_name).lower(),
        int(last_n),
        venue or "all",
        str(target_date or "")[:10],
    )
    cached = _team_recent_stats_cache.get(cache_key)
    if cached is not None:
        return cached

    # Fetch more rows when venue-filtering so we still get enough after filtering
    fetch_limit = max(last_n * 3, 12) if venue else max(last_n, 4)
    rows = get_recent_team_fixture_rows(team_name, league, limit=fetch_limit, target_date=target_date)
    if venue:
        rows = [r for r in rows
                if (r.get("meta") or {}).get("home_away", "").lower() == venue.lower()]
    rows = rows[:last_n]
    metas = [(r.get("meta") or {}) for r in rows]

    alpha = SCORING_WEIGHTS.get("recency", {}).get("alpha", 0.85)

    # Extract raw series for both weighted avg and trend computation
    corners_for_vals = [_numeric(m.get("corners_for")) for m in metas]
    sot_for_vals = [_numeric(m.get("sot_for")) for m in metas]
    cards_vals = [_numeric(m.get("cards_per_90_team")) for m in metas]
    goals_vals = [_numeric(m.get("xg_for")) for m in metas]
    cards_induced_vals = []
    yellows_induced_vals = []
    reds_induced_vals = []
    for m in metas:
        opp = str(m.get("opponent") or "")
        fixture = str(m.get("fixture") or "")
        fixture_date = str(m.get("fixture_date") or "")
        season = str(m.get("season") or "")
        if not opp or not fixture:
            cards_induced_vals.append(None)
            yellows_induced_vals.append(None)
            reds_induced_vals.append(None)
            continue
        opp_meta = _get_team_fixture_meta(opp, league, fixture, fixture_date, season=season)
        cards_induced_vals.append(_numeric((opp_meta or {}).get("cards_per_90_team")))
        yellows_induced_vals.append(_numeric((opp_meta or {}).get("yellow_cards")))
        reds_induced_vals.append(_numeric((opp_meta or {}).get("red_cards")))

    # Yellow/red card series (use metadata fields added by normalize rebuild;
    # gracefully degrade to None if not yet available)
    yellows_vals = [_numeric(m.get("yellow_cards")) for m in metas]
    reds_vals = [_numeric(m.get("red_cards")) for m in metas]
    fouls_vals = [_numeric(m.get("fouls_per_90_team")) for m in metas]

    # cards_per_foul per fixture
    cpf_vals = []
    for c, f in zip(cards_vals, fouls_vals):
        if c is not None and f is not None and f > 0:
            cpf_vals.append(c / f)
        else:
            cpf_vals.append(None)

    # fouls_drawn: opponent's fouls committed against this team
    fouls_drawn_vals = []
    for m in metas:
        opp = str(m.get("opponent") or "")
        fixture = str(m.get("fixture") or "")
        fixture_date = str(m.get("fixture_date") or "")
        season_str = str(m.get("season") or "")
        if not opp or not fixture:
            fouls_drawn_vals.append(None)
            continue
        opp_meta = _get_team_fixture_meta(opp, league, fixture, fixture_date, season=season_str)
        fouls_drawn_vals.append(_numeric((opp_meta or {}).get("fouls_per_90_team")))

    stats = {
        "n": len(metas),
        "corners_for_avg": _weighted_avg(corners_for_vals, alpha),
        "corners_against_avg": _weighted_avg([_numeric(m.get("corners_against")) for m in metas], alpha),
        "shots_for_avg": _weighted_avg([_numeric(m.get("shots_for")) for m in metas], alpha),
        "sot_for_avg": _weighted_avg(sot_for_vals, alpha),
        "shots_against_avg": _weighted_avg([_numeric(m.get("shots_against")) for m in metas], alpha),
        "sot_against_avg": _weighted_avg([_numeric(m.get("sot_against")) for m in metas], alpha),
        "cards_avg": _weighted_avg(cards_vals, alpha),
        "cards_induced_avg": _weighted_avg(cards_induced_vals, alpha),
        "aggression_avg": _weighted_avg([_numeric(m.get("aggression_index_norm")) for m in metas], alpha),
        "control_avg": _weighted_avg([_numeric(m.get("control_index")) for m in metas], alpha),
        "form_avg": _weighted_avg([_numeric(m.get("form_index_team")) for m in metas], alpha),
        "fouls_avg": _weighted_avg(fouls_vals, alpha),
        "possession_avg": _weighted_avg([_numeric(m.get("possession")) for m in metas], alpha),
        "xg_for_avg": _weighted_avg(goals_vals, alpha),
        # Yellow/red card breakdown
        "yellows_avg": _weighted_avg(yellows_vals, alpha),
        "reds_avg": _weighted_avg(reds_vals, alpha),
        "yellows_induced_avg": _weighted_avg(yellows_induced_vals, alpha),
        "reds_induced_avg": _weighted_avg(reds_induced_vals, alpha),
        "cards_per_foul_avg": _weighted_avg(cpf_vals, alpha),
        "fouls_drawn_avg": _weighted_avg(fouls_drawn_vals, alpha),
        # Trend slopes (positive = improving, units per match)
        "corners_for_slope": _linear_slope(corners_for_vals),
        "sot_for_slope": _linear_slope(sot_for_vals),
        "cards_slope": _linear_slope(cards_vals),
        "goals_slope": _linear_slope(goals_vals),
    }
    _team_recent_stats_cache[cache_key] = stats
    return stats


def get_team_recent_variance(team_name: str, league: str, last_n: int = 8,
                             target_date: Optional[str] = None) -> Dict:
    """Compute per-stat variance from recent fixtures for distribution modeling."""
    cache_key = (
        league,
        canonical_team_name(team_name).lower(),
        int(last_n),
        str(target_date or "")[:10],
    )
    cached = _team_recent_var_cache.get(cache_key)
    if cached is not None:
        return cached

    rows = get_recent_team_fixture_rows(team_name, league, limit=last_n, target_date=target_date)
    metas = [(r.get("meta") or {}) for r in rows]

    def _var(values):
        clean = [v for v in values if v is not None]
        if len(clean) < 3:
            return None
        mean = sum(clean) / len(clean)
        return sum((x - mean) ** 2 for x in clean) / (len(clean) - 1)

    result = {
        "n": len(metas),
        "corners_for_var": _var([_numeric(m.get("corners_for")) for m in metas]),
        "sot_for_var": _var([_numeric(m.get("sot_for")) for m in metas]),
        "goals_var": _var([_numeric(m.get("xg_for")) for m in metas]),
        "cards_var": _var([_numeric(m.get("cards_per_90_team")) for m in metas]),
        "yellows_var": _var([_numeric(m.get("yellow_cards")) for m in metas]),
        "reds_var": _var([_numeric(m.get("red_cards")) for m in metas]),
    }
    _team_recent_var_cache[cache_key] = result
    return result


def get_blended_variance(team_name: str, league: str, stat_key: str,
                         fixture_date: Optional[str] = None) -> Optional[float]:
    """Bayesian blend of season variance (prior) with recent variance (update).
    Returns blended variance for distribution selection (Poisson vs NegBin)."""
    season_var_key = {"goals_var": "goals_var", "corners_for_var": "corners_var",
                      "cards_var": "cards_var", "sot_for_var": "sot_var"}.get(stat_key)
    recent_dict = get_team_recent_variance(team_name, league, target_date=fixture_date)
    recent_var = recent_dict.get(stat_key)
    season_var = None
    if season_var_key:
        meta = _profile_meta(team_name, league)
        season_var = _numeric(meta.get(season_var_key)) if meta else None
    if season_var is not None and recent_var is not None:
        return 0.7 * season_var + 0.3 * recent_var
    return season_var if season_var is not None else recent_var
