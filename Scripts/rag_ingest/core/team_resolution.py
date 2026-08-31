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
from typing import Any, Dict, List, Optional, Set, Tuple

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
_team_profile_docs_cache: Dict[Tuple[str, str], List[Dict]] = {}
_team_fixture_rows_cache: Dict[Tuple[str, str], List[Dict]] = {}
_team_profile_context_cache: Dict[Tuple[str, str, str], Tuple[Dict, Dict]] = {}
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

def get_team_profile_docs(team_name: str, league: str) -> List[Dict]:
    """Return one profile document per available season, newest first.

    Team-profile documents are snapshots that are overwritten as a season
    progresses.  Callers that need a historical prediction must therefore use
    fixture rows for the active season (see ``get_team_profile_context``), but
    this helper still exposes earlier completed seasons as empirical priors.
    """
    cache_key = (league, canonical_team_name(team_name).lower())
    cached = _team_profile_docs_cache.get(cache_key)
    if cached is not None:
        return cached

    col = get_collection_handle(create_if_missing=True)
    variants = _kb_team_variants(team_name)
    rows_by_season: Dict[str, Dict] = {}
    for v in variants:
        where = build_where(league, extra_filters=[{"doc_type": "team_profile"}, {"team": v}])
        res = col.get(where=where, include=["documents", "metadatas"], limit=24)
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        for d, m in zip(docs, metas):
            if not m:
                continue
            season = str(m.get("season") or "")
            if not season:
                continue
            # Duplicate alias queries can return the same profile.  The
            # profile ID is deterministic per league/season/team, so one row
            # per season is the useful representation here.
            rows_by_season.setdefault(season, {"text": d or "", "meta": m})

    rows = sorted(
        rows_by_season.values(),
        key=lambda x: parse_season_rank((x.get("meta") or {}).get("season")),
        reverse=True,
    )
    _team_profile_docs_cache[cache_key] = rows
    _team_profile_doc_cache[cache_key] = rows[0] if rows else {}
    return rows


def get_team_profile_doc(team_name: str, league: str) -> Dict:
    """Return the newest stored team-profile document for compatibility."""
    cache_key = (league, canonical_team_name(team_name).lower())
    cached = _team_profile_doc_cache.get(cache_key)
    if cached is not None:
        return cached
    docs = get_team_profile_docs(team_name, league)
    return docs[0] if docs else {}


def get_team_profile_meta(team_name: str, league: str) -> Dict:
    cache_key = (league, canonical_team_name(team_name).lower())
    cached = _team_meta_cache.get(cache_key)
    if cached is not None:
        return cached
    doc = get_team_profile_doc(team_name, league)
    meta = (doc.get("meta") or {}) if doc else {}
    _team_meta_cache[cache_key] = meta
    return meta


# A current-season rate should not outweigh a completed season after one or
# two fixtures.  This is deliberately a transparent pseudo-sample size rather
# than a hidden model coefficient; calibration work will tune it later.
EARLY_SEASON_PRIOR_STRENGTH = 8.0


def _season_rank_for_target_date(target_date: Optional[str]) -> Optional[int]:
    """Return the domestic season's start year for an ISO-like fixture date."""
    if not target_date:
        return None
    match = re.search(r"(20\d{2})-(\d{2})", str(target_date))
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    return year if month >= 7 else year - 1


def _season_label(rank: Optional[int]) -> Optional[str]:
    if rank is None or rank < 0:
        return None
    return f"{rank}/{str(rank + 1)[-2:]}"


def _get_all_team_fixture_rows(team_name: str, league: str) -> List[Dict]:
    """Return immutable team-fixture rows once, ready for as-of filtering."""
    cache_key = (league, canonical_team_name(team_name).lower())
    cached = _team_fixture_rows_cache.get(cache_key)
    if cached is not None:
        return cached

    col = get_collection_handle(create_if_missing=True)
    dedup: Dict[str, Dict] = {}
    for variant in _kb_team_variants(team_name):
        where = build_where(league, extra_filters=[{"doc_type": "team_fixture"}, {"team": variant}])
        result = col.get(where=where, include=["documents", "metadatas"], limit=500)
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for document, meta in zip(documents, metadatas):
            if not meta:
                continue
            fixture = str(meta.get("fixture") or "")
            fixture_date = str(meta.get("fixture_date") or "")
            season = str(meta.get("season") or "")
            # Some source rows have no fixture label.  Keep those isolated so
            # they cannot accidentally overwrite a real fixture.
            key = f"{season}|{fixture_date}|{fixture or id(meta)}"
            dedup[key] = {"meta": meta, "text": document or ""}

    rows = list(dedup.values())
    rows.sort(key=lambda row: str((row.get("meta") or {}).get("fixture_date") or ""))
    _team_fixture_rows_cache[cache_key] = rows
    return rows


def _mean_numeric(values: List[Optional[float]]) -> Optional[float]:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def _sample_variance(values: List[Optional[float]]) -> Optional[float]:
    cleaned = [value for value in values if value is not None]
    if len(cleaned) < 3:
        return None
    mean = sum(cleaned) / len(cleaned)
    return sum((value - mean) ** 2 for value in cleaned) / (len(cleaned) - 1)


def _fixture_goals_for_and_against(meta: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Recover a team's score from the normalized final-score label."""
    score = str(meta.get("final_score") or "")
    match = re.search(r"(\d+)\s*[-:]\s*(\d+)", score)
    if not match:
        return None, None
    home_goals, away_goals = float(match.group(1)), float(match.group(2))
    if str(meta.get("home_away") or "").lower() == "away":
        return away_goals, home_goals
    return home_goals, away_goals


def _profile_from_fixture_rows(
    team_name: str,
    league: str,
    season: Optional[str],
    rows: List[Dict],
) -> Dict:
    """Build the rate fields used by projections from immutable fixture rows.

    This is intentionally a compact projection profile, not a replacement for
    ingestion-time feature engineering.  Its job is temporal safety: it gives
    a dated prediction only information from fixtures that had already ended.
    """
    metas = [(row.get("meta") or {}) for row in rows]
    if not metas:
        return {}

    def avg(key: str) -> Optional[float]:
        return _mean_numeric([_numeric(meta.get(key)) for meta in metas])

    def venue_avg(key: str, venue: str) -> Optional[float]:
        return _mean_numeric([
            _numeric(meta.get(key))
            for meta in metas
            if str(meta.get("home_away") or "").lower() == venue
        ])

    goals_for: List[Optional[float]] = []
    goals_against: List[Optional[float]] = []
    opponent_cards: List[Optional[float]] = []
    opponent_cards_home: List[Optional[float]] = []
    opponent_cards_away: List[Optional[float]] = []
    for meta in metas:
        goals_scored, goals_conceded = _fixture_goals_for_and_against(meta)
        goals_for.append(goals_scored)
        goals_against.append(goals_conceded)

        opponent = str(meta.get("opponent") or "")
        fixture = str(meta.get("fixture") or "")
        fixture_date = str(meta.get("fixture_date") or "")
        if not opponent or not fixture:
            opponent_cards.append(None)
            continue
        opponent_meta = _get_team_fixture_meta(
            opponent, league, fixture, fixture_date, season=str(season or "") or None,
        )
        cards = _numeric((opponent_meta or {}).get("cards_per_90_team"))
        if cards is None:
            cards = _numeric((opponent_meta or {}).get("cards_total"))
        opponent_cards.append(cards)
        if str(meta.get("home_away") or "").lower() == "home":
            opponent_cards_home.append(cards)
        elif str(meta.get("home_away") or "").lower() == "away":
            opponent_cards_away.append(cards)

    cards_per_90 = avg("cards_per_90_team")
    if cards_per_90 is None:
        cards_per_90 = avg("cards_total")

    profile = {
        "entity_type": "team",
        "doc_type": "team_profile",
        "league": league,
        "season": season,
        "team": canonical_team_name(team_name),
        "matches_played": len(metas),
        "goals_for_pm": _mean_numeric(goals_for),
        "goals_against_pm": _mean_numeric(goals_against),
        "expected_goals": avg("xg_for"),
        "corners_pm": avg("corners_for"),
        "corners_against_pm": avg("corners_against"),
        "shots_for_pm": avg("shots_for"),
        "shots_against_pm": avg("shots_against"),
        "sot_for_pm": avg("sot_for"),
        "sot_against_pm": avg("sot_against"),
        "cards_per_90_team": cards_per_90,
        "cards_pm": avg("cards_total"),
        "yellows_pm": avg("yellow_cards"),
        "reds_pm": avg("red_cards"),
        "fouls_per_90_team": avg("fouls_per_90_team"),
        "fouls_pm": avg("fouls_committed"),
        "possession": avg("possession"),
        "control_index": avg("control_index"),
        "aggression_index_norm": avg("aggression_index_norm"),
        "form_index_team": avg("form_index_team"),
        "opp_cards_induced_pm": _mean_numeric(opponent_cards),
        "goals_home_pm": _mean_numeric([
            _fixture_goals_for_and_against(meta)[0]
            for meta in metas if str(meta.get("home_away") or "").lower() == "home"
        ]),
        "goals_away_pm": _mean_numeric([
            _fixture_goals_for_and_against(meta)[0]
            for meta in metas if str(meta.get("home_away") or "").lower() == "away"
        ]),
        "xg_home_pm": venue_avg("xg_for", "home"),
        "xg_away_pm": venue_avg("xg_for", "away"),
        "corners_home_pm": venue_avg("corners_for", "home"),
        "corners_away_pm": venue_avg("corners_for", "away"),
        "corners_against_home_pm": venue_avg("corners_against", "home"),
        "corners_against_away_pm": venue_avg("corners_against", "away"),
        "sot_home_pm": venue_avg("sot_for", "home"),
        "sot_away_pm": venue_avg("sot_for", "away"),
        "sot_against_home_pm": venue_avg("sot_against", "home"),
        "sot_against_away_pm": venue_avg("sot_against", "away"),
        "cards_home_pm": venue_avg("cards_per_90_team", "home"),
        "cards_away_pm": venue_avg("cards_per_90_team", "away"),
        "cards_induced_home_pm": _mean_numeric(opponent_cards_home),
        "cards_induced_away_pm": _mean_numeric(opponent_cards_away),
        "fouls_home_pm": venue_avg("fouls_committed", "home"),
        "fouls_away_pm": venue_avg("fouls_committed", "away"),
        "goals_var": _sample_variance(goals_for),
        "corners_var": _sample_variance([_numeric(meta.get("corners_for")) for meta in metas]),
        "sot_var": _sample_variance([_numeric(meta.get("sot_for")) for meta in metas]),
        "cards_var": _sample_variance([_numeric(meta.get("cards_per_90_team")) for meta in metas]),
        "_profile_source": "fixture_rows",
    }
    return profile


def _profile_for_rank_from_docs(team_name: str, league: str, season_rank: Optional[int]) -> Dict:
    if season_rank is None:
        return {}
    for document in get_team_profile_docs(team_name, league):
        meta = dict(document.get("meta") or {})
        if parse_season_rank(meta.get("season")) == season_rank:
            return meta
    return {}


def _blend_sparse_current_profile(current: Dict, prior: Dict) -> Tuple[Dict, float, str]:
    """Shrink sparse current-season rates toward a completed prior profile."""
    current_n = _numeric((current or {}).get("matches_played")) or 0.0
    prior_n = _numeric((prior or {}).get("matches_played")) or 0.0
    if not prior:
        return dict(current or {}), 0.0, "current_only"
    if current_n <= 0:
        result = dict(prior)
        result["matches_played"] = 0
        return result, 1.0, "prior_only"
    if current_n >= EARLY_SEASON_PRIOR_STRENGTH:
        return dict(current), 0.0, "current_only"

    current_weight = current_n / (current_n + EARLY_SEASON_PRIOR_STRENGTH)
    prior_weight = 1.0 - current_weight
    result: Dict[str, Any] = dict(prior)
    result.update(current)
    excluded_numeric_fields = {"matches_played"}
    for key in set(current) | set(prior):
        current_value = current.get(key)
        prior_value = prior.get(key)
        if key in excluded_numeric_fields:
            continue
        if (
            isinstance(current_value, (int, float)) and not isinstance(current_value, bool)
            and isinstance(prior_value, (int, float)) and not isinstance(prior_value, bool)
        ):
            result[key] = current_weight * float(current_value) + prior_weight * float(prior_value)
        elif current_value is None and prior_value is not None:
            result[key] = prior_value
    result["matches_played"] = int(current_n)
    return result, prior_weight, "current_plus_prior"


def get_team_profile_context(
    team_name: str,
    league: str,
    target_date: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """Return the profile used for a prediction plus its auditable data window.

    With ``target_date`` this reconstructs the active season from immutable
    fixtures strictly before that date.  The completed prior season is then a
    shrinkage prior for a small active-season sample.  Without a date, the
    newest stored profile is retained for legacy callers but still receives the
    same early-season shrinkage where possible.
    """
    cutoff = str(target_date or "")[:10]
    cache_key = (league, canonical_team_name(team_name).lower(), cutoff)
    cached = _team_profile_context_cache.get(cache_key)
    if cached is not None:
        return cached

    documents = get_team_profile_docs(team_name, league)
    if not target_date:
        current = dict((documents[0].get("meta") or {})) if documents else {}
        current_rank = parse_season_rank(current.get("season")) if current else None
        prior = _profile_for_rank_from_docs(team_name, league, (current_rank - 1) if current_rank else None)
        effective, prior_weight, mode = _blend_sparse_current_profile(current, prior)
        current_n = int(_numeric(current.get("matches_played")) or 0)
        prior_n = int(_numeric(prior.get("matches_played")) or 0)
        audit = {
            "team": canonical_team_name(team_name),
            "as_of": None,
            "temporal_status": "latest_profile_snapshot_unbounded",
            "profile_mode": mode,
            "current_season": current.get("season") if current else None,
            "current_season_matches": current_n,
            "prior_season": prior.get("season") if prior else None,
            "prior_season_matches": prior_n,
            "prior_weight": round(prior_weight, 4),
            "effective_sample_size": round(current_n + min(prior_n, EARLY_SEASON_PRIOR_STRENGTH), 2),
        }
        _team_profile_context_cache[cache_key] = (effective, audit)
        return effective, audit

    target_rank = _season_rank_for_target_date(target_date)
    all_rows = _get_all_team_fixture_rows(team_name, league)
    rows_before = [
        row for row in all_rows
        if str((row.get("meta") or {}).get("fixture_date") or "")[:10] < cutoff
    ]
    current_rows = [
        row for row in rows_before
        if parse_season_rank((row.get("meta") or {}).get("season")) == target_rank
    ]
    prior_rank = None
    known_ranks = {
        parse_season_rank((row.get("meta") or {}).get("season"))
        for row in rows_before
    }
    known_ranks.update(
        parse_season_rank((document.get("meta") or {}).get("season"))
        for document in documents
    )
    prior_candidates = [rank for rank in known_ranks if rank >= 0 and (target_rank is None or rank < target_rank)]
    if prior_candidates:
        prior_rank = max(prior_candidates)
    prior_rows = [
        row for row in rows_before
        if parse_season_rank((row.get("meta") or {}).get("season")) == prior_rank
    ]

    current_season = (
        str((current_rows[0].get("meta") or {}).get("season") or "")
        if current_rows else _season_label(target_rank)
    )
    prior_season = (
        str((prior_rows[0].get("meta") or {}).get("season") or "")
        if prior_rows else _season_label(prior_rank)
    )
    current = _profile_from_fixture_rows(team_name, league, current_season, current_rows)
    # A prior season is complete before the target date, so fixture rows are
    # preferred and an archived profile is a safe fallback for sparse imports.
    prior = _profile_from_fixture_rows(team_name, league, prior_season, prior_rows)
    if not prior:
        prior = _profile_for_rank_from_docs(team_name, league, prior_rank)

    effective, prior_weight, mode = _blend_sparse_current_profile(current, prior)
    if effective:
        effective["season"] = current_season or effective.get("season")
        effective["_profile_source"] = "fixture_rows_as_of"
    current_n = int(_numeric(current.get("matches_played")) or 0)
    prior_n = int(_numeric(prior.get("matches_played")) or 0)
    audit = {
        "team": canonical_team_name(team_name),
        "as_of": cutoff,
        "temporal_status": "fixture_rows_strictly_before_target_date",
        "profile_mode": mode,
        "current_season": current_season,
        "current_season_matches": current_n,
        "prior_season": prior.get("season") if prior else prior_season,
        "prior_season_matches": prior_n,
        "prior_weight": round(prior_weight, 4),
        "effective_sample_size": round(current_n + min(prior_n, EARLY_SEASON_PRIOR_STRENGTH), 2),
    }
    _team_profile_context_cache[cache_key] = (effective, audit)
    return effective, audit


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


def get_blended_profile_meta(
    team_name: str,
    competition: str,
    target_date: Optional[str] = None,
) -> Dict:
    """For European competitions, blend domestic (80%) + European (20%) profiles.
    For domestic leagues, returns the standard profile unchanged."""
    if competition not in EUROPEAN_COMPETITIONS:
        return get_team_profile_context(team_name, competition, target_date=target_date)[0]

    ew = SCORING_WEIGHTS["european"]
    euro_meta = get_team_profile_context(team_name, competition, target_date=target_date)[0]
    domestic_lg = resolve_domestic_league(team_name)

    if not domestic_lg:
        return euro_meta  # Non-top-5 team -- European data only

    domestic_meta = get_team_profile_context(team_name, domestic_lg, target_date=target_date)[0]
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

def _profile_meta(team: str, league: str, target_date: Optional[str] = None) -> Dict:
    """Profile metadata -- auto-blends for European competitions."""
    if league in EUROPEAN_COMPETITIONS:
        return get_blended_profile_meta(team, league, target_date=target_date)
    return get_team_profile_context(team, league, target_date=target_date)[0]


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
    rows = list(_get_all_team_fixture_rows(team_name, league))
    target_season = season
    if target_season is None and target_date:
        target_rank = _season_rank_for_target_date(target_date)
        matching = [
            row for row in rows
            if parse_season_rank((row.get("meta") or {}).get("season")) == target_rank
        ]
        # A season can have no completed fixtures at its first kickoff.  Do not
        # silently pull prior-season fixtures into the active-season recent
        # window in that case.
        target_season = (
            str((matching[0].get("meta") or {}).get("season") or "")
            if matching else _season_label(target_rank)
        )
    elif target_season is None:
        profile = get_team_profile_doc(team_name, league)
        target_season = (profile.get("meta") or {}).get("season")

    if target_season:
        rows = [
            row for row in rows
            if str((row.get("meta") or {}).get("season") or "") == str(target_season)
        ]
    if target_date:
        cutoff = str(target_date)[:10]
        rows = [
            row for row in rows
            # Same-day rows can be later kickoffs and fixture metadata does not
            # reliably preserve a full timestamp.  Excluding the whole day is
            # conservative but guarantees that the target fixture never leaks
            # into its own prediction.
            if str((row.get("meta") or {}).get("fixture_date") or "")[:10] < cutoff
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
        meta = _profile_meta(team_name, league, target_date=fixture_date)
        season_var = _numeric(meta.get(season_var_key)) if meta else None
    if season_var is not None and recent_var is not None:
        return 0.7 * season_var + 0.3 * recent_var
    return season_var if season_var is not None else recent_var
