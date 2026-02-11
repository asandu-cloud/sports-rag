#!/usr/bin/env python3
"""
RAG CLI v2 (from scratch, odds-first, lenient intent handling)

Goals:
- Capability-first: detect what the Odds API actually provides today.
- User-friendly: avoid hard failures when constraints are impossible.
- Constraint-aware: apply hard constraints when feasible, then gracefully fallback.
- Lightweight: focused parlay assistant with concise, evidence-backed output.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations
from math import comb as _comb, log
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI
from zoneinfo import ZoneInfo

try:
    from heuristics import top_markets
except ImportError:
    from Scripts.rag_ingest.heuristics import top_markets

try:
    from chroma_backend import backend_description, env_bool, env_first, get_chroma_client
except ImportError:
    from Scripts.rag_ingest.chroma_backend import backend_description, env_bool, env_first, get_chroma_client


load_dotenv()

# ---- Config ----
ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = str(ROOT / "Index" / "chroma")
COLLECTION = env_first("CHROMA_COLLECTION", default="football_top5")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = env_first("ODDS-API", "ODDS_API_KEY")
OPENAI_API_KEY = env_first("OPENAI_API_KEY")
CHAT_MODEL = env_first("RAG_CHAT_MODEL", default="gpt-5.2")

LEAGUE_TO_ODDS_SPORT = {
    "EPL": "soccer_epl",
    "LaLiga": "soccer_spain_la_liga",
    "SerieA": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue1": "soccer_france_ligue_one",
}

# Provider-safe defaults. Some subscriptions do not expose extra market families.
DEFAULT_MARKETS = ["h2h", "totals", "spreads"]
KNOWN_GROUPS = {"corners", "cards", "moneyline", "totals", "spreads"}
AROUND_TARGET_TOLERANCE = 0.20
try:
    MIN_LEG_ODDS = max(1.01, float(env_first("MIN_LEG_ODDS", default="1.10")))
except Exception:
    MIN_LEG_ODDS = 1.10

# ---- Scoring Weights (centralized for regression testing + tuning) ----
SCORING_WEIGHTS = {
    "combo": {
        "around_target_log_penalty": 12.0,
        "neutral_anchor_base": 1.38,
        "neutral_anchor_deviation": 4.8,
        "target_abs_deviation": 0.6,
        "target_over_limit_penalty": 3.5,
        "around_over_tolerance_penalty": 15.0,
        "desired_leg_floor_ratio": 0.82,
        "desired_leg_under_penalty": 9.5,
        "soft_prefer_miss": 0.35,
        "conservative_threshold": 1.15,
        "conservative_penalty": 24.0,
        "kb_quality_weight": 0.75,
        "ml_spread_stack_penalty": 6.0,
        "hard_exclude_penalty": 25.0,
        "forbid_spread_penalty": 30.0,
        "require_total_corner_penalty": 35.0,
        "hard_include_missing": 18.0,
        "composition_missing": 22.0,
    },
    "kb_quality": {
        "heuristic_group_bonus": 1.5,
        "form_edge": 1.0,
        "control_edge": 0.8,
        "dominance_edge": 0.6,
        "sot_edge": 0.08,
        "sot_pace": 0.20,
        "goals_over_offset": 2.2,
        "goals_under_offset": 2.8,
        "goals_line_fit": 0.55,
        "corners_season_blend": 0.65,
        "corners_recent_blend": 0.35,
        "corners_line_fit": 0.6,
        "corners_side_edge": 0.8,
        "cards_line_fit": 0.7,
        "long_odds_threshold": 3.0,
        "long_odds_offset": 2.5,
        "long_odds_penalty": 0.7,
    },
    "totals_line": {
        "min_odds_hard": 1.20,
        "min_odds_preferred": 1.35,
        "ideal_odds_low": 1.55,
        "ideal_odds_high": 2.20,
        "proximity_penalty": 0.55,
        "edge_weight": 1.15,
        "short_price_penalty": 2.0,
        "ideal_range_bonus": 0.18,
        "outside_ideal_bonus": 0.10,
        "far_threshold": 3.5,
        "far_penalty": 1.3,
    },
    "projection": {
        "corners_own": 0.6,
        "corners_opp": 0.4,
        "cards_base": 0.85,
        "cards_agg": 0.15,
        "cards_agg_scale": 3.0,
        "blend_season": 0.65,
        "blend_recent": 0.35,
    },
    "comparison_cards": {
        "cards": 0.55,
        "fouls": 0.08,
        "agg": 0.35,
        "agg_scale": 2.0,
        "recent_cards": 0.25,
        "recent_fouls": 0.03,
        "opp_control": 0.25,
    },
    "comparison_corners": {
        "proj": 0.50,
        "season_for": 0.25,
        "opp_allow": 0.15,
        "recent_for": 0.15,
        "control": 0.25,
        "edge": 0.12,
    },
    "comparison_sot": {
        "proj": 0.45,
        "season_for": 0.25,
        "opp_allow": 0.15,
        "recent_for": 0.25,
        "sot_ratio": 0.20,
    },
    "projection_sot": {
        "own": 0.6,
        "opp": 0.4,
    },
    "confidence": {
        "edge_high": 0.75,
        "edge_medium": 0.35,
        "corners_gap_high": 0.60,
        "corners_gap_medium": 0.25,
        "cards_gap_high": 0.45,
        "cards_gap_medium": 0.20,
        "sot_gap_high": 0.50,
        "sot_gap_medium": 0.20,
    },
    "pool": {
        "default_anchor": 1.55,
        "quality_bonus": 0.12,
        "cap": 220,
        "beam_width": 60,
        "brute_force_threshold": 50000,
    },
}

client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

memory = {
    "last_query": None,
    "last_answer": None,
    "last_selected_event_ids": [],
    "last_selected_fixtures": [],
    "last_leg_count": None,
    "last_intent": None,
    "last_league": None,
    "last_constraint_spec": None,
    "last_selected_legs": [],
    "last_market_groups_used": [],
    "history": [],
}

_team_meta_cache: Dict[Tuple[str, str], Dict] = {}
_team_profile_doc_cache: Dict[Tuple[str, str], Dict] = {}
_team_recent_stats_cache: Dict[Tuple[str, str, int], Dict] = {}
_chroma_client = None
_chroma_collection = None


HALF_MARKET_PATTERN = re.compile(r"(?:^|_)(h1|h2|1st_half|2nd_half|first_half|second_half)(?:_|$)")


@dataclass(frozen=True)
class CandidateLeg:
    event_id: str
    fixture: str
    home_team: str
    away_team: str
    market_key: str
    outcome: str
    odds: float
    point: Optional[float]
    bookmaker: Optional[str]


@dataclass
class ConstraintSpec:
    requested_markets: Set[str]
    hard_include_groups: Set[str]
    hard_exclude_groups: Set[str]
    soft_prefer_groups: Set[str]
    required_group_counts: Dict[str, int]
    forbid_spread_keys: bool
    require_total_corner_keys: bool
    target_multiplier: Optional[float]
    target_mode: str  # none | around | max | min
    leg_count: Optional[int]
    per_match_mode: bool
    require_unique_events: bool
    time_window: str  # today | tomorrow | upcoming
    league: str


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
}


def resolve_league(user_q: str, default: str = "EPL") -> str:
    q = user_q.lower()
    hints = {
        "EPL": ["epl", "premier league", "england"],
        "LaLiga": ["laliga", "la liga", "spain"],
        "SerieA": ["serie a", "italy"],
        "Bundesliga": ["bundesliga", "germany"],
        "Ligue1": ["ligue 1", "france"],
    }
    for lg, keys in hints.items():
        if any(k in q for k in keys):
            return lg
    return default


def parse_time_window(user_q: str) -> str:
    q = user_q.lower()
    if "today" in q or "tonight" in q:
        return "today"
    if "tomorrow" in q:
        return "tomorrow"
    return "upcoming"


def parse_explicit_date(user_q: str, tz_name: str = "Europe/London") -> Optional[date]:
    """
    Parse explicit date mentions like:
    - 10th of February
    - February 10
    - Feb 10
    - 10/02 or 10-02
    """
    q = user_q.lower()
    month_map = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    tz = ZoneInfo(tz_name)
    today = datetime.now(tz).date()

    day = None
    month = None
    year = None

    # "10th of february [2026]"
    m = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:of\s+)?"
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"(?:\s*,?\s*(20\d{2}))?\b",
        q,
    )
    if m:
        day = int(m.group(1))
        month = month_map.get(m.group(2), None)
        if m.group(3):
            year = int(m.group(3))

    # "february 10 [2026]"
    if day is None or month is None:
        m = re.search(
            r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+"
            r"(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(20\d{2}))?\b",
            q,
        )
        if m:
            month = month_map.get(m.group(1), None)
            day = int(m.group(2))
            if m.group(3):
                year = int(m.group(3))

    # "10/02[/2026]" or "10-02[-2026]" (assume DD/MM)
    if day is None or month is None:
        m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](20\d{2}|\d{2}))?\b", q)
        if m:
            day = int(m.group(1))
            month = int(m.group(2))
            if m.group(3):
                y = m.group(3)
                year = int(y) if len(y) == 4 else (2000 + int(y))

    if day is None or month is None:
        return None

    if year is None:
        year = today.year
        # If the date already passed this season, roll to next year.
        try:
            tentative = date(year, month, day)
        except ValueError:
            return None
        if tentative < today:
            year += 1

    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_target(user_q: str) -> Tuple[Optional[float], str]:
    q = user_q.lower()
    patterns = [
        (r"(?:no more than|under|below|at most|max(?:imum)?)\s*(\d+(?:\.\d+)?)\s*x?", "max"),
        (r"(?:at least|over|above|min(?:imum)?)\s*(\d+(?:\.\d+)?)\s*x?", "min"),
        (r"(?:around|about|approx(?:imately)?|target(?: of)?)\s*(\d+(?:\.\d+)?)\s*x?", "around"),
        (r"(\d+(?:\.\d+)?)\s*x", "around"),
    ]
    for pat, mode in patterns:
        m = re.search(pat, q)
        if not m:
            continue
        try:
            return float(m.group(1)), mode
        except ValueError:
            break
    return None, "none"


def parse_leg_count(user_q: str) -> Optional[int]:
    q = user_q.lower()
    m = re.search(r"(\d+)\s*[- ]?leg", q)
    if not m:
        return None
    try:
        return max(1, min(int(m.group(1)), 20))
    except ValueError:
        return None


def market_group_from_key(market_key: str) -> str:
    k = (market_key or "").lower()
    if "corner" in k:
        return "corners"
    if "card" in k or "booking" in k:
        return "cards"
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
    if HALF_MARKET_PATTERN.search(k):
        return False
    return True


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", (s or "").lower()).strip()


def canonical_team_name(name: str) -> str:
    n = normalize_name(name)
    if not n:
        return name
    # Exact alias match
    if n in TEAM_ALIAS_MAP:
        return TEAM_ALIAS_MAP[n]
    # Fuzzy fallback: find closest alias key (cutoff 0.65 to catch e.g. "Brighton Hove Albion" → "Brighton and Hove Albion")
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


def add_chat_turn(user_q: str, assistant_a: str, max_turns: int = 10) -> None:
    hist = memory.get("history") or []
    hist.append({"user": user_q, "assistant": assistant_a})
    memory["history"] = hist[-max_turns:]
    memory["last_query"] = user_q
    memory["last_answer"] = assistant_a


def render_recent_chat_context(max_turns: int = 3) -> str:
    hist = memory.get("history") or []
    if not hist:
        return "None."
    lines = []
    for i, turn in enumerate(hist[-max_turns:], start=1):
        u = (turn.get("user") or "").strip()
        a = (turn.get("assistant") or "").strip()
        lines.append(f"Turn {i} User: {u}")
        lines.append(f"Turn {i} Assistant: {a}")
    return "\n".join(lines)


def is_followup_request(user_q: str) -> bool:
    q = user_q.lower()
    if not ("leg" in q or "parlay" in q or "bet" in q):
        return False
    # Direct follow-up indicators
    if re.search(r"\b(add|another|those|previous)\b", q):
        return True
    # "same" is a follow-up indicator, but NOT in "same-game parlay" (market descriptor)
    if re.search(r"\bsame\b", q) and not re.search(r"\bsame[\s-]game\b", q):
        return True
    return False


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


def _numeric(value) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _avg(values: List[Optional[float]]) -> Optional[float]:
    cleaned = [x for x in values if x is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


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


def kb_healthcheck() -> Tuple[bool, str]:
    try:
        col = get_collection_handle(create_if_missing=False)
    except Exception as exc:
        return False, f"Collection '{COLLECTION}' unavailable ({exc}). Backend={backend_description(CHROMA_DIR)}"
    try:
        cnt = col.count()
    except Exception:
        cnt = None
    if cnt is None:
        return True, f"Connected to {backend_description(CHROMA_DIR)} (count unavailable)"
    if cnt <= 0:
        return False, f"Collection '{COLLECTION}' is empty on {backend_description(CHROMA_DIR)}"
    return True, f"Connected to {backend_description(CHROMA_DIR)} | vectors={cnt}"


def parse_constraints(user_q: str, default_league: str = "EPL") -> ConstraintSpec:
    q = user_q.lower()
    target_multiplier, target_mode = parse_target(user_q)
    explicit_leg_count = parse_leg_count(user_q)
    per_match_mode = bool(
        re.search(
            r"(as many legs as|for each game|for every game|for each fixture|for every fixture"
            r"|one leg per game|one leg per match|one leg per fixture"
            r"|one\s+\w+\s+per\s+(?:game|match|fixture)"          # "one corner per game", "one total per match"
            r"|one\s+\w+\s+\w+\s+per\s+(?:game|match|fixture)"   # "one corner total per game"
            r"|per\s+(?:game|match|fixture)\b)",                   # trailing "per game"
            q,
        )
    )

    # Hard include/exclude groups
    hard_include: Set[str] = set()
    hard_exclude: Set[str] = set()
    soft_prefer: Set[str] = set()
    required_group_counts: Dict[str, int] = {}

    only_corners = bool(re.search(r"(only|just)\s+.*corner", q) or re.search(r"corner.*(only|just)", q))
    only_cards = bool(re.search(r"(only|just)\s+.*card", q) or re.search(r"card.*(only|just)", q))
    only_ml = bool(re.search(r"(only|just)\s+.*money\s*line", q) or re.search(r"(only|just)\s+.*\bml\b", q))

    if only_corners:
        hard_include.add("corners")
    if only_cards:
        hard_include.add("cards")
    if only_ml:
        hard_include.add("moneyline")

    if only_corners:
        hard_exclude |= (KNOWN_GROUPS - {"corners"})
    if only_cards:
        hard_exclude |= (KNOWN_GROUPS - {"cards"})
    if only_ml:
        hard_exclude |= (KNOWN_GROUPS - {"moneyline"})

    if re.search(r"(?:no|without|avoid|dont|don't)\s+.*money\s*line", q) or re.search(
        r"(?:no|without|avoid|dont|don't)\s+.*\bml\b", q
    ):
        hard_exclude.add("moneyline")

    # Soft preferences from query terms (ignore explicit negations)
    no_spreads_mentioned = bool(re.search(r"(?:no|without|avoid|dont|don't)\s+.*(?:spread|handicap)", q))
    no_totals_mentioned = bool(re.search(r"(?:no|without|avoid|dont|don't)\s+.*(?:total|over|under)", q))
    no_corners_mentioned = bool(re.search(r"(?:no|without|avoid|dont|don't)\s+.*corner", q))
    no_cards_mentioned = bool(re.search(r"(?:no|without|avoid|dont|don't)\s+.*(?:card|booking)", q))

    if re.search(r"\bcorner(s)?\b", q):
        if not no_corners_mentioned:
            soft_prefer.add("corners")
    if re.search(r"\bcard(s)?\b", q) or re.search(r"\bbooking(s)?\b", q):
        if not no_cards_mentioned:
            soft_prefer.add("cards")
    if re.search(r"\bspread(s)?\b", q) or re.search(r"\bhandicap(s)?\b", q):
        if not no_spreads_mentioned:
            soft_prefer.add("spreads")
    if re.search(r"\btotal(s)?\b", q) or re.search(r"\bover\b", q) or re.search(r"\bunder\b", q):
        if not no_totals_mentioned:
            soft_prefer.add("totals")
    if re.search(r"\bmoney\s*line\b", q) or re.search(r"\bml\b", q) or re.search(r"\bh2h\b", q):
        soft_prefer.add("moneyline")

    # Composition constraints (lenient NL parse)
    if re.search(r"(one|1)\s+leg.*corner", q):
        required_group_counts["corners"] = max(required_group_counts.get("corners", 0), 1)
    if re.search(r"(one|1)\s+leg.*(booking|card)", q):
        required_group_counts["cards"] = max(required_group_counts.get("cards", 0), 1)
    if re.search(r"(one|1)\s+leg.*money\s*line", q) or re.search(r"(one|1)\s+leg.*\bml\b", q):
        required_group_counts["moneyline"] = max(required_group_counts.get("moneyline", 0), 1)

    forbid_spread_keys = bool(re.search(r"(?:no|without|avoid|dont|don't)\s+.*(?:spread|handicap)", q))
    require_total_corner_keys = bool(
        re.search(r"total\s+corner", q)
        or re.search(r"corner\s+line", q)
    )

    # Requested market keys for provider fetch.
    # Keep this as provider-safe base; specialized markets (corners/cards)
    # are discovered per-event later.
    requested_markets = set(DEFAULT_MARKETS)
    if "moneyline" in hard_include:
        requested_markets = {"h2h"}
    elif hard_exclude == {"moneyline"}:
        requested_markets = {"totals", "spreads"}

    require_unique = bool(
        per_match_mode
        or re.search(r"(different matches|across matches|one per match|one per game|one per fixture)", q)
    )

    return ConstraintSpec(
        requested_markets=requested_markets,
        hard_include_groups=hard_include,
        hard_exclude_groups=hard_exclude,
        soft_prefer_groups=soft_prefer,
        required_group_counts=required_group_counts,
        forbid_spread_keys=forbid_spread_keys,
        require_total_corner_keys=require_total_corner_keys,
        target_multiplier=target_multiplier,
        target_mode=target_mode,
        leg_count=explicit_leg_count,
        per_match_mode=per_match_mode,
        require_unique_events=require_unique,
        time_window=parse_time_window(user_q),
        league=resolve_league(user_q, default=default_league),
    )


def odds_get(path: str, params: Dict) -> Tuple[Optional[object], Optional[str]]:
    api_key = env_first("ODDS-API", "ODDS_API_KEY", default=ODDS_API_KEY)
    if not api_key:
        return None, "Missing ODDS API key (ODDS-API)."
    url = f"{ODDS_API_BASE}{path}"
    p = dict(params)
    p["api_key"] = api_key
    p["oddsFormat"] = "decimal"
    p["dateFormat"] = "iso"
    p["regions"] = "uk,eu,us"
    try:
        r = requests.get(url, params=p, timeout=25)
        if r.status_code != 200:
            return None, f"Odds API HTTP {r.status_code}: {(r.text or '')[:600]}"
        return r.json(), None
    except Exception as exc:
        return None, f"Odds API request failed: {exc}"


def fetch_events(league: str, markets: Set[str]) -> Tuple[List[Dict], List[str]]:
    notes: List[str] = []
    sport_key = LEAGUE_TO_ODDS_SPORT.get(league, LEAGUE_TO_ODDS_SPORT["EPL"])
    requested = sorted(markets)

    # Try requested bundle first.
    market_str = ",".join(requested)
    rows, err = odds_get(f"/sports/{sport_key}/odds", {"markets": market_str})
    if err:
        # Fallback to safe default bundle; keep note for transparency.
        fallback = ",".join(DEFAULT_MARKETS)
        rows_fb, err_fb = odds_get(f"/sports/{sport_key}/odds", {"markets": fallback})
        if err_fb:
            notes.append(err)
            return [], notes
        rows = rows_fb
        notes.append(f"Requested markets '{market_str}' not fully supported. Fell back to '{fallback}'.")
    if not isinstance(rows, list):
        notes.append("Unexpected odds payload shape.")
        return [], notes
    return rows, notes


def extract_market_keys(payload: object) -> Set[str]:
    keys: Set[str] = set()

    def scan_block(block: object) -> None:
        if isinstance(block, dict):
            if "bookmakers" in block:
                for bm in block.get("bookmakers", []) or []:
                    for mk in bm.get("markets", []) or []:
                        k = mk.get("key")
                        if k:
                            keys.add(str(k))
            if "markets" in block:
                for mk in block.get("markets", []) or []:
                    k = mk.get("key")
                    if k:
                        keys.add(str(k))
        elif isinstance(block, list):
            for item in block:
                scan_block(item)

    scan_block(payload)
    return keys


def event_market_groups(event: Dict) -> Set[str]:
    groups: Set[str] = set()
    for bm in event.get("bookmakers", []) or []:
        for mk in bm.get("markets", []) or []:
            k = mk.get("key")
            if k:
                groups.add(market_group_from_key(str(k)))
    return groups


def discover_event_market_keys(league: str, event_id: str) -> Tuple[Set[str], Optional[str]]:
    sport_key = LEAGUE_TO_ODDS_SPORT.get(league, LEAGUE_TO_ODDS_SPORT["EPL"])
    payload, err = odds_get(f"/sports/{sport_key}/events/{event_id}/markets", {})
    if err:
        return set(), err
    keys = extract_market_keys(payload)
    return keys, None


def fetch_event_odds_for_market_keys(league: str, event_id: str, market_keys: Set[str]) -> Tuple[Optional[Dict], Optional[str]]:
    if not market_keys:
        return None, "No market keys requested."
    sport_key = LEAGUE_TO_ODDS_SPORT.get(league, LEAGUE_TO_ODDS_SPORT["EPL"])
    payload, err = odds_get(
        f"/sports/{sport_key}/events/{event_id}/odds",
        {"markets": ",".join(sorted(market_keys))},
    )
    if err:
        return None, err
    if isinstance(payload, dict):
        return payload, None
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0], None
    return None, "Unexpected event odds payload shape."


def merge_bookmakers(base: List[Dict], extra: List[Dict]) -> List[Dict]:
    """
    Merge bookmaker market blocks by bookmaker title + market key + outcome key.
    Keep highest price when duplicates occur.
    """
    rows: Dict[Tuple[str, str, str, str], Dict] = {}

    def ingest(bookmakers: List[Dict]) -> None:
        for bm in bookmakers or []:
            bname = str(bm.get("title") or "")
            for mk in bm.get("markets", []) or []:
                mkey = str(mk.get("key") or "")
                for out in mk.get("outcomes", []) or []:
                    oname = str(out.get("name") or "")
                    pkey = "" if out.get("point") is None else str(out.get("point"))
                    k = (bname, mkey, oname, pkey)
                    cur = rows.get(k)
                    try:
                        price = float(out.get("price"))
                    except Exception:
                        continue
                    if (cur is None) or (price > float(cur.get("price", 0.0))):
                        rows[k] = dict(out)

    ingest(base)
    ingest(extra)

    # Rebuild nested structure
    nested: Dict[Tuple[str, str], List[Dict]] = {}
    for (bname, mkey, _, _), out in rows.items():
        nested.setdefault((bname, mkey), []).append(out)

    bm_map: Dict[str, Dict] = {}
    for (bname, mkey), outcomes in nested.items():
        bm = bm_map.setdefault(bname, {"title": bname, "markets": []})
        bm["markets"].append({"key": mkey, "outcomes": outcomes})

    return list(bm_map.values())


def enrich_events_for_groups(events: List[Dict], league: str, desired_groups: Set[str]) -> Tuple[List[Dict], List[str]]:
    """
    For groups like corners/cards that are often unavailable at sport-level odds,
    discover event-level market keys and fetch event-specific odds.
    """
    if not desired_groups:
        return events, []

    notes: List[str] = []
    out: List[Dict] = []
    for ev in events:
        existing_groups = event_market_groups(ev)
        missing = {g for g in desired_groups if g not in existing_groups}
        if not missing:
            out.append(ev)
            continue

        event_id = str(ev.get("id") or "")
        if not event_id:
            out.append(ev)
            continue

        keys, err = discover_event_market_keys(league, event_id)
        if err:
            notes.append(f"{ev.get('home_team')} vs {ev.get('away_team')}: market discovery failed ({err})")
            out.append(ev)
            continue

        wanted_keys = {k for k in keys if market_group_from_key(k) in missing}
        if not wanted_keys:
            notes.append(
                f"{ev.get('home_team')} vs {ev.get('away_team')}: no {', '.join(sorted(missing))} keys returned by provider."
            )
            out.append(ev)
            continue

        ev_payload, err2 = fetch_event_odds_for_market_keys(league, event_id, wanted_keys)
        if err2 or not ev_payload:
            notes.append(
                f"{ev.get('home_team')} vs {ev.get('away_team')}: failed fetching event odds for discovered keys."
            )
            out.append(ev)
            continue

        merged = dict(ev)
        merged["bookmakers"] = merge_bookmakers(ev.get("bookmakers", []) or [], ev_payload.get("bookmakers", []) or [])
        out.append(merged)

    return out, notes


def totals_key_matches_group(market_key: str, stat_group: str) -> bool:
    k = (market_key or "").lower()
    if not is_full_game_market(k):
        return False
    if "total" not in k:
        return False
    if stat_group == "goals":
        if "corner" in k or "card" in k or "booking" in k:
            return False
        if "team_total" in k or "player" in k or "goalscorer" in k:
            return False
        return True
    if stat_group == "corners":
        return ("corner" in k)
    if stat_group == "cards":
        return ("card" in k or "booking" in k)
    return False


def enrich_events_for_totals_depth(events: List[Dict], league: str, stat_group: str = "goals") -> Tuple[List[Dict], List[str]]:
    """
    Expand totals markets even when 'totals' group already exists, by discovering
    event-level alternate totals keys and merging them into each event.
    """
    out: List[Dict] = []
    notes: List[str] = []
    for ev in events:
        event_id = str(ev.get("id") or "")
        if not event_id:
            out.append(ev)
            continue

        keys, err = discover_event_market_keys(league, event_id)
        if err:
            notes.append(f"{ev.get('home_team')} vs {ev.get('away_team')}: totals depth discovery failed ({err})")
            out.append(ev)
            continue

        wanted_keys = {k for k in keys if totals_key_matches_group(k, stat_group)}
        if not wanted_keys:
            notes.append(
                f"{ev.get('home_team')} vs {ev.get('away_team')}: no full-game {stat_group} totals keys returned by provider."
            )
            out.append(ev)
            continue

        # Keep request bounded while preserving a broad odds ladder.
        wanted_keys = set(sorted(wanted_keys)[:30])
        ev_payload, err2 = fetch_event_odds_for_market_keys(league, event_id, wanted_keys)
        if err2 or not ev_payload:
            notes.append(f"{ev.get('home_team')} vs {ev.get('away_team')}: totals depth fetch failed.")
            out.append(ev)
            continue

        merged = dict(ev)
        merged["bookmakers"] = merge_bookmakers(ev.get("bookmakers", []) or [], ev_payload.get("bookmakers", []) or [])
        out.append(merged)

    return out, notes


def filter_events_by_window(events: List[Dict], window: str, tz_name: str = "Europe/London") -> List[Dict]:
    if window == "upcoming":
        return events
    tz = ZoneInfo(tz_name)
    today = datetime.now(tz).date()
    target = today if window == "today" else today.fromordinal(today.toordinal() + 1)
    out: List[Dict] = []
    for ev in events:
        ts = ev.get("commence_time")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(tz)
        except Exception:
            continue
        if dt.date() == target:
            out.append(ev)
    return out


def filter_events_by_exact_date(events: List[Dict], target: date, tz_name: str = "Europe/London") -> List[Dict]:
    tz = ZoneInfo(tz_name)
    out: List[Dict] = []
    for ev in events:
        ts = ev.get("commence_time")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(tz)
        except Exception:
            continue
        if dt.date() == target:
            out.append(ev)
    return out


def event_fixtures_text(events: List[Dict]) -> str:
    if not events:
        return "No fixtures found."
    lines = []
    for i, ev in enumerate(events, start=1):
        lines.append(f"{i}. {ev.get('home_team')} vs {ev.get('away_team')} | {ev.get('commence_time')}")
    return "\n".join(lines)


def build_candidates(events: List[Dict]) -> List[CandidateLeg]:
    best_rows: Dict[Tuple[str, str, str, str], CandidateLeg] = {}
    for ev in events:
        event_id = str(ev.get("id") or "")
        home = str(ev.get("home_team") or "Home")
        away = str(ev.get("away_team") or "Away")
        fixture = f"{home} vs {away}"
        for bm in ev.get("bookmakers", []) or []:
            bm_title = bm.get("title")
            for mk in bm.get("markets", []) or []:
                market_key = str(mk.get("key") or "")
                if not is_full_game_market(market_key):
                    continue
                for out in mk.get("outcomes", []) or []:
                    name = str(out.get("name") or "")
                    if not name:
                        continue
                    point_raw = out.get("point")
                    try:
                        point = float(point_raw) if point_raw is not None else None
                    except Exception:
                        point = None
                    try:
                        odds = float(out.get("price"))
                    except Exception:
                        continue
                    if odds < MIN_LEG_ODDS:
                        continue
                    row = CandidateLeg(
                        event_id=event_id,
                        fixture=fixture,
                        home_team=home,
                        away_team=away,
                        market_key=market_key,
                        outcome=name,
                        odds=odds,
                        point=point,
                        bookmaker=bm_title,
                    )
                    k = (event_id, market_key, name, "" if point is None else str(point))
                    cur = best_rows.get(k)
                    if cur is None or row.odds > cur.odds:
                        best_rows[k] = row
    return list(best_rows.values())


def available_market_groups(candidates: List[CandidateLeg]) -> Set[str]:
    return {market_group_from_key(c.market_key) for c in candidates}


def contradictory(a: CandidateLeg, b: CandidateLeg) -> bool:
    if a.event_id != b.event_id:
        return False
    ga = market_group_from_key(a.market_key)
    gb = market_group_from_key(b.market_key)

    # One-leg-per-market-family per fixture.
    # This blocks duplicated exposure like:
    # - Over/Under corners in the same match
    # - multiple cards lines in the same match
    # - multiple spreads / multiple moneylines in the same match
    if ga == gb and ga in {"moneyline", "spreads", "totals", "corners", "cards"}:
        return True

    if (
        ga == gb
        and ga in {"totals", "spreads"}
        and a.point == b.point
        and a.outcome.lower() != b.outcome.lower()
    ):
        return True
    # If same market key + same point but opposite outcomes, treat as contradictory.
    if a.market_key == b.market_key and a.point == b.point and a.outcome.lower() != b.outcome.lower():
        return True

    def side_of_outcome(leg: CandidateLeg) -> Optional[str]:
        out = (leg.outcome or "").strip().lower()
        h = (leg.home_team or "").strip().lower()
        aw = (leg.away_team or "").strip().lower()
        if out == h:
            return "home"
        if out == aw:
            return "away"
        if out in {"draw", "tie", "x"}:
            return "draw"
        # Try prefix match for provider naming variants.
        if h and out.startswith(h):
            return "home"
        if aw and out.startswith(aw):
            return "away"
        return None

    side_a = side_of_outcome(a)
    side_b = side_of_outcome(b)

    # Equivalent result exposure:
    # ML(home) ~= spread(home -0.5), ML(away) ~= spread(away +0.5)
    if {ga, gb} == {"moneyline", "spreads"} and side_a in {"home", "away"} and side_b in {"home", "away"}:
        ml = a if ga == "moneyline" else b
        sp = b if ga == "moneyline" else a
        ml_side = side_of_outcome(ml)
        sp_side = side_of_outcome(sp)
        sp_point = sp.point
        if sp_point is not None and ml_side == sp_side:
            # Same-side spread near equivalent to ML:
            # -0.5 or 0.0 or +0.25 on selected side generally overlaps heavily with ML exposure.
            if abs(float(sp_point)) <= 0.5:
                return True
            # Also catch away +0.5 paired with away ML and home -0.5 paired with home ML.
            if (sp_side == "home" and sp_point <= -0.5) or (sp_side == "away" and sp_point >= 0.5):
                return True
    return False


def combo_valid(combo: List[CandidateLeg], require_unique_events: bool) -> bool:
    # Global product rule:
    # never allow ultra-short legs below configured minimum odds.
    if any(x.odds < MIN_LEG_ODDS for x in combo):
        return False

    if require_unique_events:
        ids = [x.event_id for x in combo]
        if len(ids) != len(set(ids)):
            return False

    # Global product rule:
    # Never mix moneyline and spread legs in the same parlay.
    groups = {market_group_from_key(x.market_key) for x in combo}
    if "moneyline" in groups and "spreads" in groups:
        return False

    for i in range(len(combo)):
        for j in range(i + 1, len(combo)):
            if contradictory(combo[i], combo[j]):
                return False
    return True


def heuristic_groups_for_event(home_meta: Dict, away_meta: Dict) -> Set[str]:
    if not home_meta or not away_meta:
        return set()
    groups: Set[str] = set()
    try:
        picks = top_markets(home_meta, away_meta, n=5)
    except Exception:
        picks = []
    for p in picks:
        label = str(p.get("market") or "").lower()
        if "corner" in label:
            groups.add("corners")
        elif "card" in label or "booking" in label:
            groups.add("cards")
        elif "goal" in label or "btts" in label or "under" in label or "over" in label:
            groups.add("totals")
    return groups


def kb_leg_quality(leg: CandidateLeg, league: str) -> float:
    """
    Positive score indicates stronger multi-signal support from local KB.
    """
    hm = get_team_profile_meta(leg.home_team, league)
    am = get_team_profile_meta(leg.away_team, league)
    h_recent = get_team_recent_stats(leg.home_team, league, last_n=6)
    a_recent = get_team_recent_stats(leg.away_team, league, last_n=6)
    heur_groups = heuristic_groups_for_event(hm, am)

    g = market_group_from_key(leg.market_key)
    score = 0.0

    h_ctrl = _numeric(hm.get("control_index"))
    a_ctrl = _numeric(am.get("control_index"))
    h_form = _numeric(hm.get("form_index_team"))
    a_form = _numeric(am.get("form_index_team"))
    h_dom = _numeric(hm.get("dominance_index"))
    a_dom = _numeric(am.get("dominance_index"))

    w = SCORING_WEIGHTS["kb_quality"]

    if g in heur_groups:
        score += w["heuristic_group_bonus"]

    if g == "moneyline" or g == "spreads":
        edge = 0.0
        home_sign = 1.0 if leg.outcome.lower().startswith(leg.home_team.lower()) else -1.0
        if h_form is not None and a_form is not None:
            edge += (h_form - a_form) * w["form_edge"] * home_sign
        if h_ctrl is not None and a_ctrl is not None:
            edge += (h_ctrl - a_ctrl) * w["control_edge"] * home_sign
        if h_dom is not None and a_dom is not None:
            edge += (h_dom - a_dom) * w["dominance_edge"] * home_sign
        h_sot = h_recent.get("sot_for_avg")
        a_sot = a_recent.get("sot_for_avg")
        if h_sot is not None and a_sot is not None:
            edge += (h_sot - a_sot) * w["sot_edge"] * home_sign
        score += edge

    if g == "totals":
        h_goal_base = _numeric(hm.get("goals_for_pm"))
        a_goal_base = _numeric(am.get("goals_for_pm"))
        h_sot = h_recent.get("sot_for_avg")
        a_sot = a_recent.get("sot_for_avg")
        pace = 0.0
        if h_goal_base is not None and a_goal_base is not None:
            pace += h_goal_base + a_goal_base
        if h_sot is not None and a_sot is not None:
            pace += w["sot_pace"] * (h_sot + a_sot)
        direction = leg.outcome.lower()
        if "over" in direction:
            score += pace - w["goals_over_offset"]
        elif "under" in direction:
            score += w["goals_under_offset"] - pace
        if leg.point is not None:
            if "over" in direction:
                score += (pace - leg.point) * w["goals_line_fit"]
            elif "under" in direction:
                score += (leg.point - pace) * w["goals_line_fit"]

    if g == "corners":
        h_proj, a_proj = projected_corners(hm, am)
        h_recent_for = h_recent.get("corners_for_avg")
        a_recent_for = a_recent.get("corners_for_avg")
        combined = None
        if h_proj is not None and a_proj is not None:
            combined = h_proj + a_proj
        if h_recent_for is not None and a_recent_for is not None:
            recent_combined = h_recent_for + a_recent_for
            combined = recent_combined if combined is None else (w["corners_season_blend"] * combined + w["corners_recent_blend"] * recent_combined)
        if combined is not None:
            direction = leg.outcome.lower()
            if leg.point is not None and "total" in leg.market_key.lower():
                if "over" in direction:
                    score += (combined - leg.point) * w["corners_line_fit"]
                elif "under" in direction:
                    score += (leg.point - combined) * w["corners_line_fit"]
            if leg.point is None or "total" not in leg.market_key.lower():
                if h_proj is not None and a_proj is not None:
                    side_edge = h_proj - a_proj
                    if leg.outcome.lower().startswith(leg.home_team.lower()):
                        score += side_edge * w["corners_side_edge"]
                    elif leg.outcome.lower().startswith(leg.away_team.lower()):
                        score -= side_edge * w["corners_side_edge"]

    if g == "cards":
        h_cards = _numeric(hm.get("cards_per_90_team"))
        a_cards = _numeric(am.get("cards_per_90_team"))
        h_recent_cards = h_recent.get("cards_avg")
        a_recent_cards = a_recent.get("cards_avg")
        combined_cards = 0.0
        n = 0
        for x in [h_cards, a_cards, h_recent_cards, a_recent_cards]:
            if x is not None:
                combined_cards += x
                n += 1
        if n > 0:
            combined_cards = combined_cards / max(1, n / 2.0)
            direction = leg.outcome.lower()
            if leg.point is not None and "total" in leg.market_key.lower():
                if "over" in direction:
                    score += (combined_cards - leg.point) * w["cards_line_fit"]
                elif "under" in direction:
                    score += (leg.point - combined_cards) * w["cards_line_fit"]

    if leg.odds >= w["long_odds_threshold"]:
        score -= (leg.odds - w["long_odds_offset"]) * w["long_odds_penalty"]
    return score


def score_combo(combo: List[CandidateLeg], c: ConstraintSpec, leg_quality: Dict[CandidateLeg, float]) -> float:
    w = SCORING_WEIGHTS["combo"]
    combo_odds = 1.0
    for leg in combo:
        combo_odds *= leg.odds

    # Base scoring
    if c.target_multiplier and c.target_mode == "around":
        score = abs(log(combo_odds) - log(c.target_multiplier)) * w["around_target_log_penalty"]
    else:
        neutral_anchor = w["neutral_anchor_base"] ** max(1, len(combo))
        score = abs(log(combo_odds) - log(neutral_anchor)) * w["neutral_anchor_deviation"]

    if c.target_multiplier and c.target_multiplier > 0:
        tgt = c.target_multiplier
        if c.target_mode == "max":
            score += abs(combo_odds - tgt) * w["target_abs_deviation"]
            if combo_odds > tgt:
                score += (combo_odds - tgt) * w["target_over_limit_penalty"]
        elif c.target_mode == "min":
            score += abs(combo_odds - tgt) * w["target_abs_deviation"]
            if combo_odds < tgt:
                score += (tgt - combo_odds) * w["target_over_limit_penalty"]
        else:
            rel_dev = abs(combo_odds - tgt) / max(tgt, 1e-9)
            if rel_dev > AROUND_TARGET_TOLERANCE:
                score += (rel_dev - AROUND_TARGET_TOLERANCE) * w["around_over_tolerance_penalty"]

        desired_leg = tgt ** (1.0 / max(1, len(combo)))
        for leg in combo:
            if c.target_mode in {"around", "min"} and leg.odds < desired_leg * w["desired_leg_floor_ratio"]:
                score += (desired_leg * w["desired_leg_floor_ratio"] - leg.odds) * w["desired_leg_under_penalty"]

    # Soft preferences
    if c.soft_prefer_groups:
        for leg in combo:
            g = market_group_from_key(leg.market_key)
            if g not in c.soft_prefer_groups:
                score += w["soft_prefer_miss"]

    # Global anti-conservative guardrail.
    for leg in combo:
        if leg.odds < w["conservative_threshold"]:
            score += (w["conservative_threshold"] - leg.odds) * w["conservative_penalty"]

    # Prefer higher multi-signal support from local KB.
    quality = sum(leg_quality.get(leg, 0.0) for leg in combo)
    score -= quality * w["kb_quality_weight"]

    # Diversity penalty for same-event, same-thesis stacking beyond contradictions.
    for i in range(len(combo)):
        for j in range(i + 1, len(combo)):
            a = combo[i]
            b = combo[j]
            if a.event_id != b.event_id:
                continue
            ga = market_group_from_key(a.market_key)
            gb = market_group_from_key(b.market_key)
            if {ga, gb} == {"moneyline", "spreads"}:
                score += w["ml_spread_stack_penalty"]

    # Hard excludes should be strongly avoided.
    for leg in combo:
        g = market_group_from_key(leg.market_key)
        if g in c.hard_exclude_groups:
            score += w["hard_exclude_penalty"]
        if c.forbid_spread_keys and (("spread" in leg.market_key.lower()) or ("handicap" in leg.market_key.lower())):
            score += w["forbid_spread_penalty"]
        if c.require_total_corner_keys:
            k = leg.market_key.lower()
            if ("corner" not in k) or ("total" not in k):
                score += w["require_total_corner_penalty"]

    # Hard includes should appear when available.
    for req in c.hard_include_groups:
        if not any(market_group_from_key(x.market_key) == req for x in combo):
            score += w["hard_include_missing"]

    # Composition constraints (e.g., 1 corner leg + 1 booking leg)
    for grp, need in c.required_group_counts.items():
        have = sum(1 for x in combo if market_group_from_key(x.market_key) == grp)
        if have < need:
            score += (need - have) * w["composition_missing"]

    return score


def check_feasibility(candidates: List[CandidateLeg], leg_count: int, c: ConstraintSpec) -> Tuple[bool, List[str]]:
    """
    Pre-check whether constraints are satisfiable before the expensive solver.
    Returns (feasible, warnings). Non-feasible means the solver would waste time.
    """
    warnings: List[str] = []
    if leg_count <= 0:
        return False, ["Invalid leg count."]
    if not candidates:
        return False, ["No odds candidates available."]

    avail_groups = {market_group_from_key(c_leg.market_key) for c_leg in candidates}
    unique_events = len({c_leg.event_id for c_leg in candidates})

    if c.require_unique_events and unique_events < leg_count:
        warnings.append(
            f"Requested {leg_count} legs across unique fixtures, but only {unique_events} fixture(s) have odds. "
            f"Reducing leg count to {unique_events}."
        )

    for grp in c.hard_include_groups:
        if grp not in avail_groups:
            warnings.append(
                f"Requested '{grp}' markets, but none are available from the current odds feed. "
                f"Available groups: {', '.join(sorted(avail_groups)) or 'none'}."
            )

    for grp, need in c.required_group_counts.items():
        grp_legs = [c_leg for c_leg in candidates if market_group_from_key(c_leg.market_key) == grp]
        if len(grp_legs) < need:
            warnings.append(
                f"Requested {need} '{grp}' leg(s), but only {len(grp_legs)} candidate(s) available."
            )

    if c.require_total_corner_keys:
        corner_total_legs = [
            c_leg for c_leg in candidates
            if "corner" in c_leg.market_key.lower() and "total" in c_leg.market_key.lower()
        ]
        if not corner_total_legs:
            warnings.append(
                "Total-corners-only requested, but no full-game total corner market keys found in feed."
            )

    feasible = not any(w.startswith("Requested") and "none are available" in w for w in warnings)
    return feasible, warnings


def _brute_force_combos(
    pool: List[CandidateLeg], leg_count: int, c: ConstraintSpec, leg_quality: Dict[CandidateLeg, float]
) -> List[Tuple[List[CandidateLeg], float, float]]:
    scored: List[Tuple[List[CandidateLeg], float, float]] = []
    for combo in combinations(pool, leg_count):
        combo_list = list(combo)
        if not combo_valid(combo_list, c.require_unique_events):
            continue
        s = score_combo(combo_list, c, leg_quality)
        scored.append((combo_list, s, combined_odds(combo_list)))
    return scored


def _beam_search_combos(
    pool: List[CandidateLeg], leg_count: int, c: ConstraintSpec, leg_quality: Dict[CandidateLeg, float]
) -> List[Tuple[List[CandidateLeg], float, float]]:
    """Greedy beam search: O(beam_width * pool_size * leg_count)."""
    pw = SCORING_WEIGHTS["pool"]
    cw = SCORING_WEIGHTS["combo"]
    beam_width = int(pw["beam_width"])

    # Seed beams with individual legs
    seeds: List[Tuple[List[CandidateLeg], float]] = []
    for leg in pool:
        if leg.odds < MIN_LEG_ODDS:
            continue
        seeds.append(([leg], -leg_quality.get(leg, 0.0)))
    seeds.sort(key=lambda x: x[1])
    beam = seeds[:beam_width]

    for step in range(1, leg_count):
        next_beam: List[Tuple[List[CandidateLeg], float]] = []
        seen_keys: Set[frozenset] = set()
        for combo, _ in beam:
            combo_set = set(id(l) for l in combo)
            for leg in pool:
                if id(leg) in combo_set:
                    continue
                extended = combo + [leg]
                if not combo_valid(extended, c.require_unique_events):
                    continue
                # Deduplicate by leg identity
                key = frozenset((l.event_id, l.market_key, l.outcome, l.point) for l in extended)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                # Heuristic for partial combos: proximity to target + KB quality
                partial_quality = sum(leg_quality.get(l, 0.0) for l in extended)
                combo_odds = 1.0
                for l in extended:
                    combo_odds *= l.odds
                if c.target_multiplier and c.target_multiplier > 0:
                    target_for_step = c.target_multiplier ** ((step + 1) / leg_count)
                    proximity = abs(log(max(1.01, combo_odds)) - log(max(1.01, target_for_step)))
                else:
                    neutral = cw["neutral_anchor_base"] ** (step + 1)
                    proximity = abs(log(max(1.01, combo_odds)) - log(neutral))
                h_score = proximity - partial_quality * 0.3
                next_beam.append((extended, h_score))
        if not next_beam:
            break
        next_beam.sort(key=lambda x: x[1])
        beam = next_beam[:beam_width]

    # Final scoring of complete combos
    scored: List[Tuple[List[CandidateLeg], float, float]] = []
    for combo, _ in beam:
        if len(combo) != leg_count:
            continue
        if not combo_valid(combo, c.require_unique_events):
            continue
        s = score_combo(combo, c, leg_quality)
        scored.append((combo, s, combined_odds(combo)))
    return scored


def select_parlay(candidates: List[CandidateLeg], leg_count: int, c: ConstraintSpec) -> Tuple[List[CandidateLeg], List[str]]:
    notes: List[str] = []
    if leg_count <= 0:
        return [], ["Invalid leg count."]
    if not candidates:
        return [], ["No odds candidates available."]

    # ---- Hard pre-filter: remove legs that violate non-negotiable constraints ----
    # This prevents the solver from ever considering invalid legs.
    pre_filter_count = len(candidates)

    # require_total_corner_keys: only keep legs whose market key contains both "corner" and "total"
    if c.require_total_corner_keys:
        candidates = [
            leg for leg in candidates
            if "corner" in leg.market_key.lower() and "total" in leg.market_key.lower()
        ]

    # hard_exclude_groups: remove legs in excluded market groups
    if c.hard_exclude_groups:
        candidates = [
            leg for leg in candidates
            if market_group_from_key(leg.market_key) not in c.hard_exclude_groups
        ]

    # hard_include_groups: if set, only keep legs in included groups
    if c.hard_include_groups:
        candidates = [
            leg for leg in candidates
            if market_group_from_key(leg.market_key) in c.hard_include_groups
        ]

    # forbid_spread_keys: remove spread/handicap legs
    if c.forbid_spread_keys:
        candidates = [
            leg for leg in candidates
            if "spread" not in leg.market_key.lower() and "handicap" not in leg.market_key.lower()
        ]

    filtered_out = pre_filter_count - len(candidates)
    if filtered_out > 0:
        notes.append(f"Hard-filtered {filtered_out} legs that violate market constraints ({len(candidates)} remaining).")

    if not candidates:
        return [], ["No candidates remain after applying market constraints (hard filter)."]

    # Reduce search size while keeping odds diversity.
    by_bucket: Dict[Tuple[str, str], List[CandidateLeg]] = {}
    for row in sorted(candidates, key=lambda x: x.odds):
        k = (row.event_id, row.market_key)
        by_bucket.setdefault(k, [])
        by_bucket[k].append(row)

    target_leg_anchor = None
    if c.target_multiplier and c.target_multiplier > 0:
        target_leg_anchor = c.target_multiplier ** (1.0 / max(1, leg_count))
    if target_leg_anchor is None:
        target_leg_anchor = SCORING_WEIGHTS["pool"]["default_anchor"]

    pool: List[CandidateLeg] = []
    for rows in by_bucket.values():
        n = len(rows)
        if n <= 8:
            pool.extend(rows)
            continue
        # Pick representative low/mid/high plus lines near target leg odds.
        picks: List[CandidateLeg] = []
        picks.extend(rows[:2])               # low odds
        picks.extend(rows[-2:])              # high odds
        rows_by_anchor = sorted(rows, key=lambda x: abs(log(max(1.01, x.odds)) - log(max(1.01, target_leg_anchor))))
        picks.extend(rows_by_anchor[:4])     # around target odds
        seen = set()
        dedup: List[CandidateLeg] = []
        for p in picks:
            key = (p.event_id, p.market_key, p.outcome, p.point, p.bookmaker, p.odds)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(p)
        pool.extend(dedup)

    # Final cap by proximity-to-target odds + KB quality, not raw low odds.
    leg_quality: Dict[CandidateLeg, float] = {}
    for leg in pool:
        leg_quality[leg] = kb_leg_quality(leg, c.league)

    pw = SCORING_WEIGHTS["pool"]

    def leg_rank(leg: CandidateLeg) -> float:
        proximity = abs(log(max(1.01, leg.odds)) - log(max(1.01, target_leg_anchor)))
        quality_bonus = pw["quality_bonus"] * leg_quality.get(leg, 0.0)
        return proximity - quality_bonus

    pool = sorted(pool, key=leg_rank)[:pw["cap"]]

    if len(pool) < leg_count:
        return [], [f"Only {len(pool)} candidate legs available for {leg_count}-leg request."]

    total_combos = _comb(len(pool), leg_count)
    threshold = SCORING_WEIGHTS["pool"]["brute_force_threshold"]
    if total_combos <= threshold:
        scored = _brute_force_combos(pool, leg_count, c, leg_quality)
    else:
        scored = _beam_search_combos(pool, leg_count, c, leg_quality)
        notes.append(f"Used beam search ({len(pool)} candidates, {leg_count} legs, ~{total_combos:,} combos avoided).")

    if not scored:
        return [], ["No valid parlay combination found."]

    if c.target_multiplier and c.target_mode in {"around", "max", "min"}:
        tgt = c.target_multiplier
        if c.target_mode == "around":
            band = AROUND_TARGET_TOLERANCE * tgt
            in_band = [row for row in scored if abs(row[2] - tgt) <= band]
            if in_band:
                best = min(in_band, key=lambda x: x[1])
                return best[0], notes
            best = min(scored, key=lambda x: abs(log(x[2]) - log(tgt)))
            notes.append(
                f"No combo within ±{int(AROUND_TARGET_TOLERANCE*100)}% of target {tgt:.2f}x; returning closest at {best[2]:.2f}x."
            )
            return best[0], notes

        if c.target_mode == "max":
            feasible = [row for row in scored if row[2] <= tgt]
            if feasible:
                best = min(feasible, key=lambda x: x[1])
                return best[0], notes
            best = min(scored, key=lambda x: abs(x[2] - tgt))
            notes.append(f"No combo <= {tgt:.2f}x; returning closest at {best[2]:.2f}x.")
            return best[0], notes

        if c.target_mode == "min":
            feasible = [row for row in scored if row[2] >= tgt]
            if feasible:
                best = min(feasible, key=lambda x: x[1])
                return best[0], notes
            best = min(scored, key=lambda x: abs(x[2] - tgt))
            notes.append(f"No combo >= {tgt:.2f}x; returning closest at {best[2]:.2f}x.")
            return best[0], notes

    # default: best score
    best = min(scored, key=lambda x: x[1])
    return best[0], notes


def _kb_team_variants(team_name: str) -> List[str]:
    """Generate team name variants for KB lookup: exact → alias → fuzzy."""
    canonical = canonical_team_name(team_name)
    variants = list({team_name, canonical, team_name.lower(), canonical.lower(), team_name.title(), canonical.title()})
    # Add token-stripped variants (e.g., "AFC Bournemouth" → "Bournemouth")
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


def get_recent_team_fixture_rows(team_name: str, league: str, limit: int = 8) -> List[Dict]:
    col = get_collection_handle(create_if_missing=True)
    variants = _kb_team_variants(team_name)

    dedup: Dict[str, Dict] = {}
    for v in variants:
        where = build_where(league, extra_filters=[{"doc_type": "team_fixture"}, {"team": v}])
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
    rows = list(dedup.values())
    rows.sort(key=lambda x: ((x.get("meta") or {}).get("fixture_date") or ""), reverse=True)
    return rows[:limit]


def get_team_recent_stats(team_name: str, league: str, last_n: int = 6) -> Dict:
    cache_key = (league, canonical_team_name(team_name).lower(), int(last_n))
    cached = _team_recent_stats_cache.get(cache_key)
    if cached is not None:
        return cached

    rows = get_recent_team_fixture_rows(team_name, league, limit=max(last_n, 4))
    rows = rows[:last_n]
    metas = [(r.get("meta") or {}) for r in rows]

    stats = {
        "n": len(metas),
        "corners_for_avg": _avg([_numeric(m.get("corners_for")) for m in metas]),
        "corners_against_avg": _avg([_numeric(m.get("corners_against")) for m in metas]),
        "shots_for_avg": _avg([_numeric(m.get("shots_for")) for m in metas]),
        "sot_for_avg": _avg([_numeric(m.get("sot_for")) for m in metas]),
        "shots_against_avg": _avg([_numeric(m.get("shots_against")) for m in metas]),
        "sot_against_avg": _avg([_numeric(m.get("sot_against")) for m in metas]),
        "cards_avg": _avg([_numeric(m.get("cards_per_90_team")) for m in metas]),
        "control_avg": _avg([_numeric(m.get("control_index")) for m in metas]),
        "form_avg": _avg([_numeric(m.get("form_index_team")) for m in metas]),
        "fouls_avg": _avg([_numeric(m.get("fouls_per_90_team")) for m in metas]),
        "xg_for_avg": _avg([_numeric(m.get("xg_for")) for m in metas]),
    }
    _team_recent_stats_cache[cache_key] = stats
    return stats


def get_team_profile_snippets_for_events(events: List[Dict], league: str, per_team: int = 1) -> List[Dict]:
    out: List[Dict] = []
    teams: List[str] = []
    for ev in events:
        for t in [ev.get("home_team"), ev.get("away_team")]:
            if t and t not in teams:
                teams.append(t)
    for team in teams:
        doc = get_team_profile_doc(str(team), league)
        if not doc:
            continue
        out.append({"text": doc.get("text") or "", "meta": doc.get("meta") or {}})
    return out[: max(1, per_team * max(len(teams), 1))]


def get_team_fixture_snippets_for_events(events: List[Dict], league: str, per_team: int = 3) -> List[Dict]:
    out: List[Dict] = []
    teams: List[str] = []
    for ev in events:
        for t in [ev.get("home_team"), ev.get("away_team")]:
            if t and t not in teams:
                teams.append(t)
    for team in teams:
        out.extend(get_recent_team_fixture_rows(str(team), league, limit=per_team))
    return out


def get_player_snippets_for_events(events: List[Dict], league: str, per_team_profiles: int = 4, per_team_fixtures: int = 6) -> List[Dict]:
    col = get_collection_handle(create_if_missing=True)
    out_map: Dict[str, Dict] = {}

    teams: List[str] = []
    for ev in events:
        for t in [ev.get("home_team"), ev.get("away_team")]:
            if t and t not in teams:
                teams.append(str(t))

    for team in teams:
        variants = _kb_team_variants(team)

        picked_profiles = 0
        for tv in variants:
            where = build_where(league, extra_filters=[{"doc_type": "player_profile"}, {"team": tv}])
            res = col.get(where=where, include=["documents", "metadatas"], limit=per_team_profiles)
            for i, d, m in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
                out_map[str(i)] = {"id": i, "text": d or "", "meta": m or {}}
                picked_profiles += 1
            if picked_profiles >= per_team_profiles:
                break

        picked_fixtures = 0
        for tv in variants:
            where = build_where(league, extra_filters=[{"doc_type": "player_fixture"}, {"team": tv}])
            res = col.get(where=where, include=["documents", "metadatas"], limit=per_team_fixtures)
            for i, d, m in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
                out_map[str(i)] = {"id": i, "text": d or "", "meta": m or {}}
                picked_fixtures += 1
            if picked_fixtures >= per_team_fixtures:
                break

    return list(out_map.values())


def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def projected_corners(home_meta: Dict, away_meta: Dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Simple blended projection:
    - own corners_for tendency
    - opponent corners_against tendency
    """
    h_for = safe_float(home_meta.get("corners_pm"))
    h_opp_allow = safe_float(away_meta.get("corners_against_pm"))
    a_for = safe_float(away_meta.get("corners_pm"))
    a_opp_allow = safe_float(home_meta.get("corners_against_pm"))

    pw = SCORING_WEIGHTS["projection"]
    h_proj = None
    a_proj = None
    if h_for is not None and h_opp_allow is not None:
        h_proj = pw["corners_own"] * h_for + pw["corners_opp"] * h_opp_allow
    elif h_for is not None:
        h_proj = h_for
    elif h_opp_allow is not None:
        h_proj = h_opp_allow

    if a_for is not None and a_opp_allow is not None:
        a_proj = pw["corners_own"] * a_for + pw["corners_opp"] * a_opp_allow
    elif a_for is not None:
        a_proj = a_for
    elif a_opp_allow is not None:
        a_proj = a_opp_allow

    return h_proj, a_proj


def projected_cards(home_meta: Dict, away_meta: Dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Cards proxy from team cards_per_90 with a small aggression adjustment.
    """
    pw = SCORING_WEIGHTS["projection"]

    def proj(meta: Dict) -> Optional[float]:
        c = safe_float(meta.get("cards_per_90_team"))
        agg = safe_float(meta.get("aggression_index_norm"))
        if c is None and agg is None:
            return None
        if c is None:
            return agg * pw["cards_agg_scale"]
        if agg is None:
            return c
        return pw["cards_base"] * c + pw["cards_agg"] * (agg * pw["cards_agg_scale"])

    return proj(home_meta), proj(away_meta)


def projected_sot(home_meta: Dict, away_meta: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Blended SoT projection: own sot_for_pm vs opponent sot_against_pm."""
    pw = SCORING_WEIGHTS["projection_sot"]
    h_for = safe_float(home_meta.get("sot_for_pm"))
    h_opp_allow = safe_float(away_meta.get("sot_against_pm"))
    a_for = safe_float(away_meta.get("sot_for_pm"))
    a_opp_allow = safe_float(home_meta.get("sot_against_pm"))

    def blend(own, opp):
        if own is not None and opp is not None:
            return pw["own"] * own + pw["opp"] * opp
        return own if own is not None else opp

    return blend(h_for, h_opp_allow), blend(a_for, a_opp_allow)


def projected_total_sot(home: str, away: str, league: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    pw = SCORING_WEIGHTS["projection"]
    hm = get_team_profile_meta(home, league)
    am = get_team_profile_meta(away, league)
    h_proj, a_proj = projected_sot(hm, am)
    season_total = (h_proj + a_proj) if (h_proj is not None and a_proj is not None) else None

    hr = get_team_recent_stats(home, league, last_n=6)
    ar = get_team_recent_stats(away, league, last_n=6)
    h_recent = hr.get("sot_for_avg")
    a_recent = ar.get("sot_for_avg")
    recent_total = (h_recent + a_recent) if (h_recent is not None and a_recent is not None) else None

    if season_total is not None and recent_total is not None:
        return (pw["blend_season"] * season_total + pw["blend_recent"] * recent_total), season_total, recent_total
    if season_total is not None:
        return season_total, season_total, None
    if recent_total is not None:
        return recent_total, None, recent_total
    return None, None, None


def leg_label(leg: CandidateLeg) -> str:
    g = market_group_from_key(leg.market_key)
    if leg.point is None:
        return f"{leg.outcome} ({leg.market_key}/{g})"
    if g == "spreads":
        return f"{leg.outcome} {leg.point:+g} ({leg.market_key}/{g})"
    return f"{leg.outcome} {leg.point:g} ({leg.market_key}/{g})"


def leg_evidence(leg: CandidateLeg, league: str) -> List[str]:
    hm = get_team_profile_meta(leg.home_team, league)
    am = get_team_profile_meta(leg.away_team, league)
    hr = get_team_recent_stats(leg.home_team, league, last_n=6)
    ar = get_team_recent_stats(leg.away_team, league, last_n=6)
    heur_groups = heuristic_groups_for_event(hm, am)
    lines: List[str] = []

    def v(meta: Dict, key: str) -> Optional[float]:
        x = meta.get(key)
        try:
            return float(x)
        except Exception:
            return None

    g = market_group_from_key(leg.market_key)
    if g == "corners":
        h_for, h_against = v(hm, "corners_pm"), v(hm, "corners_against_pm")
        a_for, a_against = v(am, "corners_pm"), v(am, "corners_against_pm")
        h_edge, a_edge = v(hm, "corner_edge_pm"), v(am, "corner_edge_pm")
        h_ctrl, a_ctrl = v(hm, "control_index"), v(am, "control_index")
        h_proj, a_proj = projected_corners(hm, am)
        if h_for is not None and h_against is not None:
            lines.append(f"{leg.home_team} corners for/against: {h_for:.2f}/{h_against:.2f} per match.")
        if a_for is not None and a_against is not None:
            lines.append(f"{leg.away_team} corners for/against: {a_for:.2f}/{a_against:.2f} per match.")
        if h_proj is not None and a_proj is not None:
            total_proj = h_proj + a_proj
            side = f"{leg.home_team} {h_proj:.2f} vs {leg.away_team} {a_proj:.2f}"
            lines.append(f"Projected corners: {side} (combined {total_proj:.2f}).")
            k = leg.market_key.lower()
            if leg.point is not None and "total" in k:
                direction = leg.outcome.lower()
                if "over" in direction:
                    lines.append(f"Line fit: projected {total_proj:.2f} vs over {leg.point:g}.")
                elif "under" in direction:
                    lines.append(f"Line fit: projected {total_proj:.2f} vs under {leg.point:g}.")
        if h_edge is not None and a_edge is not None:
            lines.append(f"Corner edge: {leg.home_team} {h_edge:+.2f}, {leg.away_team} {a_edge:+.2f} per match.")
        if h_ctrl is not None and a_ctrl is not None:
            lines.append(f"Control index: {leg.home_team} {h_ctrl:.2f}, {leg.away_team} {a_ctrl:.2f}.")
        h_recent = hr.get("corners_for_avg")
        a_recent = ar.get("corners_for_avg")
        if h_recent is not None and a_recent is not None:
            lines.append(f"Recent 6-match corners for: {leg.home_team} {h_recent:.2f}, {leg.away_team} {a_recent:.2f}.")
    elif g == "totals":
        h_gf, a_gf = v(hm, "goals_for_pm"), v(am, "goals_for_pm")
        h_ctrl, a_ctrl = v(hm, "control_index"), v(am, "control_index")
        h_dom, a_dom = v(hm, "dominance_index"), v(am, "dominance_index")
        h_edge, a_edge = v(hm, "corner_edge_pm"), v(am, "corner_edge_pm")
        if h_gf is not None and a_gf is not None:
            lines.append(f"Goal baseline: {leg.home_team} {h_gf:.2f}/match, {leg.away_team} {a_gf:.2f}/match.")
        if h_ctrl is not None and a_ctrl is not None:
            lines.append(f"Control index: {leg.home_team} {h_ctrl:.2f}, {leg.away_team} {a_ctrl:.2f}.")
        if h_dom is not None and a_dom is not None:
            lines.append(f"Dominance index: {leg.home_team} {h_dom:.2f}, {leg.away_team} {a_dom:.2f}.")
        if h_edge is not None and a_edge is not None:
            lines.append(f"Corner edge proxy: {leg.home_team} {h_edge:+.2f}, {leg.away_team} {a_edge:+.2f}.")
        h_sot, a_sot = hr.get("sot_for_avg"), ar.get("sot_for_avg")
        if h_sot is not None and a_sot is not None:
            lines.append(f"Recent 6-match SoT for: {leg.home_team} {h_sot:.2f}, {leg.away_team} {a_sot:.2f}.")
    elif g == "spreads":
        h_form, a_form = v(hm, "form_index_team"), v(am, "form_index_team")
        h_ctrl, a_ctrl = v(hm, "control_index"), v(am, "control_index")
        h_dom, a_dom = v(hm, "dominance_index"), v(am, "dominance_index")
        h_edge, a_edge = v(hm, "corner_edge_pm"), v(am, "corner_edge_pm")
        if h_form is not None and a_form is not None:
            lines.append(f"Form index: {leg.home_team} {h_form:.2f} vs {leg.away_team} {a_form:.2f}.")
        if h_ctrl is not None and a_ctrl is not None:
            lines.append(f"Control index: {leg.home_team} {h_ctrl:.2f} vs {leg.away_team} {a_ctrl:.2f}.")
        if h_dom is not None and a_dom is not None:
            lines.append(f"Dominance index: {leg.home_team} {h_dom:.2f} vs {leg.away_team} {a_dom:.2f}.")
        if h_edge is not None and a_edge is not None:
            lines.append(f"Corner edge: {leg.home_team} {h_edge:+.2f} vs {leg.away_team} {a_edge:+.2f}.")
        h_sot, a_sot = hr.get("sot_for_avg"), ar.get("sot_for_avg")
        if h_sot is not None and a_sot is not None:
            lines.append(f"Recent 6-match SoT edge: {leg.home_team} {h_sot:.2f} vs {leg.away_team} {a_sot:.2f}.")
    elif g == "moneyline":
        h_form, a_form = v(hm, "form_index_team"), v(am, "form_index_team")
        h_dom, a_dom = v(hm, "dominance_index"), v(am, "dominance_index")
        if h_form is not None and a_form is not None:
            lines.append(f"Form index: {leg.home_team} {h_form:.2f} vs {leg.away_team} {a_form:.2f}.")
        if h_dom is not None and a_dom is not None:
            lines.append(f"Dominance index: {leg.home_team} {h_dom:.2f} vs {leg.away_team} {a_dom:.2f}.")
        h_recent_form, a_recent_form = hr.get("form_avg"), ar.get("form_avg")
        if h_recent_form is not None and a_recent_form is not None:
            lines.append(f"Recent 6-match form index: {leg.home_team} {h_recent_form:.2f} vs {leg.away_team} {a_recent_form:.2f}.")
    else:
        h_ctrl, a_ctrl = v(hm, "control_index"), v(am, "control_index")
        if h_ctrl is not None and a_ctrl is not None:
            lines.append(f"Control profile: {leg.home_team} {h_ctrl:.2f}, {leg.away_team} {a_ctrl:.2f}.")

    if g in heur_groups:
        lines.append(f"Heuristic slate also favors {g} for this fixture.")

    if not lines:
        lines.append("Team-profile stats unavailable for this fixture in local KB.")
    # prioritize multi-signal support
    return lines[:3]


def combined_odds(legs: List[CandidateLeg]) -> float:
    p = 1.0
    for leg in legs:
        p *= leg.odds
    return p


def summarize_constraints(c: ConstraintSpec, available_groups: Set[str], selected: List[CandidateLeg]) -> Tuple[List[str], List[str]]:
    applied: List[str] = []
    unmet: List[str] = []

    applied.append("global rule: no moneyline + spread mix")
    applied.append("global rule: one leg per market family per fixture")
    applied.append(f"global rule: min leg odds >= {MIN_LEG_ODDS:.2f}")

    if c.per_match_mode:
        applied.append("one leg per fixture")
    elif c.require_unique_events:
        applied.append("unique fixtures")

    if c.target_multiplier is not None and c.target_mode != "none":
        combo = combined_odds(selected)
        tgt = c.target_multiplier
        if c.target_mode == "max":
            if combo <= tgt:
                applied.append(f"target max {tgt:.2f}x met")
            else:
                unmet.append(f"target max {tgt:.2f}x not met (selected {combo:.2f}x)")
        elif c.target_mode == "min":
            if combo >= tgt:
                applied.append(f"target min {tgt:.2f}x met")
            else:
                unmet.append(f"target min {tgt:.2f}x not met (selected {combo:.2f}x)")
        else:
            rel_dev = abs(combo - tgt) / max(tgt, 1e-9)
            if rel_dev <= AROUND_TARGET_TOLERANCE:
                applied.append(f"target around {tgt:.2f}x met")
            else:
                unmet.append(
                    f"target around {tgt:.2f}x not met (selected {combo:.2f}x, deviation {rel_dev*100:.1f}%)"
                )

    for req in c.hard_include_groups:
        if req not in available_groups:
            unmet.append(f"requested '{req}' markets are unavailable from current odds feed")
        elif all(market_group_from_key(x.market_key) != req for x in selected):
            unmet.append(f"requested '{req}' not represented in selected legs")
        else:
            applied.append(f"{req}-only preference honored")

    for grp, need in c.required_group_counts.items():
        have = sum(1 for x in selected if market_group_from_key(x.market_key) == grp)
        if have >= need:
            applied.append(f"{grp} legs {have}/{need}")
        else:
            unmet.append(f"required {need} {grp} leg(s), got {have}")

    for ex in c.hard_exclude_groups:
        if any(market_group_from_key(x.market_key) == ex for x in selected):
            unmet.append(f"exclude '{ex}' was violated by selected legs")
        else:
            applied.append(f"excluded '{ex}'")

    if c.forbid_spread_keys:
        if any(("spread" in x.market_key.lower()) or ("handicap" in x.market_key.lower()) for x in selected):
            unmet.append("no spreads requested, but spread/handicap market key present")
        else:
            applied.append("no spread keys")

    if c.require_total_corner_keys:
        bad = [
            x for x in selected
            if (("corner" not in x.market_key.lower()) or ("total" not in x.market_key.lower()))
        ]
        if bad:
            unmet.append("total-corners-only requested, but non-total-corner key selected")
        else:
            applied.append("total-corners-only")

    return applied, unmet


def parse_llm_leg_evidence(text: str, leg_count: int) -> Tuple[Dict[int, List[str]], Optional[str]]:
    by_leg: Dict[int, List[str]] = {}
    why_lines: List[str] = []
    current_leg: Optional[int] = None
    in_why = False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"LEG\s+(\d+)\s*:", line, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= leg_count:
                current_leg = idx
                in_why = False
                by_leg.setdefault(idx, [])
            else:
                current_leg = None
            continue
        if re.match(r"WHY\s*:", line, flags=re.IGNORECASE):
            in_why = True
            current_leg = None
            continue
        if in_why:
            why_lines.append(line.lstrip("- ").strip())
            continue
        if current_leg is not None and line.startswith("-"):
            by_leg.setdefault(current_leg, []).append(line.lstrip("- ").strip())
    why = " ".join(x for x in why_lines if x).strip() or None
    return by_leg, why


def build_reasoning_messages(
    user_q: str,
    selected: List[CandidateLeg],
    profile_docs: List[Dict],
    team_fixture_docs: List[Dict],
    player_docs: List[Dict],
    chat_context: str,
) -> List[Dict]:
    selected_lines = []
    for i, leg in enumerate(selected, start=1):
        bm = f" | bookmaker={leg.bookmaker}" if leg.bookmaker else ""
        point = "none" if leg.point is None else f"{leg.point:g}"
        selected_lines.append(
            f"LEG {i}: fixture={leg.fixture} | market_key={leg.market_key} | outcome={leg.outcome} | point={point} | odds={leg.odds:.2f}{bm}"
        )
    selected_block = "\n".join(selected_lines)
    profile_text = "\n\n".join((d.get("text") or "") for d in profile_docs[:10])
    team_fx_text = "\n\n".join((d.get("text") or "") for d in team_fixture_docs[:14])
    player_text = "\n\n".join((d.get("text") or "") for d in player_docs[:18])

    system_msg = (
        "You are a football betting analyst.\n"
        "You must explain the PROVIDED fixed legs only.\n"
        "Do not add, remove, reorder, or replace any leg.\n"
        "Use only numbers from context.\n"
        "Avoid single-match recency logic unless framed as a small supporting point.\n"
        "Use multi-signal support (form/control/dominance/corners/cards/shots/sot where available).\n"
        "If player data is sparse, still give the best available evidence from provided player/team context.\n"
        "Return EXACTLY in this format:\n"
        "LEG 1:\n"
        "- <evidence line>\n"
        "- <evidence line>\n"
        "LEG 2:\n"
        "- ...\n"
        "WHY:\n"
        "- <2-3 concise lines on why this parlay structure makes sense>\n"
    )
    user_msg = (
        f"User query:\n{user_q}\n\n"
        f"Recent chat context:\n{chat_context}\n\n"
        f"Fixed selected legs:\n{selected_block}\n\n"
        f"Team profile context:\n{profile_text or 'None'}\n\n"
        f"Recent team fixture context:\n{team_fx_text or 'None'}\n\n"
        f"Player context:\n{player_text or 'None'}\n"
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def get_llm_reasoning(
    user_q: str,
    selected: List[CandidateLeg],
    events: List[Dict],
    league: str,
) -> Tuple[Dict[int, List[str]], Optional[str], Optional[str]]:
    if not client:
        return {}, None, "OPENAI_API_KEY not set; using deterministic evidence."
    selected_event_ids = {x.event_id for x in selected}
    selected_events = [ev for ev in events if str(ev.get("id") or "") in selected_event_ids]
    profile_docs = get_team_profile_snippets_for_events(selected_events, league, per_team=1)
    team_fixture_docs = get_team_fixture_snippets_for_events(selected_events, league, per_team=3)
    player_docs = get_player_snippets_for_events(selected_events, league, per_team_profiles=4, per_team_fixtures=6)
    messages = build_reasoning_messages(
        user_q,
        selected,
        profile_docs,
        team_fixture_docs,
        player_docs,
        chat_context=render_recent_chat_context(max_turns=3),
    )
    try:
        resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
        text = (resp.choices[0].message.content or "").strip()
        evidence_map, why = parse_llm_leg_evidence(text, leg_count=len(selected))
        if not evidence_map:
            return {}, None, "LLM explanation parse failed; using deterministic evidence."
        return evidence_map, why, None
    except Exception as exc:
        return {}, None, f"LLM explanation failed: {exc}"


def llm_validate_selection(
    selected: List[CandidateLeg],
    league: str,
) -> Tuple[bool, Optional[str]]:
    """
    Optional LLM-in-the-loop validation. Gated by RAG_LLM_VALIDATION env var.
    Returns (passed, concern_text). If validation is disabled or unavailable, always passes.
    """
    if not env_bool("RAG_LLM_VALIDATION", default=False):
        return True, None
    if not client:
        return True, None

    legs_text = []
    for i, leg in enumerate(selected, start=1):
        point = "none" if leg.point is None else f"{leg.point:g}"
        legs_text.append(
            f"LEG {i}: {leg.fixture} | {leg.market_key} | {leg.outcome} | point={point} | odds={leg.odds:.2f}"
        )

    system_msg = (
        "You are a betting validation assistant. Review the proposed parlay legs below.\n"
        "Flag ONLY clear logical issues:\n"
        "- Contradictory legs (e.g., same-match Over and Under on same line)\n"
        "- Correlated exposure that compounds risk (e.g., two legs on the same team outcome)\n"
        "- Clearly unrealistic lines given the market context\n"
        "If all legs look reasonable, respond with exactly: PASS\n"
        "Otherwise respond with: CONCERN: <1-2 sentence explanation>\n"
        "Do NOT second-guess the statistical model or odds values."
    )
    user_msg = f"League: {league}\nCombined odds: {combined_odds(selected):.2f}x\n\n" + "\n".join(legs_text)

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=150,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.upper().startswith("PASS"):
            return True, None
        concern = text.replace("CONCERN:", "").strip() if "CONCERN" in text.upper() else text
        return False, concern
    except Exception:
        return True, None


def render_parlay(
    user_q: str,
    c: ConstraintSpec,
    events: List[Dict],
    selected: List[CandidateLeg],
    notes: List[str],
    llm_evidence: Optional[Dict[int, List[str]]] = None,
    llm_why: Optional[str] = None,
) -> str:
    combo = combined_odds(selected)
    available_groups = available_market_groups(build_candidates(events))
    applied, unmet = summarize_constraints(c, available_groups, selected)

    lines: List[str] = []
    lines.append("Parlay recommendation:")
    for i, leg in enumerate(selected, start=1):
        bm = f" ({leg.bookmaker})" if leg.bookmaker else ""
        lines.append(f"- Leg {i}: {leg.fixture} | {leg_label(leg)} @ {leg.odds:.2f}{bm}")
        ev_lines = (llm_evidence or {}).get(i) or leg_evidence(leg, c.league)
        for ev in ev_lines[:3]:
            lines.append(f"  Evidence: {ev}")

    lines.append(f"- Estimated combined odds: {combo:.2f}x")
    if llm_why:
        lines.append(f"- Why this parlay: {llm_why}")
    lines.append("- Constraint report:")
    lines.append(f"  Applied: {('; '.join(applied) if applied else 'none')}")
    lines.append(f"  Unmet: {('; '.join(unmet) if unmet else 'none')}")

    if notes:
        lines.append("- Notes:")
        for n in notes:
            lines.append(f"  {n}")

    # User-friendly fallback guidance, no dead-end.
    if unmet:
        lines.append(
            "- Fallback behavior: kept best possible legs from available markets instead of returning no answer."
        )

    return "\n".join(lines)


def is_schedule_query(user_q: str) -> bool:
    q = user_q.lower()
    schedule_terms = ["what matches", "fixtures", "games are on", "matches are on", "schedule", "who plays"]
    return any(t in q for t in schedule_terms)


def is_capability_query(user_q: str) -> bool:
    q = user_q.lower()
    return bool(re.search(r"\b(available|availability|have|offer|supported)\b", q) and re.search(r"\b(corner|card|booking|spread|total|moneyline|ml|h2h)\b", q))


def requested_groups_from_query(user_q: str) -> Set[str]:
    q = user_q.lower()
    out: Set[str] = set()
    if re.search(r"\bcorner", q):
        out.add("corners")
    if re.search(r"\b(card|booking)", q):
        out.add("cards")
    if re.search(r"\bspread|handicap", q):
        out.add("spreads")
    if re.search(r"\btotal|over|under", q):
        out.add("totals")
    if re.search(r"\bmoneyline\b|\bmoney\s*line\b|\bml\b|\bh2h\b", q):
        out.add("moneyline")
    return out


def filter_events_by_mentioned_teams(user_q: str, events: List[Dict]) -> List[Dict]:
    q = re.sub(r"[^a-z0-9\s]", " ", user_q.lower())
    mentions: Set[str] = set()
    for ev in events:
        for t in [ev.get("home_team"), ev.get("away_team")]:
            if not t:
                continue
            aliases = team_name_aliases(str(t))
            if any(a and (a in q) for a in aliases):
                mentions.add(str(t))
    if not mentions:
        return events
    out = [ev for ev in events if (ev.get("home_team") in mentions) or (ev.get("away_team") in mentions)]
    return out or events


def query_mentions_any_event_team(user_q: str, events: List[Dict]) -> bool:
    q = re.sub(r"[^a-z0-9\s]", " ", user_q.lower())
    for ev in events:
        for t in [ev.get("home_team"), ev.get("away_team")]:
            if not t:
                continue
            aliases = team_name_aliases(str(t))
            if any(a and (a in q) for a in aliases):
                return True
    return False


def extract_mentioned_event_teams(user_q: str, events: List[Dict]) -> Set[str]:
    q = re.sub(r"[^a-z0-9\s]", " ", user_q.lower())
    out: Set[str] = set()
    for ev in events:
        for t in [ev.get("home_team"), ev.get("away_team")]:
            if not t:
                continue
            aliases = team_name_aliases(str(t))
            if any(a and (a in q) for a in aliases):
                out.add(str(t))
    return out


def is_singular_fixture_request(user_q: str) -> bool:
    q = user_q.lower()
    if re.search(r"\b(games|matches|fixtures)\b", q):
        return False
    if re.search(r"\b(the|this|that)\s+(game|match)\b", q):
        return True
    if re.search(r"\bfor\s+the\s+.+\s+game\b", q):
        return True
    return False


def event_sort_key(ev: Dict) -> str:
    return str(ev.get("commence_time") or "")


def is_parlay_query(user_q: str) -> bool:
    q = user_q.lower()
    return "parlay" in q or "legs" in q or "bet" in q


def wants_totals_only_parlay(user_q: str) -> bool:
    q = user_q.lower()
    if not is_parlay_query(user_q):
        return False
    if "totals only" in q:
        return True
    if re.search(r"parlay.*\b(goal|goals|xg|total|totals)\b", q) and "total" in q:
        return True
    if re.search(r"parlay.*\b(corner|corners)\b", q) and "total" in q:
        return True
    if re.search(r"parlay.*\b(card|cards|booking|bookings)\b", q) and "total" in q:
        return True
    return False


def is_comparison_query(user_q: str) -> bool:
    q = user_q.lower()
    return bool(
        re.search(r"(which team|who)\s+.*(more|higher|most)", q)
        or re.search(r"for all.*games", q)
    )


def wants_deep_explanation(user_q: str) -> bool:
    q = user_q.lower()
    return bool(
        re.search(r"\b(why|reason|reasoning|explain|explanation)\b", q)
        or re.search(r"\b(solid|detailed|deep|thorough)\b", q)
    )


def requested_stat_group(user_q: str) -> Optional[str]:
    q = user_q.lower()
    if "goal" in q or re.search(r"\bxg\b", q):
        return "goals"
    if "corner" in q:
        return "corners"
    if "card" in q or "booking" in q:
        return "cards"
    if re.search(r"\b(shot|sot)\b", q) or "shots on target" in q:
        return "sot"
    return None


def is_totals_line_query(user_q: str) -> bool:
    q = user_q.lower()
    has_stat = bool(re.search(r"\b(goal|xg|corner|card|booking|shot|sot)s?\b", q) or "shots on target" in q)
    asks_total_or_line = bool(
        re.search(r"\b(how many|total|totals|line|which line|what line|should i take|take)\b", q)
        or re.search(r"\b(over|under)\b", q)
    )
    return has_stat and asks_total_or_line


def is_player_prop_query(user_q: str) -> bool:
    q = user_q.lower()
    # "shots on target" / "sot" can be team-level (comparison / totals) or
    # player-level.  Only treat as player prop when there is NO team-level
    # signal (comparison or totals patterns).
    sot_hit = bool(re.search(r"\bshots?\s+on\s+target\b|\bsot\b", q))
    team_level_signal = bool(
        re.search(r"\b(which team|for all|for every|for each)\b", q)
        or re.search(r"\b(over|under|o/u|total|totals|line)\b", q)
        or re.search(r"\b(more|higher|most|winner)\b", q)
    )
    if sot_hit and team_level_signal:
        # Suppress the player-prop classification; let comparison / totals
        # handlers pick it up instead.
        sot_hit = False
    has_player_keyword = bool(
        re.search(
            r"\b(player|goalscorer|goal\s*scorer|anytime|first\s*scorer"
            r"|assist[s]?"
            r"|who\s+scores|who\s+gets?\s+booked|who\s+gets?\s+carded"
            r"|player\s+prop|player\s+card|player\s+booking"
            r"|which\s+player)",
            q,
        )
    )
    return has_player_keyword or sot_hit


def projected_total_corners(home: str, away: str, league: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    pw = SCORING_WEIGHTS["projection"]
    hm = get_team_profile_meta(home, league)
    am = get_team_profile_meta(away, league)
    h_proj, a_proj = projected_corners(hm, am)
    season_total = (h_proj + a_proj) if (h_proj is not None and a_proj is not None) else None

    hr = get_team_recent_stats(home, league, last_n=6)
    ar = get_team_recent_stats(away, league, last_n=6)
    h_recent = hr.get("corners_for_avg")
    a_recent = ar.get("corners_for_avg")
    recent_total = (h_recent + a_recent) if (h_recent is not None and a_recent is not None) else None

    if season_total is not None and recent_total is not None:
        return (pw["blend_season"] * season_total + pw["blend_recent"] * recent_total), season_total, recent_total
    if season_total is not None:
        return season_total, season_total, None
    if recent_total is not None:
        return recent_total, None, recent_total
    return None, None, None


def projected_total_cards(home: str, away: str, league: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    pw = SCORING_WEIGHTS["projection"]
    hm = get_team_profile_meta(home, league)
    am = get_team_profile_meta(away, league)
    h_proj, a_proj = projected_cards(hm, am)
    season_total = (h_proj + a_proj) if (h_proj is not None and a_proj is not None) else None

    hr = get_team_recent_stats(home, league, last_n=6)
    ar = get_team_recent_stats(away, league, last_n=6)
    h_recent = hr.get("cards_avg")
    a_recent = ar.get("cards_avg")
    recent_total = (h_recent + a_recent) if (h_recent is not None and a_recent is not None) else None

    if season_total is not None and recent_total is not None:
        return (pw["blend_season"] * season_total + pw["blend_recent"] * recent_total), season_total, recent_total
    if season_total is not None:
        return season_total, season_total, None
    if recent_total is not None:
        return recent_total, None, recent_total
    return None, None, None


def projected_total_goals(home: str, away: str, league: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    pw = SCORING_WEIGHTS["projection"]
    hm = get_team_profile_meta(home, league)
    am = get_team_profile_meta(away, league)

    h_g = safe_float(hm.get("goals_for_pm"))
    a_g = safe_float(am.get("goals_for_pm"))
    season_total = (h_g + a_g) if (h_g is not None and a_g is not None) else None

    hr = get_team_recent_stats(home, league, last_n=6)
    ar = get_team_recent_stats(away, league, last_n=6)
    h_xg = hr.get("xg_for_avg")
    a_xg = ar.get("xg_for_avg")
    recent_total = (h_xg + a_xg) if (h_xg is not None and a_xg is not None) else None

    if season_total is not None and recent_total is not None:
        return (pw["blend_season"] * season_total + pw["blend_recent"] * recent_total), season_total, recent_total
    if season_total is not None:
        return season_total, season_total, None
    if recent_total is not None:
        return recent_total, None, recent_total
    return None, None, None


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
            elif stat_group == "cards":
                if (("card" not in k) and ("booking" not in k)) or ("total" not in k):
                    continue
            elif stat_group == "goals":
                # Match totals goals markets. Avoid corners/cards/teams/goalscorer props.
                if "total" not in k:
                    continue
                if "corner" in k or "card" in k or "booking" in k:
                    continue
                if "team_total" in k or "player" in k or "goalscorer" in k:
                    continue
                if "shot" in k:
                    continue
            elif stat_group == "sot":
                if ("shot" not in k) or ("total" not in k):
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


def choose_best_total_line(options: List[Dict], projection_total: float) -> Optional[Dict]:
    if not options:
        return None
    tw = SCORING_WEIGHTS["totals_line"]

    filtered: List[Dict] = []
    for opt in options:
        odds = safe_float(opt.get("odds"))
        if odds is None or odds < tw["min_odds_hard"]:
            continue
        filtered.append(opt)
    if not filtered:
        return None

    primary = [o for o in filtered if "alternate" not in str(o.get("market_key") or "").lower()]
    options_use = primary if primary else filtered

    best = None
    best_score = None
    for opt in options_use:
        side = str(opt.get("side") or "")
        point = safe_float(opt.get("point"))
        odds = safe_float(opt.get("odds"))
        if point is None or odds is None:
            continue
        edge = projection_total - point if side == "over" else point - projection_total

        dist = abs(projection_total - point)
        proximity_penalty = dist * tw["proximity_penalty"]
        edge_term = edge * tw["edge_weight"]

        if odds < tw["min_odds_preferred"]:
            odds_term = -tw["short_price_penalty"] * (tw["min_odds_preferred"] - odds)
        elif tw["ideal_odds_low"] <= odds <= tw["ideal_odds_high"]:
            odds_term = tw["ideal_range_bonus"] * (odds - 1.0)
        else:
            odds_term = tw["outside_ideal_bonus"] * (odds - 1.0)

        far_penalty = 0.0
        if dist >= tw["far_threshold"]:
            far_penalty = (dist - tw["far_threshold"]) * tw["far_penalty"]

        score = edge_term - proximity_penalty + odds_term - far_penalty
        if best is None or score > float(best_score):
            best = opt
            best_score = score
    return best


def confidence_from_edge(edge: float) -> str:
    cw = SCORING_WEIGHTS["confidence"]
    if edge >= cw["edge_high"]:
        return "high"
    if edge >= cw["edge_medium"]:
        return "medium"
    return "low"


def render_totals_line_answer(user_q: str, league: str, events: List[Dict]) -> str:
    group = requested_stat_group(user_q)
    if group not in {"goals", "corners", "cards", "sot"}:
        return "I can estimate totals lines for goals/corners/cards/shots-on-target when the stat type is clear."

    title = {"goals": "Goals Totals", "corners": "Corner Totals", "cards": "Card Totals", "sot": "SoT Totals"}[group]
    lines: List[str] = [f"{title} advice by fixture:"]

    for ev in events:
        home = str(ev.get("home_team") or "Home")
        away = str(ev.get("away_team") or "Away")
        fixture = f"{home} vs {away}"

        if group == "goals":
            proj_total, season_total, recent_total = projected_total_goals(home, away, league)
        elif group == "corners":
            proj_total, season_total, recent_total = projected_total_corners(home, away, league)
        elif group == "sot":
            proj_total, season_total, recent_total = projected_total_sot(home, away, league)
        else:
            proj_total, season_total, recent_total = projected_total_cards(home, away, league)

        if proj_total is None:
            lines.append(f"- {fixture}: insufficient profile data to project totals.")
            continue

        options = extract_total_line_options(ev, group)
        best = choose_best_total_line(options, proj_total)
        lines.append(f"- {fixture}: projected total {proj_total:.2f}.")

        if season_total is not None:
            lines.append(f"  Reasoning: Season-model total {season_total:.2f}.")
        if recent_total is not None:
            lines.append(f"  Reasoning: Recent 6-match total {recent_total:.2f}.")

        if best:
            side = str(best.get("side") or "").title()
            point = float(best.get("point"))
            odds = float(best.get("odds"))
            edge = (proj_total - point) if side.lower() == "over" else (point - proj_total)
            conf = confidence_from_edge(edge)
            lines.append(
                f"  Recommended line: {side} {point:g} @ {odds:.2f} ({best.get('bookmaker')}, {best.get('market_key')})."
            )
            lines.append(f"  Reasoning: Model edge {edge:+.2f} vs line ({conf} confidence).")
        else:
            # Fallback when book lines are unavailable: still provide actionable threshold.
            anchor = round(proj_total * 2.0) / 2.0
            side = "Over" if proj_total >= anchor else "Under"
            lines.append(
                f"  Recommended line: {side} {anchor:g} (no realistic-priced full-game line available in current feed; model-only recommendation)."
            )

    lines.append("Method: deterministic totals projection from local KB + available full-game market lines.")
    return "\n".join(lines)


def comparison_signals_cards(home: str, away: str, league: str) -> Tuple[Optional[str], List[str], Optional[float], Optional[float]]:
    hm = get_team_profile_meta(home, league)
    am = get_team_profile_meta(away, league)
    hr = get_team_recent_stats(home, league, last_n=6)
    ar = get_team_recent_stats(away, league, last_n=6)

    h_cards = safe_float(hm.get("cards_per_90_team"))
    a_cards = safe_float(am.get("cards_per_90_team"))
    h_fouls = safe_float(hm.get("fouls_per_90_team"))
    a_fouls = safe_float(am.get("fouls_per_90_team"))
    h_agg = safe_float(hm.get("aggression_index_norm"))
    a_agg = safe_float(am.get("aggression_index_norm"))
    h_ctrl = safe_float(hm.get("control_index"))
    a_ctrl = safe_float(am.get("control_index"))

    hr_cards = hr.get("cards_avg")
    ar_cards = ar.get("cards_avg")
    hr_fouls = hr.get("fouls_avg")
    ar_fouls = ar.get("fouls_avg")

    # Base deterministic projection from team profile.
    h_proj, a_proj = projected_cards(hm, am)
    if h_proj is None and h_cards is not None:
        h_proj = h_cards
    if a_proj is None and a_cards is not None:
        a_proj = a_cards

    # Composite card risk to enrich reasoning (kept deterministic).
    cw = SCORING_WEIGHTS["comparison_cards"]

    def risk(cards, fouls, agg, recent_cards, recent_fouls, opp_control) -> Optional[float]:
        vals = [cards, fouls, agg, recent_cards, recent_fouls, opp_control]
        if all(v is None for v in vals):
            return None
        c = cards if cards is not None else 0.0
        f = fouls if fouls is not None else 0.0
        a = agg if agg is not None else 0.0
        rc = recent_cards if recent_cards is not None else 0.0
        rf = recent_fouls if recent_fouls is not None else 0.0
        oc = opp_control if opp_control is not None else 0.0
        return (cw["cards"] * c) + (cw["fouls"] * f) + (cw["agg"] * a * cw["agg_scale"]) + (cw["recent_cards"] * rc) + (cw["recent_fouls"] * rf) + (cw["opp_control"] * oc)

    h_risk = risk(h_cards, h_fouls, h_agg, hr_cards, hr_fouls, a_ctrl)
    a_risk = risk(a_cards, a_fouls, a_agg, ar_cards, ar_fouls, h_ctrl)

    pick = None
    h_score = h_risk if h_risk is not None else h_proj
    a_score = a_risk if a_risk is not None else a_proj
    if h_score is not None and a_score is not None:
        pick = home if h_score >= a_score else away
    elif h_score is not None:
        pick = home
    elif a_score is not None:
        pick = away

    lines: List[str] = []
    if h_proj is not None and a_proj is not None:
        lines.append(f"Profile projection: {home} {h_proj:.2f} vs {away} {a_proj:.2f} cards.")
    if h_cards is not None and a_cards is not None:
        lines.append(f"Season cards/90: {home} {h_cards:.2f} vs {away} {a_cards:.2f}.")
    if h_fouls is not None and a_fouls is not None:
        lines.append(f"Season fouls/90: {home} {h_fouls:.2f} vs {away} {a_fouls:.2f}.")
    if h_agg is not None and a_agg is not None:
        lines.append(f"Aggression index: {home} {h_agg:.2f} vs {away} {a_agg:.2f}.")
    if hr_cards is not None and ar_cards is not None:
        lines.append(f"Recent 6-match cards/90 proxy: {home} {hr_cards:.2f} vs {away} {ar_cards:.2f}.")
    if h_risk is not None and a_risk is not None:
        lines.append(f"Composite card risk: {home} {h_risk:.2f} vs {away} {a_risk:.2f}.")
    return pick, lines[:6], h_score, a_score


def comparison_signals_corners(home: str, away: str, league: str) -> Tuple[Optional[str], List[str], Optional[float], Optional[float]]:
    hm = get_team_profile_meta(home, league)
    am = get_team_profile_meta(away, league)
    hr = get_team_recent_stats(home, league, last_n=6)
    ar = get_team_recent_stats(away, league, last_n=6)

    h_for = safe_float(hm.get("corners_pm"))
    a_for = safe_float(am.get("corners_pm"))
    h_against = safe_float(hm.get("corners_against_pm"))
    a_against = safe_float(am.get("corners_against_pm"))
    h_edge = safe_float(hm.get("corner_edge_pm"))
    a_edge = safe_float(am.get("corner_edge_pm"))
    h_ctrl = safe_float(hm.get("control_index"))
    a_ctrl = safe_float(am.get("control_index"))

    hr_for = hr.get("corners_for_avg")
    ar_for = ar.get("corners_for_avg")
    hr_against = hr.get("corners_against_avg")
    ar_against = ar.get("corners_against_avg")

    h_proj, a_proj = projected_corners(hm, am)

    ccw = SCORING_WEIGHTS["comparison_corners"]

    def corner_score(proj, season_for, opp_allow, recent_for, control, edge) -> Optional[float]:
        vals = [proj, season_for, opp_allow, recent_for, control, edge]
        if all(v is None for v in vals):
            return None
        p = proj if proj is not None else 0.0
        sf = season_for if season_for is not None else 0.0
        oa = opp_allow if opp_allow is not None else 0.0
        rf = recent_for if recent_for is not None else 0.0
        c = control if control is not None else 0.0
        e = edge if edge is not None else 0.0
        return ccw["proj"] * p + ccw["season_for"] * sf + ccw["opp_allow"] * oa + ccw["recent_for"] * rf + ccw["control"] * c + ccw["edge"] * e

    h_score = corner_score(h_proj, h_for, a_against, hr_for, h_ctrl, h_edge)
    a_score = corner_score(a_proj, a_for, h_against, ar_for, a_ctrl, a_edge)

    pick = None
    if h_score is not None and a_score is not None:
        pick = home if h_score >= a_score else away
    elif h_score is not None:
        pick = home
    elif a_score is not None:
        pick = away

    lines: List[str] = []
    if h_proj is not None and a_proj is not None:
        lines.append(f"Projected corners: {home} {h_proj:.2f} vs {away} {a_proj:.2f}.")
    if h_for is not None and a_for is not None:
        lines.append(f"Season corners for/match: {home} {h_for:.2f} vs {away} {a_for:.2f}.")
    if h_against is not None and a_against is not None:
        lines.append(f"Season corners against/match: {home} {h_against:.2f} vs {away} {a_against:.2f}.")
    if hr_for is not None and ar_for is not None:
        lines.append(f"Recent 6-match corners for: {home} {hr_for:.2f} vs {away} {ar_for:.2f}.")
    if h_edge is not None and a_edge is not None:
        lines.append(f"Corner edge: {home} {h_edge:+.2f} vs {away} {a_edge:+.2f}.")
    if h_score is not None and a_score is not None:
        lines.append(f"Composite corner pressure score: {home} {h_score:.2f} vs {away} {a_score:.2f}.")
    return pick, lines[:6], h_score, a_score


def comparison_signals_sot(home: str, away: str, league: str) -> Tuple[Optional[str], List[str], Optional[float], Optional[float]]:
    hm = get_team_profile_meta(home, league)
    am = get_team_profile_meta(away, league)
    hr = get_team_recent_stats(home, league, last_n=6)
    ar = get_team_recent_stats(away, league, last_n=6)

    h_for = safe_float(hm.get("sot_for_pm"))
    a_for = safe_float(am.get("sot_for_pm"))
    h_against = safe_float(hm.get("sot_against_pm"))
    a_against = safe_float(am.get("sot_against_pm"))
    h_ratio = safe_float(hm.get("_sot_ratio_for"))
    a_ratio = safe_float(am.get("_sot_ratio_for"))

    hr_for = hr.get("sot_for_avg")
    ar_for = ar.get("sot_for_avg")

    h_proj, a_proj = projected_sot(hm, am)

    sw = SCORING_WEIGHTS["comparison_sot"]

    def sot_score(proj, season_for, opp_allow, recent_for, ratio) -> Optional[float]:
        vals = [proj, season_for, opp_allow, recent_for, ratio]
        if all(v is None for v in vals):
            return None
        p = proj if proj is not None else 0.0
        sf = season_for if season_for is not None else 0.0
        oa = opp_allow if opp_allow is not None else 0.0
        rf = recent_for if recent_for is not None else 0.0
        r = ratio if ratio is not None else 0.0
        return sw["proj"] * p + sw["season_for"] * sf + sw["opp_allow"] * oa + sw["recent_for"] * rf + sw["sot_ratio"] * r

    h_score = sot_score(h_proj, h_for, a_against, hr_for, h_ratio)
    a_score = sot_score(a_proj, a_for, h_against, ar_for, a_ratio)

    pick = None
    if h_score is not None and a_score is not None:
        pick = home if h_score >= a_score else away
    elif h_score is not None:
        pick = home
    elif a_score is not None:
        pick = away

    lines: List[str] = []
    if h_proj is not None and a_proj is not None:
        lines.append(f"Projected SoT: {home} {h_proj:.2f} vs {away} {a_proj:.2f}.")
    if h_for is not None and a_for is not None:
        lines.append(f"Season SoT for/match: {home} {h_for:.2f} vs {away} {a_for:.2f}.")
    if h_against is not None and a_against is not None:
        lines.append(f"Season SoT against/match: {home} {h_against:.2f} vs {away} {a_against:.2f}.")
    if hr_for is not None and ar_for is not None:
        lines.append(f"Recent 6-match SoT for: {home} {hr_for:.2f} vs {away} {ar_for:.2f}.")
    if h_ratio is not None and a_ratio is not None:
        lines.append(f"SoT accuracy ratio: {home} {h_ratio:.2f} vs {away} {a_ratio:.2f}.")
    if h_score is not None and a_score is not None:
        lines.append(f"Composite SoT pressure score: {home} {h_score:.2f} vs {away} {a_score:.2f}.")
    return pick, lines[:6], h_score, a_score


def render_comparison_answer(user_q: str, league: str, events: List[Dict]) -> str:
    group = requested_stat_group(user_q)
    if group not in {"corners", "cards", "sot"}:
        return (
            "I can compare corners/cards/shots-on-target for fixtures, but I couldn't detect which stat you want. "
            "Try: 'for all tomorrow EPL games, which team gets more corners?'"
        )

    detailed = wants_deep_explanation(user_q)
    lines: List[str] = []
    title = {"corners": "Corners", "cards": "Cards/Bookings", "sot": "Shots on Target"}[group]
    lines.append(f"{title} comparison by fixture:")

    for ev in events:
        home = str(ev.get("home_team") or "Home")
        away = str(ev.get("away_team") or "Away")

        if group == "corners":
            pick, reasons, h_score, a_score = comparison_signals_corners(home, away, league)
            stat_label = "corners"
            gap_high_key, gap_med_key = "corners_gap_high", "corners_gap_medium"
        elif group == "sot":
            pick, reasons, h_score, a_score = comparison_signals_sot(home, away, league)
            stat_label = "shots on target"
            gap_high_key, gap_med_key = "sot_gap_high", "sot_gap_medium"
        else:
            pick, reasons, h_score, a_score = comparison_signals_cards(home, away, league)
            stat_label = "cards"
            gap_high_key, gap_med_key = "cards_gap_high", "cards_gap_medium"

        if pick is None:
            lines.append(f"- {home} vs {away}: insufficient profile stats for {stat_label}.")
            continue
        lines.append(f"- {home} vs {away}: {pick} likely more {stat_label}.")
        if detailed:
            for r in reasons[:5]:
                lines.append(f"  Reasoning: {r}")
            if h_score is not None and a_score is not None:
                margin = abs(h_score - a_score)
                cfw = SCORING_WEIGHTS["confidence"]
                conf = "high" if margin >= cfw[gap_high_key] else ("medium" if margin >= cfw[gap_med_key] else "low")
                lines.append(f"  Reasoning: Confidence={conf} (score gap {margin:.2f}).")
        elif reasons:
            lines.append(f"  Evidence: {reasons[0]}")
            if len(reasons) > 1:
                lines.append(f"  Evidence: {reasons[1]}")

    method = "Method: deterministic multi-signal projection from local KB (season + recent fixtures), not bookmaker lines."
    lines.append(method if detailed else "Method: profile-based projection from your local KB (not bookmaker line prices).")
    return "\n".join(lines)


# ---- Intent Classification (scored classifier) ----
INTENT_SCHEDULE = "schedule"
INTENT_CAPABILITY = "capability"
INTENT_COMPARISON = "comparison"
INTENT_TOTALS_LINE = "totals_line"
INTENT_PLAYER_PROP = "player_prop"
INTENT_PARLAY = "parlay"
INTENT_UNKNOWN = "unknown"

INTENT_LABELS = {
    INTENT_SCHEDULE: "Schedule & Fixtures",
    INTENT_CAPABILITY: "Market Availability",
    INTENT_COMPARISON: "Stat Comparison",
    INTENT_TOTALS_LINE: "Totals Line Advice",
    INTENT_PLAYER_PROP: "Player Props",
    INTENT_PARLAY: "Parlay Builder",
    INTENT_UNKNOWN: "General",
}


def classify_intent(user_q: str) -> List[Tuple[str, float]]:
    """Score all applicable intents, return sorted highest-first."""
    scores: List[Tuple[str, float]] = []

    if is_schedule_query(user_q):
        scores.append((INTENT_SCHEDULE, 1.0))
    if is_capability_query(user_q):
        scores.append((INTENT_CAPABILITY, 0.9))

    parlay = is_parlay_query(user_q)
    player_prop = is_player_prop_query(user_q)
    if player_prop:
        scores.append((INTENT_PLAYER_PROP, 0.85))
    if is_comparison_query(user_q):
        scores.append((INTENT_COMPARISON, 0.2 if (parlay or player_prop) else 0.7))
    if is_totals_line_query(user_q):
        scores.append((INTENT_TOTALS_LINE, 0.15 if (parlay or player_prop) else 0.65))
    if parlay and not player_prop:
        scores.append((INTENT_PARLAY, 0.6))

    if not scores:
        return [(INTENT_UNKNOWN, 0.0)]

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _handle_schedule(user_q: str, events: List[Dict], notes: List[str]) -> str:
    return "Upcoming fixtures:\n" + event_fixtures_text(events)


def _handle_capability(user_q: str, events: List[Dict], notes: List[str]) -> str:
    lines: List[str] = []
    cands = build_candidates(events)
    groups = available_market_groups(cands)
    asked = requested_groups_from_query(user_q)
    if asked:
        missing = [g for g in sorted(asked) if g not in groups]
        have = [g for g in sorted(asked) if g in groups]
        if have:
            lines.append(f"Available for requested window: {', '.join(have)}")
        if missing:
            lines.append(f"Not available from current feed/window: {', '.join(missing)}")
    else:
        lines.append("Available market groups: " + (", ".join(sorted(groups)) if groups else "none"))
    if events:
        lines.append("Fixtures considered:")
        lines.append(event_fixtures_text(events))
    if notes:
        lines.append("Notes:")
        for n in notes:
            lines.append(f"- {n}")
    return "\n".join(lines)


def _handle_comparison(user_q: str, c: ConstraintSpec, events: List[Dict], notes: List[str]) -> str:
    out = render_comparison_answer(user_q, c.league, events)
    if notes:
        out = out + "\nNotes:\n" + "\n".join(f"- {n}" for n in notes)
    return out


def _handle_totals_line(user_q: str, c: ConstraintSpec, events: List[Dict], notes: List[str]) -> str:
    out = render_totals_line_answer(user_q, c.league, events)
    if notes:
        out = out + "\nNotes:\n" + "\n".join(f"- {n}" for n in notes)
    return out


def _handle_player_props(user_q: str, c: ConstraintSpec, events: List[Dict], notes: List[str]) -> str:
    if not events:
        return "No fixtures found for the requested date. Cannot analyse player props without a fixture list."
    if client is None:
        return "OpenAI client not configured — player prop analysis requires LLM access."

    fixtures_text = event_fixtures_text(events)
    player_docs = get_player_snippets_for_events(
        events, c.league, per_team_profiles=6, per_team_fixtures=8,
    )
    player_text = "\n\n".join((d.get("text") or "") for d in player_docs[:30])

    profile_docs = get_team_profile_snippets_for_events(events, c.league, per_team=1)
    profile_text = "\n\n".join((d.get("text") or "") for d in profile_docs[:10])

    system_msg = (
        "You are a football betting analyst specialising in player props.\n"
        "The user will ask about goalscorers, shots on target, assists, cards, or other player-level markets.\n"
        "You are provided with:\n"
        "- The fixture list for the requested date.\n"
        "- Player profile docs (season averages: goals/90, assists/90, shots/90, SoT/90, fouls/90, cards/90, form index).\n"
        "- Player fixture docs (recent per-match stats: goals, assists, shots, SoT, cards, minutes, rating).\n"
        "- Team profile docs (team-level context: form, control, dominance, aggression).\n\n"
        "Guidelines:\n"
        "- Base your picks on the statistical data provided. Cite specific numbers.\n"
        "- For goalscorer picks: prioritise goals/90, shots/90, SoT/90, minutes played, and recent form index.\n"
        "- For SoT picks: use shots/90, SoT/90, and recent fixture shot counts.\n"
        "- For card picks: use cards/90, fouls/90, aggression index, and recent yellow card counts.\n"
        "- For assists: use assists/90, passes/90, and chance creation signals.\n"
        "- If data is sparse for a player, say so rather than guessing.\n"
        "- When recommending multiple picks, rank by confidence and explain the reasoning.\n"
        "- Format your answer clearly with fixture headers and bullet points.\n"
    )
    user_msg = (
        f"User question: {user_q}\n\n"
        f"Fixtures on this date:\n{fixtures_text}\n\n"
        f"Team profile context:\n{profile_text or 'None'}\n\n"
        f"Player data:\n{player_text or 'No player data available in KB.'}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.3,
        )
        out = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        out = f"LLM call failed for player props: {exc}"

    if notes:
        out = out + "\n\nNotes:\n" + "\n".join(f"- {n}" for n in notes)
    out += "\nMethod: LLM analysis backed by player profile and fixture data from local KB."
    return out


def answer_once(user_q: str, default_league: str = "EPL") -> None:
    raw_user_q = user_q
    followup_mode = is_followup_request(user_q)
    c = parse_constraints(user_q, default_league=default_league)
    parlay_intent = is_parlay_query(user_q)
    force_totals_only = wants_totals_only_parlay(user_q)
    totals_stat_group = requested_stat_group(user_q) or "goals"
    if force_totals_only:
        c.requested_markets = {"totals"}
        # Reset include/exclude to avoid contradictions with generic only_* detectors
        # from parse_constraints(), then set clean group-specific constraints.
        if totals_stat_group == "goals":
            c.hard_include_groups = {"totals"}
            c.hard_exclude_groups = {"moneyline", "spreads", "corners", "cards"}
        elif totals_stat_group == "corners":
            # Corner totals market keys (e.g. "alternate_totals_corners") are classified
            # as "corners" group, NOT "totals". So hard_include must be "corners" only.
            c.hard_include_groups = {"corners"}
            c.hard_exclude_groups = {"moneyline", "spreads", "cards", "totals"}
            c.require_total_corner_keys = True
        elif totals_stat_group == "cards":
            # Card totals market keys are classified as "cards" group.
            c.hard_include_groups = {"cards"}
            c.hard_exclude_groups = {"moneyline", "spreads", "corners", "totals"}
    events_raw, notes = fetch_events(c.league, c.requested_markets)
    if not events_raw:
        print("Could not fetch odds events.")
        for n in notes:
            print("-", n)
        return

    explicit_date = parse_explicit_date(user_q)
    if explicit_date is not None:
        events = filter_events_by_exact_date(events_raw, explicit_date)
        notes.append(f"Applied explicit date filter: {explicit_date.isoformat()}.")
    else:
        events = filter_events_by_window(events_raw, c.time_window)

    events = filter_events_by_mentioned_teams(user_q, events)
    mentioned_teams = extract_mentioned_event_teams(user_q, events_raw)
    if is_singular_fixture_request(user_q) and len(mentioned_teams) == 1 and events:
        team = next(iter(mentioned_teams))
        team_events = [ev for ev in events if (ev.get("home_team") == team) or (ev.get("away_team") == team)]
        if team_events:
            team_events.sort(key=event_sort_key)
            events = [team_events[0]]
            notes.append(
                f"Applied singular fixture lock: {events[0].get('home_team')} vs {events[0].get('away_team')}."
            )

    has_explicit_team_mention = query_mentions_any_event_team(user_q, events_raw)
    if followup_mode and memory.get("last_selected_event_ids") and not has_explicit_team_mention:
        keep = {str(x) for x in (memory.get("last_selected_event_ids") or [])}
        filtered = [ev for ev in events if str(ev.get("id") or "") in keep]
        if filtered:
            events = filtered
            notes.append("Applied follow-up memory: kept fixtures from previous recommendation.")

    # Event-level enrichment for groups the user explicitly asked about.
    asked_groups = requested_groups_from_query(user_q)
    needed_groups = set(c.hard_include_groups) | set(c.required_group_counts.keys()) | set(asked_groups)

    # For single-fixture multi-leg parlays, widen candidate market families so
    # constraints can still produce valid combos without forcing users to ask
    # explicitly for corners/cards every time.
    planned_leg_count = c.leg_count if c.leg_count is not None else 3
    if c.per_match_mode:
        planned_leg_count = len(events)
    if (
        parlay_intent
        and not force_totals_only
        and len(events) == 1
        and planned_leg_count >= 3
    ):
        needed_groups.update({"corners", "cards"})
        notes.append("Expanded single-game parlay candidate groups with corners/cards.")

    if needed_groups:
        events, enrich_notes = enrich_events_for_groups(events, c.league, needed_groups)
        notes.extend(enrich_notes)

    if parlay_intent and force_totals_only:
        events, depth_notes = enrich_events_for_totals_depth(events, c.league, stat_group=totals_stat_group)
        notes.extend(depth_notes)
        notes.append(f"Expanded full-game totals ladder for parlay optimization ({totals_stat_group}).")
    # ---- Intent-based routing via scored classifier ----
    intent_scores = classify_intent(user_q)
    top_intent = intent_scores[0][0]

    if top_intent == INTENT_SCHEDULE:
        out = _handle_schedule(user_q, events, notes)
        print(out)
        add_chat_turn(raw_user_q, out)
        return

    if top_intent == INTENT_CAPABILITY:
        out = _handle_capability(user_q, events, notes)
        print(out)
        add_chat_turn(raw_user_q, out)
        return

    if top_intent == INTENT_COMPARISON:
        out = _handle_comparison(user_q, c, events, notes)
        print(out)
        add_chat_turn(raw_user_q, out)
        return

    if top_intent == INTENT_TOTALS_LINE:
        out = _handle_totals_line(user_q, c, events, notes)
        print(out)
        add_chat_turn(raw_user_q, out)
        return

    if top_intent == INTENT_PLAYER_PROP:
        out = _handle_player_props(user_q, c, events, notes)
        print(out)
        add_chat_turn(raw_user_q, out)
        memory["last_intent"] = INTENT_PLAYER_PROP
        return

    if top_intent == INTENT_UNKNOWN:
        out = "I can help with fixture questions and parlays.\nTry: 'for all tomorrow EPL games, which team gets more corners?'"
        print(out)
        add_chat_turn(raw_user_q, out)
        return

    if not events:
        out = "No fixtures found for requested window."
        print(out)
        add_chat_turn(raw_user_q, out)
        return

    candidates = build_candidates(events)
    if not candidates:
        out = "No odds candidates found in provider response."
        print(out)
        add_chat_turn(raw_user_q, out)
        return

    leg_count = c.leg_count if c.leg_count is not None else 3
    if c.per_match_mode:
        leg_count = len(events)

    # Richer follow-up: carry over constraints from memory when applicable
    if followup_mode and memory.get("last_constraint_spec"):
        prev = memory["last_constraint_spec"]
        q_lower = user_q.lower()
        # "same markets/types" → inherit market group constraints
        if re.search(r"\b(same|keep|previous)\b.*\b(market|type|kind)\b", q_lower):
            if prev.get("hard_include_groups") and not c.hard_include_groups:
                c.hard_include_groups = set(prev["hard_include_groups"])
                notes.append("Carried over market group constraints from previous recommendation.")
            if prev.get("hard_exclude_groups") and not c.hard_exclude_groups:
                c.hard_exclude_groups = set(prev["hard_exclude_groups"])
            if prev.get("soft_prefer_groups") and not c.soft_prefer_groups:
                c.soft_prefer_groups = set(prev["soft_prefer_groups"])
        # "safer" / "cap odds" → inherit unique-events and per-match from memory
        if prev.get("per_match_mode") and c.per_match_mode is False:
            if re.search(r"\b(same|keep)\b.*\b(fixture|match|game)\b", q_lower):
                c.per_match_mode = True
                c.require_unique_events = True
        if c.leg_count is None and memory.get("last_leg_count"):
            leg_count = int(memory.get("last_leg_count") or leg_count)
        if re.search(r"\badd\b|\banother\b|\bone more\b", q_lower):
            leg_count = min(leg_count + 1, max(1, len(candidates)))

    # Keep realistic count under available unique-event capacity when required.
    if c.require_unique_events:
        unique_events = len({x.event_id for x in candidates})
        if unique_events > 0:
            leg_count = min(leg_count, unique_events)

    # Feasibility pre-check before expensive solver
    feasible, feasibility_warnings = check_feasibility(candidates, leg_count, c)
    notes.extend(feasibility_warnings)
    if not feasible:
        print("Could not construct a parlay with current constraints.")
        for n in notes:
            print("-", n)
        add_chat_turn(raw_user_q, "Could not construct a parlay with current constraints.\n" + "\n".join(f"- {n}" for n in notes))
        return
    if c.require_unique_events:
        unique_events = len({x.event_id for x in candidates})
        if unique_events < leg_count and unique_events > 0:
            leg_count = unique_events

    selected, sel_notes = select_parlay(candidates, leg_count, c)
    notes.extend(sel_notes)
    if force_totals_only:
        notes.append("Applied totals-only parlay routing from user query.")
    if not selected:
        print("Could not construct a parlay with current constraints.")
        for n in notes:
            print("-", n)
        return

    # Optional LLM validation pass (gated by RAG_LLM_VALIDATION env var)
    validation_passed, validation_concern = llm_validate_selection(selected, c.league)
    if not validation_passed and validation_concern:
        notes.append(f"LLM validation concern: {validation_concern}")

    llm_evidence, llm_why, llm_note = get_llm_reasoning(user_q, selected, events, c.league)
    if llm_note:
        notes.append(llm_note)

    out = render_parlay(
        user_q,
        c,
        events,
        selected,
        notes,
        llm_evidence=llm_evidence,
        llm_why=llm_why,
    )
    print(out)

    memory["last_selected_event_ids"] = [x.event_id for x in selected]
    memory["last_selected_fixtures"] = [x.fixture for x in selected]
    memory["last_leg_count"] = len(selected)
    memory["last_intent"] = top_intent
    memory["last_league"] = c.league
    memory["last_constraint_spec"] = {
        "hard_include_groups": list(c.hard_include_groups),
        "hard_exclude_groups": list(c.hard_exclude_groups),
        "soft_prefer_groups": list(c.soft_prefer_groups),
        "required_group_counts": dict(c.required_group_counts),
        "target_multiplier": c.target_multiplier,
        "target_mode": c.target_mode,
        "per_match_mode": c.per_match_mode,
        "require_unique_events": c.require_unique_events,
        "forbid_spread_keys": c.forbid_spread_keys,
        "require_total_corner_keys": c.require_total_corner_keys,
    }
    memory["last_selected_legs"] = [
        {"fixture": x.fixture, "market_key": x.market_key, "outcome": x.outcome,
         "odds": x.odds, "point": x.point, "market_group": market_group_from_key(x.market_key)}
        for x in selected
    ]
    memory["last_market_groups_used"] = list({market_group_from_key(x.market_key) for x in selected})
    add_chat_turn(raw_user_q, out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Odds-first betting assistant (RAG v2, lightweight).")
    ap.add_argument("--league", type=str, default="EPL", help="Default league if query does not specify.")
    ap.add_argument("--once", type=str, default=None, help="Run one query and exit.")
    args = ap.parse_args()

    if args.once:
        answer_once(args.once, default_league=args.league)
        return

    print(f"RAG v2 ready. Default league={args.league}. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nClient: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        answer_once(q, default_league=args.league)


if __name__ == "__main__":
    main()
