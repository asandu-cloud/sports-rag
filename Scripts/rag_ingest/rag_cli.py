#!/usr/bin/env python3
# Small local RAG trial (matchup-aware)

import os
import sys
import argparse
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import requests
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from heuristics import top_markets
from zoneinfo import ZoneInfo
from odds_parlay_solver import (
    OddsLeg,
    combined_odds,
    extract_candidates,
    select_best_parlay,
)


load_dotenv()

# Make project root importable for graph modules
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "Scripts"))

from graph.query_graph import get_matchup_snapshot, snapshot_to_text

# Simple in-process memory for interactive use (last matchup + parlay)
memory = {
    "last_home": None,
    "last_away": None,
    "last_parlay": None,
    "last_query": None,
    "last_answer": None,
    "history": [],
}

# ---- CONFIG ----
CHROMA_DIR = "/Users/sanduandrei/Desktop/Betting_RAG/Index/chroma"
COLLECTION = "football_top5"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-5.2"  # choose your chat model here
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
# Key is expected from process environment under this exact name.
ODDS_API_KEY = os.environ.get("ODDS-API")
LEAGUE_TO_ODDS_SPORT = {
    "EPL": "soccer_epl",
    "LaLiga": "soccer_spain_la_liga",
    "SerieA": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue1": "soccer_france_ligue_one",
}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------------
# Embedding helper
# -------------------------
def embed(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts).data
    return [r.embedding for r in resp]


# -------------------------
# League resolution
# -------------------------
def resolve_league_from_flag_or_query(league_flag: Optional[str], user_q: str) -> Optional[str]:
    if league_flag:
        return league_flag
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
    return None


# -------------------------
# Season parsing and where-clause builder
# -------------------------
def parse_seasons_arg(season_arg: Optional[str]) -> Optional[List[str]]:
    if not season_arg:
        return None
    parts = [s.strip() for s in season_arg.split(",") if s.strip()]
    return parts or None


def build_where(league: Optional[str],
                seasons: Optional[List[str]],
                extra_filters: Optional[List[dict]] = None) -> dict:
    """
    Chroma where syntax rules:
    - single dict like {"league": "EPL"} is fine
    - if multiple conditions, wrap in {"$and": [ ... ]}
    - OR across seasons uses {"$or": [{"season": "2024/25"}, {"season":"2025/26"}]}
    """
    clauses: List[dict] = []

    if league:
        clauses.append({"league": league})

    if seasons:
        if len(seasons) == 1:
            clauses.append({"season": seasons[0]})
        else:
            clauses.append({"$or": [{"season": s} for s in seasons]})

    if extra_filters:
        clauses.extend(extra_filters)

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


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


def normalize_text_for_match(value: str) -> str:
    """
    Accent-insensitive, punctuation-light normalization used for matching
    free-text player mentions (e.g., Ekitike -> Ekitike).
    """
    if not value:
        return ""
    s = unicodedata.normalize("NFKD", value)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_mentioned_player_docs(user_q: str,
                              league: Optional[str],
                              seasons: Optional[List[str]],
                              max_players: int = 3,
                              per_player_profiles: int = 1,
                              per_player_fixtures: int = 3) -> List[Dict]:
    """
    Pull player docs when the query mentions a player name with accent variance.
    Example: "Ekitike" should match stored "Ekitiké".
    """
    q_norm = normalize_text_for_match(user_q)
    if not q_norm:
        return []

    db = chromadb.PersistentClient(path=CHROMA_DIR)
    col = db.get_or_create_collection(COLLECTION)

    def collect_candidate_names(use_seasons: Optional[List[str]]) -> List[str]:
        where_profiles = build_where(league, use_seasons, extra_filters=[{"doc_type": "player_profile"}])
        res = col.get(where=where_profiles, include=["metadatas"], limit=10000)
        names: List[str] = []
        seen = set()
        for m in res.get("metadatas", []):
            name = (m or {}).get("player_name")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return names

    def match_names(names: List[str]) -> List[str]:
        scored: List[Tuple[int, str]] = []
        for name in names:
            n_norm = normalize_text_for_match(name)
            if not n_norm:
                continue

            hit = False
            # Full-name / multi-token mention
            if n_norm in q_norm:
                hit = True
            else:
                # Surname fallback for user prompts that provide only last name.
                tokens = [t for t in n_norm.split(" ") if len(t) >= 3]
                if tokens:
                    last = tokens[-1]
                    if re.search(rf"\b{re.escape(last)}\b", q_norm):
                        hit = True

            if hit:
                # Prefer more specific names first.
                scored.append((len(n_norm), name))

        scored.sort(reverse=True)
        out: List[str] = []
        seen = set()
        for _, nm in scored:
            if nm not in seen:
                seen.add(nm)
                out.append(nm)
            if len(out) >= max_players:
                break
        return out

    names = collect_candidate_names(seasons)
    matched_names = match_names(names)

    # If season filter is too narrow, retry matching over all seasons.
    if not matched_names and seasons:
        names = collect_candidate_names(None)
        matched_names = match_names(names)

    out_map: Dict[str, Dict] = {}
    for player_name in matched_names:
        where_pp = build_where(
            league,
            seasons,
            extra_filters=[{"doc_type": "player_profile"}, {"player_name": player_name}],
        )
        pp_res = col.get(where=where_pp, include=["documents", "metadatas"], limit=per_player_profiles)
        for i, d, m in zip(pp_res.get("ids", []), pp_res.get("documents", []), pp_res.get("metadatas", [])):
            out_map[i] = {"id": i, "text": d, "meta": m}

        where_pf = build_where(
            league,
            seasons,
            extra_filters=[{"doc_type": "player_fixture"}, {"player_name": player_name}],
        )
        pf_res = col.get(where=where_pf, include=["documents", "metadatas"], limit=50)
        rows = []
        for i, d, m in zip(pf_res.get("ids", []), pf_res.get("documents", []), pf_res.get("metadatas", [])):
            rows.append({"id": i, "text": d, "meta": m})
        rows.sort(key=lambda x: (x["meta"].get("fixture_date") or ""), reverse=True)
        for r in rows[:per_player_fixtures]:
            out_map[r["id"]] = r

    return list(out_map.values())


# -------------------------
# Basic semantic retrieval (fallback mode)
# -------------------------
def generic_retrieve(user_q: str,
                     league: Optional[str],
                     seasons: Optional[List[str]],
                     k: int) -> List[Dict]:
    """
    Old/default behavior: embed the query and pull top-k similar docs,
   constrained by league / seasons.
    """
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    col = db.get_or_create_collection(COLLECTION)

    vec = embed([user_q])[0]

    def q_with_extra(extra_filters: List[dict], topk: int) -> List[Dict]:
        where = build_where(league, seasons, extra_filters=extra_filters)
        res = col.query(query_embeddings=[vec], n_results=topk, where=where)
        out = []
        if res.get("documents"):
            for d, m, i in zip(res["documents"][0], res["metadatas"][0], res["ids"][0]):
                out.append({"id": i, "text": d, "meta": m})
        return out

    # Mixed retrieval so open-ended prompts include player context too.
    type_budgets = [
        ("team_profile", max(2, k // 3)),
        ("team_fixture", max(2, k // 3)),
        ("player_profile", max(2, k // 4)),
        ("player_fixture", max(2, k // 4)),
    ]

    merged: Dict[str, Dict] = {}
    for doc_type, topk in type_budgets:
        rows = q_with_extra([{"doc_type": doc_type}], topk)
        for r in rows:
            merged[r["id"]] = r

    # Deterministically include explicitly-mentioned players (accent-insensitive).
    for r in get_mentioned_player_docs(user_q, league, seasons):
        merged[r["id"]] = r

    hits = list(merged.values())
    if len(hits) < k:
        # Backfill with unrestricted search if needed
        for r in q_with_extra([], k):
            if r["id"] not in merged:
                merged[r["id"]] = r
        hits = list(merged.values())

    return hits[:k]


def parse_target_request(user_q: str) -> Tuple[Optional[float], str]:
    """
    Returns:
    - target multiplier if explicitly provided, else None
    - mode: "none" | "around" | "max" | "min"
    """
    q = user_q.lower()

    max_pat = r"(?:no more than|under|below|at most|max(?:imum)?)\s*(\d+(?:\.\d+)?)\s*x?"
    min_pat = r"(?:at least|over|above|min(?:imum)?)\s*(\d+(?:\.\d+)?)\s*x?"
    around_pat = r"(?:around|about|approx(?:imately)?|target(?: of)?)\s*(\d+(?:\.\d+)?)\s*x?"
    plain_pat = r"(\d+(?:\.\d+)?)\s*x"

    for pat, mode in [(max_pat, "max"), (min_pat, "min"), (around_pat, "around"), (plain_pat, "around")]:
        m = re.search(pat, q)
        if not m:
            continue
        try:
            return float(m.group(1)), mode
        except ValueError:
            break
    return None, "none"


def extract_leg_count(user_q: str, default: int = 3) -> int:
    m = re.search(r"(\d+)\s*[- ]?leg", user_q.lower())
    if not m:
        return default
    try:
        v = int(m.group(1))
        return max(1, min(v, 8))
    except ValueError:
        return default


def extract_match_count(user_q: str) -> Optional[int]:
    q = user_q.lower()
    m = re.search(r"\b(\d+)\s+matches?\b", q)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    words = {
        "one match": 1,
        "two matches": 2,
        "three matches": 3,
        "four matches": 4,
    }
    for k, v in words.items():
        if k in q:
            return v
    return None


def query_excludes_moneyline(user_q: str) -> bool:
    q = user_q.lower()
    return bool(
        re.search(r"(?:no|without|avoid|dont|don't)\s+.*(?:money\s*line|moneyline|ml|h2h|1x2)", q)
    )


def query_moneyline_only(user_q: str) -> bool:
    q = user_q.lower()
    return bool(
        re.search(r"(?:only|just)\s+.*(?:money\s*line|moneyline|ml|h2h|1x2)", q)
    )


def infer_requested_odds_markets(user_q: str) -> List[str]:
    """
    Decide which Odds API markets to fetch based on user intent.
    Defaults to broad set; narrows only when user asks explicitly.
    """
    q = user_q.lower()
    if query_moneyline_only(q):
        return ["h2h"]
    if query_excludes_moneyline(q):
        # Keep non-ML defaults broad.
        return ["totals", "spreads", "corners", "cards"]

    wants_totals = bool(
        re.search(r"\btotal(s)?\b", q)
        or re.search(r"\bover\b", q)
        or re.search(r"\bunder\b", q)
        or re.search(r"\bgoals?\s+over\b", q)
        or re.search(r"\bgoals?\s+under\b", q)
    )
    wants_spreads = bool(re.search(r"\bspread(s)?\b", q) or re.search(r"\bhandicap(s)?\b", q))
    wants_ml = bool(re.search(r"\bmoney\s*line\b", q) or re.search(r"\bml\b", q) or re.search(r"\b1x2\b", q) or re.search(r"\bh2h\b", q))
    wants_corners = bool(re.search(r"\bcorner(s)?\b", q))
    wants_cards = bool(re.search(r"\bcard(s)?\b", q) or re.search(r"\bbooking(s)?\b", q))

    picks: List[str] = []
    if wants_ml:
        picks.append("h2h")
    if wants_totals:
        picks.append("totals")
    if wants_spreads:
        picks.append("spreads")
    if wants_corners:
        picks.append("corners")
    if wants_cards:
        picks.append("cards")
    if not picks:
        picks = ["h2h", "totals", "spreads"]
    # Deduplicate while preserving order.
    return list(dict.fromkeys(picks))


def infer_unique_event_requirement(user_q: str) -> bool:
    q = user_q.lower()
    if any(k in q for k in ["same game parlay", "sgp", "same match", "single match"]):
        return False
    if any(k in q for k in ["different matches", "across matches", "one per match"]):
        return True
    # Default flexible behavior: allow multi-leg from same event unless user restricts it.
    return False


def parse_odds_constraints(user_q: str) -> Dict:
    """
    Parse user intent into a structured constraint object instead of hard-coded rules.
    """
    q = user_q.lower()
    target_mult, target_mode = parse_target_request(user_q)
    explicit_leg_count = re.search(r"(\d+)\s*[- ]?leg", q) is not None
    per_match_mode = bool(
        re.search(r"(as many legs as|for each game|for every game|one leg per game|one leg per match)", q)
    )

    wants_corners_only = bool(re.search(r"(only|just)\s+.*corner", q) or re.search(r"corner.*(only|just)", q))
    wants_cards_only = bool(re.search(r"(only|just)\s+.*card", q) or re.search(r"card.*(only|just)", q))

    constraints = {
        "requested_markets": infer_requested_odds_markets(user_q),
        "target_multiplier": target_mult,
        "target_mode": target_mode,
        "leg_count": extract_leg_count(user_q, default=3),
        "leg_count_explicit": explicit_leg_count,
        "leg_count_per_match_mode": per_match_mode,
        "require_unique_events": infer_unique_event_requirement(user_q) or per_match_mode,
        "exclude_moneyline": query_excludes_moneyline(user_q),
        "moneyline_only": query_moneyline_only(user_q),
        "corners_only": wants_corners_only,
        "cards_only": wants_cards_only,
    }
    return constraints


def apply_odds_constraints(constraints: Dict, events: List[Dict]) -> Dict:
    out = dict(constraints)
    if out.get("leg_count_per_match_mode"):
        out["leg_count"] = max(1, len(events))
    return out


def summarize_constraint_results(constraints: Dict, selected_legs: List[OddsLeg]) -> Tuple[List[str], List[str]]:
    applied: List[str] = []
    unmet: List[str] = []

    requested_markets = constraints.get("requested_markets") or []
    if requested_markets:
        applied.append(f"market filter={','.join(requested_markets)}")

    if constraints.get("require_unique_events"):
        applied.append("one leg per match")

    if constraints.get("exclude_moneyline"):
        if any(getattr(x, "market_key", "") == "h2h" for x in selected_legs):
            unmet.append("user asked to avoid moneyline, but selected legs include h2h")
        else:
            applied.append("excluded moneyline")

    if constraints.get("moneyline_only"):
        if all(getattr(x, "market_key", "") == "h2h" for x in selected_legs):
            applied.append("moneyline-only")
        else:
            unmet.append("user asked for moneyline-only, but non-h2h legs were selected")

    if constraints.get("corners_only"):
        if all("corner" in (getattr(x, "market_key", "") or "").lower() for x in selected_legs):
            applied.append("corners-only")
        else:
            unmet.append("corners-only requested, but corners market not available in selected legs")

    if constraints.get("cards_only"):
        if all("card" in (getattr(x, "market_key", "") or "").lower() for x in selected_legs):
            applied.append("cards-only")
        else:
            unmet.append("cards-only requested, but cards market not available in selected legs")

    target = constraints.get("target_multiplier")
    mode = constraints.get("target_mode", "none")
    if target is not None and mode != "none":
        combo = combined_odds(selected_legs)
        if mode == "max":
            if combo <= target:
                applied.append(f"target max {target:.2f}x")
            else:
                unmet.append(f"target max {target:.2f}x not met (selected {combo:.2f}x)")
        elif mode == "min":
            if combo >= target:
                applied.append(f"target min {target:.2f}x")
            else:
                unmet.append(f"target min {target:.2f}x not met (selected {combo:.2f}x)")
        else:
            applied.append(f"target around {target:.2f}x")

    return applied, unmet


def format_constraint_report(applied: List[str], unmet: List[str]) -> str:
    lines = ["Constraint report:"]
    if applied:
        lines.append("- Applied: " + "; ".join(applied))
    else:
        lines.append("- Applied: none")
    if unmet:
        lines.append("- Unmet: " + "; ".join(unmet))
    else:
        lines.append("- Unmet: none")
    return "\n".join(lines)


def is_odds_parlay_request(user_q: str) -> bool:
    q = user_q.lower()
    has_parlay = "parlay" in q
    has_odds = "odds" in q or "odd" in q or "x" in q
    has_time = "today" in q or "tonight" in q or "upcoming" in q or "tomorrow" in q
    return has_parlay and (has_odds or has_time)


def is_schedule_request(user_q: str) -> bool:
    q = user_q.lower()
    schedule_terms = ["what matches", "fixtures", "games are on", "matches are on", "who plays", "schedule"]
    time_terms = ["today", "tonight", "tomorrow", "upcoming"]
    return any(t in q for t in schedule_terms) and any(t in q for t in time_terms)


def fetch_upcoming_odds(league: Optional[str], markets: str = "h2h", max_events: int = 10) -> Tuple[List[Dict], Optional[str]]:
    if not ODDS_API_KEY:
        return [], "Missing ODDS API key (ODDS-API)."

    sport_key = LEAGUE_TO_ODDS_SPORT.get(league or "EPL", "soccer_epl")
    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    warning: Optional[str] = None

    def _request(market_string: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
        params = {
            "api_key": ODDS_API_KEY,
            "regions": "uk,eu,us",
            "markets": market_string,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code != 200:
                body = (r.text or "")[:500]
                return None, f"Odds API HTTP {r.status_code}: {body}"
            return r.json(), None
        except Exception as exc:
            return None, f"Odds API request failed: {exc}"

    rows, req_err = _request(markets)
    effective_markets = markets
    if req_err and any(x in markets for x in ["corners", "cards"]):
        fallback = "h2h,totals,spreads"
        fb_rows, fb_err = _request(fallback)
        if fb_err:
            return [], req_err
        rows = fb_rows
        effective_markets = fallback
        warning = f"Requested markets '{markets}' not fully supported. Fell back to '{fallback}'."
    elif req_err:
        return [], req_err
    rows = rows or []

    requested_market_keys = {m.strip() for m in effective_markets.split(",") if m.strip()}
    parsed: List[Dict] = []
    for ev in rows:
        best_market_rows: Dict[Tuple[str, str, str], Dict] = {}
        best_prices: Dict[str, Dict] = {}
        for bm in ev.get("bookmakers", []):
            bm_title = bm.get("title")
            for mk in bm.get("markets", []):
                mk_key = str(mk.get("key") or "")
                if requested_market_keys and mk_key not in requested_market_keys:
                    continue
                for out in mk.get("outcomes", []):
                    name = out.get("name")
                    point = out.get("point")
                    price = out.get("price")
                    try:
                        price = float(price)
                    except (TypeError, ValueError):
                        continue

                    # Keep best price per event/market/outcome/point across books.
                    point_key = "" if point is None else str(point)
                    row_key = (mk_key, str(name), point_key)
                    cur = best_market_rows.get(row_key)
                    if (cur is None) or (price > float(cur.get("price", 0.0))):
                        best_market_rows[row_key] = {
                            "market_key": mk_key,
                            "name": name,
                            "point": point,
                            "price": price,
                            "bookmaker": bm_title,
                        }

                    # Backward-compatible best h2h summary.
                    if mk_key == "h2h":
                        if name not in best_prices or price > best_prices[name]["price"]:
                            best_prices[name] = {"price": price, "bookmaker": bm_title}

        parsed.append({
            "id": ev.get("id"),
            "commence_time": ev.get("commence_time"),
            "home_team": ev.get("home_team"),
            "away_team": ev.get("away_team"),
            "best_prices": best_prices,
            "market_prices": list(best_market_rows.values()),
        })

    parsed.sort(key=lambda x: x.get("commence_time") or "")
    out = parsed[:max_events]
    if not out:
        return [], "Odds API returned 0 upcoming events for this league/region."
    return out, warning


def filter_events_by_time(events: List[Dict], user_q: str, tz_name: str = "Europe/London") -> List[Dict]:
    q = user_q.lower()
    if not events:
        return events

    tz = ZoneInfo(tz_name)
    today = datetime.now(tz).date()

    if "today" in q or "tonight" in q:
        target_date = today
    elif "tomorrow" in q:
        target_date = today.fromordinal(today.toordinal() + 1)
    else:
        return events

    out = []
    for ev in events:
        ts = ev.get("commence_time")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(tz)
        except Exception:
            continue
        if dt.date() == target_date:
            out.append(ev)
    return out


def format_upcoming_odds_text(events: List[Dict]) -> str:
    if not events:
        return "No upcoming odds events available."
    lines = []
    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        when = ev.get("commence_time")
        bp = ev.get("best_prices", {})
        h_odds = bp.get(home, {}).get("price")
        a_odds = bp.get(away, {}).get("price")
        d_odds = bp.get("Draw", {}).get("price")
        lines.append(
            f"{home} vs {away} | {when} | H:{h_odds} D:{d_odds} A:{a_odds}"
        )
    return "\n".join(lines)


def format_fixture_list(events: List[Dict]) -> str:
    if not events:
        return "No upcoming fixtures found."
    lines = []
    for i, ev in enumerate(events, start=1):
        lines.append(f"{i}. {ev.get('home_team')} vs {ev.get('away_team')} | {ev.get('commence_time')}")
    return "\n".join(lines)


def get_team_profile_snippets_for_events(events: List[Dict],
                                         league: Optional[str],
                                         seasons: Optional[List[str]],
                                         per_team: int = 1,
                                         limit_teams: int = 12) -> List[Dict]:
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    col = db.get_or_create_collection(COLLECTION)

    teams: List[str] = []
    for ev in events:
        for t in [ev.get("home_team"), ev.get("away_team")]:
            if t and t not in teams:
                teams.append(t)
    teams = teams[:limit_teams]

    out: List[Dict] = []
    for team in teams:
        variants = list({team, team.lower(), team.title()})
        picked = 0
        for tv in variants:
            where = build_where(league, seasons, extra_filters=[{"doc_type": "team_profile"}, {"team": tv}])
            res = col.get(where=where, include=["documents", "metadatas"], limit=per_team)
            for i, d, m in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
                out.append({"id": i, "text": d, "meta": m})
                picked += 1
            if picked >= per_team:
                break
    return out


def get_team_fixture_snippets_for_events(events: List[Dict],
                                         league: Optional[str],
                                         seasons: Optional[List[str]],
                                         per_team: int = 2,
                                         limit_teams: int = 12) -> List[Dict]:
    """
    Pull a few recent team_fixture docs per team so odds-mode can reference
    concrete shots-on-target and conceded stats instead of only season summaries.
    """
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    col = db.get_or_create_collection(COLLECTION)

    teams: List[str] = []
    for ev in events:
        for t in [ev.get("home_team"), ev.get("away_team")]:
            if t and t not in teams:
                teams.append(t)
    teams = teams[:limit_teams]

    out_map: Dict[str, Dict] = {}
    for team in teams:
        variants = list({team, team.lower(), team.title()})
        picked = 0
        for tv in variants:
            where = build_where(league, seasons, extra_filters=[{"doc_type": "team_fixture"}, {"team": tv}])
            res = col.get(where=where, include=["documents", "metadatas"], limit=20)
            rows = []
            for i, d, m in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
                rows.append({"id": i, "text": d, "meta": m})
            rows.sort(key=lambda x: (x["meta"].get("fixture_date") or ""), reverse=True)
            for r in rows[:per_team]:
                out_map[r["id"]] = r
                picked += 1
            if picked >= per_team:
                break
    return list(out_map.values())


def get_player_snippets_for_events(events: List[Dict],
                                   league: Optional[str],
                                   seasons: Optional[List[str]],
                                   per_team_profiles: int = 4,
                                   per_team_fixtures: int = 4,
                                   limit_teams: int = 12) -> List[Dict]:
    """
    Pull player_profile + player_fixture docs for teams in selected events.
    This is required for open-ended player-prop recommendations in odds mode.
    """
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    col = db.get_or_create_collection(COLLECTION)

    teams: List[str] = []
    for ev in events:
        for t in [ev.get("home_team"), ev.get("away_team")]:
            if t and t not in teams:
                teams.append(t)
    teams = teams[:limit_teams]

    out_map: Dict[str, Dict] = {}
    for team in teams:
        variants = list({team, team.lower(), team.title()})

        # player_profile docs
        prof_picked = 0
        for tv in variants:
            where = build_where(league, seasons, extra_filters=[{"doc_type": "player_profile"}, {"team": tv}])
            res = col.get(where=where, include=["documents", "metadatas"], limit=per_team_profiles)
            for i, d, m in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
                out_map[i] = {"id": i, "text": d, "meta": m}
                prof_picked += 1
            if prof_picked >= per_team_profiles:
                break

        # player_fixture docs (semantic pull for foul/sot/shot terms)
        vec = embed([f"best player props for {team}: shots on target, shots, fouls, cards, minutes, role"])[0]
        fix_picked = 0
        for tv in variants:
            where = build_where(league, seasons, extra_filters=[{"doc_type": "player_fixture"}, {"team": tv}])
            res = col.query(query_embeddings=[vec], n_results=per_team_fixtures, where=where)
            if res.get("documents"):
                for d, m, i in zip(res["documents"][0], res["metadatas"][0], res["ids"][0]):
                    out_map[i] = {"id": i, "text": d, "meta": m}
                    fix_picked += 1
            if fix_picked >= per_team_fixtures:
                break
    return list(out_map.values())


def build_odds_parlay_prompt(user_q: str,
                             odds_text: str,
                             profile_docs: List[Dict],
                             target_multiplier: float) -> List[Dict]:
    prof_text = "\n\n".join(d["text"] for d in profile_docs[:12])

    system_msg = (
        "You are a football betting analyst.\n"
        "Use internal reasoning, but output only concise conclusions with numeric evidence.\n"
        "Build a same-day parlay from the provided upcoming odds board.\n"
        f"Target combined decimal odds should be close to or above {target_multiplier:.2f}x.\n"
        "Use team profile stats to select higher-probability legs, not just longest prices.\n"
        "Each leg must include at least 2 independent numeric signals.\n"
        "Prefer corners/cards/ML where justified; avoid single-match recency logic.\n"
        "Do not invent lines or odds not shown in the board.\n"
        "Output format:\n"
        "- Leg 1: <market + line + odds>\n"
        "  Evidence: <2-3 short numeric points>\n"
        "- Leg 2: ...\n"
        "- Leg 3: ...\n"
        "- Estimated combined odds: <number>\n"
        "- Why this parlay: <2-4 concise sentences>\n"
    )
    user_msg = (
        f"User request: {user_q}\n\n"
        "Upcoming odds board:\n"
        f"{odds_text}\n\n"
        "Team profile context:\n"
        f"{prof_text}\n\n"
        "Build the best parlay for the request."
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def format_selected_legs(legs: List[OddsLeg]) -> str:
    def label(leg: OddsLeg) -> str:
        if leg.market_key == "h2h":
            return f"{leg.outcome} (h2h)"
        if leg.market_key == "totals":
            if leg.point is None:
                return f"{leg.outcome} (totals)"
            return f"{leg.outcome} {leg.point:g} (totals)"
        if leg.market_key == "spreads":
            if leg.point is None:
                return f"{leg.outcome} (spread)"
            return f"{leg.outcome} {leg.point:+g} (spread)"
        if leg.point is None:
            return f"{leg.outcome} ({leg.market_key})"
        return f"{leg.outcome} {leg.point:g} ({leg.market_key})"

    lines = []
    for i, leg in enumerate(legs, start=1):
        bm = f" ({leg.bookmaker})" if leg.bookmaker else ""
        lines.append(f"{i}. {leg.fixture} | {label(leg)} @ {leg.odds:.2f}{bm}")
    return "\n".join(lines)


def build_odds_explain_prompt(user_q: str,
                              selected_legs: List[OddsLeg],
                              target_multiplier: Optional[float],
                              profile_docs: List[Dict],
                              team_fixture_docs: Optional[List[Dict]] = None,
                              player_docs: Optional[List[Dict]] = None) -> List[Dict]:
    legs_text = format_selected_legs(selected_legs)
    combo = combined_odds(selected_legs)
    prof_text = "\n\n".join(d["text"] for d in profile_docs[:12])
    tf_text = "\n\n".join(d["text"] for d in (team_fixture_docs or [])[:12])
    pl_text = "\n\n".join(d["text"] for d in (player_docs or [])[:16])

    system_msg = (
        "You are a football betting analyst.\n"
        "Do not change, replace, or add legs. Use the provided selected legs exactly as given.\n"
        "Your job is to explain why each selected leg is reasonable using profile stats.\n"
        "Use concise evidence with numeric values.\n"
        "Avoid single-match recency logic.\n"
        "Player props are allowed if player context exists. If missing, state that explicitly.\n"
        "Output format:\n"
        "- Leg 1: <repeat exact market>\n"
        "  Evidence: <2-3 short numeric points>\n"
        "- Leg 2: ...\n"
        "- Leg 3: ...\n"
        "- Estimated combined odds: <number>\n"
        "- Target check: <how close to requested multiplier>\n"
    )

    target_line = (
        f"Target combined odds: ~{target_multiplier:.2f}x"
        if target_multiplier is not None
        else "Target combined odds: not explicitly specified by user."
    )
    user_msg = (
        f"User request: {user_q}\n"
        f"{target_line}\n"
        f"Selected legs (fixed):\n{legs_text}\n"
        f"Current combined odds: {combo:.2f}\n\n"
        f"Team profile context:\n{prof_text}\n\n"
        f"Team fixture context:\n{tf_text or 'None'}\n\n"
        f"Player context:\n{pl_text or 'None'}\n\n"
        "Explain these selected legs."
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


# -------------------------
# Parse "Leeds vs West Ham" style questions
# -------------------------
def parse_match_request(user_q: str) -> dict:
    """
    Extract (home_team, away_team) from a question like:
      "Leeds is playing West Ham at home today"
      "Leeds vs West Ham"
      "Arsenal @ West Ham"
    Assumptions:
      - first team mentioned is home unless "@"
      - we lowercase teams to match metadata["team"] which we stored lowercase
    """
    q = user_q.lower()

    home_team = None
    away_team = None
    home_flag = None  # may be "home"

    # case 1: "<team1> is playing <team2>"
    if " is playing " in q:
        left, right = q.split(" is playing ", 1)
        home_team_candidate = left.strip()

        # strip common filler
        right_clean = (
            right.replace("at home", "")
                 .replace("today", "")
                 .replace("tonight", "")
                 .replace("this weekend", "")
        ).strip()

        # grab first 1-2 tokens as away team guess ("west ham")
        away_team_candidate = " ".join(right_clean.split(" ")[0:2]).strip()

        if "at home" in q or "at their place" in q:
            home_team = home_team_candidate
            away_team = away_team_candidate
            home_flag = "home"
        else:
            home_team = home_team_candidate
            away_team = away_team_candidate
            home_flag = "home"

    # case 2: "leeds vs west ham"
    if home_team is None and " vs " in q:
        a, b = q.split(" vs ", 1)
        home_team = a.strip()
        away_team = " ".join(b.strip().split(" ")[0:2]).strip()
        home_flag = "home"

    # case 3: "arsenal @ west ham"
    if home_team is None and " @ " in q:
        a, b = q.split(" @ ", 1)
        away_team = a.strip()  # the team before '@' is traveling
        home_team = " ".join(b.strip().split(" ")[0:2]).strip()
        home_flag = "home"

    def norm_team(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        return name.strip().lower()

    # canonical aliases to match stored team names
    alias_map = {
        "spurs": "Tottenham",
        "tottenham hotspur": "Tottenham",
        "man utd": "Manchester United",
        "man united": "Manchester United",
        "manchester utd": "Manchester United",
        "united": "Manchester United",  # EPL-focused; adjust if multi-league ambiguity
        "wolves": "Wolves",
        "nottm forest": "Nottingham Forest",
        "forest": "Nottingham Forest",
        "brighton & hove albion": "Brighton",
    }

    def canonical(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        key = name.strip().lower()
        return alias_map.get(key, name.title())

    return {
        "home_team": canonical(home_team),
        "away_team": canonical(away_team),
        "home_flag": home_flag,
    }


# -------------------------
# Detect team names when "vs" not explicit
# -------------------------
EPL_TEAMS = {
    "arsenal", "aston villa", "bournemouth", "brentford", "brighton", "burnley",
    "chelsea", "crystal palace", "everton", "fulham", "leeds", "leicester",
    "liverpool", "luton", "manchester city", "manchester united", "man united",
    "man utd", "newcastle", "nottingham forest", "forest", "southampton",
    "tottenham", "spurs", "west ham", "wolves", "wolverhampton"
}

def detect_teams_from_query(user_q: str) -> List[str]:
    q = user_q.lower()
    found = []
    for t in EPL_TEAMS:
        if t in q:
            found.append(t)
    # de-dup preserving order of appearance
    seen = set()
    ordered = []
    for t in sorted(found, key=lambda x: q.index(x)):
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered[:2]


def is_add_leg_request(user_q: str) -> bool:
    q = user_q.lower()
    return any(phrase in q for phrase in [
        "add one more leg",
        "add another leg",
        "one more leg",
        "add a leg",
    ])


def is_followup_request(user_q: str) -> bool:
    q = user_q.lower().strip()
    follow_tokens = [
        "another",
        "one more",
        "same game",
        "that game",
        "this game",
        "for this one",
        "for this match",
        "based on that",
        "from that",
        "now add",
        "give me another",
    ]
    if any(tok in q for tok in follow_tokens):
        return True
    # Very short prompts after a prior turn are usually continuation asks.
    if len(q.split()) <= 7 and memory.get("last_query"):
        return True
    return False


def enrich_query_with_memory(user_q: str,
                             has_matchup_in_query: bool) -> str:
    """
    Make follow-up prompts self-contained by carrying last matchup context.
    """
    if has_matchup_in_query:
        return user_q
    if not (memory.get("last_home") and memory.get("last_away")):
        return user_q
    if not is_followup_request(user_q):
        return user_q
    return (
        f"For the same matchup {memory['last_home']} vs {memory['last_away']}, {user_q}\n"
        f"Previous assistant answer:\n{memory.get('last_answer') or memory.get('last_parlay') or 'n/a'}"
    )


# -------------------------
# Focused retrieval for a single team
# -------------------------
def get_team_context(team_name: str,
                     league: Optional[str],
                     seasons: Optional[List[str]],
                     k_fixture: int = 6) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns:
    - profiles: list of {id, text, meta} for team_profile docs
    - fixtures: list of {id, text, meta} for team_fixture docs
    """
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    col = db.get_or_create_collection(COLLECTION)

    # ---- team_profile docs ----
    where_profile = build_where(
        league,
        seasons,
        extra_filters=[{"doc_type": "team_profile"}, {"team": team_name}],
    )

    prof_res = col.get(
        where=where_profile,
        include=["documents", "metadatas"],  # <-- 'ids' removed
        limit=3,
    )

    profiles: List[Dict] = []
    # NOTE: col.get(...) returns dict with keys "ids", "documents", "metadatas", ...
    for i, d, m in zip(prof_res.get("ids", []),
                       prof_res.get("documents", []),
                       prof_res.get("metadatas", [])):
        profiles.append({"id": i, "text": d, "meta": m})

    # ---- team_fixture docs ----
    # Fetch a wider set and sort by fixture_date descending to avoid single-match bias
    where_fixture = build_where(
        league,
        seasons,
        extra_filters=[{"doc_type": "team_fixture"}, {"team": team_name}],
    )

    fix_res = col.get(
        where=where_fixture,
        include=["documents", "metadatas"],
        limit=200,
    )

    fixtures_all: List[Dict] = []
    # chroma returns ids separately even without include
    ids = fix_res.get("ids", [])
    docs = fix_res.get("documents", [])
    metas = fix_res.get("metadatas", [])
    for i, d, m in zip(ids, docs, metas):
        fixtures_all.append({"id": i, "text": d, "meta": m})

    def sort_key(rec):
        md = rec["meta"]
        fd = md.get("fixture_date") or ""
        return fd

    fixtures_all.sort(key=sort_key, reverse=True)
    fixtures = fixtures_all[:k_fixture]

    return profiles, fixtures


def get_team_player_context(team_name: str,
                            league: Optional[str],
                            seasons: Optional[List[str]],
                            k_player_profile: int = 6,
                            k_player_fixture: int = 8) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns:
    - player_profiles: list of {id, text, meta} for player_profile docs scoped to team
    - player_fixtures: list of {id, text, meta} for player_fixture docs scoped to team
    """
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    col = db.get_or_create_collection(COLLECTION)

    # Player docs in index currently use lowercase team metadata;
    # team docs use title-case. Query both forms and merge.
    team_variants = list({team_name, team_name.lower(), team_name.title()})

    player_profiles_map: Dict[str, Dict] = {}
    for tv in team_variants:
        where_player_profile = build_where(
            league,
            seasons,
            extra_filters=[{"doc_type": "player_profile"}, {"team": tv}],
        )
        prof_res = col.get(where=where_player_profile, include=["documents", "metadatas"], limit=k_player_profile)
        for i, d, m in zip(prof_res.get("ids", []), prof_res.get("documents", []), prof_res.get("metadatas", [])):
            player_profiles_map[i] = {"id": i, "text": d, "meta": m}

    # semantic query helps bring form-relevant player fixture docs
    vec = embed([f"recent player form, shots on target, cards, fouls for {team_name}"])[0]
    player_fixtures_map: Dict[str, Dict] = {}
    for tv in team_variants:
        where_player_fixture = build_where(
            league,
            seasons,
            extra_filters=[{"doc_type": "player_fixture"}, {"team": tv}],
        )
        fx_res = col.query(query_embeddings=[vec], n_results=k_player_fixture, where=where_player_fixture)
        if fx_res.get("documents"):
            for d, m, i in zip(fx_res["documents"][0], fx_res["metadatas"][0], fx_res["ids"][0]):
                player_fixtures_map[i] = {"id": i, "text": d, "meta": m}

    player_profiles = list(player_profiles_map.values())[:k_player_profile]
    player_fixtures = list(player_fixtures_map.values())[:k_player_fixture]
    return player_profiles, player_fixtures



# -------------------------
# Build matchup prompt (two-team parlay reasoning)
# -------------------------
def build_matchup_prompt(user_q: str,
                         home_team_block: dict,
                         away_team_block: dict,
                         home_player_block: Optional[dict] = None,
                         away_player_block: Optional[dict] = None,
                         mentioned_player_docs: Optional[List[Dict]] = None,
                         chat_context: Optional[str] = None,
                         graph_snapshot_text: Optional[str] = None) -> List[Dict]:
    """
    home_team_block = {
        "team_name": "leeds",
        "profiles": [...],
        "fixtures": [...],
    }
    away_team_block = same structure.
    We assume home_team_block["team_name"] is actually the home side in this matchup.
    """

    def pack(block):
        team = block["team_name"]
        prof_txt = "\n\n".join(d["text"] for d in block["profiles"][:2])  # season profile (corners_pm etc)
        fixt_txt = "\n\n".join(d["text"] for d in block["fixtures"][:5])  # recent fixtures (per-match)
        return (
            f"=== {team.upper()} SEASON PROFILE ===\n{prof_txt}\n\n"
            f"=== {team.upper()} RECENT FIXTURES ===\n{fixt_txt}\n"
        )

    def pack_players(block):
        if not block:
            return "No player context provided."
        team = block["team_name"]
        p_prof = "\n\n".join(d["text"] for d in block["profiles"][:3])
        p_fix = "\n\n".join(d["text"] for d in block["fixtures"][:4])
        return (
            f"=== {team.upper()} PLAYER PROFILES ===\n{p_prof}\n\n"
            f"=== {team.upper()} PLAYER FIXTURES ===\n{p_fix}\n"
        )

    # Heuristic preview from team profiles (uses metadata to prime the LLM)
    def meta_from_profile(block):
        # pick first profile meta as representative
        if not block["profiles"]:
            return {}
        return block["profiles"][0].get("meta", {})

    home_meta = meta_from_profile(home_team_block)
    away_meta = meta_from_profile(away_team_block)
    heuristic_lines = []
    if home_meta and away_meta:
        picks = top_markets(home_meta, away_meta, n=5)
        for i, p in enumerate(picks, 1):
            heuristic_lines.append(f"Heuristic {i}: {p['market']} | score={p['score']:.2f} | {p['rationale']}")
    heuristic_text = "\n".join(heuristic_lines)

    matchup_context = (
        "----- RECENT CHAT CONTEXT -----\n"
        + (chat_context or "None.")
        + "\n"
        "----- HOME TEAM BLOCK -----\n"
        + pack(home_team_block)
        + "\n----- AWAY TEAM BLOCK -----\n"
        + pack(away_team_block)
        + "\n----- HOME PLAYER BLOCK -----\n"
        + pack_players(home_player_block)
        + "\n----- AWAY PLAYER BLOCK -----\n"
        + pack_players(away_player_block)
        + "\n----- EXPLICITLY MENTIONED PLAYERS -----\n"
        + ("\n\n".join(d["text"] for d in (mentioned_player_docs or [])[:8]) if mentioned_player_docs else "None")
        + ("\n----- GRAPH SNAPSHOT -----\n" + (graph_snapshot_text or "none"))
        + ("\n----- HEURISTIC SNAPSHOT -----\n" + heuristic_text if heuristic_text else "")
    )

    system_msg = (
        "You are building a SAME-GAME PARLAY for sports betting.\n"
        "Use chain-of-thought style analysis internally, but do NOT expose hidden reasoning.\n"
        "Return only concise, evidence-backed conclusions.\n\n"
        "REASONING PROTOCOL (internal):\n"
        "1. Build a candidate slate from goals, corners, cards, and ML/double-chance angles.\n"
        "2. Score each candidate using multi-signal evidence:\n"
        "   - form_index_team, control_index, aggression_index_norm\n"
        "   - corners_pm vs corners_against_pm and corner_edge_pm\n"
        "   - cards_per_90_team and fouls_per_90_team\n"
        "   - graph snapshot cohorts and heuristic snapshot\n"
        "3. Reject weak candidates:\n"
        "   - single-stat rationale\n"
        "   - single prior H2H as main justification\n"
        "   - missing numeric support from context\n"
        "4. Select best 2-3 legs with diversified market types when data supports it.\n\n"
        "RULES:\n"
        "- Each leg MUST cite at least 2 independent numeric signals.\n"
        "- At least one NON-goal leg (corners/cards) when available.\n"
        "- Prefer shortlist markets from HEURISTIC SNAPSHOT unless contradicted by stronger evidence.\n"
        "- DO NOT invent numbers. If data is missing, say so.\n"
        "- Actively evaluate player props when player context exists, and pick the top-N most probable props requested by the user.\n"
        "- Use sample-size safety for player props: avoid tiny-sample players unless no better evidence exists.\n"
        "- Rank player props by multi-signal support (recent form + role/usage + opponent profile), not single-match recency.\n"
        "- If mentioning past H2H, treat it as secondary evidence only.\n\n"
        "OUTPUT FORMAT:\n"
        "- Leg 1: <market>\n"
        "  Evidence: <2-3 short numeric points>\n"
        "- Leg 2: <market>\n"
        "  Evidence: <2-3 short numeric points>\n"
        "- Leg 3: <market>\n"
        "  Evidence: <2-3 short numeric points>\n"
        "- Why this parlay makes sense: <2-4 concise sentences on style/form matchup>\n"
    )

    user_msg = (
        f"User request: {user_q}\n\n"
        "We consider the HOME team to be the first block below and the AWAY team to be the second block.\n"
        "Use ONLY the stats below. Do not guess form for players not mentioned.\n\n"
        f"{matchup_context}\n\n"
        "Now build the parlay."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# -------------------------
# Build generic RAG prompt (single-team or stat lookup)
# -------------------------
def build_generic_prompt(user_q: str,
                         context_docs: List[Dict],
                         league: Optional[str],
                         chat_context: Optional[str] = None) -> List[Dict]:
    has_player_context = any(
        (d.get("meta") or {}).get("doc_type") in {"player_profile", "player_fixture"}
        for d in context_docs
    )

    lead = (
        "You are a football analyst.\n"
        "Use chain-of-thought analysis internally, but do not reveal hidden reasoning.\n"
        "Return concise, evidence-first answers only.\n"
        "- Only use the provided context.\n"
        "- Quote exact numbers where possible (e.g. cards_per_90_team, corners_against_pm, control_index, form_index_team).\n"
        "- Stay in the given league unless told otherwise.\n"
        "- If data is missing, say you don't have it.\n"
        "- If predicting a fixture outcome, justify with at least 2 numeric signals and avoid single-match recency.\n"
    )
    if has_player_context:
        lead += (
            "- Player props are allowed even if the user did not name a player. "
            "Pick the top likely player props from context using multi-signal evidence "
            "(minutes/sample + role/usage + opponent profile).\n"
        )
    else:
        lead += "- If no player docs are present, explicitly state player-prop limitation.\n"

    if league:
        lead += f"\nThis league is {league}. Do not mix in other leagues.\n"

    ctx = "\n\n".join(d["text"] for d in context_docs[:8])

    return [
        {"role": "system", "content": lead},
        {
            "role": "user",
            "content": (
                f"Question: {user_q}\n\n"
                f"Recent chat context:\n{chat_context or 'None.'}\n\n"
                f"Context:\n{ctx}\n\n"
                "Answer clearly."
            ),
        }
    ]


# -------------------------
# Pretty-print sources
# -------------------------
def format_sources(hits: List[Dict], max_sources: int = 15) -> str:
    lines = []
    for h in hits[:max_sources]:
        m = h["meta"]
        tag = f"{m.get('doc_type','?')} · {m.get('league','?')} {m.get('season','?')}"
        who = m.get("player_name") or m.get("team")
        fx  = m.get("fixture")
        lines.append(f"- {who or 'N/A'} | {tag}" + (f" | {fx}" if fx else ""))
    return "Sources:\n" + "\n".join(lines)


# -------------------------
# Main chat logic
# -------------------------
def chat_once(user_q: str,
              league_flag: Optional[str],
              seasons_arg: Optional[str],
              k: int,
              show_sources: bool):

    # Resolve target league
    league = resolve_league_from_flag_or_query(league_flag, user_q)

    # Parse seasons string into list
    seasons = parse_seasons_arg(seasons_arg)

    # Carry forward context for follow-up chat turns.
    preview_matchup = parse_match_request(user_q)
    has_matchup_in_query = bool(preview_matchup.get("home_team") and preview_matchup.get("away_team"))
    user_q = enrich_query_with_memory(user_q, has_matchup_in_query)
    chat_context_text = render_recent_chat_context(max_turns=3)

    # Odds schedule mode: "What matches are on today?"
    if is_schedule_request(user_q):
        events, err = fetch_upcoming_odds(league, markets="h2h", max_events=50)
        if err and not events:
            print(f"Could not fetch upcoming fixtures: {err}")
            return
        if err:
            print(f"Odds note: {err}")
        events = filter_events_by_time(events, user_q)
        if not events:
            print("No fixtures found for the requested time window.")
            return
        print("\nUpcoming fixtures:")
        print(format_fixture_list(events))
        return

    # Odds-first mode for requests like "make me a 3x odds parlay from today's matches"
    if is_odds_parlay_request(user_q):
        constraints = parse_odds_constraints(user_q)
        req_match_count = extract_match_count(user_q)
        requested_markets = constraints["requested_markets"]
        target_mult = constraints["target_multiplier"]
        target_mode = constraints["target_mode"]
        events, err = fetch_upcoming_odds(league, markets=",".join(requested_markets), max_events=50)
        if events:
            if err:
                print(f"Odds note: {err}")
            events = filter_events_by_time(events, user_q)
            if not events:
                print("No fixtures found for the requested time window.")
                return

            constraints = apply_odds_constraints(constraints, events)
            leg_count = int(constraints["leg_count"])
            require_unique_events = bool(constraints["require_unique_events"])

            if req_match_count is not None and len(events) != req_match_count:
                print(
                    f"Requested {req_match_count} matches, but found {len(events)} for the requested window."
                )
                print("Fixtures found:")
                print(format_fixture_list(events))
                return

            candidates = extract_candidates(events, allowed_market_keys=set(requested_markets))
            chosen_legs, sel_err = select_best_parlay(
                candidates=candidates,
                leg_count=leg_count,
                target_multiplier=target_mult,
                target_mode=target_mode,
                require_unique_events=require_unique_events,
            )
            if sel_err:
                print(f"Could not build a valid parlay: {sel_err}")
                print("Tip: request fewer legs or broaden requested markets.")
                return

            applied_constraints, unmet_constraints = summarize_constraint_results(constraints, chosen_legs)

            selected_event_ids = {x.event_id for x in chosen_legs}
            selected_events = [ev for ev in events if str(ev.get("id") or "") in selected_event_ids]
            prof_docs = get_team_profile_snippets_for_events(selected_events, league, seasons, per_team=1)
            team_fixture_docs = get_team_fixture_snippets_for_events(selected_events, league, seasons, per_team=2)
            player_docs = get_player_snippets_for_events(selected_events, league, seasons, per_team_profiles=4, per_team_fixtures=4)
            messages = build_odds_explain_prompt(
                user_q,
                chosen_legs,
                target_mult,
                prof_docs,
                team_fixture_docs=team_fixture_docs,
                player_docs=player_docs,
            )
            resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
            answer = resp.choices[0].message.content
            print("\n" + answer.strip())
            print("\n" + format_constraint_report(applied_constraints, unmet_constraints))
            add_chat_turn(user_q, answer.strip())
            if show_sources:
                print("\nSources:")
                print(f"- odds_api | upcoming_board | markets={','.join(requested_markets)}")
                for d in prof_docs[:10]:
                    m = d["meta"]
                    print("-", m.get("team"), "|", m.get("doc_type"), "|", m.get("season"))
                for d in team_fixture_docs[:8]:
                    m = d["meta"]
                    print("-", m.get("team"), "|", m.get("doc_type"), "|", m.get("season"), "|", m.get("fixture"))
                for d in player_docs[:10]:
                    m = d["meta"]
                    print("-", m.get("player_name") or m.get("team"), "|", m.get("doc_type"), "|", m.get("season"), "|", m.get("fixture"))
            return
        else:
            print(f"Odds API unavailable for parlay mode: {err}")
            print("Falling back to standard RAG flow.")

    # Try matchup mode first
    # Memory-assisted "add leg" handling
    add_leg = is_add_leg_request(user_q)
    if add_leg and memory["last_home"] and memory["last_away"]:
        user_q = (
            f"Add one more leg to previous parlay for {memory['last_home']} vs {memory['last_away']}."
            f" Previous parlay:\n{memory['last_parlay'] or 'n/a'}\n"
            "Return only the new leg, distinct from prior legs, still multi-signal justified."
        )

    matchup = parse_match_request(user_q)
    home_team = matchup["home_team"]
    away_team = matchup["away_team"]

    # If parsing failed to find two teams, try detection from query tokens
    if not (home_team and away_team):
        detected = detect_teams_from_query(user_q)
        if len(detected) == 2:
            home_team, away_team = detected[0].title(), detected[1].title()

    if home_team and away_team:
        # Graph snapshot (cohort aggregates)
        graph_snap = get_matchup_snapshot(home_team, away_team, last_n=6)
        graph_snap_text = snapshot_to_text(graph_snap) if graph_snap else "Graph snapshot unavailable."

        # Pull both teams
        home_profiles, home_fixtures = get_team_context(home_team, league, seasons, k_fixture=12)
        away_profiles, away_fixtures = get_team_context(away_team, league, seasons, k_fixture=12)
        home_player_profiles, home_player_fixtures = get_team_player_context(home_team, league, seasons)
        away_player_profiles, away_player_fixtures = get_team_player_context(away_team, league, seasons)
        mentioned_player_docs = get_mentioned_player_docs(
            user_q,
            league,
            seasons,
            max_players=4,
            per_player_profiles=1,
            per_player_fixtures=3,
        )

        # If we didn't find anything for one of the teams WITH the provided seasons,
        # retry once without seasons filter (in case season mismatch like UNKNOWN)
        if (not home_profiles and not home_fixtures) or (not away_profiles and not away_fixtures):
            home_profiles, home_fixtures = get_team_context(home_team, league, None, k_fixture=6)
            away_profiles, away_fixtures = get_team_context(away_team, league, None, k_fixture=6)
            home_player_profiles, home_player_fixtures = get_team_player_context(home_team, league, None)
            away_player_profiles, away_player_fixtures = get_team_player_context(away_team, league, None)
            if not mentioned_player_docs:
                mentioned_player_docs = get_mentioned_player_docs(
                    user_q,
                    league,
                    None,
                    max_players=4,
                    per_player_profiles=1,
                    per_player_fixtures=3,
                )

        # If we now have data for both, build a matchup parlay answer
        if (home_profiles or home_fixtures) and (away_profiles or away_fixtures):
            home_block = {"team_name": home_team, "profiles": home_profiles, "fixtures": home_fixtures}
            away_block = {"team_name": away_team, "profiles": away_profiles, "fixtures": away_fixtures}
            home_player_block = {
                "team_name": home_team,
                "profiles": home_player_profiles,
                "fixtures": home_player_fixtures,
            }
            away_player_block = {
                "team_name": away_team,
                "profiles": away_player_profiles,
                "fixtures": away_player_fixtures,
            }

            messages = build_matchup_prompt(
                user_q,
                home_block,
                away_block,
                home_player_block=home_player_block,
                away_player_block=away_player_block,
                mentioned_player_docs=mentioned_player_docs,
                chat_context=chat_context_text,
                graph_snapshot_text=graph_snap_text,
            )
            resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
            answer = resp.choices[0].message.content
            print("\n" + answer.strip())

            # update memory
            memory["last_home"] = home_team
            memory["last_away"] = away_team
            memory["last_parlay"] = answer.strip()
            memory["last_query"] = user_q
            add_chat_turn(user_q, answer.strip())

            if show_sources:
                print("\nSources:")
                for d in (
                    home_profiles[:1]
                    + home_fixtures[:3]
                    + away_profiles[:1]
                    + away_fixtures[:3]
                    + home_player_profiles[:2]
                    + home_player_fixtures[:2]
                    + away_player_profiles[:2]
                    + away_player_fixtures[:2]
                    + (mentioned_player_docs[:3] if mentioned_player_docs else [])
                ):
                    m = d["meta"]
                    print("-", m.get("team"),
                          "|", m.get("doc_type"),
                          "|", m.get("season"),
                          "|", m.get("fixture"))
            return

        # if we fall through here, we just didn't have enough matchup data; continue to generic mode

    # Generic fallback mode:
    hits = generic_retrieve(user_q, league=league, seasons=seasons, k=k)

    # If no hits with season restriction, retry w/out seasons
    if not hits and seasons:
        hits = generic_retrieve(user_q, league=league, seasons=None, k=k)

    if not hits:
        print("No results found in the index for your filters. Try widening league/season or re-ingest.")
        return

    messages = build_generic_prompt(user_q, hits, league, chat_context=chat_context_text)
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
    answer = resp.choices[0].message.content

    print("\n" + answer.strip())
    add_chat_turn(user_q, answer.strip())
    if show_sources:
        print("\n" + format_sources(hits))


def main():
    ap = argparse.ArgumentParser(description="Terminal RAG over your football KB")
    ap.add_argument("--league", type=str, default="EPL",
                    help="EPL|LaLiga|SerieA|Bundesliga|Ligue1 (default: EPL)")
    ap.add_argument("--season", type=str, default=None,
                    help='Comma-separated seasons like "2024/25,2025/26", or leave unset')
    ap.add_argument("--topk", type=int, default=12,
                    help="Top-k docs to retrieve for generic mode (default: 12)")
    ap.add_argument("--no-sources", action="store_true",
                    help="Hide sources footer")
    ap.add_argument("--once", type=str, default=None,
                    help="Run a single query then exit")
    args = ap.parse_args()

    # DB sanity
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    names = [c.name for c in db.list_collections()]
    if COLLECTION not in names:
        print(f"Collection '{COLLECTION}' not found at {CHROMA_DIR}. Did you run 01_embed_and_upsert.py?")
        sys.exit(1)

    if args.once:
        chat_once(args.once, args.league, args.season, args.topk, show_sources=not args.no_sources)
        return

    # Interactive loop
    print(f"RAG ready. League={args.league or 'auto'} Season={args.season or 'any'} | type 'exit' to quit.")
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
        chat_once(q, args.league, args.season, args.topk, show_sources=not args.no_sources)


if __name__ == "__main__":
    main()
