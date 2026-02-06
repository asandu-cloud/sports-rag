#!/usr/bin/env python3
# Small local RAG trial (matchup-aware)

import os
import sys
import argparse
from typing import List, Dict, Optional, Tuple
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from heuristics import top_markets

# ---- CONFIG ----
CHROMA_DIR = "/Users/sanduandrei/Desktop/Betting_RAG/Index/chroma"
COLLECTION = "football_top5"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o"  # choose your chat model here

load_dotenv()
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

    # Prefer team profiles then team fixtures before anything else
    hits = q_with_extra([{"doc_type": "team_profile"}], k)
    if not hits:
        hits = q_with_extra([{"doc_type": "team_fixture"}], k)
    if not hits:
        hits = q_with_extra([], k)
    return hits


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



# -------------------------
# Build matchup prompt (two-team parlay reasoning)
# -------------------------
def build_matchup_prompt(user_q: str,
                         home_team_block: dict,
                         away_team_block: dict) -> List[Dict]:
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
        "----- HOME TEAM BLOCK -----\n"
        + pack(home_team_block)
        + "\n----- AWAY TEAM BLOCK -----\n"
        + pack(away_team_block)
        + ("\n----- HEURISTIC SNAPSHOT -----\n" + heuristic_text if heuristic_text else "")
    )

    system_msg = (
        "You are building a SAME-GAME PARLAY for sports betting.\n"
        "You MUST reason about the MATCHUP between the two teams, not just historical averages in isolation.\n\n"

        "INSTRUCTIONS:\n"
        "1. Matchup reasoning:\n"
        "   - Compare Home Team's attacking output vs Away Team's defensive concessions.\n"
        "   - Compare Home Team's corners_pm vs Away Team's corners_against_pm (and vice versa).\n"
        "   - Compare fouls_per_90_team and cards_per_90_team between both teams to justify card-related legs.\n"
        "   - Use corner_edge_pm (corners_for - corners_against) to argue which team tends to dominate corners.\n"
        "   - Use goals_for_pm, xG, goals_conceded trends from recent fixtures to justify BTTS / Over / Under.\n"
        "   - Explicitly say things like: 'Leeds typically wins X corners per match and West Ham typically concedes Y corners per match, so Leeds Over corners / Leeds most corners is live'.\n\n"

        "2. Output requirements:\n"
        "   - Return 2 or 3 legs max (user might say 2-leg, honor that).\n"
        "   - Each leg MUST be matchup-informed. NO generic 'Team under 1.5' unless you justify it using BOTH sides.\n"
        "   - For each leg, explain in 1-2 sentences using actual numeric stats from the context (use exact numbers like 'corners_pm 7.8', 'corners_against_pm 6.1', 'cards_per_90_team 2.0').\n"
        "   - Legs should be realistic sportsbook-style markets: "
        "      • Team Over X.5 Corners / Team To Win Corners\n"
        "      • Both Teams To Score / Over 2.5 Goals / Under 1.5 Team Goals\n"
        "      • Over X.5 Total Cards / Specific Team Over X.5 Cards\n"
        "      • etc.\n"
        "   - DO NOT propose player props unless BOTH:\n"
        "       • The user explicitly asked for that player, AND\n"
        "       • The context shows that player with meaningful sample (>=300 minutes or multiple matches).\n"
        "     Otherwise, avoid player props entirely.\n"
        "   - DO NOT invent numbers that are not explicitly in the context.\n"
        "   - If data is missing for one angle, say so and pick a different angle.\n\n"
        "   - At least one leg should be NON-GOAL (cards or corners) when the context has those stats; avoid delivering only goal totals unless nothing else is supported.\n"
        "   - Prefer picking from the HEURISTIC SNAPSHOT shortlist; only propose markets not listed there if the context strongly supports them.\n"
        "   - Each leg must be justified by at least TWO distinct stats (e.g., form_index_team + control_index, or corners_pm + corners_against_pm). Single-stat or single-H2H justifications are not allowed.\n"
        "   - Prefer evidence that crosses styles: control_index vs aggression_index_norm, corner_edge_pm vs corners_against_pm, form_index_team vs recent goals.\n"
        "   - DO NOT base legs on a single prior head-to-head. If you mention a past meeting, back it up with current form, control, aggression, and opponent archetype fit.\n"
        "   - Consider HEURISTIC SNAPSHOT above as a prior; you may override if context contradicts it.\n\n"

        "3. Tone:\n"
        "   - Present the final answer as 'Leg 1, Leg 2, (Leg 3 ... if user asked for 3)'.\n"
        "   - After the legs, include a short 'Why this parlay makes sense' section summarizing the interaction of styles.\n"
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
                         league: Optional[str]) -> List[Dict]:
    lead = (
        "You are a football analyst.\n"
        "- Only use the provided context.\n"
        "- Quote exact numbers where possible (e.g. cards_per_90_team, corners_against_pm, etc.).\n"
        "- Stay in the given league unless told otherwise.\n"
        "- If data is missing, say you don't have it.\n"
        "- If predicting a fixture outcome, explain reasoning using the stats you see.\n"
    )
    if league:
        lead += f"\nThis league is {league}. Do not mix in other leagues.\n"

    ctx = "\n\n".join(d["text"] for d in context_docs[:8])

    return [
        {"role": "system", "content": lead},
        {"role": "user", "content": f"Question: {user_q}\n\nContext:\n{ctx}\n\nAnswer clearly."}
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

    # Try matchup mode first
    matchup = parse_match_request(user_q)
    home_team = matchup["home_team"]
    away_team = matchup["away_team"]

    # If parsing failed to find two teams, try detection from query tokens
    if not (home_team and away_team):
        detected = detect_teams_from_query(user_q)
        if len(detected) == 2:
            home_team, away_team = detected[0].title(), detected[1].title()

    if home_team and away_team:
        # Pull both teams
        home_profiles, home_fixtures = get_team_context(home_team, league, seasons, k_fixture=12)
        away_profiles, away_fixtures = get_team_context(away_team, league, seasons, k_fixture=12)

        # If we didn't find anything for one of the teams WITH the provided seasons,
        # retry once without seasons filter (in case season mismatch like UNKNOWN)
        if (not home_profiles and not home_fixtures) or (not away_profiles and not away_fixtures):
            home_profiles, home_fixtures = get_team_context(home_team, league, None, k_fixture=6)
            away_profiles, away_fixtures = get_team_context(away_team, league, None, k_fixture=6)

        # If we now have data for both, build a matchup parlay answer
        if (home_profiles or home_fixtures) and (away_profiles or away_fixtures):
            home_block = {"team_name": home_team, "profiles": home_profiles, "fixtures": home_fixtures}
            away_block = {"team_name": away_team, "profiles": away_profiles, "fixtures": away_fixtures}

            messages = build_matchup_prompt(user_q, home_block, away_block)
            resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
            answer = resp.choices[0].message.content
            print("\n" + answer.strip())

            if show_sources:
                print("\nSources:")
                for d in (home_profiles[:1] + home_fixtures[:4] + away_profiles[:1] + away_fixtures[:4]):
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

    messages = build_generic_prompt(user_q, hits, league)
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
    answer = resp.choices[0].message.content

    print("\n" + answer.strip())
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
