"""Whale-exclusive conversational AI — natural language access to the prediction engine.

Whale users get a dedicated private thread under #whale-ai where they just type
naturally, like texting a personal analyst. No slash commands needed inside the thread.

Architecture:
    - on_member_update: detects Whale role assignment → DMs welcome + creates thread
    - on_message: listens in whale chat threads → GPT with tool calling → responds
    - /chat: thin shortcut from any channel → redirects to thread
    - matchday_nudge: proactive "X matches today, want me to scan?" in active threads

Conversation memory comes from Discord thread history (last 15 messages).
No changes to the existing pipeline — this is a read-only consumer of existing functions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Optional

import discord
from discord import app_commands
from discord.ext import commands, tasks

# Path setup
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag_ingest"))

from dotenv import load_dotenv
load_dotenv()

from config import COLOR_BLUE, COLOR_PURPLE, ROLE_TO_TIER, TIER_HIERARCHY

log = logging.getLogger("whale_chat")

# Thread pool for blocking calls
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="whale_chat")

# Channel ID for whale-ai
WHALE_AI_CHANNEL = int(os.getenv("DISCORD_CHANNEL_WHALE_AI", "0"))

# Thread name suffix
_THREAD_SUFFIX = "-chat"

# ---------------------------------------------------------------------------
# OpenAI setup
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
    _OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
    _client = OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None
    _CHAT_MODEL = os.getenv("WHALE_CHAT_MODEL", os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini"))
    _HAS_OPENAI = _client is not None
except ImportError:
    _HAS_OPENAI = False
    _client = None
    _CHAT_MODEL = ""

# ---------------------------------------------------------------------------
# Tier check helper
# ---------------------------------------------------------------------------

def _is_whale(member: discord.Member) -> bool:
    """Check if a member has Whale tier or higher."""
    best_rank = 0
    for role in member.roles:
        tier = ROLE_TO_TIER.get(role.name.lower())
        if tier:
            rank = TIER_HIERARCHY.get(tier, 0)
            if rank > best_rank:
                best_rank = rank
    return best_rank >= TIER_HIERARCHY.get("elite", 3)


def _get_thread_name(user: discord.User | discord.Member) -> str:
    """Get the thread name for a user."""
    return f"{user.display_name}{_THREAD_SUFFIX}"


# ---------------------------------------------------------------------------
# Thread management
# ---------------------------------------------------------------------------

async def _find_user_thread(channel: discord.TextChannel, user: discord.User | discord.Member) -> Optional[discord.Thread]:
    """Find an existing chat thread for a user in the whale-ai channel."""
    thread_name = _get_thread_name(user)

    # Check active threads
    for thread in channel.threads:
        if thread.name == thread_name and thread.is_private():
            return thread

    # Check archived threads
    try:
        async for thread in channel.archived_threads(private=True, limit=100):
            if thread.name == thread_name:
                return thread
    except Exception:
        pass

    return None


async def _create_user_thread(channel: discord.TextChannel, user: discord.User | discord.Member) -> Optional[discord.Thread]:
    """Create a new private chat thread for a Whale user."""
    thread_name = _get_thread_name(user)

    try:
        thread = await channel.create_thread(
            name=thread_name,
            type=discord.ChannelType.private_thread,
            invitable=False,
        )
        # Add the user and send greeting
        await thread.send(
            content=f"{user.mention}",
        )
        await thread.send(
            "This is your private AI channel. Ask me anything about "
            "upcoming fixtures, markets, or strategy.\n\n"
            "I remember our last 15 messages, so you can ask follow-ups naturally.\n\n"
            "What are you looking at this weekend?"
        )
        return thread
    except discord.Forbidden:
        log.warning("Missing permissions to create whale chat thread for %s", user)
        return None
    except discord.HTTPException as exc:
        log.warning("Failed to create whale chat thread for %s: %s", user, exc)
        return None


async def _get_or_create_thread(bot: commands.Bot, user: discord.User | discord.Member) -> Optional[discord.Thread]:
    """Get or create a whale chat thread for a user."""
    if not WHALE_AI_CHANNEL:
        return None
    channel = bot.get_channel(WHALE_AI_CHANNEL)
    if not isinstance(channel, discord.TextChannel):
        return None

    thread = await _find_user_thread(channel, user)
    if thread:
        return thread
    return await _create_user_thread(channel, user)


# ---------------------------------------------------------------------------
# Tool definitions for GPT function calling
# ---------------------------------------------------------------------------

_rag = None


def _ensure_rag():
    global _rag
    if _rag is None:
        import rag_cli_v2 as rag
        _rag = rag
    return _rag


LEAGUE_TO_API_ID = {
    "EPL": 39, "LaLiga": 140, "SerieA": 135,
    "Bundesliga": 78, "Ligue1": 61,
    "UCL": 2, "UEL": 3, "UECL": 848,
}


def _fetch_upcoming_from_api(league: str, team_filter: str = "") -> List[Dict]:
    """Fallback: fetch upcoming fixtures directly from API-Football."""
    import urllib.request

    api_key = os.getenv("API-FOOTBALL-KEY") or os.getenv("API_FOOTBALL_KEY", "")
    if not api_key:
        return []

    api_id = LEAGUE_TO_API_ID.get(league)
    if not api_id:
        return []

    try:
        url = f"https://v3.football.api-sports.io/fixtures?league={api_id}&season=2025&next=10"
        req = urllib.request.Request(url, headers={"x-apisports-key": api_key})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        fixtures = data.get("response", [])

        results = []
        for fx in fixtures:
            home = fx.get("teams", {}).get("home", {}).get("name", "")
            away = fx.get("teams", {}).get("away", {}).get("name", "")
            dt = fx.get("fixture", {}).get("date", "")

            if team_filter and team_filter not in home.lower() and team_filter not in away.lower():
                continue

            results.append({
                "home_team": home,
                "away_team": away,
                "commence_time": dt,
            })
        return results
    except Exception as exc:
        log.debug("API-Football upcoming fetch failed: %s", exc)
        return []


def _resolve_team(name: str, league: str = "") -> str:
    """Resolve informal team name to canonical form."""
    rag = _ensure_rag()
    try:
        canonical = rag.canonical_team_name(name)
        if canonical:
            return canonical
    except Exception:
        pass
    return name


def _validate_teams(home: str, away: str, league: str) -> Optional[str]:
    """Check both teams have profiles. Returns error message or None."""
    rag = _ensure_rag()
    h_meta = rag.get_team_profile_meta(home, league)
    a_meta = rag.get_team_profile_meta(away, league)
    missing = []
    if not h_meta:
        missing.append(home)
    if not a_meta:
        missing.append(away)
    if missing:
        return f"No profile found for {', '.join(missing)} in {league}. Check the team name(s) and league."
    return None


# Tool schemas
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_fixtures",
            "description": "Get upcoming fixtures for a league. Returns ALL upcoming fixtures. Optionally filter by date or team name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "league": {"type": "string", "enum": ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1", "UCL", "UEL", "UECL"]},
                    "date": {"type": "string", "description": "Optional: filter to a specific date (YYYY-MM-DD)."},
                    "team": {"type": "string", "description": "Optional: filter to fixtures involving this team."},
                },
                "required": ["league"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_goals",
            "description": "Project total goals for a fixture. Returns blended, season, recent projections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "home_team": {"type": "string"},
                    "away_team": {"type": "string"},
                    "league": {"type": "string"},
                },
                "required": ["home_team", "away_team", "league"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_corners",
            "description": "Project total corners for a fixture.",
            "parameters": {
                "type": "object",
                "properties": {
                    "home_team": {"type": "string"},
                    "away_team": {"type": "string"},
                    "league": {"type": "string"},
                },
                "required": ["home_team", "away_team", "league"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_cards",
            "description": "Project total cards for a fixture. Includes referee info if available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "home_team": {"type": "string"},
                    "away_team": {"type": "string"},
                    "league": {"type": "string"},
                },
                "required": ["home_team", "away_team", "league"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_sot",
            "description": "Project total shots on target for a fixture.",
            "parameters": {
                "type": "object",
                "properties": {
                    "home_team": {"type": "string"},
                    "away_team": {"type": "string"},
                    "league": {"type": "string"},
                },
                "required": ["home_team", "away_team", "league"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_btts",
            "description": "Project Both Teams To Score probability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "home_team": {"type": "string"},
                    "away_team": {"type": "string"},
                    "league": {"type": "string"},
                },
                "required": ["home_team", "away_team", "league"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_moneyline",
            "description": "Project match outcome probabilities (home win / draw / away win).",
            "parameters": {
                "type": "object",
                "properties": {
                    "home_team": {"type": "string"},
                    "away_team": {"type": "string"},
                    "league": {"type": "string"},
                },
                "required": ["home_team", "away_team", "league"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_team_profile",
            "description": "Get a team's season profile: goals, corners, cards, possession, xG, form index, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {"type": "string"},
                    "league": {"type": "string"},
                },
                "required": ["team", "league"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_goal_difference",
            "description": "Project expected goal difference (positive = home favored). For handicap/spread analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "home_team": {"type": "string"},
                    "away_team": {"type": "string"},
                    "league": {"type": "string"},
                },
                "required": ["home_team", "away_team", "league"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_odds",
            "description": "Get current bookmaker odds for a fixture. Only available when fixtures are scheduled.",
            "parameters": {
                "type": "object",
                "properties": {
                    "home_team": {"type": "string"},
                    "away_team": {"type": "string"},
                    "league": {"type": "string"},
                },
                "required": ["home_team", "away_team", "league"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_line",
            "description": "Recommend the best over/under line for a stat (goals, corners, cards, sot) in a fixture. Returns the recommended side, line, odds, model probability, value edge, and confidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "home_team": {"type": "string"},
                    "away_team": {"type": "string"},
                    "league": {"type": "string"},
                    "stat": {"type": "string", "enum": ["goals", "corners", "cards", "sot"]},
                },
                "required": ["home_team", "away_team", "league", "stat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_form",
            "description": "Get a team's recent form stats from last 6 matches: goals, corners, cards, SoT averages and trends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {"type": "string"},
                    "league": {"type": "string"},
                },
                "required": ["team", "league"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_referee_info",
            "description": "Get referee assignment and profile for a fixture. Returns referee name, cards/match average, strictness ratio, fouls/match, cards/foul rate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "home_team": {"type": "string"},
                    "away_team": {"type": "string"},
                    "league": {"type": "string"},
                },
                "required": ["home_team", "away_team", "league"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_team_corners",
            "description": "Project corners for a SPECIFIC TEAM (not match total). Use this when the user asks about one team's corners, e.g. 'how many corners will Arsenal have'. Returns that team's individual corner projection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "The team to project corners for."},
                    "opponent": {"type": "string", "description": "Their opponent."},
                    "league": {"type": "string"},
                    "is_home": {"type": "boolean", "description": "Whether the team is playing at home."},
                },
                "required": ["team", "opponent", "league", "is_home"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_team_cards",
            "description": "Project cards for a SPECIFIC TEAM (not match total). Use this when the user asks about one team's cards. Returns that team's individual card projection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "The team to project cards for."},
                    "opponent": {"type": "string", "description": "Their opponent."},
                    "league": {"type": "string"},
                    "is_home": {"type": "boolean", "description": "Whether the team is playing at home."},
                },
                "required": ["team", "opponent", "league", "is_home"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_h2h_stats",
            "description": "Get head-to-head historical stats between two teams: recent meetings, average goals/corners/cards in past encounters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "home_team": {"type": "string"},
                    "away_team": {"type": "string"},
                    "league": {"type": "string"},
                },
                "required": ["home_team", "away_team", "league"],
            },
        },
    },
]

# System prompt
SYSTEM_PROMPT = """You are Spick, the personal AI betting analyst for Spick's Picks.

You're chatting with a Whale-tier subscriber in their private thread. Be conversational, confident, and concise. This is a dialogue, not a report.

You have tools to query real statistical models and live odds. Always call a tool before answering — never invent numbers.

Tool tips:
- get_fixtures: returns ALL upcoming fixtures. Use "team" param to find a specific team's next game.
- IMPORTANT: project_corners/project_cards return MATCH TOTALS (both teams combined). If the user asks about ONE team's corners or cards, use project_team_corners or project_team_cards instead.
- recommend_line: gives the best betting line with odds, probability, and value edge. Use this when the user asks "what line should I take" or "is over X a good bet".
- get_recent_form: shows last 6 matches vs season average. Use to identify hot/cold streaks.
- get_referee_info: referee strictness, cards/match, foul rate. Always check this for card-related questions.
- get_h2h_stats: historical meetings between two teams. Check this for derby/rivalry matches.
- To build a parlay, get fixtures first, then call projection/recommend tools for each, then combine.
- You can call MULTIPLE tools in one turn.
- For team names, use the canonical name (Arsenal, not "the gunners").
- If unsure about a league: EPL for English, LaLiga for Spanish, SerieA for Italian, Bundesliga for German, Ligue1 for French.

Style:
- Use clean text formatting, not Discord embeds. You're having a conversation.
- Use ◆ headers and ━ separators for structured data.
- End responses with a natural follow-up question or suggestion.
- Keep it under 300 words unless they ask for detail.
- Always display times in CET.
- If no data is available (international break, etc.), say so clearly — don't retry.
- When building parlays interactively, show a running tally of legs and combined odds.
"""


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Execute a tool call and return JSON string result."""
    rag = _ensure_rag()

    try:
        if name == "get_fixtures":
            league = args["league"]
            target_date = args.get("date")
            team_filter = args.get("team", "").lower()

            try:
                events, _ = rag.fetch_events(league, set())
            except Exception:
                events = []

            if not events:
                events = _fetch_upcoming_from_api(league, team_filter)

            filtered = []
            for ev in events:
                ct = ev.get("commence_time", "") or ev.get("date", "")
                home = ev.get("home_team", "")
                away = ev.get("away_team", "")
                if target_date and ct and ct[:10] != target_date[:10]:
                    continue
                if team_filter and team_filter not in home.lower() and team_filter not in away.lower():
                    continue
                filtered.append({"home_team": home, "away_team": away, "commence_time": ct})

            filtered.sort(key=lambda x: x.get("commence_time", ""))

            if not filtered:
                return json.dumps({
                    "fixtures": [], "count": 0, "league": league,
                    "message": f"No upcoming fixtures found for {league}" +
                               (f" involving {team_filter}" if team_filter else "") +
                               ". Could be an international break or off-season.",
                })

            return json.dumps({"fixtures": filtered[:20], "count": len(filtered), "league": league})

        elif name == "project_goals":
            home = _resolve_team(args["home_team"]); away = _resolve_team(args["away_team"])
            err = _validate_teams(home, away, args["league"])
            if err: return json.dumps({"error": err})
            bl, s, r = rag.projected_total_goals(home, away, args["league"])
            if bl is None: return json.dumps({"error": f"No data for {home} vs {away} in {args['league']}."})
            return json.dumps({"blended": round(bl, 2), "season": round(s, 2) if s else None, "recent": round(r, 2) if r else None})

        elif name == "project_corners":
            home = _resolve_team(args["home_team"]); away = _resolve_team(args["away_team"])
            err = _validate_teams(home, away, args["league"])
            if err: return json.dumps({"error": err})
            bl, s, r = rag.projected_total_corners(home, away, args["league"])
            if bl is None: return json.dumps({"error": f"No data for {home} vs {away} in {args['league']}."})
            return json.dumps({"blended": round(bl, 2), "season": round(s, 2) if s else None, "recent": round(r, 2) if r else None})

        elif name == "project_cards":
            home = _resolve_team(args["home_team"]); away = _resolve_team(args["away_team"])
            err = _validate_teams(home, away, args["league"])
            if err: return json.dumps({"error": err})
            bl, s, r, ref = rag.projected_total_cards(home, away, args["league"])
            if bl is None: return json.dumps({"error": f"No data for {home} vs {away} in {args['league']}."})
            ref_info = {"multiplier": round(ref.multiplier, 3), "referee": ref.referee_name, "sample_size": ref.sample_size} if ref else None
            return json.dumps({"blended": round(bl, 2), "season": round(s, 2) if s else None, "recent": round(r, 2) if r else None, "referee": ref_info})

        elif name == "project_sot":
            home = _resolve_team(args["home_team"]); away = _resolve_team(args["away_team"])
            err = _validate_teams(home, away, args["league"])
            if err: return json.dumps({"error": err})
            bl, s, r = rag.projected_total_sot(home, away, args["league"])
            if bl is None: return json.dumps({"error": f"No data for {home} vs {away} in {args['league']}."})
            return json.dumps({"blended": round(bl, 2), "season": round(s, 2) if s else None, "recent": round(r, 2) if r else None})

        elif name == "project_btts":
            home = _resolve_team(args["home_team"]); away = _resolve_team(args["away_team"])
            err = _validate_teams(home, away, args["league"])
            if err: return json.dumps({"error": err})
            prob, h, a, _, _ = rag.projected_btts_prob(home, away, args["league"])
            if prob is None: return json.dumps({"error": f"No data for {home} vs {away} in {args['league']}."})
            return json.dumps({"btts_yes_prob": round(prob, 3), "home_goals_proj": round(h, 2) if h else None, "away_goals_proj": round(a, 2) if a else None})

        elif name == "project_moneyline":
            home = _resolve_team(args["home_team"]); away = _resolve_team(args["away_team"])
            err = _validate_teams(home, away, args["league"])
            if err: return json.dumps({"error": err})
            ph, pd, pa, _, _ = rag.projected_moneyline_probs(home, away, args["league"])
            if ph is None: return json.dumps({"error": f"No data for {home} vs {away} in {args['league']}."})
            return json.dumps({"home_win": round(ph, 3), "draw": round(pd, 3), "away_win": round(pa, 3)})

        elif name == "get_team_profile":
            team = _resolve_team(args["team"])
            meta = rag.get_team_profile_meta(team, args["league"])
            if not meta:
                return json.dumps({"error": f"No profile found for '{args['team']}' in {args['league']}."})
            keys = ["goals_for_pm", "goals_against_pm", "corners_pm", "corners_against_pm",
                    "cards_per_90_team", "sot_for_pm", "sot_against_pm", "possession",
                    "dominance_index", "control_index", "form_index_team",
                    "xg_home_pm", "xg_away_pm", "corners_home_pm", "corners_away_pm",
                    "matches_played"]
            profile = {k: round(float(meta[k]), 3) if meta.get(k) is not None else None for k in keys}
            return json.dumps({"team": team, "league": args["league"], "profile": profile})

        elif name == "project_goal_difference":
            home = _resolve_team(args["home_team"]); away = _resolve_team(args["away_team"])
            err = _validate_teams(home, away, args["league"])
            if err: return json.dumps({"error": err})
            bl, s, r = rag.projected_goal_difference(home, away, args["league"])
            if bl is None: return json.dumps({"error": f"No data for {home} vs {away} in {args['league']}."})
            return json.dumps({"blended_diff": round(bl, 2), "season_diff": round(s, 2) if s else None, "recent_diff": round(r, 2) if r else None, "note": "positive = home favored"})

        elif name == "get_odds":
            try:
                events, _ = rag.fetch_events(args["league"], set())
            except Exception:
                events = []
            if not events:
                return json.dumps({"error": f"No odds available for {args['league']} right now. Odds are only available when fixtures are scheduled."})
            home = _resolve_team(args["home_team"]).lower()
            away = _resolve_team(args["away_team"]).lower()
            for ev in events:
                if home in ev.get("home_team", "").lower() and away in ev.get("away_team", "").lower():
                    odds_summary = {}
                    for bm in ev.get("bookmakers", []):
                        for mkt in bm.get("markets", []):
                            key = mkt.get("key", "")
                            if key not in odds_summary:
                                outcomes = [{"name": o.get("name"), "price": o.get("price"), **({"point": o["point"]} if o.get("point") is not None else {})} for o in mkt.get("outcomes", [])]
                                if outcomes:
                                    odds_summary[key] = outcomes[:6]
                    return json.dumps({"fixture": f"{ev.get('home_team')} vs {ev.get('away_team')}", "odds": odds_summary})
            return json.dumps({"error": f"No odds found for {args['home_team']} vs {args['away_team']}."})

        elif name == "project_team_corners":
            team = _resolve_team(args["team"]); opp = _resolve_team(args["opponent"])
            league = args["league"]; is_home = args.get("is_home", True)
            err = _validate_teams(team, opp, league)
            if err: return json.dumps({"error": err})
            try:
                bl, sea, rec = rag.projected_team_corners(team, opp, league, is_home)
                if bl is None: return json.dumps({"error": f"No corner data for {team}."})
                return json.dumps({
                    "team": team, "opponent": opp, "is_home": is_home,
                    "projected_corners": round(bl, 2),
                    "season": round(sea, 2) if sea else None,
                    "recent": round(rec, 2) if rec else None,
                    "note": f"This is {team}'s individual corner projection, not the match total.",
                })
            except Exception as exc:
                return json.dumps({"error": f"Team corner projection failed: {exc}"})

        elif name == "project_team_cards":
            team = _resolve_team(args["team"]); opp = _resolve_team(args["opponent"])
            league = args["league"]; is_home = args.get("is_home", True)
            err = _validate_teams(team, opp, league)
            if err: return json.dumps({"error": err})
            try:
                bl, sea, rec = rag.projected_team_cards(team, opp, league, is_home)
                if bl is None: return json.dumps({"error": f"No card data for {team}."})
                return json.dumps({
                    "team": team, "opponent": opp, "is_home": is_home,
                    "projected_cards": round(bl, 2),
                    "season": round(sea, 2) if sea else None,
                    "recent": round(rec, 2) if rec else None,
                    "note": f"This is {team}'s individual card projection, not the match total.",
                })
            except Exception as exc:
                return json.dumps({"error": f"Team card projection failed: {exc}"})

        elif name == "recommend_line":
            home = _resolve_team(args["home_team"]); away = _resolve_team(args["away_team"])
            league = args["league"]; stat = args["stat"]
            err = _validate_teams(home, away, league)
            if err: return json.dumps({"error": err})

            # Get projection
            proj_fn = {"goals": rag.projected_total_goals, "corners": rag.projected_total_corners,
                       "sot": rag.projected_total_sot}
            if stat == "cards":
                result = rag.projected_total_cards(home, away, league)
                proj, _, _ = result[0], result[1], result[2]
            elif stat in proj_fn:
                proj, _, _ = proj_fn[stat](home, away, league)
            else:
                return json.dumps({"error": f"Unknown stat: {stat}"})

            if proj is None:
                return json.dumps({"error": f"No projection data for {stat} in {home} vs {away}."})

            # Get odds and find best line
            try:
                events, _ = rag.fetch_events(league, set())
            except Exception:
                events = []

            best = None
            for ev in events:
                if home.lower() in ev.get("home_team", "").lower() and away.lower() in ev.get("away_team", "").lower():
                    options = rag.extract_total_line_options(ev, stat if stat != "goals" else "totals")
                    if options:
                        variance = None
                        try:
                            variance = rag.get_blended_variance(home, league, f"{stat}_var")
                        except Exception:
                            pass
                        best = rag.choose_best_total_line(options, proj, variance)
                    break

            if best:
                from prob_models import over_prob, implied_prob
                model_p = best.get("_model_prob")
                val_edge = best.get("_value_edge")
                conf = rag.confidence_from_edge(
                    abs(val_edge) if val_edge else 0,
                    stat_group=stat,
                    model_prob=model_p,
                    value_edge_pct=val_edge
                )
                return json.dumps({
                    "stat": stat, "projection": round(proj, 2),
                    "recommended_side": best.get("side"), "line": best.get("point"),
                    "odds": best.get("odds"), "bookmaker": best.get("bookmaker"),
                    "model_prob": round(model_p, 3) if model_p else None,
                    "implied_prob": round(best.get("_implied_prob", 0), 3) if best.get("_implied_prob") else None,
                    "value_edge": round(val_edge, 3) if val_edge else None,
                    "confidence": conf,
                })
            else:
                # No odds — model-only recommendation
                from prob_models import over_prob
                nearest_line = round(proj * 2) / 2  # nearest 0.5
                p_over = over_prob(proj, nearest_line)
                side = "over" if p_over >= 0.5 else "under"
                model_p = p_over if side == "over" else 1 - p_over
                return json.dumps({
                    "stat": stat, "projection": round(proj, 2),
                    "recommended_side": side, "line": nearest_line,
                    "odds": None, "bookmaker": None,
                    "model_prob": round(model_p, 3),
                    "value_edge": None,
                    "confidence": "model-only (no odds available)",
                    "note": "No bookmaker odds available. Line based on model projection only.",
                })

        elif name == "get_recent_form":
            team = _resolve_team(args["team"])
            league = args["league"]
            try:
                recent = rag._recent_stats(team, league, last_n=6)
            except Exception:
                recent = {}

            if not recent:
                return json.dumps({"error": f"No recent form data for {team} in {league}."})

            # Extract key recent stats
            form = {}
            for key in ["goals_for_avg", "goals_against_avg", "corners_for_avg", "corners_against_avg",
                        "cards_avg", "sot_for_avg", "sot_against_avg", "fouls_avg",
                        "xg_for_avg", "xg_against_avg"]:
                val = recent.get(key)
                if val is not None:
                    form[key] = round(float(val), 2)

            # Get season profile for comparison
            meta = rag.get_team_profile_meta(team, league) or {}
            season = {}
            for key, meta_key in [("goals_for_pm", "goals_for_pm"), ("corners_pm", "corners_pm"),
                                   ("cards_per_90", "cards_per_90_team"), ("sot_for_pm", "sot_for_pm")]:
                val = meta.get(meta_key)
                if val is not None:
                    season[key] = round(float(val), 2)

            return json.dumps({
                "team": team, "league": league,
                "recent_6_matches": form,
                "season_averages": season,
                "note": "Compare recent to season to spot trends. Rising recent > season = hot form.",
            })

        elif name == "get_referee_info":
            home = _resolve_team(args["home_team"]); away = _resolve_team(args["away_team"])
            league = args["league"]
            try:
                from referee_data import get_referee_modifier
                ref = get_referee_modifier(home, away, league, weight=0.25)
                if ref.source == "profile" and ref.referee_name:
                    return json.dumps({
                        "referee": ref.referee_name,
                        "avg_cards_per_match": round(ref.avg_cards_per_match, 2),
                        "strictness_ratio": round(ref.strictness_ratio, 2),
                        "strictness_label": "strict" if ref.strictness_ratio >= 1.10 else ("lenient" if ref.strictness_ratio <= 0.90 else "average"),
                        "avg_fouls_per_match": round(ref.avg_fouls_per_match, 1),
                        "cards_per_foul": round(ref.cards_per_foul, 3),
                        "sample_size": ref.sample_size,
                        "modifier": round(ref.multiplier, 3),
                    })
                else:
                    return json.dumps({"referee": "Not yet assigned or data unavailable", "modifier": 1.0})
            except Exception as exc:
                return json.dumps({"error": f"Referee lookup failed: {exc}"})

        elif name == "get_h2h_stats":
            home = _resolve_team(args["home_team"]); away = _resolve_team(args["away_team"])
            league = args["league"]
            try:
                # Query Chroma for fixture docs involving both teams
                rows = rag.get_recent_team_fixture_rows(home, league, last_n=20)
                h2h_matches = []
                for row in rows:
                    meta = row.get("meta", {})
                    fixture = str(meta.get("fixture", ""))
                    if away.lower() in fixture.lower():
                        h2h_matches.append({
                            "fixture": fixture,
                            "date": meta.get("fixture_date", ""),
                            "goals": meta.get("goals"),
                            "corners": meta.get("corners"),
                            "cards_total": meta.get("cards_total"),
                        })

                if not h2h_matches:
                    return json.dumps({"message": f"No recent head-to-head data found for {home} vs {away} in Chroma.", "matches": []})

                # Compute averages
                def _avg(key):
                    vals = [float(m[key]) for m in h2h_matches if m.get(key) is not None]
                    return round(sum(vals) / len(vals), 2) if vals else None

                return json.dumps({
                    "home": home, "away": away, "league": league,
                    "meetings_found": len(h2h_matches),
                    "avg_goals_per_team": _avg("goals"),
                    "avg_corners_per_team": _avg("corners"),
                    "avg_cards_per_team": _avg("cards_total"),
                    "recent_matches": h2h_matches[:5],
                })
            except Exception as exc:
                return json.dumps({"error": f"H2H lookup failed: {exc}"})

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception as exc:
        log.error("Tool execution error for %s: %s", name, exc)
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def _build_messages(thread_history: List[Dict[str, str]], user_message: str) -> List[Dict]:
    """Build the messages array for GPT, including thread history as context."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in thread_history[-15:]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})
    return messages


def _chat_with_tools(messages: List[Dict], max_rounds: int = 8) -> str:
    """Run the GPT conversation with tool calling."""
    if not _HAS_OPENAI or not _client:
        return "AI chat is not configured. Contact an admin."

    for _ in range(max_rounds):
        try:
            resp = _client.chat.completions.create(
                model=_CHAT_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=1000,
            )
        except Exception as exc:
            log.error("GPT API error: %s", exc)
            return f"Sorry, I hit an error: {exc}"

        choice = resp.choices[0]

        if choice.finish_reason == "tool_calls" or choice.message.tool_calls:
            messages.append(choice.message.to_dict())
            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}
                log.info("Tool call: %s(%s)", fn_name, fn_args)
                result = _execute_tool(fn_name, fn_args)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
            continue

        return choice.message.content or "I couldn't generate a response."

    return "I needed too many steps to answer that. Try a simpler question."


# ---------------------------------------------------------------------------
# Welcome button view
# ---------------------------------------------------------------------------

class StartChatButton(discord.ui.View):
    """Button that creates the whale chat thread."""

    def __init__(self, bot: commands.Bot):
        super().__init__(timeout=None)  # persistent
        self.bot = bot

    @discord.ui.button(label="Start a conversation", style=discord.ButtonStyle.primary, emoji="🐋", custom_id="whale_chat_start")
    async def start_chat(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        thread = await _get_or_create_thread(self.bot, interaction.user)
        if thread:
            await interaction.followup.send(
                f"Your AI chat is ready → {thread.mention}",
                ephemeral=True,
            )
        else:
            await interaction.followup.send(
                "Couldn't create your chat thread. Make sure the whale-ai channel is set up.",
                ephemeral=True,
            )


# ---------------------------------------------------------------------------
# Discord cog
# ---------------------------------------------------------------------------

class WhaleChat(commands.Cog):
    """Whale-exclusive conversational AI assistant."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def cog_load(self):
        # Register persistent view for the welcome button
        self.bot.add_view(StartChatButton(self.bot))
        # Start matchday nudge task
        self.matchday_nudge.start()

    async def cog_unload(self):
        if self.matchday_nudge.is_running():
            self.matchday_nudge.cancel()

    # ------------------------------------------------------------------
    # on_member_update: detect Whale role assignment → DM + create thread
    # ------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_member_update(self, before: discord.Member, after: discord.Member):
        """Detect when a user gets the Whale role and send welcome DM."""
        before_roles = {r.name for r in before.roles}
        after_roles = {r.name for r in after.roles}
        if before_roles != after_roles:
            log.info("Role change for %s: %s -> %s", after, before_roles, after_roles)

        # Check if WHALE role specifically was added (not just tier-equivalent)
        before_role_names = {r.name.lower() for r in before.roles}
        after_role_names = {r.name.lower() for r in after.roles}
        whale_role_name = ROLE_TO_TIER and [k for k, v in ROLE_TO_TIER.items() if v == "elite"]
        whale_names = set(whale_role_name) if whale_role_name else {"whale"}

        had_whale = bool(before_role_names & whale_names)
        has_whale = bool(after_role_names & whale_names)

        if had_whale or not has_whale:
            return

        log.info("New Whale detected: %s — creating AI chat thread", after)

        # Create thread proactively (DM is handled by tier_welcome.py)
        await _get_or_create_thread(self.bot, after)

    # ------------------------------------------------------------------
    # on_message: listen in whale chat threads → GPT responds
    # ------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Listen for messages in Whale chat threads and respond."""
        if message.author.bot:
            return

        # Must be a private thread
        if not isinstance(message.channel, discord.Thread):
            return
        if not message.channel.is_private():
            return

        # Must be in a whale chat thread (ends with -chat)
        if not message.channel.name.endswith(_THREAD_SUFFIX):
            return

        # Must be in the whale-ai channel (if configured) or a bot- thread (legacy)
        parent = message.channel.parent
        is_whale_channel = parent and parent.id == WHALE_AI_CHANNEL if WHALE_AI_CHANNEL else False
        is_legacy_thread = message.channel.name.startswith("bot-")
        if not is_whale_channel and not is_legacy_thread:
            return

        # Check Whale tier
        guild = message.guild
        if not guild:
            return
        member = guild.get_member(message.author.id)
        if not member:
            return
        if not _is_whale(member):
            return

        if not _HAS_OPENAI:
            await message.channel.send("AI chat is not configured. Contact an admin.")
            return

        # Show typing indicator while processing
        async with message.channel.typing():
            # Fetch thread history for context
            thread_history = []
            async for hist_msg in message.channel.history(limit=16, oldest_first=False):
                if hist_msg.id == message.id:
                    continue
                role = "assistant" if hist_msg.author.bot else "user"
                content = hist_msg.content or ""
                if content:
                    thread_history.append({"role": role, "content": content})
            thread_history.reverse()

            # Run chat with tools
            loop = asyncio.get_running_loop()
            messages = _build_messages(thread_history, message.content)
            try:
                response = await loop.run_in_executor(
                    _executor, lambda: _chat_with_tools(messages)
                )
            except Exception as exc:
                log.error("Whale chat error: %s", exc)
                response = "Sorry, something went wrong. Try again."

        # Send response (split if too long)
        if len(response) <= 2000:
            await message.channel.send(response)
        else:
            for i in range(0, len(response), 1990):
                await message.channel.send(response[i:i + 1990])

    # ------------------------------------------------------------------
    # /chat: thin shortcut from any channel
    # ------------------------------------------------------------------

    @app_commands.command(
        name="chat",
        description="Open your private AI chat (Whale exclusive)",
    )
    @app_commands.describe(question="Optional: ask a question directly")
    async def chat_cmd(self, interaction: discord.Interaction, question: Optional[str] = None):
        """Whale shortcut: open chat thread or ask a question from any channel."""
        member = interaction.user
        if not isinstance(member, discord.Member) or not _is_whale(member):
            em = discord.Embed(
                title="Whale Exclusive 🐋",
                description=(
                    "The private AI analyst is available exclusively for **Whale** subscribers.\n\n"
                    "Use `/subscribe` to upgrade."
                ),
                color=COLOR_PURPLE,
            )
            await interaction.response.send_message(embed=em, ephemeral=True)
            return

        await interaction.response.defer(thinking=True, ephemeral=True)

        thread = await _get_or_create_thread(self.bot, member)
        if not thread:
            await interaction.followup.send(
                "Couldn't find or create your chat thread. Ask an admin to set up the whale-ai channel.",
                ephemeral=True,
            )
            return

        if question:
            # Post question and answer in the thread
            await thread.send(f"**{member.display_name}:** {question}")

            loop = asyncio.get_running_loop()
            # Get thread history for context
            thread_history = []
            async for hist_msg in thread.history(limit=16, oldest_first=False):
                role = "assistant" if hist_msg.author.bot else "user"
                content = hist_msg.content or ""
                if content:
                    thread_history.append({"role": role, "content": content})
            thread_history.reverse()

            messages = _build_messages(thread_history, question)
            try:
                response = await loop.run_in_executor(
                    _executor, lambda: _chat_with_tools(messages)
                )
            except Exception as exc:
                response = f"Sorry, something went wrong: {exc}"

            if len(response) <= 2000:
                await thread.send(response)
            else:
                for i in range(0, len(response), 1990):
                    await thread.send(response[i:i + 1990])

        await interaction.followup.send(
            f"🐋 Your AI chat → {thread.mention}",
            ephemeral=True,
        )

    # ------------------------------------------------------------------
    # Matchday nudge: proactive thread resurfacing
    # ------------------------------------------------------------------

    @tasks.loop(time=dt_time(hour=8, minute=30))
    async def matchday_nudge(self):
        """On matchdays, nudge active Whale threads with a scan offer."""
        if not WHALE_AI_CHANNEL:
            return

        channel = self.bot.get_channel(WHALE_AI_CHANNEL)
        if not isinstance(channel, discord.TextChannel):
            return

        # Check if there are fixtures today
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from cogs.auto_push import _get_todays_leagues
            _, active_leagues = _get_todays_leagues()
            if not active_leagues:
                return
        except Exception:
            return

        league_str = ", ".join(active_leagues[:5])
        fixture_count = len(active_leagues) * 3  # rough estimate

        # Post nudge in all active whale threads
        for thread in channel.threads:
            if not thread.is_private() or not thread.name.endswith(_THREAD_SUFFIX):
                continue
            try:
                await thread.send(
                    f"Matchday today — {league_str} in action. "
                    f"Want me to scan for value across the fixtures?"
                )
            except Exception:
                pass

    @matchday_nudge.before_loop
    async def _wait_nudge(self):
        await self.bot.wait_until_ready()


async def setup(bot: commands.Bot):
    await bot.add_cog(WhaleChat(bot))
