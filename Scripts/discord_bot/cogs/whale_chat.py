"""Whale-exclusive conversational AI — a private stats desk in the user's pocket.

Whale users get a dedicated private thread under #whale-ai where they just type
naturally, like texting a statistics team. No slash commands needed inside the thread.

Architecture:
    - on_member_update: detects Whale role assignment → DMs welcome + creates thread
    - on_message: listens in whale chat threads → GPT with tool calling → responds
    - /chat: thin shortcut from any channel → redirects to thread
    - matchday_nudge: proactive "X matches today, want me to scan?" in active threads

Conversation memory comes from Discord thread history (last 15 messages).
No changes to the core prediction pipeline — this is a read-only consumer of existing functions.
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

from config import COLOR_PURPLE, ROLE_TO_TIER, TIER_HIERARCHY
from .whale_chat_policy import BASE_SYSTEM_PROMPT, runtime_instruction_for_message

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
            "upcoming fixtures, projected totals, team form, referee trends, "
            "or matchup context.\n\n"
            "I won't build bets in here. I help you make your own read with numbers.\n\n"
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
        current_season = date.today().year if date.today().month >= 7 else date.today().year - 1
        url = f"https://v3.football.api-sports.io/fixtures?league={api_id}&season={current_season}&next=10"
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


def _recent_form_payload(rag_module, team: str, league: str) -> Dict[str, Optional[float]]:
    """Return a compact recent-form payload for a team."""
    try:
        recent = rag_module._recent_stats(team, league, last_n=6)
    except Exception:
        recent = {}

    payload: Dict[str, Optional[float]] = {}
    for key in [
        "goals_for_avg",
        "goals_against_avg",
        "corners_for_avg",
        "corners_against_avg",
        "cards_avg",
        "sot_for_avg",
        "sot_against_avg",
        "xg_for_avg",
        "xg_against_avg",
    ]:
        value = recent.get(key)
        if value is not None:
            payload[key] = round(float(value), 2)
    return payload


def _referee_payload(home: str, away: str, league: str) -> Optional[Dict[str, Any]]:
    """Return compact referee context for a fixture, if available."""
    try:
        from referee_data import get_referee_modifier

        ref = get_referee_modifier(home, away, league, weight=0.25)
    except Exception:
        return None

    if ref.source != "profile" or not ref.referee_name:
        return None

    return {
        "referee": ref.referee_name,
        "avg_cards_per_match": round(ref.avg_cards_per_match, 2),
        "strictness_ratio": round(ref.strictness_ratio, 2),
        "strictness_label": "strict" if ref.strictness_ratio >= 1.10 else ("lenient" if ref.strictness_ratio <= 0.90 else "average"),
        "avg_fouls_per_match": round(ref.avg_fouls_per_match, 1),
        "cards_per_foul": round(ref.cards_per_foul, 3),
        "sample_size": ref.sample_size,
        "modifier": round(ref.multiplier, 3),
    }


# Tool schemas
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_fixtures",
            "description": "Get upcoming fixtures for a league. Returns all upcoming fixtures and can filter by date or team name.",
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
            "name": "get_fixture_snapshot",
            "description": "Get a full pre-match statistical snapshot for a fixture: projected goals, corners, cards, shots on target, BTTS, win probabilities, recent form, and referee context.",
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
            "name": "project_goals",
            "description": "Project total goals for a fixture. Returns blended, season, and recent projections.",
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
            "description": "Project match-total corners for a fixture.",
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
            "description": "Project match-total cards for a fixture, including referee context when available.",
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
            "description": "Project match outcome probabilities (home win, draw, away win).",
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
            "description": "Get a team's season profile: goals, corners, cards, possession, xG, form index, and other core team metrics.",
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
            "description": "Project expected goal difference for a fixture. Positive means the home side is favored.",
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
            "name": "get_recent_form",
            "description": "Get a team's recent form stats from its last 6 matches: goals, corners, cards, shots on target, and xG trends.",
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
            "description": "Get referee assignment and profile for a fixture: cards per match, strictness ratio, fouls per match, and cards-per-foul rate.",
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
            "description": "Project corners for one team, not the match total.",
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
            "description": "Project cards for one team, not the match total.",
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
            "description": "Get head-to-head historical stats between two teams: recent meetings and average goals, corners, and cards in those meetings.",
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

SYSTEM_PROMPT = BASE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Execute a tool call and return JSON string result."""
    rag = _ensure_rag()

    try:
        if name == "get_fixtures":
            league = args["league"]
            target_date_str = args.get("date")
            team_filter = args.get("team", "").lower()

            from datetime import timedelta

            # Determine which dates to query
            if target_date_str:
                try:
                    dates_to_check = [date.fromisoformat(target_date_str)]
                except ValueError:
                    dates_to_check = [date.today()]
            else:
                # No date specified — fetch next 7 days
                today = date.today()
                dates_to_check = [today + timedelta(days=d) for d in range(7)]

            # Fetch fixtures across all target dates
            events = []
            for d in dates_to_check:
                try:
                    day_events, _ = rag.fetch_events(league, set(), target_date=d)
                    events.extend(day_events)
                except Exception:
                    pass

            # Fallback to API-Football direct if nothing found
            if not events:
                events = _fetch_upcoming_from_api(league, team_filter)

            # Deduplicate by fixture id
            seen_ids = set()
            unique_events = []
            for ev in events:
                eid = ev.get("id") or ev.get("_fixture_id") or f"{ev.get('home_team')}_{ev.get('away_team')}"
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    unique_events.append(ev)

            filtered = []
            for ev in unique_events:
                ct = ev.get("commence_time", "") or ev.get("date", "")
                home_name = ev.get("home_team", "")
                away_name = ev.get("away_team", "")
                if team_filter and team_filter not in home_name.lower() and team_filter not in away_name.lower():
                    continue
                filtered.append({"home_team": home_name, "away_team": away_name, "commence_time": ct})

            filtered.sort(key=lambda x: x.get("commence_time", ""))

            if not filtered:
                return json.dumps({
                    "fixtures": [], "count": 0, "league": league,
                    "message": f"No upcoming fixtures found for {league}" +
                               (f" involving {team_filter}" if team_filter else "") +
                               (f" on {target_date_str}" if target_date_str else " in the next 7 days") +
                               ". Could be an international break or off-season.",
                })

            return json.dumps({"fixtures": filtered[:20], "count": len(filtered), "league": league})

        elif name == "get_fixture_snapshot":
            home = _resolve_team(args["home_team"])
            away = _resolve_team(args["away_team"])
            league = args["league"]
            err = _validate_teams(home, away, league)
            if err:
                return json.dumps({"error": err})

            goals_bl, goals_s, goals_r = rag.projected_total_goals(home, away, league)
            corners_bl, corners_s, corners_r = rag.projected_total_corners(home, away, league)
            cards_bl, cards_s, cards_r, _ = rag.projected_total_cards(home, away, league)
            sot_bl, sot_s, sot_r = rag.projected_total_sot(home, away, league)
            btts_prob, home_goals_proj, away_goals_proj, _, _ = rag.projected_btts_prob(home, away, league)
            home_win, draw_prob, away_win, _, _ = rag.projected_moneyline_probs(home, away, league)
            goal_diff_bl, goal_diff_s, goal_diff_r = rag.projected_goal_difference(home, away, league)

            return json.dumps({
                "fixture": f"{home} vs {away}",
                "league": league,
                "projections": {
                    "goals": {"blended": round(goals_bl, 2) if goals_bl is not None else None, "season": round(goals_s, 2) if goals_s is not None else None, "recent": round(goals_r, 2) if goals_r is not None else None},
                    "corners": {"blended": round(corners_bl, 2) if corners_bl is not None else None, "season": round(corners_s, 2) if corners_s is not None else None, "recent": round(corners_r, 2) if corners_r is not None else None},
                    "cards": {"blended": round(cards_bl, 2) if cards_bl is not None else None, "season": round(cards_s, 2) if cards_s is not None else None, "recent": round(cards_r, 2) if cards_r is not None else None},
                    "shots_on_target": {"blended": round(sot_bl, 2) if sot_bl is not None else None, "season": round(sot_s, 2) if sot_s is not None else None, "recent": round(sot_r, 2) if sot_r is not None else None},
                },
                "probabilities": {
                    "btts_yes": round(btts_prob, 3) if btts_prob is not None else None,
                    "home_win": round(home_win, 3) if home_win is not None else None,
                    "draw": round(draw_prob, 3) if draw_prob is not None else None,
                    "away_win": round(away_win, 3) if away_win is not None else None,
                },
                "team_goal_projections": {
                    "home": round(home_goals_proj, 2) if home_goals_proj is not None else None,
                    "away": round(away_goals_proj, 2) if away_goals_proj is not None else None,
                    "goal_difference": {
                        "blended": round(goal_diff_bl, 2) if goal_diff_bl is not None else None,
                        "season": round(goal_diff_s, 2) if goal_diff_s is not None else None,
                        "recent": round(goal_diff_r, 2) if goal_diff_r is not None else None,
                        "note": "positive = home favored",
                    },
                },
                "recent_form": {
                    home: _recent_form_payload(rag, home, league),
                    away: _recent_form_payload(rag, away, league),
                },
                "referee": _referee_payload(home, away, league),
            })

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

        elif name == "get_recent_form":
            team = _resolve_team(args["team"])
            league = args["league"]
            form = _recent_form_payload(rag, team, league)
            if not form:
                return json.dumps({"error": f"No recent form data for {team} in {league}."})

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
                "note": "Compare recent to season to spot trends. Rising recent above season usually signals hot form.",
            })

        elif name == "get_referee_info":
            home = _resolve_team(args["home_team"])
            away = _resolve_team(args["away_team"])
            league = args["league"]
            ref_payload = _referee_payload(home, away, league)
            if ref_payload:
                return json.dumps(ref_payload)
            return json.dumps({"referee": "Not yet assigned or data unavailable", "modifier": 1.0})

        elif name == "get_h2h_stats":
            home = _resolve_team(args["home_team"]); away = _resolve_team(args["away_team"])
            league = args["league"]
            try:
                # Fetch fixture rows for BOTH teams, then cross-reference
                home_rows = rag.get_recent_team_fixture_rows(home, league, limit=30)
                away_rows = rag.get_recent_team_fixture_rows(away, league, limit=30)

                # Find fixtures where both teams appear
                h2h_matches = []
                seen = set()
                for row in home_rows + away_rows:
                    meta = row.get("meta", {})
                    fixture = str(meta.get("fixture", ""))
                    fixture_date = str(meta.get("fixture_date", ""))
                    key = f"{fixture_date}|{fixture}"
                    if key in seen:
                        continue
                    if home.lower() in fixture.lower() and away.lower() in fixture.lower():
                        seen.add(key)
                        h2h_matches.append({
                            "fixture": fixture,
                            "date": fixture_date,
                            "team": meta.get("team", ""),
                            "goals": meta.get("goals"),
                            "corners": meta.get("corners"),
                            "cards_total": meta.get("cards_total"),
                        })

                if not h2h_matches:
                    return json.dumps({"message": f"No recent head-to-head data found for {home} vs {away}.", "matches": []})

                def _avg(key):
                    vals = [float(m[key]) for m in h2h_matches if m.get(key) is not None]
                    return round(sum(vals) / len(vals), 2) if vals else None

                # Count unique meetings (2 rows per match — one per team)
                unique_dates = set(m["date"] for m in h2h_matches if m["date"])
                n_meetings = len(unique_dates) if unique_dates else len(h2h_matches) // 2

                return json.dumps({
                    "home": home, "away": away, "league": league,
                    "meetings_found": n_meetings,
                    "avg_goals_per_team": _avg("goals"),
                    "avg_corners_per_team": _avg("corners"),
                    "avg_cards_per_team": _avg("cards_total"),
                    "recent_matches": h2h_matches[:10],
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
    today = date.today()
    date_context = (
        f"\n\nToday is {today.strftime('%A, %B %d, %Y')}. "
        f"Use this to interpret relative dates like 'today', 'tomorrow', 'this weekend', 'Saturday', etc. "
        f"When the user asks about upcoming fixtures, pass the appropriate date to get_fixtures."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT + date_context}]
    runtime_instruction = runtime_instruction_for_message(user_message)
    if runtime_instruction:
        messages.append({"role": "system", "content": runtime_instruction})
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
                max_tokens=2000,
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
                f"Your stats chat is ready → {thread.mention}",
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

        log.info("New Whale detected: %s — creating AI chat thread + sending DM", after)

        # Create thread proactively
        thread = await _get_or_create_thread(self.bot, after)

        # Send Whale welcome DM with thread link
        try:
            # Resolve channel mentions
            def _ch(name):
                for guild in self.bot.guilds:
                    for ch in guild.channels:
                        if ch.name == name:
                            return ch.mention
                return f"**#{name}**"

            newsletter = _ch("newsletter")
            whale_den = _ch("whale-den")

            em = discord.Embed(
                title="Welcome to Whale tier \U0001f40b",
                description=(
                    "You now have the full arsenal. Everything in Tipster, plus tools "
                    "that nobody else in this server has access to."
                ),
                color=COLOR_PURPLE,
            )
            em.add_field(
                name="Your private stats desk",
                value=(
                    "You have a dedicated chat thread where you can ask anything in plain English \u2014 "
                    "\"break down Arsenal vs Chelsea for goals, corners and cards\" or "
                    "\"who has the stronger recent form in Bayern vs Dortmund?\" Just type. No commands needed.\n\n"
                    "Whale chat is stats-first: it helps you build your own read rather than pushing picks.\n\n"
                    + (f"\u2192 **Go to {thread.mention} and start typing.**" if thread
                       else "\u2192 **Go to #whale-ai and open your thread to start.**")
                ),
                inline=False,
            )
            em.add_field(
                name="What else you get",
                value=(
                    f"{newsletter} \u2014 Every matchday morning, you get a briefing sourced from "
                    "local sports media across Europe. Injury updates, lineup hints, tactical changes \u2014 "
                    "in English, from Kicker, Marca, L'Equipe and more.\n\n"
                    "**Unlimited commands** \u2014 No daily cap. Use `/analyze` for single-fixture "
                    "deep dives that break down where the edge comes from.\n\n"
                    f"{whale_den} \u2014 The smallest room in the server. Just you and the other Whales."
                ),
                inline=False,
            )
            em.add_field(
                name="One tip",
                value=(
                    "Start by typing something in your AI chat thread. Ask about this weekend's fixtures. "
                    "See how it feels to have a stats team in your pocket."
                ),
                inline=False,
            )
            em.set_footer(text="Spick's Picks | Welcome to the inner circle.")

            view = StartChatButton(self.bot)
            await after.send(embed=em, view=view)
        except discord.Forbidden:
            log.warning("Can't DM %s (DMs disabled). Thread created anyway.", after)
        except Exception as exc:
            log.warning("Failed to send Whale welcome DM to %s: %s", after, exc)

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
        description="Open your private stats chat (Whale exclusive)",
    )
    @app_commands.describe(question="Optional: ask a question directly")
    async def chat_cmd(self, interaction: discord.Interaction, question: Optional[str] = None):
        """Whale shortcut: open chat thread or ask a question from any channel."""
        member = interaction.user
        if not isinstance(member, discord.Member) or not _is_whale(member):
            em = discord.Embed(
                title="Whale Exclusive 🐋",
                description=(
                    "The private AI stats desk is available exclusively for **Whale** subscribers.\n\n"
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
            f"🐋 Your stats chat → {thread.mention}",
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
        # Post nudge in all active whale threads
        for thread in channel.threads:
            if not thread.is_private() or not thread.name.endswith(_THREAD_SUFFIX):
                continue
            try:
                await thread.send(
                    f"Matchday today — {league_str} in action. "
                    f"Want me to scan the board for the strongest statistical signals?"
                )
            except Exception:
                pass

    @matchday_nudge.before_loop
    async def _wait_nudge(self):
        await self.bot.wait_until_ready()


async def setup(bot: commands.Bot):
    await bot.add_cog(WhaleChat(bot))
