"""Scheduled auto-push tasks: Parlays, Market Channels, Value Alerts, Track Record.

Market channels post automatically before kickoff — no user interaction required.
Each channel gets league-by-league analysis for all fixtures on the day.
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from datetime import datetime, date, timedelta, time as dt_time, timezone

import discord
from discord.ext import commands, tasks

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "rag_ingest"))
import rag_cli_v2 as rag  # noqa: E402  (kept for text-based workflows: parlays, player props)

# Core modules for direct structured access (no stdout capture needed)
from core.events import fetch_events, filter_events_by_exact_date  # noqa: E402
from core.weights import DOMESTIC_LEAGUES as _CORE_DOMESTIC, EUROPEAN_COMPETITIONS  # noqa: E402
from core.parlay_service import ParlayBuildError, build_parlay, build_parlay_request  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from embeds import (  # noqa: E402
    rag_output_to_embeds, structured_parlay_embeds, value_alert_embed,
    consolidated_fixture_embeds, consolidated_score_embeds,
    parse_fixture_picks, parse_correct_score_picks, parse_interval_picks,
)
from config import (  # noqa: E402
    CHANNEL_DAILY_PICKS,
    CHANNEL_PARLAYS,
    CHANNEL_PLAYER_PROPS,
    CHANNEL_TEAM_LINES,
    CHANNEL_BEST_PICKS,
    CHANNEL_NEWSLETTER,
    # Legacy aliases (all map to the channels above via config.py)
    CHANNEL_PARLAY_OF_DAY,
    CHANNEL_MATCHDAY_DIGEST,
    CHANNEL_VALUE_ALERTS,
    CHANNEL_TRACK_RECORD,
    CHANNEL_MATCH_PREDICTIONS,
    CHANNEL_CORNERS_CARDS,
    CHANNEL_CORRECT_SCORE,
    DIGEST_POST_HOUR,
    ALERT_SCAN_INTERVAL_MINUTES,
    DOMESTIC_LEAGUES,
    MIN_ALERT_MODEL_PROB,
    MIN_ALERT_VALUE_EDGE,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_PURPLE,
)

# Prediction tracker (optional — graceful if unavailable)
try:
    from prediction_tracker import resolve_outcomes, get_track_record, get_daily_breakdown, capture_closing_odds
    _HAS_TRACKER = True
except ImportError:
    _HAS_TRACKER = False

# Newsletter (optional — graceful if unavailable)
try:
    from newsletter import generate_newsletter
    _HAS_NEWSLETTER = True
except ImportError:
    _HAS_NEWSLETTER = False

log = logging.getLogger("auto_push")

# Thread pool for blocking RAG calls — keeps the async event loop free
_rag_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_rag_sync(prompt: str, league: str) -> str:
    """Synchronous RAG call — runs in thread pool, never call from event loop."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rag.answer_once(prompt, default_league=league)
    except Exception as exc:
        log.error("RAG query failed: %s", exc, exc_info=True)
        return ""
    result = buf.getvalue().strip()
    if not result:
        log.warning("RAG returned empty output for prompt: %.100s", prompt)
    else:
        log.info("RAG output length: %d chars for prompt: %.80s", len(result), prompt)
    return result


async def _run_rag(prompt: str, league: str) -> str:
    """Async wrapper — runs RAG in thread pool so event loop stays free."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_rag_executor, _run_rag_sync, prompt, league)


async def _build_structured_parlay(request):
    """Run structured parlay generation in the same executor used for blocking tasks."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_rag_executor, build_parlay, request)


def _auto_parlay_presets(target_date: date, leagues: list) -> list:
    """Return the three scheduled parlay presets for #parlays."""
    leagues = list(leagues or [])
    return [
        {
            "request": build_parlay_request(
                league_value="cross-league",
                leagues_override=leagues,
                display_league="Cross-League",
                target_date=target_date,
                legs=2,
                odds=1.9,
                market="mixed",
                source_command="auto_push",
                target_mode="max",
                min_model_prob=0.65,
                allowed_confidences=["high"],
                risk_profile="lock",
            ),
            "title": "\U0001f512  The Lock (~1.8x)",
            "color": COLOR_GREEN,
        },
        {
            "request": build_parlay_request(
                league_value="cross-league",
                leagues_override=leagues,
                display_league="Cross-League",
                target_date=target_date,
                legs=3,
                odds=None,
                target_min=2.0,
                target_max=3.0,
                market="mixed",
                source_command="auto_push",
                target_mode="none",
                allowed_confidences=["high", "medium"],
                risk_profile="safe",
            ),
            "title": "\U0001f6e1\ufe0f  Safe Parlay (~2.5x)",
            "color": COLOR_BLUE,
        },
        {
            "request": build_parlay_request(
                league_value="cross-league",
                leagues_override=leagues,
                display_league="Cross-League",
                target_date=target_date,
                legs=4,
                odds=None,
                target_min=4.0,
                target_max=6.0,
                market="mixed",
                source_command="auto_push",
                target_mode="none",
                allowed_groups=["totals", "corners", "cards"],
                soft_prefer_groups=["totals", "corners", "cards"],
                required_group_counts={"totals": 1, "corners": 1, "cards": 1},
                risk_profile="standard",
            ),
            "title": "\U0001f3b2  Standard Parlay (~5x)",
            "color": COLOR_PURPLE,
        },
    ]


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"


def _today_phrase() -> str:
    d = datetime.now().date()
    return f"{_ordinal(d.day)} {d.strftime('%B')}"


def _is_valid(text: str) -> bool:
    """Check if RAG output has real content (not an error/empty message)."""
    if not text:
        return False
    low = text.lower()
    return "no fixtures" not in low and "could not" not in low and "no odds" not in low


def _get_todays_leagues() -> tuple:
    """Find which leagues have fixtures today, and the first kickoff time.

    Returns (first_kickoff_dt_or_None, active_leagues_list).
    Scans all domestic + European leagues, only returns those with matches.
    """
    all_leagues = DOMESTIC_LEAGUES + ["UCL", "UEL", "UECL"]
    first_kickoff = None
    active_leagues = []

    for league in all_leagues:
        try:
            events, _ = fetch_events(league, target_date=date.today())
            day_events = filter_events_by_exact_date(events, date.today())
            if day_events:
                active_leagues.append(league)
                for ev in day_events:
                    ko = ev.get("commence_time")
                    if ko:
                        try:
                            ko_dt = datetime.fromisoformat(ko.replace("Z", "+00:00"))
                            if first_kickoff is None or ko_dt < first_kickoff:
                                first_kickoff = ko_dt
                        except Exception:
                            pass
        except Exception:
            pass

    if active_leagues:
        log.info("Today's active leagues: %s", active_leagues)
    else:
        log.info("No fixtures found in any league today.")

    return first_kickoff, active_leagues


def _is_posting_window(first_kickoff) -> bool:
    """Return True if we're 2-3.5 hours before first kickoff."""
    if first_kickoff is None:
        return True  # no kickoff time known, post anyway
    now = datetime.now(timezone.utc)
    if first_kickoff.tzinfo is None:
        first_kickoff = first_kickoff.replace(tzinfo=timezone.utc)
    hours_until = (first_kickoff - now).total_seconds() / 3600
    log.info("First kickoff in %.1f hours (kickoff=%s, now=%s)",
             hours_until, first_kickoff, now)
    return -0.5 <= hours_until <= 3.5


async def _post_market_for_leagues(
    channel: discord.TextChannel,
    prompt_template: str,
    title_template: str,
    leagues: list,
    date_str: str,
):
    """Post a market analysis to a channel for each league that has fixtures."""
    for league in leagues:
        prompt = prompt_template.format(league=league, date=date_str)
        text = await _run_rag(prompt, league)
        if not _is_valid(text):
            continue
        embeds = rag_output_to_embeds(
            title_template.format(league=league),
            text,
            league=league,
        )
        for em in embeds:
            await channel.send(embed=em)


async def _post_consolidated_predictions(
    channel: discord.TextChannel,
    leagues: list,
    date_str: str,
):
    """Post fixture-consolidated predictions (moneyline + BTTS + spreads + SoT).

    Instead of 4 separate market posts per league, builds ONE embed per fixture
    with all markets as compact lines.
    """
    _MARKET_QUERIES = {
        "moneyline": (
            "For every {league} game on {date}, show me moneyline "
            "probabilities and the best value pick."
        ),
        "btts": (
            "For every {league} game on {date}, should I take "
            "BTTS Yes or No?"
        ),
        "spreads": (
            "For every {league} game on {date}, recommend a handicap line."
        ),
        "sot": (
            "For every {league} game on {date}, recommend a shots on target "
            "over/under line."
        ),
    }

    for league in leagues:
        # Run all 4 market queries, parse picks per fixture
        fixture_data: dict = {}  # {fixture: {market: compact_line}}

        for market_key, prompt_tpl in _MARKET_QUERIES.items():
            prompt = prompt_tpl.format(league=league, date=date_str)
            text = await _run_rag(prompt, league)
            if not _is_valid(text):
                continue
            picks = parse_fixture_picks(text)
            for fixture, line in picks.items():
                if fixture not in fixture_data:
                    fixture_data[fixture] = {}
                fixture_data[fixture][market_key] = line

        if not fixture_data:
            continue

        embeds = consolidated_fixture_embeds(league, fixture_data)
        for em in embeds:
            await channel.send(embed=em)


async def _post_consolidated_scores(
    channel: discord.TextChannel,
    leagues: list,
    date_str: str,
):
    """Post fixture-consolidated correct score + goal intervals.

    One embed per fixture with top scorelines and interval breakdown.
    """
    for league in leagues:
        fixture_data: dict = {}

        # Correct score
        text = await _run_rag(
            f"For every {league} game on {date_str}, predict the most "
            "likely correct score.",
            league,
        )
        if _is_valid(text):
            for fixture, cs_str in parse_correct_score_picks(text).items():
                if fixture not in fixture_data:
                    fixture_data[fixture] = {}
                fixture_data[fixture]["correct_score"] = cs_str

        # Goal intervals
        text = await _run_rag(
            f"For every {league} game on {date_str}, show me goal "
            "interval probabilities.",
            league,
        )
        if _is_valid(text):
            for fixture, iv_str in parse_interval_picks(text).items():
                if fixture not in fixture_data:
                    fixture_data[fixture] = {}
                fixture_data[fixture]["intervals"] = iv_str

        if not fixture_data:
            continue

        embeds = consolidated_score_embeds(league, fixture_data)
        for em in embeds:
            await channel.send(embed=em)


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Lineup fetching for player props
# ---------------------------------------------------------------------------

_LEAGUE_TO_API_ID = {
    "EPL": 39, "LaLiga": 140, "SerieA": 135,
    "Bundesliga": 78, "Ligue1": 61,
    "UCL": 2, "UEL": 3, "UECL": 848,
}


def _fetch_lineups(home: str, away: str, league: str) -> List[str]:
    """Fetch confirmed starting XI from API-Football.

    Returns list of lowercase player names if lineups are published,
    empty list if not yet available.
    """
    import os
    try:
        import requests
    except ImportError:
        return []

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("API-FOOTBALL-KEY") or os.environ.get("API_FOOTBALL_KEY")
    if not api_key:
        return []

    league_id = _LEAGUE_TO_API_ID.get(league)
    if not league_id:
        return []

    try:
        # Find the fixture ID
        resp = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers={"x-apisports-key": api_key},
            params={"league": league_id, "season": 2025, "date": date.today().isoformat()},
            timeout=15,
        )
        fixtures = resp.json().get("response", [])

        fixture_id = None
        home_lower = home.lower()
        away_lower = away.lower()
        for f in fixtures:
            h = f.get("teams", {}).get("home", {}).get("name", "").lower()
            a = f.get("teams", {}).get("away", {}).get("name", "").lower()
            if (home_lower in h or h in home_lower) and (away_lower in a or a in away_lower):
                fixture_id = f["fixture"]["id"]
                break

        if not fixture_id:
            return []

        # Fetch lineups
        resp2 = requests.get(
            "https://v3.football.api-sports.io/fixtures/lineups",
            headers={"x-apisports-key": api_key},
            params={"fixture": fixture_id},
            timeout=15,
        )
        lineups = resp2.json().get("response", [])
        if not lineups:
            return []

        starters = []
        for team_lineup in lineups:
            for p in team_lineup.get("startXI", []):
                name = p.get("player", {}).get("name", "")
                if name:
                    starters.append(name.lower())

        log.info("Fetched %d starters for fixture %d (%s vs %s)",
                 len(starters), fixture_id, home, away)
        return starters

    except Exception as exc:
        log.warning("Lineup fetch failed for %s vs %s: %s", home, away, exc)
        return []


def _filter_to_starters(rag_output: str, starters: List[str]) -> str:
    """Filter player prop RAG output to only include confirmed starters.

    Removes player entries whose name doesn't appear in the starters list.
    Preserves the === fixture header and Method footer.
    """
    if not starters:
        return rag_output

    lines = rag_output.split("\n")
    filtered: List[str] = []
    include_current = True
    starters_lower = set(starters)

    for line in lines:
        # Fixture headers, empty lines, method footer — always include
        if line.startswith("===") or line.startswith("Player Prop") or line.startswith("Method:") or not line.strip():
            filtered.append(line)
            include_current = True
            continue

        # Player header line (starts with emoji + name)
        # e.g., "⚽ Antoine Semenyo (bournemouth) [MID] (HOME)"
        import re
        player_m = re.match(r"[\u2600-\U0001ffff]+\s+(.+?)\s+\(", line)
        if player_m:
            player_name = player_m.group(1).strip().lower()
            # Check if player name matches any starter (substring match)
            include_current = any(
                player_name in s or s in player_name
                for s in starters_lower
            )
            if include_current:
                filtered.append(line)
            continue

        # Detail lines — include if current player is included
        if include_current:
            filtered.append(line)

    return "\n".join(filtered)


def _parse_player_picks_from_text(
    text: str, stat_key: str, emoji: str, bet_label: str,
) -> list:
    """Parse player prop RAG output into structured pick dicts.

    Returns list of {name, team, prob, prob_num, odds, edge, conf, emoji, bet_label, recent}.
    """
    import re
    picks = []

    # Split on player header lines
    sections = re.split(r"\n(?=[\u2600-\U0001ffff]+ [A-Z])", text)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Parse player header: "emoji Name (team) [POS] (VENUE)"
        header_m = re.match(
            r"[\u2600-\U0001ffff]+\s+(.+?)\s+\(([^)]+)\)\s+\[([A-Z]+)\]\s+\((\w+)\)",
            section,
        )
        if not header_m:
            continue

        name = header_m.group(1).strip()
        team = header_m.group(2).strip()

        # Parse probability: "— 34% chance" or "— 81% chance @ 1.17 (Bet365)"
        prob_m = re.search(r"(\d+)%\s*chance", section)
        prob_str = f"{prob_m.group(1)}%" if prob_m else "?"
        prob_num = int(prob_m.group(1)) if prob_m else 0

        # Parse odds: "@ 1.17 (Bet365)"
        odds_m = re.search(r"@\s*([\d.]+)\s*\(([^)]+)\)", section)
        odds_str = f"{odds_m.group(1)} ({odds_m.group(2)})" if odds_m else None

        # Parse edge: "Edge: +5%"
        edge_m = re.search(r"Edge:\s*([+\-]\d+%)", section)
        edge_str = edge_m.group(1) if edge_m else None

        # Parse confidence
        conf = "low"
        if re.search(r"Confidence:\s*HIGH", section, re.I):
            conf = "high"
        elif re.search(r"Confidence:\s*MEDIUM", section, re.I):
            conf = "medium"

        # Parse recent form
        recent_m = re.search(r"Recent:\s*(\[.+?\])", section)
        recent_str = recent_m.group(1) if recent_m else None

        # Determine specific bet label (might have "Over 1.5" prefix)
        specific_label = bet_label
        over_m = re.search(r">>>\s*Over\s+([\d.]+)\s+", section)
        if over_m:
            specific_label = f"Over {over_m.group(1)} {bet_label}"

        picks.append({
            "name": name,
            "team": team,
            "prob": prob_str,
            "prob_num": prob_num,
            "odds": odds_str,
            "edge": edge_str,
            "conf": conf,
            "emoji": emoji,
            "bet_label": specific_label,
            "recent": recent_str,
            "_stat_key": stat_key,
        })

    return picks


def _build_player_props_embed(
    all_picks: list,
    fixture_name: str,
    league: str,
    date_str: str,
    footer_extra: str = "",
) -> Optional[discord.Embed]:
    """Build ONE compact player props embed for a fixture.

    Layout:
    - Top 1 goalscorer pick
    - Top 1 cards pick
    - Top 1 SoT pick
    - Separator
    - Top 3 fouls picks (most likely to foul — relevant for every game)
    """
    try:
        from embeds import _get_league_logo, _conf_bar, _conf_label
    except ImportError:
        _get_league_logo = lambda x: None
        _conf_bar = lambda x: ""
        _conf_label = lambda x: ""

    # Split picks by category
    by_cat: dict = {}
    for p in all_picks:
        cat = p.get("_stat_key", "other")
        by_cat.setdefault(cat, []).append(p)

    # Sort each category by probability
    for cat in by_cat:
        by_cat[cat].sort(key=lambda x: x.get("prob_num", 0), reverse=True)

    # Select: top 1 from goalscorer, cards, SoT + top 3 from fouls
    selected = []
    for cat in ("goals", "cards", "sot"):
        picks = by_cat.get(cat, [])
        if picks:
            selected.append(picks[0])

    foul_picks = by_cat.get("fouls", [])[:3]

    if not selected and not foul_picks:
        return None

    # Determine embed color
    all_selected = selected + foul_picks
    best_conf = "low"
    for p in all_selected:
        if p.get("conf") == "high":
            best_conf = "high"
            break
        elif p.get("conf") == "medium":
            best_conf = "medium"

    embed_color = COLOR_GREEN if best_conf == "high" else (
        COLOR_YELLOW if best_conf == "medium" else COLOR_BLUE)

    league_logo = _get_league_logo(league)

    em = discord.Embed(
        title=f"\U0001f464  Player Props \u2014 {fixture_name}",
        description=f"**{date_str}** \u2022 {len(all_selected)} picks",
        color=embed_color,
    )
    if league_logo:
        em.set_thumbnail(url=league_logo)

    def _add_pick_field(p: dict):
        bar = _conf_bar(p.get("conf", "low"))
        conf_desc = _conf_label(p.get("conf", "low"))
        odds_str = f" @ {p['odds']}" if p.get("odds") else ""
        edge_str = f" \u2022 Edge: **{p['edge']}**" if p.get("edge") else ""

        em.add_field(
            name=f"{p['emoji']}  {p['name']} ({p['team']})",
            value=(
                f"**{p['bet_label']}**{odds_str}\n"
                f"{bar} {conf_desc} \u2022 **{p['prob']}** chance{edge_str}"
                + (f"\nRecent: {p['recent']}" if p.get("recent") else "")
            ),
            inline=False,
        )

    # Add goalscorer / cards / SoT picks
    for p in selected:
        _add_pick_field(p)

    # Add separator + fouls section
    if foul_picks:
        em.add_field(
            name="\U0001f6d1  Most Likely to Foul",
            value="\u200b",  # zero-width space as separator
            inline=False,
        )
        for p in foul_picks:
            bar = _conf_bar(p.get("conf", "low"))
            conf_desc = _conf_label(p.get("conf", "low"))
            odds_str = f" @ {p['odds']}" if p.get("odds") else ""

            em.add_field(
                name=f"{p['name']} ({p['team']})",
                value=(
                    f"**{p['bet_label']}**{odds_str}\n"
                    f"{bar} **{p['prob']}** chance"
                ),
                inline=True,
            )

    footer_parts = [league, "Spick's Picks"]
    if footer_extra:
        footer_parts.append(footer_extra)
    em.set_footer(text=" \u2502 ".join(footer_parts))

    return em


def _build_best_picks_embeds(breakdown: dict, date_str: str) -> list:
    """Build Discord embeds for the daily best-picks recap.

    Returns a list of Embed objects (typically 1-2):
    1. Main summary with per-market and per-league breakdown
    2. Notable hits and misses (if any)
    """
    hits = breakdown.get("hits", 0)
    misses = breakdown.get("misses", 0)
    pushes = breakdown.get("pushes", 0)
    total = breakdown.get("total", 0)
    decided = hits + misses
    hit_rate = breakdown.get("hit_rate", 0.0)

    # Color based on hit rate
    if hit_rate >= 0.60:
        color = COLOR_GREEN
    elif hit_rate >= 0.50:
        color = COLOR_YELLOW
    else:
        color = COLOR_RED

    # Header emoji based on performance
    if hit_rate >= 0.65:
        header = "Yesterday's Recap  —  Great Day"
    elif hit_rate >= 0.55:
        header = "Yesterday's Recap  —  Solid Day"
    elif hit_rate >= 0.50:
        header = "Yesterday's Recap  —  Mixed Day"
    else:
        header = "Yesterday's Recap  —  Tough Day"

    em = discord.Embed(
        title=header,
        description=(
            f"**{date_str}**\n"
            f"**{hits}/{decided}** predictions hit (**{hit_rate:.0%}**)"
            + (f"  |  {pushes} push" if pushes else "")
        ),
        color=color,
    )

    # ── By Market ──
    market_emojis = {
        "goals": "Goals", "corners": "Corners", "cards": "Cards",
        "sot": "SoT", "btts": "BTTS", "moneyline": "1X2",
        "spreads": "Spreads", "correct_score": "Correct Score",
    }
    by_market = breakdown.get("by_market", {})
    market_lines = []
    # Sort by hit rate desc, then by total desc
    sorted_markets = sorted(
        by_market.items(),
        key=lambda x: (x[1].get("hit_rate", 0), x[1].get("total", 0)),
        reverse=True,
    )
    for mkt, data in sorted_markets:
        mh = data.get("hits", 0)
        mm = data.get("misses", 0)
        md = mh + mm
        if md == 0:
            continue
        mr = data.get("hit_rate", 0)
        # Performance indicator
        if mr >= 0.70:
            ind = "+"
        elif mr >= 0.50:
            ind = "~"
        else:
            ind = "-"
        label = market_emojis.get(mkt, mkt.title())
        push_str = f" (+{data.get('pushes', 0)}P)" if data.get("pushes", 0) else ""
        market_lines.append(f"`{ind}` **{label}**: {mh}/{md} ({mr:.0%}){push_str}")

    if market_lines:
        em.add_field(name="By Market", value="\n".join(market_lines), inline=False)

    # ── By League ──
    league_emojis = {
        "EPL": "EPL", "LaLiga": "LaLiga", "SerieA": "Serie A",
        "Bundesliga": "Bundesliga", "Ligue1": "Ligue 1",
        "UCL": "UCL", "UEL": "UEL", "UECL": "UECL",
    }
    by_league = breakdown.get("by_league", {})
    league_lines = []
    sorted_leagues = sorted(
        by_league.items(),
        key=lambda x: (x[1].get("hit_rate", 0), x[1].get("total", 0)),
        reverse=True,
    )
    for lg, data in sorted_leagues:
        lh = data.get("hits", 0)
        lm = data.get("misses", 0)
        ld = lh + lm
        if ld == 0:
            continue
        lr = data.get("hit_rate", 0)
        if lr >= 0.70:
            ind = "+"
        elif lr >= 0.50:
            ind = "~"
        else:
            ind = "-"
        label = league_emojis.get(lg, lg)
        league_lines.append(f"`{ind}` **{label}**: {lh}/{ld} ({lr:.0%})")

    if league_lines:
        em.add_field(name="By League", value="\n".join(league_lines), inline=False)

    # ── By Confidence ──
    by_conf = breakdown.get("by_confidence", {})
    conf_lines = []
    conf_labels = {"high": "Strong Edge", "medium": "Leaning", "low": "Speculative"}
    for level in ["high", "medium", "low"]:
        c = by_conf.get(level, {})
        ct = c.get("total", 0)
        if ct == 0:
            continue
        ch = c.get("hits", 0)
        cr = c.get("hit_rate", 0)
        label = conf_labels.get(level, level.title())
        conf_lines.append(f"**{label}**: {ch}/{ct} ({cr:.0%})")
    if conf_lines:
        em.add_field(name="By Confidence", value="\n".join(conf_lines), inline=False)

    em.set_footer(text=f"{total} predictions graded | Spick's Picks")

    embeds = [em]

    # ── Notable Hits & Misses (second embed) ──
    notable_hits = breakdown.get("notable_hits", [])
    notable_misses = breakdown.get("notable_misses", [])

    if notable_hits or notable_misses:
        em2 = discord.Embed(
            title="Notable Picks",
            color=COLOR_BLUE,
        )

        if notable_hits:
            hit_lines = []
            for h in notable_hits[:4]:
                odds_str = f" @ {h['odds']:.2f}" if h.get("odds") else ""
                lg = h.get("league", "")
                hit_lines.append(f"**{lg}** {h['fixture']} — {h['pick']}{odds_str}")
            em2.add_field(name="Best Hits", value="\n".join(hit_lines), inline=False)

        if notable_misses:
            miss_lines = []
            for m in notable_misses[:3]:
                prob_str = f" (P={m['model_prob']:.0%})" if m.get("model_prob") else ""
                lg = m.get("league", "")
                miss_lines.append(f"**{lg}** {m['fixture']} — {m['pick']}{prob_str}")
            em2.add_field(name="Upsets / Misses", value="\n".join(miss_lines), inline=False)

        em2.set_footer(text="Spick's Picks")
        embeds.append(em2)

    return embeds


class AutoPush(commands.Cog):
    """Scheduled auto-posting tasks."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._posted_today: dict = {}  # track which tasks posted today

    def _already_posted(self, task_name: str) -> bool:
        today = date.today().isoformat()
        return self._posted_today.get(task_name) == today

    def _mark_posted(self, task_name: str):
        self._posted_today[task_name] = date.today().isoformat()

    async def cog_load(self):
        # Main scheduled tasks
        self.market_channels.start()
        if CHANNEL_PLAYER_PROPS:
            self.player_props_pre_match.start()
        if CHANNEL_PARLAYS:
            self.daily_parlay.start()
            self.value_scan.start()
        if CHANNEL_PARLAYS and _HAS_TRACKER:
            self.daily_track_record.start()
        if CHANNEL_BEST_PICKS and _HAS_TRACKER:
            self.best_picks_recap.start()
        if _HAS_TRACKER:
            self.clv_capture.start()
        if CHANNEL_NEWSLETTER and _HAS_NEWSLETTER:
            self.newsletter_digest.start()

    async def cog_unload(self):
        for task in [self.daily_parlay, self.value_scan, self.market_channels,
                     self.player_props_pre_match]:
            if task.is_running():
                task.cancel()
        if hasattr(self, "daily_track_record") and self.daily_track_record.is_running():
            self.daily_track_record.cancel()
        if hasattr(self, "best_picks_recap") and self.best_picks_recap.is_running():
            self.best_picks_recap.cancel()
        if hasattr(self, "clv_capture") and self.clv_capture.is_running():
            self.clv_capture.cancel()
        if hasattr(self, "newsletter_digest") and self.newsletter_digest.is_running():
            self.newsletter_digest.cancel()

    # ===================================================================
    # MARKET CHANNELS — auto-post before kickoff
    # ===================================================================

    @tasks.loop(minutes=30)
    async def market_channels(self):
        """Post all market analysis before first kickoff.

        Checks every 30 min. Posts once per day when within the pre-kickoff window.
        #daily-picks gets all market predictions, #player-props gets prop picks.
        """
        first_kickoff, active_leagues = _get_todays_leagues()
        if not active_leagues:
            return

        if not _is_posting_window(first_kickoff):
            return

        date_str = _today_phrase()
        leagues = active_leagues

        # --- #daily-picks: all market predictions ---
        if CHANNEL_DAILY_PICKS and not self._already_posted("daily_picks"):
            await self._post_daily_picks(leagues, date_str)

        # --- #team-lines: per-team corners & cards with odds ---
        if CHANNEL_TEAM_LINES and not self._already_posted("team_lines"):
            await self._post_team_lines(leagues, date_str)

        # Player props are posted separately — 30 min before each kickoff
        # via the player_props_pre_match task

    @market_channels.before_loop
    async def _wait_markets(self):
        await self.bot.wait_until_ready()

    # ===================================================================
    # PLAYER PROPS — 30 min before each kickoff, with confirmed lineups
    # ===================================================================

    @tasks.loop(minutes=10)
    async def player_props_pre_match(self):
        """Post player props 30 min before kickoff using confirmed lineups.

        Checks every 10 min. For each fixture kicking off in 25-35 min,
        fetches lineups from API-Football. If lineups are confirmed,
        posts player prop picks filtered to actual starters.
        """
        ch = self.bot.get_channel(CHANNEL_PLAYER_PROPS)
        if not ch:
            return

        now = datetime.now(timezone.utc)
        _, active_leagues = _get_todays_leagues()
        if not active_leagues:
            return

        date_str = _today_phrase()

        for league in active_leagues:
            events, _ = fetch_events(league, target_date=date.today())
            day_events = filter_events_by_exact_date(events, date.today())

            for ev in day_events:
                home = ev.get("home_team", "")
                away = ev.get("away_team", "")
                fixture_key = f"pp_{home}_{away}"

                # Skip if already posted for this fixture
                if self._already_posted(fixture_key):
                    continue

                # Check kickoff time — post 25-35 min before
                ko_str = ev.get("commence_time", "")
                if not ko_str:
                    continue
                try:
                    ko_dt = datetime.fromisoformat(ko_str.replace("Z", "+00:00"))
                    if ko_dt.tzinfo is None:
                        ko_dt = ko_dt.replace(tzinfo=timezone.utc)
                except Exception:
                    continue

                minutes_until = (ko_dt - now).total_seconds() / 60
                if not (25 <= minutes_until <= 35):
                    continue

                # Fetch lineups from API-Football
                starters = await asyncio.get_running_loop().run_in_executor(
                    None, _fetch_lineups, home, away, league
                )
                if not starters:
                    log.info("No lineups yet for %s vs %s, will retry", home, away)
                    continue

                log.info("Lineups confirmed for %s vs %s (%d starters), posting player props",
                         home, away, len(starters))

                # Run all 4 stat queries, collect ALL picks, rank, take top 3-4
                _pp_queries = [
                    ("who are the best anytime goalscorer picks?", "goals", "\u26bd", "Anytime Goalscorer"),
                    ("which players are most likely to be booked?", "cards", "\U0001f7e8", "To Be Booked"),
                    ("which players are best for shots on target props?", "sot", "\U0001f3af", "1+ Shot on Target"),
                    ("which players are most likely to foul?", "fouls", "\U0001f6d1", "To Commit a Foul"),
                ]

                all_picks = []
                for suffix, stat_key, emoji, bet_label in _pp_queries:
                    prompt = f"For {home} vs {away} in {league} on {date_str}, {suffix}"
                    text = await _run_rag(prompt, league)
                    if not _is_valid(text):
                        continue
                    # Filter to confirmed starters
                    filtered = _filter_to_starters(text, starters)
                    if not filtered:
                        continue
                    # Parse picks from the RAG output
                    picks = _parse_player_picks_from_text(filtered, stat_key, emoji, bet_label)
                    all_picks.extend(picks)

                if not all_picks:
                    log.info("No player prop picks for %s vs %s after filtering", home, away)
                    self._mark_posted(fixture_key)
                    continue

                em = _build_player_props_embed(
                    all_picks, f"{home} vs {away}", league, date_str,
                    footer_extra="30 min to kickoff \u2502 Lineup confirmed",
                )
                if em:
                    await ch.send(embed=em)

                self._mark_posted(fixture_key)

    @player_props_pre_match.before_loop
    async def _wait_pp(self):
        await self.bot.wait_until_ready()

    # ===================================================================
    # PARLAY OF THE DAY
    # ===================================================================

    @tasks.loop(minutes=30)
    async def daily_parlay(self):
        channel = self.bot.get_channel(CHANNEL_PARLAYS)
        if not channel:
            return

        if self._already_posted("parlay"):
            return

        first_kickoff, active_leagues = _get_todays_leagues()
        if not active_leagues:
            return
        if not _is_posting_window(first_kickoff):
            return

        target_date = date.today()
        date_str = _today_phrase()
        built = []
        used_leg_refs = []  # accumulate legs across parlays to prevent repetition
        for preset in _auto_parlay_presets(target_date, active_leagues):
            # Inject previously-used legs as exclusions for risk diversification
            if used_leg_refs:
                preset["request"].excluded_legs = list(
                    set(preset["request"].excluded_legs) | set(used_leg_refs)
                )
            try:
                result = await _build_structured_parlay(preset["request"])
            except ParlayBuildError as exc:
                log.warning("Auto parlay '%s' failed: %s | %s", preset["title"], exc.message, "; ".join(exc.notes))
                continue
            except Exception as exc:
                log.error("Auto parlay '%s' failed: %s", preset["title"], exc, exc_info=True)
                continue
            built.append((preset, result))
            # Collect used legs so subsequent parlays cannot repeat them
            for leg in result.selected_legs:
                used_leg_refs.append(leg.ref)

        if not built:
            log.warning("All 3 structured parlay presets failed.")
            return

        self._mark_posted("parlay")

        header = discord.Embed(
            title="\U0001f3af  Parlay of the Day",
            description=f"**{date_str}** — 3 parlays ranked by risk level",
            color=COLOR_GREEN,
        )
        header.set_footer(text="Spick | Auto-generated from model projections")
        await channel.send(embed=header)

        for preset, result in built:
            for em in structured_parlay_embeds(result, league="Cross-League"):
                em.title = preset["title"]
                em.color = preset["color"]
                await channel.send(embed=em)

    @daily_parlay.before_loop
    async def _wait_parlay(self):
        await self.bot.wait_until_ready()

    # ===================================================================
    # MATCHDAY DIGEST (goals totals — fixed hour)
    # ===================================================================

    @tasks.loop(time=dt_time(hour=DIGEST_POST_HOUR, minute=0))
    async def daily_digest(self):
        channel = self.bot.get_channel(CHANNEL_DAILY_PICKS)
        if not channel:
            return

        _, active_leagues = _get_todays_leagues()
        if not active_leagues:
            return

        date_str = _today_phrase()
        for league in active_leagues:
            prompt = (
                f"Give me the over/under goals line I should take for each "
                f"{league} game on {date_str}."
            )
            text = await _run_rag(prompt, league)
            if _is_valid(text):
                embeds = rag_output_to_embeds(
                    f"Goals Totals — {league}", text, league=league,
                    footer=f"Auto-posted | {date_str}",
                )
                for em in embeds:
                    await channel.send(embed=em)

    @daily_digest.before_loop
    async def _wait_digest(self):
        await self.bot.wait_until_ready()

    # ===================================================================
    # VALUE ALERTS
    # ===================================================================

    @tasks.loop(minutes=ALERT_SCAN_INTERVAL_MINUTES)
    async def value_scan(self):
        channel = self.bot.get_channel(CHANNEL_PARLAYS)
        if not channel:
            return

        _, active_leagues = _get_todays_leagues()
        if not active_leagues:
            return

        date_str = _today_phrase()
        for league in active_leagues:
            prompt = (
                f"For every {league} game on {date_str}, show moneyline "
                "probabilities and value picks where the model disagrees "
                "with the bookmakers."
            )
            text = await _run_rag(prompt, league)
            if not _is_valid(text):
                continue
            if "value edge: +" in text.lower() or "high confidence" in text.lower():
                em = value_alert_embed(text, league)
                await channel.send(embed=em)

    @value_scan.before_loop
    async def _wait_scan(self):
        await self.bot.wait_until_ready()

    # ===================================================================
    # DAILY TRACK RECORD (noon UTC)
    # ===================================================================

    @tasks.loop(time=dt_time(hour=12, minute=0))
    async def daily_track_record(self):
        channel = self.bot.get_channel(CHANNEL_PARLAYS)
        if not channel:
            return

        yesterday = (date.today() - timedelta(days=1)).isoformat()

        try:
            resolve_result = resolve_outcomes(prediction_date=yesterday)
            graded = resolve_result.get("graded", 0)
            hits = resolve_result.get("hit", 0)
            misses = resolve_result.get("miss", 0)
            log.info("Resolved %d predictions for %s: %d hits, %d misses",
                     graded, yesterday, hits, misses)
        except Exception as exc:
            log.error("Failed to resolve outcomes for %s: %s", yesterday, exc)
            graded, hits, misses = 0, 0, 0

        if graded == 0:
            return

        try:
            stats = get_track_record()
        except Exception as exc:
            log.error("Failed to get track record: %s", exc)
            return

        hit_rate_yesterday = hits / graded if graded > 0 else 0
        if hit_rate_yesterday >= 0.60:
            color = COLOR_GREEN
        elif hit_rate_yesterday >= 0.50:
            color = COLOR_YELLOW
        else:
            color = COLOR_RED

        em = discord.Embed(
            title=f"Daily Track Record — {yesterday}",
            color=color,
        )

        em.add_field(
            name="Yesterday's Results",
            value=(
                f"**{hits}/{graded}** ({hit_rate_yesterday:.0%})\n"
                f"Hits: {hits} | Misses: {misses}"
            ),
            inline=False,
        )

        by_conf = stats.get("by_confidence", {})
        conf_lines = []
        for level in ["high", "medium", "low"]:
            c = by_conf.get(level, {})
            if c.get("total", 0) > 0:
                label = {"high": "Strong Edge", "medium": "Leaning",
                         "low": "Speculative"}[level]
                conf_lines.append(
                    f"**{label}**: {c['hits']}/{c['total']} ({c['hit_rate']:.0%})"
                )
        if conf_lines:
            em.add_field(
                name="All-Time by Confidence",
                value="\n".join(conf_lines),
                inline=True,
            )

        by_market = stats.get("by_market", {})
        market_lines = []
        for market, m in sorted(by_market.items(),
                                key=lambda x: x[1].get("total", 0),
                                reverse=True)[:5]:
            if m.get("total", 0) > 0:
                market_lines.append(
                    f"**{market.title()}**: {m['hits']}/{m['total']} "
                    f"({m['hit_rate']:.0%})"
                )
        if market_lines:
            em.add_field(
                name="All-Time by Market",
                value="\n".join(market_lines),
                inline=True,
            )

        total = stats.get("total_graded", 0)
        overall_hits = stats.get("hits", 0)
        overall_rate = stats.get("hit_rate", 0)
        roi = stats.get("roi_flat_stake", 0)
        streak = stats.get("recent_streak", 0)
        streak_str = (f"W{streak}" if streak > 0
                      else (f"L{abs(streak)}" if streak < 0 else "—"))

        em.add_field(
            name="All-Time Record",
            value=(
                f"**{overall_hits}/{total}** ({overall_rate:.1%})\n"
                f"ROI: {roi:+.1%} | Streak: {streak_str}"
            ),
            inline=False,
        )
        em.set_footer(text="Spick's Picks Track Record | Updated daily at noon CET")
        await channel.send(embed=em)

    @daily_track_record.before_loop
    async def _wait_track_record(self):
        await self.bot.wait_until_ready()

    # ===================================================================
    # BEST PICKS RECAP — morning after, resolves + posts per-market/league
    # ===================================================================

    @tasks.loop(time=dt_time(hour=9, minute=0))
    async def best_picks_recap(self):
        """Resolve yesterday's predictions and post a detailed recap to #best-picks."""
        channel = self.bot.get_channel(CHANNEL_BEST_PICKS)
        if not channel:
            return

        if self._already_posted("best_picks_recap"):
            return

        yesterday = (date.today() - timedelta(days=1)).isoformat()

        # Step 1: resolve outcomes (uses API-Football fallback)
        loop = asyncio.get_running_loop()
        try:
            resolve_result = await loop.run_in_executor(
                _rag_executor, lambda: resolve_outcomes(prediction_date=yesterday)
            )
            graded = resolve_result.get("graded", 0)
            log.info("Best picks recap: resolved %d predictions for %s", graded, yesterday)
        except Exception as exc:
            log.error("Best picks recap: failed to resolve %s: %s", yesterday, exc)
            return

        if graded == 0:
            return

        # Step 2: get detailed breakdown
        try:
            breakdown = await loop.run_in_executor(
                _rag_executor, lambda: get_daily_breakdown(prediction_date=yesterday)
            )
        except Exception as exc:
            log.error("Best picks recap: failed to get breakdown: %s", exc)
            return

        if breakdown.get("total", 0) == 0:
            return

        # Step 3: build embeds and send
        embeds = _build_best_picks_embeds(breakdown, yesterday)
        for em in embeds:
            await channel.send(embed=em)

        self._mark_posted("best_picks_recap")

    @best_picks_recap.before_loop
    async def _wait_best_picks(self):
        await self.bot.wait_until_ready()

    # ===================================================================
    # CLV CAPTURE — capture closing odds before kickoff
    # ===================================================================

    @tasks.loop(minutes=10)
    async def clv_capture(self):
        """Capture closing odds for today's predictions every 10 minutes.

        Only runs on days with fixtures. The underlying capture_closing_odds()
        is idempotent — it only updates predictions where closing_odds IS NULL,
        so running frequently is safe and lightweight.
        """
        try:
            first_kickoff, active_leagues = _get_todays_leagues()
            if not active_leagues:
                return

            today_str = date.today().isoformat()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                _rag_executor, lambda: capture_closing_odds(prediction_date=today_str)
            )
            captured = result.get("captured", 0)
            if captured > 0:
                log.info("CLV capture: updated closing odds for %d predictions", captured)
        except Exception as exc:
            log.error("CLV capture failed: %s", exc, exc_info=True)

    @clv_capture.before_loop
    async def _wait_clv(self):
        await self.bot.wait_until_ready()

    # ===================================================================
    # NEWSLETTER DIGEST — matchday intel from local news sources (WHALE only)
    # ===================================================================

    @tasks.loop(time=dt_time(hour=9, minute=0))
    async def newsletter_digest(self):
        """Post matchday newsletter digest to #newsletter (WHALE only channel)."""
        channel = self.bot.get_channel(CHANNEL_NEWSLETTER)
        if not channel:
            return

        first_kickoff, active_leagues = _get_todays_leagues()
        if not active_leagues:
            return

        if self._already_posted("newsletter"):
            return

        log.info("Generating matchday newsletter for %d leagues...", len(active_leagues))
        loop = asyncio.get_running_loop()

        try:
            digests = await loop.run_in_executor(
                _rag_executor, lambda: generate_newsletter(active_leagues)
            )
        except Exception as exc:
            log.error("Newsletter generation failed: %s", exc, exc_info=True)
            return

        for league, digest_text in digests.items():
            if not digest_text or not digest_text.strip():
                continue

            em = discord.Embed(
                title=f"\U0001f4f0  Matchday Intel \u2014 {league}",
                description=digest_text[:4096],
                color=COLOR_BLUE,
            )
            em.set_footer(text=f"Spick's Picks \u2502 Sources: local sports media \u2502 WHALE exclusive")
            try:
                await channel.send(embed=em)
            except Exception as exc:
                log.error("Failed to send newsletter for %s: %s", league, exc)

        self._mark_posted("newsletter")
        log.info("Newsletter posted for %d leagues", len(digests))

    @newsletter_digest.before_loop
    async def _wait_newsletter(self):
        await self.bot.wait_until_ready()

    # ===================================================================
    # MANUAL TRIGGER (owner only)
    # ===================================================================

    @discord.app_commands.command(
        name="trigger",
        description="Manually trigger an auto-push task (owner only)",
    )
    @discord.app_commands.describe(task="Which task to trigger")
    @discord.app_commands.choices(task=[
        discord.app_commands.Choice(name="Daily Picks (all markets)", value="daily_picks"),
        discord.app_commands.Choice(name="Team Lines", value="team_lines"),
        discord.app_commands.Choice(name="Parlays", value="parlays"),
        discord.app_commands.Choice(name="Player Props", value="player_props"),
        discord.app_commands.Choice(name="Track Record", value="track_record"),
        discord.app_commands.Choice(name="Best Picks Recap", value="best_picks_recap"),
        discord.app_commands.Choice(name="CLV Capture", value="clv_capture"),
        discord.app_commands.Choice(name="Newsletter", value="newsletter"),
        discord.app_commands.Choice(name="Everything", value="everything"),
    ])
    async def trigger(
        self,
        interaction: discord.Interaction,
        task: discord.app_commands.Choice[str],
    ):
        if interaction.user.id != interaction.guild.owner_id:
            await interaction.response.send_message(
                "Only the server owner can use this command.", ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True, ephemeral=True)

        # CLV capture doesn't need today's fixtures
        if task.value == "clv_capture":
            try:
                today_str = date.today().isoformat()
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    _rag_executor, lambda: capture_closing_odds(prediction_date=today_str)
                )
                captured = result.get("captured", 0)
                skipped = result.get("skipped", 0)
                await interaction.followup.send(
                    f"CLV capture complete: {captured} updated, {skipped} skipped.",
                    ephemeral=True,
                )
            except Exception as exc:
                log.error("Manual CLV capture failed: %s", exc, exc_info=True)
                await interaction.followup.send(f"Error: {exc}", ephemeral=True)
            return

        # Newsletter doesn't need today's fixtures check (it fetches its own)
        if task.value == "newsletter":
            try:
                if not _HAS_NEWSLETTER:
                    await interaction.followup.send("Newsletter module not available.", ephemeral=True)
                    return
                _, active_leagues = _get_todays_leagues()
                if not active_leagues:
                    active_leagues = ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]
                loop = asyncio.get_running_loop()
                digests = await loop.run_in_executor(
                    _rag_executor, lambda: generate_newsletter(active_leagues)
                )
                nl_channel = self.bot.get_channel(CHANNEL_NEWSLETTER)
                posted_count = 0
                for league, text in digests.items():
                    if text and text.strip() and nl_channel:
                        em = discord.Embed(
                            title=f"\U0001f4f0  Matchday Intel \u2014 {league}",
                            description=text[:4096],
                            color=COLOR_BLUE,
                        )
                        em.set_footer(text="Spick's Picks \u2502 WHALE exclusive")
                        await nl_channel.send(embed=em)
                        posted_count += 1
                await interaction.followup.send(
                    f"Newsletter posted: {posted_count} league(s).", ephemeral=True)
            except Exception as exc:
                log.error("Manual newsletter trigger failed: %s", exc, exc_info=True)
                await interaction.followup.send(f"Error: {exc}", ephemeral=True)
            return

        # Best picks recap and track record don't need today's fixtures
        if task.value in ("best_picks_recap", "track_record"):
            posted = []
            try:
                if task.value == "best_picks_recap":
                    from prediction_tracker import resolve_outcomes, get_daily_breakdown
                    yesterday = (date.today() - timedelta(days=1)).isoformat()
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        _rag_executor, lambda: resolve_outcomes(prediction_date=yesterday)
                    )
                    breakdown = await loop.run_in_executor(
                        _rag_executor, lambda: get_daily_breakdown(prediction_date=yesterday)
                    )
                    if breakdown.get("total", 0) > 0:
                        bp_channel = self.bot.get_channel(CHANNEL_BEST_PICKS)
                        if bp_channel:
                            embeds = _build_best_picks_embeds(breakdown, yesterday)
                            for em in embeds:
                                await bp_channel.send(embed=em)
                            posted.append("best picks recap")
                        else:
                            posted.append("best picks recap (no channel configured)")
                    else:
                        posted.append("best picks recap (no graded predictions)")
                else:
                    from prediction_tracker import resolve_outcomes, get_track_record
                    yesterday = (date.today() - timedelta(days=1)).isoformat()
                    resolve_outcomes(prediction_date=yesterday)
                    tr_channel = self.bot.get_channel(CHANNEL_PARLAYS)
                    if tr_channel:
                        await self._post_track_record_embed(tr_channel)
                    posted.append("track record")
                await interaction.followup.send(
                    f"Posted: {', '.join(posted)}.", ephemeral=True)
            except Exception as exc:
                log.error("Manual trigger failed: %s", exc, exc_info=True)
                await interaction.followup.send(f"Error: {exc}", ephemeral=True)
            return

        date_str = _today_phrase()
        _, active_leagues = _get_todays_leagues()
        if not active_leagues:
            await interaction.followup.send(
                "No fixtures found in any league today.", ephemeral=True)
            return
        leagues = active_leagues
        log.info("Manual trigger '%s': active leagues = %s", task.value, leagues)

        posted = []
        try:
            if task.value in ("daily_picks", "everything"):
                await self._post_daily_picks(leagues, date_str)
                posted.append("daily picks")

            if task.value in ("team_lines", "everything"):
                await self._post_team_lines(leagues, date_str)
                posted.append("team lines")

            if task.value in ("parlays", "everything"):
                self._posted_today.pop("parlay", None)
                await self._post_parlays_now(leagues, date_str)
                posted.append("parlays")

            if task.value in ("player_props", "everything"):
                await self._post_player_props(leagues, date_str)
                posted.append("player props")

            if task.value in ("track_record", "everything"):
                try:
                    from prediction_tracker import resolve_outcomes, get_track_record
                    yesterday = (date.today() - timedelta(days=1)).isoformat()
                    resolve_result = resolve_outcomes(prediction_date=yesterday)
                    graded = resolve_result.get("graded", 0)
                    log.info("Resolved %d predictions for %s", graded, yesterday)
                    # Post the track record embed
                    tr_channel = self.bot.get_channel(CHANNEL_PARLAYS)
                    if tr_channel:
                        await self._post_track_record_embed(tr_channel)
                except Exception as exc:
                    log.error("Track record failed: %s", exc)
                posted.append("track record")

            if task.value in ("best_picks_recap", "everything"):
                try:
                    from prediction_tracker import resolve_outcomes, get_daily_breakdown
                    yesterday = (date.today() - timedelta(days=1)).isoformat()
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        _rag_executor, lambda: resolve_outcomes(prediction_date=yesterday)
                    )
                    breakdown = await loop.run_in_executor(
                        _rag_executor, lambda: get_daily_breakdown(prediction_date=yesterday)
                    )
                    if breakdown.get("total", 0) > 0:
                        bp_channel = self.bot.get_channel(CHANNEL_BEST_PICKS)
                        if bp_channel:
                            embeds = _build_best_picks_embeds(breakdown, yesterday)
                            for em in embeds:
                                await bp_channel.send(embed=em)
                except Exception as exc:
                    log.error("Best picks recap failed: %s", exc)
                posted.append("best picks recap")

            await interaction.followup.send(
                f"Posted: {', '.join(posted)}.", ephemeral=True)

        except Exception as exc:
            log.error("Manual trigger failed: %s", exc, exc_info=True)
            await interaction.followup.send(f"Error: {exc}", ephemeral=True)

    async def _post_daily_picks(self, leagues: list, date_str: str):
        """Post curated top picks per league — 5-8 best bets in ONE embed per league."""
        ch = self.bot.get_channel(CHANNEL_DAILY_PICKS)
        if not ch:
            log.warning("CHANNEL_DAILY_PICKS not configured")
            return

        # Market queries — we run all markets, collect all picks, rank, take top 5-8
        _ALL_QUERIES = {
            "moneyline": ("For every {league} game on {date}, show me moneyline probabilities and the best value pick.", "\U0001f3c6"),
            "btts": ("For every {league} game on {date}, should I take BTTS Yes or No?", "\U0001f91d"),
            "spreads": ("For every {league} game on {date}, recommend a handicap line.", "\U0001f4cf"),
            "sot": ("For every {league} game on {date}, recommend a shots on target over/under line.", "\U0001f3af"),
            "goals": ("Give me the over/under goals line I should take for each {league} game on {date}.", "\u26bd"),
            "corners": ("For every {league} game on {date}, recommend a corner totals line.", "\U0001f4d0"),
            "cards": ("For every {league} game on {date}, recommend a cards over/under line.", "\U0001f7e8"),
        }

        _MARKET_LABELS = {
            "moneyline": "Winner", "btts": "BTTS", "spreads": "Handicap",
            "sot": "SoT", "goals": "Goals", "corners": "Corners", "cards": "Cards",
        }

        for league in leagues:
            log.info("Collecting top picks for %s...", league)
            # Collect ALL picks: (fixture, market, pick_text, odds, model_p, edge, conf, emoji)
            all_picks = []

            for market_key, (prompt_tpl, emoji) in _ALL_QUERIES.items():
                prompt = prompt_tpl.format(league=league, date=date_str)
                text = await _run_rag(prompt, league)
                if not _is_valid(text):
                    continue
                picks = parse_fixture_picks(text)
                for fixture, line in picks.items():
                    if not line:
                        continue
                    # Parse components from the compact line
                    import re
                    pick_m = re.search(r"\*\*(.+?)\*\*", line)
                    odds_m = re.search(r"@\s*([\d.]+)", line)
                    model_m = re.search(r"([\d.]+)%\s*model", line)
                    edge_m = re.search(r"edge\s*([+\-][\d.]+)%", line)
                    conf = "low"
                    if re.search(r"\bhigh\b", line, re.I):
                        conf = "high"
                    elif re.search(r"\bmedium\b", line, re.I):
                        conf = "medium"

                    pick_text = pick_m.group(1) if pick_m else "?"
                    odds_val = float(odds_m.group(1)) if odds_m else 0
                    model_p = float(model_m.group(1)) if model_m else 0
                    edge_val = float(edge_m.group(1)) if edge_m else 0

                    label = _MARKET_LABELS.get(market_key, market_key.title())
                    all_picks.append({
                        "fixture": fixture,
                        "market": market_key,
                        "label": label,
                        "emoji": emoji,
                        "pick": pick_text,
                        "odds": odds_val,
                        "model_p": model_p,
                        "edge": edge_val,
                        "conf": conf,
                    })

            if not all_picks:
                continue

            # Rank: high confidence first, then medium, then low; within tier by edge descending
            conf_rank = {"high": 2, "medium": 1, "low": 0}
            all_picks.sort(key=lambda p: (
                conf_rank.get(p["conf"], 0),
                p["edge"],
            ), reverse=True)

            # Take top 8 — prefer medium+ but fill with low if needed
            medium_plus = [p for p in all_picks if p["conf"] in ("high", "medium")]
            if len(medium_plus) >= 8:
                top = medium_plus[:8]
            else:
                top = all_picks[:8]
            top = all_picks[:8]

            # Build ONE clean embed for this league
            league_logo = None
            try:
                from embeds import _get_league_logo
                league_logo = _get_league_logo(league)
            except Exception:
                pass

            n_strong = sum(1 for p in top if p["conf"] in ("high", "medium"))
            em = discord.Embed(
                title=f"\U0001f4ca  Today's Top Picks \u2014 {league}",
                description=f"**{date_str}** \u2022 {len(top)} curated picks from {len(all_picks)} analyzed",
                color=COLOR_GREEN if any(p["conf"] == "high" for p in top) else COLOR_YELLOW,
            )
            if league_logo:
                em.set_thumbnail(url=league_logo)

            for i, p in enumerate(top, 1):
                # Confidence meter
                if p["conf"] == "high":
                    meter = "\u25cf\u25cf\u25cf\u25cf\u25cb"
                    conf_label = "Strong Edge"
                elif p["conf"] == "medium":
                    meter = "\u25cf\u25cf\u25cf\u25cb\u25cb"
                    conf_label = "Leaning"
                else:
                    meter = "\u25cf\u25cb\u25cb\u25cb\u25cb"
                    conf_label = "Speculative"

                # Compact field
                odds_str = f"@ {p['odds']:.2f}" if p["odds"] else ""
                edge_str = f"+{p['edge']:.1f}%" if p["edge"] > 0 else ""

                em.add_field(
                    name=f"{p['emoji']}  {p['fixture']}",
                    value=(
                        f"**{p['pick']}** {odds_str}\n"
                        f"{meter} {conf_label} \u2022 {p['model_p']:.0f}% model"
                        + (f" \u2022 {edge_str} edge" if edge_str else "")
                    ),
                    inline=False,
                )

            em.set_footer(text=f"{league} \u2502 Spick's Picks \u2502 {len(all_picks)} picks analyzed, top {len(top)} shown")
            await ch.send(embed=em)

        self._mark_posted("daily_picks")

    async def _post_parlays_now(self, leagues: list, date_str: str):
        """Post parlays immediately — bypasses time window check (for /trigger)."""
        channel = self.bot.get_channel(CHANNEL_PARLAYS)
        if not channel:
            log.warning("CHANNEL_PARLAYS not configured")
            return

        log.info("Posting parlays (manual trigger)...")

        built = []
        for preset in _auto_parlay_presets(date.today(), leagues):
            try:
                result = await _build_structured_parlay(preset["request"])
            except ParlayBuildError as exc:
                log.warning("Manual auto parlay '%s' failed: %s | %s", preset["title"], exc.message, "; ".join(exc.notes))
                continue
            except Exception as exc:
                log.error("Manual auto parlay '%s' failed: %s", preset["title"], exc, exc_info=True)
                continue
            built.append((preset, result))

        if not built:
            log.warning("All 3 structured parlay presets failed.")
            return

        header = discord.Embed(
            title="\U0001f3af  Parlay of the Day",
            description=f"**{date_str}** \u2014 3 parlays ranked by risk level",
            color=COLOR_GREEN,
        )
        header.set_footer(text="Spick | Auto-generated from model projections")
        await channel.send(embed=header)

        for preset, result in built:
            for em in structured_parlay_embeds(result, league="Cross-League"):
                em.title = preset["title"]
                em.color = preset["color"]
                await channel.send(embed=em)

        self._mark_posted("parlay")

    async def _post_team_lines(self, leagues: list, date_str: str):
        """Post per-team corner & card lines — ONE embed per league."""
        ch = self.bot.get_channel(CHANNEL_TEAM_LINES)
        if not ch:
            log.warning("CHANNEL_TEAM_LINES not configured")
            return

        log.info("Posting team lines (compact, one embed per league)...")

        try:
            from embeds import _get_league_logo, _split_fixture_blocks, _parse_team_totals_block, _conf_bar
        except ImportError:
            log.error("Could not import embed helpers for team lines")
            return

        for league in leagues:
            prompt = (
                f"For all {league} games on {date_str}, what are the per-team "
                "corner and card lines for each fixture?"
            )
            text = await _run_rag(prompt, league)
            if not _is_valid(text):
                continue

            # Parse all fixture blocks
            blocks = _split_fixture_blocks(text)
            if not blocks:
                continue

            league_logo = _get_league_logo(league)
            em = discord.Embed(
                title=f"\U0001f4cb  Team Lines \u2014 {league}",
                description=f"**{date_str}** \u2022 Per-team corner & card lines",
                color=COLOR_BLUE,
            )
            if league_logo:
                em.set_thumbnail(url=league_logo)

            for i, block in enumerate(blocks):
                # Parse fixture name
                import re
                fx_m = re.match(r"(.+?)\s*===", block)
                fixture = fx_m.group(1).strip() if fx_m else f"Fixture {i+1}"

                # Parse teams from the block
                teams = _parse_team_totals_block(block)
                corners = [t for t in teams if t.get("stat") == "corners"]
                cards = [t for t in teams if t.get("stat") == "cards"]

                # Build fixture section — each team on its own line for readability
                lines = [f"**{fixture}**"]

                if corners:
                    lines.append("\U0001f4d0 **Corners**")
                    for t in corners:
                        rec = t.get("recommendation", "")
                        prob = t.get("model_prob", "")
                        conf = t.get("confidence", "low").lower()
                        bar = _conf_bar(conf)
                        entry = f"\u2003{bar} {t['team']}: {t['projection']} proj"
                        if rec:
                            entry += f" \u2192 **{rec}**"
                        if prob:
                            entry += f" ({prob})"
                        lines.append(entry)

                if cards:
                    lines.append("\U0001f7e8 **Cards**")
                    for t in cards:
                        rec = t.get("recommendation", "")
                        prob = t.get("model_prob", "")
                        conf = t.get("confidence", "low").lower()
                        bar = _conf_bar(conf)
                        entry = f"\u2003{bar} {t['team']}: {t['projection']} proj"
                        if rec:
                            entry += f" \u2192 **{rec}**"
                        if prob:
                            entry += f" ({prob})"
                        lines.append(entry)

                fixture_text = "\n".join(lines)

                # Add separator between fixtures
                if i > 0:
                    fixture_text = "\u2500" * 30 + "\n" + fixture_text

                # Use description for first fixture, fields for the rest
                # (Discord embeds: description has 4096 char limit, fields have 1024)
                if i == 0 and len(blocks) <= 3:
                    # For small number of fixtures, use description for all
                    pass
                em.add_field(name="\u200b", value=fixture_text[:1024], inline=False)

            em.set_footer(text=f"{league} \u2502 Spick's Picks")
            await ch.send(embed=em)

        self._mark_posted("team_lines")

    async def _post_player_props(self, leagues: list, date_str: str):
        """Post compact player prop picks — ONE embed per fixture."""
        ch = self.bot.get_channel(CHANNEL_PLAYER_PROPS)
        if not ch:
            log.warning("CHANNEL_PLAYER_PROPS not configured")
            return

        log.info("Posting player props (compact format)...")

        _pp_queries = [
            ("who are the best anytime goalscorer picks?", "goals", "\u26bd", "Anytime Goalscorer"),
            ("which players are most likely to be booked?", "cards", "\U0001f7e8", "To Be Booked"),
            ("which players are best for shots on target props?", "sot", "\U0001f3af", "1+ Shot on Target"),
            ("which players are most likely to foul?", "fouls", "\U0001f6d1", "To Commit a Foul"),
        ]

        for league in leagues:
            # Build team → fixture mapping
            team_to_fixture: dict = {}
            events, _ = fetch_events(league, target_date=date.today())
            day_events = filter_events_by_exact_date(events, date.today())
            for ev in day_events:
                h = ev.get("home_team", "")
                a = ev.get("away_team", "")
                fx_name = f"{h} vs {a}"
                team_to_fixture[h.lower()] = fx_name
                team_to_fixture[a.lower()] = fx_name

            # Collect all picks grouped by fixture
            fixture_picks: dict = {}
            for suffix, stat_key, emoji, bet_label in _pp_queries:
                prompt = f"For all {league} games on {date_str}, {suffix}"
                text = await _run_rag(prompt, league)
                if not _is_valid(text):
                    continue
                picks = _parse_player_picks_from_text(text, stat_key, emoji, bet_label)
                for p in picks:
                    team_lower = p.get("team", "").lower()
                    fx = team_to_fixture.get(team_lower, f"{league} match")
                    fixture_picks.setdefault(fx, []).append(p)

            # Build one embed per fixture using shared builder
            for fixture, picks_list in fixture_picks.items():
                em = _build_player_props_embed(picks_list, fixture, league, date_str)
                if em:
                    await ch.send(embed=em)

        self._mark_posted("player_props")

    async def _post_track_record_embed(self, channel):
        """Post detailed track record with per-market and per-confidence breakdown."""
        try:
            from prediction_tracker import get_track_record, get_calibration_data
        except ImportError:
            return

        try:
            stats = get_track_record()
        except Exception:
            return

        total = stats.get("total_graded", 0)
        if total == 0:
            em = discord.Embed(
                title="\U0001f4ca Track Record",
                description="No predictions resolved yet. Check back after today's matches finish.",
                color=COLOR_BLUE,
            )
            await channel.send(embed=em)
            return

        hits = stats.get("hits", 0)
        hit_rate = stats.get("hit_rate", 0)
        roi = stats.get("roi_flat_stake", 0)
        streak = stats.get("recent_streak", 0)

        # Color based on hit rate
        if hit_rate >= 0.55:
            color = COLOR_GREEN
        elif hit_rate >= 0.48:
            color = COLOR_YELLOW
        else:
            color = COLOR_RED

        streak_str = f"W{streak}" if streak > 0 else (f"L{abs(streak)}" if streak < 0 else "\u2014")

        em = discord.Embed(
            title="\U0001f4ca Track Record \u2014 All Time",
            description=f"**{hits}/{total}** predictions hit ({hit_rate:.1%}) | ROI: {roi:+.1%} | Streak: {streak_str}",
            color=color,
        )

        # By confidence
        by_conf = stats.get("by_confidence", {})
        conf_lines = []
        for level in ["high", "medium", "low"]:
            c = by_conf.get(level, {})
            if c.get("total", 0) > 0:
                label = {"high": "\u25cf\u25cf\u25cf\u25cf\u25cb Strong Edge", "medium": "\u25cf\u25cf\u25cf\u25cb\u25cb Leaning", "low": "\u25cf\u25cb\u25cb\u25cb\u25cb Speculative"}[level]
                conf_lines.append(f"{label}: **{c['hits']}/{c['total']}** ({c['hit_rate']:.0%})")
        if conf_lines:
            em.add_field(name="By Confidence", value="\n".join(conf_lines), inline=False)

        # By market
        by_market = stats.get("by_market", {})
        market_lines = []
        market_emojis = {"goals": "\u26bd", "corners": "\U0001f4d0", "cards": "\U0001f7e8", "sot": "\U0001f3af",
                         "btts": "\U0001f91d", "moneyline": "\U0001f3c6", "spreads": "\U0001f4cf"}
        for market, m in sorted(by_market.items(), key=lambda x: x[1].get("total", 0), reverse=True):
            if m.get("total", 0) > 0:
                emoji = market_emojis.get(market, "\U0001f4cb")
                market_lines.append(f"{emoji} {market.title()}: **{m['hits']}/{m['total']}** ({m['hit_rate']:.0%})")
        if market_lines:
            em.add_field(name="By Market", value="\n".join(market_lines[:8]), inline=False)

        # Calibration summary (if available)
        try:
            cal_data = get_calibration_data()
            if cal_data:
                cal_lines = []
                for bucket in cal_data:
                    if bucket.get("count", 0) >= 10:
                        model_p = bucket["model_avg"]
                        actual = bucket["actual_rate"]
                        diff = actual - model_p
                        arrow = "\u2705" if abs(diff) < 0.05 else ("\U0001f4c8" if diff > 0 else "\U0001f4c9")
                        cal_lines.append(f"{arrow} {bucket['bucket']}: model {model_p:.0%} \u2192 actual {actual:.0%}")
                if cal_lines:
                    em.add_field(name="Calibration", value="\n".join(cal_lines[:6]), inline=False)
        except Exception:
            pass

        em.set_footer(text=f"Based on {total} resolved predictions \u2502 Spick's Picks")
        await channel.send(embed=em)


async def setup(bot: commands.Bot):
    await bot.add_cog(AutoPush(bot))
