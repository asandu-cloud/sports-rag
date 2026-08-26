"""Slash commands — consolidated RAG-powered betting predictions.

Commands:
    /fixtures  — Browse upcoming fixtures for a league
    /analyze   — Single-fixture deep dive with market focus
    /market    — League-wide market advice (replaces 10 old commands)
    /compare   — Head-to-head stat comparison for a fixture
    /players   — Player prop recommendations
    /parlay    — Auto-generate a parlay (kept from original)
    /help      — Guided onboarding
"""

from __future__ import annotations

import io
import re
import sys
import time
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import discord
from discord import app_commands
from discord.ext import commands

# ---------------------------------------------------------------------------
# Path setup — import RAG engine and bot helpers
# ---------------------------------------------------------------------------
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "rag_ingest"))
import rag_cli_v2 as rag  # noqa: E402
from rendering.market_dispatch import (  # noqa: E402
    canonical_market_name,
    evaluate_market_results_sync,
    render_market_answer_sync,
)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from embeds import market_results_to_embeds, rag_output_to_embeds, parlay_embed  # noqa: E402
from config import ALL_LEAGUES, COLOR_BLUE, COLOR_PURPLE, COLOR_GREEN  # noqa: E402
from cogs.access import premium_only, pooled_command, tier_required, get_or_create_private_thread  # noqa: E402


# ============================================================================
# Shared Utilities
# ============================================================================

def _run_rag_sync(prompt: str, league: str) -> str:
    """Synchronous RAG call — never call from event loop directly."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rag.answer_once(prompt, default_league=league)
    return buf.getvalue().strip() or "(No output from model)"


async def _run_rag(prompt: str, league: str) -> str:
    """Async wrapper — runs RAG in thread pool so event loop stays free."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run_rag_sync, prompt, league)


async def _render_market_answer(league: str, market: str, target_date: date) -> str:
    """Async wrapper for deterministic market rendering."""
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, render_market_answer_sync, league, market, target_date)


async def _evaluate_market_results(league: str, market: str, target_date: date):
    """Run canonical market evaluation outside the Discord event loop."""
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, evaluate_market_results_sync, league, market, target_date)


# ---------------------------------------------------------------------------
# Full match analysis pipeline
# ---------------------------------------------------------------------------

# OpenAI client (shared with whale_chat pattern)
try:
    from openai import OpenAI as _OpenAI
    _analysis_client = _OpenAI(api_key=__import__("os").getenv("OPENAI_API_KEY", ""))
    _ANALYSIS_MODEL = __import__("os").getenv("WHALE_CHAT_MODEL",
                                               __import__("os").getenv("RAG_CHAT_MODEL", "gpt-4o-mini"))
except ImportError:
    _analysis_client = None
    _ANALYSIS_MODEL = ""


def _safe_round(val, decimals=2):
    if val is None:
        return None
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return None


def _build_match_analysis_sync(home: str, away: str, league: str,
                                target_date: date) -> Dict:
    """Run ALL projections and odds extraction for a single fixture.
    Returns a structured dict with every market's data. Synchronous."""

    analysis: Dict = {"home": home, "away": away, "league": league}
    fixture_date_str = target_date.isoformat()

    # --- Resolve team names ---
    home = rag.canonical_team_name(home) or home
    away = rag.canonical_team_name(away) or away
    analysis["home"] = home
    analysis["away"] = away

    # --- Team profiles ---
    h_meta = rag.get_team_profile_meta(home, league) or {}
    a_meta = rag.get_team_profile_meta(away, league) or {}
    analysis["home_profile"] = {
        k: _safe_round(h_meta.get(k)) for k in [
            "goals_for_pm", "goals_against_pm", "corners_pm", "corners_against_pm",
            "cards_per_90_team", "sot_for_pm", "sot_against_pm", "possession",
            "dominance_index", "control_index", "form_index_team",
            "xg_home_pm", "corners_home_pm", "corners_away_pm",
        ] if h_meta.get(k) is not None
    }
    analysis["away_profile"] = {
        k: _safe_round(a_meta.get(k)) for k in [
            "goals_for_pm", "goals_against_pm", "corners_pm", "corners_against_pm",
            "cards_per_90_team", "sot_for_pm", "sot_against_pm", "possession",
            "dominance_index", "control_index", "form_index_team",
            "xg_away_pm", "corners_home_pm", "corners_away_pm",
        ] if a_meta.get(k) is not None
    }

    # --- Recent form (last 6) ---
    h_recent = rag._recent_stats(home, league, last_n=6) or {}
    a_recent = rag._recent_stats(away, league, last_n=6) or {}
    analysis["home_recent"] = {k: _safe_round(v) for k, v in h_recent.items()
                                if v is not None and isinstance(v, (int, float))}
    analysis["away_recent"] = {k: _safe_round(v) for k, v in a_recent.items()
                                if v is not None and isinstance(v, (int, float))}

    # --- Core projections ---
    # Goals
    try:
        g_bl, g_s, g_r = rag.projected_total_goals(home, away, league, fixture_date=fixture_date_str)
        analysis["goals"] = {"projected": _safe_round(g_bl), "season": _safe_round(g_s),
                             "recent": _safe_round(g_r)}
    except Exception:
        analysis["goals"] = None

    # Corners
    try:
        c_bl, c_s, c_r = rag.projected_total_corners(home, away, league, fixture_date=fixture_date_str)
        analysis["corners"] = {"projected": _safe_round(c_bl), "season": _safe_round(c_s),
                               "recent": _safe_round(c_r)}
    except Exception:
        analysis["corners"] = None

    # Cards
    try:
        k_bl, k_s, k_r, ref_mod = rag.projected_total_cards(home, away, league, fixture_date=fixture_date_str)
        analysis["cards"] = {"projected": _safe_round(k_bl), "season": _safe_round(k_s),
                             "recent": _safe_round(k_r)}
        if ref_mod and ref_mod.source == "profile" and ref_mod.referee_name:
            analysis["referee"] = {
                "name": ref_mod.referee_name,
                "avg_cards": _safe_round(ref_mod.avg_cards_per_match),
                "strictness": _safe_round(ref_mod.strictness_ratio),
                "cards_per_foul": _safe_round(ref_mod.cards_per_foul, 3),
                "avg_fouls": _safe_round(ref_mod.avg_fouls_per_match, 1),
                "sample_size": ref_mod.sample_size,
            }
    except Exception:
        analysis["cards"] = None

    # SoT
    try:
        s_bl, s_s, s_r = rag.projected_total_sot(home, away, league, fixture_date=fixture_date_str)
        analysis["sot"] = {"projected": _safe_round(s_bl), "season": _safe_round(s_s),
                           "recent": _safe_round(s_r)}
    except Exception:
        analysis["sot"] = None

    # BTTS
    try:
        p_btts, h_g, a_g, _, _ = rag.projected_btts_prob(home, away, league, fixture_date=fixture_date_str)
        analysis["btts"] = {"btts_yes_prob": _safe_round(p_btts, 3),
                            "home_goals_proj": _safe_round(h_g),
                            "away_goals_proj": _safe_round(a_g)}
    except Exception:
        analysis["btts"] = None

    # Moneyline
    try:
        p_h, p_d, p_a, _, _ = rag.projected_moneyline_probs(home, away, league, fixture_date=fixture_date_str)
        ml_pairs = [
            {"side": "home", "team": home, "prob": _safe_round(p_h, 3)},
            {"side": "draw", "team": "Draw", "prob": _safe_round(p_d, 3)},
            {"side": "away", "team": away, "prob": _safe_round(p_a, 3)},
        ]
        analysis["moneyline"] = {
            "home_win": _safe_round(p_h, 3),
            "draw": _safe_round(p_d, 3),
            "away_win": _safe_round(p_a, 3),
            "most_likely": max(ml_pairs, key=lambda x: x["prob"] or 0.0),
        }
    except Exception:
        analysis["moneyline"] = None

    # Spread/handicap
    try:
        d_bl, d_s, d_r = rag.projected_goal_difference(home, away, league, fixture_date=fixture_date_str)
        analysis["spread"] = {"projected_diff": _safe_round(d_bl),
                              "season_diff": _safe_round(d_s),
                              "recent_diff": _safe_round(d_r)}
    except Exception:
        analysis["spread"] = None

    # Correct score (top 5)
    try:
        cs_probs, cs_h, cs_a = rag.projected_correct_score_probs(home, away, league, fixture_date=fixture_date_str)
        if cs_probs:
            top5 = sorted(cs_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            analysis["correct_score"] = {
                "top_scorelines": [{"score": f"{h}-{a}", "prob": _safe_round(p, 3)}
                                   for (h, a), p in top5],
            }
    except Exception:
        analysis["correct_score"] = None

    # Per-team corners
    try:
        hc_bl, hc_s, hc_r = rag.projected_team_corners(home, away, league, is_home=True, fixture_date=fixture_date_str)
        ac_bl, ac_s, ac_r = rag.projected_team_corners(away, home, league, is_home=False, fixture_date=fixture_date_str)
        analysis["team_corners"] = {
            "home": {"projected": _safe_round(hc_bl), "season": _safe_round(hc_s)},
            "away": {"projected": _safe_round(ac_bl), "season": _safe_round(ac_s)},
        }
    except Exception:
        analysis["team_corners"] = None

    # Per-team cards
    try:
        hk_bl, hk_s, hk_r = rag.projected_team_cards(home, away, league, is_home=True, fixture_date=fixture_date_str)
        ak_bl, ak_s, ak_r = rag.projected_team_cards(away, home, league, is_home=False, fixture_date=fixture_date_str)
        analysis["team_cards"] = {
            "home": {"projected": _safe_round(hk_bl), "season": _safe_round(hk_s)},
            "away": {"projected": _safe_round(ak_bl), "season": _safe_round(ak_s)},
        }
    except Exception:
        analysis["team_cards"] = None

    # --- Odds extraction + best lines ---
    try:
        events, _ = rag.fetch_events(league, set(), target_date=target_date)
    except Exception:
        events = []

    event = None
    for ev in events:
        if (home.lower() in ev.get("home_team", "").lower() and
                away.lower() in ev.get("away_team", "").lower()):
            event = ev
            break

    if event:
        analysis["kickoff"] = event.get("commence_time", "")

        # Enrich with all markets
        try:
            rag.enrich_events_for_groups([event], league, {"corners", "cards", "sot", "btts"})
        except Exception:
            pass

        # Goals line
        _extract_best_line(analysis, event, "goals", "totals",
                           analysis.get("goals", {}).get("projected"), home, league)
        # Corners line
        _extract_best_line(analysis, event, "corners", "corners",
                           analysis.get("corners", {}).get("projected"), home, league)
        # Cards line
        _extract_best_line(analysis, event, "cards", "cards",
                           analysis.get("cards", {}).get("projected"), home, league)
        # SoT line
        _extract_best_line(analysis, event, "sot", "sot",
                           analysis.get("sot", {}).get("projected"), home, league)

        # Spread line
        if analysis.get("spread", {}).get("projected_diff") is not None:
            try:
                sp_options = rag.extract_spread_line_options(event)
                if sp_options:
                    spread_result = rag.select_best_spread_recommendation(
                        sp_options,
                        analysis["spread"]["projected_diff"],
                        home,
                        away_team=away,
                        league=league,
                    )
                    best = spread_result.get("bet_recommendation")
                    if best:
                        analysis["spread"]["best_line"] = {
                            "team": best.get("team"), "point": best.get("point"),
                            "odds": best.get("odds"), "bookmaker": best.get("bookmaker"),
                            "model_prob": _safe_round(best.get("_model_prob"), 3),
                            "value_edge": _safe_round(best.get("_value_edge"), 3),
                        }
                    best_value = spread_result.get("best_value")
                    if best_value:
                        analysis["spread"]["best_value"] = {
                            "team": best_value.get("team"), "point": best_value.get("point"),
                            "odds": best_value.get("odds"), "bookmaker": best_value.get("bookmaker"),
                            "model_prob": _safe_round(best_value.get("_model_prob"), 3),
                            "value_edge": _safe_round(best_value.get("_value_edge"), 3),
                        }
                    if spread_result.get("no_bet_reason"):
                        analysis["spread"]["no_bet_reason"] = spread_result.get("no_bet_reason")
            except Exception:
                pass

        # BTTS odds
        if analysis.get("btts", {}).get("btts_yes_prob") is not None:
            try:
                btts_options = rag.extract_btts_odds(event)
                if btts_options:
                    btts_result = rag.select_best_btts_recommendation(
                        btts_options,
                        analysis["btts"]["btts_yes_prob"],
                    )
                    best = btts_result.get("bet_recommendation")
                    if best:
                        analysis["btts"]["best_side"] = {
                            "side": best.get("side"), "odds": best.get("odds"),
                            "bookmaker": best.get("bookmaker"),
                            "model_prob": _safe_round(best.get("_model_prob"), 3),
                            "value_edge": _safe_round(best.get("_value_edge"), 3),
                        }
                    best_value = btts_result.get("best_value")
                    if best_value:
                        analysis["btts"]["best_value"] = {
                            "side": best_value.get("side"), "odds": best_value.get("odds"),
                            "bookmaker": best_value.get("bookmaker"),
                            "model_prob": _safe_round(best_value.get("_model_prob"), 3),
                            "value_edge": _safe_round(best_value.get("_value_edge"), 3),
                        }
                    if btts_result.get("no_bet_reason"):
                        analysis["btts"]["no_bet_reason"] = btts_result.get("no_bet_reason")
            except Exception:
                pass

        # Moneyline odds
        if analysis.get("moneyline"):
            try:
                ml_options = rag.extract_moneyline_odds(event)
                if ml_options:
                    ml = analysis["moneyline"]
                    result = rag.choose_best_moneyline_side(
                        ml["home_win"] or 0, ml["draw"] or 0, ml["away_win"] or 0,
                        ml_options, home, away)
                    if result:
                        best_value = result.get("best_value")
                        most_likely = result.get("most_likely")
                        rec = result.get("bet_recommendation") or result.get("recommended")
                        if most_likely:
                            ml["most_likely"] = {
                                "side": most_likely.get("side"),
                                "team": most_likely.get("team"),
                                "prob": _safe_round(most_likely.get("model_prob"), 3),
                            }
                        if best_value:
                            ml["best_value"] = {
                                "side": best_value.get("side"),
                                "team": best_value.get("team"),
                                "odds": best_value.get("best_odds"),
                                "bookmaker": best_value.get("bookmaker"),
                                "model_prob": _safe_round(best_value.get("model_prob"), 3),
                                "value_edge": _safe_round(best_value.get("value_edge"), 3),
                                "ev": _safe_round(best_value.get("ev"), 3),
                            }
                        if result.get("no_bet_reason"):
                            ml["no_bet_reason"] = result.get("no_bet_reason")
                    else:
                        rec = None
                    if rec:
                        analysis["moneyline"]["best_side"] = {
                            "side": rec.get("side"),
                            "team": rec.get("team"),
                            "odds": rec.get("best_odds"),
                            "bookmaker": rec.get("bookmaker"),
                            "model_prob": _safe_round(rec.get("model_prob"), 3),
                            "value_edge": _safe_round(rec.get("value_edge"), 3),
                        }
            except Exception:
                pass

    return analysis


def _extract_best_line(analysis: Dict, event: Dict, stat: str, stat_group: str,
                        projection: Optional[float], home: str, league: str):
    """Helper: extract best over/under line for a stat and attach to analysis dict."""
    if projection is None:
        return
    try:
        options = rag.extract_total_line_options(event, stat_group)
        if not options:
            return
        variance = None
        var_keys = {"goals": "goals_var", "corners": "corners_for_var",
                    "cards": "cards_var", "sot": "sot_for_var"}
        vk = var_keys.get(stat)
        if vk:
            try:
                variance = rag.get_blended_variance(home, league, vk)
            except Exception:
                pass
        result = rag.select_best_total_recommendation(options, projection, variance)
        best = result.get("bet_recommendation")
        if best:
            key = f"{stat}_line" if stat != "goals" else "goals_line"
            analysis[key] = {
                "side": best.get("side"), "point": best.get("point"),
                "odds": best.get("odds"), "bookmaker": best.get("bookmaker"),
                "model_prob": _safe_round(best.get("_model_prob"), 3),
                "value_edge": _safe_round(best.get("_value_edge"), 3),
                "confidence": rag.confidence_from_edge(
                    abs(best.get("_value_edge", 0) or 0), stat_group=stat,
                    model_prob=best.get("_model_prob"),
                    value_edge_pct=best.get("_value_edge")),
            }
        elif result.get("best_value") or result.get("no_bet_reason"):
            key = f"{stat}_line" if stat != "goals" else "goals_line"
            best_value = result.get("best_value")
            analysis[key] = {
                "no_bet_reason": result.get("no_bet_reason"),
                "best_value": {
                    "side": best_value.get("side"), "point": best_value.get("point"),
                    "odds": best_value.get("odds"), "bookmaker": best_value.get("bookmaker"),
                    "model_prob": _safe_round(best_value.get("_model_prob"), 3),
                    "value_edge": _safe_round(best_value.get("_value_edge"), 3),
                } if best_value else None,
            }
    except Exception:
        pass


_ANALYSIS_SYSTEM_PROMPT = """You are Spix, an expert football betting analyst writing a comprehensive pre-match report.

You will receive structured data from statistical models covering every market for a fixture. Your job is to write a natural, insightful analysis that a serious bettor would find valuable.

Rules:
- ONLY use the numbers provided in the data. Never invent statistics.
- If a field is null or missing, skip that section gracefully.
- Write in a confident, concise tone. No hedging or disclaimers.
- Use the team names, not pronouns.
- Display odds in decimal format.
- When a value edge exists, highlight it clearly.
- Structure the report with these sections (skip any with no data):

◆ MATCH OVERVIEW — One-paragraph context: who's favored, why, headline pick.

◆ MONEYLINE — Win probabilities for each outcome. Best value side with odds and edge.

◆ GOALS — Projected total, best over/under line, BTTS verdict with probability.

◆ CORNERS — Match total + per-team projections. Best line.

◆ CARDS — Match total + per-team. Referee impact (if assigned). Best line.

◆ SHOTS ON TARGET — Projected total, best line.

◆ HANDICAP — Projected goal difference, best spread line.

◆ CORRECT SCORE — Top 3 most likely scorelines with probabilities.

◆ BEST VALUE — The single highest-edge pick across all markets. State the pick, odds, model probability, and value edge clearly.

Keep the full report under 600 words. Use ━ separators between sections."""


def _generate_analysis_report_sync(analysis: Dict) -> str:
    """Generate a written match analysis report from structured data via GPT."""
    import json as _json

    if not _analysis_client:
        return "AI analysis unavailable — OpenAI not configured."

    data_str = _json.dumps(analysis, indent=2, default=str)

    try:
        resp = _analysis_client.chat.completions.create(
            model=_ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": f"Write a full match analysis report from this data:\n\n{data_str}"},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        return resp.choices[0].message.content or "(No response from AI)"
    except Exception as exc:
        return f"Analysis generation failed: {exc}"


async def _run_full_analysis(home: str, away: str, league: str,
                              target_date: date) -> str:
    """Async wrapper for the full analysis pipeline."""
    import asyncio
    loop = asyncio.get_running_loop()
    analysis = await loop.run_in_executor(
        None, _build_match_analysis_sync, home, away, league, target_date)
    report = await loop.run_in_executor(
        None, _generate_analysis_report_sync, analysis)
    return report


# ---------------------------------------------------------------------------
# Date parsing — flexible input
# ---------------------------------------------------------------------------

def _parse_date(raw: Optional[str] = None) -> date:
    """Parse flexible date input.

    Accepts: None / today / tomorrow / day-name / YYYY-MM-DD / Month Day.
    """
    if not raw:
        return date.today()
    low = raw.strip().lower()
    today = date.today()
    if low == "today":
        return today
    if low == "tomorrow":
        return today + timedelta(days=1)
    day_names = [
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    ]
    if low in day_names:
        target_wd = day_names.index(low)
        diff = (target_wd - today.weekday()) % 7
        if diff == 0:
            diff = 7
        return today + timedelta(days=diff)
    # ISO format YYYY-MM-DD
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        pass
    # Natural language month+day
    for fmt in ("%B %d", "%d %B", "%b %d", "%d %b"):
        try:
            parsed = datetime.strptime(raw.strip(), fmt).date().replace(year=today.year)
            if parsed < today:
                parsed = parsed.replace(year=today.year + 1)
            return parsed
        except ValueError:
            continue
    return today


# ---------------------------------------------------------------------------
# Date phrase — for natural language prompts
# ---------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]}"


def _date_phrase(d: date) -> str:
    return f"{_ordinal(d.day)} {d.strftime('%B')}"


# ---------------------------------------------------------------------------
# Event cache — for autocomplete (5 min TTL)
# ---------------------------------------------------------------------------

_event_cache: Dict[Tuple[str, str], Tuple[float, list]] = {}
_CACHE_TTL = 300


def _get_events_cached(league: str, target: date, upcoming_only: bool = True) -> list:
    key = (league, target.isoformat())
    now_ts = time.time()
    if key in _event_cache:
        ts, evts = _event_cache[key]
        if now_ts - ts < _CACHE_TTL:
            if upcoming_only:
                return _filter_upcoming(evts)
            return evts
    try:
        events, _ = rag.fetch_events(league, set(rag.DEFAULT_MARKETS), target_date=target)
        _event_cache[key] = (now_ts, events)
        if upcoming_only:
            return _filter_upcoming(events)
        return events
    except Exception:
        return []


def _filter_upcoming(events: list) -> list:
    """Remove events that have already kicked off or finished."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    upcoming = []
    for ev in events:
        ko = ev.get("commence_time", "")
        if not ko:
            upcoming.append(ev)  # keep if no kickoff time (safety)
            continue
        try:
            ko_dt = datetime.fromisoformat(str(ko).replace("Z", "+00:00"))
            if ko_dt.tzinfo is None:
                ko_dt = ko_dt.replace(tzinfo=timezone.utc)
            if ko_dt > now:
                upcoming.append(ev)
        except Exception:
            upcoming.append(ev)  # keep on parse failure
    return upcoming


# ---------------------------------------------------------------------------
# Fixture autocomplete — shared callback
# ---------------------------------------------------------------------------

async def _match_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> List[app_commands.Choice[str]]:
    ns = interaction.namespace
    league = getattr(ns, "league", None)
    if league and hasattr(league, "value"):
        league = league.value
    if not league:
        return []
    target = _parse_date(getattr(ns, "date", None))
    events = _get_events_cached(league, target)
    choices: List[app_commands.Choice[str]] = []
    for ev in events:
        label = f"{ev['home_team']} vs {ev['away_team']}"
        if not current or current.lower() in label.lower():
            choices.append(app_commands.Choice(name=label[:100], value=label[:100]))
    return choices[:25]


# ---------------------------------------------------------------------------
# League choices
# ---------------------------------------------------------------------------

def _league_choices() -> list:
    return [app_commands.Choice(name=lg, value=lg) for lg in ALL_LEAGUES]


# ============================================================================
# Prompt Templates
# ============================================================================

# -- /analyze (single-fixture) prompts --
_ANALYZE_PROMPTS: Dict[str, str] = {
    "overview": (
        "Give me a full match analysis for {home} vs {away} in the {league} on "
        "{date}. Cover moneyline probabilities, goals over/under, BTTS, corners, "
        "and cards recommendations."
    ),
    "totals": "For {home} vs {away} in {league} on {date}, what over/under goals line should I take?",
    "corners": "For {home} vs {away} in {league} on {date}, recommend a corner totals line.",
    "cards": "For {home} vs {away} in {league} on {date}, recommend a cards over/under line.",
    "sot": "For {home} vs {away} in {league} on {date}, recommend a shots on target over/under line.",
    "btts": "For {home} vs {away} in {league} on {date}, should I take BTTS Yes or No?",
    "moneyline": (
        "For {home} vs {away} in {league} on {date}, show me moneyline probabilities "
        "and the best value pick."
    ),
    "spreads": "For {home} vs {away} in {league} on {date}, recommend a handicap line.",
    "correctscore": "For {home} vs {away} in {league} on {date}, predict the most likely correct score.",
    "intervals": "For {home} vs {away} in {league} on {date}, show me goal interval probabilities.",
    "teamlines": (
        "For {home} vs {away} in {league} on {date}, what are the per-team corner "
        "and card lines?"
    ),
}

# -- /market (league-wide) prompts --
_MARKET_PROMPTS: Dict[str, str] = {
    "totals": "Give me the over/under goals line I should take for each {league} game on {date}.",
    "corners": "For every {league} game on {date}, recommend a corner totals line for each fixture.",
    "cards": "For each {league} game on {date}, recommend a cards over/under line.",
    "sot": (
        "For every {league} game on {date}, recommend a shots on target over/under "
        "line for each fixture."
    ),
    "btts": "For all {league} games on {date}, which fixtures are best for BTTS Yes?",
    "moneyline": (
        "For every {league} game on {date}, who is most likely to win? Show moneyline "
        "probabilities and value picks."
    ),
    "spreads": "For every {league} game on {date}, recommend a handicap line for each fixture.",
    "correctscore": "For every {league} game on {date}, predict the most likely correct score.",
    "intervals": (
        "For every {league} game on {date}, show me the goal interval probabilities "
        "for each fixture."
    ),
    "teamlines": (
        "For all {league} games on {date}, what are the per-team corner and card "
        "lines for each fixture?"
    ),
}

# -- /market embed title labels --
_MARKET_LABELS: Dict[str, str] = {
    "totals": "Goals Totals",
    "corners": "Corner Totals",
    "cards": "Card Totals",
    "sot": "SoT Totals",
    "btts": "BTTS",
    "moneyline": "Moneyline",
    "spreads": "Handicap Lines",
    "correctscore": "Correct Score",
    "intervals": "Goal Intervals",
    "teamlines": "Team Lines",
}

# -- /compare stat label mapping --
_STAT_LABELS: Dict[str, str] = {
    "corners": "corners",
    "cards": "cards",
    "sot": "shots on target",
    "goals": "goals",
    "possession": "possession",
}

# -- /players prompt templates --
_PLAYER_PROMPTS_SINGLE: Dict[str, str] = {
    "goalscorer": (
        "Who is most likely to score anytime in {home} vs {away} ({league}) on {date}?"
    ),
    "sot": (
        "Which players are best for shots on target props in {home} vs {away} "
        "({league}) on {date}?"
    ),
    "cards": (
        "Which players are most likely to be booked in {home} vs {away} ({league}) "
        "on {date}?"
    ),
    "assists": (
        "Who is most likely to get an assist in {home} vs {away} ({league}) on {date}?"
    ),
}

_PLAYER_PROMPTS_ALL: Dict[str, str] = {
    "goalscorer": (
        "For all {league} games on {date}, who are the best anytime goalscorer picks?"
    ),
    "sot": (
        "For all {league} games on {date}, which players are best for shots on target props?"
    ),
    "cards": (
        "For all {league} games on {date}, which players are most likely to be booked?"
    ),
    "assists": (
        "For all {league} games on {date}, who are the best assist picks?"
    ),
}

# -- /players prop labels for embed titles --
_PROP_LABELS: Dict[str, str] = {
    "goalscorer": "Anytime Goalscorer",
    "sot": "Shots on Target Props",
    "cards": "Player Cards",
    "assists": "Assist Props",
}


# ============================================================================
# Helper: parse home/away from match string
# ============================================================================

def _parse_match(match: str) -> Tuple[str, str]:
    """Split 'Home vs Away' into (home, away). Falls back gracefully."""
    parts = match.split(" vs ")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return match.strip(), "Unknown"


# ============================================================================
# Market choices — reused by /analyze and /market
# ============================================================================

_ANALYZE_MARKET_CHOICES = [
    app_commands.Choice(name="Overview (all markets)", value="overview"),
    app_commands.Choice(name="Goals O/U", value="totals"),
    app_commands.Choice(name="Corners O/U", value="corners"),
    app_commands.Choice(name="Cards O/U", value="cards"),
    app_commands.Choice(name="SoT O/U", value="sot"),
    app_commands.Choice(name="BTTS", value="btts"),
    app_commands.Choice(name="Moneyline (1X2)", value="moneyline"),
    app_commands.Choice(name="Handicap/Spread", value="spreads"),
    app_commands.Choice(name="Correct Score", value="correctscore"),
    app_commands.Choice(name="Goal Intervals", value="intervals"),
    app_commands.Choice(name="Team Lines", value="teamlines"),
]

_MARKET_MARKET_CHOICES = [
    app_commands.Choice(name="Goals O/U", value="totals"),
    app_commands.Choice(name="Corners O/U", value="corners"),
    app_commands.Choice(name="Cards O/U", value="cards"),
    app_commands.Choice(name="SoT O/U", value="sot"),
    app_commands.Choice(name="BTTS", value="btts"),
    app_commands.Choice(name="Moneyline (1X2)", value="moneyline"),
    app_commands.Choice(name="Handicap/Spread", value="spreads"),
    app_commands.Choice(name="Correct Score", value="correctscore"),
    app_commands.Choice(name="Goal Intervals", value="intervals"),
    app_commands.Choice(name="Team Lines", value="teamlines"),
]

_STAT_CHOICES = [
    app_commands.Choice(name="Corners", value="corners"),
    app_commands.Choice(name="Cards", value="cards"),
    app_commands.Choice(name="Shots on Target", value="sot"),
    app_commands.Choice(name="Goals", value="goals"),
    app_commands.Choice(name="Possession & Control", value="possession"),
]

_PROP_CHOICES = [
    app_commands.Choice(name="Anytime Goalscorer", value="goalscorer"),
    app_commands.Choice(name="Shots on Target", value="sot"),
    app_commands.Choice(name="Cards (to be booked)", value="cards"),
    app_commands.Choice(name="Assists", value="assists"),
]

_HELP_COMMAND_CHOICES = [
    app_commands.Choice(name="fixtures", value="fixtures"),
    app_commands.Choice(name="analyze", value="analyze"),
    app_commands.Choice(name="market", value="market"),
    app_commands.Choice(name="compare", value="compare"),
    app_commands.Choice(name="parlay", value="parlay"),
    app_commands.Choice(name="build", value="build"),
    app_commands.Choice(name="players", value="players"),
    app_commands.Choice(name="teamlines", value="teamlines"),
]


# ============================================================================
# Cog
# ============================================================================

class SlashCommands(commands.Cog):
    """Slash commands for quick RAG lookups."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    async def cog_app_command_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError,
    ) -> None:
        """Handle errors from app commands — especially CheckFailure from premium_only."""
        if isinstance(error, app_commands.CheckFailure):
            # premium_only already sent a response; if it didn't, send one now
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "You don't have permission to use this command. "
                    "Check the upgrade channel for details.",
                    ephemeral=True,
                )
            return
        # Log unexpected errors
        import logging
        logging.getLogger("slash_cmds").error("Command error: %s", error, exc_info=error)

    # -----------------------------------------------------------------------
    # /fixtures — Schedule browser
    # -----------------------------------------------------------------------
    @app_commands.command(
        name="fixtures",
        description="See upcoming fixtures and available markets",
    )
    @app_commands.describe(
        league="League to check",
        date="Day to look at (today, tomorrow, saturday, or YYYY-MM-DD)",
    )
    @app_commands.choices(league=_league_choices())
    async def fixtures(
        self,
        interaction: discord.Interaction,
        league: app_commands.Choice[str],
        date: Optional[str] = None,
    ) -> None:
        await interaction.response.defer(thinking=True)
        target = _parse_date(date)
        # Keep the full day in hand so an already-started fixture is not
        # misreported as a day with no football.  This remains a pre-match
        # command, so only future kickoffs are listed as actionable fixtures.
        all_day_events = _get_events_cached(league.value, target, upcoming_only=False)
        events = _filter_upcoming(all_day_events)
        phrase = _date_phrase(target)

        if not events:
            if all_day_events:
                em = discord.Embed(
                    title="No Upcoming Fixtures",
                    description=(
                        f"Found **{len(all_day_events)}** **{league.value}** fixture"
                        f"{'s' if len(all_day_events) != 1 else ''} for **{phrase}**, "
                        "but it has already started or finished. "
                        "Use the live bot for in-play coverage."
                    ),
                    color=COLOR_BLUE,
                )
                await interaction.followup.send(embed=em)
                return
            em = discord.Embed(
                title="No Fixtures Found",
                description=f"No **{league.value}** fixtures found for **{phrase}**.",
                color=COLOR_BLUE,
            )
            await interaction.followup.send(embed=em)
            return

        lines: List[str] = []
        for ev in events:
            home = ev.get("home_team", "?")
            away = ev.get("away_team", "?")
            commence = ev.get("commence_time", "")
            kickoff = ""
            if commence:
                try:
                    dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    kickoff = dt.strftime("%H:%M UTC")
                except (ValueError, TypeError):
                    kickoff = ""
            time_str = f" - {kickoff}" if kickoff else ""
            lines.append(f"**{home}** vs **{away}**{time_str}")

        em = discord.Embed(
            title=f"Fixtures  |  {league.value}  |  {phrase}",
            description="\n".join(lines),
            color=COLOR_BLUE,
        )
        em.set_footer(text=f"{len(events)} fixture{'s' if len(events) != 1 else ''}")
        await interaction.followup.send(embed=em)

    # -----------------------------------------------------------------------
    # /analyze — Single-fixture deep dive
    # -----------------------------------------------------------------------
    @app_commands.command(
        name="analyze",
        description="Full analysis for a single match",
    )
    @app_commands.describe(
        league="League",
        match="Fixture to analyze (type to search)",
        market="Focus on a specific market (default: overview)",
        date="Day (today, tomorrow, saturday, or YYYY-MM-DD)",
    )
    @app_commands.choices(league=_league_choices())
    @app_commands.choices(market=_ANALYZE_MARKET_CHOICES)
    @tier_required("elite")
    async def analyze(
        self,
        interaction: discord.Interaction,
        league: app_commands.Choice[str],
        match: str,
        market: Optional[app_commands.Choice[str]] = None,
        date: Optional[str] = None,
    ) -> None:
        thread = await get_or_create_private_thread(interaction)
        await interaction.response.defer(thinking=True)
        target = _parse_date(date)
        home, away = _parse_match(match)

        # Run the full analysis pipeline: all projections → GPT narrative
        report = await _run_full_analysis(home, away, league.value, target)

        # Build embed
        em = discord.Embed(
            title=f"Match Analysis: {home} vs {away}",
            description=report[:4096],
            color=COLOR_GREEN,
        )
        em.set_footer(text=f"{league.value} | Spix's Picks")

        if thread:
            await interaction.followup.send(
                f"Analysis posted in your private thread {thread.mention}.",
                ephemeral=True,
            )
            await thread.send(embed=em)
            # If report exceeds embed limit, send overflow as text
            if len(report) > 4096:
                await thread.send(report[4096:])
        else:
            await interaction.followup.send(embed=em, ephemeral=True)

    @analyze.autocomplete("match")
    async def _analyze_match_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        return await _match_autocomplete(interaction, current)

    # -----------------------------------------------------------------------
    # /market — League-wide market advice
    # -----------------------------------------------------------------------
    @app_commands.command(
        name="market",
        description="Market advice for all fixtures in a league",
    )
    @app_commands.describe(
        league="League",
        market="Market type",
        date="Day (today, tomorrow, saturday, or YYYY-MM-DD)",
    )
    @app_commands.choices(league=_league_choices())
    @app_commands.choices(market=_MARKET_MARKET_CHOICES)
    @pooled_command("free")
    async def market(
        self,
        interaction: discord.Interaction,
        league: app_commands.Choice[str],
        market: app_commands.Choice[str],
        date: Optional[str] = None,
    ) -> None:
        thread = await get_or_create_private_thread(interaction)
        await interaction.response.defer(thinking=True)
        target = _parse_date(date)
        label = _MARKET_LABELS.get(market.value, market.value.title())
        if canonical_market_name(market.value) is not None:
            results, notes = await _evaluate_market_results(league.value, market.value, target)
            if results:
                embeds, files = market_results_to_embeds(
                    label, results, league=league.value,
                    footer=f"{league.value} │ Canonical market pipeline",
                )
            else:
                note_text = "\n".join(notes[:3]) if notes else "No fixtures or market data found."
                embeds, files = rag_output_to_embeds(label, note_text, league=league.value)
        else:
            # These specialist markets do not yet have a MarketResult contract.
            text = await _render_market_answer(league.value, market.value, target)
            embeds, files = rag_output_to_embeds(label, text, league=league.value)
        if thread:
            await interaction.followup.send(
                f"Results posted in your private thread {thread.mention}.",
                ephemeral=True,
            )
            await thread.send(embeds=embeds, files=files)
        else:
            await interaction.followup.send(embeds=embeds, files=files, ephemeral=True)

    # -----------------------------------------------------------------------
    # /compare — Head-to-head stat comparison
    # -----------------------------------------------------------------------
    @app_commands.command(
        name="compare",
        description="Head-to-head stat comparison for a fixture",
    )
    @app_commands.describe(
        league="League",
        match="Fixture to compare (type to search)",
        stat="Stat to compare",
        date="Day (today, tomorrow, saturday, or YYYY-MM-DD)",
    )
    @app_commands.choices(league=_league_choices())
    @app_commands.choices(stat=_STAT_CHOICES)
    @pooled_command("starter")
    async def compare(
        self,
        interaction: discord.Interaction,
        league: app_commands.Choice[str],
        match: str,
        stat: app_commands.Choice[str],
        date: Optional[str] = None,
    ) -> None:
        thread = await get_or_create_private_thread(interaction)
        await interaction.response.defer(thinking=True)
        target = _parse_date(date)
        phrase = _date_phrase(target)
        home, away = _parse_match(match)

        stat_label = _STAT_LABELS.get(stat.value, stat.value)
        prompt = (
            f"For {home} vs {away} in {league.value} on {phrase}, which team will "
            f"get more {stat_label}? Compare both sides."
        )
        text = await _run_rag(prompt, league.value)

        embeds, files = rag_output_to_embeds(
            f"Comparison: {stat_label.title()}", text, league=league.value,
        )
        if thread:
            await interaction.followup.send(
                f"Results posted in your private thread {thread.mention}.",
                ephemeral=True,
            )
            await thread.send(embeds=embeds, files=files)
        else:
            await interaction.followup.send(embeds=embeds, files=files, ephemeral=True)

    @compare.autocomplete("match")
    async def _compare_match_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        return await _match_autocomplete(interaction, current)

    # -----------------------------------------------------------------------
    # /players — Player prop recommendations
    # -----------------------------------------------------------------------
    @app_commands.command(
        name="players",
        description="Player prop recommendations",
    )
    @app_commands.describe(
        league="League",
        prop="Type of player prop",
        match="Focus on a specific fixture (type to search)",
        date="Day (today, tomorrow, saturday, or YYYY-MM-DD)",
    )
    @app_commands.choices(league=_league_choices())
    @app_commands.choices(prop=_PROP_CHOICES)
    @pooled_command("starter")
    async def players(
        self,
        interaction: discord.Interaction,
        league: app_commands.Choice[str],
        prop: app_commands.Choice[str],
        match: Optional[str] = None,
        date: Optional[str] = None,
    ) -> None:
        thread = await get_or_create_private_thread(interaction)
        await interaction.response.defer(thinking=True)
        target = _parse_date(date)
        phrase = _date_phrase(target)

        if match:
            home, away = _parse_match(match)
            template = _PLAYER_PROMPTS_SINGLE.get(prop.value, "")
            prompt = template.format(
                home=home, away=away, league=league.value, date=phrase,
            )
        else:
            template = _PLAYER_PROMPTS_ALL.get(prop.value, "")
            prompt = template.format(league=league.value, date=phrase)

        text = await _run_rag(prompt, league.value)

        label = _PROP_LABELS.get(prop.value, "Player Props")
        embeds, files = rag_output_to_embeds(label, text, league=league.value)
        if thread:
            await interaction.followup.send(
                f"Results posted in your private thread {thread.mention}.",
                ephemeral=True,
            )
            await thread.send(embeds=embeds, files=files)
        else:
            await interaction.followup.send(embeds=embeds, files=files, ephemeral=True)

    @players.autocomplete("match")
    async def _players_match_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        return await _match_autocomplete(interaction, current)

    # /parlay lives in parlay_builder.py (with follow-up buttons, cross-league, etc.)

    # -----------------------------------------------------------------------
    # /teamlines — Standalone per-team corner & card lines
    # -----------------------------------------------------------------------
    @app_commands.command(
        name="teamlines",
        description="Per-team corner and card lines (e.g. Tottenham Over 2.5 Corners)",
    )
    @app_commands.describe(
        league="League",
        match="Focus on a specific fixture (type to search)",
        date="Day (today, tomorrow, saturday, or YYYY-MM-DD)",
    )
    @app_commands.choices(league=_league_choices())
    @pooled_command("starter")
    async def teamlines(
        self,
        interaction: discord.Interaction,
        league: app_commands.Choice[str],
        match: Optional[str] = None,
        date: Optional[str] = None,
    ) -> None:
        thread = await get_or_create_private_thread(interaction)
        await interaction.response.defer(thinking=True)
        target = _parse_date(date)
        phrase = _date_phrase(target)
        lg = league.value

        if match:
            home, away = _parse_match(match)
        else:
            home, away = None, None

        text = await _render_market_answer(lg, "teamlines", target)
        if home and away:
            marker = f"=== {home} vs {away} ==="
            if marker in text:
                before, after = text.split(marker, 1)
                next_block = after.find("\n===")
                block = after[:next_block] if next_block >= 0 else after
                header = before.splitlines()[0] if before.splitlines() else "Per-Team Totals Lines by fixture:"
                text = f"{header}\n\n{marker}{block}".strip()
            else:
                text = (
                    f"No exact team-lines block found for {home} vs {away} in {lg} on {phrase}.\n\n"
                    f"{text}"
                )
        embeds, files = rag_output_to_embeds("Team Lines", text, league=lg)
        if thread:
            await interaction.followup.send(
                f"Results posted in your private thread {thread.mention}.",
                ephemeral=True,
            )
            await thread.send(embeds=embeds, files=files)
        else:
            await interaction.followup.send(embeds=embeds, files=files, ephemeral=True)

    @teamlines.autocomplete("match")
    async def _teamlines_match_ac(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        return await _match_autocomplete(interaction, current)

    # -----------------------------------------------------------------------
    # /help — Guided onboarding
    # -----------------------------------------------------------------------
    @app_commands.command(
        name="help",
        description="Learn what this bot can do",
    )
    @app_commands.describe(command="Get help for a specific command")
    @app_commands.choices(command=_HELP_COMMAND_CHOICES)
    async def help_cmd(
        self,
        interaction: discord.Interaction,
        command: Optional[app_commands.Choice[str]] = None,
    ) -> None:
        if command is None:
            em = discord.Embed(
                title="Betting RAG  --  Football Predictions",
                description=(
                    "AI-powered match predictions backed by stats from 8 leagues."
                ),
                color=COLOR_BLUE,
            )
            em.add_field(
                name="Quick Start",
                value=(
                    "`/fixtures EPL` -- see today's games\n"
                    "`/analyze EPL \"Arsenal vs Chelsea\"` -- deep dive on one match\n"
                    "`/market EPL \"Goals O/U\"` -- advice for all games\n"
                    "`/parlay EPL` -- auto-build a parlay"
                ),
                inline=False,
            )
            em.add_field(
                name="All Commands",
                value=(
                    "`/fixtures` -- Browse upcoming fixtures\n"
                    "`/analyze` -- Full analysis for a single match\n"
                    "`/market` -- League-wide market advice\n"
                    "`/compare` -- Head-to-head stat comparison\n"
                    "`/parlay` -- Generate an optimized parlay\n"
                    "`/build` -- Step-by-step interactive parlay builder\n"
                    "`/players` -- Player prop recommendations"
                ),
                inline=False,
            )
            em.add_field(
                name="Tips",
                value=(
                    "Every command accepts `date:` (today, tomorrow, saturday, YYYY-MM-DD)\n"
                    "Use `match:` to focus on a specific fixture (autocomplete will find it)\n"
                    "Parlay results have follow-up buttons to adjust"
                ),
                inline=False,
            )
            await interaction.response.send_message(embed=em, ephemeral=True)
            return

        # Per-command help
        help_pages: Dict[str, Tuple[str, str]] = {
            "fixtures": (
                "/fixtures -- Browse Fixtures",
                "See all upcoming fixtures for a league on a given date.\n\n"
                "**Parameters:**\n"
                "`league` -- League to check (required)\n"
                "`date` -- Day to look at (optional, default: today)\n\n"
                "**Examples:**\n"
                "`/fixtures EPL` -- today's EPL games\n"
                "`/fixtures LaLiga saturday` -- Saturday's La Liga games\n"
                "`/fixtures Bundesliga 2026-03-20` -- specific date",
            ),
            "analyze": (
                "/analyze -- Single Match Analysis",
                "Full deep dive on one fixture. Optionally focus on a specific market.\n\n"
                "**Parameters:**\n"
                "`league` -- League (required)\n"
                "`match` -- Fixture to analyze, autocompletes (required)\n"
                "`market` -- Focus market: Overview, Goals O/U, Corners, Cards, SoT, "
                "BTTS, Moneyline, Spreads, Correct Score, Intervals, Team Lines (optional)\n"
                "`date` -- Day (optional, default: today)\n\n"
                "**Examples:**\n"
                "`/analyze EPL \"Arsenal vs Chelsea\"` -- full overview\n"
                "`/analyze EPL \"Arsenal vs Chelsea\" \"BTTS\"` -- BTTS focus\n"
                "`/analyze LaLiga \"Barcelona vs Real Madrid\" \"Corners O/U\" saturday`",
            ),
            "market": (
                "/market -- League-Wide Market Advice",
                "Get recommendations for every fixture in a league for a given market.\n\n"
                "**Parameters:**\n"
                "`league` -- League (required)\n"
                "`market` -- Market type: Goals O/U, Corners, Cards, SoT, BTTS, "
                "Moneyline, Spreads, Correct Score, Intervals, Team Lines (required)\n"
                "`date` -- Day (optional, default: today)\n\n"
                "**Examples:**\n"
                "`/market EPL \"Goals O/U\"` -- goals totals for all EPL games today\n"
                "`/market SerieA \"BTTS\" tomorrow` -- BTTS for tomorrow's Serie A\n"
                "`/market Ligue1 \"Moneyline\" saturday`",
            ),
            "compare": (
                "/compare -- Head-to-Head Comparison",
                "Compare two teams on a specific stat for a fixture.\n\n"
                "**Parameters:**\n"
                "`league` -- League (required)\n"
                "`match` -- Fixture to compare, autocompletes (required)\n"
                "`stat` -- Stat: Corners, Cards, SoT, Goals, Possession (required)\n"
                "`date` -- Day (optional, default: today)\n\n"
                "**Examples:**\n"
                "`/compare EPL \"Arsenal vs Chelsea\" Corners`\n"
                "`/compare LaLiga \"Barcelona vs Real Madrid\" Cards saturday`",
            ),
            "parlay": (
                "/parlay -- Auto-Generate a Parlay",
                "Let the model build an optimized multi-leg parlay.\n\n"
                "**Parameters:**\n"
                "`league` -- League (required)\n"
                "`legs` -- Number of legs. If omitted with odds, the builder can choose any count (optional)\n"
                "`odds` -- Target combined odds. Defaults to 4.0 when omitted (optional)\n"
                "`date` -- Day (optional, default: today)\n\n"
                "**Examples:**\n"
                "`/parlay EPL` -- 4-leg parlay at ~4x odds\n"
                "`/parlay EPL odds:1.8` -- best bet/parlay near 1.8x, any leg count\n"
                "`/parlay LaLiga 3 3.0` -- 3 legs targeting 3x\n"
                "`/parlay Bundesliga 5 6.0 saturday`",
            ),
            "build": (
                "/build -- Interactive Parlay Builder",
                "Step-by-step parlay construction with buttons and menus.\n\n"
                "Select a league, browse fixtures, tap to add legs, "
                "see running odds, and finalize your slip.\n\n"
                "**Usage:**\n"
                "`/build EPL` -- start building an EPL parlay\n"
                "`/build LaLiga saturday` -- start for Saturday's La Liga",
            ),
            "players": (
                "/players -- Player Props",
                "Get player prop recommendations for a league or specific fixture.\n\n"
                "**Parameters:**\n"
                "`league` -- League (required)\n"
                "`prop` -- Prop type: Anytime Goalscorer, SoT, Cards, Assists (required)\n"
                "`match` -- Focus on one fixture, autocompletes (optional)\n"
                "`date` -- Day (optional, default: today)\n\n"
                "**Examples:**\n"
                "`/players EPL \"Anytime Goalscorer\"` -- best scorers across all EPL games\n"
                "`/players EPL \"SoT\" \"Arsenal vs Chelsea\"` -- SoT props for one match\n"
                "`/players LaLiga \"Cards\" saturday`",
            ),
        }
        title, body = help_pages.get(
            command.value,
            (f"/{command.value}", "No detailed help available for this command yet."),
        )
        em = discord.Embed(title=title, description=body, color=COLOR_BLUE)
        await interaction.response.send_message(embed=em, ephemeral=True)


# ============================================================================
# Setup
# ============================================================================

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(SlashCommands(bot))
