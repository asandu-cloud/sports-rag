"""Promo parlay builder — admin-only single-match parlays for marketing content.

Picks the highest-confidence leg from each market group (handicap, totals, cards,
corners) for a chosen fixture.  Entirely standalone — does not touch any other
cog's state or architecture.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import discord
from discord import app_commands
from discord.ext import commands

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "rag_ingest"))

import rag_cli_v2 as rag  # noqa: E402
from core.parlay_service import (  # noqa: E402
    build_parlay_request,
    build_parlay,
    ParlayBuildError,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from embeds import _conf_bar, _conf_label, _get_league_logo, _truncate  # noqa: E402
from config import COLOR_BLUE  # noqa: E402

log = logging.getLogger("promo_parlay")

# ---------------------------------------------------------------------------
# Market groups ranked by priority (user wants: handicap, totals, cards, corners)
# ---------------------------------------------------------------------------

_PROMO_GROUPS = ["spreads", "totals", "cards", "corners"]

# ---------------------------------------------------------------------------
# Events cache (lightweight, standalone — no shared state)
# ---------------------------------------------------------------------------

_event_cache: Dict[str, Tuple[float, list]] = {}
_CACHE_TTL = 300


def _get_events(league: str, target: date) -> list:
    key = f"{league}|{target.isoformat()}"
    now = time.time()
    if key in _event_cache:
        ts, evts = _event_cache[key]
        if now - ts < _CACHE_TTL:
            return evts
    try:
        events, _ = rag.fetch_events(league, set(rag.DEFAULT_MARKETS), target_date=target)
        _event_cache[key] = (now, events)
        return events
    except Exception:
        return []


def _filter_upcoming(events: list) -> list:
    now = datetime.now(tz=__import__("datetime").timezone.utc)
    out = []
    for ev in events:
        ko = ev.get("commence_time", "")
        if not ko:
            out.append(ev)
            continue
        try:
            ko_dt = datetime.fromisoformat(str(ko).replace("Z", "+00:00"))
            if ko_dt.tzinfo is None:
                ko_dt = ko_dt.replace(tzinfo=__import__("datetime").timezone.utc)
            if ko_dt > now:
                out.append(ev)
        except Exception:
            out.append(ev)
    return out


# ---------------------------------------------------------------------------
# Date parsing (mirrors parlay_builder._parse_date)
# ---------------------------------------------------------------------------

def _parse_date(raw: Optional[str]) -> date:
    if not raw:
        return date.today()
    text = raw.strip().lower()
    today = date.today()
    if text in ("today", ""):
        return today
    if text == "tomorrow":
        return today + timedelta(days=1)
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    if text in weekdays:
        delta = (weekdays[text] - today.weekday()) % 7
        return today if delta == 0 else today + timedelta(days=delta)
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return today


# ---------------------------------------------------------------------------
# Admin role check
# ---------------------------------------------------------------------------

_ADMIN_ROLES = {"mod", "owner"}


def _is_admin(interaction: discord.Interaction) -> bool:
    if interaction.guild and interaction.user.id == interaction.guild.owner_id:
        return True
    member = interaction.user
    if isinstance(member, discord.Member):
        return any(r.name.lower() in _ADMIN_ROLES for r in member.roles)
    return False


# ---------------------------------------------------------------------------
# League choices
# ---------------------------------------------------------------------------

_ALL_LEAGUES = [
    "EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1",
    "UCL", "UEL", "UECL",
]


def _league_choices() -> list:
    return [app_commands.Choice(name=lg, value=lg) for lg in _ALL_LEAGUES]


def _promo_safe_summary(text: str) -> str:
    summary = str(text or "").strip()
    if not summary:
        return ""

    replacements = [
        ("Confidence mix:", "Confidence overview:"),
        ("Market mix stays concentrated around", "Game insights stay concentrated around"),
        ("model-versus-market pricing is positive on average", "model alignment remains positive on average"),
        ("model-versus-market pricing is negative on average", "model alignment remains mixed on average"),
        ("Combined odds landed at", "Combined analysis score landed at"),
        ("against a", "relative to a"),
        ("target.", "target range."),
        ("No standalone projection conflicts were flagged.", "No conflicting signals were flagged."),
        ("leg(s) conflict with standalone projections.", "signal(s) conflict with the broader model view."),
    ]
    for old, new in replacements:
        summary = summary.replace(old, new)
    return summary


def _promo_embeds(result, league: str, match: str) -> List[discord.Embed]:
    selected_legs = list(getattr(result, "selected_legs", []) or [])
    league_logo = _get_league_logo(league) if league else None

    if not selected_legs:
        em = discord.Embed(
            title=f"\U0001f4e3 Match Analysis \u2014 {match}",
            description="No predictions were available for this fixture.",
            color=COLOR_BLUE,
        )
        if league_logo:
            em.set_author(name=f"League: {league}", icon_url=league_logo)
        em.set_footer(text="Spix's Picks | Match analysis", icon_url=league_logo)
        return [em]

    em = discord.Embed(
        title=f"\U0001f4e3 Match Analysis \u2014 {match}",
        color=COLOR_BLUE,
    )
    if league_logo:
        em.set_author(name=f"League: {league}", icon_url=league_logo)

    for index, leg in enumerate(selected_legs, start=1):
        confidence = str(getattr(leg, "confidence", "") or "").lower()
        model_prob = getattr(leg, "model_prob", None)
        confidence_bits: List[str] = []
        if confidence:
            confidence_bits.append(f"{_conf_bar(confidence)} {_conf_label(confidence)}")
        if model_prob is not None:
            confidence_bits.append(f"{model_prob:.0%} model confidence")
        confidence_joined = " • ".join(confidence_bits)
        confidence_line = f"\n\u2728 {confidence_joined}" if confidence_joined else ""

        warning = getattr(leg, "warning", None)
        note_line = f"\n\U0001f4dd {warning}" if warning else ""

        signal_score = getattr(leg, "odds", None)
        signal_line = ""
        if signal_score is not None:
            signal_line = f"\n\U0001f4ca Analysis score: {float(signal_score):.2f}"

        em.add_field(
            name=f"Insight {index}",
            value=(
                f"**{getattr(leg, 'fixture', '')}**\n"
                f"\U0001f4cb Prediction: {getattr(leg, 'pick_display', '')}"
                f"{signal_line}{confidence_line}{note_line}"
            ),
            inline=False,
        )

    combined_score = getattr(result, "combined_odds", None)
    if combined_score is not None:
        em.add_field(
            name="Overview",
            value=f"**Combined analysis score: {float(combined_score):.2f}**",
            inline=False,
        )

    confidence_breakdown = str(getattr(result, "confidence_breakdown", "") or "").strip()
    if confidence_breakdown:
        em.add_field(name="\u2b50 Confidence", value=confidence_breakdown, inline=True)

    cross_warnings = list(getattr(result, "cross_warnings", []) or [])
    if cross_warnings:
        em.add_field(
            name="\U0001f4dd Context",
            value=_truncate("; ".join(cross_warnings), 1024),
            inline=True,
        )

    summary = _promo_safe_summary(getattr(result, "summary_text", "") or "")
    if summary:
        em.add_field(name="\U0001f4a1 Game Insights", value=_truncate(summary, 1024), inline=False)

    em.set_footer(text="Spix's Picks | Match analysis", icon_url=league_logo)
    return [em]


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class PromoParlayCog(commands.Cog, name="promo_parlay"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="promo",
        description="Build a ~2.5x same-game promo parlay for a fixture (admin only)",
    )
    @app_commands.describe(
        league="League of the fixture",
        match="Fixture (autocomplete)",
        date="Date (today / tomorrow / Saturday / YYYY-MM-DD)",
    )
    @app_commands.choices(league=_league_choices())
    async def promo(
        self,
        interaction: discord.Interaction,
        league: app_commands.Choice[str],
        match: str,
        date: Optional[str] = None,
    ):
        if not _is_admin(interaction):
            await interaction.response.send_message(
                "This command is restricted to admins.", ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True, ephemeral=True)

        target = _parse_date(date)
        lg = league.value
        groups = _PROMO_GROUPS[:3]

        request = build_parlay_request(
            league_value=lg,
            target_date=target,
            legs=3,
            odds=None,
            target_min=2.2,
            target_max=3.0,
            market="mixed",
            match=match,
            source_command="promo",
            target_mode="none",
            allowed_groups=groups,
            required_group_counts={g: 1 for g in groups},
            risk_profile="standard",
            discord_user_id=interaction.user.id,
            guild_id=interaction.guild.id if interaction.guild else None,
            channel_id=interaction.channel.id if interaction.channel else None,
        )

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, build_parlay, request)
        except ParlayBuildError as exc:
            notes = "\n".join(f"- {n}" for n in exc.notes) if exc.notes else ""
            await interaction.followup.send(
                f"Could not build promo parlay: **{exc.message}**\n{notes}",
                ephemeral=True,
            )
            return
        except Exception as exc:
            log.error("Promo parlay failed: %s", exc, exc_info=True)
            await interaction.followup.send(f"Error: {exc}", ephemeral=True)
            return

        embeds = _promo_embeds(result, league=lg, match=match)

        await interaction.followup.send(embeds=embeds, ephemeral=True)

    @promo.autocomplete("match")
    async def _promo_match_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        ns = interaction.namespace
        league = getattr(ns, "league", None)
        if hasattr(league, "value"):
            league = league.value
        if not league:
            return []
        target = _parse_date(getattr(ns, "date", None))
        events = _filter_upcoming(_get_events(league, target))
        choices: List[app_commands.Choice[str]] = []
        for ev in events:
            label = f"{ev['home_team']} vs {ev['away_team']}"
            if not current or current.lower() in label.lower():
                choices.append(app_commands.Choice(name=label[:100], value=label[:100]))
        return choices[:25]


# ---------------------------------------------------------------------------
# Cog loader
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(PromoParlayCog(bot))
