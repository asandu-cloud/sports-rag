"""Interactive parlay builder with date selection, cross-league support,
market focus, same-game parlays, and follow-up action buttons."""

from __future__ import annotations

import sys
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import discord
from discord import app_commands
from discord.ext import commands

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "rag_ingest"))
import rag_cli_v2 as rag  # noqa: E402
from core.parlay_service import (  # noqa: E402
    ParlayBuildError,
    ParlaySessionStore,
    build_and_store_parlay,
    build_parlay_request,
    derive_followup_request,
)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from embeds import parlay_error_embeds, structured_parlay_embeds  # noqa: E402
from config import ALL_LEAGUES, COLOR_PURPLE  # noqa: E402
from cogs.access import pooled_command, get_or_create_private_thread  # noqa: E402

_SESSION_STORE = ParlaySessionStore()


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    """Return ordinal string for an integer (1st, 2nd, 3rd, etc.)."""
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"


def _parse_date(raw: Optional[str]) -> date:
    """Parse a user-supplied date string into a date object.

    Accepts: ``None`` (today), ``"today"``, ``"tomorrow"``, weekday names
    (``"saturday"``), and ISO format (``"2026-03-15"``).  Weekday names
    resolve to the *next* occurrence (including today if it matches).
    """
    if raw is None:
        return date.today()

    text = raw.strip().lower()
    today = date.today()

    if text in ("today", ""):
        return today
    if text == "tomorrow":
        return today + timedelta(days=1)

    # Weekday name
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    if text in weekdays:
        target_wd = weekdays[text]
        delta = (target_wd - today.weekday()) % 7
        if delta == 0:
            # Same weekday = today
            return today
        return today + timedelta(days=delta)

    # ISO date (YYYY-MM-DD)
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        pass

    return today


def _date_phrase(d: date) -> str:
    """Human-friendly date phrase, e.g. '13th March'."""
    return f"{_ordinal(d.day)} {d.strftime('%B')}"


async def _send_parlay_result(
    interaction: discord.Interaction,
    result,
    thread: Optional[discord.Thread],
) -> None:
    """Send a structured parlay result to the thread or ephemeral fallback."""
    display_league = result.request.display_league
    embeds = structured_parlay_embeds(result, league=display_league)
    view = ParlayFollowUpView(
        session_id=result.session_id,
        league=display_league,
        date_phrase=result.request.date_phrase,
        thread_id=thread.id if thread else None,
    )
    if thread:
        await interaction.followup.send(
            f"Parlay posted in your private thread {thread.mention}.",
            ephemeral=True,
        )
        message = await thread.send(embeds=embeds, view=view)
        _SESSION_STORE.update_message_context(
            result.session_id,
            message_id=message.id,
            channel_id=getattr(message.channel, "id", None),
            thread_id=getattr(message.channel, "id", None),
        )
    else:
        await interaction.followup.send(embeds=embeds, view=view, ephemeral=True)


async def _send_parlay_error(
    interaction: discord.Interaction,
    message: str,
    notes: Optional[List[str]],
    *,
    league: str,
    thread: Optional[discord.Thread],
) -> None:
    embeds = parlay_error_embeds(message, notes or [], league=league)
    if thread:
        await interaction.followup.send(
            f"Could not build your parlay. Details were posted in {thread.mention}.",
            ephemeral=True,
        )
        await thread.send(embeds=embeds)
    else:
        await interaction.followup.send(embeds=embeds, ephemeral=True)


async def _build_structured(request):
    """Run the structured parlay builder off the event loop."""
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, build_and_store_parlay, request, _SESSION_STORE)


# ---------------------------------------------------------------------------
# Events cache for match autocomplete
# ---------------------------------------------------------------------------

_events_cache: Dict[str, Tuple[float, list]] = {}
_CACHE_TTL = 300  # 5 minutes


def _get_events_cached(league: str, target_date: Optional[date] = None) -> list:
    """Fetch events for a league on a specific date with a short TTL cache."""
    from datetime import date as _date
    from core.events import fetch_events as _fetch_events

    if target_date is None:
        target_date = _date.today()

    cache_key = f"{league}|{target_date.isoformat()}"
    now = time.time()
    if cache_key in _events_cache:
        ts, events = _events_cache[cache_key]
        if now - ts < _CACHE_TTL:
            return events

    try:
        events, _notes = _fetch_events(league, target_date=target_date)
        _events_cache[cache_key] = (now, events)
        return events
    except Exception as exc:
        import logging
        logging.getLogger("parlay_builder").warning("Event fetch failed for %s on %s: %s", league, target_date, exc)
        return []


def _filter_upcoming(events: list) -> list:
    """Remove events that have already kicked off or finished."""
    from datetime import datetime as _dt, timezone as _tz
    now = _dt.now(_tz.utc)
    upcoming = []
    for ev in events:
        ko = ev.get("commence_time", "")
        if not ko:
            upcoming.append(ev)
            continue
        try:
            ko_dt = _dt.fromisoformat(str(ko).replace("Z", "+00:00"))
            if ko_dt.tzinfo is None:
                ko_dt = ko_dt.replace(tzinfo=_tz.utc)
            if ko_dt > now:
                upcoming.append(ev)
        except Exception:
            upcoming.append(ev)
    return upcoming


async def _match_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> List[app_commands.Choice[str]]:
    """Autocomplete for match selection — only shows upcoming (not started) fixtures."""
    namespace = interaction.namespace
    league = getattr(namespace, "league", None)

    if hasattr(league, "value"):
        league = league.value
    if not league or league == "cross-league":
        return []

    date_str = getattr(namespace, "date", None)
    target = _parse_date(date_str)
    events = _get_events_cached(league, target_date=target)

    from core.events import filter_events_by_exact_date as _filter_date
    try:
        events = _filter_date(events, target)
    except Exception:
        pass

    # Only show upcoming matches
    events = _filter_upcoming(events)

    choices: List[app_commands.Choice[str]] = []
    query = current.lower()

    for ev in events:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        label = f"{home} vs {away}"
        if query and query not in label.lower():
            continue
        choices.append(app_commands.Choice(name=label[:100], value=label[:100]))
        if len(choices) >= 25:
            break

    return choices


# ---------------------------------------------------------------------------
# Follow-up buttons (attached to every parlay response)
# ---------------------------------------------------------------------------

class ParlayFollowUpView(discord.ui.View):
    """Follow-up action buttons attached to every parlay response."""

    def __init__(self, session_id: int, league: str, date_phrase: str, thread_id: Optional[int] = None):
        super().__init__(timeout=None)
        self.session_id = int(session_id)
        self.league = league
        self.date_phrase = date_phrase
        self.thread_id = thread_id
        self.add_leg.custom_id = f"parlay:add_leg:{self.session_id}"
        self.swap_weakest.custom_id = f"parlay:swap_weakest:{self.session_id}"
        self.safer.custom_id = f"parlay:safer:{self.session_id}"
        self.riskier.custom_id = f"parlay:riskier:{self.session_id}"

    async def _resolve_thread(self, interaction: discord.Interaction):
        if not self.thread_id:
            return None
        channel = interaction.client.get_channel(self.thread_id)
        if channel is not None:
            return channel
        try:
            return await interaction.client.fetch_channel(self.thread_id)
        except Exception:
            return None

    async def _publish_result(self, interaction: discord.Interaction, result) -> None:
        embeds = structured_parlay_embeds(result, league=self.league)
        view = ParlayFollowUpView(
            session_id=result.session_id,
            league=result.request.display_league,
            date_phrase=result.request.date_phrase,
            thread_id=result.request.thread_id,
        )
        thread = await self._resolve_thread(interaction)
        if thread is not None:
            await interaction.followup.send(
                f"Updated parlay posted in your private thread {thread.mention}.",
                ephemeral=True,
            )
            message = await thread.send(embeds=embeds, view=view)
            _SESSION_STORE.update_message_context(
                result.session_id,
                message_id=message.id,
                channel_id=getattr(message.channel, "id", None),
                thread_id=getattr(message.channel, "id", None),
            )
        else:
            await interaction.followup.send(embeds=embeds, view=view, ephemeral=True)

    async def _publish_error(self, interaction: discord.Interaction, message: str, notes: Optional[List[str]] = None) -> None:
        embeds = parlay_error_embeds(message, notes or [], league=self.league)
        thread = await self._resolve_thread(interaction)
        if thread is not None:
            await interaction.followup.send(
                f"Could not update your parlay. Details were posted in {thread.mention}.",
                ephemeral=True,
            )
            await thread.send(embeds=embeds)
        else:
            await interaction.followup.send(embeds=embeds, ephemeral=True)

    async def _handle_action(self, interaction: discord.Interaction, action: str) -> None:
        await interaction.response.defer(thinking=True)
        record = _SESSION_STORE.get_session(self.session_id)
        if record is None:
            await self._publish_error(interaction, "This parlay session is no longer available.")
            return

        try:
            request = derive_followup_request(record, action)
            request.discord_user_id = interaction.user.id
            request.guild_id = interaction.guild.id if interaction.guild else None
            request.channel_id = interaction.channel.id if interaction.channel else None
            request.thread_id = self.thread_id or getattr(interaction.channel, "id", None)
            result = await _build_structured(request)
        except ParlayBuildError as exc:
            await self._publish_error(interaction, exc.message, exc.notes)
            return
        except Exception as exc:
            await self._publish_error(interaction, f"Parlay update failed: {exc}")
            return

        await self._publish_result(interaction, result)

    @discord.ui.button(label="Add a leg", style=discord.ButtonStyle.green, emoji="\u2795")
    async def add_leg(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._handle_action(interaction, "add_leg")

    @discord.ui.button(label="Swap weakest", style=discord.ButtonStyle.blurple, emoji="\U0001f504")
    async def swap_weakest(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._handle_action(interaction, "swap_weakest")

    @discord.ui.button(label="Safer", style=discord.ButtonStyle.grey, emoji="\U0001f6e1\ufe0f")
    async def safer(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._handle_action(interaction, "safer")

    @discord.ui.button(label="Riskier", style=discord.ButtonStyle.red, emoji="\U0001f3b2")
    async def riskier(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._handle_action(interaction, "riskier")


# ---------------------------------------------------------------------------
# /build — Interactive step-by-step builder
# ---------------------------------------------------------------------------

# Step 1: Date selection

class DateButton(discord.ui.Button):
    """A single date-picker button."""

    def __init__(self, label: str, target_date: date,
                 thread: Optional[discord.Thread] = None):
        super().__init__(label=label, style=discord.ButtonStyle.blurple)
        self.target_date = target_date
        self.thread = thread

    async def callback(self, interaction: discord.Interaction):
        em = discord.Embed(
            title=f"Build Parlay \u2014 {_date_phrase(self.target_date)}",
            description="Step 2: Pick a league",
            color=COLOR_PURPLE,
        )
        view = BuildLeagueSelectView(self.target_date, self.thread)
        await interaction.response.edit_message(embed=em, view=view)


class DateSelectView(discord.ui.View):
    """Step 1 view: choose a date."""

    def __init__(self, thread: Optional[discord.Thread] = None):
        super().__init__(timeout=120)
        today = date.today()
        dates = [
            ("Today", today),
            ("Tomorrow", today + timedelta(days=1)),
        ]
        # Next 3 calendar days after tomorrow
        for i in range(2, 5):
            d = today + timedelta(days=i)
            dates.append((d.strftime("%A"), d))

        for label, d in dates:
            self.add_item(DateButton(label=label, target_date=d, thread=thread))


# Step 2: League selection

class BuildLeagueSelect(discord.ui.Select):
    """Dropdown to pick a league (step 2 of /build)."""

    def __init__(self, target_date: date, thread: Optional[discord.Thread] = None):
        self.target_date = target_date
        self.thread = thread
        options = [discord.SelectOption(label=lg, value=lg) for lg in ALL_LEAGUES]
        super().__init__(placeholder="Pick a league...", options=options)

    async def callback(self, interaction: discord.Interaction):
        league = self.values[0]
        em = discord.Embed(
            title=f"Build Parlay \u2014 {league} \u2014 {_date_phrase(self.target_date)}",
            description="Step 3: Pick a market type",
            color=COLOR_PURPLE,
        )
        view = BuildMarketSelectView(league, self.target_date, self.thread)
        await interaction.response.edit_message(embed=em, view=view)


class BuildLeagueSelectView(discord.ui.View):
    def __init__(self, target_date: date, thread: Optional[discord.Thread] = None):
        super().__init__(timeout=120)
        self.add_item(BuildLeagueSelect(target_date, thread))


# Step 3: Market selection

class BuildMarketSelect(discord.ui.Select):
    """Dropdown to pick a market type (step 3 of /build)."""

    def __init__(self, league: str, target_date: date,
                 thread: Optional[discord.Thread] = None):
        self.league = league
        self.target_date = target_date
        self.thread = thread
        options = [
            discord.SelectOption(label="Goals Totals", value="totals", emoji="\u26bd"),
            discord.SelectOption(label="Corners", value="corners", emoji="\U0001f4d0"),
            discord.SelectOption(label="Cards", value="cards", emoji="\U0001f7e8"),
            discord.SelectOption(label="SoT", value="sot", emoji="\U0001f3af"),
            discord.SelectOption(label="BTTS", value="btts", emoji="\U0001f91d"),
            discord.SelectOption(label="Moneyline", value="moneyline", emoji="\U0001f3c6"),
            discord.SelectOption(label="Mixed Markets", value="mixed", emoji="\U0001f3b2"),
        ]
        super().__init__(placeholder="Pick a market type...", options=options)

    async def callback(self, interaction: discord.Interaction):
        market = self.values[0]
        em = discord.Embed(
            title=f"Build Parlay \u2014 {self.league} \u2014 {market.title()}",
            description="Step 4: Pick your risk level",
            color=COLOR_PURPLE,
        )
        view = BuildOddsTargetView(self.league, self.target_date, market, self.thread)
        await interaction.response.edit_message(embed=em, view=view)


class BuildMarketSelectView(discord.ui.View):
    def __init__(self, league: str, target_date: date,
                 thread: Optional[discord.Thread] = None):
        super().__init__(timeout=120)
        self.add_item(BuildMarketSelect(league, target_date, thread))


# Step 4: Risk level / execute

class BuildOddsTargetView(discord.ui.View):
    """Step 4 view: pick a risk level and execute the parlay query."""

    def __init__(self, league: str, target_date: date, market: str,
                 thread: Optional[discord.Thread] = None):
        super().__init__(timeout=120)
        self.league = league
        self.target_date = target_date
        self.market = market
        self.thread = thread

    async def _build(self, interaction: discord.Interaction, legs: int, odds: float, risk_profile: str):
        await interaction.response.defer(thinking=True)
        request = build_parlay_request(
            league_value=self.league,
            target_date=self.target_date,
            legs=legs,
            odds=odds,
            market=self.market,
            match=None,
            source_command="build",
            target_mode="max",
            risk_profile=risk_profile,
            discord_user_id=interaction.user.id,
            guild_id=interaction.guild.id if interaction.guild else None,
            channel_id=interaction.channel.id if interaction.channel else None,
            thread_id=self.thread.id if self.thread else None,
        )
        try:
            result = await _build_structured(request)
        except ParlayBuildError as exc:
            await _send_parlay_error(
                interaction,
                exc.message,
                exc.notes,
                league=self.league,
                thread=self.thread,
            )
            return
        except Exception as exc:
            await _send_parlay_error(
                interaction,
                f"Parlay build failed: {exc}",
                [],
                league=self.league,
                thread=self.thread,
            )
            return

        await _send_parlay_result(interaction, result, self.thread)

    @discord.ui.button(label="Safe (3-leg, ~2.5x)", style=discord.ButtonStyle.green)
    async def safe(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._build(interaction, 3, 2.5, "safe")

    @discord.ui.button(label="Standard (4-leg, ~4x)", style=discord.ButtonStyle.blurple)
    async def standard(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._build(interaction, 4, 4.0, "standard")

    @discord.ui.button(label="Risky (5-leg, ~7x)", style=discord.ButtonStyle.red)
    async def risky(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._build(interaction, 5, 7.0, "risky")


# ---------------------------------------------------------------------------
# /parlay — Enhanced parlay generator (replaces old /parlay + /quickparlay)
# ---------------------------------------------------------------------------

PARLAY_LEAGUE_CHOICES = [
    app_commands.Choice(name=lg, value=lg) for lg in ALL_LEAGUES
] + [app_commands.Choice(name="Cross-League (Top 5)", value="cross-league")]

PARLAY_MARKET_CHOICES = [
    app_commands.Choice(name="Mixed (best value)", value="mixed"),
    app_commands.Choice(name="Goals only", value="totals"),
    app_commands.Choice(name="Corners only", value="corners"),
    app_commands.Choice(name="Cards only", value="cards"),
    app_commands.Choice(name="SoT only", value="sot"),
]


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class ParlayBuilder(commands.Cog):
    """Interactive parlay building commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def cog_load(self):
        for record in _SESSION_STORE.list_persistent_sessions():
            if not record.message_id:
                continue
            view = ParlayFollowUpView(
                session_id=record.id,
                league=record.result.request.display_league,
                date_phrase=record.result.request.date_phrase,
                thread_id=record.thread_id,
            )
            self.bot.add_view(view, message_id=int(record.message_id))

    # --- /build ---

    @app_commands.command(name="build", description="Step-by-step interactive parlay builder")
    @pooled_command("pro")
    async def build(self, interaction: discord.Interaction):
        thread = await get_or_create_private_thread(interaction)
        em = discord.Embed(
            title="Build a Parlay",
            description="Step 1: Pick a date",
            color=COLOR_PURPLE,
        )
        view = DateSelectView(thread)
        if thread:
            await interaction.response.send_message(
                f"Building your parlay in {thread.mention}...",
                ephemeral=True,
            )
            await thread.send(embed=em, view=view)
        else:
            await interaction.response.send_message(embed=em, view=view, ephemeral=True)

    # --- /parlay ---

    @app_commands.command(name="parlay", description="Generate an optimized parlay")
    @app_commands.describe(
        league="League (or Cross-League for multi-league)",
        legs="Number of legs. Leave blank to let odds-only requests choose the best count.",
        odds="Target combined decimal odds. Defaults to 4.0 when omitted.",
        market="Focus on specific market type (optional)",
        match="Lock to a specific fixture (type to search)",
        date="Day (today, tomorrow, saturday, or YYYY-MM-DD)",
    )
    @app_commands.choices(league=PARLAY_LEAGUE_CHOICES)
    @app_commands.choices(market=PARLAY_MARKET_CHOICES)
    @pooled_command("pro")
    async def parlay_cmd(
        self,
        interaction: discord.Interaction,
        league: app_commands.Choice[str],
        legs: Optional[int] = None,
        odds: Optional[float] = None,
        market: Optional[app_commands.Choice[str]] = None,
        match: Optional[str] = None,
        date: Optional[str] = None,
    ):
        thread = await get_or_create_private_thread(interaction)
        await interaction.response.defer(thinking=True)

        target_date = _parse_date(date)
        lg = league.value
        market_val = market.value if market else None
        flexible_leg_count = legs is None and odds is not None
        target_odds = odds if odds is not None else 4.0
        target_mode = "around" if (flexible_leg_count or lg == "cross-league" or match) else "max"
        request = build_parlay_request(
            league_value=lg,
            target_date=target_date,
            legs=legs,
            odds=target_odds,
            market=market_val,
            match=match,
            source_command="parlay",
            target_mode=target_mode,
            flexible_leg_count=flexible_leg_count,
            risk_profile="standard",
            discord_user_id=interaction.user.id,
            guild_id=interaction.guild.id if interaction.guild else None,
            channel_id=interaction.channel.id if interaction.channel else None,
            thread_id=thread.id if thread else None,
        )
        try:
            result = await _build_structured(request)
        except ParlayBuildError as exc:
            await _send_parlay_error(
                interaction,
                exc.message,
                exc.notes,
                league="Cross-League" if lg == "cross-league" else lg,
                thread=thread,
            )
            return
        except Exception as exc:
            await _send_parlay_error(
                interaction,
                f"Parlay build failed: {exc}",
                [],
                league="Cross-League" if lg == "cross-league" else lg,
                thread=thread,
            )
            return

        await _send_parlay_result(interaction, result, thread)

    @parlay_cmd.autocomplete("match")
    async def _parlay_match_ac(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        return await _match_autocomplete(interaction, current)


async def setup(bot: commands.Bot):
    await bot.add_cog(ParlayBuilder(bot))
