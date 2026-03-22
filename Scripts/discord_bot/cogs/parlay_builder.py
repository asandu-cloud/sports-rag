"""Interactive parlay builder with date selection, cross-league support,
market focus, same-game parlays, and follow-up action buttons."""

from __future__ import annotations

import io
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import discord
from discord import app_commands
from discord.ext import commands

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "rag_ingest"))
import rag_cli_v2 as rag  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from embeds import parlay_embed  # noqa: E402
from config import ALL_LEAGUES, COLOR_PURPLE, COLOR_BLUE  # noqa: E402
from cogs.access import premium_only, get_or_create_private_thread, _has_premium_role  # noqa: E402


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


def _run_rag_sync(prompt: str, league: str) -> str:
    """Synchronous RAG call — never call from event loop directly."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rag.answer_once(prompt, default_league=league)
    return buf.getvalue().strip() or "(No output from model)"


async def _run_rag(prompt: str, league: str) -> str:
    """Async wrapper — runs RAG in thread pool so event loop stays free."""
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run_rag_sync, prompt, league)


# ---------------------------------------------------------------------------
# Events cache for match autocomplete
# ---------------------------------------------------------------------------

_events_cache: Dict[str, Tuple[float, list]] = {}
_CACHE_TTL = 300  # 5 minutes


def _get_events_cached(league: str) -> list:
    """Fetch today's events for a league with a short TTL cache."""
    now = time.time()
    if league in _events_cache:
        ts, events = _events_cache[league]
        if now - ts < _CACHE_TTL:
            return events

    try:
        events, _notes = rag.fetch_events(league, set(rag.DEFAULT_MARKETS))
        _events_cache[league] = (now, events)
        return events
    except Exception as exc:
        import logging
        logging.getLogger("parlay_builder").warning("Event fetch failed for %s: %s", league, exc)
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

    events = _get_events_cached(league)

    date_str = getattr(namespace, "date", None)
    target = _parse_date(date_str)
    try:
        events = rag.filter_events_by_exact_date(events, target)
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

    def __init__(self, league: str, date_phrase: str,
                 thread: Optional[discord.Thread] = None,
                 previous_parlay: str = ""):
        super().__init__(timeout=300)  # 5 minutes
        self.league = league
        self.date_phrase = date_phrase
        self.thread = thread
        self.previous_parlay = previous_parlay  # raw text of the parlay output

    async def _send_to_thread(
        self,
        interaction: discord.Interaction,
        embeds: list,
        view: "ParlayFollowUpView",
    ):
        """Send results to the user's private thread (or ephemeral fallback)."""
        if self.thread:
            await interaction.followup.send(
                f"Updated parlay posted in your private thread {self.thread.mention}.",
                ephemeral=True,
            )
            await self.thread.send(embeds=embeds, view=view)
        else:
            await interaction.followup.send(embeds=embeds, view=view, ephemeral=True)

    @discord.ui.button(label="Add a leg", style=discord.ButtonStyle.green, emoji="\u2795")
    async def add_leg(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)
        if self.previous_parlay:
            import re
            legs = re.findall(
                r"Leg\s+\d+:.*?\|.*?@\s*[\d.]+\s*\([^)]+\)(?:\s*\([^)]+\))?",
                self.previous_parlay,
            )
            n_legs = len(legs)
            legs_text = "\n".join(legs) if legs else self.previous_parlay[:500]
            prompt = (
                f"For {self.league} games on {self.date_phrase}, "
                f"I have this {n_legs}-leg parlay:\n{legs_text}\n\n"
                f"Add one more leg to make it a {n_legs + 1}-leg parlay. "
                "Pick a strong high-confidence leg from a different fixture than the existing legs."
            )
        else:
            prompt = (
                f"For {self.league} games on {self.date_phrase}, "
                "build a 5-leg parlay with high confidence picks."
            )
        text = await _run_rag(prompt, self.league)
        embeds = parlay_embed(text, league=self.league)
        view = ParlayFollowUpView(self.league, self.date_phrase, self.thread, previous_parlay=text)
        await self._send_to_thread(interaction, embeds, view)

    @discord.ui.button(label="Swap weakest", style=discord.ButtonStyle.blurple, emoji="\U0001f504")
    async def swap_weakest(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)
        # Include the previous parlay so the model knows which leg to swap
        if self.previous_parlay:
            # Extract legs from previous output to identify the weakest
            import re
            legs = re.findall(
                r"Leg\s+\d+:.*?\|.*?@\s*[\d.]+\s*\([^)]+\)(?:\s*\([^)]+\))?",
                self.previous_parlay,
            )
            legs_text = "\n".join(legs) if legs else self.previous_parlay[:500]
            prompt = (
                f"For {self.league} games on {self.date_phrase}, "
                f"I have this parlay:\n{legs_text}\n\n"
                "Identify the weakest leg (lowest confidence or smallest edge) "
                "and replace it with a stronger alternative from the same or different fixture. "
                "Keep the other legs unchanged and rebuild the parlay."
            )
        else:
            prompt = (
                f"For {self.league} games on {self.date_phrase}, "
                "build a parlay with 4 legs, replacing any weak picks with stronger alternatives."
            )
        text = await _run_rag(prompt, self.league)
        embeds = parlay_embed(text, league=self.league)
        view = ParlayFollowUpView(self.league, self.date_phrase, self.thread, previous_parlay=text)
        await self._send_to_thread(interaction, embeds, view)

    @discord.ui.button(label="Safer", style=discord.ButtonStyle.grey, emoji="\U0001f6e1\ufe0f")
    async def safer(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)
        if self.previous_parlay:
            import re
            legs = re.findall(
                r"Leg\s+\d+:.*?\|.*?@\s*[\d.]+\s*\([^)]+\)(?:\s*\([^)]+\))?",
                self.previous_parlay,
            )
            legs_text = "\n".join(legs) if legs else self.previous_parlay[:500]
            prompt = (
                f"For {self.league} games on {self.date_phrase}, "
                f"I have this parlay:\n{legs_text}\n\n"
                "Make it safer — keep the same fixtures but pick shorter-odds, "
                "higher-probability lines. Cap combined odds at 2.5x."
            )
        else:
            prompt = (
                f"For {self.league} games on {self.date_phrase}, "
                "build a safe parlay with combined odds around 2.5x using only high confidence picks."
            )
        text = await _run_rag(prompt, self.league)
        embeds = parlay_embed(text, league=self.league)
        view = ParlayFollowUpView(self.league, self.date_phrase, self.thread, previous_parlay=text)
        await self._send_to_thread(interaction, embeds, view)

    @discord.ui.button(label="Riskier", style=discord.ButtonStyle.red, emoji="\U0001f3b2")
    async def riskier(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)
        if self.previous_parlay:
            import re
            legs = re.findall(
                r"Leg\s+\d+:.*?\|.*?@\s*[\d.]+\s*\([^)]+\)(?:\s*\([^)]+\))?",
                self.previous_parlay,
            )
            legs_text = "\n".join(legs) if legs else self.previous_parlay[:500]
            prompt = (
                f"For {self.league} games on {self.date_phrase}, "
                f"I have this parlay:\n{legs_text}\n\n"
                "Make it riskier — keep the same fixtures but pick longer-odds, "
                "higher-upside lines. Push combined odds up to around 5.0x-7.0x."
            )
        else:
            prompt = (
                f"For {self.league} games on {self.date_phrase}, "
                "build a risky parlay with combined odds around 6.0x mixing different markets."
            )
        text = await _run_rag(prompt, self.league)
        embeds = parlay_embed(text, league=self.league)
        view = ParlayFollowUpView(self.league, self.date_phrase, self.thread, previous_parlay=text)
        await self._send_to_thread(interaction, embeds, view)


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

_MARKET_CONSTRAINTS = {
    "totals": "using only full-game over/under goals totals lines",
    "corners": "with ONLY CORNER TOTALS, one corner total per game, full-game markets only",
    "cards": "with ONLY CARD TOTALS, one card total per game, full-game markets only",
    "sot": "using ONLY shots on target totals, one SoT leg per game, full-game markets only",
    "btts": "using only BTTS Yes/No picks",
    "moneyline": "using only moneyline picks",
    "mixed": "mixing goals, corners, and cards markets, at least two different fixtures",
}


class BuildOddsTargetView(discord.ui.View):
    """Step 4 view: pick a risk level and execute the parlay query."""

    def __init__(self, league: str, target_date: date, market: str,
                 thread: Optional[discord.Thread] = None):
        super().__init__(timeout=120)
        self.league = league
        self.target_date = target_date
        self.market = market
        self.thread = thread

    async def _build(self, interaction: discord.Interaction, legs: int, odds: float):
        await interaction.response.defer(thinking=True)
        dp = _date_phrase(self.target_date)
        constraint = _MARKET_CONSTRAINTS.get(self.market, "")
        prompt = (
            f"For all {self.league} games on {dp}, make a {legs}-leg parlay "
            f"{constraint} with combined decimal odds at or below {odds}x."
        )
        text = await _run_rag(prompt, self.league)
        embeds = parlay_embed(text, league=self.league)
        view = ParlayFollowUpView(self.league, dp, self.thread, previous_parlay=text)
        if self.thread:
            await interaction.followup.send(
                f"Parlay posted in your private thread {self.thread.mention}.",
                ephemeral=True,
            )
            await self.thread.send(embeds=embeds, view=view)
        else:
            await interaction.followup.send(embeds=embeds, view=view, ephemeral=True)

    @discord.ui.button(label="Safe (3-leg, ~2.5x)", style=discord.ButtonStyle.green)
    async def safe(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._build(interaction, 3, 2.5)

    @discord.ui.button(label="Standard (4-leg, ~4x)", style=discord.ButtonStyle.blurple)
    async def standard(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._build(interaction, 4, 4.0)

    @discord.ui.button(label="Risky (5-leg, ~7x)", style=discord.ButtonStyle.red)
    async def risky(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._build(interaction, 5, 7.0)


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

    # --- /build ---

    @app_commands.command(name="build", description="Step-by-step interactive parlay builder")
    @premium_only()
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
        legs="Number of legs (default 4)",
        odds="Target combined decimal odds (default 4.0)",
        market="Focus on specific market type (optional)",
        match="Lock to a specific fixture (type to search)",
        date="Day (today, tomorrow, saturday, or YYYY-MM-DD)",
    )
    @app_commands.choices(league=PARLAY_LEAGUE_CHOICES)
    @app_commands.choices(market=PARLAY_MARKET_CHOICES)
    @premium_only()
    async def parlay_cmd(
        self,
        interaction: discord.Interaction,
        league: app_commands.Choice[str],
        legs: Optional[int] = 4,
        odds: Optional[float] = 4.0,
        market: Optional[app_commands.Choice[str]] = None,
        match: Optional[str] = None,
        date: Optional[str] = None,
    ):
        thread = await get_or_create_private_thread(interaction)
        await interaction.response.defer(thinking=True)

        target_date = _parse_date(date)
        dp = _date_phrase(target_date)
        lg = league.value
        market_val = market.value if market else None

        # Build the prompt based on context
        if lg == "cross-league":
            prompt = (
                f"For all top 5 league games on {dp}, build a cross-league parlay "
                f"with {legs} legs and combined odds around {odds}x."
            )
            if market_val and market_val != "mixed":
                constraint = _MARKET_CONSTRAINTS.get(market_val, "")
                prompt = (
                    f"For all top 5 league games on {dp}, build a cross-league parlay "
                    f"{constraint} with {legs} legs and combined odds around {odds}x."
                )
            display_league = "Cross-League"
            rag_league = "EPL"
        elif match:
            prompt = (
                f"Build a {legs}-leg same-game parlay for {match} in {lg} on {dp} "
                f"with combined odds around {odds}x."
            )
            if market_val and market_val != "mixed":
                constraint = _MARKET_CONSTRAINTS.get(market_val, "")
                prompt = (
                    f"Build a {legs}-leg same-game parlay for {match} in {lg} on {dp} "
                    f"{constraint} with combined odds around {odds}x."
                )
            display_league = lg
            rag_league = lg
        else:
            if market_val and market_val != "mixed":
                constraint = _MARKET_CONSTRAINTS.get(market_val, "")
                prompt = (
                    f"For all {lg} games on {dp}, make a {legs}-leg parlay "
                    f"{constraint} with combined decimal odds at or below {odds}x."
                )
            else:
                prompt = (
                    f"For all {lg} games on {dp}, make a {legs}-leg parlay "
                    f"with combined decimal odds at or below {odds}x."
                )
            display_league = lg
            rag_league = lg

        text = await _run_rag(prompt, rag_league)
        embeds = parlay_embed(text, league=display_league)
        view = ParlayFollowUpView(display_league, dp, thread, previous_parlay=text)
        if thread:
            await interaction.followup.send(
                f"Parlay posted in your private thread {thread.mention}.",
                ephemeral=True,
            )
            await thread.send(embeds=embeds, view=view)
        else:
            await interaction.followup.send(embeds=embeds, view=view, ephemeral=True)

    @parlay_cmd.autocomplete("match")
    async def _parlay_match_ac(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        return await _match_autocomplete(interaction, current)


async def setup(bot: commands.Bot):
    await bot.add_cog(ParlayBuilder(bot))
