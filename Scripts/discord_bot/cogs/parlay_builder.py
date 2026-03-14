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


def _run_rag(prompt: str, league: str) -> str:
    """Run a RAG query and capture the text output."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rag.answer_once(prompt, default_league=league)
    return buf.getvalue().strip() or "(No output from model)"


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
        # fetch_events takes league code ("EPL") and a set of markets,
        # returns (events_list, notes)
        events, _notes = rag.fetch_events(league, set(rag.DEFAULT_MARKETS))
        _events_cache[league] = (now, events)
        return events
    except Exception as exc:
        import logging
        logging.getLogger("parlay_builder").warning("Event fetch failed for %s: %s", league, exc)
        return []


async def _match_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> List[app_commands.Choice[str]]:
    """Autocomplete for match selection — fetches fixtures for the selected league."""
    namespace = interaction.namespace
    league = getattr(namespace, "league", None)

    # Handle Choice objects (Discord may pass the raw value or the Choice)
    if hasattr(league, "value"):
        league = league.value
    if not league or league == "cross-league":
        return []

    # Fetch events (cached 5 min)
    events = _get_events_cached(league)

    # Filter by date — defaults to today when no date specified
    date_str = getattr(namespace, "date", None)
    target = _parse_date(date_str)  # returns today when date_str is None
    try:
        events = rag.filter_events_by_exact_date(events, target)
    except Exception:
        pass

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

    def __init__(self, league: str, date_phrase: str):
        super().__init__(timeout=300)  # 5 minutes
        self.league = league
        self.date_phrase = date_phrase

    @discord.ui.button(label="Add a leg", style=discord.ButtonStyle.green, emoji="\u2795")
    async def add_leg(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)
        prompt = (
            f"For {self.league} games on {self.date_phrase}, "
            "add one more leg to the parlay above."
        )
        text = _run_rag(prompt, self.league)
        embeds = parlay_embed(text, league=self.league)
        view = ParlayFollowUpView(self.league, self.date_phrase)
        await interaction.followup.send(embeds=embeds, view=view)

    @discord.ui.button(label="Swap weakest", style=discord.ButtonStyle.blurple, emoji="\U0001f504")
    async def swap_weakest(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)
        prompt = (
            f"For {self.league} games on {self.date_phrase}, "
            "replace the weakest leg with a better alternative from the same fixture."
        )
        text = _run_rag(prompt, self.league)
        embeds = parlay_embed(text, league=self.league)
        view = ParlayFollowUpView(self.league, self.date_phrase)
        await interaction.followup.send(embeds=embeds, view=view)

    @discord.ui.button(label="Safer", style=discord.ButtonStyle.grey, emoji="\U0001f6e1\ufe0f")
    async def safer(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)
        prompt = (
            f"For {self.league} games on {self.date_phrase}, "
            "keep the same fixtures, but make the parlay safer and cap combined odds at 2.6x."
        )
        text = _run_rag(prompt, self.league)
        embeds = parlay_embed(text, league=self.league)
        view = ParlayFollowUpView(self.league, self.date_phrase)
        await interaction.followup.send(embeds=embeds, view=view)

    @discord.ui.button(label="Riskier", style=discord.ButtonStyle.red, emoji="\U0001f3b2")
    async def riskier(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)
        prompt = (
            f"For {self.league} games on {self.date_phrase}, "
            "same fixtures and market types, but push the combined odds up to around 5.0x."
        )
        text = _run_rag(prompt, self.league)
        embeds = parlay_embed(text, league=self.league)
        view = ParlayFollowUpView(self.league, self.date_phrase)
        await interaction.followup.send(embeds=embeds, view=view)


# ---------------------------------------------------------------------------
# /build — Interactive step-by-step builder
# ---------------------------------------------------------------------------

# Step 1: Date selection

class DateButton(discord.ui.Button):
    """A single date-picker button."""

    def __init__(self, label: str, target_date: date):
        super().__init__(label=label, style=discord.ButtonStyle.blurple)
        self.target_date = target_date

    async def callback(self, interaction: discord.Interaction):
        em = discord.Embed(
            title=f"Build Parlay \u2014 {_date_phrase(self.target_date)}",
            description="Step 2: Pick a league",
            color=COLOR_PURPLE,
        )
        view = BuildLeagueSelectView(self.target_date)
        await interaction.response.edit_message(embed=em, view=view)


class DateSelectView(discord.ui.View):
    """Step 1 view: choose a date."""

    def __init__(self):
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
            self.add_item(DateButton(label=label, target_date=d))


# Step 2: League selection

class BuildLeagueSelect(discord.ui.Select):
    """Dropdown to pick a league (step 2 of /build)."""

    def __init__(self, target_date: date):
        self.target_date = target_date
        options = [discord.SelectOption(label=lg, value=lg) for lg in ALL_LEAGUES]
        super().__init__(placeholder="Pick a league...", options=options)

    async def callback(self, interaction: discord.Interaction):
        league = self.values[0]
        em = discord.Embed(
            title=f"Build Parlay \u2014 {league} \u2014 {_date_phrase(self.target_date)}",
            description="Step 3: Pick a market type",
            color=COLOR_PURPLE,
        )
        view = BuildMarketSelectView(league, self.target_date)
        await interaction.response.edit_message(embed=em, view=view)


class BuildLeagueSelectView(discord.ui.View):
    def __init__(self, target_date: date):
        super().__init__(timeout=120)
        self.add_item(BuildLeagueSelect(target_date))


# Step 3: Market selection

class BuildMarketSelect(discord.ui.Select):
    """Dropdown to pick a market type (step 3 of /build)."""

    def __init__(self, league: str, target_date: date):
        self.league = league
        self.target_date = target_date
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
        view = BuildOddsTargetView(self.league, self.target_date, market)
        await interaction.response.edit_message(embed=em, view=view)


class BuildMarketSelectView(discord.ui.View):
    def __init__(self, league: str, target_date: date):
        super().__init__(timeout=120)
        self.add_item(BuildMarketSelect(league, target_date))


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

    def __init__(self, league: str, target_date: date, market: str):
        super().__init__(timeout=120)
        self.league = league
        self.target_date = target_date
        self.market = market

    async def _build(self, interaction: discord.Interaction, legs: int, odds: float):
        await interaction.response.defer(thinking=True)
        dp = _date_phrase(self.target_date)
        constraint = _MARKET_CONSTRAINTS.get(self.market, "")
        prompt = (
            f"For all {self.league} games on {dp}, make a {legs}-leg parlay "
            f"{constraint} with combined decimal odds at or below {odds}x."
        )
        text = _run_rag(prompt, self.league)
        embeds = parlay_embed(text, league=self.league)
        view = ParlayFollowUpView(self.league, dp)
        await interaction.followup.send(embeds=embeds, view=view)

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
    async def build(self, interaction: discord.Interaction):
        em = discord.Embed(
            title="Build a Parlay",
            description="Step 1: Pick a date",
            color=COLOR_PURPLE,
        )
        view = DateSelectView()
        await interaction.response.send_message(embed=em, view=view)

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
        await interaction.response.defer(thinking=True)

        target_date = _parse_date(date)
        dp = _date_phrase(target_date)
        lg = league.value
        market_val = market.value if market else None

        # Build the prompt based on context
        if lg == "cross-league":
            # Cross-league parlay
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
            rag_league = "EPL"  # default league for RAG engine; cross-league detected via prompt
        elif match:
            # Same-game parlay locked to a specific fixture
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
            # Standard single-league parlay
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

        text = _run_rag(prompt, rag_league)
        embeds = parlay_embed(text, league=display_league)
        view = ParlayFollowUpView(display_league, dp)
        await interaction.followup.send(embeds=embeds, view=view)

    @parlay_cmd.autocomplete("match")
    async def _parlay_match_ac(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        return await _match_autocomplete(interaction, current)


async def setup(bot: commands.Bot):
    await bot.add_cog(ParlayBuilder(bot))
