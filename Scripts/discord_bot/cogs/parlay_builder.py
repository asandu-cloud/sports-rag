"""Interactive parlay builder using Discord Views and Buttons."""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from datetime import datetime
from typing import Dict, List, Optional

import discord
from discord import app_commands
from discord.ext import commands

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "rag_ingest"))
import rag_cli_v2 as rag  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from embeds import parlay_embed  # noqa: E402
from config import ALL_LEAGUES, COLOR_PURPLE, COLOR_BLUE  # noqa: E402


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"


def _today_phrase() -> str:
    d = datetime.now().date()
    return f"{_ordinal(d.day)} {d.strftime('%B')}"


def _run_rag(prompt: str, league: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rag.answer_once(prompt, default_league=league)
    return buf.getvalue().strip() or ""


# ---- Market type selector ----

class MarketSelect(discord.ui.Select):
    """Dropdown to pick market type after league is chosen."""

    def __init__(self, league: str):
        self.league = league
        options = [
            discord.SelectOption(label="Goals Totals", value="totals", emoji="⚽"),
            discord.SelectOption(label="Corners", value="corners", emoji="📐"),
            discord.SelectOption(label="Cards", value="cards", emoji="🟨"),
            discord.SelectOption(label="SoT", value="sot", emoji="🎯"),
            discord.SelectOption(label="BTTS", value="btts", emoji="🤝"),
            discord.SelectOption(label="Moneyline", value="moneyline", emoji="🏆"),
            discord.SelectOption(label="Mixed Markets", value="mixed", emoji="🎲"),
        ]
        super().__init__(placeholder="Pick a market type...", options=options)

    async def callback(self, interaction: discord.Interaction):
        market = self.values[0]
        await interaction.response.defer(thinking=True)

        date = _today_phrase()
        league = self.league

        prompts = {
            "totals": (
                f"For all {league} games on {date}, make a 4-leg parlay with exactly "
                "one leg per fixture using only full-game over/under goals totals lines. "
                "Keep combined decimal odds at or below 4.3x."
            ),
            "corners": (
                f"For all {league} games on {date}, make a parlay with ONLY CORNER TOTALS, "
                "one corner total per game, full-game markets only, around 3.0x combined odds."
            ),
            "cards": (
                f"For all {league} games on {date}, build a parlay with ONLY CARD TOTALS, "
                "one card total per game, full-game markets only, around 3.5x combined odds."
            ),
            "sot": (
                f"For all {league} games on {date}, build a parlay using ONLY shots on target "
                "totals, one SoT leg per game, full-game markets only, around 3.0x combined odds."
            ),
            "btts": (
                f"For all {league} games on {date}, which fixtures are best for BTTS Yes?"
            ),
            "moneyline": (
                f"For every {league} game on {date}, who is most likely to win? "
                "Show moneyline probabilities and value picks."
            ),
            "mixed": (
                f"Build a 4-leg {league} parlay for {date} around 4.0x. "
                "Mix goals, corners, and cards markets. At least two different fixtures."
            ),
        }

        prompt = prompts.get(market, prompts["totals"])
        text = _run_rag(prompt, league)
        embeds = parlay_embed(text, league=league)
        await interaction.followup.send(embeds=embeds)


class MarketSelectView(discord.ui.View):
    def __init__(self, league: str):
        super().__init__(timeout=120)
        self.add_item(MarketSelect(league))


# ---- League selector ----

class LeagueSelect(discord.ui.Select):
    """Dropdown to pick a league."""

    def __init__(self):
        options = [
            discord.SelectOption(label=lg, value=lg) for lg in ALL_LEAGUES
        ]
        super().__init__(placeholder="Pick a league...", options=options)

    async def callback(self, interaction: discord.Interaction):
        league = self.values[0]
        em = discord.Embed(
            title=f"Build Parlay — {league}",
            description="Now pick a market type:",
            color=COLOR_PURPLE,
        )
        view = MarketSelectView(league)
        await interaction.response.edit_message(embed=em, view=view)


class LeagueSelectView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=120)
        self.add_item(LeagueSelect())


# ---- Odds target buttons ----

class OddsTargetView(discord.ui.View):
    """Quick parlay builder with preset odds targets."""

    def __init__(self, league: str):
        super().__init__(timeout=120)
        self.league = league

    async def _build(self, interaction: discord.Interaction, legs: int, odds: float):
        await interaction.response.defer(thinking=True)
        date = _today_phrase()
        prompt = (
            f"For all {self.league} games on {date}, make a {legs}-leg parlay "
            f"with combined decimal odds at or below {odds}x."
        )
        text = _run_rag(prompt, self.league)
        embeds = parlay_embed(text, league=self.league)
        await interaction.followup.send(embeds=embeds)

    @discord.ui.button(label="Safe (3-leg, ~2.5x)", style=discord.ButtonStyle.green)
    async def safe(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._build(interaction, 3, 2.5)

    @discord.ui.button(label="Standard (4-leg, ~4x)", style=discord.ButtonStyle.blurple)
    async def standard(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._build(interaction, 4, 4.0)

    @discord.ui.button(label="Risky (5-leg, ~7x)", style=discord.ButtonStyle.red)
    async def risky(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._build(interaction, 5, 7.0)


# ---- Cog ----

class ParlayBuilder(commands.Cog):
    """Interactive parlay building commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="build", description="Interactive parlay builder")
    async def build(self, interaction: discord.Interaction):
        em = discord.Embed(
            title="Build a Parlay",
            description="Step 1: Pick a league to start.",
            color=COLOR_PURPLE,
        )
        view = LeagueSelectView()
        await interaction.response.send_message(embed=em, view=view)

    @app_commands.command(name="quickparlay", description="Quick parlay with preset risk levels")
    @app_commands.describe(league="League to query")
    @app_commands.choices(league=[app_commands.Choice(name=lg, value=lg) for lg in ALL_LEAGUES])
    async def quickparlay(self, interaction: discord.Interaction, league: app_commands.Choice[str]):
        em = discord.Embed(
            title=f"Quick Parlay — {league.value}",
            description="Pick your risk level:",
            color=COLOR_PURPLE,
        )
        view = OddsTargetView(league.value)
        await interaction.response.send_message(embed=em, view=view)


async def setup(bot: commands.Bot):
    await bot.add_cog(ParlayBuilder(bot))
