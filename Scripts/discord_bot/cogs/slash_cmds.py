"""Slash commands — each maps to a RAG template."""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

# Add rag_ingest to path so we can import the engine
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "rag_ingest"))
import rag_cli_v2 as rag  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from embeds import rag_output_to_embeds, parlay_embed  # noqa: E402
from config import ALL_LEAGUES  # noqa: E402


def _run_rag(prompt: str, league: str) -> str:
    """Run a RAG query and capture the text output."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rag.answer_once(prompt, default_league=league)
    return buf.getvalue().strip() or "(No output from model)"


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"


def _today_phrase() -> str:
    d = datetime.now().date()
    return f"{_ordinal(d.day)} {d.strftime('%B')}"


def _league_choices() -> list:
    return [app_commands.Choice(name=lg, value=lg) for lg in ALL_LEAGUES]


class SlashCommands(commands.Cog):
    """Slash commands for quick RAG lookups."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    # --- /totals ---
    @app_commands.command(name="totals", description="Goals over/under advice for all fixtures")
    @app_commands.describe(league="League to query")
    @app_commands.choices(league=_league_choices())
    async def totals(self, interaction: discord.Interaction, league: app_commands.Choice[str]):
        await interaction.response.defer(thinking=True)
        date = _today_phrase()
        prompt = f"Give me the over/under goals line I should take for each {league.value} game on {date}."
        text = _run_rag(prompt, league.value)
        embeds = rag_output_to_embeds("Goals Totals", text, league=league.value)
        await interaction.followup.send(embeds=embeds)

    # --- /corners ---
    @app_commands.command(name="corners", description="Corner totals advice for all fixtures")
    @app_commands.describe(league="League to query")
    @app_commands.choices(league=_league_choices())
    async def corners(self, interaction: discord.Interaction, league: app_commands.Choice[str]):
        await interaction.response.defer(thinking=True)
        date = _today_phrase()
        prompt = f"For every {league.value} game on {date}, recommend a corner totals line for each fixture."
        text = _run_rag(prompt, league.value)
        embeds = rag_output_to_embeds("Corner Totals", text, league=league.value)
        await interaction.followup.send(embeds=embeds)

    # --- /cards ---
    @app_commands.command(name="cards", description="Card totals advice for all fixtures")
    @app_commands.describe(league="League to query")
    @app_commands.choices(league=_league_choices())
    async def cards(self, interaction: discord.Interaction, league: app_commands.Choice[str]):
        await interaction.response.defer(thinking=True)
        date = _today_phrase()
        prompt = f"For each {league.value} game on {date}, recommend a cards over/under line."
        text = _run_rag(prompt, league.value)
        embeds = rag_output_to_embeds("Card Totals", text, league=league.value)
        await interaction.followup.send(embeds=embeds)

    # --- /btts ---
    @app_commands.command(name="btts", description="Both Teams To Score advice for all fixtures")
    @app_commands.describe(league="League to query")
    @app_commands.choices(league=_league_choices())
    async def btts(self, interaction: discord.Interaction, league: app_commands.Choice[str]):
        await interaction.response.defer(thinking=True)
        date = _today_phrase()
        prompt = f"For all {league.value} games on {date}, which fixtures are best for BTTS Yes?"
        text = _run_rag(prompt, league.value)
        embeds = rag_output_to_embeds("BTTS", text, league=league.value)
        await interaction.followup.send(embeds=embeds)

    # --- /moneyline ---
    @app_commands.command(name="moneyline", description="Match winner predictions for all fixtures")
    @app_commands.describe(league="League to query")
    @app_commands.choices(league=_league_choices())
    async def moneyline(self, interaction: discord.Interaction, league: app_commands.Choice[str]):
        await interaction.response.defer(thinking=True)
        date = _today_phrase()
        prompt = (
            f"For every {league.value} game on {date}, who is most likely to win? "
            "Show moneyline probabilities and value picks."
        )
        text = _run_rag(prompt, league.value)
        embeds = rag_output_to_embeds("Moneyline", text, league=league.value)
        await interaction.followup.send(embeds=embeds)

    # --- /spreads ---
    @app_commands.command(name="spreads", description="Handicap/spread advice for all fixtures")
    @app_commands.describe(league="League to query")
    @app_commands.choices(league=_league_choices())
    async def spreads(self, interaction: discord.Interaction, league: app_commands.Choice[str]):
        await interaction.response.defer(thinking=True)
        date = _today_phrase()
        prompt = f"For every {league.value} game on {date}, recommend a handicap line for each fixture."
        text = _run_rag(prompt, league.value)
        embeds = rag_output_to_embeds("Handicap Lines", text, league=league.value)
        await interaction.followup.send(embeds=embeds)

    # --- /correctscore ---
    @app_commands.command(name="correctscore", description="Correct score predictions for all fixtures")
    @app_commands.describe(league="League to query")
    @app_commands.choices(league=_league_choices())
    async def correctscore(self, interaction: discord.Interaction, league: app_commands.Choice[str]):
        await interaction.response.defer(thinking=True)
        date = _today_phrase()
        prompt = f"For every {league.value} game on {date}, predict the most likely correct score."
        text = _run_rag(prompt, league.value)
        embeds = rag_output_to_embeds("Correct Score", text, league=league.value)
        await interaction.followup.send(embeds=embeds)

    # --- /intervals ---
    @app_commands.command(name="intervals", description="Goal interval probabilities for all fixtures")
    @app_commands.describe(league="League to query")
    @app_commands.choices(league=_league_choices())
    async def intervals(self, interaction: discord.Interaction, league: app_commands.Choice[str]):
        await interaction.response.defer(thinking=True)
        date = _today_phrase()
        prompt = (
            f"For every {league.value} game on {date}, show me the goal interval "
            "probabilities for each fixture."
        )
        text = _run_rag(prompt, league.value)
        embeds = rag_output_to_embeds("Goal Intervals", text, league=league.value)
        await interaction.followup.send(embeds=embeds)

    # --- /sot ---
    @app_commands.command(name="sot", description="Shots on target totals for all fixtures")
    @app_commands.describe(league="League to query")
    @app_commands.choices(league=_league_choices())
    async def sot(self, interaction: discord.Interaction, league: app_commands.Choice[str]):
        await interaction.response.defer(thinking=True)
        date = _today_phrase()
        prompt = (
            f"For every {league.value} game on {date}, recommend a shots on target "
            "over/under line for each fixture."
        )
        text = _run_rag(prompt, league.value)
        embeds = rag_output_to_embeds("SoT Totals", text, league=league.value)
        await interaction.followup.send(embeds=embeds)

    # --- /parlay ---
    @app_commands.command(name="parlay", description="Auto-generate a parlay")
    @app_commands.describe(
        league="League to query",
        legs="Number of legs (default 4)",
        odds="Target combined odds (default 4.0)",
    )
    @app_commands.choices(league=_league_choices())
    async def parlay_cmd(
        self,
        interaction: discord.Interaction,
        league: app_commands.Choice[str],
        legs: Optional[int] = 4,
        odds: Optional[float] = 4.0,
    ):
        await interaction.response.defer(thinking=True)
        date = _today_phrase()
        prompt = (
            f"For all {league.value} games on {date}, make a {legs}-leg parlay "
            f"with combined decimal odds at or below {odds}x."
        )
        text = _run_rag(prompt, league.value)
        embeds = parlay_embed(text, league=league.value)
        await interaction.followup.send(embeds=embeds)

    # --- /teamlines ---
    @app_commands.command(name="teamlines", description="Per-team corner and card lines for all fixtures")
    @app_commands.describe(league="League to query")
    @app_commands.choices(league=_league_choices())
    async def teamlines(self, interaction: discord.Interaction, league: app_commands.Choice[str]):
        await interaction.response.defer(thinking=True)
        date = _today_phrase()
        prompt = (
            f"For all {league.value} games on {date}, what are the per-team corner "
            "and card lines for each fixture?"
        )
        text = _run_rag(prompt, league.value)
        embeds = rag_output_to_embeds("Team Lines", text, league=league.value)
        await interaction.followup.send(embeds=embeds)


async def setup(bot: commands.Bot):
    await bot.add_cog(SlashCommands(bot))
