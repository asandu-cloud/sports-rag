"""Slash commands for the live Discord bot.

Commands:
    /livescores              — Show all currently live fixtures with scores
    /live <match>            — Full live analysis for one fixture (scoreboard embed)
    /liveprob <match> <market> — Live probability for a specific market
    /livestats <match>       — Raw live stats (corners, shots, cards, possession)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import discord
from discord import app_commands
from discord.ext import commands

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "Scripts" / "live"))
sys.path.insert(0, str(_ROOT / "Scripts" / "rag_ingest"))
sys.path.insert(0, str(_ROOT / "Scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import COLOR_BLUE, COLOR_RED
from embeds import (
    live_fixtures_embed,
    live_prob_embed,
    live_scoreboard_embed,
    live_stats_embed,
)

log = logging.getLogger("live_bot.commands_cog")


# ---------------------------------------------------------------------------
# Market choices for /liveprob
# ---------------------------------------------------------------------------
MARKET_CHOICES = [
    app_commands.Choice(name="Goals", value="goals"),
    app_commands.Choice(name="Corners", value="corners"),
    app_commands.Choice(name="Cards", value="cards"),
    app_commands.Choice(name="Shots on Target", value="sot"),
    app_commands.Choice(name="BTTS", value="btts"),
    app_commands.Choice(name="Moneyline", value="moneyline"),
]


class LiveCommandsCog(commands.Cog):
    """Slash commands for querying live match data."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    def _get_poller(self):
        """Get the LivePoller from the alerts cog."""
        alerts_cog = self.bot.get_cog("LiveAlertsCog")
        if alerts_cog is None:
            return None
        return getattr(alerts_cog, "poller", None)

    def _find_fixture(self, query: str) -> Optional[dict]:
        """Find a live fixture matching a search query.

        Matches against home/away team names (case-insensitive substring).
        Returns the fixture info dict or None.
        """
        poller = self._get_poller()
        if poller is None:
            return None

        active = poller.get_active_fixtures()
        query_lower = query.lower().strip()

        # Exact match first
        for f in active:
            home = f.get("home", "").lower()
            away = f.get("away", "").lower()
            if query_lower in home or query_lower in away:
                return f

        # Fuzzy: try matching parts of query separated by spaces
        query_parts = query_lower.split()
        for f in active:
            combined = f"{f.get('home', '')} {f.get('away', '')}".lower()
            if all(part in combined for part in query_parts):
                return f

        return None

    # ------------------------------------------------------------------
    # /livescores
    # ------------------------------------------------------------------

    @app_commands.command(
        name="livescores",
        description="Show all currently live fixtures with scores",
    )
    async def livescores(self, interaction: discord.Interaction) -> None:
        poller = self._get_poller()
        if poller is None:
            await interaction.response.send_message(
                "Live poller is not running. No live data available.",
                ephemeral=True,
            )
            return

        fixtures = poller.get_active_fixtures()
        em = live_fixtures_embed(fixtures)
        await interaction.response.send_message(embed=em)

    # ------------------------------------------------------------------
    # /live <match>
    # ------------------------------------------------------------------

    @app_commands.command(
        name="live",
        description="Full live analysis for one fixture",
    )
    @app_commands.describe(match="Team name or partial match (e.g. 'Arsenal')")
    async def live(self, interaction: discord.Interaction, match: str) -> None:
        poller = self._get_poller()
        if poller is None:
            await interaction.response.send_message(
                "Live poller is not running. No live data available.",
                ephemeral=True,
            )
            return

        fixture = self._find_fixture(match)
        if fixture is None:
            active = poller.get_active_fixtures()
            if not active:
                msg = "No matches are currently live."
            else:
                fixture_list = "\n".join(
                    f"  {f.get('home', '?')} vs {f.get('away', '?')} [{f.get('league', '')}]"
                    for f in active
                )
                msg = f"No match found for '{match}'. Currently live:\n```\n{fixture_list}\n```"
            await interaction.response.send_message(msg, ephemeral=True)
            return

        fid = fixture["fixture_id"]
        state = poller.get_state(fid)
        snapshot = poller.get_snapshot(fid)

        if state is None:
            await interaction.response.send_message(
                f"State not available for fixture {fid}.",
                ephemeral=True,
            )
            return

        if snapshot is None:
            # No snapshot yet — show basic info
            em = discord.Embed(
                title=f"\u26bd {fixture.get('home', '?')} {fixture.get('score', '?-?')} {fixture.get('away', '?')}",
                description="Projections not yet available. Waiting for first data update.",
                color=COLOR_BLUE,
            )
            em.set_footer(text="Matchwise Live")
            await interaction.response.send_message(embed=em)
            return

        em = live_scoreboard_embed(snapshot, state)
        await interaction.response.send_message(embed=em)

    # ------------------------------------------------------------------
    # /liveprob <match> <market>
    # ------------------------------------------------------------------

    @app_commands.command(
        name="liveprob",
        description="Live probability for a specific market",
    )
    @app_commands.describe(
        match="Team name or partial match (e.g. 'Arsenal')",
        market="Market type",
    )
    @app_commands.choices(market=MARKET_CHOICES)
    async def liveprob(
        self,
        interaction: discord.Interaction,
        match: str,
        market: app_commands.Choice[str],
    ) -> None:
        poller = self._get_poller()
        if poller is None:
            await interaction.response.send_message(
                "Live poller is not running.",
                ephemeral=True,
            )
            return

        fixture = self._find_fixture(match)
        if fixture is None:
            await interaction.response.send_message(
                f"No live match found for '{match}'.",
                ephemeral=True,
            )
            return

        fid = fixture["fixture_id"]
        state = poller.get_state(fid)
        snapshot = poller.get_snapshot(fid)

        if snapshot is None:
            await interaction.response.send_message(
                "Projections not yet available for this fixture.",
                ephemeral=True,
            )
            return

        em = live_prob_embed(snapshot, state, market.value)
        await interaction.response.send_message(embed=em)

    # ------------------------------------------------------------------
    # /livestats <match>
    # ------------------------------------------------------------------

    @app_commands.command(
        name="livestats",
        description="Raw live stats for a fixture",
    )
    @app_commands.describe(match="Team name or partial match (e.g. 'Arsenal')")
    async def livestats(self, interaction: discord.Interaction, match: str) -> None:
        poller = self._get_poller()
        if poller is None:
            await interaction.response.send_message(
                "Live poller is not running.",
                ephemeral=True,
            )
            return

        fixture = self._find_fixture(match)
        if fixture is None:
            await interaction.response.send_message(
                f"No live match found for '{match}'.",
                ephemeral=True,
            )
            return

        fid = fixture["fixture_id"]
        state = poller.get_state(fid)
        snapshot = poller.get_snapshot(fid)

        if state is None:
            await interaction.response.send_message(
                "State not available for this fixture.",
                ephemeral=True,
            )
            return

        em = live_stats_embed(snapshot, state)
        await interaction.response.send_message(embed=em)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(LiveCommandsCog(bot))
