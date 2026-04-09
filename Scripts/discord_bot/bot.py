"""Betting RAG Discord Bot — entry point.

Usage:
    cd Scripts/discord_bot
    python bot.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import discord
from discord.ext import commands

# Ensure rag_ingest is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from config import BOT_TOKEN, PREMIUM_ROLE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("betting_rag_bot")

COGS = [
    "cogs.slash_cmds",
    "cogs.auto_push",
    "cogs.parlay_builder",
    "cogs.subscribe",
    "cogs.whale_chat",
    "cogs.trial",
    "cogs.tier_welcome",
    "cogs.promo_parlay",
]


class BettingRAGBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True  # needed for role management, on_member_update, and DMs
        super().__init__(
            command_prefix="!",
            intents=intents,
            chunk_guilds_at_startup=True,  # cache all members so on_member_update works
        )

        # Block all slash commands in DMs — must be used in the server
        self.tree.interaction_check = self._guild_only_check

    async def _guild_only_check(self, interaction: discord.Interaction) -> bool:
        """Reject all slash commands sent via DM."""
        if interaction.guild is None:
            await interaction.response.send_message(
                "Commands can only be used in the server, not in DMs.",
                ephemeral=True,
            )
            return False
        return True

    async def setup_hook(self):
        for cog in COGS:
            try:
                await self.load_extension(cog)
                log.info(f"Loaded cog: {cog}")
            except Exception as e:
                log.error(f"Failed to load cog {cog}: {e}")

    async def on_ready(self):
        log.info(f"Bot ready as {self.user} (ID: {self.user.id})")
        log.info(f"Guilds: {[g.name for g in self.guilds]}")

        # Sync slash commands per guild (instant, unlike global sync which takes ~1hr)
        for guild in self.guilds:
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            log.info(f"Synced {len(synced)} commands to '{guild.name}'")

        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="football markets",
            )
        )


def main():
    if not BOT_TOKEN:
        log.error(
            "DISCORD_BOT_TOKEN not set. Add it to your .env file:\n"
            "  DISCORD_BOT_TOKEN=your_token_here\n"
            "\n"
            "To get a token:\n"
            "  1. Go to https://discord.com/developers/applications\n"
            "  2. Create a New Application\n"
            "  3. Go to Bot tab → Reset Token → Copy\n"
            "  4. Enable MESSAGE CONTENT INTENT under Privileged Intents\n"
            "  5. Go to OAuth2 → URL Generator:\n"
            "     - Scopes: bot, applications.commands\n"
            "     - Permissions: Send Messages, Embed Links, Use Slash Commands,\n"
            "       Create Private Threads, Send Messages in Threads,\n"
            "       Manage Threads\n"
            "  6. Copy the generated URL and open it to invite the bot to your server"
        )
        sys.exit(1)

    print(f"[STARTUP] Premium roles configured: {PREMIUM_ROLE_NAMES}")
    log.info("Premium roles configured: %s", PREMIUM_ROLE_NAMES)
    bot = BettingRAGBot()
    bot.run(BOT_TOKEN)


if __name__ == "__main__":
    main()
