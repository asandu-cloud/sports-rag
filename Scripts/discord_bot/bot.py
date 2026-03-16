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

from config import BOT_TOKEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("betting_rag_bot")

COGS = [
    "cogs.slash_cmds",
    "cogs.auto_push",
    "cogs.parlay_builder",
]


class BettingRAGBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self):
        for cog in COGS:
            try:
                await self.load_extension(cog)
                log.info(f"Loaded cog: {cog}")
            except Exception as e:
                log.error(f"Failed to load cog {cog}: {e}")

        # Sync slash commands with Discord
        synced = await self.tree.sync()
        log.info(f"Synced {len(synced)} slash commands")

    async def on_ready(self):
        log.info(f"Bot ready as {self.user} (ID: {self.user.id})")
        log.info(f"Guilds: {[g.name for g in self.guilds]}")
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

    bot = BettingRAGBot()
    bot.run(BOT_TOKEN)


if __name__ == "__main__":
    main()
