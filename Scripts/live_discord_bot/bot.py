"""Matchwise LIVE Discord Bot — in-play alerts and live projections.

This is a SEPARATE bot from the pre-match prediction bot at Scripts/discord_bot/.
Requires its own Discord application and token: DISCORD_LIVE_BOT_TOKEN

Usage:
    cd Scripts/live_discord_bot
    python bot.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import discord
from discord.ext import commands

# ---------------------------------------------------------------------------
# Path setup — ensure live engine and rag_ingest are importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "Scripts" / "live"))
sys.path.insert(0, str(_ROOT / "Scripts" / "rag_ingest"))
sys.path.insert(0, str(_ROOT / "Scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LIVE_BOT_TOKEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("live_bot")

COGS = [
    "cogs.live_alerts",
    "cogs.live_commands",
]


class MatchwiseLiveBot(commands.Bot):
    """Discord bot for live in-play betting alerts and projections."""

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!live ", intents=intents)

    async def setup_hook(self) -> None:
        for cog in COGS:
            try:
                await self.load_extension(cog)
                log.info("Loaded cog: %s", cog)
            except Exception as exc:
                log.error("Failed to load cog %s: %s", cog, exc, exc_info=True)

        # Sync slash commands with Discord
        synced = await self.tree.sync()
        log.info("Synced %d slash commands", len(synced))

    async def on_ready(self) -> None:
        log.info("Bot ready as %s (ID: %s)", self.user, self.user.id)
        log.info("Guilds: %s", [g.name for g in self.guilds])
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="live matches",
            )
        )


def main() -> None:
    if not LIVE_BOT_TOKEN:
        log.error(
            "DISCORD_LIVE_BOT_TOKEN not set. Add it to your .env file:\n"
            "  DISCORD_LIVE_BOT_TOKEN=your_token_here\n"
            "\n"
            "This is a SEPARATE token from DISCORD_BOT_TOKEN (pre-match bot).\n"
            "\n"
            "To get a token:\n"
            "  1. Go to https://discord.com/developers/applications\n"
            "  2. Create a New Application (e.g. 'Matchwise Live')\n"
            "  3. Go to Bot tab -> Reset Token -> Copy\n"
            "  4. Enable MESSAGE CONTENT INTENT under Privileged Intents\n"
            "  5. Go to OAuth2 -> URL Generator:\n"
            "     - Scopes: bot, applications.commands\n"
            "     - Permissions: Send Messages, Embed Links, Use Slash Commands\n"
            "  6. Copy the generated URL and open it to invite the bot to your server\n"
            "\n"
            "Also set these channel IDs in .env:\n"
            "  DISCORD_CHANNEL_LIVE_ALERTS=<channel_id>\n"
            "  DISCORD_CHANNEL_LIVE_SCORES=<channel_id>"
        )
        sys.exit(1)

    bot = MatchwiseLiveBot()
    bot.run(LIVE_BOT_TOKEN)


if __name__ == "__main__":
    main()
