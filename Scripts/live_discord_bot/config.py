"""Live Discord bot configuration.

SEPARATE bot from the pre-match prediction bot at Scripts/discord_bot/.
Requires its own Discord application and token: DISCORD_LIVE_BOT_TOKEN
"""

import os
from dotenv import load_dotenv

load_dotenv()

# SEPARATE token from the pre-match bot
LIVE_BOT_TOKEN = os.getenv("DISCORD_LIVE_BOT_TOKEN", "")

# Channel IDs for live alerts
CHANNEL_LIVE_ALERTS = int(os.getenv("DISCORD_CHANNEL_LIVE_ALERTS", "0"))
CHANNEL_LIVE_SCORES = int(os.getenv("DISCORD_CHANNEL_LIVE_SCORES", "0"))

# Poll interval (seconds) — how often the bot checks for live fixture updates
LIVE_POLL_INTERVAL = int(os.getenv("LIVE_POLL_INTERVAL", "45"))

# Leagues to track
DOMESTIC_LEAGUES = ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]
EUROPEAN_COMPS = ["UCL", "UEL", "UECL"]
ALL_LEAGUES = DOMESTIC_LEAGUES + EUROPEAN_COMPS

# Embed colors (same brand as pre-match bot)
COLOR_GREEN = 0x2ECC71
COLOR_YELLOW = 0xF1C40F
COLOR_RED = 0xE74C3C
COLOR_BLUE = 0x3498DB
COLOR_PURPLE = 0x9B59B6
COLOR_ORANGE = 0xE67E22  # for live-specific alerts
