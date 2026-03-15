"""Discord bot configuration."""

import os
from dotenv import load_dotenv

load_dotenv()

# Discord bot token — set in .env as DISCORD_BOT_TOKEN
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")

# Channel IDs for auto-push (set these after creating channels in your server)
CHANNEL_PARLAY_OF_DAY = int(os.getenv("DISCORD_CHANNEL_PARLAY", "0"))
CHANNEL_MATCHDAY_DIGEST = int(os.getenv("DISCORD_CHANNEL_DIGEST", "0"))
CHANNEL_VALUE_ALERTS = int(os.getenv("DISCORD_CHANNEL_ALERTS", "0"))
CHANNEL_TRACK_RECORD = int(os.getenv("DISCORD_CHANNEL_TRACK_RECORD", "0"))

# Auto-push schedule (hours in UTC)
DIGEST_POST_HOUR = int(os.getenv("DISCORD_DIGEST_HOUR", "8"))  # 8 AM UTC
ALERT_SCAN_INTERVAL_MINUTES = int(os.getenv("DISCORD_ALERT_INTERVAL", "120"))

# Value alert thresholds
MIN_ALERT_MODEL_PROB = 0.65
MIN_ALERT_VALUE_EDGE = 0.08

# Leagues
DOMESTIC_LEAGUES = ["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1"]
EUROPEAN_COMPS = ["UCL", "UEL", "UECL"]
ALL_LEAGUES = DOMESTIC_LEAGUES + EUROPEAN_COMPS

# Embed colors
COLOR_GREEN = 0x2ECC71   # high confidence
COLOR_YELLOW = 0xF1C40F  # medium confidence
COLOR_RED = 0xE74C3C     # low confidence
COLOR_BLUE = 0x3498DB    # info / neutral
COLOR_PURPLE = 0x9B59B6  # parlay
