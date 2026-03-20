"""Discord bot configuration."""

import os
from dotenv import load_dotenv

load_dotenv()

# Discord bot token — set in .env as DISCORD_BOT_TOKEN
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")

# Channel IDs for auto-push (3 channels)
CHANNEL_DAILY_PICKS = int(os.getenv("DISCORD_CHANNEL_DAILY_PICKS", "0"))    # all market predictions
CHANNEL_PARLAYS = int(os.getenv("DISCORD_CHANNEL_PARLAYS", "0"))            # parlays + value alerts + track record
CHANNEL_PLAYER_PROPS = int(os.getenv("DISCORD_CHANNEL_PLAYER_PROPS", "0"))  # player prop picks
CHANNEL_TEAM_LINES = int(os.getenv("DISCORD_CHANNEL_TEAM_LINES", "0"))      # per-team corner & card lines

# Legacy channel IDs (kept for backward compatibility, map to new channels)
CHANNEL_PARLAY_OF_DAY = int(os.getenv("DISCORD_CHANNEL_PARLAYS",
                            os.getenv("DISCORD_CHANNEL_PARLAY", "0")))
CHANNEL_MATCHDAY_DIGEST = int(os.getenv("DISCORD_CHANNEL_DAILY_PICKS",
                              os.getenv("DISCORD_CHANNEL_DIGEST", "0")))
CHANNEL_VALUE_ALERTS = int(os.getenv("DISCORD_CHANNEL_PARLAYS",
                           os.getenv("DISCORD_CHANNEL_ALERTS", "0")))
CHANNEL_TRACK_RECORD = int(os.getenv("DISCORD_CHANNEL_PARLAYS",
                           os.getenv("DISCORD_CHANNEL_TRACK_RECORD", "0")))
CHANNEL_MATCH_PREDICTIONS = int(os.getenv("DISCORD_CHANNEL_DAILY_PICKS",
                                os.getenv("DISCORD_CHANNEL_PREDICTIONS", "0")))
CHANNEL_CORNERS_CARDS = int(os.getenv("DISCORD_CHANNEL_DAILY_PICKS",
                            os.getenv("DISCORD_CHANNEL_CORNERS_CARDS", "0")))
CHANNEL_CORRECT_SCORE = int(os.getenv("DISCORD_CHANNEL_DAILY_PICKS",
                            os.getenv("DISCORD_CHANNEL_CORRECT_SCORE", "0")))

# Auto-push schedule (hours in UTC)
DIGEST_POST_HOUR = int(os.getenv("DISCORD_DIGEST_HOUR", "8"))  # 8 AM UTC
ALERT_SCAN_INTERVAL_MINUTES = int(os.getenv("DISCORD_ALERT_INTERVAL", "120"))

# Value alert thresholds
MIN_ALERT_MODEL_PROB = 0.65
MIN_ALERT_VALUE_EDGE = 0.08

# Access control — role names that can interact with the bot
# Users with ANY of these roles can use premium commands.
# Set via env (comma-separated) or override here.
PREMIUM_ROLE_NAMES = [
    r.strip()
    for r in os.getenv("DISCORD_PREMIUM_ROLES", "Premium").split(",")
    if r.strip()
]

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
