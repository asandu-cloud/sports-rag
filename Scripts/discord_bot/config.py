"""Discord bot configuration."""

import os
from dotenv import load_dotenv

load_dotenv()

# Discord bot token — set in .env as DISCORD_BOT_TOKEN
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")

# Channel IDs for auto-push (7 channels)
CHANNEL_DAILY_PICKS = int(os.getenv("DISCORD_CHANNEL_DAILY_PICKS", "0"))    # all market predictions
CHANNEL_PARLAYS = int(os.getenv("DISCORD_CHANNEL_PARLAYS", "0"))            # parlays + value alerts
CHANNEL_PLAYER_PROPS = int(os.getenv("DISCORD_CHANNEL_PLAYER_PROPS", "0"))  # player prop picks
CHANNEL_TEAM_LINES = int(os.getenv("DISCORD_CHANNEL_TEAM_LINES", "0"))      # per-team corner & card lines
CHANNEL_BEST_PICKS = int(os.getenv("DISCORD_CHANNEL_BEST_PICKS", "0"))      # daily track record summary
CHANNEL_NEWSLETTER = int(os.getenv("DISCORD_CHANNEL_NEWSLETTER", "0"))      # matchday intel (WHALE only)
CHANNEL_HIGH_RISK = int(os.getenv("DISCORD_CHANNEL_HIGH_RISK", "0"))        # best 1.85-2.4 odds unit bets

# Legacy channel IDs (kept for backward compatibility, map to new channels)
CHANNEL_PARLAY_OF_DAY = int(os.getenv("DISCORD_CHANNEL_PARLAYS",
                            os.getenv("DISCORD_CHANNEL_PARLAY", "0")))
CHANNEL_MATCHDAY_DIGEST = int(os.getenv("DISCORD_CHANNEL_DAILY_PICKS",
                              os.getenv("DISCORD_CHANNEL_DIGEST", "0")))
CHANNEL_VALUE_ALERTS = int(os.getenv("DISCORD_CHANNEL_PARLAYS",
                           os.getenv("DISCORD_CHANNEL_ALERTS", "0")))
CHANNEL_TRACK_RECORD = int(os.getenv("DISCORD_CHANNEL_BEST_PICKS",
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

# ---------------------------------------------------------------------------
# Subscription tier system
# ---------------------------------------------------------------------------

# Tier role display names in Discord
TIER_ROLES = {
    "starter": os.getenv("DISCORD_ROLE_TIER1_NAME", "ROOKIE"),
    "pro": os.getenv("DISCORD_ROLE_TIER2_NAME", "TIPSTER"),
    "elite": os.getenv("DISCORD_ROLE_TIER3_NAME", "WHALE"),
}

# Guild ID for role management via REST API
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID", "")

# Role name (case-insensitive) -> internal tier
ROLE_TO_TIER = {
    "rookie": "starter",
    "tipster": "pro",
    "whale": "elite",
    "friends & family": "unlimited",
    "mod": "unlimited",
    "owner": "unlimited",
}

# Tier hierarchy for permission checks
TIER_HIERARCHY = {
    "free": 0,
    "starter": 1,
    "pro": 2,
    "elite": 3,
    "unlimited": 99,
}

# Daily command pool sizes per tier (0 = unlimited)
TIER_POOL_LIMITS = {
    "free": 1,
    "starter": 5,
    "pro": 15,
    "elite": 0,
    "unlimited": 0,
}

# All roles that grant any level of access (used by legacy premium_only check)
PREMIUM_ROLE_NAMES = list(TIER_ROLES.values()) + ["OWNER", "MOD", "FRIENDS & FAMILY"]

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
