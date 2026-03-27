"""Access control — tier-based command gating, rate limiting, and private thread routing.

Roles map to internal tiers:
    OWNER / MOD / FRIENDS & FAMILY → "unlimited" (bypass all checks)
    WHALE                          → "elite"     (unlimited commands)
    TIPSTER                        → "pro"       (35 commands/day pool)
    ROOKIE                         → "starter"   (5 commands/day pool)
    (no tier role)                 → "free"      (1 command/day, /market only)

Decorators:
    @tier_required("elite")    — only Whale+ can use this command
    @pooled_command("free")    — any tier can use, costs 1 from daily pool
    @pooled_command("starter") — Rookie+ can use, costs 1 from daily pool
    @pooled_command("pro")     — Tipster+ can use, costs 1 from daily pool
    @premium_only()            — legacy alias for tier_required("starter")
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, Optional, Tuple

import discord
from discord import app_commands

from config import (
    ROLE_TO_TIER, TIER_HIERARCHY, TIER_POOL_LIMITS,
    PREMIUM_ROLE_NAMES, COLOR_BLUE, COLOR_PURPLE,
)

log = logging.getLogger("access")

# ---------------------------------------------------------------------------
# Daily command pool tracker (in-memory, resets on date change)
# ---------------------------------------------------------------------------
# Key: (user_id, date_iso_str) → commands used today
_daily_pool: Dict[Tuple[int, str], int] = {}


def _get_pool_usage(user_id: int) -> int:
    """Return how many pooled commands this user has used today."""
    return _daily_pool.get((user_id, date.today().isoformat()), 0)


def _increment_pool(user_id: int) -> int:
    """Increment and return the new usage count."""
    key = (user_id, date.today().isoformat())
    _daily_pool[key] = _daily_pool.get(key, 0) + 1
    return _daily_pool[key]


# ---------------------------------------------------------------------------
# Tier resolution
# ---------------------------------------------------------------------------

def _get_user_tier(interaction: discord.Interaction) -> str:
    """Determine the user's highest tier from their Discord roles.

    Returns one of: "unlimited", "elite", "pro", "starter", "free".
    """
    if not interaction.guild:
        return "free"
    member = interaction.user
    if not isinstance(member, discord.Member):
        return "free"

    best_tier = "free"
    best_rank = 0

    for role in member.roles:
        tier = ROLE_TO_TIER.get(role.name.lower())
        if tier:
            rank = TIER_HIERARCHY.get(tier, 0)
            if rank > best_rank:
                best_tier = tier
                best_rank = rank

    return best_tier


def _has_premium_role(interaction: discord.Interaction) -> bool:
    """Legacy check — returns True if user has any paid or bypass role."""
    return _get_user_tier(interaction) != "free"


# ---------------------------------------------------------------------------
# Tier gate decorator (hard gate, no pool)
# ---------------------------------------------------------------------------

def tier_required(minimum_tier: str):
    """App-command check that requires a minimum tier (no pool cost)."""

    min_rank = TIER_HIERARCHY.get(minimum_tier, 0)
    tier_labels = {"starter": "Rookie", "pro": "Tipster", "elite": "Whale"}
    label = tier_labels.get(minimum_tier, minimum_tier.title())

    async def predicate(interaction: discord.Interaction) -> bool:
        user_tier = _get_user_tier(interaction)
        user_rank = TIER_HIERARCHY.get(user_tier, 0)

        if user_rank >= min_rank:
            return True

        em = discord.Embed(
            title="Upgrade Required",
            description=(
                f"This command requires the **{label}** plan or higher.\n\n"
                "Use `/subscribe` to view plans."
            ),
            color=COLOR_PURPLE,
        )
        try:
            await interaction.response.send_message(embed=em, ephemeral=True)
        except Exception as exc:
            log.warning("Failed to send tier gate message: %s", exc)
        return False

    return app_commands.check(predicate)


# ---------------------------------------------------------------------------
# Pooled command decorator (tier gate + daily pool)
# ---------------------------------------------------------------------------

def pooled_command(minimum_tier: str):
    """App-command check that requires a minimum tier AND costs 1 from the daily pool.

    - Whale+ and unlimited bypass the pool entirely.
    - Lower tiers deduct from their daily pool.
    - Shows remaining commands in the error message.
    """

    min_rank = TIER_HIERARCHY.get(minimum_tier, 0)
    tier_labels = {"free": "Free", "starter": "Rookie", "pro": "Tipster", "elite": "Whale"}

    async def predicate(interaction: discord.Interaction) -> bool:
        user_tier = _get_user_tier(interaction)
        user_rank = TIER_HIERARCHY.get(user_tier, 0)

        # 1. Check minimum tier
        if user_rank < min_rank:
            label = tier_labels.get(minimum_tier, minimum_tier.title())
            em = discord.Embed(
                title="Upgrade Required",
                description=(
                    f"This command requires the **{label}** plan or higher.\n\n"
                    "Use `/subscribe` to view plans."
                ),
                color=COLOR_PURPLE,
            )
            try:
                await interaction.response.send_message(embed=em, ephemeral=True)
            except Exception:
                pass
            return False

        # 2. Check pool (elite+ and unlimited skip)
        pool_limit = TIER_POOL_LIMITS.get(user_tier, 0)
        if pool_limit == 0:
            # Unlimited — no tracking needed
            return True

        user_id = interaction.user.id
        used = _get_pool_usage(user_id)

        if used >= pool_limit:
            # Pool exhausted
            current_label = tier_labels.get(user_tier, user_tier.title())
            # Suggest next tier
            if user_tier == "free":
                upgrade_msg = "Upgrade to **Rookie** for 5/day or **Tipster** for 35/day."
            elif user_tier == "starter":
                upgrade_msg = "Upgrade to **Tipster** for 35/day or **Whale** for unlimited."
            elif user_tier == "pro":
                upgrade_msg = "Upgrade to **Whale** for unlimited access."
            else:
                upgrade_msg = ""

            em = discord.Embed(
                title="Daily Limit Reached",
                description=(
                    f"You've used all **{pool_limit}** commands today ({current_label} plan).\n\n"
                    f"{upgrade_msg}\n"
                    "Use `/subscribe` to upgrade."
                ),
                color=COLOR_BLUE,
            )
            try:
                await interaction.response.send_message(embed=em, ephemeral=True)
            except Exception:
                pass
            return False

        # 3. Deduct from pool
        new_count = _increment_pool(user_id)
        remaining = pool_limit - new_count
        log.debug("Pool usage for %s: %d/%d (tier=%s)",
                  interaction.user, new_count, pool_limit, user_tier)
        return True

    return app_commands.check(predicate)


# ---------------------------------------------------------------------------
# Legacy decorator (backward compatible)
# ---------------------------------------------------------------------------

def premium_only():
    """Legacy decorator — equivalent to tier_required("starter")."""
    return tier_required("starter")


# ---------------------------------------------------------------------------
# Private thread routing
# ---------------------------------------------------------------------------

_THREAD_PREFIX = "bot-"


async def get_or_create_private_thread(
    interaction: discord.Interaction,
) -> Optional[discord.Thread]:
    """Find or create a private thread for the invoking user.

    Returns the thread, or None if threads are unsupported in this channel.
    """
    channel = interaction.channel

    if not isinstance(channel, discord.TextChannel):
        return None

    user = interaction.user
    thread_name = f"{_THREAD_PREFIX}{user.name}"

    for thread in channel.threads:
        if thread.name == thread_name and thread.is_private():
            return thread

    async for thread in channel.archived_threads(private=True, limit=100):
        if thread.name == thread_name:
            return thread

    try:
        thread = await channel.create_thread(
            name=thread_name,
            type=discord.ChannelType.private_thread,
            invitable=False,
        )
        await thread.send(
            content=f"{user.mention}",
            embed=discord.Embed(
                title="Your Private Bot Channel",
                description=(
                    "All your bot interactions will appear here.\n"
                    "Only you and the server admins can see this thread."
                ),
                color=COLOR_BLUE,
            ),
        )
        return thread
    except discord.Forbidden:
        log.warning("Missing permissions to create private thread for %s", user)
        return None
    except discord.HTTPException as exc:
        log.warning("Failed to create private thread for %s: %s", user, exc)
        return None
