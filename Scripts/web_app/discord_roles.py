"""
Discord role management via REST API for the Matchwise subscription system.

Used by the Stripe webhook handler to assign or remove tier roles when
subscriptions change.  All functions degrade gracefully -- errors are logged
and ``False`` is returned rather than raising exceptions.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, List

import aiohttp
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (all from environment)
# ---------------------------------------------------------------------------

DISCORD_BOT_TOKEN: str = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_GUILD_ID: str = os.getenv("DISCORD_GUILD_ID", "")

TIER_TO_ROLE_ID: Dict[str, str] = {
    "starter": os.getenv("DISCORD_ROLE_TIER1", ""),
    "pro": os.getenv("DISCORD_ROLE_TIER2", ""),
    "elite": os.getenv("DISCORD_ROLE_TIER3", ""),
}

ALL_TIER_ROLE_IDS: List[str] = [rid for rid in TIER_TO_ROLE_ID.values() if rid]

_API_BASE = "https://discord.com/api/v10"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _headers() -> Dict[str, str]:
    """Return the authorization and content-type headers."""
    return {
        "Authorization": f"Bot {DISCORD_BOT_TOKEN}",
        "Content-Type": "application/json",
    }


async def _add_role(discord_id: str, role_id: str) -> bool:
    """Add a single role to a guild member via the Discord REST API.

    Returns ``True`` on success (including 204 No Content), ``False`` on any
    error.
    """
    if not DISCORD_BOT_TOKEN or not DISCORD_GUILD_ID or not role_id:
        logger.warning(
            "_add_role skipped: missing token, guild, or role_id"
        )
        return False

    url = (
        f"{_API_BASE}/guilds/{DISCORD_GUILD_ID}"
        f"/members/{discord_id}/roles/{role_id}"
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.put(url, headers=_headers()) as resp:
                if resp.status == 429:
                    retry_after = (await resp.json()).get("retry_after", 1)
                    logger.warning(
                        "Rate limited adding role %s to %s, retrying in %.1fs",
                        role_id,
                        discord_id,
                        retry_after,
                    )
                    await asyncio.sleep(retry_after)
                    async with session.put(url, headers=_headers()) as retry:
                        if retry.status in (200, 204):
                            return True
                        logger.error(
                            "_add_role retry failed: %s %s",
                            retry.status,
                            await retry.text(),
                        )
                        return False

                if resp.status in (200, 204):
                    return True

                logger.error(
                    "_add_role failed: %s %s", resp.status, await resp.text()
                )
                return False

    except Exception:
        logger.exception(
            "_add_role error for discord_id=%s role_id=%s",
            discord_id,
            role_id,
        )
        return False


async def _remove_role(discord_id: str, role_id: str) -> bool:
    """Remove a single role from a guild member via the Discord REST API.

    Returns ``True`` on success (including 204 No Content), ``False`` on any
    error.
    """
    if not DISCORD_BOT_TOKEN or not DISCORD_GUILD_ID or not role_id:
        logger.warning(
            "_remove_role skipped: missing token, guild, or role_id"
        )
        return False

    url = (
        f"{_API_BASE}/guilds/{DISCORD_GUILD_ID}"
        f"/members/{discord_id}/roles/{role_id}"
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=_headers()) as resp:
                if resp.status == 429:
                    retry_after = (await resp.json()).get("retry_after", 1)
                    logger.warning(
                        "Rate limited removing role %s from %s, retrying in %.1fs",
                        role_id,
                        discord_id,
                        retry_after,
                    )
                    await asyncio.sleep(retry_after)
                    async with session.delete(url, headers=_headers()) as retry:
                        if retry.status in (200, 204):
                            return True
                        logger.error(
                            "_remove_role retry failed: %s %s",
                            retry.status,
                            await retry.text(),
                        )
                        return False

                if resp.status in (200, 204):
                    return True

                logger.error(
                    "_remove_role failed: %s %s",
                    resp.status,
                    await resp.text(),
                )
                return False

    except Exception:
        logger.exception(
            "_remove_role error for discord_id=%s role_id=%s",
            discord_id,
            role_id,
        )
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def assign_tier_role(discord_id: str, tier: str) -> bool:
    """Remove all tier roles, then assign the role matching *tier*.

    Returns ``True`` if the target role was added successfully (role removals
    are best-effort and do not affect the return value).  Returns ``False``
    if the tier has no mapped role or the add fails.
    """
    target_role_id = TIER_TO_ROLE_ID.get(tier, "")
    if not target_role_id:
        logger.warning(
            "assign_tier_role: no role mapped for tier=%s", tier
        )
        return False

    # Remove all tier roles first (best-effort) ---------------------------
    for role_id in ALL_TIER_ROLE_IDS:
        if role_id != target_role_id:
            await _remove_role(discord_id, role_id)

    # Assign the correct role ---------------------------------------------
    success = await _add_role(discord_id, target_role_id)
    if success:
        logger.info(
            "Assigned tier=%s role to discord_id=%s", tier, discord_id
        )
    else:
        logger.error(
            "Failed to assign tier=%s role to discord_id=%s", tier, discord_id
        )
    return success


async def remove_all_tier_roles(discord_id: str) -> bool:
    """Remove all three tier roles from a guild member.

    Returns ``True`` if all removals succeeded, ``False`` if any failed.
    """
    if not ALL_TIER_ROLE_IDS:
        logger.warning("remove_all_tier_roles: no tier role IDs configured")
        return False

    all_ok = True
    for role_id in ALL_TIER_ROLE_IDS:
        ok = await _remove_role(discord_id, role_id)
        if not ok:
            all_ok = False

    if all_ok:
        logger.info(
            "Removed all tier roles from discord_id=%s", discord_id
        )
    else:
        logger.warning(
            "Some tier role removals failed for discord_id=%s", discord_id
        )
    return all_ok
