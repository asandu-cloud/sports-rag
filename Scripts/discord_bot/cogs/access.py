"""Access control — premium role gate and private thread routing.

Premium commands are restricted to users with a configured role.
Responses are sent in a per-user private thread so conversations
are invisible to other members.
"""

from __future__ import annotations

import logging
from typing import Optional

import discord
from discord import app_commands

from config import PREMIUM_ROLE_NAMES, COLOR_BLUE

log = logging.getLogger("access")

# ---------------------------------------------------------------------------
# Role check
# ---------------------------------------------------------------------------

def _has_premium_role(interaction: discord.Interaction) -> bool:
    """Return True if the invoking member has any of the configured premium roles."""
    if not interaction.guild:
        log.warning("Premium check: no guild context")
        return False
    member = interaction.user
    if not isinstance(member, discord.Member):
        log.warning("Premium check: user %s is not a Member", member)
        return False
    role_names_lower = {r.name.lower() for r in member.roles}
    allowed_lower = {r.lower() for r in PREMIUM_ROLE_NAMES}
    matched = role_names_lower & allowed_lower
    if not matched:
        actual_roles = {r.name for r in member.roles}
        log.info("Premium check DENIED for %s — has roles: %s | allowed: %s",
                 member, actual_roles, set(PREMIUM_ROLE_NAMES))
    else:
        log.debug("Premium check OK for %s — matched: %s", member, matched)
    return bool(matched)


def premium_only():
    """App-command check that restricts usage to premium role holders."""

    async def predicate(interaction: discord.Interaction) -> bool:
        if _has_premium_role(interaction):
            return True
        roles_list = ", ".join(f"**{r}**" for r in PREMIUM_ROLE_NAMES)
        em = discord.Embed(
            title="Premium Required",
            description=(
                f"This command requires one of these roles: {roles_list}.\n\n"
                "Check the upgrade channel for details on how to get access."
            ),
            color=COLOR_BLUE,
        )
        try:
            await interaction.response.send_message(embed=em, ephemeral=True)
        except Exception as exc:
            log.warning("Failed to send premium gate message: %s", exc)
        return False

    return app_commands.check(predicate)


# ---------------------------------------------------------------------------
# Private thread routing
# ---------------------------------------------------------------------------

# Thread name prefix — used to find existing threads for a user.
_THREAD_PREFIX = "bot-"


async def get_or_create_private_thread(
    interaction: discord.Interaction,
) -> Optional[discord.Thread]:
    """Find or create a private thread for the invoking user.

    Returns the thread, or None if threads are unsupported in this channel.
    """
    channel = interaction.channel

    # Private threads require a text channel in a guild.
    if not isinstance(channel, discord.TextChannel):
        return None

    user = interaction.user
    thread_name = f"{_THREAD_PREFIX}{user.name}"

    # Look for an existing private thread for this user.
    # guild.threads only contains threads the bot can see (active + joined).
    for thread in channel.threads:
        if thread.name == thread_name and thread.is_private():
            return thread

    # Also check archived threads.
    async for thread in channel.archived_threads(private=True, limit=100):
        if thread.name == thread_name:
            # Unarchive by sending — discord auto-unarchives on new message.
            return thread

    # Create a new private thread.
    try:
        thread = await channel.create_thread(
            name=thread_name,
            type=discord.ChannelType.private_thread,
            invitable=False,
        )
        # Mention the user so they get added to the thread and can see it.
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
