"""Tier welcome DMs — personalized messages sent when users get a subscription role.

Detects ROOKIE, TIPSTER, and WHALE role assignments via on_member_update
and sends tier-specific welcome DMs automatically.
"""

from __future__ import annotations

import logging
import os

import discord
from discord.ext import commands

from config import ROLE_TO_TIER, COLOR_GREEN, COLOR_BLUE, COLOR_PURPLE

BOT_OWNER_ID = int(os.getenv("DISCORD_BOT_OWNER_ID", "0"))

log = logging.getLogger("tier_welcome")

# ---------------------------------------------------------------------------
# Role detection helpers
# ---------------------------------------------------------------------------

# Build reverse lookup: tier -> set of role names (lowercase)
_TIER_ROLE_NAMES = {}
for _role_name, _tier in ROLE_TO_TIER.items():
    _TIER_ROLE_NAMES.setdefault(_tier, set()).add(_role_name.lower())

# We only send DMs for these tiers
# Whale DM is handled by whale_chat.py (needs thread link)
_WELCOME_TIERS = {"starter", "pro"}


def _detect_new_tier(before: discord.Member, after: discord.Member) -> str | None:
    """Detect if a subscription tier role was just added.

    Returns the tier string ("starter", "pro", "elite") or None.
    Only triggers on role ADDITION, not removal.
    """
    before_roles = {r.name.lower() for r in before.roles}
    after_roles = {r.name.lower() for r in after.roles}

    for tier in _WELCOME_TIERS:
        role_names = _TIER_ROLE_NAMES.get(tier, set())
        had_role = bool(before_roles & role_names)
        has_role = bool(after_roles & role_names)
        if not had_role and has_role:
            return tier

    return None


# ---------------------------------------------------------------------------
# Welcome messages
# ---------------------------------------------------------------------------

def _rookie_welcome() -> discord.Embed:
    em = discord.Embed(
        title="Welcome to Spix's Picks \U0001f94a",
        description=(
            "You just unlocked access to data-driven predictions across 5 European leagues."
        ),
        color=COLOR_GREEN,
    )
    em.add_field(
        name="What's waiting for you",
        value=(
            "**#daily-picks** \u2014 Our top curated picks drop every matchday morning. "
            "This is where most members start their gameday.\n\n"
            "**#best-picks** \u2014 Every morning after, we show exactly what hit and what "
            "didn't. Full transparency, no hiding misses.\n\n"
            "**Bot commands** \u2014 Head to #bot-commands and try `/market EPL goals` to see "
            "league-wide predictions. You have 5 commands per day \u2014 make them count.\n\n"
            "**Voice channels** \u2014 On matchdays, jump into the league voice chats. "
            "Watch the games with the community."
        ),
        inline=False,
    )
    em.add_field(
        name="Quick tip",
        value=(
            "Check **#best-picks** first. See our track record before you place anything. "
            "That's what we'd do."
        ),
        inline=False,
    )
    em.set_footer(text="Spix's Picks | See you on matchday.")
    return em


def _tipster_welcome() -> discord.Embed:
    em = discord.Embed(
        title="Welcome to Spix's Picks \U0001f4a1",
        description=(
            "You just joined the tier that most serious bettors land on. "
            "Here's what you've unlocked."
        ),
        color=COLOR_BLUE,
    )
    em.add_field(
        name="Everything in Rookie, plus",
        value=(
            "**#parlays** \u2014 Three auto-generated parlays every matchday: "
            "a Lock (~1.8x), a Safe play (~2.5x), and a Standard (~5x). Pick your risk level.\n\n"
            "**#player-props** \u2014 Player prop picks posted 30 minutes before kickoff "
            "with confirmed lineups. Goals, cards, SoT, fouls.\n\n"
            "**#team-lines** \u2014 Per-team corner and card lines for every fixture. "
            "This is where the quiet edges live.\n\n"
            "**All 8 leagues** \u2014 Champions League, Europa League, and Conference League "
            "are now in your feed.\n\n"
            "**15 commands/day** \u2014 Try `/parlay EPL` to build a custom parlay, "
            "or `/players EPL goals` for goalscorer props."
        ),
        inline=False,
    )
    em.add_field(
        name="The Tipster Lounge",
        value=(
            "**#lounge** is your space. Chat strategy, post your slips in **#my-bets**, "
            "celebrate in **#wins**. The community is what keeps people here longer than the picks."
        ),
        inline=False,
    )
    em.set_footer(text="Spix's Picks | See you in the lounge.")
    return em


def _whale_welcome() -> discord.Embed:
    em = discord.Embed(
        title="Welcome to Whale tier \U0001f40b",
        description=(
            "You now have the full arsenal. Everything in Tipster, plus tools "
            "that nobody else in this server has access to."
        ),
        color=COLOR_PURPLE,
    )
    em.add_field(
        name="Your private stats desk",
        value=(
            "You have a dedicated chat thread where you can ask anything in plain English \u2014 "
            "\"break down Arsenal vs Chelsea for goals, corners and cards\" or "
            "\"who has the stronger recent form in Bayern vs Dortmund?\" Just type. No commands needed.\n\n"
            "Whale chat is stats-first: it helps you build your own read rather than pushing picks.\n\n"
            "\u2192 **Go to #whale-ai and open your thread to start.**"
        ),
        inline=False,
    )
    em.add_field(
        name="What else you get",
        value=(
            "**#newsletter** \u2014 Every matchday morning, you get a briefing sourced from "
            "local sports media across Europe. Injury updates, lineup hints, tactical changes \u2014 "
            "in English, from Kicker, Marca, L'Equipe and more.\n\n"
            "**Unlimited commands** \u2014 No daily cap. Use `/analyze` for single-fixture "
            "deep dives that break down where the edge comes from.\n\n"
            "**#whale-den** \u2014 The smallest room in the server. Just you and the other Whales."
        ),
        inline=False,
    )
    em.add_field(
        name="One tip",
        value=(
            "Start by typing something in your AI chat thread. Ask about this weekend's fixtures. "
            "See how it feels to have a stats team in your pocket."
        ),
        inline=False,
    )
    em.set_footer(text="Spix's Picks | Welcome to the inner circle.")
    return em


_TIER_TO_EMBED = {
    "starter": _rookie_welcome,
    "pro": _tipster_welcome,
    "elite": _whale_welcome,
}


# ---------------------------------------------------------------------------
# Channel mention helper
# ---------------------------------------------------------------------------

def _ch(bot: commands.Bot, name: str) -> str:
    """Try to get a clickable channel mention by name. Falls back to bold text."""
    for guild in bot.guilds:
        for channel in guild.channels:
            if channel.name == name:
                return channel.mention
    return f"**#{name}**"


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class TierWelcome(commands.Cog):
    """Send personalized welcome DMs when users receive subscription roles."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    def _build_rookie_dm(self) -> discord.Embed:
        """Build Rookie welcome with live channel links."""
        dp = _ch(self.bot, "daily-picks")
        bp = _ch(self.bot, "best-picks")
        bc = _ch(self.bot, "bot-commands")

        em = discord.Embed(
            title="Welcome to Spix's Picks \U0001f94a",
            description="You just unlocked access to data-driven predictions across 5 European leagues.",
            color=COLOR_GREEN,
        )
        em.add_field(
            name="What's waiting for you",
            value=(
                f"{dp} \u2014 Our top curated picks drop every matchday morning. "
                "This is where most members start their gameday.\n\n"
                f"{bp} \u2014 Every morning after, we show exactly what hit and what "
                "didn't. Full transparency, no hiding misses.\n\n"
                f"**Bot commands** \u2014 Head to {bc} and try `/market EPL goals` to see "
                "league-wide predictions. You have 5 commands per day \u2014 make them count.\n\n"
                "**Voice channels** \u2014 On matchdays, jump into the league voice chats. "
                "Watch the games with the community."
            ),
            inline=False,
        )
        em.add_field(
            name="Quick tip",
            value=f"Check {bp} first. See our track record before you place anything. That's what we'd do.",
            inline=False,
        )
        em.set_footer(text="Spix's Picks | See you on matchday.")
        return em

    def _build_tipster_dm(self) -> discord.Embed:
        """Build Tipster welcome with live channel links."""
        parlays = _ch(self.bot, "parlays")
        pp = _ch(self.bot, "player-props")
        tl = _ch(self.bot, "team-lines")
        lounge = _ch(self.bot, "lounge")
        my_bets = _ch(self.bot, "my-bets")
        wins = _ch(self.bot, "wins")

        em = discord.Embed(
            title="Welcome to Spix's Picks \U0001f4a1",
            description="You just joined the tier that most serious bettors land on. Here's what you've unlocked.",
            color=COLOR_BLUE,
        )
        em.add_field(
            name="Everything in Rookie, plus",
            value=(
                f"{parlays} \u2014 Three auto-generated parlays every matchday: "
                "a Lock (~1.8x), a Safe play (~2.5x), and a Standard (~5x). Pick your risk level.\n\n"
                f"{pp} \u2014 Player prop picks posted 30 minutes before kickoff "
                "with confirmed lineups. Goals, cards, SoT, fouls.\n\n"
                f"{tl} \u2014 Per-team corner and card lines for every fixture. "
                "This is where the quiet edges live.\n\n"
                "**All 8 leagues** \u2014 Champions League, Europa League, and Conference League "
                "are now in your feed.\n\n"
                "**15 commands/day** \u2014 Try `/parlay EPL` to build a custom parlay, "
                "or `/players EPL goals` for goalscorer props."
            ),
            inline=False,
        )
        em.add_field(
            name="The Tipster Lounge",
            value=(
                f"{lounge} is your space. Chat strategy, post your slips in {my_bets}, "
                f"celebrate in {wins}. The community is what keeps people here longer than the picks."
            ),
            inline=False,
        )
        em.set_footer(text="Spix's Picks | See you in the lounge.")
        return em

    def _build_whale_dm(self) -> discord.Embed:
        """Build Whale welcome with live channel links."""
        whale_ai = _ch(self.bot, "whale-ai")
        newsletter = _ch(self.bot, "newsletter")
        whale_den = _ch(self.bot, "whale-den")

        em = discord.Embed(
            title="Welcome to Whale tier \U0001f40b",
            description=(
                "You now have the full arsenal. Everything in Tipster, plus tools "
                "that nobody else in this server has access to."
            ),
            color=COLOR_PURPLE,
        )
        em.add_field(
            name="Your private stats desk",
            value=(
                "You have a dedicated chat thread where you can ask anything in plain English \u2014 "
                "\"break down Arsenal vs Chelsea for goals, corners and cards\" or "
                "\"who has the stronger recent form in Bayern vs Dortmund?\" Just type. No commands needed.\n\n"
                "Whale chat is stats-first: it helps you build your own read rather than pushing picks.\n\n"
                f"\u2192 **Go to {whale_ai} and open your thread to start.**"
            ),
            inline=False,
        )
        em.add_field(
            name="What else you get",
            value=(
                f"{newsletter} \u2014 Every matchday morning, you get a briefing sourced from "
                "local sports media across Europe. Injury updates, lineup hints, tactical changes \u2014 "
                "in English, from Kicker, Marca, L'Equipe and more.\n\n"
                "**Unlimited commands** \u2014 No daily cap. Use `/analyze` for single-fixture "
                "deep dives that break down where the edge comes from.\n\n"
                f"{whale_den} \u2014 The smallest room in the server. Just you and the other Whales."
            ),
            inline=False,
        )
        em.add_field(
            name="One tip",
            value=(
                "Start by typing something in your AI chat thread. Ask about this weekend's fixtures. "
                "See how it feels to have a stats team in your pocket."
            ),
            inline=False,
        )
        em.set_footer(text="Spix's Picks | Welcome to the inner circle.")
        return em

    @commands.Cog.listener()
    async def on_member_update(self, before: discord.Member, after: discord.Member):
        """Detect tier role assignments and send welcome DMs."""
        new_tier = _detect_new_tier(before, after)
        if not new_tier:
            return

        log.info("New %s subscriber detected: %s", new_tier, after)

        # Build tier-specific embed with live channel links
        if new_tier == "starter":
            em = self._build_rookie_dm()
        elif new_tier == "pro":
            em = self._build_tipster_dm()
        elif new_tier == "elite":
            em = self._build_whale_dm()
        else:
            return

        try:
            await after.send(embed=em)
            log.info("Sent %s welcome DM to %s", new_tier, after)
        except discord.Forbidden:
            log.warning("Can't DM %s (DMs disabled)", after)
        except Exception as exc:
            log.warning("Failed to send %s welcome DM to %s: %s", new_tier, after, exc)


    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member):
        """Send an onboarding DM when a new user joins the server."""
        em = discord.Embed(
            title="Welcome to Spix's Picks",
            description=(
                "We use AI models to analyze every match across Europe's top leagues "
                "— and turn that into daily betting picks, parlays, and player props.\n\n"
                "Here's how to get started:"
            ),
            color=0x2ECC71,
        )
        em.add_field(
            name="Start Your Free Trial",
            value=(
                "Run `/trial` in any channel to activate your free trial. "
                "We'll post 2 curated parlays for you every matchday — no card required, no time limit.\n\n"
                "Once 2 of our picks hit, we'll show you exactly how much you would've made. "
                "Then you can decide whether you want to pay for a subscription. "
                "That's our idea of a free trial — helping you pay for your first subscription through free hits."
            ),
            inline=False,
        )
        em.add_field(
            name="Browse the Commands",
            value=(
                "`/fixtures` — see today's upcoming matches\n"
                "`/market` — get a prediction for any league and market (1 free per day)\n"
                "`/help` — full guide to every command and tier"
            ),
            inline=False,
        )
        em.add_field(
            name="Ready for More?",
            value=(
                "Run `/subscribe` to unlock full access — daily picks, parlays, "
                "player props, team lines, and more.\n"
                "Use `/referral` after subscribing to earn free months."
            ),
            inline=False,
        )
        em.set_footer(text="Spix's Picks | Powered by data, not gut feeling")

        try:
            await member.send(embed=em)
            log.info("Sent onboarding DM to new member %s", member)
        except discord.Forbidden:
            log.warning("Can't DM %s (DMs disabled)", member)
        except Exception as exc:
            log.warning("Failed to send onboarding DM to %s: %s", member, exc)

        # Notify bot owner about the new member
        if BOT_OWNER_ID:
            try:
                owner = await self.bot.fetch_user(BOT_OWNER_ID)
                await owner.send(
                    f"**New member joined:** {member} ({member.id})\n"
                    f"Server: {member.guild.name} | "
                    f"Members: {member.guild.member_count}"
                )
            except Exception as exc:
                log.debug("Could not notify owner about new member: %s", exc)


async def setup(bot: commands.Bot):
    await bot.add_cog(TierWelcome(bot))
