"""Subscription management cog — Stripe-backed plan selection, account info, cancellation.

Provides three slash commands:
    /subscribe  — display tier embed with plan selector, create Stripe Checkout
    /account    — show current subscription status (ephemeral)
    /cancel     — generate Stripe Customer Portal link for self-service cancellation
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import discord
from discord import app_commands
from discord.ext import commands

from config import COLOR_BLUE, COLOR_GREEN, COLOR_PURPLE, COLOR_YELLOW

log = logging.getLogger("subscribe")

# ---------------------------------------------------------------------------
# Lazy Stripe import — graceful degradation if stripe is not installed
# ---------------------------------------------------------------------------

try:
    import stripe

    stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
    _HAS_STRIPE = bool(stripe.api_key)
except ImportError:
    stripe = None  # type: ignore[assignment]
    _HAS_STRIPE = False

# ---------------------------------------------------------------------------
# Lazy user DB import — shared SQLite with the web app
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web_app"))
try:
    from users import get_user_by_discord_id, init_db, upsert_user

    _HAS_USERS = True
except ImportError:
    _HAS_USERS = False
    log.warning("users module not available — account/cancel commands will be limited")

# ---------------------------------------------------------------------------
# Price ID mapping from environment variables
# ---------------------------------------------------------------------------

TIER_PRICES: Dict[str, str] = {
    "starter_monthly": os.getenv("STRIPE_PRICE_STARTER_MONTHLY", ""),
    "starter_annual": os.getenv("STRIPE_PRICE_STARTER_ANNUAL", ""),
    "pro_monthly": os.getenv("STRIPE_PRICE_PRO_MONTHLY", ""),
    "pro_annual": os.getenv("STRIPE_PRICE_PRO_ANNUAL", ""),
    "elite_monthly": os.getenv("STRIPE_PRICE_ELITE_MONTHLY", ""),
    "elite_annual": os.getenv("STRIPE_PRICE_ELITE_ANNUAL", ""),
}

# Human-readable tier names for display
TIER_LABELS: Dict[str, str] = {
    "starter_monthly": "Rookie Monthly — $9.99/mo",
    "starter_annual": "Rookie Annual — $99/yr (save $21)",
    "pro_monthly": "Tipster Monthly — $14.99/mo",
    "pro_annual": "Tipster Annual — $149/yr (save $31)",
    "elite_monthly": "Whale Monthly — $29.99/mo",
    "elite_annual": "Whale Annual — $299/yr (save $61)",
}

# ---------------------------------------------------------------------------
# Stripe checkout helpers
# ---------------------------------------------------------------------------

_SUCCESS_URL = (
    os.getenv("STRIPE_SUCCESS_URL", "")
    or (
        os.getenv("DISCORD_REDIRECT_URI", "https://discord.com").rsplit("/", 1)[0]
        + "/checkout/success"
    )
)

_CANCEL_URL = (
    os.getenv("STRIPE_CANCEL_URL", "")
    or "https://discord.com"
)


def _create_checkout_session(
    price_id: str,
    discord_user_id: str,
    discord_username: str,
) -> Optional[str]:
    """Create a Stripe Checkout Session and return the URL, or None on failure."""
    if not _HAS_STRIPE or not price_id:
        return None
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            client_reference_id=str(discord_user_id),
            subscription_data={"trial_period_days": 7},
            success_url=_SUCCESS_URL + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=_CANCEL_URL,
            metadata={
                "discord_id": str(discord_user_id),
                "discord_username": discord_username,
            },
        )
        return session.url
    except Exception as exc:
        log.exception("Stripe checkout session creation failed: %s", exc)
        return None


def _create_portal_session(stripe_customer_id: str) -> Optional[str]:
    """Create a Stripe Billing Portal session and return the URL."""
    if not _HAS_STRIPE or not stripe_customer_id:
        return None
    try:
        session = stripe.billing_portal.Session.create(
            customer=stripe_customer_id,
            return_url=_CANCEL_URL,
        )
        return session.url
    except Exception as exc:
        log.exception("Stripe portal session creation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Subscribe embed builder
# ---------------------------------------------------------------------------


def _build_subscribe_embed() -> discord.Embed:
    """Build the rich embed showing all three tiers."""
    em = discord.Embed(
        title="Spick's Picks \u2014 Choose Your Plan",
        description=(
            "Start your **7-day free trial**. Cancel anytime.\n"
            "Select a plan below to get started."
        ),
        color=COLOR_PURPLE,
    )

    em.add_field(
        name="ROOKIE \u2014 $9.99/mo",
        value=(
            "\u25b8 5 domestic leagues\n"
            "\u25b8 Daily picks & match predictions\n"
            "\u25b8 Track record access\n"
            "\u25b8 5 bot commands/day"
        ),
        inline=False,
    )

    em.add_field(
        name="TIPSTER \u2014 $14.99/mo  \u2190 Most Popular",
        value=(
            "\u25b8 Everything in Rookie\n"
            "\u25b8 All 8 leagues (+ European comps)\n"
            "\u25b8 Parlays, player props, team lines\n"
            "\u25b8 Unlimited bot commands\n"
            "\u25b8 Website dashboard"
        ),
        inline=False,
    )

    em.add_field(
        name="WHALE \u2014 $29.99/mo",
        value=(
            "\u25b8 Everything in Tipster\n"
            "\u25b8 Pre-match lineup alerts\n"
            "\u25b8 CLV tracking dashboard\n"
            "\u25b8 Priority support thread\n"
            "\u25b8 Early access to new features"
        ),
        inline=False,
    )

    em.set_footer(text="Annual plans save up to 17%. Select a plan from the menu below.")
    return em


# ---------------------------------------------------------------------------
# Account embed builder
# ---------------------------------------------------------------------------


def _build_account_embed(user: Optional[Dict[str, Any]]) -> discord.Embed:
    """Build an embed showing the user's subscription status."""
    if user is None:
        em = discord.Embed(
            title="Account Status",
            description=(
                "No account found.\n\n"
                "Use `/subscribe` to choose a plan and get started."
            ),
            color=COLOR_BLUE,
        )
        return em

    tier = user.get("tier", "free")
    status = user.get("status", "inactive")
    is_founding = user.get("is_founding_member", 0)

    # Status color
    if status in ("active", "trialing"):
        color = COLOR_GREEN
    elif status == "cancelled":
        color = COLOR_YELLOW
    else:
        color = COLOR_BLUE

    em = discord.Embed(
        title="Account Status",
        color=color,
    )

    # Tier display
    tier_display = tier.capitalize() if tier != "free" else "Free"
    if is_founding:
        tier_display += "  \u2b50 Founding Member"

    em.add_field(name="Plan", value=tier_display, inline=True)
    em.add_field(name="Status", value=status.capitalize(), inline=True)

    # Renewal / expiry date
    sub_end = user.get("subscription_end")
    trial_end = user.get("trial_end")
    if status == "trialing" and trial_end:
        em.add_field(name="Trial Ends", value=trial_end[:10], inline=True)
    elif sub_end:
        label = "Expires" if status == "cancelled" else "Next Renewal"
        em.add_field(name=label, value=sub_end[:10], inline=True)

    # Member since
    created = user.get("created_at")
    if created:
        em.add_field(name="Member Since", value=created[:10], inline=True)

    if tier == "free" or status not in ("active", "trialing"):
        em.set_footer(text="Use /subscribe to choose a plan.")

    return em


# ---------------------------------------------------------------------------
# Tier select menu
# ---------------------------------------------------------------------------


class TierSelect(discord.ui.Select):
    """Dropdown menu for selecting a subscription plan."""

    def __init__(self) -> None:
        options = [
            discord.SelectOption(
                label="Rookie \u2014 $9.99/mo",
                value="starter_monthly",
                description="5 leagues, daily picks, 5 commands/day",
            ),
            discord.SelectOption(
                label="Rookie \u2014 $99/yr (save $21)",
                value="starter_annual",
                description="Same as Rookie Monthly, billed annually",
            ),
            discord.SelectOption(
                label="Tipster \u2014 $14.99/mo",
                value="pro_monthly",
                description="All 8 leagues, parlays, unlimited commands",
                default=True,
            ),
            discord.SelectOption(
                label="Tipster \u2014 $149/yr (save $31)",
                value="pro_annual",
                description="Same as Tipster Monthly, billed annually",
            ),
            discord.SelectOption(
                label="Whale \u2014 $29.99/mo",
                value="elite_monthly",
                description="Lineup alerts, CLV tracking, priority support",
            ),
            discord.SelectOption(
                label="Whale \u2014 $299/yr (save $61)",
                value="elite_annual",
                description="Same as Elite Monthly, billed annually",
            ),
        ]
        super().__init__(
            placeholder="Select a plan...",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        selected = self.values[0]
        price_id = TIER_PRICES.get(selected, "")

        if not _HAS_STRIPE:
            await interaction.response.send_message(
                "Subscriptions are not available yet. Stay tuned!",
                ephemeral=True,
            )
            return

        if not price_id:
            await interaction.response.send_message(
                "This plan is not configured yet. Please try another option.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=True)

        # Ensure user exists in DB before checkout
        if _HAS_USERS:
            upsert_user(
                discord_id=str(interaction.user.id),
                discord_username=str(interaction.user),
            )

        checkout_url = _create_checkout_session(
            price_id=price_id,
            discord_user_id=str(interaction.user.id),
            discord_username=str(interaction.user),
        )

        if checkout_url:
            label = TIER_LABELS.get(selected, selected)
            em = discord.Embed(
                title="Checkout Ready",
                description=(
                    f"**Plan:** {label}\n\n"
                    f"[Click here to complete checkout]({checkout_url})\n\n"
                    "You will have a **7-day free trial** before being charged.\n"
                    "Your Discord roles will be updated automatically after payment."
                ),
                color=COLOR_GREEN,
            )
            await interaction.followup.send(embed=em, ephemeral=True)
        else:
            await interaction.followup.send(
                "Failed to create checkout session. Please try again later.",
                ephemeral=True,
            )


class SubscribeView(discord.ui.View):
    """View containing the tier selection dropdown."""

    def __init__(self) -> None:
        super().__init__(timeout=300.0)
        self.add_item(TierSelect())

    async def on_timeout(self) -> None:
        # Disable the select menu after timeout
        for item in self.children:
            if isinstance(item, discord.ui.Select):
                item.disabled = True


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------


class SubscribeCog(commands.Cog, name="subscribe"):
    """Subscription management commands."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

        # Initialize user DB on cog load
        if _HAS_USERS:
            try:
                init_db()
            except Exception as exc:
                log.warning("Failed to initialize user DB: %s", exc)

    # ── /subscribe ────────────────────────────────────────────────────────

    @app_commands.command(
        name="subscribe",
        description="Choose a subscription plan and start your 7-day free trial",
    )
    async def subscribe_cmd(self, interaction: discord.Interaction) -> None:
        if not _HAS_STRIPE:
            em = discord.Embed(
                title="Coming Soon",
                description=(
                    "Subscriptions are not available yet.\n"
                    "Stay tuned for launch!"
                ),
                color=COLOR_BLUE,
            )
            await interaction.response.send_message(embed=em, ephemeral=True)
            return

        embed = _build_subscribe_embed()
        view = SubscribeView()
        await interaction.response.send_message(
            embed=embed,
            view=view,
            ephemeral=True,
        )

    # ── /account ──────────────────────────────────────────────────────────

    @app_commands.command(
        name="account",
        description="Check your current subscription status",
    )
    async def account_cmd(self, interaction: discord.Interaction) -> None:
        if not _HAS_USERS:
            await interaction.response.send_message(
                "Account system is not available yet.",
                ephemeral=True,
            )
            return

        user = get_user_by_discord_id(str(interaction.user.id))
        embed = _build_account_embed(user)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # ── /cancel ───────────────────────────────────────────────────────────

    @app_commands.command(
        name="cancel",
        description="Manage or cancel your subscription via Stripe",
    )
    async def cancel_cmd(self, interaction: discord.Interaction) -> None:
        if not _HAS_STRIPE:
            await interaction.response.send_message(
                "Subscriptions are not available yet.",
                ephemeral=True,
            )
            return

        if not _HAS_USERS:
            await interaction.response.send_message(
                "Account system is not available yet.",
                ephemeral=True,
            )
            return

        user = get_user_by_discord_id(str(interaction.user.id))

        if not user or not user.get("stripe_customer_id"):
            em = discord.Embed(
                title="No Subscription Found",
                description=(
                    "You don't have an active subscription.\n\n"
                    "Use `/subscribe` to choose a plan and get started."
                ),
                color=COLOR_BLUE,
            )
            await interaction.response.send_message(embed=em, ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)

        portal_url = _create_portal_session(user["stripe_customer_id"])

        if portal_url:
            em = discord.Embed(
                title="Manage Subscription",
                description=(
                    f"[Click here to manage your subscription]({portal_url})\n\n"
                    "From the portal you can:\n"
                    "\u25b8 Update payment method\n"
                    "\u25b8 Change your plan\n"
                    "\u25b8 Cancel your subscription\n"
                    "\u25b8 View billing history"
                ),
                color=COLOR_YELLOW,
            )
            await interaction.followup.send(embed=em, ephemeral=True)
        else:
            await interaction.followup.send(
                "Failed to open the billing portal. Please try again later.",
                ephemeral=True,
            )


# ---------------------------------------------------------------------------
# Cog loader
# ---------------------------------------------------------------------------


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(SubscribeCog(bot))
