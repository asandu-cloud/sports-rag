"""
Stripe webhook handler + checkout session creation — FastAPI APIRouter.

Handles subscription lifecycle events from Stripe and exposes endpoints
for creating Checkout Sessions and Customer Portal sessions.

Env vars:
    STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET, STRIPE_PUBLISHABLE_KEY,
    FOUNDING_MEMBER_ACTIVE, FRONTEND_URL,
    STRIPE_PRICE_STARTER_MONTHLY, STRIPE_PRICE_STARTER_ANNUAL,
    STRIPE_PRICE_PRO_MONTHLY, STRIPE_PRICE_PRO_ANNUAL,
    STRIPE_PRICE_ELITE_MONTHLY, STRIPE_PRICE_ELITE_ANNUAL
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from users import (
    get_user_by_discord_id,
    get_user_by_stripe_customer,
    update_user_subscription,
    cancel_user_subscription,
    log_subscription_event,
)
from discord_roles import assign_tier_role, remove_all_tier_roles
from auth import get_current_user, require_auth

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PUBLISHABLE_KEY: str = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:8000")
FOUNDING_MEMBER_ACTIVE: bool = os.getenv("FOUNDING_MEMBER_ACTIVE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Price-to-tier mapping — populated from env vars at module load
# ---------------------------------------------------------------------------

PRICE_TO_TIER: Dict[str, str] = {}
for _tier, _suffixes in [("starter", ["STARTER"]), ("pro", ["PRO"]), ("elite", ["ELITE"])]:
    for _interval in ["MONTHLY", "ANNUAL"]:
        _price_id = os.getenv(f"STRIPE_PRICE_{_suffixes[0]}_{_interval}", "")
        if _price_id:
            PRICE_TO_TIER[_price_id] = _tier

# Reverse lookup: tier + interval -> price_id
TIER_TO_PRICE: Dict[str, str] = {}
for _tier, _suffixes in [("starter", ["STARTER"]), ("pro", ["PRO"]), ("elite", ["ELITE"])]:
    for _interval in ["MONTHLY", "ANNUAL"]:
        _price_id = os.getenv(f"STRIPE_PRICE_{_suffixes[0]}_{_interval}", "")
        if _price_id:
            TIER_TO_PRICE[f"{_tier}_{_interval.lower()}"] = _price_id

# Track processed event IDs to handle duplicates gracefully
_processed_events: set = set()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CheckoutRequest(BaseModel):
    tier: str  # "starter", "pro", "elite"
    interval: str  # "monthly", "annual"


# ---------------------------------------------------------------------------
# Router — webhooks
# ---------------------------------------------------------------------------

webhook_router = APIRouter(tags=["webhooks"])


@webhook_router.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    """
    Handle incoming Stripe webhook events.

    IMPORTANT: Uses raw request body for signature verification.
    Returns 200 for all events (even unhandled) to prevent Stripe retries.
    """
    # Read raw body — Stripe signature verification requires raw bytes
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    # Verify webhook signature
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        log.error("Stripe webhook: invalid payload")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        log.error("Stripe webhook: invalid signature")
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_id = event.get("id", "unknown")
    event_type = event.get("type", "unknown")

    # Deduplicate — Stripe may retry delivery
    if event_id in _processed_events:
        log.info("Stripe webhook: duplicate event %s, skipping", event_id)
        return {"status": "ok", "message": "duplicate event"}
    _processed_events.add(event_id)

    # Cap the in-memory set to prevent unbounded growth
    if len(_processed_events) > 10000:
        _processed_events.clear()

    log.info("Stripe webhook: %s (%s)", event_type, event_id)

    # Log every event to the subscription_events table
    _log_event_safe(event_id, event_type, event)

    # Dispatch to handler
    try:
        if event_type == "checkout.session.completed":
            await _handle_checkout_completed(event)
        elif event_type == "invoice.paid":
            await _handle_invoice_paid(event)
        elif event_type == "customer.subscription.updated":
            await _handle_subscription_updated(event)
        elif event_type == "customer.subscription.deleted":
            await _handle_subscription_deleted(event)
        elif event_type == "invoice.payment_failed":
            await _handle_payment_failed(event)
        else:
            log.debug("Stripe webhook: unhandled event type %s", event_type)
    except Exception:
        # Log but still return 200 to prevent retries for code errors
        log.exception("Stripe webhook handler error for %s", event_type)

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Webhook event handlers
# ---------------------------------------------------------------------------


async def _handle_checkout_completed(event: Dict[str, Any]) -> None:
    """Handle checkout.session.completed — new subscription created."""
    session = event["data"]["object"]
    discord_id = session.get("client_reference_id")
    customer_id = session.get("customer")
    subscription_id = session.get("subscription")

    if not discord_id:
        log.warning("checkout.session.completed: no client_reference_id, skipping")
        return

    if not subscription_id:
        log.warning("checkout.session.completed: no subscription_id (one-time payment?), skipping")
        return

    # Retrieve subscription to get price and trial info
    subscription = stripe.Subscription.retrieve(subscription_id)
    price_id = subscription["items"]["data"][0]["price"]["id"]
    tier = PRICE_TO_TIER.get(price_id, "starter")

    trial_end = subscription.get("trial_end")
    status = "trial" if trial_end else "active"

    is_founding = 1 if FOUNDING_MEMBER_ACTIVE else 0

    update_user_subscription(
        discord_id=discord_id,
        tier=tier,
        status=status,
        stripe_customer_id=customer_id,
        stripe_subscription_id=subscription_id,
        trial_end=trial_end,
        is_founding_member=is_founding,
    )

    log.info(
        "checkout.session.completed: discord=%s tier=%s status=%s founding=%s",
        discord_id, tier, status, is_founding,
    )

    # Assign Discord role
    await assign_tier_role(discord_id, tier)


async def _handle_invoice_paid(event: Dict[str, Any]) -> None:
    """Handle invoice.paid — successful recurring payment."""
    invoice = event["data"]["object"]
    customer_id = invoice.get("customer")

    if not customer_id:
        return

    user = get_user_by_stripe_customer(customer_id)
    if not user:
        log.warning("invoice.paid: no user found for customer %s", customer_id)
        return

    discord_id = user["discord_id"]

    # Extract subscription period end from invoice line items
    subscription_end = None
    lines = invoice.get("lines", {}).get("data", [])
    if lines:
        period = lines[0].get("period", {})
        subscription_end = period.get("end")

    update_user_subscription(
        discord_id=discord_id,
        tier=user.get("tier", "starter"),
        status="active",
        subscription_end=subscription_end,
    )

    log.info("invoice.paid: discord=%s renewed until %s", discord_id, subscription_end)

    # Ensure Discord role is assigned (in case it was removed during past_due)
    await assign_tier_role(discord_id, user.get("tier", "starter"))


async def _handle_subscription_updated(event: Dict[str, Any]) -> None:
    """Handle customer.subscription.updated — tier or status changes."""
    subscription = event["data"]["object"]
    customer_id = subscription.get("customer")

    if not customer_id:
        return

    user = get_user_by_stripe_customer(customer_id)
    if not user:
        log.warning("subscription.updated: no user found for customer %s", customer_id)
        return

    discord_id = user["discord_id"]

    # Determine new tier from price
    price_id = subscription["items"]["data"][0]["price"]["id"]
    new_tier = PRICE_TO_TIER.get(price_id, user.get("tier", "starter"))

    # Map Stripe status to our status
    stripe_status = subscription.get("status", "active")
    status_map = {
        "active": "active",
        "trialing": "trial",
        "past_due": "past_due",
        "canceled": "cancelled",
        "unpaid": "past_due",
        "incomplete": "past_due",
        "incomplete_expired": "expired",
    }
    new_status = status_map.get(stripe_status, "active")

    old_tier = user.get("tier", "free")

    update_user_subscription(
        discord_id=discord_id,
        tier=new_tier,
        status=new_status,
    )

    log.info(
        "subscription.updated: discord=%s tier=%s->%s status=%s",
        discord_id, old_tier, new_tier, new_status,
    )

    # Swap Discord roles if tier changed
    if new_tier != old_tier:
        await remove_all_tier_roles(discord_id)
        await assign_tier_role(discord_id, new_tier)


async def _handle_subscription_deleted(event: Dict[str, Any]) -> None:
    """Handle customer.subscription.deleted — subscription cancelled."""
    subscription = event["data"]["object"]
    customer_id = subscription.get("customer")

    if not customer_id:
        return

    user = get_user_by_stripe_customer(customer_id)
    if not user:
        log.warning("subscription.deleted: no user found for customer %s", customer_id)
        return

    discord_id = user["discord_id"]

    cancel_user_subscription(discord_id)
    log.info("subscription.deleted: discord=%s cancelled", discord_id)

    # Remove all tier roles
    await remove_all_tier_roles(discord_id)


async def _handle_payment_failed(event: Dict[str, Any]) -> None:
    """Handle invoice.payment_failed — mark subscription as past_due."""
    invoice = event["data"]["object"]
    customer_id = invoice.get("customer")

    if not customer_id:
        return

    user = get_user_by_stripe_customer(customer_id)
    if not user:
        log.warning("payment_failed: no user found for customer %s", customer_id)
        return

    discord_id = user["discord_id"]

    update_user_subscription(
        discord_id=discord_id,
        tier=user.get("tier", "starter"),
        status="past_due",
    )

    log.info("payment_failed: discord=%s marked past_due", discord_id)


# ---------------------------------------------------------------------------
# Helper — safe event logging
# ---------------------------------------------------------------------------


def _log_event_safe(event_id: str, event_type: str, event: Dict[str, Any]) -> None:
    """Log a Stripe event to the DB, suppressing errors."""
    try:
        # Extract discord_id or customer_id for reference
        data_obj = event.get("data", {}).get("object", {})
        discord_id = data_obj.get("client_reference_id") or ""
        customer_id = data_obj.get("customer") or ""

        if not discord_id and customer_id:
            user = get_user_by_stripe_customer(customer_id)
            if user:
                discord_id = user.get("discord_id", "")

        log_subscription_event(
            stripe_event_id=event_id,
            event_type=event_type,
            discord_id=discord_id,
            data=event,
        )
    except Exception:
        log.debug("Failed to log subscription event %s", event_id, exc_info=True)


# ---------------------------------------------------------------------------
# Router — checkout endpoints
# ---------------------------------------------------------------------------

checkout_router = APIRouter(prefix="/api/checkout", tags=["checkout"])


@checkout_router.post("/create-session")
async def create_checkout_session(
    body: CheckoutRequest,
    user: Dict[str, Any] = Depends(require_auth),
):
    """
    Create a Stripe Checkout Session for a new subscription.

    Request body: {"tier": "pro", "interval": "monthly"}
    Returns: {"url": "https://checkout.stripe.com/..."}
    """
    lookup_key = f"{body.tier}_{body.interval}"
    price_id = TIER_TO_PRICE.get(lookup_key)

    if not price_id:
        raise HTTPException(
            status_code=400,
            detail=f"No price configured for tier='{body.tier}' interval='{body.interval}'. "
                   f"Available: {list(TIER_TO_PRICE.keys())}",
        )

    discord_id = user["discord_id"]
    email = user.get("email")

    # Check if user already has a Stripe customer ID (returning customer)
    existing_customer_id = user.get("stripe_customer_id")

    # Build checkout session params
    session_params: Dict[str, Any] = {
        "payment_method_types": ["card"],
        "mode": "subscription",
        "client_reference_id": discord_id,
        "line_items": [{"price": price_id, "quantity": 1}],
        "success_url": f"{FRONTEND_URL}/checkout/success?session_id={{CHECKOUT_SESSION_ID}}",
        "cancel_url": f"{FRONTEND_URL}/#pricing",
    }

    # Attach existing customer or set email for new customers
    if existing_customer_id:
        session_params["customer"] = existing_customer_id
    elif email:
        session_params["customer_email"] = email

    # Offer 7-day trial only for new customers (no existing Stripe customer)
    if not existing_customer_id:
        session_params["subscription_data"] = {"trial_period_days": 7}

    try:
        session = stripe.checkout.Session.create(**session_params)
    except stripe.error.StripeError as exc:
        log.error("Stripe checkout session creation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create checkout session")

    return {"url": session.url}


@checkout_router.post("/portal")
async def create_portal_session(
    user: Dict[str, Any] = Depends(require_auth),
):
    """
    Create a Stripe Customer Portal session for managing subscriptions.

    Returns: {"url": "https://billing.stripe.com/..."}
    """
    customer_id = user.get("stripe_customer_id")
    if not customer_id:
        raise HTTPException(
            status_code=400,
            detail="No active subscription found. Subscribe first.",
        )

    try:
        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"{FRONTEND_URL}/app",
        )
    except stripe.error.StripeError as exc:
        log.error("Stripe portal session creation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create portal session")

    return {"url": portal_session.url}
