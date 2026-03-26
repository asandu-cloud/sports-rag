"""
Discord OAuth2 + JWT authentication — FastAPI APIRouter.

Handles Discord login flow, JWT token issuance, and auth dependencies
for protecting endpoints by tier.

Env vars:
    DISCORD_CLIENT_ID, DISCORD_CLIENT_SECRET, DISCORD_REDIRECT_URI,
    JWT_SECRET, FRONTEND_URL
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
load_dotenv()
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx
import jwt as pyjwt
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import RedirectResponse

from users import upsert_user, get_user_by_discord_id, init_db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

DISCORD_CLIENT_ID: str = os.getenv("DISCORD_CLIENT_ID", "")
DISCORD_CLIENT_SECRET: str = os.getenv("DISCORD_CLIENT_SECRET", "")
DISCORD_REDIRECT_URI: str = os.getenv("DISCORD_REDIRECT_URI", "http://localhost:8000/auth/discord/callback")
JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me-in-production")
FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:8000")

JWT_ALGORITHM = "HS256"
JWT_EXPIRY_SECONDS = 86400  # 24 hours

DISCORD_API = "https://discord.com/api/v10"
DISCORD_AUTHORIZE_URL = "https://discord.com/api/oauth2/authorize"
DISCORD_TOKEN_URL = "https://discord.com/api/oauth2/token"

OAUTH_SCOPES = "identify email"

# Tier hierarchy (higher index = higher tier)
TIER_HIERARCHY: Dict[str, int] = {
    "free": 0,
    "starter": 1,
    "pro": 2,
    "elite": 3,
}

# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def create_jwt(
    discord_id: str,
    username: str,
    tier: str = "free",
    status: str = "active",
) -> str:
    """Create a signed JWT token with user claims."""
    now = int(time.time())
    payload = {
        "sub": discord_id,
        "username": username,
        "tier": tier,
        "status": status,
        "iat": now,
        "exp": now + JWT_EXPIRY_SECONDS,
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT token. Returns payload dict or None."""
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except pyjwt.ExpiredSignatureError:
        log.debug("JWT expired")
        return None
    except pyjwt.InvalidTokenError as exc:
        log.debug("JWT invalid: %s", exc)
        return None


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


async def get_current_user(
    authorization: Optional[str] = Header(None),
) -> Optional[Dict[str, Any]]:
    """
    Extract the current user from the Authorization header.

    Returns the full user dict from the DB (with refreshed tier/status),
    or None if the token is missing/invalid.  Does NOT raise 401 — this
    allows unauthenticated access on free-tier endpoints.
    """
    if not authorization:
        return None

    # Support "Bearer <token>" format
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1]
    elif len(parts) == 1:
        token = parts[0]
    else:
        return None

    payload = decode_jwt(token)
    if payload is None:
        return None

    discord_id = payload.get("sub")
    if not discord_id:
        return None

    # Fetch fresh user record from DB (tier/status may have changed)
    user = get_user_by_discord_id(discord_id)
    if user is None:
        return None

    return user


async def require_auth(
    user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Dependency that raises 401 if not authenticated."""
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def require_tier(minimum: str):
    """
    Dependency factory that checks the user meets a minimum tier.

    Usage:
        @router.get("/premium-endpoint")
        async def endpoint(user = Depends(require_tier("pro"))):
            ...
    """
    min_level = TIER_HIERARCHY.get(minimum, 0)

    async def _check_tier(
        user: Dict[str, Any] = Depends(require_auth),
    ) -> Dict[str, Any]:
        user_level = TIER_HIERARCHY.get(user.get("tier", "free"), 0)
        if user_level < min_level:
            raise HTTPException(
                status_code=403,
                detail=f"This endpoint requires '{minimum}' tier or above. "
                       f"Your current tier: '{user.get('tier', 'free')}'",
            )
        # Also check subscription status
        status = user.get("status", "active")
        if status in ("cancelled", "past_due", "expired"):
            raise HTTPException(
                status_code=403,
                detail=f"Your subscription status is '{status}'. "
                       "Please update your payment to continue.",
            )
        return user

    return _check_tier


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/discord/login")
async def discord_login():
    """Redirect the user to Discord's OAuth2 authorize page."""
    params = urlencode({
        "client_id": DISCORD_CLIENT_ID,
        "redirect_uri": DISCORD_REDIRECT_URI,
        "response_type": "code",
        "scope": OAUTH_SCOPES,
    })
    return RedirectResponse(url=f"{DISCORD_AUTHORIZE_URL}?{params}")


@router.get("/discord/callback")
async def discord_callback(code: str):
    """
    Handle the OAuth2 callback from Discord.

    Exchanges the authorization code for an access token, fetches the user's
    profile, upserts into the DB, generates a JWT, and redirects to the
    frontend with the token as a query parameter.
    """
    # 1. Exchange code for access token
    token_data = {
        "client_id": DISCORD_CLIENT_ID,
        "client_secret": DISCORD_CLIENT_SECRET,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": DISCORD_REDIRECT_URI,
    }

    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            DISCORD_TOKEN_URL,
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if token_resp.status_code != 200:
            error_body = token_resp.text
            log.error(
                "Discord token exchange failed: %s %s",
                token_resp.status_code,
                error_body,
            )
            raise HTTPException(
                status_code=400,
                detail=f"Discord OAuth failed: {error_body[:200]}",
            )

        token_json = token_resp.json()
        access_token = token_json.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="No access token received from Discord")

        # 2. Fetch user info from Discord
        user_resp = await client.get(
            f"{DISCORD_API}/users/@me",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        if user_resp.status_code != 200:
            log.error(
                "Discord user fetch failed: %s %s",
                user_resp.status_code,
                user_resp.text,
            )
            raise HTTPException(status_code=400, detail="Failed to fetch Discord user info")

        discord_user = user_resp.json()

    discord_id = discord_user["id"]
    username = discord_user.get("username", "unknown")
    email = discord_user.get("email")

    # Build avatar URL
    avatar_hash = discord_user.get("avatar")
    if avatar_hash:
        avatar_url = f"https://cdn.discordapp.com/avatars/{discord_id}/{avatar_hash}.png"
    else:
        # Default Discord avatar
        discriminator = discord_user.get("discriminator", "0")
        default_idx = int(discriminator) % 5 if discriminator != "0" else int(discord_id) >> 22 % 6
        avatar_url = f"https://cdn.discordapp.com/embed/avatars/{default_idx}.png"

    # 3. Upsert user in DB
    upsert_user(
        discord_id=discord_id,
        discord_username=username,
        discord_avatar_url=avatar_url,
        email=email,
    )

    # 4. Get current tier/status from DB
    user = get_user_by_discord_id(discord_id)
    tier = user.get("tier", "free") if user else "free"
    status = user.get("status", "active") if user else "active"

    # 5. Generate JWT
    token = create_jwt(
        discord_id=discord_id,
        username=username,
        tier=tier,
        status=status,
    )

    # 6. Redirect to frontend with token
    return RedirectResponse(url=f"{FRONTEND_URL}/app?token={token}", status_code=302)


@router.get("/me")
async def get_me(user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Return the current user's profile (refreshed from DB)."""
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {
        "discord_id": user.get("discord_id"),
        "username": user.get("username"),
        "avatar_url": user.get("avatar_url"),
        "email": user.get("email"),
        "tier": user.get("tier", "free"),
        "status": user.get("status", "active"),
        "is_founding_member": bool(user.get("is_founding_member", 0)),
        "trial_end": user.get("trial_end"),
        "subscription_end": user.get("subscription_end"),
        "created_at": user.get("created_at"),
    }


@router.post("/logout")
async def logout():
    """
    Logout endpoint — returns success.

    Actual token invalidation is handled client-side by deleting
    the JWT from localStorage. Server-side token blocklisting can
    be added later if needed.
    """
    return {"ok": True, "message": "Logged out — clear token on client"}
