"""
Authentication — Discord OAuth2 + email/password + JWT.

Handles Discord login flow, email signup/login, JWT token issuance,
and auth dependencies for protecting endpoints by tier.

Env vars:
    DISCORD_CLIENT_ID, DISCORD_CLIENT_SECRET, DISCORD_REDIRECT_URI,
    JWT_SECRET, FRONTEND_URL
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time

from dotenv import load_dotenv
load_dotenv()
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx
import jwt as pyjwt
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr

from users import (
    upsert_user, get_user_by_discord_id, get_user_by_email,
    get_user_by_id, create_email_user, init_db,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

DISCORD_CLIENT_ID: str = os.getenv("DISCORD_CLIENT_ID", "")
DISCORD_CLIENT_SECRET: str = os.getenv("DISCORD_CLIENT_SECRET", "")
DISCORD_REDIRECT_URI: str = os.getenv("DISCORD_REDIRECT_URI", "http://localhost:8000/auth/discord/callback")
JWT_SECRET: str = os.getenv("JWT_SECRET", "")
FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:8000")

# Fail loudly if JWT_SECRET is not configured or is the insecure default
if not JWT_SECRET or JWT_SECRET == "change-me-in-production":
    raise RuntimeError(
        "JWT_SECRET must be set to a strong random value in the environment. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )

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

# In-memory stores for OAuth state and one-time auth codes
# In production, use Redis or a DB table with TTL
_oauth_states: Dict[str, float] = {}   # state -> created_at
_auth_codes: Dict[str, Dict] = {}      # code -> {jwt, created_at}
_STATE_TTL = 600       # 10 minutes
_CODE_TTL = 120        # 2 minutes


def _cleanup_expired(store: Dict[str, Any], ttl: int) -> None:
    """Remove expired entries from an in-memory TTL store."""
    now = time.time()
    expired = [k for k, v in store.items()
               if now - (v if isinstance(v, (int, float)) else v.get("created_at", 0)) > ttl]
    for k in expired:
        store.pop(k, None)

# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def create_jwt(
    user_id: str,
    username: str,
    tier: str = "free",
    status: str = "active",
    auth_provider: str = "discord",
) -> str:
    """Create a signed JWT token with user claims."""
    now = int(time.time())
    payload = {
        "sub": user_id,
        "username": username,
        "tier": tier,
        "status": status,
        "auth_provider": auth_provider,
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

    subject = payload.get("sub")
    if not subject:
        return None

    auth_provider = payload.get("auth_provider", "discord")

    # Fetch fresh user record from DB (tier/status may have changed)
    if auth_provider == "email":
        # For email users, sub is the user ID
        try:
            user = get_user_by_id(int(subject))
        except (ValueError, TypeError):
            return None
    else:
        # For Discord users, sub is the discord_id
        user = get_user_by_discord_id(subject)

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
    _cleanup_expired(_oauth_states, _STATE_TTL)
    state = secrets.token_urlsafe(32)
    _oauth_states[state] = time.time()
    params = urlencode({
        "client_id": DISCORD_CLIENT_ID,
        "redirect_uri": DISCORD_REDIRECT_URI,
        "response_type": "code",
        "scope": OAUTH_SCOPES,
        "state": state,
    })
    return RedirectResponse(url=f"{DISCORD_AUTHORIZE_URL}?{params}")


@router.get("/discord/callback")
async def discord_callback(code: str, state: str = ""):
    """
    Handle the OAuth2 callback from Discord.

    Exchanges the authorization code for an access token, fetches the user's
    profile, upserts into the DB, generates a JWT, and redirects to the
    frontend with a short-lived one-time code (NOT the JWT itself).
    """
    # 0. Validate OAuth state to prevent CSRF
    if not state or state not in _oauth_states:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")
    state_created = _oauth_states.pop(state)
    if time.time() - state_created > _STATE_TTL:
        raise HTTPException(status_code=400, detail="OAuth state expired")

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
            log.error(
                "Discord token exchange failed: %s %s",
                token_resp.status_code,
                token_resp.text,
            )
            raise HTTPException(
                status_code=400,
                detail="Discord authentication failed. Please try again.",
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
        user_id=discord_id,
        username=username,
        tier=tier,
        status=status,
        auth_provider="discord",
    )

    # 6. Issue a one-time code (not the JWT itself) to avoid token leakage in URL
    _cleanup_expired(_auth_codes, _CODE_TTL)
    auth_code = secrets.token_urlsafe(48)
    _auth_codes[auth_code] = {"jwt": token, "created_at": time.time()}

    return RedirectResponse(url=f"{FRONTEND_URL}/app?code={auth_code}", status_code=302)


@router.post("/exchange")
async def exchange_code(request: Request):
    """
    Exchange a one-time auth code for a JWT token.

    The frontend calls this after the OAuth redirect to securely obtain
    the JWT without it appearing in the URL or browser history.
    """
    body = await request.json()
    auth_code = body.get("code", "")

    entry = _auth_codes.pop(auth_code, None)
    if not entry:
        raise HTTPException(status_code=400, detail="Invalid or expired auth code")
    if time.time() - entry["created_at"] > _CODE_TTL:
        raise HTTPException(status_code=400, detail="Auth code expired")

    return {"token": entry["jwt"]}


@router.get("/me")
async def get_me(user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Return the current user's profile (refreshed from DB)."""
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    provider = user.get("auth_provider", "discord")
    username = user.get("discord_username") or user.get("email", "").split("@")[0]
    return {
        "id": user.get("id"),
        "discord_id": user.get("discord_id"),
        "username": username,
        "avatar_url": user.get("discord_avatar_url"),
        "email": user.get("email"),
        "auth_provider": provider,
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


# ---------------------------------------------------------------------------
# Email/password authentication
# ---------------------------------------------------------------------------

_HASH_ITERATIONS = 260_000  # OWASP recommendation for PBKDF2-SHA256


def _hash_password(password: str) -> str:
    """Hash a password using PBKDF2-SHA256 with a random salt."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _HASH_ITERATIONS)
    return f"{salt}${dk.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored PBKDF2-SHA256 hash."""
    try:
        salt, dk_hex = stored_hash.split("$", 1)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _HASH_ITERATIONS)
        return secrets.compare_digest(dk.hex(), dk_hex)
    except (ValueError, AttributeError):
        return False


class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/signup")
async def email_signup(body: SignupRequest):
    """
    Create a new account with email and password.

    Returns a JWT token on success.
    """
    email = body.email.strip().lower()
    password = body.password

    # Basic validation
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email address")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    # Check if email is already taken
    existing = get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    # Create user
    pw_hash = _hash_password(password)
    user = create_email_user(email, pw_hash)
    if not user:
        raise HTTPException(status_code=500, detail="Failed to create account")

    # Generate JWT
    token = create_jwt(
        user_id=str(user["id"]),
        username=email.split("@")[0],
        tier=user.get("tier", "free"),
        status=user.get("status", "active"),
        auth_provider="email",
    )

    return {"token": token}


@router.post("/login")
async def email_login(body: LoginRequest):
    """
    Log in with email and password.

    Returns a JWT token on success.
    """
    email = body.email.strip().lower()
    password = body.password

    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    stored_hash = user.get("password_hash", "")
    if not _verify_password(password, stored_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_jwt(
        user_id=str(user["id"]),
        username=email.split("@")[0],
        tier=user.get("tier", "free"),
        status=user.get("status", "active"),
        auth_provider="email",
    )

    return {"token": token}
