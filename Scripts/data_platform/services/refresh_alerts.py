"""Local refresh watchdog; checked by existing scheduled workers, not the writer.

    REFRESH_ALERT_AFTER_SECONDS=3600
    REFRESH_ALERT_TELEGRAM_TOKEN=<private Telegram bot token, optional>
    REFRESH_ALERT_TELEGRAM_CHAT_ID=<the chat the bot writes to>
    REFRESH_ALERT_WEBHOOK_URL=<private Discord webhook, optional>

The ledger is beside the database (independent of its writer lock). Alerts
are rate-limited across processes; delivery failure cannot bypass the gate.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
from pathlib import Path
import time

from Scripts.refresh_runtime import positive_seconds

logger = logging.getLogger(__name__)
TELEGRAM = "https://api.telegram.org/bot{token}/sendMessage"


def _state_path(marker):
    return Path(str(marker) + ".alert.json")


def _read(path):
    try:
        state = json.loads(path.read_text())
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def blockage_status(marker, *, now=None):
    now = time.time() if now is None else now
    state = _read(_state_path(marker))
    since = state.get("blocked_since")
    try:
        marker_since = marker.stat().st_mtime
        since = min(since, marker_since) if since is not None else marker_since
    except FileNotFoundError:
        pass
    age = max(0, now - since) if since is not None else 0
    threshold = positive_seconds("REFRESH_ALERT_AFTER_SECONDS", 3600)
    return {"blocked": since is not None, "blocked_seconds": round(age),
            "prolonged": since is not None and age >= threshold,
            "threshold_seconds": threshold, "incomplete_marker": marker.exists(),
            "webhook_configured": configured(),
            "notification_delivered": state.get("notification_delivered", False),
            "alert_file": str(_state_path(marker))}


def _telegram():
    token = os.environ.get("REFRESH_ALERT_TELEGRAM_TOKEN")
    chat = os.environ.get("REFRESH_ALERT_TELEGRAM_CHAT_ID")
    return (token, chat) if token and chat else None


def configured():
    """Whether a private alert destination is set: a Telegram bot and chat, or a Discord webhook."""
    return bool(_telegram() or os.environ.get("REFRESH_ALERT_WEBHOOK_URL"))


def _post(url, payload, accepted):
    import requests
    try:
        with requests.post(url, json=payload, timeout=(3, 5)) as response:
            if 200 <= response.status_code < 300 and accepted(response):
                return True
            logger.warning("Private alert delivery refused (status %s); local alert retained", response.status_code)
            return False
    except requests.RequestException:
        # Bot and webhook tokens are in URLs; never log the exception/URL.
        logger.warning("Private alert delivery failed; local alert retained")
        return False


def _telegram_ok(response):
    try:
        return response.json().get("ok") is True
    except (ValueError, AttributeError):
        return False


def _send(message):
    """Delivers to every configured destination; True when at least one took it."""
    delivered = False
    if telegram := _telegram():
        token, chat = telegram
        delivered |= _post(TELEGRAM.format(token=token),
                           {"chat_id": chat, "text": message, "disable_web_page_preview": True}, _telegram_ok)
    if destination := os.environ.get("REFRESH_ALERT_WEBHOOK_URL"):
        delivered |= _post(destination, {"content": message, "allowed_mentions": {"parse": []}}, lambda r: True)
    return delivered


def observe_gate(marker, *, blocked, now=None, sender=None):
    """One observation per scheduled gate attempt, plus a recovery event.

    An old incomplete marker keeps the first failure's age across retries.
    A lock held without a marker is timed from the first blocked observation.
    """
    now = time.time() if now is None else now
    sender = sender or _send
    path = _state_path(marker)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with Path(str(path) + ".lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            state = _read(path)
            if blocked:
                since = state.get("blocked_since", now)
                if marker.exists():
                    since = min(since, marker.stat().st_mtime)
                state.update(blocked_since=since, last_seen=now)
                age = max(0, now - since)
                if age >= positive_seconds("REFRESH_ALERT_AFTER_SECONDS", 3600):
                    message = (f"Spix refresh blocked for {int(age // 60)} minutes. "
                               "Match Reads and prediction measurement cannot resume until a full refresh succeeds. "
                               "Check daily-data-refresh logs; do not delete the safety gate.")
                    if "last_alert_at" not in state or now - state["last_alert_at"] >= 21600:
                        logger.error(message)
                        state.update(last_alert_at=now, notification_delivered=False, message=message)
                    if not state.get("notification_delivered") and ("last_attempt_at" not in state or now - state["last_attempt_at"] >= 300):
                        state["notification_delivered"] = bool(sender(message))
                        state["last_attempt_at"] = now
            else:
                if not state.get("blocked_since") and not state.get("recovery_pending"):
                    return
                if state.get("notification_delivered"):
                    # Keep recovery pending if transport is down; retry on
                    # the next healthy tick instead of silently losing it.
                    if now - state.get("last_attempt_at", 0) < 300 and state.get("recovery_pending"):
                        return
                    message = "Spix refresh recovered: the data gate is open; scheduled workers may resume."
                    if not sender(message):
                        state.update(recovery_pending=True, last_attempt_at=now)
                    else:
                        state = {"recovered_at": now}
                else:
                    state = {"recovered_at": now}
                logger.info("Refresh gate recovered")
                # A pending recovery notification is not an active blockage.
                state.pop("blocked_since", None)
            temporary = path.with_suffix(f".tmp.{os.getpid()}")
            temporary.write_text(json.dumps(state, indent=2))
            temporary.replace(path)
    except (OSError, ValueError):
        logger.warning("Could not persist refresh alert state; inspect refresh logs")
