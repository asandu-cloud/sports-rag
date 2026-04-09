"""Live alerts cog — polls for live match data and posts alerts to Discord.

Runs the LivePoller within the bot's asyncio event loop and uses the
live engine + alert detector to generate and post alerts.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Set

import discord
from discord.ext import commands, tasks

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "Scripts" / "live"))
sys.path.insert(0, str(_ROOT / "Scripts" / "rag_ingest"))
sys.path.insert(0, str(_ROOT / "Scripts"))

# ---------------------------------------------------------------------------
# Bot config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CHANNEL_LIVE_ALERTS, CHANNEL_LIVE_SCORES, LIVE_POLL_INTERVAL

# ---------------------------------------------------------------------------
# Embed builders
# ---------------------------------------------------------------------------
from embeds import (
    full_time_embed,
    goal_alert_embed,
    live_alert_embed,
    match_start_embed,
)
from live.live_runtime import get_live_poller

log = logging.getLogger("live_bot.alerts_cog")


class LiveAlertsCog(commands.Cog):
    """Cog that runs the live polling loop and posts alerts to Discord channels."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.poller = None  # LivePoller instance, initialized in cog_load
        self._last_alert_snapshot_ts: Dict[int, str] = {}
        self._known_fixtures: Set[int] = set()  # fixtures we have posted "match started" for
        self._prev_event_counts: Dict[int, int] = {}  # fixture_id -> event count at last check
        self._finished: Set[int] = set()  # fixtures we have posted "full time" for

    async def cog_load(self) -> None:
        """Initialize the LivePoller and start the alert loop."""
        try:
            self.poller = get_live_poller(auto_subscribe=True)
            await self.poller.start()
            log.info("LivePoller started inside alerts cog")
        except Exception as exc:
            log.error("Failed to start LivePoller: %s", exc, exc_info=True)
            self.poller = None
            return

        self.alert_loop.start()
        log.info("Alert loop started (interval=%ds)", LIVE_POLL_INTERVAL)

    async def cog_unload(self) -> None:
        """Stop the alert loop and clean up the poller."""
        self.alert_loop.cancel()
        if self.poller:
            await self.poller.stop()
            log.info("LivePoller stopped")

    # ------------------------------------------------------------------
    # Alert loop
    # ------------------------------------------------------------------

    @tasks.loop(seconds=LIVE_POLL_INTERVAL)
    async def alert_loop(self) -> None:
        """Main polling tick: check each tracked fixture for alerts."""
        if self.poller is None:
            return

        alerts_channel = self._get_channel(CHANNEL_LIVE_ALERTS)
        scores_channel = self._get_channel(CHANNEL_LIVE_SCORES)

        active = self.poller.get_active_fixtures()

        for fixture_info in active:
            fid = fixture_info.get("fixture_id")
            if fid is None:
                continue

            state = self.poller.get_state(fid)
            snapshot = self.poller.get_snapshot(fid)
            if state is None:
                continue

            # --- Match started ---
            if fid not in self._known_fixtures:
                self._known_fixtures.add(fid)
                if scores_channel:
                    try:
                        em = match_start_embed(state)
                        await scores_channel.send(embed=em)
                    except Exception as exc:
                        log.warning("Failed to post match start for %d: %s", fid, exc)

            # --- Goal detection ---
            await self._check_for_goals(fid, state, snapshot, scores_channel)

            # --- Alert posting (already computed inside the poller) ---
            if snapshot is not None:
                snapshot_ts = str(getattr(snapshot, "timestamp", "") or "")
                if snapshot_ts and snapshot_ts != self._last_alert_snapshot_ts.get(fid):
                    alerts = list(getattr(snapshot, "alerts", []) or [])
                    for alert in alerts:
                        if alerts_channel:
                            try:
                                em = live_alert_embed(alert, snapshot, state)
                                await alerts_channel.send(embed=em)
                            except Exception as exc:
                                log.warning("Failed to post alert for %d: %s", fid, exc)
                    self._last_alert_snapshot_ts[fid] = snapshot_ts

        # --- Full time detection ---
        archived_ids = getattr(self.poller, "archived", set())
        for fid in archived_ids:
            if fid in self._finished:
                continue
            self._finished.add(fid)
            state = self.poller.get_state(fid)
            snapshot = self.poller.get_snapshot(fid)
            if state and scores_channel:
                try:
                    em = full_time_embed(snapshot, state)
                    await scores_channel.send(embed=em)
                except Exception as exc:
                    log.warning("Failed to post full time for %d: %s", fid, exc)

    @alert_loop.before_loop
    async def _before_alert_loop(self) -> None:
        """Wait until the bot is ready before starting the loop."""
        await self.bot.wait_until_ready()

    # ------------------------------------------------------------------
    # Goal detection
    # ------------------------------------------------------------------

    async def _check_for_goals(
        self,
        fid: int,
        state: Any,
        snapshot: Any,
        channel: Optional[discord.TextChannel],
    ) -> None:
        """Detect new goal events by comparing event counts."""
        if channel is None:
            return

        events = getattr(state, "events", [])
        prev_count = self._prev_event_counts.get(fid, 0)
        self._prev_event_counts[fid] = len(events)

        if prev_count == 0:
            # First time seeing this fixture, don't alert on existing goals
            return

        # Check for new goal events since last check
        new_events = events[prev_count:]
        for ev in new_events:
            ev_type = getattr(ev, "event_type", "") if hasattr(ev, "event_type") else ev.get("event_type", "")
            if ev_type == "goal":
                ev_dict = asdict(ev) if hasattr(ev, "__dataclass_fields__") else ev
                try:
                    em = goal_alert_embed(ev_dict, snapshot, state)
                    await channel.send(embed=em)
                except Exception as exc:
                    log.warning("Failed to post goal alert for %d: %s", fid, exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_channel(self, channel_id: int) -> Optional[discord.TextChannel]:
        """Get a channel by ID, returning None if not found or not configured."""
        if channel_id == 0:
            return None
        ch = self.bot.get_channel(channel_id)
        if ch is None:
            log.debug("Channel %d not found (bot may not have access)", channel_id)
        return ch

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(LiveAlertsCog(bot))
