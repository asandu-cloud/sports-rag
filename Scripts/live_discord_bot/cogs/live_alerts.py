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
from typing import Any, Dict, List, Optional, Set

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
# Live engine imports (graceful degradation)
# ---------------------------------------------------------------------------
_HAS_ALERTS = False
detect_alerts = None
try:
    from live.live_alerts import detect_alerts as _detect_alerts
    detect_alerts = _detect_alerts
    _HAS_ALERTS = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Embed builders
# ---------------------------------------------------------------------------
from embeds import (
    full_time_embed,
    goal_alert_embed,
    live_alert_embed,
    match_start_embed,
)

log = logging.getLogger("live_bot.alerts_cog")


class LiveAlertsCog(commands.Cog):
    """Cog that runs the live polling loop and posts alerts to Discord channels."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.poller = None  # LivePoller instance, initialized in cog_load
        self._prev_snapshots: Dict[int, Any] = {}  # fixture_id -> previous snapshot
        self._known_fixtures: Set[int] = set()  # fixtures we have posted "match started" for
        self._prev_event_counts: Dict[int, int] = {}  # fixture_id -> event count at last check
        self._finished: Set[int] = set()  # fixtures we have posted "full time" for

    async def cog_load(self) -> None:
        """Initialize the LivePoller and start the alert loop."""
        try:
            from live.live_poller import LivePoller
            self.poller = LivePoller(auto_subscribe=True)
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

            # --- Alert detection ---
            if snapshot is not None and _HAS_ALERTS and detect_alerts is not None:
                prev = self._prev_snapshots.get(fid)
                try:
                    # detect_alerts expects the engine's LiveMatchState, but we
                    # have the poller's LiveMatchState. Build a bridge object.
                    engine_state = self._to_engine_state(state, snapshot)
                    alerts = detect_alerts(snapshot, prev, engine_state)
                except Exception as exc:
                    log.warning("detect_alerts failed for fixture %d: %s", fid, exc)
                    alerts = []

                for alert in alerts:
                    if alerts_channel:
                        try:
                            em = live_alert_embed(alert, snapshot, state)
                            await alerts_channel.send(embed=em)
                        except Exception as exc:
                            log.warning("Failed to post alert for %d: %s", fid, exc)

                self._prev_snapshots[fid] = snapshot

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

    def _to_engine_state(self, poller_state: Any, snapshot: Any) -> Any:
        """Bridge the poller's LiveMatchState to the engine's LiveMatchState.

        The engine's detect_alerts expects live_state.LiveMatchState with flat
        fields (goals_home, corners_home, etc.), while the poller uses
        home_stats/away_stats sub-objects. This method creates a compatible
        object.
        """
        try:
            from live.live_state import LiveMatchState as EngineState
        except ImportError:
            # If we can't import the engine state, return the poller state
            # and hope for the best (detect_alerts uses getattr defensively)
            return poller_state

        hs = getattr(poller_state, "home_stats", None)
        as_ = getattr(poller_state, "away_stats", None)

        kwargs = dict(
            fixture_id=getattr(poller_state, "fixture_id", 0),
            league=getattr(poller_state, "league", ""),
            home_team=getattr(poller_state, "home_team", ""),
            away_team=getattr(poller_state, "away_team", ""),
            status=getattr(poller_state, "status", ""),
            elapsed_min=float(getattr(poller_state, "elapsed_min", 0)),
        )

        if hs:
            kwargs["goals_home"] = getattr(hs, "goals", 0)
            kwargs["corners_home"] = getattr(hs, "corners", 0)
            kwargs["cards_home"] = getattr(hs, "yellow_cards", 0) + getattr(hs, "red_cards", 0)
            kwargs["sot_home"] = getattr(hs, "sot", 0)
            kwargs["shots_home"] = getattr(hs, "shots", 0)
            kwargs["fouls_home"] = getattr(hs, "fouls", 0)
            kwargs["possession_home"] = getattr(hs, "possession", 50.0)
            kwargs["red_cards_home"] = getattr(hs, "red_cards", 0)

        if as_:
            kwargs["goals_away"] = getattr(as_, "goals", 0)
            kwargs["corners_away"] = getattr(as_, "corners", 0)
            kwargs["cards_away"] = getattr(as_, "yellow_cards", 0) + getattr(as_, "red_cards", 0)
            kwargs["sot_away"] = getattr(as_, "sot", 0)
            kwargs["shots_away"] = getattr(as_, "shots", 0)
            kwargs["fouls_away"] = getattr(as_, "fouls", 0)
            kwargs["possession_away"] = getattr(as_, "possession", 50.0)
            kwargs["red_cards_away"] = getattr(as_, "red_cards", 0)

        # Transfer pre-match priors
        kwargs["prior_goals"] = getattr(poller_state, "prior_total_goals", None)
        kwargs["prior_corners"] = getattr(poller_state, "prior_total_corners", None)
        kwargs["prior_cards"] = getattr(poller_state, "prior_total_cards", None)
        kwargs["prior_sot"] = getattr(poller_state, "prior_total_sot", None)
        kwargs["prior_goals_home"] = getattr(poller_state, "prior_home_goals", None)
        kwargs["prior_goals_away"] = getattr(poller_state, "prior_away_goals", None)
        kwargs["prior_btts_prob"] = getattr(poller_state, "prior_btts_prob", None)

        # Transfer events (convert poller MatchEvent to engine MatchEvent)
        poller_events = getattr(poller_state, "events", [])
        try:
            from live.live_state import MatchEvent as EngineEvent
            engine_events = []
            for ev in poller_events:
                if hasattr(ev, "__dataclass_fields__"):
                    # Poller's MatchEvent has team (name), engine's has team_side
                    team_name = getattr(ev, "team", "")
                    # Determine side from team name
                    home_name = getattr(poller_state, "home_team", "")
                    team_side = "home" if team_name == home_name else "away"
                    engine_events.append(EngineEvent(
                        minute=getattr(ev, "minute", 0),
                        event_type=getattr(ev, "event_type", ""),
                        team_side=team_side,
                        player=getattr(ev, "player", "") or "",
                        detail=getattr(ev, "detail", "") or "",
                    ))
            kwargs["events"] = engine_events
        except ImportError:
            pass

        try:
            return EngineState(**kwargs)
        except Exception as exc:
            log.debug("Failed to construct engine state: %s", exc)
            return poller_state


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(LiveAlertsCog(bot))
