"""Scheduled auto-push tasks: Parlays, Market Channels, Value Alerts, Track Record.

Market channels post automatically before kickoff — no user interaction required.
Each channel gets league-by-league analysis for all fixtures on the day.
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from datetime import datetime, date, timedelta, time as dt_time, timezone

import discord
from discord.ext import commands, tasks

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "rag_ingest"))
import rag_cli_v2 as rag  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from embeds import (  # noqa: E402
    rag_output_to_embeds, parlay_embed, value_alert_embed,
    consolidated_fixture_embeds, consolidated_score_embeds,
    parse_fixture_picks, parse_correct_score_picks, parse_interval_picks,
)
from config import (  # noqa: E402
    CHANNEL_PARLAY_OF_DAY,
    CHANNEL_MATCHDAY_DIGEST,
    CHANNEL_VALUE_ALERTS,
    CHANNEL_TRACK_RECORD,
    CHANNEL_MATCH_PREDICTIONS,
    CHANNEL_CORNERS_CARDS,
    CHANNEL_PLAYER_PROPS,
    CHANNEL_CORRECT_SCORE,
    DIGEST_POST_HOUR,
    ALERT_SCAN_INTERVAL_MINUTES,
    DOMESTIC_LEAGUES,
    MIN_ALERT_MODEL_PROB,
    MIN_ALERT_VALUE_EDGE,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_PURPLE,
)

# Prediction tracker (optional — graceful if unavailable)
try:
    from prediction_tracker import resolve_outcomes, get_track_record
    _HAS_TRACKER = True
except ImportError:
    _HAS_TRACKER = False

log = logging.getLogger("auto_push")

# Thread pool for blocking RAG calls — keeps the async event loop free
_rag_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_rag_sync(prompt: str, league: str) -> str:
    """Synchronous RAG call — runs in thread pool, never call from event loop."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rag.answer_once(prompt, default_league=league)
    except Exception as exc:
        log.error("RAG query failed: %s", exc, exc_info=True)
        return ""
    result = buf.getvalue().strip()
    if not result:
        log.warning("RAG returned empty output for prompt: %.100s", prompt)
    else:
        log.info("RAG output length: %d chars for prompt: %.80s", len(result), prompt)
    return result


async def _run_rag(prompt: str, league: str) -> str:
    """Async wrapper — runs RAG in thread pool so event loop stays free."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_rag_executor, _run_rag_sync, prompt, league)


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"


def _today_phrase() -> str:
    d = datetime.now().date()
    return f"{_ordinal(d.day)} {d.strftime('%B')}"


def _is_valid(text: str) -> bool:
    """Check if RAG output has real content (not an error/empty message)."""
    if not text:
        return False
    low = text.lower()
    return "no fixtures" not in low and "could not" not in low and "no odds" not in low


def _get_todays_leagues() -> tuple:
    """Find which leagues have fixtures today, and the first kickoff time.

    Returns (first_kickoff_dt_or_None, active_leagues_list).
    Scans all domestic + European leagues, only returns those with matches.
    """
    all_leagues = DOMESTIC_LEAGUES + ["UCL", "UEL", "UECL"]
    first_kickoff = None
    active_leagues = []

    for league in all_leagues:
        try:
            events, _ = rag.fetch_events(league, set(rag.DEFAULT_MARKETS))
            day_events = rag.filter_events_by_exact_date(events, date.today())
            if day_events:
                active_leagues.append(league)
                for ev in day_events:
                    ko = ev.get("commence_time")
                    if ko:
                        try:
                            ko_dt = datetime.fromisoformat(ko.replace("Z", "+00:00"))
                            if first_kickoff is None or ko_dt < first_kickoff:
                                first_kickoff = ko_dt
                        except Exception:
                            pass
        except Exception:
            pass

    if active_leagues:
        log.info("Today's active leagues: %s", active_leagues)
    else:
        log.info("No fixtures found in any league today.")

    return first_kickoff, active_leagues


def _is_posting_window(first_kickoff) -> bool:
    """Return True if we're 2-3.5 hours before first kickoff."""
    if first_kickoff is None:
        return True  # no kickoff time known, post anyway
    now = datetime.now(timezone.utc)
    if first_kickoff.tzinfo is None:
        first_kickoff = first_kickoff.replace(tzinfo=timezone.utc)
    hours_until = (first_kickoff - now).total_seconds() / 3600
    log.info("First kickoff in %.1f hours (kickoff=%s, now=%s)",
             hours_until, first_kickoff, now)
    return -0.5 <= hours_until <= 3.5


async def _post_market_for_leagues(
    channel: discord.TextChannel,
    prompt_template: str,
    title_template: str,
    leagues: list,
    date_str: str,
):
    """Post a market analysis to a channel for each league that has fixtures."""
    for league in leagues:
        prompt = prompt_template.format(league=league, date=date_str)
        text = await _run_rag(prompt, league)
        if not _is_valid(text):
            continue
        embeds = rag_output_to_embeds(
            title_template.format(league=league),
            text,
            league=league,
        )
        for em in embeds:
            await channel.send(embed=em)


async def _post_consolidated_predictions(
    channel: discord.TextChannel,
    leagues: list,
    date_str: str,
):
    """Post fixture-consolidated predictions (moneyline + BTTS + spreads + SoT).

    Instead of 4 separate market posts per league, builds ONE embed per fixture
    with all markets as compact lines.
    """
    _MARKET_QUERIES = {
        "moneyline": (
            "For every {league} game on {date}, show me moneyline "
            "probabilities and the best value pick."
        ),
        "btts": (
            "For every {league} game on {date}, should I take "
            "BTTS Yes or No?"
        ),
        "spreads": (
            "For every {league} game on {date}, recommend a handicap line."
        ),
        "sot": (
            "For every {league} game on {date}, recommend a shots on target "
            "over/under line."
        ),
    }

    for league in leagues:
        # Run all 4 market queries, parse picks per fixture
        fixture_data: dict = {}  # {fixture: {market: compact_line}}

        for market_key, prompt_tpl in _MARKET_QUERIES.items():
            prompt = prompt_tpl.format(league=league, date=date_str)
            text = await _run_rag(prompt, league)
            if not _is_valid(text):
                continue
            picks = parse_fixture_picks(text)
            for fixture, line in picks.items():
                if fixture not in fixture_data:
                    fixture_data[fixture] = {}
                fixture_data[fixture][market_key] = line

        if not fixture_data:
            continue

        embeds = consolidated_fixture_embeds(league, fixture_data)
        for em in embeds:
            await channel.send(embed=em)


async def _post_consolidated_scores(
    channel: discord.TextChannel,
    leagues: list,
    date_str: str,
):
    """Post fixture-consolidated correct score + goal intervals.

    One embed per fixture with top scorelines and interval breakdown.
    """
    for league in leagues:
        fixture_data: dict = {}

        # Correct score
        text = await _run_rag(
            f"For every {league} game on {date_str}, predict the most "
            "likely correct score.",
            league,
        )
        if _is_valid(text):
            for fixture, cs_str in parse_correct_score_picks(text).items():
                if fixture not in fixture_data:
                    fixture_data[fixture] = {}
                fixture_data[fixture]["correct_score"] = cs_str

        # Goal intervals
        text = await _run_rag(
            f"For every {league} game on {date_str}, show me goal "
            "interval probabilities.",
            league,
        )
        if _is_valid(text):
            for fixture, iv_str in parse_interval_picks(text).items():
                if fixture not in fixture_data:
                    fixture_data[fixture] = {}
                fixture_data[fixture]["intervals"] = iv_str

        if not fixture_data:
            continue

        embeds = consolidated_score_embeds(league, fixture_data)
        for em in embeds:
            await channel.send(embed=em)


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class AutoPush(commands.Cog):
    """Scheduled auto-posting tasks."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._posted_today: dict = {}  # track which tasks posted today

    def _already_posted(self, task_name: str) -> bool:
        today = date.today().isoformat()
        return self._posted_today.get(task_name) == today

    def _mark_posted(self, task_name: str):
        self._posted_today[task_name] = date.today().isoformat()

    async def cog_load(self):
        if CHANNEL_MATCHDAY_DIGEST:
            self.daily_digest.start()
        if CHANNEL_PARLAY_OF_DAY:
            self.daily_parlay.start()
        if CHANNEL_VALUE_ALERTS:
            self.value_scan.start()
        if CHANNEL_TRACK_RECORD and _HAS_TRACKER:
            self.daily_track_record.start()
        # Market channels — all share the same 30-min check loop
        self.market_channels.start()

    async def cog_unload(self):
        for task in [self.daily_digest, self.daily_parlay, self.value_scan,
                     self.market_channels]:
            if task.is_running():
                task.cancel()
        if hasattr(self, "daily_track_record") and self.daily_track_record.is_running():
            self.daily_track_record.cancel()

    # ===================================================================
    # MARKET CHANNELS — auto-post before kickoff
    # ===================================================================

    @tasks.loop(minutes=30)
    async def market_channels(self):
        """Post market analysis to dedicated channels before first kickoff.

        Checks every 30 min. Posts once per day when within the pre-kickoff window.
        Each market type goes to its own channel.
        """
        first_kickoff, active_leagues = _get_todays_leagues()
        if not active_leagues:
            return

        if not _is_posting_window(first_kickoff):
            return

        date_str = _today_phrase()
        leagues = active_leagues

        # --- Match Predictions: consolidated by fixture ---
        if CHANNEL_MATCH_PREDICTIONS and not self._already_posted("predictions"):
            ch = self.bot.get_channel(CHANNEL_MATCH_PREDICTIONS)
            if ch:
                log.info("Posting consolidated match predictions...")
                await _post_consolidated_predictions(ch, leagues, date_str)
                self._mark_posted("predictions")

        # --- Corners + Cards ---
        if CHANNEL_CORNERS_CARDS and not self._already_posted("corners_cards"):
            ch = self.bot.get_channel(CHANNEL_CORNERS_CARDS)
            if ch:
                log.info("Posting corner & card lines...")
                await _post_market_for_leagues(
                    ch,
                    "For all {league} games on {date}, what are the per-team "
                    "corner and card lines for each fixture?",
                    "Corner & Card Lines — {league}",
                    leagues, date_str,
                )
                self._mark_posted("corners_cards")

        # --- Correct Score + Goal Intervals: consolidated by fixture ---
        if CHANNEL_CORRECT_SCORE and not self._already_posted("correct_score"):
            ch = self.bot.get_channel(CHANNEL_CORRECT_SCORE)
            if ch:
                log.info("Posting consolidated correct score + intervals...")
                await _post_consolidated_scores(ch, leagues, date_str)
                self._mark_posted("correct_score")

        # --- Player Props: Goals + Cards + SoT + Assists ---
        if CHANNEL_PLAYER_PROPS and not self._already_posted("player_props"):
            ch = self.bot.get_channel(CHANNEL_PLAYER_PROPS)
            if ch:
                log.info("Posting player props (goals + cards + SoT + assists)...")
                for league in leagues:
                    prompt = (
                        f"For all {league} games on {date_str}, who are the "
                        f"best anytime goalscorer picks?"
                    )
                    text = await _run_rag(prompt, league)
                    if _is_valid(text):
                        for em in rag_output_to_embeds(
                            f"Goalscorer Picks — {league}", text, league=league,
                        ):
                            await ch.send(embed=em)

                    prompt = (
                        f"For all {league} games on {date_str}, which players "
                        f"are most likely to be booked?"
                    )
                    text = await _run_rag(prompt, league)
                    if _is_valid(text):
                        for em in rag_output_to_embeds(
                            f"Player Cards — {league}", text, league=league,
                        ):
                            await ch.send(embed=em)

                    prompt = (
                        f"For all {league} games on {date_str}, which players "
                        f"are best for shots on target props?"
                    )
                    text = await _run_rag(prompt, league)
                    if _is_valid(text):
                        for em in rag_output_to_embeds(
                            f"Player SoT — {league}", text, league=league,
                        ):
                            await ch.send(embed=em)

                    prompt = (
                        f"For all {league} games on {date_str}, who are the "
                        f"best assist picks?"
                    )
                    text = await _run_rag(prompt, league)
                    if _is_valid(text):
                        for em in rag_output_to_embeds(
                            f"Assist Picks — {league}", text, league=league,
                        ):
                            await ch.send(embed=em)
                self._mark_posted("player_props")

    @market_channels.before_loop
    async def _wait_markets(self):
        await self.bot.wait_until_ready()

    # ===================================================================
    # PARLAY OF THE DAY
    # ===================================================================

    @tasks.loop(minutes=30)
    async def daily_parlay(self):
        channel = self.bot.get_channel(CHANNEL_PARLAY_OF_DAY)
        if not channel:
            return

        if self._already_posted("parlay"):
            return

        first_kickoff, active_leagues = _get_todays_leagues()
        if not active_leagues:
            return
        if not _is_posting_window(first_kickoff):
            return

        date_str = _today_phrase()

        # 1. THE LOCK — ultra-safe, ~1.8x odds
        lock_text = await _run_rag(
            f"For all league games on {date_str} across all leagues, "
            "build a cross-league 2-leg parlay using ONLY the two strongest "
            "high-confidence picks. Keep combined decimal odds at or below 1.9x. "
            "Prioritize picks where the model probability is above 65%. "
            "This is the safest possible parlay.",
            "EPL",
        )

        # 2. SAFE — conservative, 2-3x odds
        safe_text = await _run_rag(
            f"For all league games on {date_str} across all leagues, "
            "build a cross-league 3-leg parlay with combined odds between "
            "2.0x and 3.0x. Only use high or medium confidence picks.",
            "EPL",
        )

        # 3. STANDARD — balanced risk, 4-6x odds
        std_text = await _run_rag(
            f"For all league games on {date_str} across all leagues, "
            "build a cross-league 4-leg parlay with combined odds between "
            "4x and 6x. Mix goals, corners, and cards markets for diversity.",
            "EPL",
        )

        has_any = _is_valid(lock_text) or _is_valid(safe_text) or _is_valid(std_text)
        if not has_any:
            log.warning("All 3 parlay prompts returned no usable output.")
            return

        self._mark_posted("parlay")

        header = discord.Embed(
            title="\U0001f3af  Parlay of the Day",
            description=f"**{date_str}** — 3 parlays ranked by risk level",
            color=COLOR_GREEN,
        )
        header.set_footer(text="Spick | Auto-generated from model projections")
        await channel.send(embed=header)

        if _is_valid(lock_text):
            for em in parlay_embed(lock_text, league="Cross-League"):
                em.title = "\U0001f512  The Lock (~1.8x)"
                em.color = COLOR_GREEN
                await channel.send(embed=em)

        if _is_valid(safe_text):
            for em in parlay_embed(safe_text, league="Cross-League"):
                em.title = "\U0001f6e1\ufe0f  Safe Parlay (~2.5x)"
                em.color = COLOR_BLUE
                await channel.send(embed=em)

        if _is_valid(std_text):
            for em in parlay_embed(std_text, league="Cross-League"):
                em.title = "\U0001f3b2  Standard Parlay (~5x)"
                em.color = COLOR_PURPLE
                await channel.send(embed=em)

    @daily_parlay.before_loop
    async def _wait_parlay(self):
        await self.bot.wait_until_ready()

    # ===================================================================
    # MATCHDAY DIGEST (goals totals — fixed hour)
    # ===================================================================

    @tasks.loop(time=dt_time(hour=DIGEST_POST_HOUR, minute=0))
    async def daily_digest(self):
        channel = self.bot.get_channel(CHANNEL_MATCHDAY_DIGEST)
        if not channel:
            return

        _, active_leagues = _get_todays_leagues()
        if not active_leagues:
            return

        date_str = _today_phrase()
        for league in active_leagues:
            prompt = (
                f"Give me the over/under goals line I should take for each "
                f"{league} game on {date_str}."
            )
            text = await _run_rag(prompt, league)
            if _is_valid(text):
                embeds = rag_output_to_embeds(
                    f"Goals Totals — {league}", text, league=league,
                    footer=f"Auto-posted | {date_str}",
                )
                for em in embeds:
                    await channel.send(embed=em)

    @daily_digest.before_loop
    async def _wait_digest(self):
        await self.bot.wait_until_ready()

    # ===================================================================
    # VALUE ALERTS
    # ===================================================================

    @tasks.loop(minutes=ALERT_SCAN_INTERVAL_MINUTES)
    async def value_scan(self):
        channel = self.bot.get_channel(CHANNEL_VALUE_ALERTS)
        if not channel:
            return

        _, active_leagues = _get_todays_leagues()
        if not active_leagues:
            return

        date_str = _today_phrase()
        for league in active_leagues:
            prompt = (
                f"For every {league} game on {date_str}, show moneyline "
                "probabilities and value picks where the model disagrees "
                "with the bookmakers."
            )
            text = await _run_rag(prompt, league)
            if not _is_valid(text):
                continue
            if "value edge: +" in text.lower() or "high confidence" in text.lower():
                em = value_alert_embed(text, league)
                await channel.send(embed=em)

    @value_scan.before_loop
    async def _wait_scan(self):
        await self.bot.wait_until_ready()

    # ===================================================================
    # DAILY TRACK RECORD (noon UTC)
    # ===================================================================

    @tasks.loop(time=dt_time(hour=12, minute=0))
    async def daily_track_record(self):
        channel = self.bot.get_channel(CHANNEL_TRACK_RECORD)
        if not channel:
            return

        yesterday = (date.today() - timedelta(days=1)).isoformat()

        try:
            resolve_result = resolve_outcomes(prediction_date=yesterday)
            graded = resolve_result.get("graded", 0)
            hits = resolve_result.get("hit", 0)
            misses = resolve_result.get("miss", 0)
            log.info("Resolved %d predictions for %s: %d hits, %d misses",
                     graded, yesterday, hits, misses)
        except Exception as exc:
            log.error("Failed to resolve outcomes for %s: %s", yesterday, exc)
            graded, hits, misses = 0, 0, 0

        if graded == 0:
            return

        try:
            stats = get_track_record()
        except Exception as exc:
            log.error("Failed to get track record: %s", exc)
            return

        hit_rate_yesterday = hits / graded if graded > 0 else 0
        if hit_rate_yesterday >= 0.60:
            color = COLOR_GREEN
        elif hit_rate_yesterday >= 0.50:
            color = COLOR_YELLOW
        else:
            color = COLOR_RED

        em = discord.Embed(
            title=f"Daily Track Record — {yesterday}",
            color=color,
        )

        em.add_field(
            name="Yesterday's Results",
            value=(
                f"**{hits}/{graded}** ({hit_rate_yesterday:.0%})\n"
                f"Hits: {hits} | Misses: {misses}"
            ),
            inline=False,
        )

        by_conf = stats.get("by_confidence", {})
        conf_lines = []
        for level in ["high", "medium", "low"]:
            c = by_conf.get(level, {})
            if c.get("total", 0) > 0:
                label = {"high": "Strong Edge", "medium": "Leaning",
                         "low": "Speculative"}[level]
                conf_lines.append(
                    f"**{label}**: {c['hits']}/{c['total']} ({c['hit_rate']:.0%})"
                )
        if conf_lines:
            em.add_field(
                name="All-Time by Confidence",
                value="\n".join(conf_lines),
                inline=True,
            )

        by_market = stats.get("by_market", {})
        market_lines = []
        for market, m in sorted(by_market.items(),
                                key=lambda x: x[1].get("total", 0),
                                reverse=True)[:5]:
            if m.get("total", 0) > 0:
                market_lines.append(
                    f"**{market.title()}**: {m['hits']}/{m['total']} "
                    f"({m['hit_rate']:.0%})"
                )
        if market_lines:
            em.add_field(
                name="All-Time by Market",
                value="\n".join(market_lines),
                inline=True,
            )

        total = stats.get("total_graded", 0)
        overall_hits = stats.get("hits", 0)
        overall_rate = stats.get("hit_rate", 0)
        roi = stats.get("roi_flat_stake", 0)
        streak = stats.get("recent_streak", 0)
        streak_str = (f"W{streak}" if streak > 0
                      else (f"L{abs(streak)}" if streak < 0 else "—"))

        em.add_field(
            name="All-Time Record",
            value=(
                f"**{overall_hits}/{total}** ({overall_rate:.1%})\n"
                f"ROI: {roi:+.1%} | Streak: {streak_str}"
            ),
            inline=False,
        )
        em.set_footer(text="Matchwise Track Record | Updated daily at noon UTC")
        await channel.send(embed=em)

    @daily_track_record.before_loop
    async def _wait_track_record(self):
        await self.bot.wait_until_ready()

    # ===================================================================
    # MANUAL TRIGGER (owner only)
    # ===================================================================

    @discord.app_commands.command(
        name="trigger",
        description="Manually trigger an auto-push task (owner only)",
    )
    @discord.app_commands.describe(task="Which task to trigger")
    @discord.app_commands.choices(task=[
        discord.app_commands.Choice(name="Parlay of the Day", value="parlay"),
        discord.app_commands.Choice(name="Goals Digest", value="digest"),
        discord.app_commands.Choice(name="Value Alerts", value="alerts"),
        discord.app_commands.Choice(name="Predictions (ML+BTTS+Spreads+SoT)", value="predictions"),
        discord.app_commands.Choice(name="Corners & Cards", value="corners_cards"),
        discord.app_commands.Choice(name="Correct Score & Intervals", value="correct_score"),
        discord.app_commands.Choice(name="Player Props", value="player_props"),
        discord.app_commands.Choice(name="All Markets", value="all_markets"),
    ])
    async def trigger(
        self,
        interaction: discord.Interaction,
        task: discord.app_commands.Choice[str],
    ):
        if interaction.user.id != interaction.guild.owner_id:
            await interaction.response.send_message(
                "Only the server owner can use this command.", ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True, ephemeral=True)
        date_str = _today_phrase()
        _, active_leagues = _get_todays_leagues()
        if not active_leagues:
            await interaction.followup.send(
                "No fixtures found in any league today.", ephemeral=True)
            return
        leagues = active_leagues
        log.info("Manual trigger: active leagues = %s", leagues)

        try:
            if task.value == "parlay":
                self._posted_today.pop("parlay", None)
                await self.daily_parlay()
                await interaction.followup.send(
                    "Parlay of the Day posted.", ephemeral=True)

            elif task.value == "digest":
                ch = self.bot.get_channel(CHANNEL_MATCHDAY_DIGEST)
                if not ch:
                    await interaction.followup.send(
                        "Digest channel not configured.", ephemeral=True)
                    return
                for league in leagues:
                    prompt = (
                        f"Give me the over/under goals line I should take "
                        f"for each {league} game on {date_str}."
                    )
                    text = await _run_rag(prompt, league)
                    if _is_valid(text):
                        for em in rag_output_to_embeds(
                            f"Goals Totals — {league}", text, league=league,
                        ):
                            await ch.send(embed=em)
                await interaction.followup.send(
                    "Goals digest posted.", ephemeral=True)

            elif task.value == "alerts":
                ch = self.bot.get_channel(CHANNEL_VALUE_ALERTS)
                if not ch:
                    await interaction.followup.send(
                        "Alerts channel not configured.", ephemeral=True)
                    return
                posted = 0
                for league in leagues:
                    prompt = (
                        f"For every {league} game on {date_str}, show "
                        "moneyline probabilities and value picks where "
                        "the model disagrees with the bookmakers."
                    )
                    text = await _run_rag(prompt, league)
                    if _is_valid(text) and (
                        "value edge: +" in text.lower()
                        or "high confidence" in text.lower()
                    ):
                        await ch.send(embed=value_alert_embed(text, league))
                        posted += 1
                await interaction.followup.send(
                    f"Value Alerts: {posted} alert(s) posted.", ephemeral=True)

            elif task.value in ("predictions", "corners_cards", "correct_score",
                                "player_props", "all_markets"):
                # Force-post specific or all market channels
                targets = ([task.value] if task.value != "all_markets"
                           else ["predictions", "corners_cards",
                                 "correct_score", "player_props"])
                for t in targets:
                    self._posted_today.pop(t, None)
                # Trigger the market_channels loop logic directly
                await self._post_markets_now(targets, date_str, leagues)
                await interaction.followup.send(
                    f"Market channels posted: {', '.join(targets)}",
                    ephemeral=True,
                )

        except Exception as exc:
            log.error("Manual trigger failed: %s", exc, exc_info=True)
            await interaction.followup.send(f"Error: {exc}", ephemeral=True)

    async def _post_markets_now(self, targets: list, date_str: str,
                                leagues: list):
        """Post specific market channels immediately (for /trigger)."""
        if "predictions" in targets and CHANNEL_MATCH_PREDICTIONS:
            ch = self.bot.get_channel(CHANNEL_MATCH_PREDICTIONS)
            if ch:
                await _post_consolidated_predictions(ch, leagues, date_str)
                self._mark_posted("predictions")

        if "corners_cards" in targets and CHANNEL_CORNERS_CARDS:
            ch = self.bot.get_channel(CHANNEL_CORNERS_CARDS)
            if ch:
                await _post_market_for_leagues(
                    ch,
                    "For all {league} games on {date}, what are the per-team "
                    "corner and card lines for each fixture?",
                    "Corner & Card Lines — {league}",
                    leagues, date_str,
                )
                self._mark_posted("corners_cards")

        if "correct_score" in targets and CHANNEL_CORRECT_SCORE:
            ch = self.bot.get_channel(CHANNEL_CORRECT_SCORE)
            if ch:
                await _post_consolidated_scores(ch, leagues, date_str)
                self._mark_posted("correct_score")

        if "player_props" in targets and CHANNEL_PLAYER_PROPS:
            ch = self.bot.get_channel(CHANNEL_PLAYER_PROPS)
            if ch:
                _pp = [
                    ("who are the best anytime goalscorer picks?", "Goalscorer Picks"),
                    ("which players are most likely to be booked?", "Player Cards"),
                    ("which players are best for shots on target props?", "Player SoT"),
                    ("who are the best assist picks?", "Assist Picks"),
                ]
                for league in leagues:
                    for suffix, label in _pp:
                        prompt = f"For all {league} games on {date_str}, {suffix}"
                        text = await _run_rag(prompt, league)
                        if _is_valid(text):
                            for em in rag_output_to_embeds(
                                f"{label} — {league}", text, league=league,
                            ):
                                await ch.send(embed=em)
                self._mark_posted("player_props")


async def setup(bot: commands.Bot):
    await bot.add_cog(AutoPush(bot))
