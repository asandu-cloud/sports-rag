"""Scheduled auto-push tasks: Parlay of the Day, Matchday Digest, Value Alerts, Track Record."""

from __future__ import annotations

import io
import logging
import sys
from contextlib import redirect_stdout
from datetime import datetime, date, timedelta, time as dt_time

import discord
from discord.ext import commands, tasks

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "rag_ingest"))
import rag_cli_v2 as rag  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from embeds import rag_output_to_embeds, parlay_embed, value_alert_embed  # noqa: E402
from config import (  # noqa: E402
    CHANNEL_PARLAY_OF_DAY,
    CHANNEL_MATCHDAY_DIGEST,
    CHANNEL_VALUE_ALERTS,
    CHANNEL_TRACK_RECORD,
    DIGEST_POST_HOUR,
    ALERT_SCAN_INTERVAL_MINUTES,
    DOMESTIC_LEAGUES,
    MIN_ALERT_MODEL_PROB,
    MIN_ALERT_VALUE_EDGE,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_RED,
    COLOR_BLUE,
)

# Prediction tracker (optional — graceful if unavailable)
try:
    from prediction_tracker import resolve_outcomes, get_track_record
    _HAS_TRACKER = True
except ImportError:
    _HAS_TRACKER = False

log = logging.getLogger("auto_push")


def _run_rag(prompt: str, league: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rag.answer_once(prompt, default_league=league)
    return buf.getvalue().strip() or ""


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"


def _today_phrase() -> str:
    d = datetime.now().date()
    return f"{_ordinal(d.day)} {d.strftime('%B')}"


class AutoPush(commands.Cog):
    """Scheduled auto-posting tasks."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def cog_load(self):
        if CHANNEL_MATCHDAY_DIGEST:
            self.daily_digest.start()
        if CHANNEL_PARLAY_OF_DAY:
            self.daily_parlay.start()
        if CHANNEL_VALUE_ALERTS:
            self.value_scan.start()
        if CHANNEL_TRACK_RECORD and _HAS_TRACKER:
            self.daily_track_record.start()

    async def cog_unload(self):
        self.daily_digest.cancel()
        self.daily_parlay.cancel()
        self.value_scan.cancel()
        if hasattr(self, "daily_track_record") and self.daily_track_record.is_running():
            self.daily_track_record.cancel()

    # --- Daily Matchday Digest ---
    @tasks.loop(time=dt_time(hour=DIGEST_POST_HOUR, minute=0))
    async def daily_digest(self):
        channel = self.bot.get_channel(CHANNEL_MATCHDAY_DIGEST)
        if not channel:
            return

        date_str = _today_phrase()
        for league in DOMESTIC_LEAGUES:
            prompt = (
                f"Give me the over/under goals line I should take for each "
                f"{league} game on {date_str}."
            )
            text = _run_rag(prompt, league)
            if text and "no fixtures" not in text.lower():
                embeds = rag_output_to_embeds(
                    f"Matchday Digest — {league}", text, league=league,
                    footer=f"Auto-posted | {date_str}",
                )
                for em in embeds:
                    await channel.send(embed=em)

    @daily_digest.before_loop
    async def _wait_digest(self):
        await self.bot.wait_until_ready()

    # --- Daily Parlay of the Day ---
    @tasks.loop(time=dt_time(hour=DIGEST_POST_HOUR, minute=15))
    async def daily_parlay(self):
        channel = self.bot.get_channel(CHANNEL_PARLAY_OF_DAY)
        if not channel:
            return

        date_str = _today_phrase()
        prompt = (
            f"For all top 5 league games on {date_str}, build a cross-league parlay "
            "with odds between 3x and 5x. Pick the best combination across all leagues."
        )
        text = _run_rag(prompt, "EPL")
        if text and "no fixtures" not in text.lower():
            embeds = parlay_embed(text, league="Cross-League")
            for em in embeds:
                await channel.send(embed=em)

    @daily_parlay.before_loop
    async def _wait_parlay(self):
        await self.bot.wait_until_ready()

    # --- Value Alert Scan ---
    @tasks.loop(minutes=ALERT_SCAN_INTERVAL_MINUTES)
    async def value_scan(self):
        channel = self.bot.get_channel(CHANNEL_VALUE_ALERTS)
        if not channel:
            return

        date_str = _today_phrase()
        for league in DOMESTIC_LEAGUES:
            prompt = (
                f"For every {league} game on {date_str}, show moneyline probabilities "
                "and value picks where the model disagrees with the bookmakers."
            )
            text = _run_rag(prompt, league)
            if not text or "no fixtures" in text.lower():
                continue
            if "value edge: +" in text.lower() or "high confidence" in text.lower():
                em = value_alert_embed(text, league)
                await channel.send(embed=em)

    @value_scan.before_loop
    async def _wait_scan(self):
        await self.bot.wait_until_ready()

    # --- Daily Track Record Report (noon UTC) ---
    @tasks.loop(time=dt_time(hour=12, minute=0))
    async def daily_track_record(self):
        channel = self.bot.get_channel(CHANNEL_TRACK_RECORD)
        if not channel:
            return

        yesterday = (date.today() - timedelta(days=1)).isoformat()

        # Step 1: Resolve yesterday's outcomes
        try:
            resolve_result = resolve_outcomes(prediction_date=yesterday)
            graded = resolve_result.get("graded", 0)
            hits = resolve_result.get("hit", 0)
            misses = resolve_result.get("miss", 0)
            log.info("Resolved %d predictions for %s: %d hits, %d misses", graded, yesterday, hits, misses)
        except Exception as exc:
            log.error("Failed to resolve outcomes for %s: %s", yesterday, exc)
            graded, hits, misses = 0, 0, 0

        if graded == 0:
            # No predictions to report — likely no matches yesterday
            return

        # Step 2: Build the track record embed
        try:
            stats = get_track_record()
        except Exception as exc:
            log.error("Failed to get track record: %s", exc)
            return

        # Yesterday's results
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

        # Yesterday's performance
        em.add_field(
            name="Yesterday's Results",
            value=(
                f"**{hits}/{graded}** ({hit_rate_yesterday:.0%})\n"
                f"Hits: {hits} | Misses: {misses}"
            ),
            inline=False,
        )

        # By confidence (yesterday only not available easily, so show overall)
        by_conf = stats.get("by_confidence", {})
        conf_lines = []
        for level in ["high", "medium", "low"]:
            c = by_conf.get(level, {})
            if c.get("total", 0) > 0:
                label = {"high": "Strong Edge", "medium": "Leaning", "low": "Speculative"}[level]
                conf_lines.append(
                    f"**{label}**: {c['hits']}/{c['total']} ({c['hit_rate']:.0%})"
                )
        if conf_lines:
            em.add_field(
                name="All-Time by Confidence",
                value="\n".join(conf_lines),
                inline=True,
            )

        # By market (top 5)
        by_market = stats.get("by_market", {})
        market_lines = []
        for market, m in sorted(by_market.items(), key=lambda x: x[1].get("total", 0), reverse=True)[:5]:
            if m.get("total", 0) > 0:
                market_lines.append(
                    f"**{market.title()}**: {m['hits']}/{m['total']} ({m['hit_rate']:.0%})"
                )
        if market_lines:
            em.add_field(
                name="All-Time by Market",
                value="\n".join(market_lines),
                inline=True,
            )

        # Overall stats
        total = stats.get("total_graded", 0)
        overall_hits = stats.get("hits", 0)
        overall_rate = stats.get("hit_rate", 0)
        roi = stats.get("roi_flat_stake", 0)
        streak = stats.get("recent_streak", 0)

        streak_str = f"W{streak}" if streak > 0 else (f"L{abs(streak)}" if streak < 0 else "—")

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


async def setup(bot: commands.Bot):
    await bot.add_cog(AutoPush(bot))
