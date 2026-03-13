"""Scheduled auto-push tasks: Parlay of the Day, Matchday Digest, Value Alerts."""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from datetime import datetime, time as dt_time

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
    DIGEST_POST_HOUR,
    ALERT_SCAN_INTERVAL_MINUTES,
    DOMESTIC_LEAGUES,
    MIN_ALERT_MODEL_PROB,
    MIN_ALERT_VALUE_EDGE,
)


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

    async def cog_unload(self):
        self.daily_digest.cancel()
        self.daily_parlay.cancel()
        self.value_scan.cancel()

    # --- Daily Matchday Digest ---
    @tasks.loop(time=dt_time(hour=DIGEST_POST_HOUR, minute=0))
    async def daily_digest(self):
        channel = self.bot.get_channel(CHANNEL_MATCHDAY_DIGEST)
        if not channel:
            return

        date = _today_phrase()
        for league in DOMESTIC_LEAGUES:
            prompt = (
                f"Give me the over/under goals line I should take for each "
                f"{league} game on {date}."
            )
            text = _run_rag(prompt, league)
            if text and "no fixtures" not in text.lower():
                embeds = rag_output_to_embeds(
                    f"Matchday Digest — {league}", text, league=league,
                    footer=f"Auto-posted | {date}",
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

        date = _today_phrase()
        # Try cross-league parlay first
        prompt = (
            f"For all top 5 league games on {date}, build a cross-league parlay "
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

        date = _today_phrase()
        for league in DOMESTIC_LEAGUES:
            prompt = (
                f"For every {league} game on {date}, show moneyline probabilities "
                "and value picks where the model disagrees with the bookmakers."
            )
            text = _run_rag(prompt, league)
            if not text or "no fixtures" in text.lower():
                continue
            # Only post if there are genuine value edges in the output
            if "value edge: +" in text.lower() or "high confidence" in text.lower():
                em = value_alert_embed(text, league)
                await channel.send(embed=em)

    @value_scan.before_loop
    async def _wait_scan(self):
        await self.bot.wait_until_ready()


async def setup(bot: commands.Bot):
    await bot.add_cog(AutoPush(bot))
