"""Trial / free-picks funnel — user acquisition via curated low-odds parlays.

Flow:
    1. User joins server, has FREE MEMBER role (or no paid role).
    2. User runs /trial  -> gets TRIAL role -> can see #free-picks channel.
    3. Bot auto-posts 2 low-odds parlays (1.5-1.8x) to #free-picks each matchday.
    4. After matches conclude, bot resolves whether parlays hit.
    5. Once a user has seen 2 correct parlays since activating trial, bot DMs them
       with a graduation message and removes the TRIAL role.

Database:  Index/predictions.db  (shared SQLite — trial tables created idempotently)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, time as dt_time, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import discord
from discord import app_commands
from discord.ext import commands, tasks

# ---------------------------------------------------------------------------
# Path setup — same pattern as auto_push.py
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_BOT_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _BOT_DIR.parents[1]

sys.path.insert(0, str(_PROJECT_ROOT / "Scripts" / "rag_ingest"))
from core.events import fetch_events, filter_events_by_exact_date  # noqa: E402
from core.parlay_service import ParlayBuildError, build_parlay, build_parlay_request  # noqa: E402

sys.path.insert(0, str(_BOT_DIR))
from config import (  # noqa: E402
    DOMESTIC_LEAGUES,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_PURPLE,
    ROLE_TO_TIER,
    TIER_HIERARCHY,
)

log = logging.getLogger("trial")

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
CHANNEL_FREE_PICKS = int(os.getenv("DISCORD_CHANNEL_FREE_PICKS", "0"))
ROLE_TRIAL = os.getenv("DISCORD_ROLE_TRIAL", "TRIAL")

# Paid role names for checking existing subscriptions
_PAID_ROLE_NAMES = {"rookie", "tipster", "whale"}

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH = _PROJECT_ROOT / "Index" / "predictions.db"

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS trial_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    discord_id TEXT NOT NULL UNIQUE,
    discord_username TEXT,
    activated_at TEXT NOT NULL DEFAULT (datetime('now')),
    hits INTEGER NOT NULL DEFAULT 0,
    total_parlays_seen INTEGER NOT NULL DEFAULT 0,
    total_profit REAL NOT NULL DEFAULT 0.0,
    graduated INTEGER NOT NULL DEFAULT 0,
    graduated_at TEXT
);

CREATE TABLE IF NOT EXISTS trial_parlays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    posted_at TEXT NOT NULL DEFAULT (datetime('now')),
    matchday_date TEXT NOT NULL,
    legs TEXT NOT NULL,
    combined_odds REAL NOT NULL,
    outcome TEXT,
    resolved_at TEXT,
    stake REAL NOT NULL DEFAULT 10.0,
    profit REAL
);
"""


def _get_db() -> sqlite3.Connection:
    """Open a connection with WAL mode and return it."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_tables():
    """Create trial tables idempotently."""
    conn = _get_db()
    try:
        conn.executescript(_CREATE_TABLES)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Thread pool for blocking calls
# ---------------------------------------------------------------------------
_trial_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trial")

# ---------------------------------------------------------------------------
# Helpers — structured parlay generation
# ---------------------------------------------------------------------------


async def _build_structured_trial_parlay(request):
    """Async wrapper for structured trial parlay generation."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_trial_executor, build_parlay, request)


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"


def _today_phrase() -> str:
    d = datetime.now().date()
    return f"{_ordinal(d.day)} {d.strftime('%B')}"


def _get_todays_leagues() -> Tuple[Optional[datetime], List[str]]:
    """Find which leagues have fixtures today, and the first kickoff time."""
    all_leagues = DOMESTIC_LEAGUES + ["UCL", "UEL", "UECL"]
    first_kickoff = None
    active_leagues: List[str] = []

    for league in all_leagues:
        try:
            events, _ = fetch_events(league, target_date=date.today())
            day_events = filter_events_by_exact_date(events, date.today())
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

    return first_kickoff, active_leagues


def _is_posting_window(first_kickoff) -> bool:
    """Return True if we're 2-3.5 hours before first kickoff."""
    if first_kickoff is None:
        return True
    now = datetime.now(timezone.utc)
    if first_kickoff.tzinfo is None:
        first_kickoff = first_kickoff.replace(tzinfo=timezone.utc)
    hours_until = (first_kickoff - now).total_seconds() / 3600
    return -0.5 <= hours_until <= 3.5


def _trial_leg_strings(result) -> List[str]:
    """Convert structured parlay legs into stored/displayed trial strings."""
    legs: List[str] = []
    for leg in getattr(result, "selected_legs", []) or []:
        line = f"{leg.fixture} | {leg.pick_display} @ {leg.odds:.2f}"
        legs.append(line[:200])
    return legs[:4]


def _trial_parlay_presets(target_date: date, leagues: List[str]) -> List[dict]:
    """Return ordered low-risk trial presets for #free-picks.

    The first two are the primary daily free picks. The later presets are
    controlled fallbacks so the scheduler can still reach two posts on thinner
    slates without drifting into longshot territory.
    """
    leagues = list(leagues or [])
    return [
        {
            "name": "safe_mixed_primary",
            "request": build_parlay_request(
                league_value="cross-league",
                leagues_override=leagues,
                display_league="Cross-League",
                target_date=target_date,
                legs=2,
                odds=None,
                target_min=1.4,
                target_max=1.7,
                market="mixed",
                source_command="trial",
                target_mode="none",
                min_model_prob=0.65,
                allowed_confidences=["high"],
                risk_profile="trial-safe",
            ),
        },
        {
            "name": "safe_moneyline_btts_primary",
            "request": build_parlay_request(
                league_value="cross-league",
                leagues_override=leagues,
                display_league="Cross-League",
                target_date=target_date,
                legs=2,
                odds=None,
                target_min=1.5,
                target_max=1.8,
                market="mixed",
                source_command="trial",
                target_mode="none",
                min_model_prob=0.60,
                allowed_confidences=["high"],
                allowed_groups=["moneyline", "btts"],
                soft_prefer_groups=["moneyline", "btts"],
                risk_profile="trial-safe",
            ),
        },
        {
            "name": "safe_mixed_fallback",
            "request": build_parlay_request(
                league_value="cross-league",
                leagues_override=leagues,
                display_league="Cross-League",
                target_date=target_date,
                legs=2,
                odds=None,
                target_min=1.4,
                target_max=1.8,
                market="mixed",
                source_command="trial",
                target_mode="none",
                min_model_prob=0.60,
                allowed_confidences=["high", "medium"],
                risk_profile="trial-safe",
            ),
        },
        {
            "name": "safe_focus_fallback",
            "request": build_parlay_request(
                league_value="cross-league",
                leagues_override=leagues,
                display_league="Cross-League",
                target_date=target_date,
                legs=2,
                odds=None,
                target_min=1.45,
                target_max=1.85,
                market="mixed",
                source_command="trial",
                target_mode="none",
                min_model_prob=0.58,
                allowed_confidences=["high", "medium"],
                allowed_groups=["moneyline", "btts", "totals"],
                soft_prefer_groups=["moneyline", "btts"],
                risk_profile="trial-safe",
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

def _get_trial_user(discord_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a trial user record, or None."""
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM trial_users WHERE discord_id = ?", (discord_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _activate_trial(discord_id: str, username: str):
    """Insert a new trial user."""
    conn = _get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO trial_users (discord_id, discord_username) VALUES (?, ?)",
            (discord_id, username),
        )
        conn.commit()
    finally:
        conn.close()


def _log_trial_parlay(matchday_date: str, legs: List[str], combined_odds: float):
    """Log a trial parlay to the database."""
    conn = _get_db()
    try:
        conn.execute(
            "INSERT INTO trial_parlays (matchday_date, legs, combined_odds) VALUES (?, ?, ?)",
            (matchday_date, json.dumps(legs), combined_odds),
        )
        conn.commit()
    finally:
        conn.close()


def _get_unresolved_parlays() -> List[Dict[str, Any]]:
    """Fetch all unresolved trial parlays."""
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM trial_parlays WHERE outcome IS NULL ORDER BY posted_at"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _resolve_parlay(parlay_id: int, outcome: str, profit: float):
    """Mark a trial parlay as resolved."""
    conn = _get_db()
    try:
        conn.execute(
            "UPDATE trial_parlays SET outcome = ?, profit = ?, resolved_at = datetime('now') "
            "WHERE id = ?",
            (outcome, profit, parlay_id),
        )
        conn.commit()
    finally:
        conn.close()


def _get_active_trial_users() -> List[Dict[str, Any]]:
    """Fetch all non-graduated trial users."""
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM trial_users WHERE graduated = 0"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _get_hit_parlays_since(activated_at: str) -> List[Dict[str, Any]]:
    """Fetch trial parlays that hit since a given activation date."""
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM trial_parlays WHERE outcome = 'hit' AND resolved_at >= ? "
            "ORDER BY resolved_at",
            (activated_at,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _get_all_resolved_since(activated_at: str) -> List[Dict[str, Any]]:
    """Fetch all resolved trial parlays since activation."""
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM trial_parlays WHERE outcome IS NOT NULL AND resolved_at >= ? "
            "ORDER BY resolved_at",
            (activated_at,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _graduate_user(discord_id: str, hits: int, total_seen: int, total_profit: float):
    """Mark a user as graduated."""
    conn = _get_db()
    try:
        conn.execute(
            "UPDATE trial_users SET graduated = 1, graduated_at = datetime('now'), "
            "hits = ?, total_parlays_seen = ?, total_profit = ? "
            "WHERE discord_id = ?",
            (hits, total_seen, total_profit, discord_id),
        )
        conn.commit()
    finally:
        conn.close()


def _count_today_parlays() -> int:
    """Count how many trial parlays were posted today."""
    today = date.today().isoformat()
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM trial_parlays WHERE matchday_date = ?",
            (today,),
        ).fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


def _get_today_parlay_signatures() -> List[Tuple[str, ...]]:
    """Return exact leg signatures already posted today.

    This lets the scheduler retry later in the day without reposting the same
    free pick if only one parlay succeeded on an earlier attempt.
    """
    today = date.today().isoformat()
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT legs FROM trial_parlays WHERE matchday_date = ? ORDER BY posted_at",
            (today,),
        ).fetchall()
        signatures: List[Tuple[str, ...]] = []
        for row in rows:
            try:
                legs = json.loads(row["legs"])
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(legs, list) and legs:
                signatures.append(tuple(str(leg) for leg in legs))
        return signatures
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Resolution logic
# ---------------------------------------------------------------------------

def _resolve_trial_parlays_sync() -> int:
    """Resolve unresolved trial parlays using the existing prediction_tracker.

    Checks API-Football results for each leg. Returns number of parlays resolved.
    """
    try:
        from prediction_tracker import _fetch_actual_results
    except ImportError:
        log.warning("prediction_tracker not available — cannot resolve trial parlays")
        return 0

    unresolved = _get_unresolved_parlays()
    if not unresolved:
        return 0

    resolved_count = 0

    for parlay in unresolved:
        matchday = parlay["matchday_date"]
        legs_json = parlay["legs"]
        combined_odds = parlay["combined_odds"]
        stake = parlay["stake"]

        try:
            legs = json.loads(legs_json)
        except (json.JSONDecodeError, TypeError):
            log.warning("Could not parse legs for trial parlay %d", parlay["id"])
            continue

        # Check if the matchday has passed (give 6 hours buffer for late matches)
        try:
            matchday_dt = datetime.strptime(matchday, "%Y-%m-%d")
            cutoff = matchday_dt + timedelta(hours=30)  # next day 6am
            if datetime.utcnow() < cutoff:
                continue  # Too early to resolve
        except ValueError:
            continue

        # Try to fetch results for this matchday
        try:
            results = _fetch_actual_results(matchday)
        except Exception as exc:
            log.warning("Failed to fetch results for %s: %s", matchday, exc)
            continue

        if not results:
            continue

        # Grade: check if all legs hit
        # We match leg text against results using fuzzy team name matching
        all_hit = True
        any_resolved = False

        for leg in legs:
            leg_lower = leg.lower()

            # Try to determine what the leg is about by pattern matching
            # This is best-effort — parlay legs are free-text from RAG output
            matched = False
            for result in results:
                home = result.get("home", "").lower()
                away = result.get("away", "").lower()
                home_goals = result.get("home_goals")
                away_goals = result.get("away_goals")

                if home_goals is None or away_goals is None:
                    continue

                total_goals = home_goals + away_goals

                # Check if this result's teams appear in the leg
                if not (home in leg_lower or away in leg_lower
                        or _fuzzy_team_match(home, leg_lower)
                        or _fuzzy_team_match(away, leg_lower)):
                    continue

                matched = True
                any_resolved = True

                # Grade the leg based on its type
                if _grade_leg(leg_lower, home, away, home_goals, away_goals,
                              total_goals, result):
                    pass  # This leg hit
                else:
                    all_hit = False
                break

            if not matched:
                # Could not match this leg to any result — skip entire parlay
                all_hit = False
                break

        if not any_resolved:
            continue

        # Record outcome
        outcome = "hit" if all_hit else "miss"
        profit = round((combined_odds * stake) - stake, 2) if all_hit else round(-stake, 2)
        _resolve_parlay(parlay["id"], outcome, profit)
        resolved_count += 1
        log.info("Trial parlay %d resolved as %s (profit: %.2f)",
                 parlay["id"], outcome, profit)

    return resolved_count


def _fuzzy_team_match(team: str, text: str) -> bool:
    """Check if a team name approximately appears in the text."""
    # Short team names (3+ chars) — check word boundary
    if len(team) >= 3:
        parts = team.split()
        for part in parts:
            if len(part) >= 4 and part in text:
                return True
    return False


def _grade_leg(
    leg: str,
    home: str, away: str,
    home_goals: int, away_goals: int,
    total_goals: int,
    result: Dict[str, Any],
) -> bool:
    """Grade a single parlay leg against actual results.

    Returns True if the leg hit, False otherwise. Best-effort parsing.
    """
    # --- Moneyline / Match Winner ---
    ml_patterns = [
        (r"(?:moneyline|ml|win|match winner)", None),
    ]
    if re.search(r"\b(?:ml|moneyline|win(?:ner)?)\b", leg):
        if home in leg:
            return home_goals > away_goals
        elif away in leg:
            return away_goals > home_goals
        elif "draw" in leg:
            return home_goals == away_goals

    # --- Over/Under Goals ---
    over_m = re.search(r"over\s+([\d.]+)\s*(?:goals?|total)", leg)
    if over_m:
        line = float(over_m.group(1))
        return total_goals > line

    under_m = re.search(r"under\s+([\d.]+)\s*(?:goals?|total)", leg)
    if under_m:
        line = float(under_m.group(1))
        return total_goals < line

    # --- BTTS ---
    if "btts" in leg:
        both_scored = home_goals > 0 and away_goals > 0
        if "yes" in leg:
            return both_scored
        elif "no" in leg:
            return not both_scored

    # --- Over/Under (generic — could be corners, cards, SoT, goals) ---
    # For corners/cards/SoT we don't have actuals in basic results
    # so we skip those — they'll remain unresolved
    generic_over = re.search(r"over\s+([\d.]+)", leg)
    if generic_over and not over_m:
        line = float(generic_over.group(1))
        # If the leg mentions goals or no other stat, assume goals
        if not any(kw in leg for kw in ["corner", "card", "shot", "sot", "foul"]):
            return total_goals > line

    generic_under = re.search(r"under\s+([\d.]+)", leg)
    if generic_under and not under_m:
        line = float(generic_under.group(1))
        if not any(kw in leg for kw in ["corner", "card", "shot", "sot", "foul"]):
            return total_goals < line

    # --- Handicap/Spread ---
    spread_m = re.search(r"([+-][\d.]+)\s*(?:spread|handicap|hcap)", leg)
    if spread_m:
        point = float(spread_m.group(1))
        diff = home_goals - away_goals
        if home in leg:
            return (diff + point) > 0
        elif away in leg:
            return (-diff + point) > 0

    # Could not determine — conservatively mark as miss
    log.debug("Could not grade leg: %s", leg[:100])
    return False


# ---------------------------------------------------------------------------
# Graduation check
# ---------------------------------------------------------------------------

async def _check_graduations(bot: commands.Bot):
    """Check all active trial users and graduate those with 2+ hits."""
    loop = asyncio.get_running_loop()

    active_users = await loop.run_in_executor(_trial_executor, _get_active_trial_users)
    if not active_users:
        return

    for user in active_users:
        discord_id = user["discord_id"]
        activated_at = user["activated_at"]

        hit_parlays = await loop.run_in_executor(
            _trial_executor, _get_hit_parlays_since, activated_at
        )
        all_resolved = await loop.run_in_executor(
            _trial_executor, _get_all_resolved_since, activated_at
        )

        hits = len(hit_parlays)
        total_seen = len(all_resolved)

        if hits < 2:
            continue

        # Calculate total profit from hit parlays
        total_profit = sum(p.get("profit", 0.0) for p in hit_parlays)

        # Graduate this user
        await loop.run_in_executor(
            _trial_executor,
            _graduate_user, discord_id, hits, total_seen, total_profit,
        )

        # Send graduation DM and remove role
        await _send_graduation_dm(bot, discord_id, hit_parlays, total_profit)

        log.info("Graduated trial user %s: %d hits, $%.2f profit",
                 discord_id, hits, total_profit)


async def _send_graduation_dm(
    bot: commands.Bot,
    discord_id: str,
    hit_parlays: List[Dict[str, Any]],
    total_profit: float,
):
    """Send graduation DM and remove TRIAL role."""
    try:
        discord_user = await bot.fetch_user(int(discord_id))
    except Exception as exc:
        log.warning("Could not fetch user %s for graduation DM: %s", discord_id, exc)
        return

    # Build hit details
    hit_lines = []
    for i, p in enumerate(hit_parlays[:2], 1):
        try:
            legs = json.loads(p["legs"])
            legs_str = " + ".join(legs[:3])
        except (json.JSONDecodeError, TypeError):
            legs_str = "Parlay"
        odds = p.get("combined_odds", 0)
        profit = p.get("profit", 0)
        hit_lines.append(
            f"**Parlay {i}:** {legs_str} @ {odds:.2f} \u2192 HIT (+${profit:.2f})"
        )

    hit_details = "\n".join(hit_lines) if hit_lines else "2 winning parlays!"

    em = discord.Embed(
        title="Your Free Trial Results",
        description=(
            "\u2705 **2 winning parlays landed!**\n\n"
            f"{hit_details}\n\n"
            f"**Total profit on $10 stakes: +${total_profit:.2f}**\n\n"
            "Ready to keep winning? Use `/subscribe` to unlock full access.\n\n"
            "\U0001f94a **Rookie** \u2014 $9.99/mo (5 commands/day)\n"
            "\U0001f4a1 **Tipster** \u2014 $14.99/mo (35 commands/day)\n"
            "\U0001f40b **Whale** \u2014 $29.99/mo (unlimited + AI assistant)"
        ),
        color=COLOR_GREEN,
    )
    em.set_footer(text="Spix's Picks | Your trial has ended")

    try:
        await discord_user.send(embed=em)
        log.info("Sent graduation DM to %s", discord_id)
    except discord.Forbidden:
        log.warning("Cannot DM user %s (DMs disabled)", discord_id)
    except Exception as exc:
        log.warning("Failed to send graduation DM to %s: %s", discord_id, exc)

    # Remove TRIAL role from all mutual guilds
    for guild in bot.guilds:
        try:
            member = guild.get_member(int(discord_id))
            if member is None:
                member = await guild.fetch_member(int(discord_id))
        except Exception:
            continue

        if member is None:
            continue

        trial_role = discord.utils.find(
            lambda r: r.name.upper() == ROLE_TRIAL.upper(), guild.roles
        )
        if trial_role and trial_role in member.roles:
            try:
                await member.remove_roles(trial_role, reason="Trial graduated")
                log.info("Removed TRIAL role from %s in %s", member, guild.name)
            except Exception as exc:
                log.warning("Failed to remove TRIAL role from %s: %s", member, exc)


# ---------------------------------------------------------------------------
# Embed builders
# ---------------------------------------------------------------------------

def _build_trial_parlay_embed(
    legs: List[str],
    combined_odds: float,
    parlay_num: int,
    date_str: str,
) -> discord.Embed:
    """Build a clean embed for a trial/free-pick parlay."""
    em = discord.Embed(
        title=f"\U0001f3af  Free Pick #{parlay_num}",
        description=f"**{date_str}** \u2022 Low-risk parlay",
        color=COLOR_GREEN,
    )

    for i, leg in enumerate(legs, 1):
        # Clean up the leg for display
        display = leg.strip()
        if len(display) > 200:
            display = display[:197] + "..."
        em.add_field(
            name=f"Leg {i}",
            value=display,
            inline=False,
        )

    em.add_field(
        name="Combined Odds",
        value=f"**{combined_odds:.2f}x**",
        inline=True,
    )
    em.add_field(
        name="Stake",
        value="$10.00 (tracked)",
        inline=True,
    )
    em.add_field(
        name="Potential Profit",
        value=f"+${(combined_odds * 10) - 10:.2f}",
        inline=True,
    )

    em.set_footer(text="Spix's Picks | Free Trial Pick | Use /subscribe for full access")
    return em


def _build_resolution_embed(
    parlay: Dict[str, Any],
    outcome: str,
) -> discord.Embed:
    """Build a resolution embed for a trial parlay."""
    try:
        legs = json.loads(parlay["legs"])
    except (json.JSONDecodeError, TypeError):
        legs = ["Unknown"]

    odds = parlay.get("combined_odds", 0)
    profit = parlay.get("profit", 0)

    if outcome == "hit":
        em = discord.Embed(
            title="\u2705  Free Pick \u2014 HIT!",
            description=(
                f"**+${profit:.2f}** profit on $10 stake\n\n"
                + "\n".join(f"\u2022 {leg}" for leg in legs)
            ),
            color=COLOR_GREEN,
        )
    else:
        em = discord.Embed(
            title="\u274c  Free Pick \u2014 Miss",
            description=(
                f"-$10.00\n\n"
                + "\n".join(f"\u2022 {leg}" for leg in legs)
            ),
            color=COLOR_RED,
        )

    em.add_field(name="Odds", value=f"{odds:.2f}x", inline=True)
    em.set_footer(text="Spix's Picks | Use /subscribe for full access")
    return em


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class TrialCog(commands.Cog, name="trial"):
    """Trial / free-picks user acquisition funnel."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._posted_today: Dict[str, str] = {}

        # Initialize DB tables
        try:
            _init_tables()
            log.info("Trial DB tables initialized")
        except Exception as exc:
            log.error("Failed to initialize trial DB: %s", exc)

    def _already_posted(self, task_name: str) -> bool:
        return self._posted_today.get(task_name) == date.today().isoformat()

    def _mark_posted(self, task_name: str):
        self._posted_today[task_name] = date.today().isoformat()

    async def cog_load(self):
        if CHANNEL_FREE_PICKS:
            self.post_trial_parlays.start()
            self.resolve_trial_parlays.start()
            log.info("Trial tasks started (channel=%d)", CHANNEL_FREE_PICKS)
        else:
            log.warning("DISCORD_CHANNEL_FREE_PICKS not configured — trial tasks disabled")

    async def cog_unload(self):
        if self.post_trial_parlays.is_running():
            self.post_trial_parlays.cancel()
        if self.resolve_trial_parlays.is_running():
            self.resolve_trial_parlays.cancel()

    # ==================================================================
    # /trial command
    # ==================================================================

    @app_commands.command(
        name="trial",
        description="Activate your free trial to see our free picks",
    )
    async def trial_cmd(self, interaction: discord.Interaction) -> None:
        if not interaction.guild:
            await interaction.response.send_message(
                "This command can only be used in a server.", ephemeral=True,
            )
            return

        member = interaction.user
        if not isinstance(member, discord.Member):
            await interaction.response.send_message(
                "Could not determine your server roles.", ephemeral=True,
            )
            return

        discord_id = str(member.id)
        member_roles_lower = {r.name.lower() for r in member.roles}

        # Check if already a paid subscriber
        if member_roles_lower & _PAID_ROLE_NAMES:
            em = discord.Embed(
                title="Already Subscribed",
                description=(
                    "You already have a paid subscription!\n"
                    "You have full access to all channels and features."
                ),
                color=COLOR_BLUE,
            )
            await interaction.response.send_message(embed=em, ephemeral=True)
            return

        # Check if already has TRIAL role
        has_trial = any(r.name.upper() == ROLE_TRIAL.upper() for r in member.roles)
        if has_trial:
            em = discord.Embed(
                title="Trial Already Active",
                description=(
                    "You're already on a free trial!\n\n"
                    "Head to the **#free-picks** channel to see today's picks."
                ),
                color=COLOR_YELLOW,
            )
            await interaction.response.send_message(embed=em, ephemeral=True)
            return

        # Check if already graduated
        existing = await asyncio.get_running_loop().run_in_executor(
            _trial_executor, _get_trial_user, discord_id
        )
        if existing and existing.get("graduated"):
            em = discord.Embed(
                title="Trial Completed",
                description=(
                    "Your free trial has ended.\n\n"
                    "Use `/subscribe` to unlock full access and keep winning!"
                ),
                color=COLOR_PURPLE,
            )
            await interaction.response.send_message(embed=em, ephemeral=True)
            return

        # Activate trial: assign role + insert DB record
        trial_role = discord.utils.find(
            lambda r: r.name.upper() == ROLE_TRIAL.upper(),
            interaction.guild.roles,
        )
        if not trial_role:
            log.error("TRIAL role '%s' not found in guild", ROLE_TRIAL)
            await interaction.response.send_message(
                "Trial system is not configured yet. Please contact an admin.",
                ephemeral=True,
            )
            return

        try:
            await member.add_roles(trial_role, reason="Trial activated via /trial")
        except discord.Forbidden:
            log.error("Bot lacks permission to assign TRIAL role")
            await interaction.response.send_message(
                "I don't have permission to assign roles. Please contact an admin.",
                ephemeral=True,
            )
            return
        except Exception as exc:
            log.error("Failed to assign TRIAL role: %s", exc)
            await interaction.response.send_message(
                "Something went wrong. Please try again later.",
                ephemeral=True,
            )
            return

        # Insert into DB
        username = str(member)
        await asyncio.get_running_loop().run_in_executor(
            _trial_executor, _activate_trial, discord_id, username
        )

        em = discord.Embed(
            title="Trial Activated!",
            description=(
                "You now have access to **#free-picks** where we post 2 parlays "
                "every matchday. Once 2 of our picks land, we'll let you know "
                "how much you would've made.\n\n"
                "Head to **#free-picks** to see today's picks!"
            ),
            color=COLOR_GREEN,
        )
        em.set_footer(text="Spix's Picks | Free Trial")
        await interaction.response.send_message(embed=em, ephemeral=True)
        log.info("Trial activated for %s (%s)", member, discord_id)

        # Send detailed DM with channel links
        try:
            # Find channel mentions
            free_picks = None
            bot_commands = None
            subscribe = None
            if interaction.guild:
                for ch in interaction.guild.channels:
                    if ch.name == "free-picks":
                        free_picks = ch.mention
                    elif ch.name == "bot-commands":
                        bot_commands = ch.mention
                    elif ch.name == "subscribe":
                        subscribe = ch.mention

            fp = free_picks or "**#free-picks**"
            bc = bot_commands or "**#bot-commands**"
            sc = subscribe or "**#subscribe**"

            dm_em = discord.Embed(
                title="Your Free Trial Has Started",
                description=(
                    "Here's exactly how this works."
                ),
                color=COLOR_GREEN,
            )
            dm_em.add_field(
                name="What happens now",
                value=(
                    f"We post **2 safe parlays** in {fp} every matchday. "
                    "These are low-odds, high-confidence picks designed to hit.\n\n"
                    "We track every parlay automatically. When **2 of them land**, "
                    "we'll DM you with exactly how much you would've made on a $10 stake."
                ),
                inline=False,
            )
            dm_em.add_field(
                name="What you need to do",
                value=(
                    f"Nothing \u2014 just check {fp} before each matchday and follow along. "
                    "You don't need to place any bets during the trial. Just watch the results."
                ),
                inline=False,
            )
            dm_em.add_field(
                name="What happens after",
                value=(
                    "Once 2 parlays hit, we'll show you the profit and your trial ends. "
                    f"Use the profit to pay for your first month \u2014 check {sc} for plans.\n\n"
                    f"You can also explore the bot in {bc} \u2014 try `/market EPL goals` "
                    "to see a preview of what subscribers get."
                ),
                inline=False,
            )
            dm_em.set_footer(text="Spix's Picks | No card required. No time limit. Just results.")
            await member.send(embed=dm_em)
        except discord.Forbidden:
            log.warning("Can't DM %s (DMs disabled)", member)
        except Exception as exc:
            log.warning("Failed to send trial DM to %s: %s", member, exc)

    # ==================================================================
    # Scheduled task: Post trial parlays
    # ==================================================================

    @tasks.loop(minutes=30)
    async def post_trial_parlays(self, *, force: bool = False):
        """Post 2 low-odds trial parlays to #free-picks each matchday.

        Parameters
        ----------
        force : bool
            When True (manual trigger), skip posting-window and count guards.
        """
        if not force and self._already_posted("trial_parlays"):
            return

        ch = self.bot.get_channel(CHANNEL_FREE_PICKS)
        if not ch:
            log.warning("CHANNEL_FREE_PICKS not found — cannot post trial parlays")
            return

        first_kickoff, active_leagues = _get_todays_leagues()
        if not active_leagues:
            log.info("No active leagues today — skipping trial parlays")
            return

        if not force and not _is_posting_window(first_kickoff):
            return

        loop = asyncio.get_running_loop()
        existing_today = 0
        existing_signatures: List[Tuple[str, ...]] = []

        # Check we haven't already posted 2 today (skip on manual trigger)
        if not force:
            existing_today = await loop.run_in_executor(_trial_executor, _count_today_parlays)
            if existing_today >= 2:
                self._mark_posted("trial_parlays")
                return
            existing_signatures = await loop.run_in_executor(
                _trial_executor, _get_today_parlay_signatures
            )

        date_str = _today_phrase()
        today_iso = date.today().isoformat()
        parlays_posted = 0
        remaining_slots = 2 if force else max(0, 2 - existing_today)
        if remaining_slots <= 0:
            self._mark_posted("trial_parlays")
            return
        seen_signatures = set(existing_signatures)
        used_event_groups = []  # (event_id, market_group) — block same market+match across free picks

        for i, preset in enumerate(_trial_parlay_presets(date.today(), active_leagues)):
            if parlays_posted >= remaining_slots:
                break

            # Inject previously-used event+group combos as exclusions
            if used_event_groups:
                preset["request"].excluded_event_groups = list(
                    set(preset["request"].excluded_event_groups) | set(used_event_groups)
                )

            try:
                result = await _build_structured_trial_parlay(preset["request"])
            except ParlayBuildError as exc:
                log.warning(
                    "Trial parlay preset %d (%s) failed: %s | %s",
                    i + 1,
                    preset.get("name", "unnamed"),
                    exc.message,
                    "; ".join(exc.notes),
                )
                continue
            except Exception as exc:
                log.error(
                    "Trial parlay preset %d (%s) failed: %s",
                    i + 1,
                    preset.get("name", "unnamed"),
                    exc,
                    exc_info=True,
                )
                continue

            legs = _trial_leg_strings(result)
            combined_odds = float(getattr(result, "combined_odds", 0.0) or 0.0)
            if not legs:
                log.warning("Could not derive legs from structured trial parlay %d", i + 1)
                continue

            leg_signature = tuple(legs)
            if leg_signature in seen_signatures:
                log.info(
                    "Skipping duplicate trial parlay from preset %d (%s)",
                    i + 1,
                    preset.get("name", "unnamed"),
                )
                continue

            # Validate odds are in safe range
            if combined_odds < 1.2 or combined_odds > 2.0:
                log.warning(
                    "Trial parlay %d odds %.2f outside safe range, adjusting display",
                    i + 1, combined_odds,
                )
                # Still post it — the odds might just be display issue
                if combined_odds == 0:
                    combined_odds = 1.60  # Default safe fallback

            # Log to database
            await loop.run_in_executor(
                _trial_executor,
                _log_trial_parlay, today_iso, legs, combined_odds,
            )

            # Collect used event+group combos so subsequent parlays avoid same market+match
            for leg in result.selected_legs:
                used_event_groups.append((leg.event_id, leg.market_group))

            # Build and post embed
            parlays_posted += 1
            seen_signatures.add(leg_signature)
            parlay_num = existing_today + parlays_posted
            em = _build_trial_parlay_embed(legs, combined_odds, parlay_num, date_str)
            try:
                await ch.send(embed=em)
                log.info("Posted trial parlay %d (odds: %.2f, legs: %d)",
                         parlay_num, combined_odds, len(legs))
            except Exception as exc:
                log.error("Failed to post trial parlay: %s", exc)

        total_posted_today = existing_today + parlays_posted
        if total_posted_today >= 2:
            self._mark_posted("trial_parlays")
            log.info("Posted %d total trial parlays for %s", total_posted_today, today_iso)
        elif parlays_posted > 0:
            log.warning(
                "Only %d/2 trial parlays are available for %s so far; will retry on the next scheduler tick.",
                total_posted_today,
                today_iso,
            )
        else:
            log.warning("Could not generate any trial parlays for %s", today_iso)

    @post_trial_parlays.before_loop
    async def _wait_trial_parlays(self):
        await self.bot.wait_until_ready()

    # ==================================================================
    # Scheduled task: Resolve trial parlays + check graduations
    # ==================================================================

    @tasks.loop(time=dt_time(hour=23, minute=0))
    async def resolve_trial_parlays(self):
        """Resolve trial parlays and check for user graduations.

        Runs at 23:00 UTC daily — most European matches are done by then.
        """
        loop = asyncio.get_running_loop()

        # Step 1: Resolve unresolved parlays
        try:
            resolved = await loop.run_in_executor(
                _trial_executor, _resolve_trial_parlays_sync
            )
            if resolved > 0:
                log.info("Resolved %d trial parlays", resolved)
        except Exception as exc:
            log.error("Trial parlay resolution failed: %s", exc, exc_info=True)
            return

        # Step 2: Post resolution results to #free-picks
        if resolved > 0:
            ch = self.bot.get_channel(CHANNEL_FREE_PICKS)
            if ch:
                # Fetch recently resolved parlays
                conn = _get_db()
                try:
                    rows = conn.execute(
                        "SELECT * FROM trial_parlays WHERE resolved_at IS NOT NULL "
                        "AND date(resolved_at) = date('now') ORDER BY resolved_at"
                    ).fetchall()
                    for row in rows:
                        p = dict(row)
                        em = _build_resolution_embed(p, p["outcome"])
                        try:
                            await ch.send(embed=em)
                        except Exception as exc:
                            log.error("Failed to post resolution embed: %s", exc)
                finally:
                    conn.close()

        # Step 3: Check graduations (runs after resolution)
        try:
            await _check_graduations(self.bot)
        except Exception as exc:
            log.error("Graduation check failed: %s", exc, exc_info=True)

    @resolve_trial_parlays.before_loop
    async def _wait_resolve(self):
        await self.bot.wait_until_ready()

    # ==================================================================
    # Manual trigger support (owner only)
    # ==================================================================

    @app_commands.command(
        name="trial-trigger",
        description="Manually trigger trial parlay posting or resolution (owner only)",
    )
    @app_commands.describe(action="What to trigger")
    @app_commands.choices(action=[
        app_commands.Choice(name="Post trial parlays", value="post"),
        app_commands.Choice(name="Resolve parlays", value="resolve"),
        app_commands.Choice(name="Check graduations", value="graduate"),
    ])
    async def trial_trigger(
        self,
        interaction: discord.Interaction,
        action: app_commands.Choice[str],
    ):
        if not interaction.guild or interaction.user.id != interaction.guild.owner_id:
            await interaction.response.send_message(
                "Only the server owner can use this command.", ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True, ephemeral=True)

        if action.value == "post":
            # Reset the posted flag and run with force=True to skip window guards
            self._posted_today.pop("trial_parlays", None)
            await self.post_trial_parlays(force=True)
            await interaction.followup.send("Trial parlays posted.", ephemeral=True)

        elif action.value == "resolve":
            loop = asyncio.get_running_loop()
            resolved = await loop.run_in_executor(
                _trial_executor, _resolve_trial_parlays_sync
            )
            await interaction.followup.send(
                f"Resolved {resolved} trial parlays.", ephemeral=True,
            )

        elif action.value == "graduate":
            await _check_graduations(self.bot)
            await interaction.followup.send(
                "Graduation check complete.", ephemeral=True,
            )


# ---------------------------------------------------------------------------
# Cog loader
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(TrialCog(bot))
