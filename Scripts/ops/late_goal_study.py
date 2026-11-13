"""Late-goal study (tracker D103): how often another goal comes after 75', and what a fair "one more goal" price
does from there. Step 1 of a possible in-play side project; measurement only, with no provider call and no write.

Sources, all already on disk and opened read-only:
- goal and red card events: the staged player-history download of 30 Sep (2021/22 to Sep 2026) and the refresh's
  events since 20 Sep 2026 (``platform.db``, inside the data gate); the refresh's copy wins where both have a match;
- final scores, teams and kickoffs: ``platform.db``, in the same gated read;
- the pre-match goal expectation: our read at kickoff (D70), else football-data's closing over/under 2.5;
- the in-play first look: ``Index/live.db`` (82 matches, 4–12 Apr 2026);
- account-wide API-Football use per day: the odds archive's saved response headers.

Minutes are as the data shows them. A goal shown at 75' came before the clock reached 75:00, so "after 75'" is 76'
or later, stoppage time included; extra time is left out. Since 2025/26 the provider logs most stoppage goals as a
plain 90', so the minute-by-minute clock uses 2021–2024, which record the added minute.

    python -m Scripts.ops.late_goal_study --out Research/late-goal-study-<date>
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys

from Scripts.live.live_config import GOAL_TIME_DISTRIBUTION

ROOT = Path(__file__).resolve().parents[2]
STAGED = ROOT / "Index/history_staging/player-stats-2021-2026/platform.db"
FOOTBALL_DATA = ROOT / "Index/football_data.sqlite3"
LIVE_DB = ROOT / "Index/live.db"
ODDS_ARCHIVE = ROOT / "Index/odds_archive/odds.sqlite3"
RECENT_FROM = date(2026, 9, 20)  # the refresh has saved events since this backfill (D64)
READS_FROM = date(2026, 9, 5)    # the first Match Read

MINUTES = (75, 77, 80, 85)
GOALS = ("Normal Goal", "Own Goal", "Penalty")
REDS = ("Red Card", "Second Yellow card")
ADDED_MINUTE_SEASONS = (2021, 2022, 2023, 2024)  # their stoppage goals carry the added minute
BANDS = ((2.3, "under 2.3"), (2.7, "2.3 to 2.7"), (3.1, "2.7 to 3.1"), (math.inf, "3.1 or more"))
TARGETS = (1.80, 3.00)
START_PRICES = (1.40, 1.50, 1.60, 1.70, 1.80)
WINDOWS = (((0, 15), "1'–15'"), ((15, 30), "16'–30'"), ((30, 45), "31'–45' and stoppage"),
           ((45, 60), "46'–60'"), ((60, 75), "61'–75'"), ((75, 90), "76'–90' and stoppage"))
INPLAY_BOOKMAKER = "API-Football Live"  # the live feed; a snapshot with other books is the pre-match fallback
CALLS_PER_MATCH = 30  # one in-play price a minute from 70' to the final whistle


# --- one match -----------------------------------------------------------------------------------------------------

def shown_minute(elapsed, extra):
    """The minute the scoreboard showed, on the 90-minute clock: 45+2 stays 45 (first half), 90+3 is 93. None for
    extra time, a missing minute or the provider's negative sentinel."""
    if not isinstance(elapsed, int) or elapsed < 0 or elapsed > 90:
        return None
    if elapsed == 90 and isinstance(extra, int) and extra > 0:
        return 90 + extra
    return elapsed


def build_match(events, fixture):
    """Goals and red cards as (shown minute, side) from API-Football events, or None when a goal cannot be placed or
    the goals do not add up to the stored score. A goal counts for the team the event lists, own goals included (that
    rebuilds the stored score of 21,244 of 21,253 staged matches with goals)."""
    sides = {fixture["home_id"]: "home", fixture["away_id"]: "away"}
    goals, reds, cancelled = [], [], []
    for event in events:
        timing = event.get("time") or {}
        minute = shown_minute(timing.get("elapsed"), timing.get("extra"))
        side = sides.get((event.get("team") or {}).get("id"))
        kind, detail, player = event.get("type"), event.get("detail"), (event.get("player") or {}).get("id")
        if kind == "Goal" and detail in GOALS:
            if minute is None or side is None:
                return None
            goals.append((minute, side))
        elif kind == "Card" and detail in REDS and minute is not None and side is not None:
            reds.append((minute, side, player))
        elif kind == "Var" and detail == "Red card cancelled" and player is not None:
            cancelled.append(player)
    for player in cancelled:
        mine = [r for r in reds if r[2] == player]
        if mine:
            reds.remove(mine[-1])
    if [sum(s == "home" for _, s in goals), sum(s == "away" for _, s in goals)] != \
            [fixture["home_goals"], fixture["away_goals"]]:
        return None
    kept = {k: v for k, v in fixture.items() if k not in ("home_id", "away_id", "home_goals", "away_goals")}
    return {**kept, "goals": sorted(goals, key=lambda g: g[0]), "reds": [(m, s) for m, s, _ in reds]}


def score_at(match, minute):
    shown = [s for m, s in match["goals"] if m <= minute]
    return shown.count("home"), shown.count("away")


def goal_after(match, minute):
    return any(m > minute for m, _ in match["goals"])


def red_before(match, minute):
    return any(m <= minute for m, _ in match["reds"])


def gap_group(home, away):
    return {0: "level", 1: "one apart"}.get(abs(home - away), "two or more")


def goals_group(total):
    return str(total) if total < 4 else "4+"


def expectation_band(expected):
    return next(label for top, label in BANDS if expected < top)


# --- rates ---------------------------------------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    """The 95% Wilson interval of k in n."""
    if not n:
        return None, None
    p, z2 = k / n, z * z
    centre, spread = p + z2 / (2 * n), z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (centre - spread) / (1 + z2 / n), (centre + spread) / (1 + z2 / n)


def rate_row(group, flags):
    flags = list(flags)
    k, n = sum(flags), len(flags)
    low, high = wilson(k, n)
    return {"group": group, "matches": n, "with_goal": k, "rate": k / n if n else None, "low": low, "high": high}


def split(matches, minute, key, order=None):
    """Another goal after ``minute``, by ``key(match, home, away)`` on the score at that minute (None leaves a match
    out). Rows in ``order``, else largest first."""
    groups = defaultdict(list)
    for m in matches:
        label = key(m, *score_at(m, minute))
        if label is not None:
            groups[label].append(goal_after(m, minute))
    rows = [rate_row(label, flags) for label, flags in groups.items()]
    if order:
        return sorted(rows, key=lambda r: order.index(r["group"]) if r["group"] in order else len(order))
    return sorted(rows, key=lambda r: -r["matches"])


# --- the fair-price clock ------------------------------------------------------------------------------------------

def minute_intensity(matches, start=75):
    """Goals per match shown at each minute after ``start``, stoppage minutes (90+X as 90+X) included."""
    counts = Counter(m for match in matches for m, _ in match["goals"] if m > start)
    if not counts:
        return {}
    return {minute: counts[minute] / len(matches) for minute in range(start + 1, max(counts) + 1)}


def price_clock(intensity, start_price, targets=TARGETS, start=75):
    """A match whose fair "one more goal" price at ``start`` is ``start_price``: with no goal, the first minute its
    fair price reaches each target, the chance no goal came first (the bet fires), and its chance then.

    The goals still to come follow the pooled shape of ``intensity``; only their level is the match's own (from its
    price, a Poisson: chance = 1 - exp(-goals left))."""
    minutes = sorted(m for m in intensity if m > start)
    total = sum(intensity[m] for m in minutes)
    lam = -math.log(1 - 1 / start_price)

    def left(t):
        return lam * sum(intensity[m] for m in minutes if m > t) / total

    rows = []
    for target in targets:
        need = -math.log(1 - 1 / target)
        minute = next((t for t in [start] + minutes if left(t) <= need + 1e-12), None)
        rows.append({"target": target, "minute": minute,
                     "fires": None if minute is None else math.exp(-(lam - left(minute))),
                     "chance_then": None if minute is None else 1 - math.exp(-left(minute))})
    return {"start_price": start_price, "lambda": lam, "targets": rows}


def clock_table(matches, minutes, start=75):
    """For each minute: the chance of another goal after it, and the same among matches with no goal since
    ``start`` (what a bettor waiting for the price to drift faces)."""
    rows = []
    for minute in minutes:
        quiet = [m for m in matches if not any(start < g <= minute for g, _ in m["goals"])]
        after = [goal_after(m, minute) for m in quiet]
        rows.append({"minute": minute, "chance_after": sum(goal_after(m, minute) for m in matches) / len(matches),
                     "quiet_since_75": len(quiet), "chance_after_if_quiet": sum(after) / len(after) if after else None})
    return rows


# --- the pre-match expectation -------------------------------------------------------------------------------------

def devig_over(over, under):
    return (1 / over) / (1 / over + 1 / under)


def lambda_from_over25(p_over):
    """The Poisson total whose chance of 3 or more goals is ``p_over``."""
    low, high = 0.05, 10.0
    for _ in range(80):
        mid = (low + high) / 2
        if 1 - math.exp(-mid) * (1 + mid + mid * mid / 2) < p_over:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def goals_projection(read):
    """The read's expected total goals as its card shows it (``match_read_metrics``): the market-anchored total where
    the read has one, else the model's projection."""
    for result in (read.get("data") or {}).get("canonical_results") or ():
        market = result.get("market") or {}
        if market.get("group") == "goals" and market.get("key") == "totals":
            anchored = (((result.get("context") or {}).get("market_anchor") or {}).get("goal_figures") or {})
            value = anchored.get("total_goals")
            if not isinstance(value, (int, float)):
                value = (result.get("projection") or {}).get("value")
            return float(value) if isinstance(value, (int, float)) else None
    return None


def calibration(matches, minute, *, share, groups=5):
    """Pre-match expectation against what happened: matches in ``groups`` equal parts by expectation, each with the
    chance it implies (1 - exp(-expected goals x the share of goals after ``minute``)) and the share with a goal."""
    ranked = sorted((m for m in matches if m.get("expectation") is not None), key=lambda m: m["expectation"])
    rows = []
    for i in range(groups):
        part = ranked[len(ranked) * i // groups: len(ranked) * (i + 1) // groups]
        if not part:
            continue
        row = rate_row(f"{part[0]['expectation']:.2f} to {part[-1]['expectation']:.2f}",
                       (goal_after(m, minute) for m in part))
        rows.append({**row, "happened": row["rate"],
                     "predicted": statistics.fmean(1 - math.exp(-m["expectation"] * share) for m in part)})
    return rows


def difference_interval(a, b):
    """The difference of two rates (rows from ``rate_row``) with a normal 95% interval."""
    diff = a["rate"] - b["rate"]
    se = math.sqrt(a["rate"] * (1 - a["rate"]) / a["matches"] + b["rate"] * (1 - b["rate"]) / b["matches"])
    return {"difference": diff, "low": diff - 1.96 * se, "high": diff + 1.96 * se}


# --- the live engine's timing table --------------------------------------------------------------------------------

def timing_shares(matches):
    """The share of goals in each of the live engine's windows (stoppage time in the half it belongs to)."""
    counts = Counter()
    for m in matches:
        for minute, _ in m["goals"]:
            counts[next(i for i, ((lo, hi), _) in enumerate(WINDOWS) if minute <= hi or i == len(WINDOWS) - 1)] += 1
    total = sum(counts.values())
    rows = []
    for i, (window, label) in enumerate(WINDOWS):
        low, high = wilson(counts[i], total)
        rows.append({"window": label, "goals": counts[i], "share": counts[i] / total if total else None,
                     "low": low, "high": high, "engine": GOAL_TIME_DISTRIBUTION[window]})
    return rows


# --- the in-play first look ----------------------------------------------------------------------------------------

def one_more_goal(odds, total):
    """The live "one more goal" price (over ``total`` + 0.5) and its fair chance, or None when the snapshot is the
    pre-match fallback or its lines do not fit the score (a stale score, or a settled line still listed)."""
    if odds.get("bookmakers_count") != 1:
        return None
    offers = {}
    for line in (odds.get("totals") or {}).get("goals") or ():
        live = [o for o in line.get("offers") or () if o.get("bookmaker") == INPLAY_BOOKMAKER and o.get("over_odds")]
        if live and float(line["line"]) % 1 == 0.5:
            offers[float(line["line"])] = live
    if not offers or min(offers) != total + 0.5:
        return None
    chosen = offers[total + 0.5]
    fair = [o["fair_over_prob"] for o in chosen if o.get("fair_over_prob") is not None]
    return statistics.median(o["over_odds"] for o in chosen), statistics.median(fair) if fair else None


def inplay_path(snaps, final_total, goal_minutes=None):
    """One match's price at 75' (the snapshot nearest it, 73'–77'), the first minute its price reached each target
    with no goal since, and whether another goal came after 75'. None without a price near 75'."""
    near = [s for s in snaps if 73 <= s["minute"] <= 77]
    if not near:
        return None
    base = min(near, key=lambda s: (abs(s["minute"] - 75), s["minute"]))
    reached = {str(t): None for t in TARGETS}
    for s in sorted(snaps, key=lambda s: s["minute"]):
        if s["minute"] < 75:
            continue
        if s["total"] != base["total"]:
            break  # a goal: the strategy's wait is over
        for t in TARGETS:
            if reached[str(t)] is None and s["price"] >= t:
                reached[str(t)] = s["minute"]
    after = any(m > 75 for m in goal_minutes) if goal_minutes is not None else final_total > base["total"]
    return {"minute_75": base["minute"], "price_75": base["price"], "fair_prob_75": base["fair_prob"],
            "total_75": base["total"], "qualifies": base["price"] < TARGETS[0], "reached": reached,
            "goal_after_75": after}


def _time(text):
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def load_inplay(path=LIVE_DB):
    """Each match's consistent in-play snapshots ({minute, total, price, fair_prob}) and its final total."""
    out = {}
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        leagues = dict(db.execute("SELECT fixture_id, league FROM live_fixtures"))
        states = defaultdict(list)
        for fid, at, status, elapsed, raw in db.execute(
                "SELECT fixture_id, captured_at, status, elapsed_min, state_json FROM live_state_snapshots "
                "ORDER BY fixture_id, captured_at"):
            state = json.loads(raw)
            total = (state.get("home_stats") or {}).get("goals", 0) + (state.get("away_stats") or {}).get("goals", 0)
            states[fid].append((_time(at), status, elapsed, total))
        for fid, at, raw in db.execute("SELECT fixture_id, captured_at, odds_json FROM live_odds_snapshots "
                                       "ORDER BY fixture_id, captured_at"):
            match = out.setdefault(fid, {"league": leagues.get(fid), "snaps": [], "final_total": None})
            moment = _time(at)
            near = [s for s in states.get(fid, ()) if abs((s[0] - moment).total_seconds()) <= 30]
            if not near:
                continue
            _, status, elapsed, total = min(near, key=lambda s: abs((s[0] - moment).total_seconds()))
            price = one_more_goal(json.loads(raw), total) if status == "2H" and elapsed is not None else None
            if price:
                match["snaps"].append({"minute": float(elapsed), "total": total, "price": price[0],
                                       "fair_prob": price[1]})
        for fid, rows in states.items():
            finished = [total for _, status, _, total in rows if status == "FT"]
            if fid in out and finished:
                out[fid]["final_total"] = finished[-1]
    return out


def inplay_summary(paths):
    """The first look: matches with a price at 75', those under 1.80, how often each target was reached, and the
    fair chance at 75' against what happened."""
    priced = [p for p in paths if p]
    qualifying = [p for p in priced if p["qualifies"]]

    def look(rows):
        happened = rate_row("", (p["goal_after_75"] for p in rows))
        fair = [p["fair_prob_75"] for p in rows if p["fair_prob_75"] is not None]
        return {"matches": len(rows), "reached_1.8": sum(p["reached"]["1.8"] is not None for p in rows),
                "reached_3.0": sum(p["reached"]["3.0"] is not None for p in rows),
                "goal_after_75": happened["with_goal"], "happened": happened["rate"], "low": happened["low"],
                "high": happened["high"], "fair_chance_75": statistics.fmean(fair) if fair else None,
                "median_price_75": statistics.median(p["price_75"] for p in rows) if rows else None}
    return {"with_price_at_75": look(priced), "under_1.80": look(qualifying)}


# --- step 3's error bars and step 2's cost -------------------------------------------------------------------------

def roi_half_width(price, p, n):
    """Half the 95% interval of the return per bet over ``n`` bets at ``price`` that win with chance ``p``."""
    return 1.96 * price * math.sqrt(p * (1 - p) / n)


def api_use_by_day(rows):
    """Account-wide API-Football use per UTC day, read from saved response headers [(requested_at, headers_json)]:
    the day's limit less its lowest remaining count, and the time of its last reading."""
    days = {}
    for at, raw in rows:
        headers = {k.lower(): v for k, v in (json.loads(raw) if raw else {}).items()}
        try:
            limit = int(headers["x-ratelimit-requests-limit"])
            remaining = int(headers["x-ratelimit-requests-remaining"])
        except (KeyError, TypeError, ValueError):
            continue
        moment = _time(at).astimezone(timezone.utc)
        day = days.setdefault(moment.date().isoformat(), {"used": 0, "limit": limit, "last_seen": None, "_last": None})
        if limit - remaining >= day["used"]:
            day["used"], day["limit"] = limit - remaining, limit
        if day["_last"] is None or moment > day["_last"]:
            day["_last"], day["last_seen"] = moment, moment.strftime("%H:%M")
    return {d: {k: v for k, v in day.items() if k != "_last"} for d, day in sorted(days.items())}


def load_api_use(path=ODDS_ARCHIVE):
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        return api_use_by_day(db.execute("SELECT requested_at, headers_json FROM requests "
                                         "WHERE headers_json IS NOT NULL"))


# --- loading -------------------------------------------------------------------------------------------------------

def load_platform(since=RECENT_FROM, *, gate=None, session_factory=None):
    """Finished fixtures from 2021/22 on (score, teams, league, season, kickoff) by API-Football id, the ids with
    events saved by the refresh since ``since``, and their goal, card and VAR events; one gated read."""
    from sqlalchemy import text

    from Scripts.data_platform.db import get_sessionmaker
    from Scripts.data_platform.services.refresh_coordination import data_access
    with (gate or data_access)(), (session_factory or get_sessionmaker())() as session:
        fixtures = {str(r[0]): {"fixture_id": str(r[0]), "home_id": r[1], "away_id": r[2], "home_goals": r[3],
                                "away_goals": r[4], "league": r[5], "season": r[6], "kickoff": str(r[7])}
                    for r in session.execute(text(
                        "SELECT f.api_football_id, ht.api_football_id, at.api_football_id, f.home_goals, "
                        "f.away_goals, c.code, s.year, f.kickoff_utc FROM fixtures f "
                        "JOIN teams ht ON ht.id = f.home_team_id JOIN teams at ON at.id = f.away_team_id "
                        "JOIN competitions c ON c.id = f.competition_id JOIN seasons s ON s.id = f.season_id "
                        "WHERE s.year >= 2021 AND f.status = 'FT' AND f.home_goals IS NOT NULL"))}
        start = datetime.combine(since, datetime.min.time())
        ids = {str(r[0]) for r in session.execute(text(
            "SELECT DISTINCT f.api_football_id FROM fixture_match_events e JOIN fixtures f ON f.id = e.fixture_id "
            "WHERE f.kickoff_utc >= :start"), {"start": start})}
        events = defaultdict(list)
        for api_id, raw in session.execute(text(
                "SELECT f.api_football_id, e.raw_json FROM fixture_match_events e "
                "JOIN fixtures f ON f.id = e.fixture_id WHERE f.kickoff_utc >= :start "
                "AND e.event_type IN ('Goal', 'Card', 'Var') ORDER BY e.fixture_id, e.event_index"), {"start": start}):
            events[str(api_id)].append(json.loads(raw) if isinstance(raw, str) else raw)
    return fixtures, ids, events


def load_staged(path=STAGED):
    """The staged download's matches with events (collected whole or with gaps flagged; an empty answer stays out,
    as a goalless match and missing coverage look alike) and their goal, card and VAR events."""
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        ids = {str(r[0]) for r in db.execute(
            "SELECT f.api_football_id FROM fixture_endpoint_collection x JOIN fixtures f ON f.id = x.fixture_id "
            "WHERE x.endpoint = '/fixtures/events' AND x.state IN ('ready', 'partial')")}
        events = defaultdict(list)
        for api_id, raw in db.execute(
                "SELECT f.api_football_id, e.raw_json FROM fixture_events e "
                "JOIN fixture_endpoint_collection x ON x.fixture_id = e.fixture_id AND x.archive_id = e.archive_id "
                "AND x.endpoint = '/fixtures/events' JOIN fixtures f ON f.id = e.fixture_id "
                "WHERE x.state IN ('ready', 'partial') AND e.event_type IN ('Goal', 'Card', 'Var') "
                "ORDER BY e.fixture_id, e.event_index"):
            events[str(api_id)].append(json.loads(raw))
    return ids, events


def load_market_expectations(path=FOOTBALL_DATA):
    """The expected total from each linked match's closing over/under 2.5: Pinnacle, else the market average."""
    prices = defaultdict(dict)
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        for api_id, book, selection, price in db.execute(
                "SELECT m.fixture_api_id, o.bookmaker, o.selection, o.price FROM odds o JOIN matches m ON m.id = "
                "o.match_id WHERE o.market = 'ou25' AND o.closing = 1 AND o.suspect = 0 AND m.fixture_api_id IS NOT "
                "NULL AND o.bookmaker IN ('Pinnacle', 'Market average') AND o.price > 1"):
            prices[(str(api_id), book)][selection] = price
    out = {}
    for book in ("Market average", "Pinnacle"):  # Pinnacle last, so it wins
        for (api_id, b), pair in prices.items():
            if b == book and {"over", "under"} <= set(pair):
                out[api_id] = lambda_from_over25(devig_over(pair["over"], pair["under"]))
    return out


def load_read_expectations(days, *, now, gate=None):
    """Our read's expected total for each match as the board showed it at kickoff (D70), by API-Football id; one
    gate hold per day, as the wins reel does."""
    scripts = str(ROOT / "Scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from web_app.routers import match_reads as board

    from Scripts.data_platform.services.refresh_coordination import data_access
    from Scripts.data_platform.settlement import provider_id
    service, out = board._get_match_read_service(), {}
    for day in days:
        with (gate or data_access)():
            for league in board.PUBLIC_MATCH_READ_LEAGUES:
                for read, _ in board._shown_at_kickoff(
                        board._load_matchday_reads(service, league=league, target_date=day), now=now):
                    api_id, expected = provider_id((read.get("fixture") or {}).get("event_id")), goals_projection(read)
                    if api_id and expected:
                        out[api_id] = expected
    return out


def assemble(fixtures, staged_ids, staged, recent_ids, recent, market, reads):
    """Every match with events, built, with its expectation and where it came from; and what was left out."""
    matches, left_out = [], Counter()
    for api_id in sorted(staged_ids | recent_ids):
        fixture = fixtures.get(api_id)
        if fixture is None:
            left_out["no finished fixture with a score"] += 1
            continue
        built = build_match(recent.get(api_id, []) if api_id in recent_ids else staged.get(api_id, []), fixture)
        if built is None:
            left_out["goals do not add up or cannot be placed"] += 1
            continue
        source = "read" if api_id in reads else "market" if api_id in market else None
        built["expectation"] = reads.get(api_id, market.get(api_id))
        built["expectation_source"] = source
        matches.append(built)
    return matches, dict(left_out)


# --- the study -----------------------------------------------------------------------------------------------------

def busiest_days(fixtures, season, top=5):
    days = Counter(f["kickoff"][:10] for f in fixtures.values() if f["season"] == season)
    weeks = {date.fromisoformat(d).isocalendar()[:2] for d in days}
    return {"days": days.most_common(top), "per_week": sum(days.values()) / len(weeks) if weeks else None}


def study(matches, *, inplay=None, api_use=None, fixtures=None):
    minute = MINUTES[0]
    coded = [m for m in matches if m["season"] in ADDED_MINUTE_SEASONS]
    with_expectation = [m for m in matches if m.get("expectation") is not None]
    goals_after = sum(sum(g > minute for g, _ in m["goals"]) for m in with_expectation)
    share = goals_after / max(1, sum(len(m["goals"]) for m in with_expectation))
    fifths = calibration(matches, minute, share=share)
    intensity = minute_intensity(coded, start=minute)
    remaining = {t: sum(v for m, v in intensity.items() if m > t) for t in MINUTES + (89, 90)}
    by_id = {m["fixture_id"]: m for m in matches}
    paths = []
    for fid, item in (inplay or {}).items():
        known = by_id.get(str(fid))
        path = inplay_path(item["snaps"], item["final_total"],
                           [g for g, _ in known["goals"]] if known else None) if item["snaps"] else None
        if path and (known or item["final_total"] is not None):
            paths.append({**path, "fixture_id": fid, "league": item["league"]})
    return {
        "matches": len(matches),
        "seasons": dict(sorted(Counter(m["season"] for m in matches).items())),
        "overall": [{**rate_row(f"after {t}'", (goal_after(m, t) for m in matches)), "minute": t} for t in MINUTES],
        "splits": {
            "score at 75'": split(matches, minute, lambda m, h, a: gap_group(h, a),
                                  ["level", "one apart", "two or more"]),
            "goals so far": split(matches, minute, lambda m, h, a: goals_group(h + a), ["0", "1", "2", "3", "4+"]),
            "pre-match expectation": split(
                matches, minute, lambda m, h, a: None if m.get("expectation") is None
                else expectation_band(m["expectation"]), [label for _, label in BANDS]),
            "league": split(matches, minute, lambda m, h, a: m["league"]),
            "season": split(matches, minute, lambda m, h, a: m["season"], sorted({m["season"] for m in matches})),
            "red card before 75'": split(matches, minute, lambda m, h, a: "yes" if red_before(m, minute) else "no",
                                         ["no", "yes"]),
            "score at 75' and expectation": split(
                matches, minute, lambda m, h, a: None if m.get("expectation") is None
                else f"{gap_group(h, a)}, {expectation_band(m['expectation'])}",
                [f"{g}, {b}" for g in ("level", "one apart", "two or more") for _, b in BANDS]),
        },
        "expectation_sources": dict(Counter(m.get("expectation_source") or "none" for m in matches)),
        "clock": {"seasons": list(ADDED_MINUTE_SEASONS), "matches": len(coded),
                  "goals_left_per_match": remaining,
                  "table": clock_table(coded, (75, 77, 80, 82, 85, 87, 89, 90)),
                  "poisson_check": [{"minute": t, "poisson": 1 - math.exp(-remaining[t]),
                                     "happened": sum(goal_after(m, t) for m in coded) / len(coded)}
                                    for t in MINUTES],
                  "prices": [price_clock(intensity, p) for p in START_PRICES]},
        "calibration": {"share_after_75": share, "fifths": fifths,
                        "top_minus_bottom": difference_interval(fifths[-1], fifths[0]) if len(fifths) > 1 else None,
                        "reads_only": calibration([m for m in matches if m.get("expectation_source") == "read"],
                                                  minute, share=share, groups=3)},
        "timing": {"windows": timing_shares(matches),
                   "after_90_share": sum(sum(g > 90 for g, _ in m["goals"]) for m in coded)
                   / max(1, sum(len(m["goals"]) for m in coded))},
        "inplay": {"summary": inplay_summary(paths), "paths": paths},
        "error_bars": [{"bets": n, "at_1.80": roi_half_width(1.80, 1 / 1.80, n),
                        "at_3.00": roi_half_width(3.00, 1 / 3.00, n)} for n in (100, 300, 500, 1000, 2000)],
        "cost": {"api_use_by_day": api_use or {}, "calls_per_match": CALLS_PER_MATCH,
                 "busiest_2025_26": busiest_days(fixtures, 2025) if fixtures else None},
    }


def _pct(x, digits=0):
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def _rate(r):
    return f"{_pct(r['rate'])} ({_pct(r['low'])}–{_pct(r['high'])})"


def summary(result):
    lines = [f"# Late-goal study ({result['matches']} matches)", "",
             "Seasons: " + ", ".join(f"{s}: {n}" for s, n in result["seasons"].items()), "",
             "## Another goal after the minute (stoppage time included, 95% interval)", "",
             "| From | Matches | With another goal | Rate |", "|---|---|---|---|"]
    lines += [f"| {r['minute']}' | {r['matches']} | {r['with_goal']} | {_rate(r)} |" for r in result["overall"]]
    for name, rows in result["splits"].items():
        lines += ["", f"## After 75', by {name}", "", "| Group | Matches | Rate |", "|---|---|---|"]
        lines += [f"| {r['group']} | {r['matches']} | {_rate(r)} |" for r in rows]
    clock = result["clock"]
    lines += ["", f"## The clock ({clock['matches']} matches, {clock['seasons'][0]}–{clock['seasons'][-1]})", "",
              "| Minute | Chance of another goal after it | If none since 75' (matches) |", "|---|---|---|"]
    lines += [f"| {r['minute']}' | {_pct(r['chance_after'])} | {_pct(r['chance_after_if_quiet'])} "
              f"({r['quiet_since_75']}) |" for r in clock["table"]]
    lines.append("\nA Poisson on the pooled clock against what happened: " + ", ".join(
        f"after {r['minute']}' {_pct(r['poisson'], 1)} vs {_pct(r['happened'], 1)}" for r in clock["poisson_check"]))
    lines += ["", "| Fair price at 75' | Reaches 1.80 at | Fires | Reaches 3.00 at | Fires |", "|---|---|---|---|---|"]
    for p in clock["prices"]:
        a, b = p["targets"]
        lines.append(f"| {p['start_price']:.2f} | {a['minute']}' | {_pct(a['fires'])} | {b['minute']}' | "
                     f"{_pct(b['fires'])} |")
    cal = result["calibration"]
    lines += ["", f"## Pre-match expectation against what happened (share of goals after 75': "
              f"{_pct(cal['share_after_75'], 1)})", "", "| Expected goals | Matches | Predicted | Happened |",
              "|---|---|---|---|"]
    lines += [f"| {r['group']} | {r['matches']} | {_pct(r['predicted'])} | {_rate(r)} |" for r in cal["fifths"]]
    if cal["top_minus_bottom"]:
        d = cal["top_minus_bottom"]
        lines.append(f"\nTop fifth minus bottom fifth: {d['difference'] * 100:+.1f} pts "
                     f"({d['low'] * 100:+.1f} to {d['high'] * 100:+.1f}).")
    lines.append("\nExpectation from: " + ", ".join(f"{k} {v}" for k, v in result["expectation_sources"].items()))
    lines += ["", "Our reads at kickoff only, in thirds:", "", "| Expected goals | Matches | Predicted | Happened |",
              "|---|---|---|---|"]
    lines += [f"| {r['group']} | {r['matches']} | {_pct(r['predicted'])} | {_rate(r)} |" for r in cal["reads_only"]]
    lines += ["", "## The live engine's timing table", "", "| Window | Our goals | Our share | Engine |",
              "|---|---|---|---|"]
    lines += [f"| {r['window']} | {r['goals']} | {_pct(r['share'], 1)} ({_pct(r['low'], 1)}–{_pct(r['high'], 1)}) "
              f"| {_pct(r['engine'], 1)} |" for r in result["timing"]["windows"]]
    lines.append(f"\nGoals after 90' (2021–2024): {_pct(result['timing']['after_90_share'], 1)}.")
    inplay = result["inplay"]["summary"]
    lines += ["", "## In-play first look (Index/live.db, Apr 2026; a small sample)", "",
              "| Matches | n | Reached 1.80 | Reached 3.00 | Goal after 75' | Fair chance at 75' | Median price |",
              "|---|---|---|---|---|---|---|"]
    for name, row in inplay.items():
        lines.append(f"| {name} | {row['matches']} | {row['reached_1.8']} | {row['reached_3.0']} | "
                     f"{_pct(row['happened'])} ({_pct(row['low'])}–{_pct(row['high'])}) | "
                     f"{_pct(row['fair_chance_75'])} | {row['median_price_75'] or 'n/a'} |")
    lines += ["", "## Error bars on the return per bet (95%)", "", "| Bets | At 1.80 | At 3.00 |", "|---|---|---|"]
    lines += [f"| {r['bets']} | ±{r['at_1.80'] * 100:.1f} pts | ±{r['at_3.00'] * 100:.1f} pts |"
              for r in result["error_bars"]]
    cost = result["cost"]
    if cost["busiest_2025_26"]:
        busy = cost["busiest_2025_26"]
        lines += ["", "## Step 2's cost", "", f"Matches per week in 2025/26: {busy['per_week']:.0f}; busiest days: "
                  + ", ".join(f"{d} ({n})" for d, n in busy["days"]) + f". At {cost['calls_per_match']} calls a "
                  "match, a 50-match day is " + f"{50 * cost['calls_per_match']:,} calls."]
    lines += ["", "| Day (UTC) | Used | Limit | Last reading |", "|---|---|---|---|"]
    lines += [f"| {d} {date.fromisoformat(d):%a} | {v['used']:,} | {v['limit']:,} | {v['last_seen']} |"
              for d, v in cost["api_use_by_day"].items() if v["limit"] == 7500]
    if result.get("left_out"):
        lines.append("\nLeft out: " + ", ".join(f"{k}: {v}" for k, v in result["left_out"].items()))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-reads", action="store_true", help="leave out our reads' expectation (faster)")
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc)
    fixtures, recent_ids, recent = load_platform()
    staged_ids, staged = load_staged()
    market = load_market_expectations()
    reads = {}
    if not args.no_reads:
        days = sorted({date.fromisoformat(fixtures[i]["kickoff"][:10]) for i in staged_ids | recent_ids
                       if i in fixtures and fixtures[i]["kickoff"][:10] >= READS_FROM.isoformat()})
        reads = load_read_expectations(days, now=now)
    matches, left_out = assemble(fixtures, staged_ids, staged, recent_ids, recent, market, reads)
    result = {**study(matches, inplay=load_inplay(), api_use=load_api_use(), fixtures=fixtures),
              "left_out": left_out, "made_at": now.isoformat()}
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "late_goal_study.json").write_text(json.dumps(result, indent=2, default=str))
        (args.out / "summary.md").write_text(summary(result) + "\n")
    print(summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
