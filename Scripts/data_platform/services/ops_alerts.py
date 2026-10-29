"""Operations messages to the owner's private chat (owner, 28 Oct).

Every message goes through ``refresh_alerts._send``, so it reaches the same Telegram chat (or Discord webhook) as
the refresh watchdog. Problems make a sound: the Match Read worker, measurement or the true close failing several
runs in a row, a failed daily refresh and the day it stops retrying, OddsPapi's allowance running low, and (stage
4) price confirmation unable to check most of a day's picks. The day's rhythm arrives without a sound: the refresh
starting and finishing, Match Reads published (at most one message an hour) and the Monday scorecard.

Messages carry counts, names and times only, never error text (an OddsPapi error can hold its key). A message
that cannot be sent, or a hook that breaks, never fails the job. State per topic is kept in ``SPIX_OPS_ALERT_DIR``
(default ``Index/ops-alerts``).
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import functools
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
MATCH_READ_FAILURES = 2      # worker runs every 10 minutes; 14-27 Oct: 3 double failures, none longer
MEASUREMENT_FAILURES = 10    # runs every minute; short error runs are routine (up to 9 in a row)
MEASUREMENT_BUSY = 30        # runs in a row out of time before any request (a busy database), and ...
MEASUREMENT_BUSY_FOR = timedelta(hours=3)  # ... for this long; 29 Oct evening: 33 runs in 113 min, then 15 more
TRUE_CLOSE_FAILURES = 3      # runs hourly
RELEASE_EVERY = timedelta(hours=1)
ODDSPAPI_LOW = 50            # requests left on the account
REFRESH_TRIES = 6            # refresh_retry.MAX_ATTEMPTS
CONFIRMATION_FAULTS = frozenset({"fixture_list_error", "market_list_error", "history_error", "fixture_not_matched",
                                 "no_oddspapi_key"})
CONFIRMATION_MIN_FAULTS = 5
DEFERRED = frozenset({"refresh_deferred", "overlap_skipped", "disabled", "planned"})


def notify(message, *, silent=False):
    """Sends ``message`` to the owner's chat; True when delivered. Never raises."""
    try:
        from .refresh_alerts import _send
        return bool(_send(message, silent=silent))
    except Exception:
        logger.warning("Operations message not sent")
        return False


def _safe(hook):
    @functools.wraps(hook)
    def run(*args, **kwargs):
        try:
            return hook(*args, **kwargs)
        except Exception:
            logger.warning("Operations alert %s failed", hook.__name__)
            return None
    return run


def _folder():
    return Path(os.environ.get("SPIX_OPS_ALERT_DIR") or ROOT / "Index" / "ops-alerts")


@contextmanager
def _topic(name):
    """The topic's saved state, locked across processes and written back on exit."""
    folder = _folder()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    with (folder / f"{name}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            state = {}
        yield state
        temporary = path.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(state, indent=2, default=str))
        temporary.replace(path)


def streak(topic, *, failed, after, alert, recovered, lasting=None, now=None):
    """Counts failed runs in a row: one alert with a sound at ``after`` (and, with ``lasting``, once the first of
    them is that long ago), one quiet message when a run succeeds."""
    now = now or datetime.now(timezone.utc)
    with _topic(topic) as state:
        if failed:
            state["failures"] = state.get("failures", 0) + 1
            state.setdefault("since", now.isoformat())
            long_enough = lasting is None or now - datetime.fromisoformat(state["since"]) >= lasting
            if state["failures"] >= after and long_enough and not state.get("alerted"):
                state["alerted"] = notify(alert(state["failures"]))
        else:
            if state.get("alerted"):
                notify(recovered(state.get("failures", 0)), silent=True)
            state.clear()


def once(topic, key, message, *, silent=False):
    """Sends ``message`` unless this topic already delivered one for ``key`` (a day, say)."""
    with _topic(topic) as state:
        if state.get("sent") == key:
            return False
        if notify(message, silent=silent):
            state["sent"] = key
            return True
        return False


def _plural(n, word):
    return f"{n:,} {word}{'' if n == 1 else 'es' if word.endswith('ch') else 's'}"


def _local(now=None):
    return (now or datetime.now(timezone.utc)).astimezone()


# The Match Read worker -------------------------------------------------------------------------------------------

@_safe
def match_read_cycle(report, *, now=None, tz=None):
    """After each Match Read worker run; ``report`` is the cycle report as a dict."""
    if report.get("dry_run"):
        return
    errors = len(report.get("errors") or [])
    if errors or not report.get("skipped_reason"):  # a run skipped for the shut gate neither fails nor recovers
        streak("match-read-worker", failed=bool(errors), after=MATCH_READ_FAILURES,
               alert=lambda n: (f"Spix · the Match Read worker failed {n} runs in a row ({_plural(errors, 'error')} in "
                                "the last). Cards may be missing or out of date. "
                                "Log: Index/logs/match-read-worker.stdout.log"),
               recovered=lambda n: f"Spix · the Match Read worker is running normally again after {n} failed runs.")
    releases_due(release_items(report), now=now, tz=tz)


def release_items(report):
    """The cards a worker run made visible for the first time."""
    items = []
    for release in report.get("releases") or []:
        for card in release.get("released") or []:
            if card.get("delivery_created"):
                items.append({"league": release.get("league"), "date": release.get("date"),
                              "stage": release.get("stage"), "kickoff": release.get("kickoff"),
                              "fixture": release.get("fixture"), "fixture_id": str(card.get("fixture_id")),
                              "picks": int(card.get("recommendations_created") or 0)})
    return items


def releases_message(items, *, tz=None):
    """Published cards by day and league; a match read again (confirmed lineups) counts once, at its latest read."""
    latest, lineups = {}, set()
    for item in items:
        latest[item["fixture_id"]] = item
        if item.get("stage") == "confirmed_lineups":
            lineups.add(item["fixture_id"])

    def kickoff(item):
        value = item.get("kickoff")
        return datetime.fromisoformat(value).astimezone(tz) if value else None

    days = {}
    for item in latest.values():
        start = kickoff(item)
        day = start.date() if start else datetime.fromisoformat(item["date"]).date()
        days.setdefault(day, {}).setdefault(item["league"], []).append(item)
    lines = [f"Spix · Match Reads published: {_plural(len(latest), 'match')}, "
             f"{_plural(sum(i['picks'] for i in latest.values()), 'pick')}"]
    for day in sorted(days):
        lines.append(f"{day:%a} {day.day} {day:%b}")
        for league, group in sorted(days[day].items()):
            group.sort(key=lambda i: kickoff(i) or datetime.max.replace(tzinfo=timezone.utc))
            names = (", ".join(i.get("fixture") or i["fixture_id"] for i in group) if len(group) <= 3
                     else _plural(len(group), "match"))
            parts = [f"{league}: {names}", _plural(sum(i["picks"] for i in group), "pick")]
            if kickoff(group[0]):
                parts.append(f"first kickoff {kickoff(group[0]):%H:%M}")
            confirmed = sum(i["fixture_id"] in lineups for i in group)
            if confirmed:
                parts.append(f"{confirmed} with confirmed lineups")
            lines.append("  " + " · ".join(parts))
    return "\n".join(lines)[:3900]


@_safe
def releases_due(items, *, now=None, tz=None):
    """Adds a run's new cards and sends what has gathered, quietly, at most once every ``RELEASE_EVERY``."""
    now = now or datetime.now(timezone.utc)
    with _topic("match-reads-published") as state:
        pending = ((state.get("pending") or []) + list(items))[-1000:]
        last = state.get("last_sent")
        if pending and (last is None or now - datetime.fromisoformat(last) >= RELEASE_EVERY):
            if notify(releases_message(pending, tz=tz), silent=True):
                state.update(pending=[], last_sent=now.isoformat())
                return
        state["pending"] = pending


# The daily refresh -----------------------------------------------------------------------------------------------

def _took(seconds):
    minutes = int(seconds // 60)
    if minutes < 1:
        return "under a minute"
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


@_safe
def refresh_started(attempt=None):
    notify(f"Spix · daily refresh started{f' (try {attempt} of {REFRESH_TRIES} today)' if attempt else ''}.",
           silent=True)


@_safe
def refresh_finished(*, ok, seconds, attempt=None, now=None):
    if ok:
        notify(f"Spix · daily refresh done in {_took(seconds)}. Match Reads and measurement resume.", silent=True)
        return
    now = _local(now)
    if attempt is not None and attempt >= REFRESH_TRIES:
        refresh_gave_up(now=now)
        return
    retry = now.replace(minute=15, second=0, microsecond=0)
    retry = retry if retry > now else retry + timedelta(hours=1)
    notify(f"Spix · daily refresh failed after {_took(seconds)}. The data gate stays shut: no new Match Reads or "
           f"measurement until a refresh completes. The hourly job tries again at {retry:%H:%M}"
           f"{f' (try {attempt + 1} of {REFRESH_TRIES})' if attempt else ''}. "
           "Log: Index/logs/daily-data-refresh.stderr.log")


def refresh_gave_up(*, now=None):
    once("refresh-gave-up", _local(now).date().isoformat(),
         f"Spix · the daily refresh failed {REFRESH_TRIES} times today and has stopped trying. The data gate stays "
         "shut: no Match Reads or measurement until a refresh completes. Run it by hand: "
         "python -m Scripts.data_platform refresh-season --include-europe")


@_safe
def refresh_skipped(reason, *, gate_shut=None, now=None):
    """``refresh-season --if-due`` decided not to run: only the day's limit with the gate shut is news."""
    if not str(reason).startswith("attempt limit"):
        return
    if gate_shut is None:
        from .refresh_coordination import _paths
        gate_shut = _paths()[1].exists()
    if gate_shut:
        refresh_gave_up(now=now)


# OddsPapi, measurement, the true close -----------------------------------------------------------------------------

@_safe
def oddspapi_left(left, *, reserve):
    """After the true close reads the account (``left``: requests left of its allowance)."""
    levels = (("low", ODDSPAPI_LOW, f"Spix · OddsPapi: {left} requests left of the account's allowance. The true "
               f"close stops buying fixture lists at {reserve} left; at none left, price confirmation (stage 4) "
               "turns every pick into a no-bet."),
              ("reserve", reserve, f"Spix · OddsPapi: {left} requests left, so the true close has stopped buying "
               f"fixture lists (it keeps {reserve} in reserve). Closing prices for new weeks wait until the "
               "allowance renews."))
    with _topic("oddspapi") as state:
        sent = set(state.get("sent") or [])
        for level, threshold, message in levels:
            if left > threshold:
                sent.discard(level)
            elif level not in sent and notify(message):
                sent.add(level)
        state["sent"] = sorted(sent)


@_safe
def measurement_cycle(result, *, now=None):
    """After each measurement run (settlement and closing prices); deferred runs neither fail nor recover.

    A busy run (out of time before asking the provider anything, as on a busy match evening) is not a failure:
    it has its own, longer streak and leaves the failure streak as it was."""
    if result.get("status") in DEFERRED:
        return
    busy = bool(result.get("busy"))
    streak("measurement-busy", failed=busy, after=MEASUREMENT_BUSY, lasting=MEASUREMENT_BUSY_FOR, now=now,
           alert=lambda n: (f"Spix · measurement ran out of time {n} runs in a row over 3+ hours: the database is "
                            "busy, so results and closing prices are late. "
                            "Log: Index/logs/prediction-measurement.stdout.log"),
           recovered=lambda n: f"Spix · measurement is finishing its runs again after {n} busy runs.")
    if busy:
        return
    streak("measurement", failed=bool(result.get("errors")) or result.get("status") == "error",
           after=MEASUREMENT_FAILURES,
           alert=lambda n: (f"Spix · measurement failed {n} runs in a row: results and closing prices are not being "
                            "recorded. Log: Index/logs/prediction-measurement.stdout.log"),
           recovered=lambda n: f"Spix · measurement is running normally again after {n} failed runs.")


@_safe
def true_close_run(counts=None, *, problem=None):
    """After each true-close run. It fails when it cannot run (``problem``) or records nothing on 3+ matches with
    errors; a single match that keeps erroring is retried until it is given up, as designed."""
    counts = counts or {}
    failed = problem is not None or (counts.get("fixtures", 0) >= 3 and bool(counts.get("errors"))
                                     and not (counts.get("final") or counts.get("unavailable")))
    streak("true-close", failed=failed, after=TRUE_CLOSE_FAILURES,
           alert=lambda n: (f"Spix · the true close failed {n} runs in a row: no closing prices from OddsPapi are "
                            "being recorded. Log: Index/logs/true-close.stdout.log"),
           recovered=lambda n: f"Spix · the true close is running normally again after {n} failed runs.")


# Price confirmation (stage 4) --------------------------------------------------------------------------------------

@_safe
def price_confirmations(records, *, now=None):
    """After price confirmation checks a fixture's picks (``records``: their ``price_confirmation`` records).

    A pick the market moved away from is a no-bet by design; a fault is a pick that could not be checked at all.
    One message a day when faults are most of the day's checked picks (at least ``CONFIRMATION_MIN_FAULTS``).
    """
    day = _local(now).date().isoformat()
    with _topic("price-confirmation") as state:
        if state.get("day") != day:
            state.clear()
            state.update(day=day, checked=0, reasons={})
        state["checked"] += len(records)
        reasons = Counter(state["reasons"])
        reasons.update(r.get("reason") for r in records if r.get("reason") in CONFIRMATION_FAULTS)
        state["reasons"] = dict(reasons)
        faults = sum(reasons.values())
        if not state.get("alerted") and faults >= CONFIRMATION_MIN_FAULTS and faults * 2 > state["checked"]:
            from .price_confirmation import REASONS
            why = "; ".join(f"{REASONS.get(reason, reason)} ({n})" for reason, n in reasons.most_common())
            state["alerted"] = notify(f"Spix · price confirmation: {faults} of today's {state['checked']} picks could "
                                      f"not be checked against OddsPapi, so they were not published. {why}.")


# The Monday scorecard ----------------------------------------------------------------------------------------------

def scorecard_message(card):
    overall = card["cohorts"]["selections"]["overall"]

    def value(row):
        if row.get("true_close_value") is None:
            return "no picks at the close yet"
        interval = (row.get("intervals") or {}).get("true_close_value")
        spread = f" ({interval[0]:+.1%} to {interval[1]:+.1%})" if interval else ""
        return f"{row['true_close_value']:+.1%}{spread} on {row['true_close_count']:,}"

    lines = [f"Spix · scorecard {card['as_of']}",
             f"{overall['published']:,} published, {overall['settled']:,} settled.",
             f"Value at Pinnacle's close: {value(overall)} picks.",
             f"Spix said {overall['stated']:.1%}, won {overall['win_rate']:.1%}; the market said {overall['market']:.1%}.",
             "By stage, value at the close:"]
    for engine, row in sorted(card["cohorts"]["selections"].get("by_engine", {}).items()):
        lines.append(f"  {engine.replace('SPIX_', '')}: {value(row)}")
    lines.append(f"Full card: Research/scorecards/{card['as_of']}/scorecard.md")
    return "\n".join(lines)
