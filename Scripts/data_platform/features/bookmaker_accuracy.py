"""Historical accuracy of each bookmaker's closing price (rework A1 step 4, the accuracy half of the weights).

For every bookmaker in the football-data history, on fixtures linked to ours with
agreeing scores: its closing fair probability (proportional margin removal) is
scored by log loss on the result, against the consensus of the *other*
bookmakers on the same fixture (per-outcome median of their fair
probabilities, renormalised). A negative difference means the bookmaker beat
the rest of the market. Intervals: percentile bootstrap over calendar weeks.

Aggregates (market average/maximum, BetBrain) are not bookmakers and are left
out; Pinnacle prices flagged suspect (2025/26 onward) are dropped.
"""
from collections import defaultdict
from contextlib import closing
from datetime import date
import math
import random
import sqlite3
import statistics

import numpy as np

VERSION = "bookmaker-accuracy.v1"
EVENTS = {"winner": ("1x2", ("home", "draw", "away")), "over2.5": ("ou25", ("over", "under"))}
SEED = 20260928
EPS = 1e-15


def _aggregate(name):
    lowered = name.lower()
    return "average" in lowered or "maximum" in lowered


def _week(day):
    year, week, _ = date.fromisoformat(day[:10]).isocalendar()
    return f"{year}-W{week:02d}"


def _outcome(event, home_goals, away_goals):
    if event == "winner":
        return 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2
    return 0 if home_goals + away_goals > 2.5 else 1


def _fair(prices, selections):
    if any(prices.get(side) is None or prices[side] <= 1 for side in selections):
        return None
    inverse = [1 / prices[side] for side in selections]
    total = sum(inverse)
    return [p / total for p in inverse], total - 1


def load(db_path, event, *, timing="closing"):
    market, selections = EVENTS[event]
    query = """SELECT m.id, m.match_date, m.season, m.fthg, m.ftag, o.bookmaker, o.selection, o.price
               FROM odds o JOIN matches m ON m.id = o.match_id
               WHERE m.link_status = 'matched' AND o.market = ? AND o.closing = ? AND o.suspect = 0
                 AND m.fthg IS NOT NULL AND m.ftag IS NOT NULL"""
    fixtures = {}
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        for mid, day, season, hg, ag, book, side, price in db.execute(query, (market, 1 if timing == "closing" else 0)):
            if _aggregate(book):
                continue
            entry = fixtures.setdefault(mid, {"week": _week(day), "season": season,
                                              "y": _outcome(event, hg, ag), "books": defaultdict(dict)})
            entry["books"][book][side] = price
    for entry in fixtures.values():
        fair = {}
        for book, prices in entry["books"].items():
            result = _fair(prices, selections)
            if result:
                fair[book] = result
        entry["fair"] = fair
    return fixtures


def _consensus(others):
    medians = [statistics.median(p[i] for p in others) for i in range(len(others[0]))]
    total = sum(medians)
    return [m / total for m in medians]


def _interval(weeks, deltas, resamples, seed):
    groups = defaultdict(list)
    for week, delta in zip(weeks, deltas):
        groups[week].append(delta)
    keys = sorted(groups)
    sums = np.array([sum(groups[k]) for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    draws = np.random.default_rng(seed).integers(0, len(keys), size=(resamples, len(keys)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return np.quantile(boot, [.025, .975]).tolist()


def evaluate(db_path, event, *, timing="closing", resamples=2000, seed=SEED):
    fixtures = load(db_path, event, timing=timing)
    per_book = defaultdict(lambda: {"weeks": [], "deltas": [], "book": [], "consensus": [], "margins": [], "seasons": set()})
    for entry in fixtures.values():
        fair, y = entry["fair"], entry["y"]
        for book, (p, margin) in fair.items():
            others = [q for other, (q, _) in fair.items() if other != book]
            if len(others) < 2:
                continue
            consensus = _consensus(others)
            lb, lc = -math.log(max(p[y], EPS)), -math.log(max(consensus[y], EPS))
            record = per_book[book]
            record["weeks"].append(entry["week"])
            record["deltas"].append(lb - lc)
            record["book"].append(lb)
            record["consensus"].append(lc)
            record["margins"].append(margin)
            record["seasons"].add(entry["season"])
    rows = []
    for book, r in per_book.items():
        low, high = _interval(r["weeks"], r["deltas"], resamples, seed)
        book_ll, cons_ll = float(np.mean(r["book"])), float(np.mean(r["consensus"]))
        rows.append({"bookmaker": book, "event": event, "fixtures": len(r["deltas"]),
                     "weeks": len(set(r["weeks"])), "seasons": sorted(r["seasons"]),
                     "book_logloss": book_ll, "consensus_logloss": cons_ll,
                     "mean_difference": float(np.mean(r["deltas"])), "relative_change": book_ll / cons_ll - 1,
                     "interval95": [low, high], "beats_consensus": high < 0, "worse_than_consensus": low > 0,
                     "mean_margin": float(np.mean(r["margins"]))})
    return sorted(rows, key=lambda r: r["mean_difference"])
