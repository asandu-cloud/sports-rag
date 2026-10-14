"""Read-only freshness and accuracy study of bookmaker prices in the odds archive.

Answers whether a bookmaker's archived prices (Pinnacle by default) are live,
lagging or stale, using four measures per market:

* activity: how often the price changes between consecutive captures;
* following: when the other books' median moves, does this book move with it,
  one capture later, or not at all;
* outliers: how often it sits far from the other books;
* accuracy: log loss of its fair price at the last pre-kickoff capture on
  settled fixtures, against an equal-weight consensus, with a matchweek
  bootstrap interval.

Captures within ``--near-hours`` of kick-off are reported separately; with the
10-minute near-kickoff collector they give lead-lag at that resolution.

Both databases are opened read-only. Raw receipts are parsed directly (the
archive's ``normalize`` builds every quote row, far more than this study needs),
applying the same pre-kickoff rule: a capture at or after kick-off is ignored.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timedelta, timezone
import gzip
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics

from Scripts.ops.odds_archive import ROOT, archive_path, parse_time

PLATFORM_DB = ROOT / "Index/platform.db"
MARKETS = {1: "match_result", 5: "goals_over_under", 4: "asian_handicap"}
MOVE = 0.02          # consensus move worth checking (probability points)
FOLLOW = 0.005       # smallest move counted as "moved"
OUTLIER = 0.03       # distance from the other books' median
MIN_BOOKS = 4        # books needed to form a consensus
BOOTSTRAP_DRAWS = 2000
SEED = 20261014


def connect_ro(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    db.execute("PRAGMA query_only=ON")
    return db


def fair_first(values):
    """Margin-free probability of the first selection, and the margin."""
    inverse = [1 / v for v in values]
    total = sum(inverse)
    return inverse[0] / total, total - 1


def selections(market_id, values):
    """Map one archived market to {line: decimal odds tuple, first side first}."""
    prices = {}
    for value in values:
        try:
            odd = float(value.get("odd"))
        except (TypeError, ValueError):
            continue
        label = value.get("value")
        if odd > 1 and isinstance(label, str):
            prices[label] = odd
    if market_id == 1:
        if all(k in prices for k in ("Home", "Draw", "Away")):
            return {"1x2": (prices["Home"], prices["Draw"], prices["Away"])}
        return {}
    first, second = ("Over", "Under") if market_id == 5 else ("Home", "Away")
    lines = {}
    for label, odd in prices.items():
        side, _, line = label.partition(" ")
        if side == first and f"{second} {line}" in prices:
            lines[line] = (odd, prices[f"{second} {line}"])
    return lines


def load_captures(db, *, books=None):
    """{(fixture, market, line): {received_at: {book: fair_p}}} plus fixture kickoffs."""
    series = defaultdict(lambda: defaultdict(dict))
    margins = defaultdict(list)
    kickoffs = {}
    parsed = {}
    rows = db.execute("""SELECT r.received_at, r.raw_sha256 FROM requests r
        WHERE r.status='ok' AND r.raw_sha256 IS NOT NULL ORDER BY r.received_at""").fetchall()
    for received_raw, sha in rows:
        received = parse_time(received_raw)
        if received is None:
            continue
        if sha not in parsed:
            body = db.execute("SELECT body_gzip FROM raw_bodies WHERE sha256=?", (sha,)).fetchone()[0]
            parsed[sha] = json.loads(gzip.decompress(body)).get("response") or []
        for event in parsed[sha]:
            fixture = event.get("fixture") or {}
            fixture_id, kickoff = fixture.get("id"), parse_time(fixture.get("date"))
            if not isinstance(fixture_id, int) or kickoff is None or received >= kickoff:
                continue
            kickoffs[fixture_id] = kickoff
            for book in event.get("bookmakers") or []:
                name = book.get("name")
                if not isinstance(name, str) or (books and name not in books):
                    continue
                for market in book.get("bets") or []:
                    market_id = market.get("id")
                    if market_id not in MARKETS:
                        continue
                    for line, odds in selections(market_id, market.get("values") or []).items():
                        probability, margin = fair_first(odds)
                        series[(fixture_id, MARKETS[market_id], line)][received][name] = probability
                        margins[(MARKETS[market_id], name)].append(margin)
    return series, kickoffs, margins


def _empty_stats():
    return {"pairs": 0, "changed": 0, "moves": 0, "with": 0, "next": 0, "missed": 0,
            "captures": 0, "outliers": 0}


def freshness(series, kickoffs, *, near_hours=3.0):
    """Activity, following and outlier counts per (window, market, book)."""
    stats = defaultdict(_empty_stats)
    for (fixture_id, market, _line), by_time in series.items():
        times = sorted(by_time)
        kickoff = kickoffs[fixture_id]
        for i, t in enumerate(times):
            window = "near_kickoff" if kickoff - t <= timedelta(hours=near_hours) else "earlier"
            books = by_time[t]
            if len(books) >= MIN_BOOKS:
                for book, p in books.items():
                    others = [q for b, q in books.items() if b != book]
                    s = stats[(window, market, book)]
                    s["captures"] += 1
                    s["outliers"] += abs(p - statistics.median(others)) > OUTLIER
            if i + 1 >= len(times):
                continue
            nxt = times[i + 1]
            later = times[i + 2] if i + 2 < len(times) else None
            common = set(books) & set(by_time[nxt])
            for book in common:
                s = stats[(window, market, book)]
                s["pairs"] += 1
                s["changed"] += abs(by_time[nxt][book] - books[book]) > 1e-9
                others = [b for b in common if b != book]
                if len(others) < MIN_BOOKS - 1:
                    continue
                consensus = statistics.median(by_time[nxt][b] - books[b] for b in others)
                if abs(consensus) < MOVE:
                    continue
                s["moves"] += 1
                step = by_time[nxt][book] - books[book]
                if step * consensus > 0 and abs(step) >= FOLLOW:
                    s["with"] += 1
                elif later is not None and book in by_time[later] and \
                        (by_time[later][book] - books[book]) * consensus > 0 and \
                        abs(by_time[later][book] - books[book]) >= FOLLOW:
                    s["next"] += 1
                else:
                    s["missed"] += 1
    return stats


def load_results(path):
    """Regulation-time results only (FT); extra-time and penalty finishes excluded."""
    with closing(connect_ro(path)) as db:
        return {fid: (home, away) for fid, home, away in db.execute(
            "SELECT api_football_id, home_goals, away_goals FROM fixtures "
            "WHERE status='FT' AND home_goals IS NOT NULL AND away_goals IS NOT NULL")}


def outcome(market, line, score):
    home, away = score
    if market == "match_result":
        return 0 if home > away else 1 if home == away else 2
    if market == "goals_over_under" and line == "2.5":
        return 0 if home + away > 2.5 else 1
    return None


def accuracy(series, kickoffs, results, *, books, seed=SEED):
    """Log loss at the last pre-kickoff capture, per book versus consensus."""
    rows = defaultdict(list)  # (market, book) -> [(week, loss_book, loss_consensus)]
    for (fixture_id, market, line), by_time in series.items():
        if fixture_id not in results or market == "asian_handicap":
            continue
        y = outcome(market, line, results[fixture_id])
        if y is None:
            continue
        last = {}
        for t in sorted(by_time):
            last.update(by_time[t])
        if len(last) < MIN_BOOKS:
            continue
        consensus = statistics.fmean(last.values())
        week = kickoffs[fixture_id].isocalendar()[:2]

        def loss(p):
            # match_result stores P(home); other outcomes need the full vector,
            # so only the home/not-home binary is scored for 1X2.
            hit = p if y == 0 else 1 - p
            return -math.log(min(max(hit, 1e-9), 1 - 1e-9))
        for book in books:
            if book in last:
                rows[(market, book)].append((week, loss(last[book]), loss(consensus)))
    report = {}
    rng = random.Random(seed)
    for key, values in rows.items():
        weeks = defaultdict(list)
        for week, lb, lc in values:
            weeks[week].append(lb - lc)
        keys = list(weeks)
        draws = []
        for _ in range(BOOTSTRAP_DRAWS):
            sample = [d for _ in keys for d in weeks[rng.choice(keys)]]
            draws.append(statistics.fmean(sample))
        draws.sort()
        report[f"{key[0]}|{key[1]}"] = dict(
            fixtures=len(values),
            log_loss=round(statistics.fmean(v[1] for v in values), 4),
            consensus_log_loss=round(statistics.fmean(v[2] for v in values), 4),
            difference=round(statistics.fmean(v[1] - v[2] for v in values), 4),
            interval_95=[round(draws[int(.025 * len(draws))], 4), round(draws[int(.975 * len(draws)) - 1], 4)],
            matchweeks=len(keys))
    return report


def verdict(stats, accuracy_report, book):
    """Apply the agreed decision rule; say 'insufficient' rather than guess."""
    near = [s for (w, _m, b), s in stats.items() if w == "near_kickoff" and b == book]
    moves = sum(s["moves"] for s in near)
    trailing = sum(s["next"] + s["missed"] for s in near)
    lag = None if moves < 30 else trailing / moves
    acc = accuracy_report.get(f"match_result|{book}")
    worse = None if not acc or acc["matchweeks"] < 4 else acc["interval_95"][0] > 0
    return {
        "near_kickoff_moves": moves,
        "near_kickoff_trailing_share": None if lag is None else round(lag, 3),
        "lag_test": "insufficient" if lag is None else ("pass" if lag <= 0.10 else "fail"),
        "accuracy_test": "insufficient" if worse is None else ("fail" if worse else "pass"),
        "spot_check_test": "pending: compare ~20 prices with an independent timestamped source",
    }


def build_report(archive_db, platform_db, *, book="Pinnacle", near_hours=3.0):
    with closing(connect_ro(archive_db)) as db:
        series, kickoffs, margins = load_captures(db)
    stats = freshness(series, kickoffs, near_hours=near_hours)
    books = sorted({b for (_w, _m, b) in stats})
    acc = accuracy(series, kickoffs, load_results(platform_db), books=books)
    table = {}
    for (window, market, name), s in sorted(stats.items()):
        table[f"{window}|{market}|{name}"] = dict(
            s, change_rate=round(s["changed"] / s["pairs"], 3) if s["pairs"] else None,
            follow_rate=round(s["with"] / s["moves"], 3) if s["moves"] else None,
            outlier_rate=round(s["outliers"] / s["captures"], 3) if s["captures"] else None)
    return dict(
        generated_at=datetime.now(timezone.utc).isoformat(), book=book, near_hours=near_hours,
        fixtures=len(kickoffs), series=len(series),
        thresholds=dict(move=MOVE, follow=FOLLOW, outlier=OUTLIER, min_books=MIN_BOOKS),
        margins={f"{m}|{b}": round(statistics.fmean(v), 4) for (m, b), v in sorted(margins.items())},
        freshness=table, accuracy=acc, verdict=verdict(stats, acc, book))


def markdown(report):
    book = report["book"]
    lines = [f"# Odds freshness: {book}", "",
             f"Generated {report['generated_at']}. {report['fixtures']} fixtures, {report['series']} price series.", "",
             "| Window | Market | Book | Change rate | Follows market | Next capture | Missed | Outlier rate |",
             "|---|---|---|---|---|---|---|---|"]
    for key, s in report["freshness"].items():
        window, market, name = key.split("|")
        if s["pairs"] < 50:
            continue
        lines.append(f"| {window} | {market} | {name} | {s['change_rate']} | "
                     f"{s['follow_rate']} ({s['moves']} moves) | {s['next']} | {s['missed']} | {s['outlier_rate']} |")
    lines += ["", "| Market | Book | Fixtures | Log loss | Consensus | Difference (95% interval) |", "|---|---|---|---|---|---|"]
    for key, a in sorted(report["accuracy"].items()):
        market, name = key.split("|")
        lines.append(f"| {market} | {name} | {a['fixtures']} | {a['log_loss']} | {a['consensus_log_loss']} | "
                     f"{a['difference']:+} ({a['interval_95'][0]:+}, {a['interval_95'][1]:+}) |")
    lines += ["", "## Verdict", "", "```", json.dumps(report["verdict"], indent=2), "```", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", default="Pinnacle")
    parser.add_argument("--near-hours", type=float, default=3.0)
    parser.add_argument("--out", type=Path, help="Directory for report.json and report.md")
    args = parser.parse_args(argv)
    report = build_report(archive_path(ROOT), PLATFORM_DB, book=args.book, near_hours=args.near_hours)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=2))
        (args.out / "report.md").write_text(markdown(report))
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
