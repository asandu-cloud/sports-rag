"""Check our archived Pinnacle prices against Pinnacle's own timestamped history (rework A1 step 4).

OddsPapi's ``/v4/historical-odds`` (free, history since January 2026) returns
every Pinnacle price change for a fixture. For each Pinnacle 1X2 price our odds
archive captured from API-Football, this finds Pinnacle's actual price at the
capture moment and reports:

* ``gap_pts``: the largest difference in fair probability (percentage points);
* ``lag_minutes``: how long ago Pinnacle last showed the price we captured
  (0 = current, None = Pinnacle never showed it);
* the closing check: our "closing" snapshot (API-Football's final pre-kickoff
  refresh) against Pinnacle's true last price before kickoff.

Billable OddsPapi calls (fixture lists, one per league) are cached in
``Index/oddspapi/`` and capped by ``--budget``; historical odds are free.
The API key is read from ``ODDSPAPI_API_KEY`` and never written to disk.

    python -m Scripts.ops.pinnacle_check --out Research/pinnacle-check-<date>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
from datetime import datetime, timedelta, timezone
import gzip
import json
import os
from pathlib import Path
import sqlite3
import statistics

from Scripts.data_platform import oddspapi
from Scripts.data_platform.oddspapi import (  # noqa: F401  (re-exported)
    BudgetExceeded, OddsPapiClient, match_fixtures, utc as _utc)
from Scripts.data_platform.services.refresh_coordination import data_access
from Scripts.ops.odds_archive import ROOT

MARKETS = {"winner": ("101", ("101", "102", "103"))}
TOURNAMENTS = oddspapi.TOURNAMENTS


def timeline(body, bookmaker, market, *, kickoff):
    """[(time, prices)] for one market, forward-filled, only changes before kickoff."""
    market_id, outcomes = market
    data = (((body.get("bookmakers") or {}).get(bookmaker) or {}).get("markets") or {}).get(market_id) or {}
    changes = []
    for index, outcome in enumerate(outcomes):
        for entry in ((data.get("outcomes") or {}).get(outcome) or {}).get("players", {}).get("0", []):
            moment = _utc(entry["createdAt"])
            if moment < kickoff and entry.get("price"):
                changes.append((moment, index, float(entry["price"])))
    current, states = [None] * len(outcomes), []
    for moment, index, price in sorted(changes):
        current[index] = price
        if all(current):
            if states and states[-1][0] == moment:
                states[-1] = (moment, tuple(current))
            else:
                states.append((moment, tuple(current)))
    return states


def state_at(states, moment):
    best = None
    for entry in states:
        if entry[0] <= moment:
            best = entry
        else:
            break
    return best


def closing(states):
    return states[-1] if states else None


def _fair(prices):
    inverse = [1 / p for p in prices]
    total = sum(inverse)
    return [p / total for p in inverse]


def _same(a, b, tolerance=0.006):
    return all(abs(x / y - 1) <= tolerance for x, y in zip(a, b))


def compare(our_prices, states, moment):
    """Gap to Pinnacle's price at ``moment`` and how long ago Pinnacle last showed ours."""
    current = state_at(states, moment)
    if current is None:
        return {"gap_pts": None, "lag_minutes": None, "pinnacle": None}
    gap = max(abs(a - b) for a, b in zip(_fair(our_prices), _fair(current[1]))) * 100
    index = states.index(current)
    lag = None
    for i in range(index, -1, -1):
        if _same(our_prices, states[i][1]):
            lag = 0.0 if i == index else (moment - states[i + 1][0]).total_seconds() / 60
            break
    return {"gap_pts": gap, "lag_minutes": lag, "pinnacle": current[1]}


# --- the study ----------------------------------------------------------------------------------

def _pinnacle_1x2(row):
    for book in row.get("bookmakers", []):
        if book.get("name") == "Pinnacle":
            for bet in book.get("bets", []):
                if bet.get("id") == 1:
                    odds = {v["value"]: float(v["odd"]) for v in bet.get("values", []) if v.get("odd")}
                    if {"Home", "Draw", "Away"} <= set(odds):
                        return odds["Home"], odds["Draw"], odds["Away"]
    return None


def our_fixtures(since, until, *, db=None):
    """Finished fixtures kicking off in [since, until), read behind the refresh gate (one short statement)."""
    with data_access(), contextlib.closing(sqlite3.connect(f"file:{db or ROOT / 'Index/platform.db'}?mode=ro",
                                                           uri=True, timeout=60)) as platform:
        rows = platform.execute("""SELECT f.api_football_id, f.kickoff_utc, c.code, h.name, a.name FROM fixtures f
            JOIN competitions c ON c.id = f.competition_id JOIN teams h ON h.id = f.home_team_id
            JOIN teams a ON a.id = f.away_team_id WHERE f.kickoff_utc >= ? AND f.kickoff_utc < ? AND f.status = 'FT'""",
                                (since, until)).fetchall()
    return [{"fixture_id": fid, "kickoff": _utc(str(ko)[:19] + "+00:00"), "league": code, "home": home, "away": away}
            for fid, ko, code, home, away in rows]


def our_captures(fixture_ids):
    """{fixture id: [(received_at, api_update, prices)]} for Pinnacle 1X2 in the odds archive."""
    archive = sqlite3.connect(f"file:{ROOT / 'Index/odds_archive/odds.sqlite3'}?mode=ro", uri=True, timeout=60)
    wanted = set(fixture_ids)
    captures = defaultdict(list)
    query = """SELECT DISTINCT r.raw_sha256, r.received_at FROM fixture_snapshots f JOIN requests r ON r.id = f.request_id
               WHERE f.fixture_id IN ({}) AND r.raw_sha256 IS NOT NULL""".format(",".join("?" * len(wanted)))
    for sha, received in archive.execute(query, list(wanted)):
        body = json.loads(gzip.decompress(archive.execute("SELECT body_gzip FROM raw_bodies WHERE sha256 = ?", (sha,)).fetchone()[0]))
        for row in body.get("response", []):
            fid = row["fixture"]["id"]
            prices = _pinnacle_1x2(row) if fid in wanted else None
            if prices:
                captures[fid].append((_utc(received), _utc(row.get("update")) if row.get("update") else None, prices))
    return captures


def our_closing(fixture_id):
    path = ROOT / "Index/closing_backfill" / f"{fixture_id}.json.gz"
    if not path.exists():
        return None
    rows = json.load(gzip.open(path, "rt"))["body"].get("response") or []
    if not rows:
        return None
    prices = _pinnacle_1x2(rows[0])
    return (_utc(rows[0]["update"]), prices) if prices else None


def _summary(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0}
    pick = lambda q: values[min(len(values) - 1, int(q * (len(values) - 1)))]
    return {"n": len(values), "median": statistics.median(values), "p90": pick(.9), "max": values[-1]}


def run(client, since, until):
    fixtures = [f for f in our_fixtures(since, until) if f["league"] in TOURNAMENTS]
    by_league = defaultdict(list)
    for f in fixtures:
        by_league[f["league"]].append(f)
    mapping = {}
    for league, items in sorted(by_league.items()):
        theirs = client.get("/v4/fixtures", tournamentId=TOURNAMENTS[league],
                            **{"from": (min(f["kickoff"] for f in items) - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
                               "to": (max(f["kickoff"] for f in items) + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")})
        mapping.update(match_fixtures(items, theirs if isinstance(theirs, list) else []))
    captures = our_captures(mapping)
    records, closings = [], []
    kickoffs = {f["fixture_id"]: f for f in fixtures}
    for fid, odds_id in mapping.items():
        if fid not in captures and our_closing(fid) is None:
            continue
        body = client.get("/v4/historical-odds", fixtureId=odds_id, bookmakers="pinnacle")
        states = timeline(body, "pinnacle", MARKETS["winner"], kickoff=kickoffs[fid]["kickoff"])
        if not states:
            continue
        for received, update, prices in captures.get(fid, []):
            if received >= kickoffs[fid]["kickoff"]:
                continue
            result = compare(prices, states, received)
            result.update(fixture_id=fid, league=kickoffs[fid]["league"],
                          minutes_before_kickoff=(kickoffs[fid]["kickoff"] - received).total_seconds() / 60)
            records.append(result)
        ours = our_closing(fid)
        if ours:
            true_close = closing(states)
            gap = max(abs(a - b) for a, b in zip(_fair(ours[1]), _fair(true_close[1]))) * 100
            closings.append({"fixture_id": fid, "gap_pts": gap,
                             "our_snapshot_minutes_before_kickoff": (kickoffs[fid]["kickoff"] - ours[0]).total_seconds() / 60,
                             "true_close_minutes_before_kickoff": (kickoffs[fid]["kickoff"] - true_close[0]).total_seconds() / 60})
    return summarize(fixtures, mapping, records, closings)


def summarize(fixtures, mapping, records, closings):
    buckets = {"over 24 h": lambda m: m > 1440, "3–24 h": lambda m: 180 < m <= 1440, "last 3 h": lambda m: m <= 180}
    report = {"fixtures": len(fixtures), "matched": len(mapping), "captures": len(records), "by_time_to_kickoff": {}}
    for name, test in buckets.items():
        rows = [r for r in records if test(r["minutes_before_kickoff"])]
        known = [r for r in rows if r["gap_pts"] is not None]
        report["by_time_to_kickoff"][name] = {
            "captures": len(rows),
            "within_1pt": sum(r["gap_pts"] <= 1 for r in known) / len(known) if known else None,
            "current_price": sum(r["lag_minutes"] == 0 for r in known) / len(known) if known else None,
            "gap_pts": _summary(r["gap_pts"] for r in known),
            "lag_minutes": _summary(r["lag_minutes"] for r in known),
            "never_shown_by_pinnacle": sum(r["lag_minutes"] is None for r in known)}
    report["closing"] = {"fixtures": len(closings), "gap_pts": _summary(c["gap_pts"] for c in closings),
                         "within_1pt": sum(c["gap_pts"] <= 1 for c in closings) / len(closings) if closings else None,
                         "our_snapshot_minutes_before_kickoff": _summary(c["our_snapshot_minutes_before_kickoff"] for c in closings),
                         "true_close_minutes_before_kickoff": _summary(c["true_close_minutes_before_kickoff"] for c in closings)}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="2026-10-09")
    parser.add_argument("--until", default=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    parser.add_argument("--budget", type=int, default=40, help="maximum billable OddsPapi requests (cumulative)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.environ.get("ODDSPAPI_API_KEY")
    if not key:
        parser.error("ODDSPAPI_API_KEY is not set")
    report = run(OddsPapiClient(key, budget=args.budget), args.since, args.until)
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "pinnacle-check.json").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
