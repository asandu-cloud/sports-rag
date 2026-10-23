"""A5.3: were the best prices real? Each pick's quote checked against the bookmaker's own price history.

A5.2 (D31) found that pricing against the market consensus leaves value at the close, but that value is
price shopping: a pick takes the best price any bookmaker showed in the read. API-Football copies prices
with a delay (D15: our Pinnacle copy is a median 130 minutes old in the last three hours before kickoff),
so a best price may be one the bookmaker had already withdrawn. For every pick a selection policy makes on
the audited reads (the A5.2 replay: latest read before kickoff per fixture and stage), OddsPapi's free
``/v4/historical-odds`` gives the quoting bookmaker's own timestamped prices for that selection. At the
moment of the read each pick is one of:

* ``real``: the bookmaker's own price then was at least the quoted price (0.5% tolerance);
* ``stale``: it had already moved shorter; ``lag_minutes`` is how long before the read it moved away;
* ``never``: it had never shown the quoted price before the read (a feed error or a differently defined market);
* ``suspended``: the selection was suspended at that moment;
* ``not_listed``: the bookmaker had not priced the line yet;
* ``no_history``: OddsPapi has no prices for this bookmaker and line (or the fixture is unmatched).
* ``history_error``: OddsPapi answered the history request with an error (22 Oct: Betfair requested with
  other bookmakers failed the whole request; such picks were counted as ``no_history`` before).

Value is recomputed at the bookmaker's own price (published probability times that price, minus one), and
the bookmaker's own last price before kickoff gives its true close for the quoted price.

Billable calls (one fixture list per competition) are cached and capped by ``--budget``; historical odds
are free. The API key is read from ``ODDSPAPI_API_KEY`` and never written to disk.

    python -m Scripts.ops.price_check --out Research/price-check-<date>
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics

from Scripts.data_platform import disagreement as dis
from Scripts.data_platform import selection_policies as sp
from Scripts.data_platform.calibration_data import _time
from Scripts.data_platform.oddspapi import (  # noqa: F401
    BOOKS, EXCHANGES, OddsPapiError, histories, last_before, locate, market_index, series)
from Scripts.data_platform.services.refresh_coordination import data_access
from Scripts.ops import pinnacle_check as pc
from Scripts.ops.odds_archive import ROOT

# The quotes are also checked at the bookmaker's other country sites (Bet365's differ, see oddspapi.BOOKS).
ALSO = {"Bet365": {"Bet365 UK site": "bet365", "Bet365 German site": "bet365.de"}}
TOLERANCE = 0.005
ERROR = {"status": "history_error", "own_price": None, "lag_minutes": None, "own_close": None}
POLICIES = ("market", "current")


def check(quoted, points, at, kickoff, *, tolerance=TOLERANCE):
    """The bookmaker's own price at ``at`` against the quoted price, and its last price before kickoff."""
    if not points:
        return {"status": "no_history", "own_price": None, "lag_minutes": None, "own_close": None}
    floor = quoted * (1 - tolerance)
    close = last_before(points, kickoff)  # None when the line was withdrawn before kickoff
    out = {"own_close": close[1] if close else None, "lag_minutes": None}
    before = [p for p in points if p[0] <= at]
    if not before:
        return {**out, "status": "not_listed", "own_price": None}
    _, own, active = before[-1]
    out["own_price"] = own if active else None
    if not active or not own:
        out["status"] = "suspended"
    elif own >= floor:
        out["status"] = "real"
    else:
        shown = [i for i, p in enumerate(before) if p[1] and p[2] and p[1] >= floor]
        out["status"] = "stale" if shown else "never"
        if shown:
            out["lag_minutes"] = (at - before[shown[-1] + 1][0]).total_seconds() / 60
    return out


def picks_by_policy(rows, policies=POLICIES, *, min_ev=0.02):
    """{policy: [(row, published probability, claimed value)]} for every stage."""
    named = {p.name: p for p in sp.policies()}
    out = {}
    for name in policies:
        out[name] = [c for stage in sorted({r["stage"] for r in rows})
                     for c in sp.pick([r for r in rows if r["stage"] == stage], named[name], min_ev=min_ev)]
    return out


def fixture_ids(client, fixtures):
    """{our fixture id (str): OddsPapi fixtureId}, one fixture list per competition."""
    by_league = defaultdict(list)
    for f in fixtures:
        if f["league"] in pc.TOURNAMENTS:
            by_league[f["league"]].append(f)
    mapping = {}
    for league, items in sorted(by_league.items()):
        theirs = client.get("/v4/fixtures", tournamentId=pc.TOURNAMENTS[league],
                            **{"from": (min(f["kickoff"] for f in items) - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
                               "to": (max(f["kickoff"] for f in items) + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")})
        mapping.update({str(k): v for k, v in pc.match_fixtures(items, theirs if isinstance(theirs, list) else []).items()})
    return mapping


def run(client, rows, fixtures, index, *, policies=POLICIES, min_ev=0.02):
    picks = picks_by_policy(rows, policies, min_ev=min_ev)
    mapping = fixture_ids(client, fixtures)
    kickoffs = {str(f["fixture_id"]): f["kickoff"] for f in fixtures}
    needed, outcomes = defaultdict(set), defaultdict(lambda: defaultdict(set))
    for chosen in picks.values():
        for r, _, _ in chosen:
            slug = BOOKS.get(r["best_book"])
            if slug and r["fixture_id"] in mapping:
                needed[r["fixture_id"]].update({slug, *ALSO.get(r["best_book"], {}).values()})
                place = locate(index, r) if slug in EXCHANGES else None
                if place:
                    outcomes[r["fixture_id"]][slug].add(place[1])
    results = {name: [] for name in picks}
    also = {name: [] for name in picks}  # the same quotes checked at other country sites of the bookmaker
    for fixture in sorted(needed):
        try:
            bodies = histories(client, mapping[fixture], needed[fixture], outcomes[fixture], missing_ok=True)
        except OddsPapiError:
            bodies = None
        for name, chosen in picks.items():
            for r, p, ev in chosen:
                if r["fixture_id"] != fixture or not BOOKS.get(r["best_book"]):
                    continue  # bookmakers OddsPapi does not carry are recorded below
                place = locate(index, r)
                at, kickoff = _time(r["evaluated_at"]), kickoffs.get(fixture) or _time(r["kickoff"])
                for label, slug, out in [(r["best_book"], BOOKS.get(r["best_book"]), results[name])] + [
                        (label, slug, also[name]) for label, slug in ALSO.get(r["best_book"], {}).items()]:
                    points = series(bodies.get(slug), slug, *place) if bodies and slug and place else []
                    found = check(r["best_odds"], points, at, kickoff) if bodies is not None else ERROR
                    out.append({**_record(r, p, ev, found), "book": label})
    for name, chosen in picks.items():  # picks whose bookmaker or fixture OddsPapi cannot check
        for r, p, ev in chosen:
            if not (BOOKS.get(r["best_book"]) and r["fixture_id"] in mapping):
                reason = "bookmaker_not_carried" if not BOOKS.get(r["best_book"]) else "fixture_unmatched"
                results[name].append(_record(r, p, ev, {"status": reason, "own_price": None, "lag_minutes": None,
                                                       "own_close": None}))
    return {"fixtures": len({r["fixture_id"] for r in rows}), "matched": len(mapping), "min_ev": min_ev,
            "picks": results, "summary": {name: summarize(records) for name, records in results.items()},
            "other_sites": {name: {k: _block([r for r in records if r["book"] == k])
                                   for k in sorted({r["book"] for r in records})} for name, records in also.items()}}


def _record(r, p, ev, outcome):
    own = outcome.get("own_price")
    return {"fixture_id": r["fixture_id"], "stage": r["stage"], "market": r["market"], "side": r["side"],
            "point": r["point"], "book": r["best_book"], "quoted": r["best_odds"], "published_p": p,
            "claimed_ev": ev, "evaluated_at": str(r["evaluated_at"]), "kickoff": str(r["kickoff"]),
            "minutes_before_kickoff": (_time(r["kickoff"]) - _time(r["evaluated_at"])).total_seconds() / 60,
            "own_ev": p * own - 1 if own else None, **outcome}


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _block(records):
    statuses = Counter(r["status"] for r in records)
    checked = [r for r in records if r["status"] in ("real", "stale", "never", "suspended", "not_listed")]
    real = [r for r in checked if r["status"] == "real"]
    closed = [r for r in checked if r["own_close"]]
    return {
        "picks": len(records), "statuses": dict(statuses), "checked": len(checked),
        "real_share": len(real) / len(checked) if checked else None,
        "stale_lag_minutes_median": _median(r["lag_minutes"] for r in records if r["status"] == "stale"),
        "claimed_ev_mean": statistics.fmean(r["claimed_ev"] for r in checked) if checked else None,
        # value at the bookmaker's own price; a withdrawn or suspended price counts as no bet (0)
        "own_ev_mean": statistics.fmean((r["own_ev"] if r["own_ev"] is not None and r["status"] != "suspended" else 0.0)
                                        for r in checked) if checked else None,
        "still_value_share": (sum(1 for r in checked if r["own_ev"] is not None and r["status"] != "suspended"
                                  and r["own_ev"] >= 0.02) / len(checked)) if checked else None,
        "own_close_vs_quoted": {
            "shorter": sum(1 for r in closed if r["own_close"] < r["quoted"] * (1 - TOLERANCE)),
            "same": sum(1 for r in closed if abs(r["own_close"] / r["quoted"] - 1) <= TOLERANCE),
            "longer": sum(1 for r in closed if r["own_close"] > r["quoted"] * (1 + TOLERANCE))},
        "price_clv_vs_own_close_mean": statistics.fmean(r["quoted"] / r["own_close"] - 1 for r in closed) if closed else None,
    }


def summarize(records):
    out = {"all": _block(records)}
    for field in ("stage", "book", "market"):
        out[f"by_{field}"] = {k: _block([r for r in records if r[field] == k])
                               for k in sorted({r[field] for r in records})}
    windows = {"over 24 h": lambda m: m > 1440, "3–24 h": lambda m: 180 < m <= 1440, "last 3 h": lambda m: m <= 180}
    out["by_time_to_kickoff"] = {k: _block([r for r in records if test(r["minutes_before_kickoff"])])
                                 for k, test in windows.items()}
    return out


def markdown(report):
    def pct(v, signed=False):
        return "n/a" if v is None else (f"{v * 100:+.1f}%" if signed else f"{v * 100:.0f}%")
    lines = [f"Fixtures {report['fixtures']}, matched in OddsPapi {report['matched']}, min value {report['min_ev']:.0%}"]
    for name, summary in report["summary"].items():
        lines += [f"\n## Policy: {name}", "",
                  "| Slice | Picks | Checked | Real | Stale (median lag, min) | Never | Suspended | Value claimed | "
                  "Value at own price | Still ≥2% | Own close shorter / same / longer |",
                  "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|"]
        slices = [("all", summary["all"])] + [(f"{g[3:]}: {k}", v) for g in ("by_stage", "by_time_to_kickoff", "by_book", "by_market")
                                               for k, v in summary[g].items()] + [
            (f"same quotes at {k}", v) for k, v in report.get("other_sites", {}).get(name, {}).items()]
        for label, b in slices:
            s, lag = b["statuses"], b["stale_lag_minutes_median"]
            moves = b["own_close_vs_quoted"]
            lines.append(f"| {label} | {b['picks']} | {b['checked']} | {pct(b['real_share'])} | {s.get('stale', 0)}"
                         f" ({'n/a' if lag is None else round(lag)}) | {s.get('never', 0)} | {s.get('suspended', 0)} | "
                         f"{pct(b['claimed_ev_mean'], True)} | {pct(b['own_ev_mean'], True)} | {pct(b['still_value_share'])} | "
                         f"{moves['shorter']} / {moves['same']} / {moves['longer']} |")
    return "\n".join(lines)



def load_inputs(db, backfill, since, until):
    """(audited lines, our finished fixtures among them in [since, until)), each read behind the refresh gate.

    The lines are read inside the gate; ``our_fixtures`` takes its own afterwards, never nested (a nested gate fails
    when the refresh is waiting for it)."""
    with data_access():
        rows = dis.build(db, backfill_dir=backfill, all_sides=True)
    ids = {r["fixture_id"] for r in rows}
    return rows, [f for f in pc.our_fixtures(since, until) if str(f["fixture_id"]) in ids]

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=ROOT / "Index/platform.db")
    parser.add_argument("--backfill", type=Path, default=ROOT / "Index/closing_backfill")
    parser.add_argument("--since", default="2026-10-08")
    parser.add_argument("--until", default="2026-10-21")
    parser.add_argument("--min-ev", type=float, default=0.02)
    parser.add_argument("--budget", type=int, default=60, help="maximum billable OddsPapi requests (cumulative)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.environ.get("ODDSPAPI_API_KEY")
    if not key:
        parser.error("ODDSPAPI_API_KEY is not set")
    client = pc.OddsPapiClient(key, budget=args.budget, compress=True)
    rows, fixtures = load_inputs(args.db, args.backfill, args.since, args.until)
    report = run(client, rows, fixtures, market_index(client.get("/v4/markets")), min_ev=args.min_ev)
    text = markdown(report)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "price-check.json").write_text(json.dumps(report, indent=1, default=str))
        (args.out / "price-check.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
