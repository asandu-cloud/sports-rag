"""A5.3 price-age check (tracker D44): is a price seen recently at its bookmaker more often still on offer?

For every market-price pick of the audited reads (A5.2) whose quoted price was on offer at its bookmaker at the read
(OddsPapi's history cut at the read, as price confirmation would have seen it, D42), this records how long before the
read OddsPapi last saw the bookmaker on any line of the match, and whether the quoted price was still there at the
bookmaker's first check after the read. Fresh picks (seen within ``--fresh-minutes``) are compared with older ones; the
label's age rule is supported when fresh prices are still there more often (interval of the difference above zero).

No billable request: the budget is the billable count already used, so a fixture list that is not cached raises
``BudgetExceeded`` instead of being bought, and lists come first from the true close's weekly ones. Histories are read
one bookmaker at a time, as the D42 replay cached them, and only from the download cache unless ``--fetch-histories``
(free requests). The key is read from ``.env`` and never printed.

    python -m Scripts.ops.confidence_age_study --out Research/confidence-age-<date>
"""
from __future__ import annotations

import argparse
import bisect
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import time

from Scripts.data_platform import disagreement as dis, oddspapi
from Scripts.data_platform.calibration_data import _time
from Scripts.data_platform.services.true_close import week_window
from Scripts.ops import pinnacle_check as pc, price_check as prc
from Scripts.ops.odds_archive import ROOT

TOLERANCE = 0.005  # as the price check and price confirmation


def seen_age_minutes(moments, at):
    """Minutes since the bookmaker's newest record at or before ``at`` (``moments`` sorted), or None."""
    i = bisect.bisect_right(moments, at)
    return None if i == 0 else (at - moments[i - 1]).total_seconds() / 60


def next_check_verdict(points, moments, at, kickoff, quoted, *, tolerance=TOLERANCE):
    """Whether the quoted price was there at the bookmaker's first record after ``at``, before kickoff."""
    i = bisect.bisect_right(moments, at)
    following = moments[i] if i < len(moments) and moments[i] < kickoff else None
    if following is None:
        return "no check before kickoff"
    then = [p for p in points if p[0] <= following]
    if not then:
        return "gone"
    _, price, active = then[-1]
    return "still there" if active and price and price >= quoted * (1 - tolerance) else "gone"


def _share(records):
    checked = [r for r in records if r["verdict"] != "no check before kickoff"]
    return sum(r["verdict"] == "still there" for r in checked) / len(checked) if checked else None


def _value(records):
    closed = [r["value_at_close"] for r in records if r.get("value_at_close") is not None]
    return sum(closed) / len(closed) if closed else None


def still_there_comparison(fresh, older, *, resamples=1000):
    """Fresh against older prices: the share still on offer at the next check, with a fixture-resampled interval."""
    def group(records):
        return {"picks": len(records), "no_check": sum(r["verdict"] == "no check before kickoff" for r in records),
                "still_there_share": _share(records), "value_at_close": _value(records)}

    def difference(sample):
        a, b = _share([r for r in sample if r["_fresh"]]), _share([r for r in sample if not r["_fresh"]])
        return None if a is None or b is None else a - b
    tagged = [{**r, "_fresh": True} for r in fresh] + [{**r, "_fresh": False} for r in older]
    return {"fresh": group(fresh), "older": group(older), "diff": difference(tagged),
            "diff_interval95": dis.fixture_bootstrap(tagged, difference, resamples=resamples)}


def _redact(text):
    return re.sub(r"apiKey=[^&\s'\"]+", "apiKey=REDACTED", str(text))


def _fixture_map(client, fixtures):
    """{our fixture id: OddsPapi fixtureId} from fixture lists already bought, never new ones.

    First the true close's weekly lists (one per competition and week, ``true_close.week_window``), which cover every
    week it has processed; then, for fixtures still unmatched, the league-wide list the price check bought for the
    D42 replay. A week with neither cached is reported in ``skipped``.
    """
    mapping, skipped, weeks = {}, [], {}
    for f in fixtures:
        if f["league"] in pc.TOURNAMENTS:
            weeks.setdefault((f["league"], week_window(f["kickoff"])), []).append(f)
    for (league, (start, end)), items in sorted(weeks.items()):
        try:
            theirs = client.get("/v4/fixtures", tournamentId=oddspapi.TOURNAMENTS[league], **{"from": start, "to": end})
        except (oddspapi.OddsPapiError, oddspapi.BudgetExceeded):
            continue
        mapping.update({str(k): v for k, v in oddspapi.match_fixtures(items, theirs if isinstance(theirs, list) else []).items()})
    for league in sorted({league for league, _ in weeks}):
        left = [f for (lg, _), items in weeks.items() if lg == league for f in items if str(f["fixture_id"]) not in mapping]
        if not left:
            continue
        try:
            mapping.update(prc.fixture_ids(client, left))
        except (oddspapi.OddsPapiError, oddspapi.BudgetExceeded) as exc:
            skipped.append(f"{league}: {len(left)} fixtures, no cached fixture list ({_redact(exc)})")
    return mapping, skipped


def _history(client, odds_id, slug, *, cached_only=True):
    """One bookmaker's price history for the match; with ``cached_only``, None unless it is already downloaded."""
    downloaded = client.cached("/v4/historical-odds", fixtureId=odds_id, bookmakers=slug)
    if cached_only and not downloaded:
        return None
    if not downloaded:
        _leave_the_true_close_its_window()
    try:
        return oddspapi.histories(client, odds_id, {slug}, missing_ok=True).get(slug)
    except (oddspapi.OddsPapiError, oddspapi.BudgetExceeded):
        return None


def _leave_the_true_close_its_window():
    while 25 <= datetime.now(timezone.utc).minute < 40:  # the hourly true close runs at half past
        time.sleep(30)


def records(client, rows, fixtures, mapping, index, *, cached_only=True):
    """One record per pick whose quoted price was on offer at the read."""
    kickoffs = {str(f["fixture_id"]): f["kickoff"] for f in fixtures}
    picks = prc.picks_by_policy(rows, ("market",))["market"]
    out, cache, current = [], {}, None
    for r, p, _ in sorted(picks, key=lambda x: str(x[0]["fixture_id"])):
        slug = oddspapi.BOOKS.get(r["best_book"])
        if not slug or slug in oddspapi.EXCHANGES or r["fixture_id"] not in mapping:
            continue
        if r["fixture_id"] != current:  # one fixture's histories in memory at a time
            current, cache = r["fixture_id"], {}
        if slug not in cache:
            body = _history(client, mapping[r["fixture_id"]], slug, cached_only=cached_only)
            book = ((body or {}).get("bookmakers") or {}).get(slug) or {}
            moments = sorted({oddspapi.utc(e["createdAt"]) for m in (book.get("markets") or {}).values()
                              for o in (m.get("outcomes") or {}).values()
                              for e in (o.get("players") or {}).get("0") or [] if e.get("createdAt")})
            cache[slug] = (body, moments)
        body, moments = cache[slug]
        read, kickoff = _time(r["evaluated_at"]), kickoffs.get(r["fixture_id"]) or _time(r["kickoff"])
        place = oddspapi.locate(index, r)
        points = oddspapi.series(body, slug, *place) if place else []
        before = [x for x in points if x[0] <= read]
        if not before or seen_age_minutes(moments, read) is None:
            continue
        _, own, active = before[-1]
        if not active or not own or own < r["best_odds"] * (1 - TOLERANCE):
            continue  # not on offer at the read: price confirmation would not have published it as quoted
        out.append({"fixture_id": r["fixture_id"], "market": r["market"], "book": r["best_book"], "p": float(p),
                    "odds": r["best_odds"], "minutes_before_kickoff": (kickoff - read).total_seconds() / 60,
                    "age_minutes": seen_age_minutes(moments, read),
                    "verdict": next_check_verdict(points, moments, read, kickoff, r["best_odds"]),
                    "value_at_close": None if r.get("p_close") is None else r["p_close"] * r["best_odds"] - 1,
                    "y": r["y"], "w": r["w"]})
    return out


def summary(result):
    def line(name, g):
        share = "n/a" if g["still_there_share"] is None else f"{g['still_there_share']:.0%}"
        value = "n/a" if g["value_at_close"] is None else f"{g['value_at_close'] * 100:+.1f}%"
        return f"| {name} | {g['picks']} | {g['no_check']} | {share} | {value} |"
    c, lo_hi = result["comparison"], result["comparison"]["diff_interval95"]
    diff = "n/a" if c["diff"] is None else f"{c['diff'] * 100:+.1f} pts"
    interval = "n/a" if None in lo_hi else f"{lo_hi[0] * 100:+.1f} to {lo_hi[1] * 100:+.1f}"
    return "\n".join([
        f"Picks on offer at the read: {result['records']} ({result['fixtures']} fixtures); fresh = seen within "
        f"{result['fresh_minutes']} min. Fixture lists not cached: {result['skipped'] or 'none'}", "",
        "| Price age | Picks | No check before kickoff | Still there at the next check | Value at close |",
        "|---|---|---|---|---|", line("fresh", c["fresh"]), line("older", c["older"]), "",
        f"Fresh minus older, still there: {diff} ({interval}); age rule supported: "
        f"{'yes' if result['supported'] else 'no'}"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--from", dest="since", default="2026-10-08")
    parser.add_argument("--to", dest="until", default=datetime.now(timezone.utc).date().isoformat(),
                        help="kickoffs before this date (default: today, UTC)")
    parser.add_argument("--fresh-minutes", type=float, default=30.0)
    parser.add_argument("--fetch-histories", action="store_true",
                        help="request histories not yet downloaded (free, but real OddsPapi requests)")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    client = oddspapi.OddsPapiClient(os.environ["ODDSPAPI_API_KEY"], compress=True)
    client.budget = client._usage()["billable"]  # cached answers only for anything billable
    rows, fixtures = prc.load_inputs(ROOT / "Index/platform.db", ROOT / "Index/closing_backfill", args.since, args.until)
    mapping, skipped = _fixture_map(client, fixtures)
    index = oddspapi.market_index(client.get("/v4/markets"))
    found = records(client, rows, fixtures, mapping, index, cached_only=not args.fetch_histories)
    fresh = [r for r in found if r["age_minutes"] <= args.fresh_minutes]
    older = [r for r in found if r["age_minutes"] > args.fresh_minutes]
    comparison = still_there_comparison(fresh, older, resamples=args.resamples)
    lo = comparison["diff_interval95"][0]
    result = {"records": len(found), "fixtures": len({r["fixture_id"] for r in found}), "fresh_minutes": args.fresh_minutes,
              "skipped": skipped, "comparison": comparison, "supported": lo is not None and lo > 0}
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "confidence_age.json").write_text(json.dumps({**result, "picks": found}, indent=2, default=str))
    print(summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
