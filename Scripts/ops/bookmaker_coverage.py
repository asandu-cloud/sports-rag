"""Bookmaker coverage watch: alert when a bookmaker drops out of API-Football's odds (tracker open item).

16-18 Oct: Betano, Betfair, SBO and Dafabet fell from ~95% of fixtures to ~2% of them, and the drop went unnoticed
for a day. This counts, per UTC day of capture, the fixtures each bookmaker priced in the odds archive's raw /odds
responses, among fixtures kicking off within 48 h of the capture (bookmakers price far-off fixtures unevenly), as a
share of every such fixture captured that day. A bookmaker whose share today is under half its median share over the
previous 7 days (where that median is at least 20%) raises an alert: to the private Discord webhook in
``REFRESH_ALERT_WEBHOOK_URL`` when set, and always to the log and ``Index/bookmaker_coverage/latest.json``. Each
bookmaker is alerted once a day; a failed delivery is retried at the next run. Complete days are cached, so a run
decodes only today's responses. Replayed on 7-22 Oct it alerts on 16, 17 and 18 Oct only, for those four.

    python -m Scripts.ops.bookmaker_coverage [--days 8]
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import gzip
import json
import os
from pathlib import Path
import sqlite3
import statistics

from Scripts.ops.odds_archive import ROOT, parse_time

HORIZON = timedelta(hours=48)
WINDOW, DROP, FLOOR, MIN_FIXTURES, MIN_HISTORY = 7, 0.5, 0.2, 20, 3
OUT = ROOT / "Index/bookmaker_coverage"
ARCHIVE = ROOT / "Index/odds_archive/odds.sqlite3"


def _priced(book):
    return any(bet.get("values") for bet in book.get("bets") or [])


def day_counts(captures):
    """{day: {"fixtures": n, "books": {bookmaker: n}}} from (received_at, /odds body) pairs."""
    fixtures, books = {}, {}
    for received, body in captures:
        day = received.date().isoformat()
        for row in (body or {}).get("response") or []:
            fixture = row.get("fixture") or {}
            kickoff = parse_time(str(fixture.get("date")))
            if fixture.get("id") is None or kickoff is None or not timedelta(0) < kickoff - received <= HORIZON:
                continue
            fixtures.setdefault(day, set()).add(fixture["id"])
            for book in row.get("bookmakers") or []:
                if book.get("name") and _priced(book):
                    books.setdefault(day, {}).setdefault(book["name"], set()).add(fixture["id"])
    return {day: {"fixtures": len(ids), "books": {name: len(v) for name, v in sorted(books.get(day, {}).items())}}
            for day, ids in sorted(fixtures.items())}


def alerts(days, today, *, window=WINDOW, drop=DROP, floor=FLOOR, min_fixtures=MIN_FIXTURES, min_history=MIN_HISTORY):
    """Bookmakers whose share of today's fixtures is under ``drop`` times their usual (median) share."""
    start = datetime.fromisoformat(today).date()
    earlier = [days[d] for d in ((start - timedelta(days=n)).isoformat() for n in range(1, window + 1))
               if d in days and days[d]["fixtures"] >= min_fixtures]
    now = days.get(today)
    if len(earlier) < min_history or not now or now["fixtures"] < min_fixtures:
        return []
    found = []
    for book in sorted({b for day in earlier for b in day["books"]}):
        usual = statistics.median(day["books"].get(book, 0) / day["fixtures"] for day in earlier)
        priced = now["books"].get(book, 0)
        if usual >= floor and priced / now["fixtures"] < drop * usual:
            found.append({"bookmaker": book, "share_today": priced / now["fixtures"], "usual_share": usual,
                          "fixtures_today": now["fixtures"], "priced_today": priced})
    return found


def unsent(found, sent, day):
    return [a for a in found if a["bookmaker"] not in sent.get(day, [])]


def message(found, day):
    lines = [f"Spix odds feed, {day}: bookmaker coverage dropped (fixtures kicking off within 48 h)."]
    lines += [f"- {a['bookmaker']}: priced {a['priced_today']} of {a['fixtures_today']} fixtures today "
              f"({a['share_today']:.0%}; usually {a['usual_share']:.0%})" for a in found]
    return "\n".join(lines)


def captures(db, start, end):
    """(received_at, body) of every successful archived /odds response received in [start, end).

    One short query per body: the archive is in rollback-journal mode, so a long read would block its writer.
    """
    rows = db.execute("SELECT received_at, raw_sha256 FROM requests WHERE received_at >= ? AND received_at < ? "
                      "AND http_status = 200 AND raw_sha256 IS NOT NULL ORDER BY received_at",
                      (start.isoformat(), end.isoformat())).fetchall()
    seen = set()
    for received, sha in rows:
        at = parse_time(received)
        if at is None or (at.date(), sha) in seen:
            continue
        seen.add((at.date(), sha))
        blob = db.execute("SELECT body_gzip FROM raw_bodies WHERE sha256 = ?", (sha,)).fetchone()
        if blob:
            yield at, json.loads(gzip.decompress(blob[0]))


def _read(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def run(*, now=None, days_back=8, archive=ARCHIVE, out=OUT, send=None):
    now = now or datetime.now(timezone.utc)
    today = now.date().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    cached = _read(out / "days.json", {})
    wanted = [(now.date() - timedelta(days=n)).isoformat() for n in range(days_back, -1, -1)]
    missing = [d for d in wanted if d not in cached or d == today]
    if missing:
        start = datetime.fromisoformat(missing[0]).replace(tzinfo=timezone.utc)
        with closing(sqlite3.connect(archive.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)) as db:
            fresh = day_counts(captures(db, start, now + timedelta(minutes=1)))
        for day in missing:
            cached[day] = fresh.get(day, {"fixtures": 0, "books": {}})
    days = {d: cached[d] for d in wanted}
    (out / "days.json").write_text(json.dumps({d: v for d, v in days.items() if d != today}, indent=1))
    found = alerts(days, today)
    sent = _read(out / "alerts_sent.json", {})
    new, delivered = unsent(found, sent, today), None
    if new:
        configured = send is not None or bool(os.environ.get("REFRESH_ALERT_WEBHOOK_URL"))
        if send is None:
            from Scripts.data_platform.services.refresh_alerts import _send as send
        delivered = bool(send(message(new, today)))
        if delivered or not configured:  # a failed delivery is retried at the next run; with no webhook, logged once
            sent[today] = sorted(set(sent.get(today, [])) | {a["bookmaker"] for a in new})
            (out / "alerts_sent.json").write_text(json.dumps({d: v for d, v in sent.items() if d in wanted}, indent=1))
    report = {"checked_at": now.isoformat(), "today": today, "alerts": found, "new_alerts": new,
              "webhook_delivered": delivered, "days": days}
    (out / "latest.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=8, help="days of history to keep (the window needs 7)")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    report = run(days_back=args.days)
    today = report["days"][report["today"]]
    print(json.dumps({"checked_at": report["checked_at"], "fixtures_today": today["fixtures"],
                      "alerts": report["alerts"], "new_alerts": len(report["new_alerts"]),
                      "webhook_delivered": report["webhook_delivered"]}, indent=1))
    if report["new_alerts"]:
        print(message(report["new_alerts"], report["today"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
