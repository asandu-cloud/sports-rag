"""Does OddsPapi's post-match rewrite change bookmakers' pre-kickoff prices? (tracker D42 open item)

On 22 Oct OddsPapi rewrote two matches' histories about 2 h after kickoff, adding in-play markets. Pinnacle's
pre-kickoff prices were unchanged, but soft bookmakers then had fewer markets with a pre-kickoff price than 27
minutes before kickoff (Betano 227 -> 214). The true close reads the history after the rewrite.

``capture`` saves each fixture's history twice from OddsPapi's free history endpoint, never through the client's
cache: just after kickoff (before the rewrite) and 3-12 h after kickoff (after it). It stops when no window is
left, and starts no capture in minutes 25-39 of each hour or while the true close runs. Rerunning resumes:
a capture already saved is not repeated. Fixtures come from OddsPapi's weekly lists (cached, shared with the true
close). ``compare`` reports per bookmaker the lines whose last pre-kickoff price differs between the two captures,
and for published picks whether their true close would change.

    python -m Scripts.ops.history_rewrite_check capture --out Research/history-rewrite-2026-10-24 \\
        --until 2026-10-26T00:00:00Z
    python -m Scripts.ops.history_rewrite_check compare --out Research/history-rewrite-2026-10-24
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import gzip
import json
import os
from pathlib import Path
import subprocess
import time

from Scripts.data_platform import oddspapi
from Scripts.data_platform.services.true_close import kickoff_of, true_close, week_window
from Scripts.ops.odds_archive import ROOT

PHASES = {"at_kickoff": (timedelta(minutes=2), timedelta(minutes=100)),
          "after_rewrite": (timedelta(hours=3), timedelta(hours=12))}
QUIET = range(25, 40)  # minutes past the hour: the true close (hourly at :30)
# The bookmakers our picks quote, three at most per request. William Hill alone: some fixtures have no
# William Hill history, and "No historical odds found" fails every bookmaker asked for with it (D41).
GROUPS = (("pinnacle",), ("betano", "1xbet", "bet365.es"), ("marathonbet",), ("williamhill",))
LEAGUES = ("EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1", "Championship", "PrimeiraLiga", "SuperLig")


def due_phase(kickoff, now, done):
    """The capture due for a fixture at ``now`` ("at_kickoff", "after_rewrite") or None."""
    if now.minute in QUIET:
        return None
    for phase, (start, end) in PHASES.items():
        if phase not in done and start <= now - kickoff < end:
            return phase
    return None


def capture(client, fixture, phase, out, *, now):
    """Save one fixture's histories for one phase; an OddsPapi error saves nothing (the next pass retries)."""
    bodies, groups = {}, {}
    for group in GROUPS:
        key = ",".join(group)
        try:
            bodies[key] = client.fresh("/v4/historical-odds", fixtureId=fixture["fixtureId"], bookmakers=key)
            groups[key] = "ok"
        except oddspapi.OddsPapiError as exc:
            if exc.code != "NOT_FOUND":
                return {"fixture": fixture["fixtureId"], "phase": phase, "status": "error", "error": exc.code or exc.status}
            bodies[key], groups[key] = None, "NOT_FOUND"
    folder = Path(out) / str(fixture["fixtureId"])
    folder.mkdir(parents=True, exist_ok=True)
    saved = {"fixture": fixture, "phase": phase, "requested_at": now.isoformat(), "bodies": bodies}
    (folder / f"{phase}.json.gz").write_bytes(gzip.compress(json.dumps(saved).encode()))
    return {"fixture": fixture["fixtureId"], "phase": phase, "status": "saved", "groups": groups}


def bodies_by_slug(saved):
    """{slug: history body} from a saved capture; a bookmaker with no history found gets an empty body."""
    out = {}
    for key, body in saved["bodies"].items():
        for slug in key.split(","):
            out[slug] = body if body is not None else {}
    return out


def _closes(body, slug, kickoff):
    """{(market id, outcome id): (time, price)} of every line with an active price at kickoff."""
    markets = (((body or {}).get("bookmakers") or {}).get(slug) or {}).get("markets") or {}
    out = {}
    for market_id, market in markets.items():
        for outcome_id in (market.get("outcomes") or {}):
            last = oddspapi.last_before(oddspapi.series(body, slug, market_id, outcome_id), kickoff)
            if last:
                out[(market_id, outcome_id)] = last
    return out


def compare_closes(before, after, slug, kickoff):
    """Line-by-line comparison of one bookmaker's last pre-kickoff prices in the two captures."""
    a, b = _closes(before, slug, kickoff), _closes(after, slug, kickoff)
    counts, lines = Counter({"same": 0, "changed": 0, "dropped": 0, "added": 0}), []
    for key in sorted(set(a) | set(b)):
        kind = ("added" if key not in a else "dropped" if key not in b else "same" if a[key] == b[key] else "changed")
        counts[kind] += 1
        if kind != "same":
            lines.append({"market_id": key[0], "outcome_id": key[1], "kind": kind,
                          "at_kickoff": [a[key][0].isoformat(), a[key][1]] if key in a else None,
                          "after_rewrite": [b[key][0].isoformat(), b[key][1]] if key in b else None})
    return {"counts": dict(counts), "lines": lines,
            "markets_with_close": {"at_kickoff": len({m for m, _ in a}), "after_rewrite": len({m for m, _ in b})}}


def pick_change(prediction, before, after, index, kickoff):
    """Whether a pick's true close (Pinnacle's prices, its own bookmaker's) differs between the captures."""
    a, b = true_close(prediction, before, index, kickoff), true_close(prediction, after, index, kickoff)
    out = {}
    for part, key in (("consensus", "prices"), ("same_book", "odds")):
        va, vb = (a.get(part) or {}).get(key), (b.get(part) or {}).get(key)
        out[part] = ("missing" if va is None and vb is None else "dropped" if vb is None else "added" if va is None
                     else "same" if va == vb else "changed")
    return out


def weekend_fixtures(client, start, until):
    """OddsPapi fixtures of ``LEAGUES`` kicking off in [start, until), from the weekly lists the true close uses."""
    windows = {week_window(start + timedelta(days=d)) for d in range((until - start).days + 1)}
    out = []
    for league in LEAGUES:
        for first, last in sorted(windows):
            listed = client.get("/v4/fixtures", tournamentId=oddspapi.TOURNAMENTS[league], **{"from": first, "to": last})
            for f in listed if isinstance(listed, list) else []:
                if start <= oddspapi.utc(f["startTime"]) < until:
                    out.append({k: f.get(k) for k in ("fixtureId", "startTime", "participant1Name", "participant2Name")}
                               | {"tournament": league})
    return sorted({f["fixtureId"]: f for f in out}.values(), key=lambda f: f["startTime"])


def true_close_running():
    """The hourly true close is running: two clients on one key slow each other with "rate limited" answers
    (23 Oct: a run started at 21:30 UTC was still fetching at 22:15)."""
    return subprocess.run(["pgrep", "-f", "Scripts.ops.true_close"], capture_output=True).returncode == 0


def run_captures(client, fixtures, out, *, clock, sleep=time.sleep, log=print, busy=true_close_running):
    """Capture every fixture in its windows; stops when no window is left to capture."""
    def done(f):
        return {p for p in PHASES if (Path(out) / str(f["fixtureId"]) / f"{p}.json.gz").exists()}

    while any(p not in done(f) and clock() < oddspapi.utc(f["startTime"]) + end
              for f in fixtures for p, (_, end) in PHASES.items()):
        for f in fixtures:
            if busy():
                break
            phase = due_phase(oddspapi.utc(f["startTime"]), clock(), done(f))
            if phase:
                record = capture(client, f, phase, out, now=clock())
                log(json.dumps({"utc": clock().strftime("%Y-%m-%d %H:%M"), **record}))
        sleep(60)


def _load(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def compare(out, index, predictions):
    """Report of every fixture captured twice, by bookmaker and for the picks with a true close there."""
    by_slug, missed, picks = {}, Counter(), Counter()
    captured = {}
    for folder in sorted(p for p in Path(out).iterdir() if p.is_dir()):
        files = {phase: folder / f"{phase}.json.gz" for phase in PHASES}
        if not all(f.exists() for f in files.values()):
            missed["|".join(sorted(p for p, f in files.items() if not f.exists()))] += 1
            continue
        before, after = _load(files["at_kickoff"]), _load(files["after_rewrite"])
        kickoff = oddspapi.utc(before["fixture"]["startTime"])
        a, b = bodies_by_slug(before), bodies_by_slug(after)
        captured[folder.name] = (a, b)
        for slug in sorted(set(a) & set(b)):
            diff = compare_closes(a[slug], b[slug], slug, kickoff)
            entry = by_slug.setdefault(slug, {"fixtures": 0, "counts": Counter(), "markets_with_close": Counter(),
                                              "examples": []})
            entry["fixtures"] += 1
            entry["counts"].update(diff["counts"])
            entry["markets_with_close"].update(diff["markets_with_close"])
            entry["examples"].extend({"fixture": folder.name, **line} for line in diff["lines"][:3])
    for p in predictions:
        odds_id = str((p.get("true_close") or {}).get("oddspapi_fixture_id") or "")
        if odds_id in captured:
            change = pick_change(p, *captured[odds_id], index, kickoff_of(p))
            picks.update(f"{part}: {kind}" for part, kind in change.items())
    return {"fixtures_compared": len(captured), "missed": dict(missed), "picks": dict(picks),
            "bookmakers": {s: {**e, "counts": dict(e["counts"]), "markets_with_close": dict(e["markets_with_close"]),
                               "examples": e["examples"][:20]} for s, e in by_slug.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("capture", "compare"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--until", default=None, help="last kickoff (UTC, ISO), for capture")
    parser.add_argument("--lists", type=int, default=3, help="billable fixture lists allowed, for capture")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.environ.get("ODDSPAPI_API_KEY")
    if not key:
        parser.error("ODDSPAPI_API_KEY is not set")
    client = oddspapi.OddsPapiClient(key, compress=True)
    client.budget = client._usage()["billable"] + args.lists
    clock = lambda: datetime.now(timezone.utc)  # noqa: E731
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "capture":
        if not args.until:
            parser.error("--until is required for capture")
        fixtures = weekend_fixtures(client, clock() - timedelta(hours=12), oddspapi.utc(args.until))
        (args.out / "fixtures.json").write_text(json.dumps(fixtures, indent=1))
        print(json.dumps({"fixtures": len(fixtures), "by_league": Counter(f["tournament"] for f in fixtures)}), flush=True)

        def log(line):
            print(line, flush=True)
            with (args.out / "captures.jsonl").open("a") as handle:
                handle.write(line + "\n")
        run_captures(client, fixtures, args.out, clock=clock, log=log)
        return 0
    from Scripts.data_platform.repositories.predictions import PredictionRepository
    from Scripts.data_platform.services.refresh_coordination import data_access
    with data_access():
        rows = PredictionRepository().get_recent(days=30, published_only=True, publication_scope="all", limit=10 ** 6)
    report = compare(args.out, oddspapi.market_index(client.get("/v4/markets")), rows)
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("fixtures_compared", "missed", "picks")}, indent=1))
    for slug, e in sorted(report["bookmakers"].items()):
        print(slug, e["fixtures"], e["counts"], e["markets_with_close"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
