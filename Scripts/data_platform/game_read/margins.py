"""Game-read bench (D77): how much margin the bookmakers' typical prices carry on each line of our saved reads.

The bench rebuilds the live Most likely pool on football-data's history, where only the result, the main handicap
and over/under 2.5 have prices. Every other line gets a typical price from its chance and the margin measured
here: the median over our saved reads of the line's booksum, the sum of ``1 / typical price`` over its two sides,
with the typical price the live list uses (``recommendations.options``). Measured on the reads the board showed at
kickoff (D70) in the 10 domestic leagues the bench tests, and saved in ``margins.json`` beside this file so the
bench never needs the platform database.

    python -m Scripts.data_platform.game_read.margins --since 2026-09-05
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Iterable, Mapping, Optional

from Scripts.data_platform import recommendations
from Scripts.ops.football_data_import import DIVISIONS

VERSION = "game-read-margins.v1"
MARGINS_PATH = Path(__file__).with_name("margins.json")
DEFAULT_BOOKSUM = 1.05
LEAGUES = tuple(competition for competition, _ in DIVISIONS.values())
PAIRS = {"goals": ("over", "under"), "spreads": ("home", "away"), "btts": ("yes", "no")}


def _key(option: Mapping) -> tuple:
    market, line = option["market"], option.get("line")
    if market == "btts":
        return market, None
    if market == "spreads" and option["side"] == "away":
        line = -line  # a handicap is keyed by the home side's line
    return market, float(line)


def booksums(options_by_market: Mapping[str, list]) -> dict:
    """Each goals line, home handicap line and both-teams-to-score's booksum in one read; lines missing a side are
    left out."""
    sides = defaultdict(dict)
    for market in PAIRS:
        for option in options_by_market.get(market) or []:
            sides[_key(option)][option["side"]] = option["typical_price"]
    return {key: sum(1 / price for price in prices.values())
            for key, prices in sides.items() if set(prices) == set(PAIRS[key[0]])}


def measure(reads: Iterable[Iterable[Mapping]]) -> dict:
    """``{(market, line): {"booksum": median over the reads, "n": reads with the line}}``."""
    found = defaultdict(list)
    for canonical_results in reads:
        for key, value in booksums(recommendations.options(canonical_results)).items():
            found[key].append(value)
    return {key: {"booksum": statistics.median(values), "n": len(values)} for key, values in found.items()}


def booksum(table: Mapping[tuple, float], market: str, line: Optional[float]) -> float:
    """The line's median booksum, else the median of the market's lines, else ``DEFAULT_BOOKSUM``."""
    key = (market, None if line is None else float(line))
    if key in table:
        return table[key]
    same_market = [value for (m, _), value in table.items() if m == market]
    return statistics.median(same_market) if same_market else DEFAULT_BOOKSUM


def save(table: Mapping[tuple, Mapping], path: Path = MARGINS_PATH, **meta) -> None:
    lines = [{"market": market, "line": line, "booksum": round(entry["booksum"], 4), "n": entry["n"]}
             for (market, line), entry in sorted(table.items(), key=lambda item: (item[0][0], item[0][1] or 0.0))]
    Path(path).write_text(json.dumps({"version": VERSION, **meta, "lines": lines}, indent=1) + "\n")


def load(path: Path = MARGINS_PATH) -> dict:
    data = json.loads(Path(path).read_text())
    return {(entry["market"], entry["line"]): entry["booksum"] for entry in data["lines"]}


def _board():
    scripts = str(Path(__file__).resolve().parents[2])
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from web_app.routers import match_reads as board
    return board


def export(days: Iterable[date], *, now: datetime, gate=None) -> dict:
    """``measure`` over the reads the board showed at kickoff on ``days``; one hold of the database gate per day,
    as the wins reel and the late-goal study read them."""
    board = _board()
    from Scripts.data_platform.services.refresh_coordination import data_access
    service, reads = board._get_match_read_service(), []
    for day in days:
        with (gate or data_access)():
            for league in LEAGUES:
                for shown, _ in board._shown_at_kickoff(
                        board._load_matchday_reads(service, league=league, target_date=day), now=now):
                    reads.append((shown.get("data") or {}).get("canonical_results") or [])
    return measure(reads)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", type=date.fromisoformat, required=True)
    parser.add_argument("--until", type=date.fromisoformat, help="last day (default: yesterday, UTC)")
    parser.add_argument("--out", type=Path, default=MARGINS_PATH)
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc)
    until = args.until or now.date() - timedelta(days=1)
    days = [args.since + timedelta(days=i) for i in range((until - args.since).days + 1)]
    table = export(days, now=now)
    save(table, args.out, measured_on=now.date().isoformat(), leagues=list(LEAGUES),
         days={"from": args.since.isoformat(), "to": until.isoformat(), "count": len(days)})
    values = [entry["booksum"] for entry in table.values()]
    print(f"{len(table)} lines, booksums {min(values):.3f} to {max(values):.3f}, written to {args.out}"
          if table else "no reads found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
