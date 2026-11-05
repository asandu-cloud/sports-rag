"""Write example pick notes for a day's matches (tracker D75), without writing to the database.

    python -m Scripts.ops.pick_notes_preview [--date 2026-11-05] [--limit 12] [--gpt] [--out DIR]

Takes the cards the website serves (``--site``, the live site by default, so the picks are exactly what visitors
see), reads each team's finished matches in the fixture's competition inside ``data_access()``, and writes
``notes.json`` and ``notes.md`` to ``Research/pick-notes-<date>/``. Without ``--gpt`` every note has the fixed
wording; with it, GPT-5.6 is asked once per match (a paid request) and the fixed wording covers any note it fails.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional
from urllib.request import urlopen

from Scripts.data_platform import pick_notes as pn
from Scripts.ops.odds_archive import ROOT

SITE = "http://127.0.0.1:8000"
_KINDS = {"most_likely": "Most likely", "value": "Value"}


def notes_for_cards(cards: Iterable[Mapping[str, Any]], history: Callable[[str], Optional[Mapping[str, Any]]], *,
                    generator: Callable[[Mapping[str, Any]], Mapping]) -> List[dict]:
    """``{fixture, facts, notes}`` for each card with at least one pick; ``history(event_id)`` gives the
    repository's ``{competition, home, away}`` (None when the fixture is unknown)."""
    entries = []
    for card in cards:
        block = card.get("recommendations") or {}
        if not (block.get("most_likely") or block.get("value")):
            continue
        fixture = card.get("fixture") or {}
        found = history(str(fixture.get("event_id"))) or {}
        facts = pn.match_facts(block, {**fixture, "competition": found.get("competition")},
                               {"home": found.get("home") or [], "away": found.get("away") or []})
        entries.append({"fixture": fixture, "facts": facts, "notes": pn.notes_for_match(facts, generator=generator)})
    return entries


def markdown(entries: Iterable[Mapping[str, Any]], day: str) -> str:
    lines = [f"# Pick notes, {day}", ""]
    for entry in entries:
        fixture = entry["fixture"]
        kickoff = str(fixture.get("kickoff") or "")[11:16]
        lines += [f"## {fixture.get('home_team')} v {fixture.get('away_team')} ({fixture.get('league')}, "
                  f"{kickoff} UTC)", ""]
        for pick in entry["facts"]["picks"]:
            note = entry["notes"]["notes"][pick["id"]]
            wording = "written by GPT" if note["source"] == "gpt" else "fixed wording"
            lines += [f"**{pick['label']}** ({_KINDS[pick['kind']]}, {pick['chance_percent']}%) · {wording}",
                      f"- For: {note['for']}", f"- Against: {note['against']}", ""]
    return "\n".join(lines)


def _get(url: str):
    with urlopen(url, timeout=30) as response:  # the local website, plain HTTP on this machine
        return json.load(response)


def board_cards(site: str, day: str) -> List[dict]:
    """Every current card of the day, league by league, as the website serves them."""
    rows = _get(f"{site}/api/match-reads/schedule/{day}").get("fixtures") or []
    leagues = sorted({(row.get("fixture") or {}).get("league") for row in rows} - {None})
    cards = []
    for league in leagues:
        cards += _get(f"{site}/api/match-reads/{league}/{day}").get("cards") or []
    return sorted(cards, key=lambda c: ((c.get("fixture") or {}).get("kickoff") or "", str(c.get("fixture"))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--limit", type=int, default=0, help="at most this many matches (0: all)")
    parser.add_argument("--gpt", action="store_true", help="ask GPT-5.6 once per match (paid)")
    parser.add_argument("--site", default=SITE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from Scripts.data_platform.repositories.team_form import TeamFormRepository
    from Scripts.data_platform.services.refresh_coordination import RefreshBusy, data_access

    cards = [c for c in board_cards(args.site, args.date) if (c.get("recommendations") or {}).get("most_likely")
             or (c.get("recommendations") or {}).get("value")]
    if args.limit:
        cards = cards[:args.limit]
    repository, histories = TeamFormRepository(), {}
    try:
        with data_access():
            for card in cards:
                event_id = str((card.get("fixture") or {}).get("event_id"))
                histories[event_id] = repository.for_fixture(int(event_id)) if event_id.isdigit() else None
    except RefreshBusy:
        print("The daily refresh holds the database; try again once it finishes.")
        return 1
    generator = pn.generate_gpt_notes if args.gpt else (lambda _: {})
    entries = notes_for_cards(cards, histories.get, generator=generator)

    out = args.out or ROOT / f"Research/pick-notes-{args.date}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "notes.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    (out / "notes.md").write_text(markdown(entries, args.date))
    sources = [n["source"] for e in entries for n in e["notes"]["notes"].values()]
    print(f"{len(entries)} matches, {len(sources)} picks: {sources.count('gpt')} written by GPT, "
          f"{sources.count('fixed')} fixed wording -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
