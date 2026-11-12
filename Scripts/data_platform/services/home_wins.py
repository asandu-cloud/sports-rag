"""The home page's wins reel (tracker D99): last week's best wins as the board showed them at kickoff, with the
week's whole record beside them, losses included (identity.md section 7). Design:
``docs/superpowers/specs/2026-11-12-home-wins-reel-design.md``.

- **Which picks:** the board's before-kickoff record (D70), Most likely and Value, from 4 Nov 19:50 UTC (D68).
- **The week:** the latest full Monday-to-Sunday week (London) with a graded pick, so a break keeps the last one.
- **Graded** on the full-time score as the scorecard does: a half won counts with the wins, a half lost with the
  losses, a push is void; extra time, a missing score or missing corners wait.
- **A win on the reel** is a full win with a typical price: each match's dearest, highest first, at most 12.

Rebuilding a week's reads is too slow for a page request, so the daily refresh writes ``Index/home_wins.json``
(``python -m Scripts.data_platform home-wins``) and the website serves that file.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from . import before_kickoff
from ..settlement import asian_outcome, provider_id

logger = logging.getLogger(__name__)

VERSION = "home_wins.v1"
LIVE_FROM = datetime(2026, 11, 4, 19, 50, tzinfo=timezone.utc)  # D68: the lists went on the site
MAX_WINS = 12
MAX_WEEKS = 26  # how far back a break may reach
LONDON = ZoneInfo("Europe/London")
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "Index" / "home_wins.json"
LISTS = ("most_likely", "value")
WON, LOST, VOID = {"hit", "half_hit"}, {"miss", "half_miss"}, {"push"}


def empty() -> dict:
    return {"version": VERSION, "week": None, "record": [], "wins": []}


def grade(pick: Mapping[str, Any], facts: Mapping[str, Any]) -> Optional[str]:
    """``hit``, ``half_hit``, ``push``, ``half_miss`` or ``miss`` on the full-time score; None while it waits."""
    if facts.get("status") != "FT":
        return None  # extra time and penalties settle these markets on 90 minutes, which the score does not hold
    market, side, line = pick.get("market"), pick.get("side"), pick.get("line")
    if market == "corners":
        corners = facts.get("corners_home"), facts.get("corners_away")
        if None in corners or side not in ("over", "under"):
            return None
        return asian_outcome(sum(corners), line, over=side == "over")
    home, away = facts.get("home_goals"), facts.get("away_goals")
    if home is None or away is None:
        return None
    if market == "goals" and side in ("over", "under"):
        return asian_outcome(home + away, line, over=side == "over")
    if market == "spreads" and side in ("home", "away") and line is not None:
        margin = home - away if side == "home" else away - home
        return asian_outcome(margin, -float(line))
    if market == "btts" and side in ("yes", "no"):
        return "hit" if (home > 0 and away > 0) == (side == "yes") else "miss"
    if market == "moneyline" and side in ("home", "draw", "away"):
        return "hit" if {"home": home > away, "draw": home == away, "away": away > home}[side] else "miss"
    return None


def _kickoff(card: Mapping[str, Any]) -> Optional[datetime]:
    try:
        moment = datetime.fromisoformat(str((card.get("fixture") or {}).get("kickoff")).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else None


def _price(card: Mapping[str, Any], pick: Mapping[str, Any], listed: str) -> Optional[float]:
    """The typical price the card showed: a Most likely pick's own, a Value pick's from its option in the markets."""
    block = card.get("recommendations") or {}
    key = (pick.get("market"), pick.get("side"), pick.get("line"))
    if listed == "most_likely":
        for item in block.get("most_likely") or ():
            if (item.get("market"), item.get("side"), item.get("line")) == key and item.get("typical_price"):
                return item["typical_price"]
    for market in block.get("markets") or ():
        if market.get("market") != key[0]:
            continue
        for item in market.get("options") or ():
            if (item.get("side"), item.get("line")) == key[1:] and item.get("typical_price"):
                return item["typical_price"]
    return None


def _result(pick: Mapping[str, Any], facts: Mapping[str, Any]) -> Optional[str]:
    if pick.get("market") == "goals":
        total = facts["home_goals"] + facts["away_goals"]
        return f"{total} goal{'' if total == 1 else 's'}"
    if pick.get("market") == "corners":
        return f"{facts['corners_home'] + facts['corners_away']} corners"
    return None


def _record(listed: str, graded: Sequence[tuple]) -> dict:
    chances = [pick["chance"] for pick, _ in graded]
    won = sum(outcome in WON for _, outcome in graded)
    return {"list": listed, "picks": len(graded), "won": won,
            "lost": sum(outcome in LOST for _, outcome in graded),
            "void": sum(outcome in VOID for _, outcome in graded),
            "said": round(sum(chances) / len(chances), 4) if chances else None,
            "expected_wins": round(sum(chances), 1),
            "happened": round(won / len(graded), 4) if graded else None}


def block(cards: Iterable[Mapping[str, Any]], facts: Mapping[str, Mapping[str, Any]], *, monday: date) -> dict:
    """The reel for the London week starting ``monday``, from the full cards shown at kickoff and each match's
    facts by event id. ``week`` is None when no pick that week is graded yet."""
    graded: Dict[str, List[tuple]] = {listed: [] for listed in LISTS}
    days, best = [], []
    for card in cards:
        kickoff = _kickoff(card)
        if kickoff is None or kickoff < LIVE_FROM or not 0 <= (kickoff.astimezone(LONDON).date() - monday).days < 7:
            continue
        shown = before_kickoff.entry(card, read_at=None)  # the lists exactly as the board showed them
        match = facts.get(str((card.get("fixture") or {}).get("event_id")))
        if shown is None or match is None:
            continue
        top = None
        for listed in ("value", "most_likely"):  # an option in both lists counts as the Value pick it also was
            for pick in shown[listed]:
                outcome = grade(pick, match) if isinstance(pick.get("chance"), (int, float)) else None
                if outcome is None:
                    continue
                graded[listed].append((pick, outcome))
                days.append(kickoff.astimezone(LONDON).date())
                price = _price(card, pick, listed) if outcome == "hit" else None
                if price is not None and (top is None or price > top[0]):
                    top = (price, listed, pick)
        if top is not None:
            best.append((shown, match, kickoff, *top))
    if not days:
        return {**empty(), "record": [_record(listed, []) for listed in LISTS]}
    best.sort(key=lambda row: (-row[3], row[2], str(row[0]["fixture"].get("event_id"))))
    wins = [{"fixture": shown["fixture"], "visuals": shown["visuals"],
             "score": {"home": match["home_goals"], "away": match["away_goals"]}, "list": listed,
             "pick": {"label": pick.get("label"), "market": pick.get("market"), "side": pick.get("side"),
                      "line": pick.get("line"), "chance": pick.get("chance"), "typical_price": price},
             "result": _result(pick, match)}
            for shown, match, _, price, listed, pick in best[:MAX_WINS]]
    return {"version": VERSION, "week": {"from": min(days).isoformat(), "to": max(days).isoformat()},
            "record": [_record(listed, graded[listed]) for listed in LISTS], "wins": wins}


def last_full_week(now: datetime) -> date:
    """The Monday of the latest Monday-to-Sunday week, London time, that is over."""
    today = now.astimezone(LONDON).date()
    return today - timedelta(days=today.weekday() + 7)


def shown_at_kickoff(days: Sequence[date], *, now: datetime, gate=None) -> List[dict]:
    """Every public league's cards as the board showed them at kickoff on these UTC days (D70), one gate hold per
    day so the refresh can take the database between days. Built without pick notes or evidence."""
    scripts = str(Path(__file__).resolve().parents[2])
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from data_platform.services.match_read_cards import MatchReadCardError, build_match_read_card
    from web_app.routers import match_reads as board

    from .refresh_coordination import data_access
    gate = gate or data_access
    service, cards = board._get_match_read_service(), []
    for day in days:
        with gate():
            for league in board.PUBLIC_MATCH_READ_LEAGUES:
                reads = board._load_matchday_reads(service, league=league, target_date=day)
                for read, _ in board._shown_at_kickoff(reads, now=now):
                    try:
                        cards.append(build_match_read_card(read))
                    except MatchReadCardError as exc:
                        logger.warning("Skipping unreadable Match Read %s: %s", read.get("id"), exc)
    return cards


def stored_facts(event_ids: Iterable[str], *, session_factory=None, gate=None) -> Dict[str, dict]:
    """Status, score and both teams' corners for each match by its API-Football id, in one short read."""
    from sqlalchemy import select

    from ..db import get_sessionmaker
    from ..models import Fixture, FixtureTeamStats
    from .refresh_coordination import data_access
    ids = sorted({int(i) for i in map(provider_id, event_ids) if i})
    if not ids:
        return {}
    found: Dict[str, dict] = {}
    with (gate or data_access)(), (session_factory or get_sessionmaker())() as session:
        internal = {}
        for row in session.execute(select(Fixture.id, Fixture.api_football_id, Fixture.status, Fixture.home_goals,
                                          Fixture.away_goals).where(Fixture.api_football_id.in_(ids))):
            found[str(row.api_football_id)] = {"status": row.status, "home_goals": row.home_goals,
                                               "away_goals": row.away_goals, "corners_home": None,
                                               "corners_away": None}
            internal[row.id] = str(row.api_football_id)
        if internal:
            for row in session.execute(select(FixtureTeamStats.fixture_id, FixtureTeamStats.is_home,
                                              FixtureTeamStats.corners)
                                       .where(FixtureTeamStats.fixture_id.in_(list(internal)))):
                found[internal[row.fixture_id]]["corners_home" if row.is_home else "corners_away"] = row.corners
    return found


def build(target: Path = DEFAULT_PATH, *, now: Optional[datetime] = None,
          load_cards: Optional[Callable[..., List[dict]]] = None,
          load_facts: Optional[Callable[[Iterable[str]], Dict[str, dict]]] = None) -> dict:
    """Write the reel for the latest full week with a graded pick (an empty one when none since go-live) beside
    ``target`` and swap it in."""
    now = now or datetime.now(timezone.utc)
    load_cards, load_facts = load_cards or shown_at_kickoff, load_facts or stored_facts
    reel, monday = empty(), last_full_week(now)
    for _ in range(MAX_WEEKS):
        sunday = monday + timedelta(days=6)
        if sunday < LIVE_FROM.astimezone(LONDON).date():
            break
        # From the Sunday before: the board files a match under its UTC day, and London runs ahead in summer.
        cards = load_cards([monday + timedelta(days=offset) for offset in range(-1, 7)], now=now)
        week = block(cards, load_facts([str((c.get("fixture") or {}).get("event_id")) for c in cards]),
                     monday=monday)
        if week["week"] is not None:
            reel = week
            break
        monday -= timedelta(days=7)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(reel, ensure_ascii=False))
    os.replace(tmp, target)
    return reel


def load(path: Path = DEFAULT_PATH) -> dict:
    """The reel the last build wrote, or an empty one (the home page then hides the section)."""
    try:
        reel = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return empty()
    if not isinstance(reel, dict) or reel.get("version") != VERSION or not isinstance(reel.get("wins"), list):
        return empty()
    return reel
