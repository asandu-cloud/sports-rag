"""This season's finished league matches, with corners and final pre-kickoff 1X2 prices, for the pick evidence (D82).
Read only; snapshots are read ``BATCH`` at a time so each statement stays short."""

import math
import statistics
from typing import Dict, List, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from ..db import retry_database_busy, session_scope
from ..fair_prices import MIN_BOOKS
from ..models import Competition, Fixture, FixtureTeamStats, OddsSnapshot, Season, Team

BATCH = 20
FINAL = "final_prematch"
MATCH_WINNER = 1  # API-Football's bet id for the 1X2
SIDES = {"Home": "home", "Draw": "draw", "Away": "away"}


def _cent(value: float) -> float:
    return math.floor(value * 100 + 0.5 + 1e-9) / 100


def median_prices(payload) -> Optional[dict]:
    """The median across bookmakers of each 1X2 price, to the cent, or None below ``MIN_BOOKS`` on any side."""
    found = {side: [] for side in SIDES.values()}
    for row in (payload or {}).get("response") or []:
        for book in (row or {}).get("bookmakers") or []:
            for bet in book.get("bets") or []:
                if bet.get("id") != MATCH_WINNER:
                    continue
                for value in bet.get("values") or []:
                    side = SIDES.get(value.get("value"))
                    try:
                        price = float(value.get("odd"))
                    except (TypeError, ValueError):
                        continue
                    if side and math.isfinite(price) and price > 1:
                        found[side].append(price)
    if any(len(prices) < MIN_BOOKS for prices in found.values()):
        return None
    return {side: _cent(statistics.median(prices)) for side, prices in found.items()}


class PickEvidenceRepository:
    def __init__(self, session_factory=session_scope):
        self._factory = session_factory

    @retry_database_busy
    def finished(self, *, after_season: int, competitions: Sequence[str]) -> List[dict]:
        """Full-time matches of ``competitions`` in seasons after ``after_season``; corners are the two teams' total,
        None unless both teams' statistics have them."""
        home, away = aliased(Team), aliased(Team)
        with self._factory() as session:
            rows = session.execute(
                select(Fixture.id, Fixture.api_football_id, Fixture.kickoff_utc, Season.label, Competition.code,
                       home.api_football_id, away.api_football_id, home.name, away.name, Fixture.home_goals,
                       Fixture.away_goals)
                .join(Season, Season.id == Fixture.season_id)
                .join(Competition, Competition.id == Fixture.competition_id)
                .join(home, home.id == Fixture.home_team_id).join(away, away.id == Fixture.away_team_id)
                .where(Competition.code.in_(list(competitions)), Season.year > after_season, Fixture.status == "FT",
                       Fixture.home_goals.is_not(None), Fixture.away_goals.is_not(None))).all()
            corners = dict(session.execute(
                select(FixtureTeamStats.fixture_id, func.sum(FixtureTeamStats.corners))
                .where(FixtureTeamStats.fixture_id.in_([r[0] for r in rows]), FixtureTeamStats.corners.is_not(None))
                .group_by(FixtureTeamStats.fixture_id).having(func.count() == 2)).all()) if rows else {}
        return [{"event_id": str(api_id), "date": kickoff.date().isoformat(), "season": label, "competition": code,
                 "home_id": home_id, "away_id": away_id, "home": home_name, "away": away_name, "home_goals": hg,
                 "away_goals": ag, "corners": None if corners.get(fid) is None else int(corners[fid])}
                for fid, api_id, kickoff, label, code, home_id, away_id, home_name, away_name, hg, ag in rows
                if kickoff is not None]

    def prices(self, event_ids: Sequence[str]) -> Dict[str, dict]:
        """``{event_id: {home, draw, away}}`` from each event's latest final pre-kickoff snapshot; events without a
        complete price are left out."""
        latest = self._latest([str(e) for e in event_ids])
        out: Dict[str, dict] = {}
        ids = list(latest.items())
        for start in range(0, len(ids), BATCH):
            for event_id, payload in self._payloads([row_id for _, row_id in ids[start:start + BATCH]]):
                prices = median_prices(payload)
                if prices:
                    out[event_id] = prices
        return out

    @retry_database_busy
    def _latest(self, event_ids: List[str]) -> Dict[str, int]:
        if not event_ids:
            return {}
        with self._factory() as session:
            rows = session.execute(
                select(OddsSnapshot.event_id, OddsSnapshot.captured_at, OddsSnapshot.id)
                .where(OddsSnapshot.kind == FINAL, OddsSnapshot.event_id.in_(event_ids))).all()
        latest: Dict[str, tuple] = {}
        for event_id, captured, row_id in rows:
            if event_id not in latest or (captured, row_id) > latest[event_id]:
                latest[event_id] = (captured, row_id)
        return {event_id: value[1] for event_id, value in latest.items()}

    @retry_database_busy
    def _payloads(self, row_ids: List[int]) -> list:
        with self._factory() as session:
            return session.execute(
                select(OddsSnapshot.event_id, OddsSnapshot.odds_json).where(OddsSnapshot.id.in_(row_ids))).all()

    @retry_database_busy
    def teams(self, event_id: str) -> Optional[tuple]:
        """(home, away) API-Football team ids of a fixture, or None."""
        if not str(event_id).isdigit():
            return None
        home, away = aliased(Team), aliased(Team)
        with self._factory() as session:
            row = session.execute(
                select(home.api_football_id, away.api_football_id).select_from(Fixture)
                .join(home, home.id == Fixture.home_team_id).join(away, away.id == Fixture.away_team_id)
                .where(Fixture.api_football_id == int(event_id))).first()
        return tuple(row) if row else None

    @retry_database_busy
    def names(self, api_ids: Sequence[int]) -> Dict[int, str]:
        with self._factory() as session:
            return dict(session.execute(
                select(Team.api_football_id, Team.name).where(Team.api_football_id.in_(list(api_ids)))).all())
