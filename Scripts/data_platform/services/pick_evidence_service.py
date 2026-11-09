"""Pick evidence on the website's cards (tracker D82), behind the display switch ``SPIX_PICK_EVIDENCE``.

Each card's ``recommendations.v1`` block gains ``story`` (How Spix reads it) and, in the ten domestic leagues,
``evidence`` (the shared groups of past matches) plus each Most likely item's ``evidence`` (``pick_evidence.build``).
The history is football-data's 2024/25 and 2025/26 (a static file, read once) and this season's finished matches
from the platform database, refreshed once every ``KEEP``; a match's final pre-kickoff price is read once. Every
database read goes through the refresh gate without waiting (``refresh_coordination.read_gate``), so nothing is read
while the daily refresh writes. After a failed read the next try waits ``RETRY``; meanwhile the last good history is
kept, or the picks show no evidence if there is none yet. Nothing that was on the card changes.
"""
from __future__ import annotations

from contextlib import closing, nullcontext
from datetime import timedelta
import logging
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from .. import pick_evidence

logger = logging.getLogger(__name__)
KEEP = timedelta(hours=1)
RETRY = timedelta(minutes=5)
FIRST_SEASON = int(pick_evidence.SINCE[:4])
FOOTBALL_DATA = Path(__file__).resolve().parents[3] / "Index/football_data.sqlite3"
PRICE = ("1x2", "Market average", 0)  # football-data's average of the bookmakers' pre-closing 1X2 prices


def enabled() -> bool:
    return os.environ.get("SPIX_PICK_EVIDENCE", "").strip().lower() == "on"


class HistoryUnavailable(RuntimeError):
    pass


def football_data_rows(path: Path) -> List[dict]:
    """Seasons from ``FIRST_SEASON`` of the ten leagues, each match with both team ids, corners and the 1X2."""
    if not Path(path).exists():
        return []
    leagues = pick_evidence.DOMESTIC
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)) as db:
        rows = db.execute(f"""
            select m.season, m.match_date, m.competition, m.home_team_api_id, m.away_team_api_id, m.home_name,
                   m.away_name, m.fthg, m.ftag, m.hc + m.ac,
                   max(case when o.selection = 'home' then o.price end),
                   max(case when o.selection = 'draw' then o.price end),
                   max(case when o.selection = 'away' then o.price end)
            from matches m join odds o on o.match_id = m.id and o.market = ? and o.bookmaker = ? and o.closing = ?
            where m.season >= ? and m.competition in ({','.join('?' * len(leagues))})
              and m.home_team_api_id is not null and m.away_team_api_id is not null
              and m.fthg is not null and m.ftag is not null and m.hc is not null and m.ac is not null
            group by m.id""", (*PRICE, FIRST_SEASON, *leagues)).fetchall()
    return [{"date": day, "season": f"{season}/{(season + 1) % 100:02d}", "competition": competition,
             "home_id": home_id, "away_id": away_id, "home": home, "away": away, "home_goals": hg, "away_goals": ag,
             "corners": corners, "home_price": ph, "draw_price": pd, "away_price": pa}
            for season, day, competition, home_id, away_id, home, away, hg, ag, corners, ph, pd, pa in rows
            if all(p and p > 1 for p in (ph, pd, pa))]


class History:
    def __init__(self, *, football_data: Path = FOOTBALL_DATA, platform=None, keep: timedelta = KEEP, gate=None,
                 retry: timedelta = RETRY):
        """``gate`` makes the context each database read runs in; by default the refresh gate for the real database
        and none for an injected ``platform`` (tests)."""
        self._path, self._platform, self._keep, self._retry = Path(football_data), platform, keep, retry
        self._gate = gate if gate is not None else (nullcontext if platform is not None else None)
        self._failed_at = None
        self._static: Optional[List[dict]] = None
        self._names: Dict[int, str] = {}
        self._prices: Dict[str, dict] = {}
        self._teams: Dict[str, tuple] = {}
        self._loaded = None  # (when, rows, by_team, by_league)
        self._lock = threading.Lock()

    def _repository(self):
        if self._platform is None:
            from ..repositories.pick_evidence import PickEvidenceRepository
            self._platform = PickEvidenceRepository()
        return self._platform

    def _load(self) -> List[dict]:
        if self._static is None:
            self._static = football_data_rows(self._path)
        platform = self._repository()
        if not self._names and self._static:
            self._names = platform.names(sorted({r[k] for r in self._static for k in ("home_id", "away_id")}))
        after = max((int(r["season"][:4]) for r in self._static), default=FIRST_SEASON - 1)
        finished = platform.finished(after_season=after, competitions=pick_evidence.DOMESTIC)
        missing = [r["event_id"] for r in finished if r["event_id"] not in self._prices]
        if missing:
            self._prices.update(platform.prices(missing))
        rows = [{**r, "home": self._names.get(r["home_id"], r["home"]),
                 "away": self._names.get(r["away_id"], r["away"])} for r in self._static]
        for r in finished:
            price = self._prices.get(r["event_id"])
            if price and r["corners"] is not None:
                rows.append({key: value for key, value in r.items() if key != "event_id"} | {
                    "home_price": price["home"], "draw_price": price["draw"], "away_price": price["away"]})
        return rows

    def _read(self, now, read):
        """Run one database read inside the gate, or refuse at once while the last failure is under ``RETRY`` old."""
        if self._failed_at is not None and now - self._failed_at < self._retry:
            raise HistoryUnavailable("the platform database was busy; trying again later")
        if self._gate is None:
            from .refresh_coordination import read_gate
            self._gate = read_gate
        try:
            with self._gate():
                value = read()
        except Exception:
            self._failed_at = now
            raise
        self._failed_at = None
        return value

    def _current(self, now):
        with self._lock:
            if self._loaded is None or now - self._loaded[0] >= self._keep:
                try:
                    rows = self._read(now, self._load)
                except Exception as exc:
                    if self._loaded is None:
                        raise
                    if not isinstance(exc, HistoryUnavailable):
                        logger.warning("Pick evidence keeps the history loaded at %s (%s)",
                                       self._loaded[0].isoformat(), type(exc).__name__)
                    return self._loaded
                by_team: Dict[int, list] = {}
                by_league: Dict[str, list] = {}
                for row in rows:
                    by_team.setdefault(row["home_id"], []).append(row)
                    by_team.setdefault(row["away_id"], []).append(row)
                    by_league.setdefault(row["competition"], []).append(row)
                self._loaded = (now, rows, by_team, by_league)
            return self._loaded

    def rows(self, now) -> List[dict]:
        return self._current(now)[1]

    def subset(self, now, league: str, team_ids: Sequence[int]) -> List[dict]:
        """The matches either team played or tonight's league had, each once."""
        _, _, by_team, by_league = self._current(now)
        chosen = {id(r): r for r in by_league.get(league, [])}
        for team in team_ids:
            chosen.update((id(r), r) for r in by_team.get(team, []))
        return list(chosen.values())

    def teams(self, event_id: str, now) -> Optional[tuple]:
        if event_id not in self._teams:
            found = self._read(now, lambda: self._repository().teams(event_id))
            if not found:
                return None
            self._teams[event_id] = tuple(found)
        return self._teams[event_id]


def _corners(canonical_results) -> Optional[float]:
    from ..fair_prices import _results

    results = canonical_results if isinstance(canonical_results, Sequence) and not isinstance(
        canonical_results, (str, bytes)) else []
    value = ((_results(results).get("corners") or {}).get("projection") or {}).get("value")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class PickEvidence:
    def __init__(self, history: Optional[History] = None):
        self._history = history or History()

    def attach(self, card: MutableMapping[str, Any], canonical_results, *, now) -> None:
        """Add ``story``, ``evidence`` and the picks' ``evidence`` to ``card`` in place; on any failure the card keeps
        what it had."""
        block = card.get("recommendations")
        if not isinstance(block, MutableMapping):
            return
        fixture, metrics = card.get("fixture") or {}, card.get("metrics") or {}
        figures = {"result_chances": metrics.get("result_probabilities"), "goals_expected": metrics.get("team_goals"),
                   "corners_expected": _corners(canonical_results)}
        history, teams = [], None
        if fixture.get("league") in pick_evidence.DOMESTIC and block.get("most_likely"):
            try:
                teams = self._history.teams(str(fixture.get("event_id") or ""), now)
                history = self._history.subset(now, fixture["league"], teams) if teams else []
            except Exception as exc:  # the card must never fail for its evidence
                logger.warning("Pick evidence history unavailable for %s (%s)", fixture.get("event_id"),
                               type(exc).__name__)
                history, teams = [], None
        try:
            built = pick_evidence.build(block, fixture, history=history, team_ids=teams, figures=figures)
        except Exception as exc:
            logger.warning("Pick evidence skipped for %s (%s)", fixture.get("event_id"), type(exc).__name__)
            return
        if built["story"]:
            block["story"] = built["story"]
        if built["evidence"]:
            block["evidence"] = built["evidence"]
        for index, evidence in built["picks"].items():
            block["most_likely"][index]["evidence"] = evidence
