"""Game-read bench (D77, phase 2): each bench match's statistics, exported once from our own database.

``export`` makes one read-only pass over ``Index/platform.db`` inside the refresh gate (``data_access``), and writes
``Index/game_read_stats.sqlite3``; the bench reads only that file, never the platform database. Per football-data
match (keyed by its id in the bench's history file):

- shots inside and outside the box, per side, from our fixture statistics (linked matches only, 2019/20 on);
- shots on target, per side, from football-data's own columns (2012/13 on in most leagues);
- real xG, per side: exported only to report how close the derived shot-location xG comes on 2022/23 (a tuning
  season); no model reads it (owner, 16 Nov);
- rest, per side: whole days since the team's previous finished fixture in any competition we hold (domestic
  leagues and the three European cups, not domestic cups), from 2019/20. None for a team's first fixture we hold.

Sides follow our fixture's home and away teams. An unlinked match has no box, xG or rest.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from contextlib import closing
from dataclasses import astuple, dataclass, fields
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Callable, Optional

from Scripts.data_platform.game_read.history import BENCH_DB
from Scripts.data_platform.services.refresh_coordination import data_access
from Scripts.ops.football_data_import import PLATFORM_DB

STATS_DB = BENCH_DB.with_name("game_read_stats.sqlite3")
FINISHED = ("FT", "AET", "PEN")


@dataclass(frozen=True)
class MatchStats:
    home_box: Optional[int] = None  # shots inside the box
    away_box: Optional[int] = None
    home_outside: Optional[int] = None  # shots outside the box
    away_outside: Optional[int] = None
    home_target: Optional[int] = None  # shots on target
    away_target: Optional[int] = None
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    home_rest: Optional[int] = None  # whole days since the previous fixture
    away_rest: Optional[int] = None


COLUMNS = [f.name for f in fields(MatchStats)]


def _kickoff(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _platform(path: Path) -> tuple:
    """Our finished fixtures by API id (fixture id, home team, away team, kickoff), each team's kickoffs in order,
    and each fixture's per-team statistics."""
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)) as db:
        marks = ", ".join("?" * len(FINISHED))
        fixtures = {api: (id, home, away, _kickoff(kickoff)) for id, api, home, away, kickoff in db.execute(
            f"SELECT id, api_football_id, home_team_id, away_team_id, kickoff_utc FROM fixtures "
            f"WHERE status IN ({marks})", FINISHED)}
        team_stats = {(fixture, team): (inside, outside, xg) for fixture, team, inside, outside, xg in db.execute(
            "SELECT fixture_id, team_id, shots_inside_box, shots_outside_box, expected_goals FROM fixture_team_stats")}
    kickoffs = defaultdict(list)
    for _, home, away, kickoff in fixtures.values():
        kickoffs[home].append(kickoff)
        kickoffs[away].append(kickoff)
    for times in kickoffs.values():
        times.sort()
    return fixtures, kickoffs, team_stats


def _rest(kickoffs: list, kickoff: datetime) -> Optional[int]:
    before = bisect_left(kickoffs, kickoff)
    return None if before == 0 else (kickoff - kickoffs[before - 1]).days


def export(out: Path = STATS_DB, *, history_db: Path = BENCH_DB, platform_db: Path = PLATFORM_DB,
           gate: Callable = data_access) -> dict:
    """Write every bench match's statistics to ``out``; raises ``RefreshBusy`` while the refresh writes, and then
    writes nothing. Returns counts of what was found."""
    with closing(sqlite3.connect(f"file:{history_db}?mode=ro", uri=True)) as db:
        matches = db.execute("SELECT id, fixture_api_id, hst, ast FROM matches").fetchall()
    with gate():
        fixtures, kickoffs, team_stats = _platform(platform_db)
    rows, counts = [], defaultdict(int)
    for match_id, fixture_api, home_target, away_target in matches:
        found = {"home_target": home_target, "away_target": away_target}
        fixture = fixtures.get(fixture_api) if fixture_api is not None else None
        if fixture is not None:
            id, home, away, kickoff = fixture
            for side, team in (("home", home), ("away", away)):
                inside, outside, xg = team_stats.get((id, team), (None, None, None))
                found.update({f"{side}_box": inside, f"{side}_outside": outside, f"{side}_xg": xg,
                              f"{side}_rest": _rest(kickoffs[team], kickoff)})
            counts["linked"] += 1
        stat = MatchStats(**found)
        counts["box"] += stat.home_box is not None and stat.away_box is not None
        counts["rest"] += stat.home_rest is not None and stat.away_rest is not None
        rows.append((match_id, *astuple(stat)))
    building = out.with_suffix(".building")
    building.unlink(missing_ok=True)
    with closing(sqlite3.connect(building)) as db:
        db.execute(f"CREATE TABLE match_stats (match_id INTEGER PRIMARY KEY, {', '.join(COLUMNS)})")
        db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        db.executemany(f"INSERT INTO match_stats VALUES ({', '.join('?' * (len(COLUMNS) + 1))})", rows)
        db.executemany("INSERT INTO meta VALUES (?, ?)", [("built", datetime.now().isoformat(timespec="seconds")),
                                                          ("history", str(history_db)), ("matches", str(len(rows)))])
        db.commit()
    building.replace(out)
    return {"matches": len(rows), **counts}


def load(path: Path = STATS_DB) -> dict:
    """Every match's statistics by its id in the bench's history file; opened read-only."""
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        return {row[0]: MatchStats(*row[1:]) for row in db.execute(
            f"SELECT match_id, {', '.join(COLUMNS)} FROM match_stats")}
