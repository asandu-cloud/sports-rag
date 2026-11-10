"""The stats file: flat, pre-joined tables built from platform.db, one short read per season.

Questions only ever open this file, read-only. The build reads the platform database through the Research Area's own
queries and count rules, so the two never disagree, and swaps the new file in whole.
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from Scripts.data_platform.models import Competition, Fixture, Player, Team
from Scripts.data_platform.repositories.research import FINISHED, ResearchRepository, comparable_rounds, records
from Scripts.data_platform.research_stats import normalize_research_row
from Scripts.data_platform.services.refresh_coordination import data_access
from Scripts.data_platform.services.research_profiles import add_known, referee_aliases, starter
from Scripts.ops.odds_archive import ROOT

DEFAULT_PATH = ROOT / "Index" / "stats_questions" / "stats.db"
# stats-file name -> research row field. Each team stat is stored for the team and against it (the opponent's).
TEAM_FIELDS = {"corners": "corners", "shots": "shots_total", "shots_on_target": "shots_on", "xg": "expected_goals",
               "fouls": "fouls_committed", "offsides": "offsides", "saves": "goalkeeper_saves",
               "yellow_cards": "yellow_cards", "red_cards": "red_cards", "possession": "possession"}
PLAYER_FIELDS = {"goals": "goals", "assists": "assists", "shots": "shots_total", "shots_on_target": "shots_on",
                 "fouls_committed": "fouls_committed", "fouls_drawn": "fouls_drawn", "yellow_cards": "yellow_cards",
                 "red_cards": "red_cards", "tackles": "tackles", "interceptions": "interceptions",
                 "passes": "passes_total", "pass_accuracy": "pass_accuracy", "rating": "rating"}
TEAM_STATS = (*TEAM_FIELDS, "cards")
PLAYER_STATS = (*PLAYER_FIELDS, "cards")

SCHEMA = f"""
CREATE TABLE build (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE competitions (code TEXT PRIMARY KEY, name TEXT, type TEXT, country TEXT);
CREATE TABLE teams (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT, league TEXT);
CREATE TABLE players (id INTEGER PRIMARY KEY, name TEXT NOT NULL, team_id INTEGER, position TEXT);
CREATE TABLE referees (referee_key TEXT PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE matches (fixture_id INTEGER PRIMARY KEY, competition TEXT, season INTEGER, kickoff TEXT,
  home_id INTEGER, away_id INTEGER, home_goals INTEGER, away_goals INTEGER, referee_key TEXT);
CREATE TABLE team_matches (fixture_id INTEGER, team_id INTEGER, opponent_id INTEGER, is_home INTEGER,
  competition TEXT, season INTEGER, kickoff TEXT, goals_for INTEGER, goals_against INTEGER,
  {", ".join(f"{s}_for REAL, {s}_against REAL" for s in TEAM_STATS)},
  PRIMARY KEY (fixture_id, team_id));
CREATE TABLE player_matches (fixture_id INTEGER, player_id INTEGER, team_id INTEGER, competition TEXT,
  season INTEGER, kickoff TEXT, started INTEGER, position TEXT, minutes INTEGER,
  {", ".join(f"{s} REAL" for s in PLAYER_STATS)}, PRIMARY KEY (fixture_id, player_id));
CREATE INDEX team_matches_team ON team_matches (team_id, kickoff);
CREATE INDEX team_matches_scope ON team_matches (competition, season);
CREATE INDEX player_matches_player ON player_matches (player_id, kickoff);
CREATE INDEX player_matches_scope ON player_matches (competition, season);
CREATE INDEX matches_referee ON matches (referee_key, kickoff);
CREATE INDEX matches_scope ON matches (competition, season);
"""


def connect(path: Path = DEFAULT_PATH) -> sqlite3.Connection:
    """Read-only: questions can never write, even through free SQL."""
    conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _iso(value):
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _slug(raw: str) -> str:
    return "ref-" + re.sub(r"[^\w]+", "-", raw.lower(), flags=re.UNICODE).strip("-")


def _team_row(row, scope):
    row = normalize_research_row("team", row)
    out = {"fixture_id": row["fixture_id"], "team_id": row["team_id"], "opponent_id": row["opponent_team_id"],
           "is_home": int(bool(row["is_home"])), "competition": scope["code"], "season": scope["year"],
           "kickoff": _iso(row["kickoff_utc"]), "goals_for": row["result_goals_for"],
           "goals_against": row["result_goals_against"]}
    for name, field in TEAM_FIELDS.items():
        out[name + "_for"], out[name + "_against"] = row.get(field), row.get("opponent_" + field)
    out["possession_for"] = None if row.get("possession") is None else round(row["possession"] * 100, 1)
    out["possession_against"] = (None if row.get("opponent_possession") is None
                                 else round(row["opponent_possession"] * 100, 1))
    out["cards_for"] = add_known(row.get("yellow_cards"), row.get("red_cards"))
    out["cards_against"] = add_known(row.get("opponent_yellow_cards"), row.get("opponent_red_cards"))
    return out


def _player_row(row, scope):
    row = normalize_research_row("player", row)
    out = {"fixture_id": row["fixture_id"], "player_id": row["player_id"], "team_id": row["team_id"],
           "competition": scope["code"], "season": scope["year"], "kickoff": _iso(row["kickoff_utc"]),
           "started": int(starter(row)[0]), "position": row.get("position"), "minutes": row["minutes"]}
    for name, field in PLAYER_FIELDS.items():
        out[name] = row.get(field)
    out["cards"] = add_known(row.get("yellow_cards"), row.get("red_cards"))
    return out


def _insert(conn, table, rows):
    if rows:
        cols = list(rows[0])
        conn.executemany(f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                         [tuple(r[c] for c in cols) for r in rows])


def build(target: Path = DEFAULT_PATH, *, session_factory=None, gate=data_access, clock=time.perf_counter) -> dict:
    """Write a new stats file beside ``target`` and swap it in. One read per season and table, each in its own
    session, so no read holds the platform database for long (writers wait at most 2 s, D88)."""
    started = clock()
    repo = ResearchRepository(session_factory) if session_factory else ResearchRepository()
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.unlink(missing_ok=True)
    out = sqlite3.connect(tmp)
    try:
        out.executescript(SCHEMA)
        with gate():
            with repo.session_factory() as s:
                scopes = repo.scopes(s)
                comps = records(s, select(Competition.code, Competition.name, Competition.competition_type,
                                          Competition.country))
                aliases = {a["raw_name"]: a for a in referee_aliases(repo.referee_names(s))}
            _insert(out, "competitions", [{"code": c["code"], "name": c["name"], "type": c["competition_type"],
                                           "country": c["country"]} for c in comps])
            referees = {}
            for scope in scopes:
                with repo.session_factory() as s:
                    teams = [_team_row(r, scope) for r in repo.team_rows(s, scope["season_id"])]
                with repo.session_factory() as s:
                    players = [_player_row(r, scope) for r in repo.player_rows(s, scope["season_id"])]
                with repo.session_factory() as s:
                    f = Fixture
                    fixtures = records(s, select(f.id, f.kickoff_utc, f.home_team_id, f.away_team_id, f.home_goals,
                                                 f.away_goals, f.referee).join(Competition, Competition.id ==
                                                 f.competition_id).where(f.season_id == scope["season_id"],
                                                 f.status.in_(FINISHED), comparable_rounds()))
                matches = []
                for m in fixtures:
                    raw, key = m["referee"], None
                    if raw:
                        alias = aliases.get(raw)
                        key = alias["referee_key"] if alias else _slug(raw)
                        referees.setdefault(key, alias["name"] if alias else raw.split(",")[0].strip())
                    matches.append({"fixture_id": m["id"], "competition": scope["code"], "season": scope["year"],
                                    "kickoff": _iso(m["kickoff_utc"]), "home_id": m["home_team_id"],
                                    "away_id": m["away_team_id"], "home_goals": m["home_goals"],
                                    "away_goals": m["away_goals"], "referee_key": key})
                _insert(out, "matches", matches)
                _insert(out, "team_matches", teams)
                _insert(out, "player_matches", players)
            with repo.session_factory() as s:
                team_rows = records(s, select(Team.id, Team.name, Team.country))
            with repo.session_factory() as s:
                player_rows = records(s, select(Player.id, Player.name))
        _insert(out, "referees", [{"referee_key": k, "name": v} for k, v in referees.items()])
        seen_teams = {r[0] for r in out.execute("SELECT team_id FROM team_matches")}
        _insert(out, "teams", [{"id": t["id"], "name": t["name"], "country": t["country"], "league": None}
                               for t in team_rows if t["id"] in seen_teams])
        seen_players = {r[0] for r in out.execute("SELECT DISTINCT player_id FROM player_matches")}
        _insert(out, "players", [{"id": p["id"], "name": p["name"], "team_id": None, "position": None}
                                 for p in player_rows if p["id"] in seen_players])
        # A team's league: the domestic league it played most in, in its latest season with one.
        out.execute("""UPDATE teams SET league = (
            SELECT tm.competition FROM team_matches tm JOIN competitions c ON c.code = tm.competition
            WHERE tm.team_id = teams.id AND c.type = 'domestic_league'
            GROUP BY tm.competition, tm.season ORDER BY tm.season DESC, count(*) DESC LIMIT 1)""")
        out.execute("""UPDATE players SET (team_id, position) = (
            SELECT pm.team_id, pm.position FROM player_matches pm WHERE pm.player_id = players.id
            ORDER BY pm.kickoff DESC LIMIT 1)""")
        report = {key: out.execute(f"SELECT count(*) FROM {key}").fetchone()[0]
                  for key in ("matches", "team_matches", "player_matches", "teams", "players", "referees")}
        report["data_up_to"] = out.execute("SELECT max(kickoff) FROM matches").fetchone()[0]
        report["built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        report["seconds"] = round(clock() - started, 1)
        _insert(out, "build", [{"key": k, "value": str(v)} for k, v in report.items()])
        out.commit()
    finally:
        out.close()
    os.replace(tmp, target)
    return report
