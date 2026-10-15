"""Load the football-data.co.uk history CSVs into a backtest database.

The raw files in ``Research/football-data/raw/`` (named ``{DIV}_{YYZZ}.csv``,
e.g. ``E0_2425.csv``) are never modified. Each run rebuilds
``Index/football_data.sqlite3`` from them:

* ``matches``: one row per match with result and match stats, the
  API-Football team ids and, where we hold the season, the linked fixture;
* ``odds``: one row per price (market, bookmaker, pre-closing or closing,
  selection, handicap line). Column names follow notes.txt: a ``C`` after the
  bookmaker marks the closing price. Pinnacle prices from 2025/26 are flagged
  ``suspect`` (football-data.co.uk: its feed is unreliable since 23 July 2025);
* ``team_names``: the football-data name of each team and the API-Football
  team it was learned to be;
* ``files``: the source file hashes.

Team names are learned from our own fixtures: a football-data match whose date
(±1 day) and score fit exactly one fixture of the same league-season votes for
its two names. Rows are then linked by teams and date (±3 days), and the score
is compared, so a disagreement is reported instead of guessed. The platform
database is opened read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import csv
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from Scripts.ops.odds_archive import ROOT

RAW_DIR = ROOT / "Research/football-data/raw"
OUT_DB = ROOT / "Index/football_data.sqlite3"
PLATFORM_DB = ROOT / "Index/platform.db"
PINNACLE_UNRELIABLE_FROM = 2025  # season start year

# football-data division -> (platform competition code, country)
DIVISIONS = {
    "E0": ("EPL", "England"), "E1": ("Championship", "England"), "SP1": ("LaLiga", "Spain"),
    "I1": ("SerieA", "Italy"), "D1": ("Bundesliga", "Germany"), "F1": ("Ligue1", "France"),
    "N1": ("Eredivisie", "Netherlands"), "P1": ("PrimeiraLiga", "Portugal"),
    "B1": ("BelgianProLeague", "Belgium"), "T1": ("SuperLig", "Turkey"),
}
BOOKMAKERS = {
    "1XB": "1xBet", "B365": "Bet365", "BF": "Betfair", "BFD": "Betfred", "BFE": "Betfair Exchange",
    "BMGM": "BetMGM", "BS": "Blue Square", "BV": "BetVictor", "BW": "Bet&Win", "CL": "Coral",
    "GB": "Gamebookers", "IW": "Interwetten", "LB": "Ladbrokes", "P": "Pinnacle", "PS": "Pinnacle",
    "PP": "Paddy Power", "SB": "Sportingbet", "SJ": "Stan James", "SK": "Skybet", "SO": "Sporting Odds",
    "VC": "VC Bet", "WH": "William Hill", "Max": "Market maximum", "Avg": "Market average",
    "BbMx": "BetBrain maximum", "BbAv": "BetBrain average",
}
SELECTIONS = {
    "H": ("1x2", "home"), "D": ("1x2", "draw"), "A": ("1x2", "away"),
    ">2.5": ("ou25", "over"), "<2.5": ("ou25", "under"),
    "AHH": ("ah", "home"), "AHA": ("ah", "away"),
}
STATS = {"HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast", "HF": "hf", "AF": "af", "HC": "hc", "AC": "ac",
         "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar"}
FILE_NAME = re.compile(r"^([A-Z]+\d)_(\d{2})(\d{2})$")


def file_identity(path):
    """``E0_2425.csv`` -> ("E0", 2024)."""
    m = FILE_NAME.match(Path(path).stem)
    if not m or m.group(1) not in DIVISIONS or int(m.group(3)) != (int(m.group(2)) + 1) % 100:
        raise ValueError(f"expected DIV_YYZZ.csv for a known division, got {Path(path).name}")
    return m.group(1), 2000 + int(m.group(2))


def parse_date(text):
    return datetime.strptime(text.strip(), "%d/%m/%Y" if len(text.strip()) == 10 else "%d/%m/%y").date()


def odds_column(column):
    """``PSCH`` -> ("Pinnacle", True, "1x2", "home"); None for anything that is not a price."""
    for size in range(min(len(column), 4), 0, -1):  # longest bookmaker prefix first: BFDD is Betfred, BFD Betfair
        prefix, rest = column[:size], column[size:]
        if prefix not in BOOKMAKERS:
            continue
        closing_price = rest.startswith("C") and rest[1:] in SELECTIONS
        side = rest[1:] if closing_price else rest
        if side in SELECTIONS:
            return (BOOKMAKERS[prefix], closing_price, *SELECTIONS[side])
    return None


def _number(text, cast=float):
    try:
        return cast(text.strip()) if text and text.strip() else None
    except ValueError:
        return None


def _handicap_line(record, column, closing_price):
    if column.startswith("Bb"):
        key = "BbAHh"
    else:
        key = "AHCh" if closing_price else "AHh"
    return _number(record.get(key, ""))


def read_file(path):
    div, season = file_identity(path)
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        columns = [(c, odds_column(c)) for c in reader.fieldnames or []]
        priced = [(c, spec) for c, spec in columns if spec]
        matches = []
        for record in reader:
            if (record.get("Div") or "").strip() != div:
                continue  # trailing blank / spreadsheet residue rows
            match = dict(
                div=div, season=season, competition=DIVISIONS[div][0], file=Path(path).name,
                match_date=parse_date(record["Date"]).isoformat(),
                kickoff_time=(record.get("Time") or "").strip() or None,
                home_name=record["HomeTeam"].strip(), away_name=record["AwayTeam"].strip(),
                fthg=_number(record.get("FTHG"), int), ftag=_number(record.get("FTAG"), int),
                ftr=(record.get("FTR") or "").strip() or None,
                hthg=_number(record.get("HTHG"), int), htag=_number(record.get("HTAG"), int),
                htr=(record.get("HTR") or "").strip() or None,
                referee=(record.get("Referee") or "").strip() or None,
                **{name: _number(record.get(col), int) for col, name in STATS.items()},
                home_team_api_id=None, away_team_api_id=None, fixture_api_id=None, link_status=None,
            )
            match["odds"] = []
            for column, (bookmaker, closing_price, market, selection) in priced:
                price = _number(record.get(column))
                if price is None or price <= 1.0:
                    continue
                line = _handicap_line(record, column, closing_price) if market == "ah" else None
                if market == "ah" and line is None:
                    continue
                match["odds"].append(dict(
                    bookmaker=bookmaker, closing=closing_price, market=market, selection=selection,
                    line=line, price=price, column=column,
                    suspect=bookmaker == "Pinnacle" and season >= PINNACLE_UNRELIABLE_FROM))
            matches.append(match)
    return matches


def _fixtures(platform_db):
    """Finished fixtures per (competition code, season year), with team api ids."""
    with closing(sqlite3.connect(f"file:{platform_db}?mode=ro", uri=True)) as db:
        rows = db.execute("""
            SELECT c.code, s.year, f.api_football_id, h.api_football_id, a.api_football_id,
                   f.kickoff_utc, f.home_goals, f.away_goals
            FROM fixtures f JOIN competitions c ON c.id = f.competition_id
            JOIN seasons s ON s.id = f.season_id
            JOIN teams h ON h.id = f.home_team_id JOIN teams a ON a.id = f.away_team_id
            WHERE f.status IN ('FT', 'AET', 'PEN')""").fetchall()
    by_scope = defaultdict(list)
    for code, year, fixture, home, away, kickoff, hg, ag in rows:
        by_scope[(code, year)].append(dict(fixture=fixture, home=home, away=away,
                                           day=date.fromisoformat(str(kickoff)[:10]), hg=hg, ag=ag))
    return by_scope


def learn_team_names(matches, platform_db, *, min_votes=3, min_share=0.8):
    """Map (country, football-data name) -> API-Football team id from unambiguous date+score matches."""
    fixtures = _fixtures(platform_db)
    votes = defaultdict(Counter)
    for m in matches:
        scope = fixtures.get((m["competition"], m["season"]))
        if not scope or m["fthg"] is None:
            continue
        day = date.fromisoformat(m["match_date"])
        fits = [f for f in scope if abs((f["day"] - day).days) <= 1 and (f["hg"], f["ag"]) == (m["fthg"], m["ftag"])]
        if len(fits) == 1:
            country = DIVISIONS[m["div"]][1]
            votes[(country, m["home_name"])][fits[0]["home"]] += 1
            votes[(country, m["away_name"])][fits[0]["away"]] += 1
    names = {}
    for key in {(DIVISIONS[m["div"]][1], n) for m in matches for n in (m["home_name"], m["away_name"])}:
        tally = votes.get(key, Counter())
        total = sum(tally.values())
        team, count = tally.most_common(1)[0] if tally else (None, 0)
        accepted = count >= min_votes and count / total >= min_share
        names[key] = dict(team_api_id=team if accepted else None, votes=count, total_votes=total,
                          status="learned" if accepted else ("ambiguous" if total else "unseen"))
    return names


def link_fixtures(matches, names, platform_db, *, window_days=3):
    """Set team ids, fixture id and link_status on every match (in place)."""
    fixtures = _fixtures(platform_db)
    held = {scope for scope in fixtures}
    taken = set()
    for m in matches:
        country = DIVISIONS[m["div"]][1]
        m["home_team_api_id"] = names.get((country, m["home_name"]), {}).get("team_api_id")
        m["away_team_api_id"] = names.get((country, m["away_name"]), {}).get("team_api_id")
        scope = (m["competition"], m["season"])
        if m["home_team_api_id"] is None or m["away_team_api_id"] is None:
            m["link_status"] = "team_unmapped"
            continue
        if scope not in held:
            m["link_status"] = "season_not_covered"
            continue
        day = date.fromisoformat(m["match_date"])
        fits = [f for f in fixtures[scope] if (f["home"], f["away"]) == (m["home_team_api_id"], m["away_team_api_id"])
                and abs((f["day"] - day).days) <= window_days]
        if not fits:
            m["link_status"] = "no_fixture"
            continue
        best = min(fits, key=lambda f: abs((f["day"] - day).days))
        m["fixture_api_id"] = best["fixture"]
        if best["fixture"] in taken:
            m["link_status"] = "duplicate_fixture"
        elif (best["hg"], best["ag"]) != (m["fthg"], m["ftag"]):
            m["link_status"] = "score_mismatch"
        else:
            m["link_status"] = "matched"
        taken.add(best["fixture"])
    return matches


SCHEMA = """
CREATE TABLE files (file TEXT PRIMARY KEY, div TEXT, season INTEGER, sha256 TEXT, matches INTEGER);
CREATE TABLE matches (
    id INTEGER PRIMARY KEY, file TEXT, div TEXT, competition TEXT, season INTEGER,
    match_date TEXT, kickoff_time TEXT, home_name TEXT, away_name TEXT,
    home_team_api_id INTEGER, away_team_api_id INTEGER, fixture_api_id INTEGER, link_status TEXT,
    fthg INTEGER, ftag INTEGER, ftr TEXT, hthg INTEGER, htag INTEGER, htr TEXT,
    hs INTEGER, as_ INTEGER, hst INTEGER, ast INTEGER, hf INTEGER, af INTEGER, hc INTEGER, ac INTEGER,
    hy INTEGER, ay INTEGER, hr INTEGER, ar INTEGER, referee TEXT);
CREATE TABLE odds (
    match_id INTEGER REFERENCES matches(id), market TEXT, bookmaker TEXT, closing INTEGER,
    selection TEXT, line REAL, price REAL, source_column TEXT, suspect INTEGER);
CREATE TABLE team_names (
    country TEXT, fd_name TEXT, team_api_id INTEGER, votes INTEGER, total_votes INTEGER, status TEXT,
    PRIMARY KEY (country, fd_name));
CREATE INDEX odds_match ON odds(match_id, market, closing);
CREATE INDEX matches_fixture ON matches(fixture_api_id);
CREATE INDEX matches_scope ON matches(competition, season, match_date);
"""
MATCH_COLUMNS = ["file", "div", "competition", "season", "match_date", "kickoff_time", "home_name", "away_name",
                 "home_team_api_id", "away_team_api_id", "fixture_api_id", "link_status", "fthg", "ftag", "ftr",
                 "hthg", "htag", "htr", *STATS.values(), "referee"]


def build(raw_dir=RAW_DIR, out_path=OUT_DB, platform_db=PLATFORM_DB, *, min_votes=3):
    raw_dir, out_path = Path(raw_dir), Path(out_path)
    paths = sorted(raw_dir.glob("*.csv"))
    files = {p.name: (file_identity(p), hashlib.sha256(p.read_bytes()).hexdigest(), read_file(p)) for p in paths}
    matches = [m for _, _, rows in files.values() for m in rows]
    names = learn_team_names(matches, platform_db, min_votes=min_votes)
    link_fixtures(matches, names, platform_db)

    temp = out_path.with_suffix(".building")
    temp.unlink(missing_ok=True)
    with closing(sqlite3.connect(temp)) as db:
        db.executescript(SCHEMA)
        for name, ((div, season), digest, rows) in files.items():
            db.execute("INSERT INTO files VALUES (?, ?, ?, ?, ?)", (name, div, season, digest, len(rows)))
        marks = ", ".join("?" * len(MATCH_COLUMNS))
        for m in matches:
            cursor = db.execute(f"INSERT INTO matches ({', '.join(MATCH_COLUMNS)}) VALUES ({marks})",
                                [m[c] for c in MATCH_COLUMNS])
            db.executemany("INSERT INTO odds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                (cursor.lastrowid, o["market"], o["bookmaker"], int(o["closing"]), o["selection"], o["line"],
                 o["price"], o["column"], int(o["suspect"])) for o in m["odds"]])
        db.executemany("INSERT INTO team_names VALUES (?, ?, ?, ?, ?, ?)", [
            (country, fd_name, v["team_api_id"], v["votes"], v["total_votes"], v["status"])
            for (country, fd_name), v in sorted(names.items())])
        db.commit()
    temp.replace(out_path)
    return summary(files, matches, names)


def summary(files, matches, names):
    return dict(
        files=len(files), matches=len(matches), odds=sum(len(m["odds"]) for m in matches),
        link_status=dict(Counter(m["link_status"] for m in matches).most_common()),
        team_names=dict(Counter(v["status"] for v in names.values()).most_common()),
        unmapped_names=sorted(f"{c}: {n}" for (c, n), v in names.items() if v["team_api_id"] is None),
        score_mismatches=[f"{m['file']} {m['match_date']} {m['home_name']} v {m['away_name']}"
                          for m in matches if m["link_status"] == "score_mismatch"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DB)
    parser.add_argument("--platform-db", type=Path, default=PLATFORM_DB)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.raw, args.out, args.platform_db), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
