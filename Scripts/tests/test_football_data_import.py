from datetime import date
import sqlite3

import pytest

from Scripts.ops import football_data_import as fd

HEADER = ("Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,"
          "PSH,PSD,PSA,PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA,P>2.5,P<2.5,AvgC>2.5,AvgC<2.5,AHh,PAHH,PAHA,AHCh,AvgCAHH,AvgCAHA")


def row(home, away, day, hg, ag, *, div="E0", odds="2.1,3.4,3.6,2.0,3.5,3.8,1.95,3.4,3.7,1.9,2.0,1.85,1.95,-0.5,1.9,2.0,-0.25,1.88,1.98"):
    result = "H" if hg > ag else "A" if ag > hg else "D"
    return f"{div},{day},15:00,{home},{away},{hg},{ag},{result},0,0,D,10,8,4,3,11,12,6,4,2,1,0,0,{odds}"


def write(path, lines, bom=True):
    path.write_bytes(("﻿" if bom else "").encode() + ("\r\n".join(lines) + "\r\n").encode())
    return path


def platform(tmp_path, fixtures):
    """A minimal copy of the platform tables the importer reads."""
    db = sqlite3.connect(tmp_path / "platform.db")
    db.executescript("""
        CREATE TABLE competitions (id INTEGER PRIMARY KEY, code TEXT, api_football_id INTEGER);
        CREATE TABLE seasons (id INTEGER PRIMARY KEY, competition_id INTEGER, year INTEGER);
        CREATE TABLE teams (id INTEGER PRIMARY KEY, api_football_id INTEGER, name TEXT);
        CREATE TABLE fixtures (id INTEGER PRIMARY KEY, api_football_id INTEGER, competition_id INTEGER,
            season_id INTEGER, home_team_id INTEGER, away_team_id INTEGER, kickoff_utc TEXT,
            status TEXT, home_goals INTEGER, away_goals INTEGER);
        INSERT INTO competitions VALUES (1, 'EPL', 39);
        INSERT INTO seasons VALUES (1, 1, 2024);
        INSERT INTO teams VALUES (1, 33, 'Manchester United'), (2, 36, 'Fulham'), (3, 40, 'Liverpool'),
                                 (4, 57, 'Ipswich');
    """)
    for i, (home, away, kickoff, hg, ag) in enumerate(fixtures, start=1):
        db.execute("INSERT INTO fixtures VALUES (?, ?, 1, 1, ?, ?, ?, 'FT', ?, ?)",
                   (i, 1000 + i, home, away, kickoff, hg, ag))
    db.commit()
    db.close()
    return tmp_path / "platform.db"


def test_division_and_season_come_from_the_file_name(tmp_path):
    assert fd.file_identity(tmp_path / "E0_2425.csv") == ("E0", 2024)
    assert fd.file_identity(tmp_path / "SP1_1213.csv") == ("SP1", 2012)
    with pytest.raises(ValueError):
        fd.file_identity(tmp_path / "EO_2425.csv")
    with pytest.raises(ValueError):
        fd.file_identity(tmp_path / "E0_2427.csv")


def test_both_date_formats_parse():
    assert fd.parse_date("13/08/12") == date(2012, 8, 13)
    assert fd.parse_date("16/08/2024") == date(2024, 8, 16)


@pytest.mark.parametrize("column, expected", [
    ("PSCH", ("Pinnacle", True, "1x2", "home")),
    ("PSA", ("Pinnacle", False, "1x2", "away")),
    ("P>2.5", ("Pinnacle", False, "ou25", "over")),
    ("PC<2.5", ("Pinnacle", True, "ou25", "under")),
    ("PCAHH", ("Pinnacle", True, "ah", "home")),
    ("BFD", ("Betfair", False, "1x2", "draw")),
    ("BFDD", ("Betfred", False, "1x2", "draw")),
    ("BFDCA", ("Betfred", True, "1x2", "away")),
    ("BFECAHH", ("Betfair Exchange", True, "ah", "home")),
    ("AvgCAHA", ("Market average", True, "ah", "away")),
    ("BbAvAHH", ("BetBrain average", False, "ah", "home")),
    ("BbMx>2.5", ("BetBrain maximum", False, "ou25", "over")),
    ("1XBCH", ("1xBet", True, "1x2", "home")),
])
def test_odds_columns_resolve_to_bookmaker_timing_market_and_side(column, expected):
    assert fd.odds_column(column) == expected


@pytest.mark.parametrize("column", ["Bb1X2", "BbAH", "BbAHh", "AHh", "AHCh", "HC", "FTHG", "Referee", ""])
def test_non_price_columns_are_not_odds(column):
    assert fd.odds_column(column) is None


def test_file_reads_bom_and_skips_rows_without_a_division(tmp_path):
    path = write(tmp_path / "E0_2425.csv", [HEADER, row("Man United", "Fulham", "16/08/2024", 1, 0),
                                            ",,,,,,#REF!", ",,,"])
    matches = fd.read_file(path)
    assert len(matches) == 1
    m = matches[0]
    assert (m["div"], m["season"], m["match_date"], m["kickoff_time"]) == ("E0", 2024, "2024-08-16", "15:00")
    assert (m["home_name"], m["away_name"], m["fthg"], m["ftag"], m["hc"], m["ay"]) == ("Man United", "Fulham", 1, 0, 6, 1)


def test_asian_handicap_prices_carry_their_own_line(tmp_path):
    path = write(tmp_path / "E0_2425.csv", [HEADER, row("Man United", "Fulham", "16/08/2024", 1, 0)])
    odds = {(o["bookmaker"], o["closing"], o["market"], o["selection"]): o for o in fd.read_file(path)[0]["odds"]}
    assert odds[("Pinnacle", False, "ah", "home")]["line"] == -0.5
    assert odds[("Market average", True, "ah", "away")]["line"] == -0.25
    assert odds[("Pinnacle", True, "1x2", "home")]["price"] == 2.0
    assert odds[("Pinnacle", True, "1x2", "home")]["line"] is None


def test_betbrain_handicap_uses_the_betbrain_line(tmp_path):
    path = write(tmp_path / "E0_1213.csv", ["Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,BbAHh,BbAvAHH,BbAvAHA",
                                            "E0,18/08/12,Arsenal,Sunderland,0,0,D,-1.5,2.05,1.83"], bom=False)
    (home, away) = fd.read_file(path)[0]["odds"]
    assert (home["line"], home["price"], away["selection"]) == (-1.5, 2.05, "away")


def test_blank_and_invalid_prices_are_dropped(tmp_path):
    odds = "2.1,,3.6,2.0,3.5,1.0,1.95,3.4,3.7,x,2.0,1.85,1.95,,1.9,2.0,-0.25,1.88,1.98"
    path = write(tmp_path / "E0_2425.csv", [HEADER, row("Man United", "Fulham", "16/08/2024", 1, 0, odds=odds)])
    kept = {(o["bookmaker"], o["closing"], o["market"], o["selection"]) for o in fd.read_file(path)[0]["odds"]}
    assert ("Pinnacle", False, "1x2", "draw") not in kept      # blank
    assert ("Pinnacle", True, "1x2", "away") not in kept       # 1.0 is not a real price
    assert ("Pinnacle", False, "ou25", "over") not in kept     # not a number
    assert ("Pinnacle", False, "ah", "home") not in kept       # no handicap line
    assert ("Pinnacle", True, "1x2", "home") in kept


def test_pinnacle_prices_are_flagged_from_2025_26(tmp_path):
    old = write(tmp_path / "E0_2425.csv", [HEADER, row("Man United", "Fulham", "16/08/2024", 1, 0)])
    new = write(tmp_path / "E0_2526.csv", [HEADER, row("Man United", "Fulham", "16/08/2025", 1, 0)])
    assert not any(o["suspect"] for o in fd.read_file(old)[0]["odds"])
    flagged = {o["bookmaker"] for o in fd.read_file(new)[0]["odds"] if o["suspect"]}
    assert flagged == {"Pinnacle"}


def test_team_names_are_learned_and_rows_linked_to_fixtures(tmp_path):
    db = platform(tmp_path, [(1, 2, "2024-08-16 19:00:00", 1, 0), (4, 3, "2024-08-17 11:30:00", 0, 2),
                             (3, 2, "2024-09-14 14:00:00", 2, 0), (2, 4, "2024-09-21 14:00:00", 1, 1),
                             (1, 3, "2024-10-05 14:00:00", 0, 3), (2, 1, "2024-11-02 15:00:00", 0, 1)])
    write(tmp_path / "E0_2425.csv", [HEADER,
                                     row("Man United", "Fulham", "16/08/2024", 1, 0),
                                     row("Ipswich", "Liverpool", "17/08/2024", 0, 2),
                                     row("Liverpool", "Fulham", "14/09/2024", 2, 0),
                                     row("Fulham", "Ipswich", "21/09/2024", 1, 1),
                                     row("Man United", "Liverpool", "05/10/2024", 0, 2),
                                     row("Fulham", "Man United", "02/11/2024", 0, 1)])
    matches = [m for f in sorted(tmp_path.glob("*.csv")) for m in fd.read_file(f)]
    names = fd.learn_team_names(matches, db, min_votes=2)
    assert names[("England", "Man United")]["team_api_id"] == 33
    assert names[("England", "Liverpool")]["team_api_id"] == 40
    fd.link_fixtures(matches, names, db)
    status = {(m["home_name"], m["away_name"]): (m["link_status"], m["fixture_api_id"]) for m in matches}
    assert status[("Man United", "Fulham")] == ("matched", 1001)
    assert status[("Fulham", "Ipswich")] == ("matched", 1004)
    assert status[("Man United", "Liverpool")] == ("score_mismatch", 1005)


def test_seasons_we_do_not_hold_keep_team_ids_without_a_fixture(tmp_path):
    db = platform(tmp_path, [(1, 2, "2024-08-16 19:00:00", 1, 0), (2, 1, "2025-02-01 15:00:00", 2, 2)])
    write(tmp_path / "E0_2425.csv", [HEADER, row("Man United", "Fulham", "16/08/2024", 1, 0),
                                     row("Fulham", "Man United", "01/02/2025", 2, 2)])
    write(tmp_path / "E0_1213.csv", [HEADER, row("Man United", "Fulham", "18/08/2012", 3, 1),
                                     row("Wigan", "Fulham", "25/08/2012", 0, 0)])
    matches = [m for f in sorted(tmp_path.glob("*.csv")) for m in fd.read_file(f)]
    names = fd.learn_team_names(matches, db, min_votes=2)
    fd.link_fixtures(matches, names, db)
    old = {m["home_name"]: m for m in matches if m["season"] == 2012}
    assert (old["Man United"]["link_status"], old["Man United"]["home_team_api_id"]) == ("season_not_covered", 33)
    assert old["Wigan"]["link_status"] == "team_unmapped"
    assert old["Wigan"]["home_team_api_id"] is None


def test_build_writes_a_fresh_database_with_a_summary(tmp_path):
    db = platform(tmp_path, [(1, 2, "2024-08-16 19:00:00", 1, 0), (2, 1, "2025-02-01 15:00:00", 2, 2)])
    raw = tmp_path / "raw"
    raw.mkdir()
    write(raw / "E0_2425.csv", [HEADER, row("Man United", "Fulham", "16/08/2024", 1, 0),
                                row("Fulham", "Man United", "01/02/2025", 2, 2)])
    (raw / "notes.txt").write_text("notes")
    out = tmp_path / "football_data.sqlite3"
    for _ in range(2):  # a rebuild replaces, never appends
        summary = fd.build(raw, out, db, min_votes=2)
    assert summary["files"] == 1 and summary["matches"] == 2
    assert summary["link_status"] == {"matched": 2}
    con = sqlite3.connect(out)
    assert con.execute("SELECT count(*) FROM matches").fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM odds").fetchone()[0] == 2 * 17  # 19 odds columns, 2 are handicap lines
    assert con.execute("SELECT sha256 FROM files").fetchone()[0]
    assert con.execute("SELECT count(*) FROM team_names WHERE team_api_id IS NOT NULL").fetchone()[0] == 2
