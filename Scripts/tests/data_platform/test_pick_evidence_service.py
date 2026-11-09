"""Pick evidence on the website's cards (D82): the history (football-data's two seasons and this season from the
platform database) is loaded once an hour, prices are read once per match, and a failure leaves the card as it was."""
import copy
from datetime import datetime, timedelta, timezone
import sqlite3

from Scripts.tests.data_platform.test_pick_evidence import FIGURES, block, pick

NOW = datetime(2026, 11, 21, 12, tzinfo=timezone.utc)
H, A = 40, 52


def football_data(path, n=12):
    """A football-data file with ``n`` home wins for H and ``n`` away draws for A in 2024/25, plus rows that must be
    left out: an older season, a cup-less division, a match without corners and one without prices."""
    db = sqlite3.connect(path)
    db.executescript("""
        create table matches (id integer primary key, competition text, season integer, match_date text,
            home_name text, away_name text, home_team_api_id integer, away_team_api_id integer,
            fthg integer, ftag integer, hc integer, ac integer);
        create table odds (match_id integer, market text, bookmaker text, closing integer, selection text,
            price real);""")
    def add(i, comp, season, home_id, away_id, hg, ag, hc=5, ac=4, prices=(1.6, 4.0, 5.5)):
        db.execute("insert into matches values (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (i, comp, season, f"2025-01-{1 + i % 28:02d}", f"fd{home_id}", f"fd{away_id}", home_id, away_id,
                    hg, ag, hc, ac))
        for selection, price in zip(("home", "draw", "away"), prices or ()):
            db.execute("insert into odds values (?,?,?,?,?,?)", (i, "1x2", "Market average", 0, selection, price))
            db.execute("insert into odds values (?,?,?,?,?,?)", (i, "1x2", "Pinnacle", 0, selection, 9.9))
    for i in range(n):
        add(i, "EPL", 2024, H, 1000 + i, 1, 0)
        add(100 + i, "EPL", 2024, 2000 + i, A, 1, 1, prices=(2.0, 3.4, 5.5))
    for i in range(40):
        add(200 + i, "EPL", 2025, 3000 + i, 4000 + i, 2, 1)
    add(900, "EPL", 2023, H, 5000, 1, 0)                   # before August 2024
    add(901, "Liga Portugal 2", 2024, H, 5000, 1, 0)        # not one of the ten leagues
    add(902, "EPL", 2024, H, 5000, 1, 0, hc=None)           # no corners
    add(903, "EPL", 2024, H, 5000, 1, 0, prices=None)       # no prices
    db.commit(); db.close()
    return path


class Platform:
    """The platform database's side, counting what it was asked."""

    def __init__(self, finished=(), prices=None, teams=(H, A), fail=False):
        self._finished, self._prices, self._teams, self.fail = list(finished), prices or {}, teams, fail
        self.asked_prices, self.asked_finished = [], 0

    def finished(self, *, after_season, competitions):
        if self.fail:
            raise RuntimeError("database is locked")
        self.asked_finished += 1
        assert after_season == 2025
        return [dict(r) for r in self._finished]

    def prices(self, event_ids):
        self.asked_prices.append(sorted(event_ids))
        return {e: self._prices[e] for e in event_ids if e in self._prices}

    def teams(self, event_id):
        return self._teams

    def names(self, api_ids):
        return {H: "Manchester City", A: "Nottingham Forest"}


def this_season(event_id, day, corners=10):
    return {"event_id": event_id, "date": f"2026-10-{day:02d}", "season": "2026/27", "competition": "EPL",
            "home_id": H, "away_id": 77, "home": "Manchester City", "away": "Wolves", "home_goals": 3,
            "away_goals": 0, "corners": corners}


def card(league="EPL", most_likely=None):
    over = pick("goals", "over", 1.5, chance=0.7, price=1.33, label="Over 1.5 goals")
    return {"fixture": {"event_id": "1557465", "league": league, "home_team": "Manchester City",
                        "away_team": "Nottingham Forest", "kickoff": "2026-11-21T15:00:00+00:00"},
            "metrics": {"result_probabilities": FIGURES["result_chances"], "team_goals": FIGURES["goals_expected"],
                        "projected_total_goals": 2.7},
            "recommendations": block([over] if most_likely is None else most_likely)}


CORNERS = [{"market": {"group": "corners"}, "projection": {"value": 10.4}}]


def service(tmp_path, platform, **kw):
    from data_platform.services.pick_evidence_service import History, PickEvidence

    return PickEvidence(History(football_data=football_data(tmp_path / "fd.sqlite3", **kw), platform=platform))


def test_history_keeps_two_seasons_of_the_ten_leagues_with_prices_and_corners_and_adds_this_season(tmp_path):
    from data_platform.services.pick_evidence_service import History

    platform = Platform(finished=[this_season("1", 3), this_season("2", 4), this_season("3", 5, corners=None)],
                        prices={"1": {"home": 1.4, "draw": 4.5, "away": 7.0}, "3": {"home": 1.5, "draw": 4, "away": 6}})
    rows = History(football_data=football_data(tmp_path / "fd.sqlite3"), platform=platform).rows(NOW)
    assert len(rows) == 12 + 12 + 40 + 1
    first = next(r for r in rows if r["home_id"] == H and r["season"] == "2024/25")
    assert first == {"date": first["date"], "season": "2024/25", "competition": "EPL", "home_id": H, "away_id": 1000,
                     "home": "Manchester City", "away": "fd1000", "home_goals": 1, "away_goals": 0, "corners": 9,
                     "home_price": 1.6, "draw_price": 4.0, "away_price": 5.5}
    ours = next(r for r in rows if r["season"] == "2026/27")
    assert (ours["date"], ours["home_price"], ours["away_price"], ours["corners"]) == ("2026-10-03", 1.4, 7.0, 10)


def test_history_is_read_once_an_hour_and_each_matchs_price_once(tmp_path):
    from data_platform.services.pick_evidence_service import History

    platform = Platform(finished=[this_season("1", 3), this_season("2", 4)],
                        prices={"1": {"home": 1.4, "draw": 4.5, "away": 7.0}})
    history = History(football_data=football_data(tmp_path / "fd.sqlite3"), platform=platform)
    history.rows(NOW); history.rows(NOW + timedelta(minutes=59))
    assert platform.asked_finished == 1
    history.rows(NOW + timedelta(minutes=61))
    assert platform.asked_finished == 2
    assert platform.asked_prices == [["1", "2"], ["2"]]  # a priced match is not asked again


def test_attach_adds_the_story_the_groups_and_each_picks_evidence_and_changes_nothing_else(tmp_path):
    target = card()
    before = copy.deepcopy(target)
    service(tmp_path, Platform()).attach(target, CORNERS, now=NOW)
    added = target["recommendations"]
    assert added["story"]["lead"] == "Manchester City are clear favourites, with most goals expected from " \
                                     "Manchester City."
    assert added["story"]["corners_expected"] == 10.4
    assert [g["key"] for g in added["evidence"]["groups"]] == ["home", "away", "league"]
    assert added["most_likely"][0]["evidence"]["verdict"]
    stripped = copy.deepcopy(target)
    del stripped["recommendations"]["story"], stripped["recommendations"]["evidence"]
    for item in stripped["recommendations"]["most_likely"]:
        del item["evidence"]
    assert stripped == before


def test_a_european_cup_card_gets_the_story_only_and_no_history_is_read(tmp_path):
    platform = Platform()
    target = card(league="UCL")
    service(tmp_path, platform).attach(target, CORNERS, now=NOW)
    assert target["recommendations"]["story"] and "evidence" not in target["recommendations"]
    assert platform.asked_finished == 0


def test_a_failed_history_read_leaves_the_picks_as_they_were_and_is_tried_again(tmp_path):
    platform = Platform(fail=True)
    evidence = service(tmp_path, platform)
    target = card()
    evidence.attach(target, CORNERS, now=NOW)
    assert "evidence" not in target["recommendations"] and "evidence" not in target["recommendations"]["most_likely"][0]
    assert target["recommendations"]["story"]  # the story needs no history
    platform.fail = False
    evidence.attach(target, CORNERS, now=NOW + timedelta(minutes=1))
    assert target["recommendations"]["evidence"]


def test_a_card_without_recommendations_is_left_alone(tmp_path):
    target = {**card(), "recommendations": None}
    service(tmp_path, Platform()).attach(target, CORNERS, now=NOW)
    assert target["recommendations"] is None


def test_the_switch_is_off_unless_it_says_on(monkeypatch):
    from data_platform.services import pick_evidence_service

    monkeypatch.delenv("SPIX_PICK_EVIDENCE", raising=False)
    assert not pick_evidence_service.enabled()
    monkeypatch.setenv("SPIX_PICK_EVIDENCE", " On ")
    assert pick_evidence_service.enabled()
