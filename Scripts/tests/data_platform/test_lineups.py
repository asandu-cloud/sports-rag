"""D135: confirmed line-ups for the match page, asked of API-Football on demand, kept in memory, capped a day."""
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_platform.services import lineups
from data_platform.services.lineups import Lineups
from data_platform.services.research_match import likely_eleven
from Scripts.tests.data_platform.test_research_match_teams import SQUAD, history, service  # noqa: F401 (fixture)

KICKOFF = datetime(2026, 9, 30, 15, tzinfo=timezone.utc)  # Chelsea v Leeds (3200) in ``history``
CHELSEA, LEEDS = 500, 501  # API-Football team ids in ``history``; Chelsea's players are 700 + their SQUAD index


def side(team, first_player, *, bench=(("Sub", 960),)):
    xi = [{"player": {"id": first_player + i, "name": f"P{i}", "number": i + 1, "pos": "G" if i == 0 else "D",
                      "grid": f"{1 if i == 0 else 2}:{max(i, 1)}"}} for i in range(11)]
    return {"team": {"id": team, "name": str(team)}, "formation": "4-3-3", "coach": {"id": 1, "name": f"Coach {team}"},
            "startXI": xi, "substitutes": [{"player": {"id": pid, "name": n, "number": 30, "pos": "M", "grid": None}}
                                           for n, pid in bench]}


def reply(event_id, *, out=True):
    return {"fixture": {"id": int(event_id), "status": {"short": "NS"}},
            "teams": {"home": {"id": CHELSEA}, "away": {"id": LEEDS}},
            "lineups": [side(LEEDS, 950, bench=()), side(CHELSEA, 700)] if out else []}


class Provider:
    def __init__(self, out=()):
        self.out, self.calls = set(out), []

    def __call__(self, ids):
        self.calls.append(list(ids))
        return [reply(e, out=e in self.out) for e in ids]


def test_the_switch_is_off_unless_on(monkeypatch):
    monkeypatch.setenv("SPIX_LINEUPS", "")
    assert not lineups.enabled()
    monkeypatch.setenv("SPIX_LINEUPS", " On ")
    assert lineups.enabled()


def test_nothing_is_asked_before_the_hour_before_kickoff():
    provider = Provider()
    known = Lineups(provider).lookup("3200", KICKOFF, KICKOFF - timedelta(minutes=61))
    assert provider.calls == [] and known == {"status": "not_out_yet", "checked_at": None, "home": None, "away": None}


def test_asks_every_five_minutes_until_both_elevens_are_out_then_once_at_kickoff():
    provider = Provider()
    store = Lineups(provider)
    t = KICKOFF - timedelta(minutes=60)
    assert store.lookup("3200", KICKOFF, t)["status"] == "not_out_yet"
    store.lookup("3200", KICKOFF, t + timedelta(minutes=4))  # asked 4 minutes ago: not again
    assert len(provider.calls) == 1
    provider.out.add("3200")
    known = store.lookup("3200", KICKOFF, t + timedelta(minutes=5))
    assert len(provider.calls) == 2 and known["status"] == "confirmed"
    assert known["checked_at"] == (t + timedelta(minutes=5)).isoformat()
    assert known["home"]["formation"] == "4-3-3" and known["home"]["coach"] == f"Coach {CHELSEA}"
    assert known["home"]["start_xi"][0] == {"api_player_id": 700, "name": "P0", "number": 1, "position": "G",
                                            "grid": "1:1"}
    assert len(known["home"]["start_xi"]) == 11 and known["home"]["bench"][0]["api_player_id"] == 960
    assert known["away"]["start_xi"][0]["api_player_id"] == 950  # matched by team, not by order
    store.lookup("3200", KICKOFF, KICKOFF - timedelta(minutes=1))
    assert len(provider.calls) == 2  # out: no more asks before kickoff
    store.lookup("3200", KICKOFF, KICKOFF + timedelta(minutes=2))
    store.lookup("3200", KICKOFF, KICKOFF + timedelta(minutes=40))
    assert len(provider.calls) == 3  # the one check at kickoff for late changes, then never again


def test_matches_wanted_at_the_same_time_share_one_request_of_up_to_20():
    provider = Provider()
    store = Lineups(provider, batch=2)
    t = KICKOFF - timedelta(minutes=30)
    store.lookup("3200", KICKOFF, t)
    for event_id in ("3201", "3202"):
        store.lookup(event_id, KICKOFF, t + timedelta(minutes=1))  # 3200 not due yet
    store.lookup("3203", KICKOFF, t + timedelta(minutes=5))  # 3200 due again; 3201 and 3202 not yet
    store.lookup("3203", KICKOFF, t + timedelta(minutes=6))
    assert provider.calls == [["3200"], ["3201"], ["3202"], ["3200", "3203"], ["3201", "3202"]]


def test_the_daily_cap_stops_every_request_until_the_next_utc_day():
    provider = Provider()
    store = Lineups(provider, daily_cap=2)
    t = KICKOFF - timedelta(minutes=40)
    for minutes in (0, 5, 10, 15):
        store.lookup("3200", KICKOFF, t + timedelta(minutes=minutes))
    assert len(provider.calls) == 2
    late = datetime(2026, 10, 1, 0, 30, tzinfo=timezone.utc)
    store.lookup("3300", late + timedelta(minutes=30), late)
    assert len(provider.calls) == 3


def test_unavailable_once_a_check_after_kickoff_finds_none_or_the_window_closes():
    provider = Provider()
    store = Lineups(provider)
    assert store.lookup("3200", KICKOFF, KICKOFF - timedelta(minutes=10))["status"] == "not_out_yet"
    known = store.lookup("3200", KICKOFF, KICKOFF + timedelta(minutes=1))
    assert known["status"] == "unavailable" and len(provider.calls) == 2
    store.lookup("3200", KICKOFF, KICKOFF + timedelta(minutes=20))
    assert len(provider.calls) == 2  # every 30 minutes after kickoff
    store.lookup("3200", KICKOFF, KICKOFF + timedelta(minutes=31))
    assert len(provider.calls) == 3
    fresh = Lineups(Provider())
    assert fresh.lookup("3200", KICKOFF, KICKOFF + timedelta(minutes=151))["status"] == "unavailable"
    assert fresh._fetch.calls == []


def test_a_failed_request_is_not_an_answer_and_still_counts():
    def broken(ids):
        raise ConnectionError("https://example.invalid/?key=secret")
    store = Lineups(broken, daily_cap=1)
    known = store.lookup("3200", KICKOFF, KICKOFF + timedelta(minutes=1))
    assert known["status"] == "not_out_yet" and known["checked_at"] is None
    assert store._calls[KICKOFF.date()] == 1


def test_likely_eleven_is_the_keeper_then_ten_outfield_by_starts_in_the_last_five():
    rows = [{"player_id": i, "name": n, "position": pos, "minutes": 90 if i != 11 else 0, "substitute": False}
            for i, (n, pos) in enumerate(SQUAD + (("Second keeper", "G"), ("Unused", "M")))]
    rows += [{"player_id": 11, "name": "Second keeper", "position": "G", "minutes": 0, "substitute": True}]
    eleven = likely_eleven(rows * 3)
    assert eleven[0] == {"player_id": 0, "name": "Sanchez", "position": "G", "starts_last_5": 3}
    assert len(eleven) == 11 and all(p["position"] != "G" for p in eleven[1:])


def client(monkeypatch, session_factory, provider, *, now):
    from web_app.routers import match_reads, research
    monkeypatch.setattr(research, "_get_match_service", lambda: service(session_factory))
    monkeypatch.setattr(research, "_get_lineups", lambda: store)
    monkeypatch.setattr(match_reads, "_utc_now", lambda: now)
    store = Lineups(provider)
    app = FastAPI(); app.include_router(research.router)
    return TestClient(app)


def test_the_lineups_route_maps_players_to_research_profiles(history, monkeypatch):
    monkeypatch.setenv("SPIX_LINEUPS", "on")
    provider = Provider(out={"3200"})
    api = client(monkeypatch, history, provider, now=KICKOFF - timedelta(minutes=30))
    body = api.get("/api/research/matches/3200/lineups").json()
    assert body["schema_version"] == "research-lineups.v1" and body["event_id"] == "3200"
    block = body["lineups"]
    assert block["status"] == "confirmed" and set(block) == {"status", "checked_at", "home", "away", "likely"}
    assert block["home"]["start_xi"][0] == {"player_id": block["home"]["start_xi"][0]["player_id"], "name": "P0",
                                            "number": 1, "position": "G", "grid": "1:1"}
    assert block["home"]["start_xi"][0]["player_id"] is not None
    assert block["home"]["bench"][0]["player_id"] is None  # 960: no profile
    assert all(p["player_id"] is None for p in block["away"]["start_xi"])  # Leeds' players are unknown here
    likely = block["likely"]["home"]
    assert likely[0]["name"] == "Sanchez" and likely[0]["position"] == "G" and likely[0]["player_id"] is not None
    assert [p["name"] for p in likely[-2:]] == ["Nkunku", "Jackson"] and len(likely) == 10  # George: no start
    assert block["likely"]["away"] == []
    full = api.get("/api/research/matches/3200").json()
    assert full["lineups"] == block and len(provider.calls) == 1  # the full view reads the same memory
    assert api.get("/api/research/matches/999/lineups").status_code == 404


def test_with_the_switch_off_lineups_stay_null_and_nothing_is_asked(history, monkeypatch):
    monkeypatch.setenv("SPIX_LINEUPS", "")
    provider = Provider(out={"3200"})
    api = client(monkeypatch, history, provider, now=KICKOFF - timedelta(minutes=30))
    assert api.get("/api/research/matches/3200/lineups").json()["lineups"] is None
    assert api.get("/api/research/matches/3200").json()["lineups"] is None
    assert provider.calls == []


def test_not_out_yet_still_carries_the_likely_elevens(history, monkeypatch):
    monkeypatch.setenv("SPIX_LINEUPS", "on")
    api = client(monkeypatch, history, Provider(), now=KICKOFF - timedelta(hours=3))
    block = api.get("/api/research/matches/3200/lineups").json()["lineups"]
    assert block["status"] == "not_out_yet" and block["home"] is None and block["away"] is None
    assert block["likely"]["home"][0]["name"] == "Sanchez"
