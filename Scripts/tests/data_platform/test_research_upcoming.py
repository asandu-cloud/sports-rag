"""A team's and a referee's next matches for the Research Area (D110, the overhaul's B1)."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_platform.models import Competition, Fixture, ResearchRefereeAlias, Season, Team
from data_platform.repositories.research import ResearchRepository
from data_platform.services.refresh_coordination import RefreshBusy
from data_platform.services.research_profiles import referee_aliases
from data_platform.services.research_upcoming import ResearchUpcomingService, read_summary

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
TAYLOR = "Anthony Taylor, England"


def card(*, preliminary=False):
    return {"timing": {"is_preliminary": preliminary}, "created_at": "2026-11-20T09:00:00+00:00"}


def results(goals=2.66, corners=10.14, cards=None):
    out = [{"market": {"group": "goals"}, "projection": {"value": goals}},
           {"market": {"group": "corners"}, "projection": {"value": corners}}]
    if cards is not None:
        out.append({"market": {"group": "cards"}, "projection": {"value": cards}})
    return out


@pytest.fixture()
def schedule(session_factory):
    with session_factory() as s:
        epl = Competition(code="EPL", name="Premier League", api_football_id=39)
        ucl = Competition(code="UCL", name="UEFA Champions League", api_football_id=2,
                          competition_type="continental_cup")
        s.add_all([epl, ucl]); s.flush()
        epl_season, ucl_season = Season(competition_id=epl.id, year=2026, label="2026/27"), \
            Season(competition_id=ucl.id, year=2026, label="2026/27")
        teams = {n: Team(name=n, api_football_id=100 + i, short_code=n[:3].upper(),
                         logo_url=f"https://media.example/{n}.png")
                 for i, n in enumerate(("Chelsea", "Leeds", "Napoli", "Newcastle", "Arsenal"))}
        s.add_all([epl_season, ucl_season, *teams.values()]); s.flush()

        def fixture(api_id, comp, season, home, away, kickoff, status="NS", referee=None):
            s.add(Fixture(api_football_id=api_id, competition_id=comp.id, season_id=season.id,
                          home_team_id=teams[home].id, away_team_id=teams[away].id,
                          kickoff_utc=kickoff, status=status, referee=referee))

        fixture(1, epl, epl_season, "Chelsea", "Arsenal", datetime(2026, 11, 8, 15, tzinfo=timezone.utc),
                status="FT", referee=TAYLOR)
        fixture(2, epl, epl_season, "Arsenal", "Chelsea", datetime(2026, 11, 20, 15, tzinfo=timezone.utc),
                status="PST")
        fixture(3, epl, epl_season, "Chelsea", "Leeds", datetime(2026, 11, 21, 15, tzinfo=timezone.utc),
                referee=TAYLOR)
        fixture(4, ucl, ucl_season, "Napoli", "Chelsea", datetime(2026, 11, 25, 20, tzinfo=timezone.utc))
        fixture(5, epl, epl_season, "Newcastle", "Chelsea", datetime(2026, 11, 29, 15, tzinfo=timezone.utc))
        fixture(6, epl, epl_season, "Chelsea", "Napoli", datetime(2026, 12, 3, 20, tzinfo=timezone.utc))
        fixture(7, epl, epl_season, "Leeds", "Newcastle", datetime(2026, 11, 28, 15, tzinfo=timezone.utc),
                referee="A. Taylor")  # a spelling the alias table has not seen yet
        key = referee_aliases([TAYLOR])[0]["referee_key"]
        s.add(ResearchRefereeAlias(raw_name=TAYLOR, name="Anthony Taylor", referee_key=key))
        ids = {name: team.id for name, team in teams.items()}
    return ids, key


def service(session_factory, reads=None):
    reads = reads or {}
    return ResearchUpcomingService(ResearchRepository(session_factory), load_read=lambda event_id: reads.get(event_id),
                                   now=lambda: NOW, outlook_hours=96)


def test_team_upcoming_lists_every_competition_in_kickoff_order(session_factory, schedule):
    ids, key = schedule
    out = service(session_factory).team(ids["Chelsea"], limit=3)
    assert out["schema_version"] == "research-upcoming.v1"
    assert out["subject"] == {"type": "team", "id": ids["Chelsea"], "name": "Chelsea"}
    assert [m["event_id"] for m in out["matches"]] == ["3", "4", "5"]  # finished and postponed left out, limit 3
    first, second = out["matches"][:2]
    assert first["kickoff"] == "2026-11-21T15:00:00+00:00"
    assert first["competition"] == {"code": "EPL", "name": "Premier League"}
    assert first["home"] == {"id": ids["Chelsea"], "name": "Chelsea", "short_code": "CHE",
                             "logo_url": "https://media.example/Chelsea.png"}
    assert first["venue"] == "home" and second["venue"] == "away"
    assert second["competition"]["code"] == "UCL"
    assert first["referee"] == {"name": "Anthony Taylor", "referee_key": key}
    assert second["referee"] is None
    assert isinstance(first["fixture_id"], int)


def test_referee_upcoming_uses_the_alias_key_and_null_until_named(session_factory, schedule):
    ids, key = schedule
    out = service(session_factory).referee(key, limit=2)
    assert out["subject"] == {"type": "referee", "referee_key": key, "name": "Anthony Taylor"}
    assert [m["event_id"] for m in out["matches"]] == ["3", "7"]  # "A. Taylor" counts under the same key
    assert {m["referee"]["referee_key"] for m in out["matches"]} == {key}
    assert {m["referee"]["name"] for m in out["matches"]} == {"Anthony Taylor"}
    assert all(m["venue"] is None for m in out["matches"])
    assert service(session_factory).referee(key, limit=1)["matches"][0]["event_id"] == "3"


def test_unknown_subjects_raise_lookup_errors(session_factory, schedule):
    with pytest.raises(LookupError):
        service(session_factory).team(99999, limit=3)
    with pytest.raises(LookupError):
        service(session_factory).referee("no-such-ref-00000000", limit=1)


def test_read_status_ready_preliminary_waiting_and_opens_at():
    kickoff = datetime(2026, 11, 21, 15, tzinfo=timezone.utc)
    ready = read_summary("3", kickoff, load_read=lambda _: (card(), results(cards=4.62)), now=NOW, outlook_hours=96)
    assert ready == {"status": "ready", "read_at": "2026-11-20T09:00:00+00:00", "opens_at": None,
                     "expected": {"goals": 2.7, "corners": 10.1, "cards": 4.6}}
    early = read_summary("3", kickoff, load_read=lambda _: (card(preliminary=True), results()), now=NOW,
                         outlook_hours=96)
    assert early["status"] == "preliminary"
    assert early["expected"] == {"goals": 2.7, "corners": 10.1, "cards": None}
    waiting = read_summary("3", kickoff, load_read=lambda _: None, now=NOW, outlook_hours=96)
    assert waiting == {"status": "waiting", "read_at": None, "opens_at": "2026-11-17T15:00:00+00:00",
                       "expected": None}
    due = read_summary("3", kickoff, load_read=lambda _: None, now=kickoff - timedelta(hours=10), outlook_hours=96)
    assert due["status"] == "waiting" and due["opens_at"] is None


def test_upcoming_reads_carry_into_each_match(session_factory, schedule):
    ids, _ = schedule
    out = service(session_factory, reads={"3": (card(), results())}).team(ids["Chelsea"], limit=2)
    assert out["matches"][0]["read"]["status"] == "ready"
    assert out["matches"][1]["read"]["status"] == "waiting"


def test_upcoming_routes_404_422_and_503_while_the_gate_is_shut(session_factory, schedule, monkeypatch):
    from contextlib import contextmanager
    from web_app.routers import research as router_module
    ids, key = schedule
    monkeypatch.setattr(router_module, "_get_upcoming_service", lambda: service(session_factory))
    app = FastAPI(); app.include_router(router_module.router)
    client = TestClient(app)
    base = "/api/research"
    team = client.get(f"{base}/teams/{ids['Chelsea']}/upcoming")
    assert team.status_code == 200 and len(team.json()["matches"]) == 3
    assert len(client.get(f"{base}/referees/{key}/upcoming").json()["matches"]) == 1
    assert client.get(f"{base}/teams/99999/upcoming").status_code == 404
    assert client.get(f"{base}/referees/no-such-ref/upcoming").status_code == 404
    for route in (f"teams/{ids['Chelsea']}/upcoming?limit=0", f"teams/{ids['Chelsea']}/upcoming?limit=11",
                  "teams/0/upcoming", f"referees/{key}/upcoming?limit=0"):
        assert client.get(f"{base}/{route}").status_code == 422

    @contextmanager
    def shut():
        raise RefreshBusy("refresh running")
        yield

    monkeypatch.setattr(router_module, "_gate", shut)
    busy = client.get(f"{base}/teams/{ids['Chelsea']}/upcoming")
    assert busy.status_code == 503
    assert busy.json()["detail"] == "Research is updating; try again shortly."
