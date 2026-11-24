"""D131: the board's team names open their Research Area pages, without slowing the board."""
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_platform.models import Competition, Fixture, Season, Team
from data_platform.repositories.research import ResearchRepository
from data_platform.services.research_profiles import ResearchProfileService
from Scripts.tests.data_platform.test_before_kickoff_api import BOARD, started_read, switches
from Scripts.tests.data_platform.test_match_read_api import _client, _service

KEYS = {"home_team_id", "away_team_id"}


def fixtures(session_factory, event_ids):
    """Alpha (one finished match, so a Research Area page) v Beta (none yet) in every match of ``event_ids``."""
    with session_factory() as s:
        comp = Competition(code="EPL", name="Premier League", api_football_id=39, competition_type="domestic_league")
        s.add(comp); s.flush()
        season = Season(competition_id=comp.id, year=2026, label="2026/27")
        alpha, beta, gamma = (Team(name=n, api_football_id=i) for i, n in enumerate(("Alpha", "Beta", "Gamma"), 1))
        s.add_all([season, alpha, beta, gamma]); s.flush()
        s.add(Fixture(api_football_id=3999, competition_id=comp.id, season_id=season.id, home_team_id=alpha.id,
                      away_team_id=gamma.id, status="FT", round="Regular Season - 1", home_goals=1, away_goals=0,
                      kickoff_utc=datetime(2026, 8, 30, 15, tzinfo=timezone.utc)))
        for event_id in event_ids:
            s.add(Fixture(api_football_id=int(event_id), competition_id=comp.id, season_id=season.id,
                          home_team_id=alpha.id, away_team_id=beta.id, status="NS", round="Regular Season - 3",
                          kickoff_utc=datetime(2026, 9, 6, 15, tzinfo=timezone.utc)))
        s.commit()
        return alpha.id, beta.id


def busy(monkeypatch):
    from data_platform.services import refresh_coordination

    @contextmanager
    def gate():
        raise refresh_coordination.RefreshBusy("refresh")
        yield
    monkeypatch.setattr(refresh_coordination, "read_gate", gate)


def test_board_cards_and_before_kickoff_entries_carry_the_research_team_ids(settings, engine, session_factory,
                                                                             monkeypatch):
    service = _service(session_factory)
    for event_id in ("4001", "fixture-unknown"):
        started_read(service, event_id, kickoff="2026-09-06T15:00:00Z", evaluated_at="2026-09-06T12:00:00Z")
    started_read(service, "4002")  # kicked off at 12:00, read at 11:50: a before-kickoff entry
    alpha, beta = fixtures(session_factory, ("4001", "4002"))
    switches(monkeypatch)
    body = _client(monkeypatch, service).get(BOARD).json()
    cards = {c["fixture"]["event_id"]: c["fixture"] for c in body["cards"]}
    assert {k: cards["4001"][k] for k in KEYS} == {"home_team_id": alpha, "away_team_id": None}  # Beta: no page yet
    assert not KEYS & set(cards["fixture-unknown"])  # not in the database: left out, never guessed
    [entry] = body["before_kickoff"]
    assert {k: entry["fixture"][k] for k in KEYS} == {"home_team_id": alpha, "away_team_id": None}

    research = FastAPI()
    from web_app.routers import research as research_router
    monkeypatch.setattr(research_router, "_get_research_service",
                        lambda: ResearchProfileService(ResearchRepository(session_factory)))
    research.include_router(research_router.router)
    client = TestClient(research)
    assert client.get(f"/api/research/teams/{cards['4001']['home_team_id']}").status_code == 200
    assert client.get(f"/api/research/teams/{beta}").status_code == 404  # why Beta's id is null

    busy(monkeypatch)
    body = _client(monkeypatch, service).get(BOARD).json()
    assert all(not KEYS & set(c["fixture"]) for c in body["cards"] + body["before_kickoff"])


def test_schedule_rows_carry_the_research_team_ids(settings, engine, session_factory, monkeypatch):
    from data_platform.repositories.fixture_schedule import FixtureScheduleRepository
    from web_app.routers import match_reads
    alpha, _ = fixtures(session_factory, ("4001",))
    monkeypatch.setattr(match_reads, "_utc_now", lambda: datetime(2026, 9, 5, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(match_reads, "_get_fixture_schedule_repository",
                        lambda: FixtureScheduleRepository(session_factory))
    monkeypatch.setenv("SPIX_LIVE_STATUS", "")
    app = FastAPI(); app.include_router(match_reads.router)
    client = TestClient(app)
    [row] = client.get("/api/match-reads/schedule/2026-09-06").json()["fixtures"]
    assert {k: row["fixture"][k] for k in KEYS} == {"home_team_id": alpha, "away_team_id": None}
    busy(monkeypatch)
    [row] = client.get("/api/match-reads/schedule/2026-09-06").json()["fixtures"]
    assert not KEYS & set(row["fixture"])
