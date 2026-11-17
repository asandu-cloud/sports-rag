"""D124: the board's cards name the referee as the Research Area does, without slowing the board."""
from contextlib import contextmanager
from datetime import datetime, timezone

from data_platform.models import Competition, Fixture, Season, Team
from data_platform.services.research_upcoming import referee_refs
from Scripts.tests.data_platform.test_before_kickoff_api import BOARD, started_read, switches
from Scripts.tests.data_platform.test_match_read_api import _client, _service


def fixtures(session_factory, referees):
    """One canonical fixture per event id, each with its raw referee name (None: not named yet)."""
    with session_factory() as s:
        comp = Competition(code="EPL", name="Premier League", api_football_id=39)
        s.add(comp); s.flush()
        season = Season(competition_id=comp.id, year=2026, label="2026/27")
        home, away = Team(name="Alpha", api_football_id=1), Team(name="Beta", api_football_id=2)
        s.add_all([season, home, away]); s.flush()
        for event_id, raw in referees.items():
            s.add(Fixture(api_football_id=int(event_id), competition_id=comp.id, season_id=season.id,
                          home_team_id=home.id, away_team_id=away.id, status="NS", referee=raw,
                          kickoff_utc=datetime(2026, 9, 6, 15, tzinfo=timezone.utc)))
        s.commit()


def board(monkeypatch, service):
    return {c["fixture"]["event_id"]: c["fixture"] for c in _client(monkeypatch, service).get(BOARD).json()["cards"]}


def test_board_cards_name_the_referee_like_the_research_area(settings, engine, session_factory, monkeypatch):
    service = _service(session_factory)
    for event_id in ("4001", "4002", "fixture-unknown"):
        started_read(service, event_id, kickoff="2026-09-06T15:00:00Z", evaluated_at="2026-09-06T12:00:00Z")
    fixtures(session_factory, {"4001": "Anthony Taylor, England", "4002": None})
    switches(monkeypatch)
    cards = board(monkeypatch, service)
    raw = "Anthony Taylor, England"
    assert cards["4001"]["referee"] == referee_refs([raw], [])[raw]
    assert set(cards["4001"]["referee"]) == {"name", "referee_key"}
    assert cards["4002"]["referee"] is None  # known match, no referee named yet
    assert "referee" not in cards["fixture-unknown"]  # not in the database: not available

    from data_platform.services import refresh_coordination

    @contextmanager
    def busy():
        raise refresh_coordination.RefreshBusy("refresh")
        yield
    monkeypatch.setattr(refresh_coordination, "read_gate", busy)
    assert all("referee" not in fixture for fixture in board(monkeypatch, service).values())
