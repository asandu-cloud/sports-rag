"""Started matches keep the picks they showed at kickoff, without prices, and may carry a live status (D70)."""
import copy

from Scripts.tests.data_platform.test_match_read_api import _client, _priced_goals_result, _release, _save, _service

BOARD = "/api/match-reads/EPL/2026-09-06"  # _client's clock: 2026-09-06 12:30 UTC
KEYS = {"market", "side", "line", "label", "chance"}


def started_read(service, fixture_id="fixture-started", *, kickoff="2026-09-06T12:00:00Z",
                 evaluated_at="2026-09-06T11:50:00Z"):
    result = _priced_goals_result(fixture_id)
    result["fixture"]["kickoff"] = kickoff
    saved = _save(service, result, thesis="Read before kickoff.", evaluated_at=evaluated_at)
    _release(service, saved)
    return saved


def switches(monkeypatch, *, recs="on", before="on", live=""):
    for name, value in (("SPIX_RECOMMENDATIONS", recs), ("SPIX_BEFORE_KICKOFF", before), ("SPIX_LIVE_STATUS", live)):
        monkeypatch.setenv(name, value)


def test_the_board_is_unchanged_while_the_switch_is_off(settings, engine, session_factory, monkeypatch):
    service = _service(session_factory)
    started_read(service)
    switches(monkeypatch, before="")
    body = _client(monkeypatch, service).get(BOARD).json()
    assert "before_kickoff" not in body and body["cards"] == []


def test_a_started_match_keeps_the_picks_it_showed_at_kickoff_without_any_price(
    settings, engine, session_factory, monkeypatch,
):
    from data_platform.services.match_read_cards import build_match_read_card

    service = _service(session_factory)
    saved = started_read(service)
    switches(monkeypatch)
    body = _client(monkeypatch, service).get(BOARD).json()
    assert body["cards"] == []  # still withheld as a card: nothing after kickoff reads as a bet to place
    shown = build_match_read_card(saved)["recommendations"]
    [entry] = body["before_kickoff"]
    assert entry == {
        "version": "before-kickoff.v1",
        "fixture": {"event_id": "fixture-started", "league": "EPL", "home_team": "Home fixture-started",
                    "away_team": "Away fixture-started", "kickoff": "2026-09-06T12:00:00Z"},
        "visuals": {"home_team_logo": None, "away_team_logo": None},
        "read_at": "2026-09-06T11:50:00Z",
        "most_likely": [{key: item[key] for key in KEYS} for item in shown["most_likely"]],
        "value": [{key: item[key] for key in KEYS} for item in shown["value"]],
        "status": None,
    }
    assert entry["most_likely"] and entry["value"][0]["label"] == "Over 2.5 goals"
    assert all(set(item) == KEYS for item in entry["most_likely"] + entry["value"])


def test_only_started_matches_whose_read_was_current_at_kickoff_are_listed(
    settings, engine, session_factory, monkeypatch,
):
    service = _service(session_factory)
    started_read(service, "fixture-upcoming", kickoff="2026-09-06T15:00:00Z", evaluated_at="2026-09-06T12:00:00Z")
    started_read(service, "fixture-stale", evaluated_at="2026-09-06T08:00:00Z")  # four hours old at kickoff
    started_read(service, "fixture-late", evaluated_at="2026-09-06T12:10:00Z")  # made after kickoff
    started_read(service, "fixture-shown")
    switches(monkeypatch)
    client = _client(monkeypatch, service)
    body = client.get(BOARD).json()
    assert [card["fixture"]["event_id"] for card in body["cards"]] == ["fixture-upcoming"]
    assert [entry["fixture"]["event_id"] for entry in body["before_kickoff"]] == ["fixture-shown"]
    switches(monkeypatch, recs="")  # no recommendation layer, nothing to show
    assert client.get(BOARD).json()["before_kickoff"] == []


class Schedule:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def list_for_matchday(self, *, target_date, leagues):
        self.calls.append((target_date, tuple(leagues)))
        return [copy.deepcopy(row) for row in self.rows if row["fixture"]["league"] in leagues]  # fresh rows, as stored


def schedule_row(event_id, league, kickoff):
    return {"fixture": {"event_id": event_id, "league": league, "home_team": "H", "away_team": "A",
                        "kickoff": kickoff}, "status": "NS", "visuals": {"home_team_logo": None, "away_team_logo": None}}


def test_live_status_reaches_started_schedule_rows_and_before_kickoff_entries_from_one_request(
    settings, engine, session_factory, monkeypatch,
):
    from data_platform.services import live_status
    from web_app.routers import match_reads

    service = _service(session_factory)
    started_read(service)
    asked = []

    def provider(ids):
        asked.append(list(ids))
        return [{"fixture": {"id": event_id, "status": {"short": "2H", "elapsed": 61}}} for event_id in ids]

    statuses = live_status.LiveStatuses(provider)
    schedule = Schedule([schedule_row("fixture-started", "EPL", "2026-09-06T12:00:00Z"),
                         schedule_row("fixture-other", "LaLiga", "2026-09-06T11:30:00Z"),
                         schedule_row("fixture-upcoming", "EPL", "2026-09-06T19:00:00Z")])
    monkeypatch.setattr(match_reads, "_get_live_statuses", lambda: statuses)
    monkeypatch.setattr(match_reads, "_get_fixture_schedule_repository", lambda: schedule)
    switches(monkeypatch, live="on")
    client = _client(monkeypatch, service)

    entry = client.get(BOARD).json()["before_kickoff"][0]
    assert entry["status"] == {"short": "2H", "elapsed": 61}
    assert asked == [["fixture-started", "fixture-other"]]  # every started match of the day, in one request
    rows = {row["fixture"]["event_id"]: row for row in client.get("/api/match-reads/schedule/2026-09-06").json()["fixtures"]}
    assert rows["fixture-other"]["live_status"] == {"short": "2H", "elapsed": 61}
    assert rows["fixture-upcoming"]["live_status"] is None and rows["fixture-upcoming"]["status"] == "NS"
    assert len(asked) == 1  # the schedule used what the board had just asked for

    switches(monkeypatch, live="")
    assert "live_status" not in client.get("/api/match-reads/schedule/2026-09-06").json()["fixtures"][0]
    assert client.get(BOARD).json()["before_kickoff"][0]["status"] is None
