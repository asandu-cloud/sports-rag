"""The board and the match page carry pick notes only while ``SPIX_PICK_NOTES`` is on (D75)."""
from Scripts.tests.data_platform.test_before_kickoff_api import started_read, switches
from Scripts.tests.data_platform.test_match_read_api import _client, _service

BOARD = "/api/match-reads/EPL/2026-09-06"  # _client's clock: 2026-09-06 12:30 UTC


def items(card):
    block = card["recommendations"]
    return block["most_likely"] + block["value"]


def client_with(monkeypatch, session_factory, notes):
    from data_platform.services.pick_note_service import PickNotes
    from web_app.routers import match_reads

    service = _service(session_factory)
    started_read(service, "fixture-upcoming", kickoff="2026-09-06T15:00:00Z", evaluated_at="2026-09-06T12:00:00Z")
    switches(monkeypatch)
    monkeypatch.setenv("SPIX_PICK_NOTES", notes)
    monkeypatch.setattr(match_reads, "_get_pick_notes", lambda: PickNotes(lambda _: None))
    return _client(monkeypatch, service)


def test_the_board_is_unchanged_while_the_switch_is_off(settings, engine, session_factory, monkeypatch):
    [card] = client_with(monkeypatch, session_factory, "").get(BOARD).json()["cards"]
    assert items(card) and all("note" not in item for item in items(card))


def test_every_pick_on_the_board_and_the_match_page_has_its_two_lines(settings, engine, session_factory, monkeypatch):
    client = client_with(monkeypatch, session_factory, "on")
    [card] = client.get(BOARD).json()["cards"]
    effective = client.get("/api/match-reads/fixtures/fixture-upcoming").json()["effective"]
    for target in (card, effective):
        assert items(target) and all(set(item["note"]) == {"for", "against"} for item in items(target))
    assert items(card)[0]["note"]["against"].endswith("times in 10.")
