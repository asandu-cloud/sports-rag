"""The board and the match page carry pick evidence only while ``SPIX_PICK_EVIDENCE`` is on (D82), and turning it on
adds keys without changing anything that was on the card."""
import copy

from Scripts.tests.data_platform.test_before_kickoff_api import started_read, switches
from Scripts.tests.data_platform.test_match_read_api import _client, _service

BOARD = "/api/match-reads/EPL/2026-09-06"  # _client's clock: 2026-09-06 12:30 UTC
MATCH = "/api/match-reads/fixtures/fixture-upcoming"


class Recording:
    """Stands in for the evidence service: adds a marker so the test can see where it ran."""

    def __init__(self):
        self.calls = []

    def attach(self, card, canonical_results, *, now):
        self.calls.append((card["fixture"]["event_id"], len(canonical_results or [])))
        card["recommendations"]["story"] = {"lead": "marker"}


def client_with(monkeypatch, session_factory, evidence, service_object):
    from web_app.routers import match_reads

    service = _service(session_factory)
    started_read(service, "fixture-upcoming", kickoff="2026-09-06T15:00:00Z", evaluated_at="2026-09-06T12:00:00Z")
    switches(monkeypatch)
    monkeypatch.setenv("SPIX_PICK_EVIDENCE", evidence)
    monkeypatch.setattr(match_reads, "_get_pick_evidence", lambda: service_object)
    return _client(monkeypatch, service)


def test_nothing_is_added_while_the_switch_is_off(settings, engine, session_factory, monkeypatch):
    recording = Recording()
    client = client_with(monkeypatch, session_factory, "", recording)
    [card] = client.get(BOARD).json()["cards"]
    assert "story" not in card["recommendations"] and recording.calls == []


def test_the_board_and_the_match_page_get_the_evidence_with_the_reads_canonical_results(
        settings, engine, session_factory, monkeypatch):
    recording = Recording()
    client = client_with(monkeypatch, session_factory, "on", recording)
    [card] = client.get(BOARD).json()["cards"]
    effective = client.get(MATCH).json()["effective"]
    assert card["recommendations"]["story"] == effective["recommendations"]["story"] == {"lead": "marker"}
    assert recording.calls[0][0] == "fixture-upcoming" and recording.calls[0][1] > 0


def test_switching_it_on_changes_nothing_that_was_on_the_card(settings, engine, session_factory, monkeypatch):
    from data_platform.services.pick_evidence_service import History, PickEvidence
    from Scripts.tests.data_platform.test_pick_evidence_service import Platform, football_data

    import tempfile
    from pathlib import Path

    folder = Path(tempfile.mkdtemp(prefix="pick-evidence-"))
    real = PickEvidence(History(football_data=football_data(folder / "fd.sqlite3"), platform=Platform()))
    off = client_with(monkeypatch, session_factory, "", real).get(BOARD).json()
    on = _client(monkeypatch, _service(session_factory))
    monkeypatch.setenv("SPIX_PICK_EVIDENCE", "on")
    on = on.get(BOARD).json()
    assert off["cards"] and on["cards"]
    stripped = copy.deepcopy(on)
    for card in stripped["cards"]:
        block = card["recommendations"]
        block.pop("story", None), block.pop("evidence", None)
        for item in block["most_likely"]:
            item.pop("evidence", None)
    assert stripped == off
