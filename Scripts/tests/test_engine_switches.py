"""A3.7: every engine switch is read in one place and recorded with each result and pick."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from core import engine_switches, prediction_guardrails, team_resolution  # noqa: E402
from data_platform.publication_identity import tracking_identity  # noqa: E402


def test_defaults_are_todays_behaviour(monkeypatch):
    for name in engine_switches.SWITCHES:
        monkeypatch.delenv(name, raising=False)
    assert engine_switches.snapshot() == {"version": "engine-switches.v1", "DOCUMENT_STORE": "chroma",
                                          "SPIX_COUNT_DISPERSION": "floors"}
    assert prediction_guardrails.fixed_dispersion_enabled() is False
    assert team_resolution.document_store_handle() is None


def test_values_are_case_insensitive_and_unknown_values_fall_back(monkeypatch):
    monkeypatch.setenv("SPIX_COUNT_DISPERSION", " FIXED ")
    monkeypatch.setenv("DOCUMENT_STORE", "pgvector")
    assert prediction_guardrails.fixed_dispersion_enabled() is True
    assert team_resolution.document_store_handle() is None
    assert engine_switches.snapshot()["ignored"] == {"DOCUMENT_STORE": "pgvector"}


def test_the_store_switch_opens_the_document_store(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCUMENT_STORE", "sqlite")
    monkeypatch.setenv("DOCUMENT_STORE_PATH", str(tmp_path / "documents.sqlite3"))
    assert team_resolution.document_store_handle() is not None


def test_each_published_pick_carries_the_switches_of_its_read():
    switches = {"version": "engine-switches.v1", "DOCUMENT_STORE": "sqlite", "SPIX_COUNT_DISPERSION": "fixed"}
    result = {"fixture": {"event_id": "1", "kickoff": "2026-10-20T19:00:00Z"}, "market": {"key": "totals", "group": "goals"},
              "decision": {"quote": {"market_key": "totals", "side": "over", "line": 2.5, "bookmaker": "Book",
                                     "period": "regulation_time"}},
              "provenance": {"system_version": "v"}, "context": {"engine_switches": switches}}
    assert tracking_identity(result)["engine_switches"] == switches
