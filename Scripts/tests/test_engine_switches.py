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
                                          "SPIX_COUNT_DISPERSION": "floors", "SPIX_DEVIG_METHOD": "proportional",
                                          "SPIX_MARKET_ANCHOR": "off", "SPIX_SELECTION_POLICY": "legacy",
                                          "SPIX_BOOKMAKER_SET": "all"}
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


# --- Stage 2: power margin removal in pick selection (A3.1, D17) ---------------------------------

from core import line_selection  # noqa: E402
from data_platform import devig  # noqa: E402
from prob_models import remove_vig_two_way  # noqa: E402


def test_selection_removes_margin_proportionally_until_switched(monkeypatch):
    monkeypatch.delenv("SPIX_DEVIG_METHOD", raising=False)
    assert line_selection._two_way_fair(1.45, 2.80) == remove_vig_two_way(1.45, 2.80)
    raw = [1 / 1.9, 1 / 3.6, 1 / 4.2]
    assert line_selection._three_way_fair([1.9, 3.6, 4.2]) == [p / sum(raw) for p in raw]


def test_switched_on_selection_uses_the_power_method(monkeypatch):
    monkeypatch.setenv("SPIX_DEVIG_METHOD", "power")
    assert line_selection._two_way_fair(1.45, 2.80) == tuple(devig.power([1.45, 2.80]))
    assert line_selection._three_way_fair([1.9, 3.6, 4.2]) == devig.power([1.9, 3.6, 4.2])
    # Power takes more margin from the longshot than proportional does.
    assert line_selection._two_way_fair(1.45, 2.80)[1] < remove_vig_two_way(1.45, 2.80)[1]


def test_switched_on_totals_selection_prices_against_the_power_fair(monkeypatch):
    options = [{"side": "over", "point": 2.5, "odds": 1.45, "bookmaker": "Book", "market_key": "totals"},
               {"side": "under", "point": 2.5, "odds": 2.80, "bookmaker": "Book", "market_key": "totals"}]
    implied = {}
    for method in ("proportional", "power"):
        monkeypatch.setenv("SPIX_DEVIG_METHOD", method)
        result = line_selection.select_best_total_recommendation([dict(o) for o in options], 3.1)
        implied[method] = {row["side"]: row["_implied_prob"] for row in result["all_lines"]}
    assert implied["proportional"]["under"] == remove_vig_two_way(1.45, 2.80)[1]
    assert implied["power"]["under"] == devig.power([1.45, 2.80])[1]
