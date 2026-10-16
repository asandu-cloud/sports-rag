"""A3.5: fixed count dispersions replace the variance floors for corners and SoT, behind a switch."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from core import prediction_guardrails as guard  # noqa: E402


def test_switched_off_the_floors_are_unchanged(monkeypatch):
    monkeypatch.delenv("SPIX_COUNT_DISPERSION", raising=False)
    variance, details = guard.resolve_total_variance("corners", 10.0, 3.0, 3.0)
    assert variance == 15.0 and details["source"] == "conservative_floor_over_profile_variance"


@pytest.mark.parametrize("market,alpha", [("corners", 0.025), ("sot", 0.01)])
def test_switched_on_corners_and_sot_use_the_qualified_dispersion(monkeypatch, market, alpha):
    monkeypatch.setenv("SPIX_COUNT_DISPERSION", "fixed")
    variance, details = guard.resolve_total_variance(market, 10.0, 3.0, 3.0)
    assert variance == pytest.approx(10.0 + alpha * 100)
    assert details["source"] == "fixed_dispersion" and details["dispersion"] == alpha
    assert details["version"] == "count-dispersion.v1" and details["profile_variance"] == 6.0


@pytest.mark.parametrize("market", ["cards", "goals"])
def test_switched_on_other_markets_keep_their_floors(monkeypatch, market):
    monkeypatch.setenv("SPIX_COUNT_DISPERSION", "fixed")
    monkeypatch.delenv("SPIX_COUNT_DISPERSION", raising=False)
    expected = guard.resolve_total_variance(market, 4.0, 1.0, 1.0)
    monkeypatch.setenv("SPIX_COUNT_DISPERSION", "fixed")
    assert guard.resolve_total_variance(market, 4.0, 1.0, 1.0) == expected


def test_fixed_dispersion_does_not_trigger_the_floor_confidence_cap(monkeypatch):
    monkeypatch.setenv("SPIX_COUNT_DISPERSION", "fixed")
    _, details = guard.resolve_total_variance("corners", 10.0, None, None)
    assert details["source"] not in {"conservative_fallback_floor", "conservative_floor_over_profile_variance"}
