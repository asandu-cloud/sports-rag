"""V3 competition registry — load + validation + filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_platform.registry import load_registry
from data_platform.registry.loader import DEFAULT_REGISTRY_PATH, RegistryError


def test_default_registry_loads():
    registry = load_registry()
    codes = set(registry.codes())
    # All 8 competitions that existed in V1/V2 must still be present
    assert {"EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1", "UCL", "UEL", "UECL"} <= codes

    epl = registry.require("EPL")
    assert epl.competition_type == "domestic_league"
    provider = epl.provider("api_football")
    assert provider is not None
    assert provider.provider_id == 39


def test_filter_by_label():
    registry = load_registry()
    top5 = registry.filter(label="top5")
    codes = {e.code for e in top5}
    assert codes == {"EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1"}


def test_team_aliases_present_for_quirks():
    registry = load_registry()
    seriea = registry.require("SerieA")
    provider = seriea.provider("api_football")
    assert provider is not None
    assert provider.team_aliases.get("Verona") == "Hellas Verona"


def test_invalid_registry_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("competitions:\n  FAKE:\n    name: Fake\n")
    with pytest.raises(RegistryError):
        load_registry(bad)  # no providers block


def test_registry_feeds_sync_competitions():
    """The legacy ``COMPETITIONS`` dict in sync/apifootball derives from the registry now."""
    from data_platform.sync.apifootball import COMPETITIONS, CompetitionSpec
    assert isinstance(COMPETITIONS["EPL"], CompetitionSpec)
    assert COMPETITIONS["EPL"].api_football_id == 39
    assert COMPETITIONS["UCL"].competition_type == "continental_cup"
