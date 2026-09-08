"""Verification tests for the confirmed-lineup Match Read input path."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "Scripts" / "rag_ingest", ROOT / "Scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Scripts.rag_ingest import lineup_context  # noqa: E402


def _xi(team: str, prefix: str, count: int = 11) -> dict:
    return {
        "team": {"name": team},
        "startXI": [
            {"player": {"name": f"{prefix} Player {index}"}}
            for index in range(1, count + 1)
        ],
    }


def test_direct_fixture_id_uses_the_canonical_event_identity_and_requires_both_xis():
    rows = [_xi("Arsenal", "Home"), _xi("Chelsea", "Away")]
    with mock.patch.object(lineup_context, "_get_api_key", return_value="test-key"), mock.patch.object(
        lineup_context, "_find_fixture_id"
    ) as find_fixture, mock.patch.object(
        lineup_context, "_fetch_lineups_raw", return_value=rows
    ) as fetch_lineups:
        home, away, available = lineup_context.fetch_match_lineups(
            "Arsenal",
            "Chelsea",
            "EPL",
            "2026-09-06",
            fixture_id="12345",
        )

    find_fixture.assert_not_called()
    fetch_lineups.assert_called_once_with(12345, "test-key")
    assert available is True
    assert len(home) == len(away) == 11


def test_partial_provider_lineups_are_not_labelled_confirmed():
    rows = [_xi("Arsenal", "Home"), _xi("Chelsea", "Away", count=10)]
    with mock.patch.object(lineup_context, "_get_api_key", return_value="test-key"), mock.patch.object(
        lineup_context, "_fetch_lineups_raw", return_value=rows
    ):
        home, away, available = lineup_context.fetch_match_lineups(
            "Arsenal",
            "Chelsea",
            "EPL",
            "2026-09-06",
            fixture_id=12345,
        )

    assert (home, away, available) == ([], [], False)


def test_event_provider_passes_the_canonical_fixture_id_to_context_resolution():
    context = lineup_context.LineupContext(source="lineups", is_available=True)
    event = {
        "id": "12345",
        "_fixture_id": 12345,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
    }
    with mock.patch.object(lineup_context, "get_lineup_context", return_value=context) as resolver:
        result = lineup_context.get_confirmed_lineup_context_for_event(
            event,
            "EPL",
            date(2026, 9, 6),
        )

    assert result is context
    resolver.assert_called_once_with(
        "Arsenal",
        "Chelsea",
        "EPL",
        "2026-09-06",
        fixture_id=12345,
    )


def test_fixture_lookup_season_moves_with_the_campaign_calendar():
    assert lineup_context._api_football_season_for_fixture_date("2026-09-06") == 2026
    assert lineup_context._api_football_season_for_fixture_date("2027-05-30") == 2026
