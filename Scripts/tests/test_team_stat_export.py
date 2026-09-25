"""Collector -> saved export: explicit zero must survive without inventing it."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from Scripts.pull_season import LEAGUES
from Scripts.team_stat_export import EVIDENCE_KEY, export_statistics, verified_export_values


def block(team_id, yellow, red):
    return {"team": {"id": team_id, "name": "One" if team_id == 1 else "Two"},
            "statistics": [{"type": "Yellow Cards", "value": yellow},
                           {"type": "Red Cards", "value": red},
                           {"type": "Corner Kicks", "value": None},
                           {"type": "Ball Possession", "value": "50%"}]}


@pytest.mark.parametrize("script", [c["team_stats"] for c in LEAGUES.values()])
def test_collectors_preserve_zero_null_and_evidence_through_saved_json(script, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [script, "--season", "2026"])
    namespace = runpy.run_path(script)["fetch_fixture_team_stats"].__globals__
    namespace["legacy_json"] = lambda *a, **kw: {"response": [block(1, 0, 0), block(2, None, None)]}
    namespace["OUTPUT_DIR"] = tmp_path
    context = {"fixture_id": 123, "fixture_name": "One vs Two", "home_team": "One", "away_team": "Two",
               "home_goals": 0, "away_goals": 1, "fixture_date_utc": "2026-09-01T19:00:00+00:00",
               "fixture_date": "2026-09-01", "final_score": "0-1", "final_score_string": "One 0 - Two 1"}
    rows = namespace["fetch_fixture_team_stats"](123, context)
    per_fixture, aggregate = namespace["aggregate_team_stats"](rows)
    namespace["save_team_outputs"](per_fixture, aggregate)
    saved = json.loads(next(tmp_path.glob("*team_fixture_stats_2026.json")).read_text())
    assert saved[0]["Red Cards"] == saved[0]["Yellow Cards"] == 0
    assert saved[1]["Red Cards"] is saved[1]["Yellow Cards"] is None
    assert all(r["Corner Kicks"] is None for r in saved)
    assert all(r["Ball Possession"] == 0.5 for r in saved)
    for row, team_id in zip(saved, (1, 2)):
        evidence = verified_export_values(row, fixture_id=123, team_id=team_id)
        assert evidence["Red Cards"] == row["Red Cards"]
        assert row[EVIDENCE_KEY]["values"]["Ball Possession"] == "50%"
    assert not any(c.startswith(EVIDENCE_KEY) for c in aggregate.columns)
    aggregates = json.loads(next(tmp_path.glob("*team_aggregate_stats_2026.json")).read_text())
    assert next(r for r in aggregates if r["team"] == "Two")["Red Cards_for"] is None


def test_omitted_fields_and_invalid_numbers_do_not_become_zero():
    sample = block(1, None, None)
    sample["statistics"] = [{"type": "Red Cards", "value": "not reported"}]
    result = export_statistics(sample, 123)
    assert result["Red Cards"] is None
    assert "Yellow Cards" not in result


def test_duplicate_statistics_are_rejected():
    sample = block(1, 2, 0)
    sample["statistics"].append({"type": "Red Cards", "value": None})
    with pytest.raises(ValueError, match="duplicate"):
        export_statistics(sample, 123)


@pytest.mark.parametrize("mutation", ["fixture", "team", "raw_null", "missing_raw", "timezone", "schema"])
def test_untrusted_export_marker_cannot_certify_a_zero(mutation):
    row = deepcopy(export_statistics(block(1, 2, 0), 123))
    evidence = row[EVIDENCE_KEY]
    if mutation == "fixture": evidence["fixture_id"] = 124
    if mutation == "team": evidence["team_id"] = 2
    if mutation == "raw_null": evidence["values"]["Red Cards"] = None
    if mutation == "missing_raw": evidence.pop("values")
    if mutation == "timezone": evidence["observed_at"] = "2026-09-25T12:00:00"
    if mutation == "schema": evidence["schema"] = "unverified"
    with pytest.raises(ValueError):
        verified_export_values(row, fixture_id=123, team_id=1)
