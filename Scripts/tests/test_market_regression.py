"""Replay deterministic canonical-market regression scenarios from JSON fixtures."""

from __future__ import annotations

import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Scripts" / "rag_ingest"))

from core import market_service  # noqa: E402
from core.market_regression import assert_regression_match  # noqa: E402


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "pipeline_v1"
_OVERRIDABLE_PROJECTIONS = {
    "projected_total_goals",
    "projected_total_corners",
    "projected_total_cards",
    "projected_total_sot",
    "projected_btts_prob",
    "projected_moneyline_probs",
    "projected_goal_difference",
}


def load_scenario(scenario_id: str) -> dict:
    """Load one sanitized replay recording by its manifest identifier."""
    return json.loads((FIXTURE_ROOT / scenario_id / "input.json").read_text(encoding="utf-8"))


def replay_scenario(scenario: dict):
    """Replay frozen normalized inputs through the real service and selectors.

    Synthetic recordings freeze model projection outputs because this repository
    does not yet contain archived odds/profile snapshots.  They deliberately
    test the canonical selection + delivery boundary.  Archive recordings can
    use the same expected-result format without these test overrides.
    """
    overrides = dict(scenario.get("projection_overrides") or {})
    unknown = set(overrides) - _OVERRIDABLE_PROJECTIONS
    if unknown:
        raise ValueError(f"Unsupported projection overrides: {sorted(unknown)}")

    with ExitStack() as stack:
        for function_name, result in overrides.items():
            stack.enter_context(mock.patch.object(
                market_service, function_name, return_value=tuple(result),
            ))
        return market_service.evaluate_event(
            scenario["event"],
            scenario["league"],
            markets=scenario["markets"],
            fixture_date=scenario["fixture_date"],
            input_snapshot_id=scenario["input_snapshot_id"],
            generated_at=scenario["generated_at"],
            model_version="recorded-synthetic-v1",
        )


class MarketRegressionTests(unittest.TestCase):
    def test_manifest_scenarios_replay_to_approved_decision_baselines(self):
        manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "pipeline-regression.v1")
        self.assertEqual(manifest["recording_policy"], "synthetic_selector_delivery")

        for scenario_id in manifest["scenarios"]:
            with self.subTest(scenario=scenario_id):
                scenario = load_scenario(scenario_id)
                self.assertGreaterEqual(scenario.get("baseline_revision", 0), 1)
                self.assertTrue(scenario.get("baseline_reason"))
                self.assertTrue(scenario.get("expected"), "Every scenario needs an explicit approved baseline.")

                results = replay_scenario(scenario)
                self.assertEqual(
                    [result.market.group for result in results], scenario["markets"],
                )
                for result in results:
                    expected = scenario["expected"].get(result.market.group)
                    self.assertIsNotNone(expected, f"No expected result for {result.market.group}")
                    assert_regression_match(result, expected)


if __name__ == "__main__":
    unittest.main()
