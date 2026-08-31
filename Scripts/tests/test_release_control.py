"""Tests for fail-safe canonical recommendation release modes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag_ingest"))

from core.market_result import Decision, DecisionStatus, PriceQuote
from core.release_control import apply_release_policy, resolve_release_policy


def _recommended() -> Decision:
    return Decision(
        status=DecisionStatus.RECOMMENDED,
        quote=PriceQuote(side="over", line=2.5, odds=2.0, bookmaker="Book"),
        model_probability=0.62,
        implied_probability=0.50,
        value_edge=0.12,
        expected_value=0.24,
        confidence="medium",
    )


class ReleaseControlTests(unittest.TestCase):
    def test_live_mode_preserves_an_eligible_recommendation(self):
        context = {}
        decision = apply_release_policy(_recommended(), context, policy=resolve_release_policy("live"))

        self.assertEqual(decision.status, DecisionStatus.RECOMMENDED)
        self.assertEqual(context["release_policy"]["mode"], "live")
        self.assertNotIn("release_candidate", context)

    def test_shadow_mode_withholds_but_keeps_the_exact_candidate(self):
        context = {}
        original = _recommended()
        decision = apply_release_policy(original, context, policy=resolve_release_policy("shadow"))

        self.assertEqual(decision.status, DecisionStatus.NO_BET)
        self.assertIn("Shadow mode", decision.reason)
        self.assertEqual(context["release_candidate"], original.to_dict())
        self.assertEqual(context["release_policy"]["mode"], "shadow")

    def test_invalid_explicit_configuration_fails_closed(self):
        context = {}
        policy = resolve_release_policy("publsih")
        decision = apply_release_policy(_recommended(), context, policy=policy)

        self.assertEqual(policy.mode, "disabled")
        self.assertFalse(policy.configuration_valid)
        self.assertEqual(decision.status, DecisionStatus.NO_BET)
        self.assertIn("invalid", decision.reason.lower())


if __name__ == "__main__":
    unittest.main()
