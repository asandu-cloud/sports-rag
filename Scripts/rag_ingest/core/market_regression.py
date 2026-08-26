"""Stable comparison helpers for recorded canonical-market regressions.

Recorded scenarios intentionally compare decision-relevant fields rather than
rendered copy.  A model correction is allowed to change an approved baseline,
but the resulting semantic difference must be visible in review and covered by
the same replay tests.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

try:
    from core.market_result import MarketResult
except ImportError:
    from .market_result import MarketResult  # type: ignore[no-redef]


class RegressionMismatch(AssertionError):
    """A recorded expected market decision differs from the current result."""


def regression_view(result: MarketResult) -> dict[str, Any]:
    """Return the reproducible, decision-facing subset of a market result."""
    payload = result.to_dict()
    return {
        "fixture": payload["fixture"],
        "market": payload["market"],
        "projection": {
            "value": payload["projection"]["value"],
            "season_component": payload["projection"]["season_component"],
            "recent_component": payload["projection"]["recent_component"],
        },
        "decision": payload["decision"],
        "provenance": {
            "pipeline_version": payload["provenance"]["pipeline_version"],
            "input_snapshot_id": payload["provenance"]["input_snapshot_id"],
            "model_version": payload["provenance"]["model_version"],
            "as_of": payload["provenance"]["as_of"],
        },
        "evidence_codes": [item["code"] for item in payload["evidence"]],
    }


def assert_regression_match(
    result: MarketResult,
    expected: Mapping[str, Any],
    *,
    float_tolerance: float = 1e-9,
) -> None:
    """Assert that a recorded expected subset matches a canonical result.

    Expected fixtures are intentionally allowed to omit presentation-irrelevant
    fields.  Any field they do state must match, including ``None`` values.
    Floats use a small tolerance to avoid JSON/platform representation noise.
    """
    _assert_subset(regression_view(result), dict(expected), "result", float_tolerance)


def _assert_subset(actual: Any, expected: Any, path: str, tolerance: float) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise RegressionMismatch(f"{path}: expected an object, got {actual!r}")
        for key, expected_value in expected.items():
            if key not in actual:
                raise RegressionMismatch(f"{path}.{key}: expected key is missing")
            _assert_subset(actual[key], expected_value, f"{path}.{key}", tolerance)
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise RegressionMismatch(f"{path}: expected a list, got {actual!r}")
        if len(actual) != len(expected):
            raise RegressionMismatch(
                f"{path}: expected {len(expected)} items, got {len(actual)}"
            )
        for index, expected_value in enumerate(expected):
            _assert_subset(actual[index], expected_value, f"{path}[{index}]", tolerance)
        return

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise RegressionMismatch(f"{path}: expected numeric {expected!r}, got {actual!r}")
        if not (math.isfinite(float(actual)) and math.isfinite(float(expected))):
            raise RegressionMismatch(f"{path}: non-finite numeric comparison")
        if not math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=tolerance):
            raise RegressionMismatch(f"{path}: expected {expected!r}, got {actual!r}")
        return

    if actual != expected:
        raise RegressionMismatch(f"{path}: expected {expected!r}, got {actual!r}")
