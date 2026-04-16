"""Feature pipeline versions.

Keep it simple: one semver-ish string per pipeline. Bump when a change
would invalidate pre-existing snapshots (new source, different formula,
changed blend weights).
"""

from __future__ import annotations

from typing import Mapping, Optional

CURRENT_FEATURE_VERSION = "2026.04-v1"

_PIPELINE_OVERRIDES: Mapping[str, str] = {
    # Example: "Bundesliga": "2026.04-v1b"
}


def feature_version_for(pipeline: str) -> str:
    """Return the version string associated with a pipeline name."""
    return _PIPELINE_OVERRIDES.get(pipeline, CURRENT_FEATURE_VERSION)
