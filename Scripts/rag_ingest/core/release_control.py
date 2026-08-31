"""Fail-safe release modes for canonical pre-match recommendations.

The market service remains presentation-free, but it is the one place every
canonical Discord and website path passes through.  A release mode here makes
it possible to evaluate a revised selector without accidentally publishing its
picks.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterator, MutableMapping, Optional

try:
    from core.market_result import Decision, DecisionStatus
except ImportError:
    from .market_result import Decision, DecisionStatus  # type: ignore[no-redef]


RELEASE_POLICY_VERSION = "canonical-release-policy.v1"
RELEASE_MODE_ENV = "PREDICTION_RELEASE_MODE"
VALID_RELEASE_MODES = frozenset({"live", "shadow", "disabled"})


@dataclass(frozen=True)
class ReleasePolicy:
    """The effective publication policy for one canonical evaluation."""

    mode: str
    source: str
    configuration_valid: bool
    configuration_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": RELEASE_POLICY_VERSION,
            "mode": self.mode,
            "source": self.source,
            "configuration_valid": self.configuration_valid,
            "configuration_error": self.configuration_error,
        }


def resolve_release_policy(raw_mode: Optional[str] = None) -> ReleasePolicy:
    """Resolve the deployment mode, failing closed for an invalid setting.

    Existing installations keep their current public behaviour when the
    variable is absent (``live``).  Once an operator explicitly configures a
    mode, a typo must never silently turn into public recommendations.
    """
    if raw_mode is None:
        raw_mode = os.getenv(RELEASE_MODE_ENV)
    normalized = str(raw_mode or "").strip().lower()
    if not normalized:
        return ReleasePolicy(mode="live", source="default", configuration_valid=True)
    if normalized in VALID_RELEASE_MODES:
        return ReleasePolicy(mode=normalized, source="environment", configuration_valid=True)
    return ReleasePolicy(
        mode="disabled",
        source="invalid_environment",
        configuration_valid=False,
        configuration_error=(
            f"{RELEASE_MODE_ENV}={raw_mode!r} is invalid; expected one of "
            f"{', '.join(sorted(VALID_RELEASE_MODES))}."
        ),
    )


def apply_release_policy(
    decision: Decision,
    context: MutableMapping[str, Any],
    *,
    policy: Optional[ReleasePolicy] = None,
) -> Decision:
    """Attach release evidence and suppress publication when required.

    In shadow mode the unmodified candidate decision is retained in context so
    the shadow runner can persist and later settle it.  The decision returned
    to the website/Discord adapters is deliberately a no-bet, which means a
    shadow deployment cannot leak a candidate as a public pick.
    """
    policy = policy or resolve_release_policy()
    context["release_policy"] = policy.to_dict()

    if policy.mode == "live" or not decision.is_recommended:
        return decision

    context["release_candidate"] = decision.to_dict()
    if policy.mode == "shadow":
        reason = "Shadow mode: candidate captured for evaluation and withheld from public publication."
    else:
        reason = "Recommendation publication is disabled by the release policy."
        if policy.configuration_error:
            reason = f"{reason} {policy.configuration_error}"
    return replace(
        decision,
        # A string avoids enum identity drift when legacy direct-script
        # imports and package imports both exist in the same interpreter.
        status=DecisionStatus.NO_BET.value,
        confidence="low",
        reason=reason,
    )


@contextmanager
def temporary_release_mode(mode: str) -> Iterator[None]:
    """Temporarily set a mode for an offline/shadow capture process.

    This avoids requiring an operator to alter the shell environment merely to
    run the capture command. It only affects the current Python process.
    """
    previous = os.environ.get(RELEASE_MODE_ENV)
    os.environ[RELEASE_MODE_ENV] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(RELEASE_MODE_ENV, None)
        else:
            os.environ[RELEASE_MODE_ENV] = previous
