"""Diff stored values vs freshly regenerated values.

Used by operators when investigating drift (e.g. after a feature
pipeline version bump). Compares numeric, string, and simple-dict
fields; ignores auto-generated timestamps and ids by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Tuple


_IGNORED = {"id", "created_at", "updated_at", "last_fetched_at", "last_api_updated_at",
            "built_at", "feature_build_id", "build_run_id", "payload_digest"}


@dataclass(frozen=True)
class ValueDiff:
    path: str
    expected: Any
    actual: Any

    def as_tuple(self) -> Tuple[str, Any, Any]:
        return (self.path, self.expected, self.actual)


def _normalize(value: Any) -> Any:
    if value is None:
        return None
    try:
        from decimal import Decimal
        if isinstance(value, Decimal):
            return float(value)
    except Exception:
        pass
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items() if k not in _IGNORED}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def compare_values(
    stored: Mapping[str, Any],
    regenerated: Mapping[str, Any],
    *,
    ignore: Iterable[str] = (),
    tolerate: float = 1e-5,
) -> List[ValueDiff]:
    """Return a list of per-field diffs.

    Numeric fields are compared with a small tolerance so floating-point
    drift doesn't produce spurious diffs.
    """
    ignore_set = set(_IGNORED) | set(ignore)
    diffs: List[ValueDiff] = []
    keys = (set(stored) | set(regenerated)) - ignore_set
    for key in sorted(keys):
        left = _normalize(stored.get(key))
        right = _normalize(regenerated.get(key))
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if abs(float(left) - float(right)) > tolerate:
                diffs.append(ValueDiff(path=key, expected=left, actual=right))
        elif left != right:
            diffs.append(ValueDiff(path=key, expected=left, actual=right))
    return diffs
