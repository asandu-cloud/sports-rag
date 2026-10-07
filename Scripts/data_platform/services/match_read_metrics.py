"""Fixture-level model figures read from a Match Read's canonical results.

The briefing fact pack and the delivery card both describe the same numbers:
projected match totals, the team-goal split and the home/draw/away balance.
They are read here, once, from the persisted canonical market results so the
website never has to recover a figure from editorial prose.  Nothing in this
module runs or changes a model.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence


def match_metrics(canonical_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return the projected figures carried by a fixture's canonical results.

    Only figures that are present and finite are included; a missing market
    leaves its key absent rather than becoming zero.
    """
    markets = {
        _text(_mapping(result.get("market")).get("group")).lower(): result
        for result in canonical_results
        if isinstance(result, Mapping) and _text(_mapping(result.get("market")).get("group"))
    }
    metrics: Dict[str, Any] = {}

    for group, label in (("goals", "goals"), ("corners", "corners"), ("cards", "cards"), ("sot", "shots_on_target")):
        projection = _mapping(markets.get(group, {}).get("projection"))
        value = _finite(projection.get("value"))
        if value is None:
            continue
        metric: Dict[str, Any] = {"projected": _round(value)}
        season = _finite(projection.get("season_component"))
        recent = _finite(projection.get("recent_component"))
        if season is not None:
            metric["season_baseline"] = _round(season)
        if recent is not None:
            metric["recent_baseline"] = _round(recent)
        metrics[label] = metric

    for group in ("btts", "moneyline"):
        components = _mapping(_mapping(markets.get(group, {}).get("projection")).get("components"))
        if group == "btts":
            home_goals = _finite(components.get("home_goals"))
            away_goals = _finite(components.get("away_goals"))
            both_score = _finite(components.get("yes_probability"))
            if any(value is not None for value in (home_goals, away_goals, both_score)):
                metrics["team_goals"] = {
                    key: _round(value)
                    for key, value in (
                        ("home", home_goals),
                        ("away", away_goals),
                        ("both_teams_score_probability", both_score),
                    )
                    if value is not None
                }
        else:
            home = _finite(components.get("home_probability"))
            draw = _finite(components.get("draw_probability"))
            away = _finite(components.get("away_probability"))
            if any(value is not None for value in (home, draw, away)):
                metrics["result_probabilities"] = {
                    key: _round(value)
                    for key, value in (("home", home), ("draw", draw), ("away", away))
                    if value is not None
                }
    return metrics


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: float) -> float:
    return round(value, 4)
