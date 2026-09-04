"""Pure season-to-date aggregation for team feature snapshots.

The legacy pipeline writes one ``team_engineered_features`` row per team per
completed fixture.  It does not always write the rolled-up
``team_profiles`` file that the original platform importer expected.  This
module turns those fixture rows into the same family of season-to-date fields
without importing the RAG normalizer, a vector store, or any web dependency.

The aggregation deliberately follows the normalizer's established semantics:
event totals are divided by matches played, shape/rate fields are averaged,
and conceded/opponent-induced fields come from the opponent's row in the same
fixture.  Keeping the logic here dependency-free makes it safe to use during a
database-only feature build.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


def _number(value: Any) -> Optional[float]:
    """Return a finite float, treating missing/non-numeric values as absent."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value.endswith("%"):
            value = value[:-1]
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_number(row: Mapping[str, Any], keys: Iterable[str], default: Optional[float] = None) -> Optional[float]:
    for key in keys:
        value = _number(row.get(key))
        if value is not None:
            return value
    return default


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _sum(rows: Iterable[Mapping[str, Any]], keys: Iterable[str]) -> float:
    return sum(_first_number(row, keys, 0.0) or 0.0 for row in rows)


def _key(value: Any) -> str:
    return str(value or "").strip().casefold()


def _fixture_key(row: Mapping[str, Any]) -> Optional[str]:
    """Prefer the stable API fixture id; legacy files can fall back to text."""
    fixture_id = row.get("fixture_id")
    if fixture_id not in (None, ""):
        return f"id:{str(fixture_id).strip()}"
    fixture = _key(row.get("fixture"))
    return f"name:{fixture}" if fixture else None


def _is_home(row: Mapping[str, Any]) -> bool:
    """Match the normalizer's explicit-home, fixture-text fallback behaviour."""
    team = _key(row.get("team"))
    home_team = _key(row.get("home_team"))
    if team and home_team:
        return team == home_team

    fixture = str(row.get("fixture") or "")
    if " vs " in fixture.casefold():
        left, _right = fixture.split(" vs ", 1)
        return team == _key(left)
    # This is also how the existing normalizer treats unparseable rows: not
    # explicitly home means they fall into the away bucket.
    return False


def _team_identity(rows: Iterable[Mapping[str, Any]]) -> Optional[Any]:
    for row in rows:
        for key in ("team_id", "api_football_id"):
            value = row.get(key)
            if value not in (None, ""):
                return value
    return None


def _opponent_lookup(rows: Iterable[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    """Return a counterpart row for each source row that has one.

    Engineered output normally has two rows for every fixture.  Prefer the
    explicit home/away identity when it is available, then fall back to the
    only other team in that fixture.  This supports older files that have a
    fixture string but no ``fixture_id``.
    """
    by_fixture: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    source_rows = list(rows)
    for row in source_rows:
        key = _fixture_key(row)
        if key is not None:
            by_fixture[key].append(row)

    opponents: Dict[int, Mapping[str, Any]] = {}
    for row in source_rows:
        key = _fixture_key(row)
        team = _key(row.get("team"))
        if not key or not team:
            continue
        candidates = [candidate for candidate in by_fixture[key] if _key(candidate.get("team")) != team]
        if not candidates:
            continue

        home = _key(row.get("home_team"))
        away = _key(row.get("away_team"))
        expected = away if team == home else home if team == away else ""
        opponent = next((candidate for candidate in candidates if _key(candidate.get("team")) == expected), None)
        opponents[id(row)] = opponent or candidates[0]
    return opponents


def _split_mean(
    rows: Iterable[Mapping[str, Any]],
    getter: Callable[[Mapping[str, Any]], Optional[float]],
) -> Dict[str, Optional[float]]:
    home: List[Optional[float]] = []
    away: List[Optional[float]] = []
    for row in rows:
        (home if _is_home(row) else away).append(getter(row))
    return {"home": _mean(home), "away": _mean(away)}


def _sample_variance(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if value is not None]
    if len(valid) < 5:
        return None
    average = sum(valid) / len(valid)
    return sum((value - average) ** 2 for value in valid) / (len(valid) - 1)


def _archetype(
    *,
    control_index: Optional[float],
    possession: Optional[float],
    dominance_index: Optional[float],
    aggression_index_norm: Optional[float],
) -> str:
    """Use the normalizer's deterministic style bucket, not a fresh cluster."""
    if control_index is not None and control_index >= 0.55:
        return (
            "possession-aggressive"
            if (aggression_index_norm or 0.0) >= 0.25
            else "possession-disciplined"
        )
    if (possession is not None and possession <= 0.45) or (
        dominance_index is not None and dominance_index < 0.45
    ):
        return "low-block-counter"
    return "transition-pace"


def derive_team_snapshot_rows(team_rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Roll fixture-level engineered rows into one snapshot source row per team.

    Each returned row has a ``team``/optional ``team_id`` identity plus only
    fields understood by ``TeamFeatureSnapshot``.  Rows without a team name
    are ignored.  The function is intentionally deterministic and pure so it
    can be unit-tested or reused without a DB, API client, Chroma, or browser.
    """
    source_rows = [dict(row) for row in team_rows if _key(row.get("team"))]
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[_key(row.get("team"))].append(row)

    opponents = _opponent_lookup(source_rows)
    derived: List[Dict[str, Any]] = []
    for team_key in sorted(grouped):
        rows = grouped[team_key]
        matches_played = len(rows)
        if not matches_played:
            continue

        goals = _sum(rows, ("goals",))
        assists = _sum(rows, ("assists",))
        corners = _sum(rows, ("corners",))
        fouls = _sum(rows, ("fouls_committed",))
        yellows = _sum(rows, ("yellow_cards",))
        reds = _sum(rows, ("red_cards",))
        shots_for = _sum(rows, ("shots_total_x", "shots_total"))
        sot_for = _sum(rows, ("shots_on_x", "shots_on"))

        cards_by_row = [
            _first_number(row, ("cards_total",), None)
            for row in rows
        ]
        cards_total = sum(
            card if card is not None else (_first_number(row, ("yellow_cards",), 0.0) or 0.0)
            + (_first_number(row, ("red_cards",), 0.0) or 0.0)
            for row, card in zip(rows, cards_by_row)
        )

        expected_goals = _mean(_first_number(row, ("expected_goals",)) for row in rows)
        possession = _mean(_first_number(row, ("possession",)) for row in rows)
        dominance_index = _mean(_first_number(row, ("dominance_index",)) for row in rows)
        control_index = _mean(_first_number(row, ("control_index",)) for row in rows)
        aggression_index_norm = _mean(_first_number(row, ("aggression_index_norm",)) for row in rows)
        form_index_team = _mean(_first_number(row, ("form_index_team",)) for row in rows)
        fouls_per_90_team = _mean(_first_number(row, ("fouls_per_90_team",)) for row in rows)
        cards_per_90_team = _mean(_first_number(row, ("cards_per_90_team",)) for row in rows)
        cards_per_foul_team = _mean(_first_number(row, ("cards_per_foul_team",)) for row in rows)

        passes_total = _sum(rows, ("passes_total",))
        accurate_passes = _sum(rows, ("accurate_passes",))
        pass_accuracy_values: List[Optional[float]] = []
        for row in rows:
            pass_accuracy = _first_number(row, ("pass_accuracy_team",))
            if pass_accuracy is None:
                row_passes = _first_number(row, ("passes_total",), 0.0) or 0.0
                row_accurate = _first_number(row, ("accurate_passes",), 0.0) or 0.0
                pass_accuracy = 100.0 * row_accurate / row_passes if row_passes > 0 else None
            elif pass_accuracy <= 1.0:
                pass_accuracy *= 100.0
            pass_accuracy_values.append(pass_accuracy)
        pass_accuracy_pct = (
            100.0 * accurate_passes / passes_total
            if passes_total > 0
            else _mean(pass_accuracy_values)
        )

        own_goals_split = _split_mean(rows, lambda row: _first_number(row, ("goals",)))
        own_corners_split = _split_mean(rows, lambda row: _first_number(row, ("corners",)))
        own_cards_split = _split_mean(
            rows,
            lambda row: _first_number(
                row,
                ("cards_total",),
                (_first_number(row, ("yellow_cards",), 0.0) or 0.0)
                + (_first_number(row, ("red_cards",), 0.0) or 0.0),
            ),
        )
        own_xg_split = _split_mean(rows, lambda row: _first_number(row, ("expected_goals",)))
        own_sot_split = _split_mean(rows, lambda row: _first_number(row, ("shots_on_x", "shots_on")))
        own_fouls_split = _split_mean(rows, lambda row: _first_number(row, ("fouls_committed",)))

        opponent_goals: List[Optional[float]] = []
        opponent_corners: List[Optional[float]] = []
        opponent_sot: List[Optional[float]] = []
        opponent_cards: List[Optional[float]] = []
        opponent_corners_home: List[Optional[float]] = []
        opponent_corners_away: List[Optional[float]] = []
        opponent_sot_home: List[Optional[float]] = []
        opponent_sot_away: List[Optional[float]] = []
        opponent_cards_home: List[Optional[float]] = []
        opponent_cards_away: List[Optional[float]] = []
        for row in rows:
            opponent = opponents.get(id(row))
            if opponent is None:
                continue
            opp_goals = _first_number(opponent, ("goals",))
            opp_corners = _first_number(opponent, ("corners",))
            opp_sot = _first_number(opponent, ("sot_for", "shots_on_x", "shots_on"))
            opp_cards = _first_number(
                opponent,
                ("cards_total",),
                (_first_number(opponent, ("yellow_cards",), 0.0) or 0.0)
                + (_first_number(opponent, ("red_cards",), 0.0) or 0.0),
            )
            opponent_goals.append(opp_goals)
            opponent_corners.append(opp_corners)
            opponent_sot.append(opp_sot)
            opponent_cards.append(opp_cards)
            if _is_home(row):
                opponent_corners_home.append(opp_corners)
                opponent_sot_home.append(opp_sot)
                opponent_cards_home.append(opp_cards)
            else:
                opponent_corners_away.append(opp_corners)
                opponent_sot_away.append(opp_sot)
                opponent_cards_away.append(opp_cards)

        derived.append({
            "team": str(rows[0].get("team")).strip(),
            "team_id": _team_identity(rows),
            "matches_played": matches_played,
            "goals_for_pm": goals / matches_played,
            "goals_against_pm": _mean(opponent_goals),
            "assists_pm": assists / matches_played,
            "corners_pm": corners / matches_played,
            "corners_against_pm": _mean(opponent_corners),
            "sot_for_pm": sot_for / matches_played,
            "sot_against_pm": _mean(opponent_sot),
            "shots_for_pm": shots_for / matches_played,
            "cards_pm": cards_total / matches_played,
            "yellows_pm": yellows / matches_played,
            "reds_pm": reds / matches_played,
            "cards_per_90_team": cards_per_90_team if cards_per_90_team is not None else cards_total / matches_played,
            "fouls_per_90_team": fouls_per_90_team if fouls_per_90_team is not None else fouls / matches_played,
            "cards_per_foul_team": cards_per_foul_team,
            "aggression_index_norm": aggression_index_norm,
            "form_index_team": form_index_team,
            "control_index": control_index,
            "dominance_index": dominance_index,
            "possession": possession,
            "expected_goals": expected_goals,
            "pass_accuracy_pct": pass_accuracy_pct,
            "opp_cards_induced_pm": _mean(opponent_cards),
            "goals_home_pm": own_goals_split["home"],
            "goals_away_pm": own_goals_split["away"],
            "corners_home_pm": own_corners_split["home"],
            "corners_away_pm": own_corners_split["away"],
            "cards_home_pm": own_cards_split["home"],
            "cards_away_pm": own_cards_split["away"],
            "corners_against_home_pm": _mean(opponent_corners_home),
            "corners_against_away_pm": _mean(opponent_corners_away),
            "sot_against_home_pm": _mean(opponent_sot_home),
            "sot_against_away_pm": _mean(opponent_sot_away),
            "cards_induced_home_pm": _mean(opponent_cards_home),
            "cards_induced_away_pm": _mean(opponent_cards_away),
            "xg_home_pm": own_xg_split["home"],
            "xg_away_pm": own_xg_split["away"],
            "sot_home_pm": own_sot_split["home"],
            "sot_away_pm": own_sot_split["away"],
            "fouls_home_pm": own_fouls_split["home"],
            "fouls_away_pm": own_fouls_split["away"],
            "corners_var": _sample_variance(_first_number(row, ("corners",)) for row in rows),
            "goals_var": _sample_variance(_first_number(row, ("goals",)) for row in rows),
            "cards_var": _sample_variance(
                _first_number(
                    row,
                    ("cards_total",),
                    (_first_number(row, ("yellow_cards",), 0.0) or 0.0)
                    + (_first_number(row, ("red_cards",), 0.0) or 0.0),
                )
                for row in rows
            ),
            "sot_var": _sample_variance(_first_number(row, ("shots_on", "shots_on_x")) for row in rows),
            "archetype": _archetype(
                control_index=control_index,
                possession=possession,
                dominance_index=dominance_index,
                aggression_index_norm=aggression_index_norm,
            ),
        })

    return derived
