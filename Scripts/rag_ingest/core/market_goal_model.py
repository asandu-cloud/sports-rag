"""The market's own goal model (M1) for every goal market (rework A3.7 stage 4, D29).

Solved per read from the bookmakers' prices the engine already holds, with the targets
A3.2 validated (D19): the 1X2, over/under 2.5 and the main Asian handicap line (the one
most bookmakers price on both sides), each the median across books of that book's fair
probability, with selection's margin removal. The model is the independent Poisson of
``data_platform.market_goals`` (no Dixon–Coles correction, as validated).

Inputs are the option lists the selectors receive (``extract_moneyline_odds``,
``extract_total_line_options``, ``extract_spread_line_options``), which the decision audit
also stores, so research replays the exact engine computation.
"""
from __future__ import annotations

from collections import defaultdict
import statistics
from typing import Dict, List, Optional, Tuple

from .line_selection import _quote_identity, _three_way_fair, _two_way_fair, safe_float

VERSION = "market-goal-model.v1"
TOTAL_LINE = 2.5


def _goals():
    try:
        from data_platform import market_goals
    except ImportError:
        from Scripts.data_platform import market_goals  # type: ignore[import]
    return market_goals


def _median_fair(per_book: List[List[float]]) -> List[float]:
    medians = [statistics.median(p[i] for p in per_book) for i in range(len(per_book[0]))]
    total = sum(medians)
    return [m / total for m in medians]


def targets(h2h_options, total_options, spread_options, home_team: str, away_team: str) -> Dict:
    """``market_goals.solve`` targets: {"1x2": [...], "totals": {2.5: p_over}, "handicaps": {home_line: p_home}}."""
    out: Dict = {"totals": {}, "handicaps": {}}

    books: Dict[Tuple, Dict[str, float]] = defaultdict(dict)
    for o in h2h_options or []:
        side, odds = str(o.get("side") or "").lower().strip(), safe_float(o.get("odds"))
        if side in ("home", "draw", "away") and odds is not None and odds > 1.0:
            books[_quote_identity(o)][side] = max(odds, books[_quote_identity(o)].get(side, 0.0))
    per_book = [_three_way_fair([p["home"], p["draw"], p["away"]]) for p in books.values()
                if {"home", "draw", "away"} <= set(p)]
    per_book = [p for p in per_book if p is not None]
    if per_book:
        out["1x2"] = _median_fair(per_book)

    pairs: Dict[Tuple, Dict[str, float]] = defaultdict(dict)
    for o in total_options or []:
        side, odds, point = str(o.get("side") or "").lower().strip(), safe_float(o.get("odds")), safe_float(o.get("point"))
        if point == TOTAL_LINE and side in ("over", "under") and odds is not None and odds > 1.0:
            pairs[_quote_identity(o)][side] = odds
    per_book = [list(_two_way_fair(p["over"], p["under"])) for p in pairs.values() if {"over", "under"} <= set(p)]
    if per_book:
        out["totals"][TOTAL_LINE] = _median_fair(per_book)[0]

    home, away = str(home_team or "").lower().strip(), str(away_team or "").lower().strip()
    by_line: Dict[float, Dict[Tuple, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for o in spread_options or []:
        team, odds, point = str(o.get("team") or "").lower().strip(), safe_float(o.get("odds")), safe_float(o.get("point"))
        if odds is None or odds <= 1.0 or point is None or team not in (home, away) or home == away:
            continue
        home_line = point if team == home else -point  # the line from the home side's view
        by_line[home_line][_quote_identity(o)]["home" if team == home else "away"] = odds
    complete = {line: [list(_two_way_fair(p["home"], p["away"])) for p in books_.values() if {"home", "away"} <= set(p)]
                for line, books_ in by_line.items()}
    complete = {line: rows for line, rows in complete.items() if rows}
    if complete:
        main = max(complete, key=lambda line: (len(complete[line]), -abs(line)))
        out["handicaps"][main] = _median_fair(complete[main])[0]
    return out


def solve(h2h_options, total_options, spread_options, home_team: str, away_team: str) -> Optional[Dict]:
    """M1's goal means and the targets they reproduce, or None when the prices cannot identify both teams."""
    target = targets(h2h_options, total_options, spread_options, home_team, away_team)
    fit = _goals().solve(target)
    if fit is None:
        return None
    return {"version": VERSION, "home": fit["home"], "away": fit["away"], "error": fit["error"],
            "inputs": fit["inputs"], "targets": {"1x2": target.get("1x2"),
                                                  "totals": {str(k): v for k, v in target["totals"].items()},
                                                  "handicaps": {str(k): v for k, v in target["handicaps"].items()}}}


def score_probabilities(fit: Dict) -> Dict[Tuple[int, int], float]:
    matrix = _goals()._matrix(fit["home"], fit["away"])
    total = float(matrix.sum())
    return {(h, a): float(matrix[h, a]) / total for h in range(matrix.shape[0]) for a in range(matrix.shape[1])}


def total_probabilities(fit: Dict) -> Dict[int, float]:
    out: Dict[int, float] = defaultdict(float)
    for (h, a), p in score_probabilities(fit).items():
        out[h + a] += p
    return dict(out)


def btts_probability(fit: Dict) -> float:
    return sum(p for (h, a), p in score_probabilities(fit).items() if h > 0 and a > 0)


def figures(fit: Dict) -> Dict[str, float]:
    """What the surfaces publish under stage 4: M1's expected goals per team and in total, goal difference, BTTS yes."""
    return {"home_goals": fit["home"], "away_goals": fit["away"], "total_goals": fit["home"] + fit["away"],
            "goal_difference": fit["home"] - fit["away"], "btts_yes": btts_probability(fit)}
