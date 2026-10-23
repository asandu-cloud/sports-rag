"""D29 gate for A3.7 stage 4: does the market's goal model (M1) beat Spix on the goal markets it would price?

For every audited read (the latest before kickoff, per fixture and stage), M1 is solved again
from the price lists the decision audit stored, with the engine's own
``core.market_goal_model.solve``, so the study replays exactly what stage 4 would publish.
Each evaluated line is taken once, on the side Spix prefers (``disagreement.read_lines``;
log loss is the same on either side), with three probabilities: Spix's, M1's and the
bookmakers' consensus (the median of each book's own fair price, power margin removal).

Scope (D29): goal lines other than 2.5 (2.5 keeps stage 3's direct consensus), the Asian
handicap and BTTS. The handicap line M1 is fitted to is flagged, because M1 reproduces it by
construction.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

from .calibration_data import _label, fixture_outcomes, latest_read_rows
from .disagreement import read_lines
from .reference_price import norm
from .scorecard import outcome_weight

MARKETS = ("goals", "spreads", "btts")
_RAG_INGEST = Path(__file__).resolve().parents[1] / "rag_ingest"


def _engine():
    """The engine's own modules, imported the way the Match Read worker loads them."""
    if str(_RAG_INGEST) not in sys.path:
        sys.path.insert(0, str(_RAG_INGEST))
    from core import line_selection, market_goal_model
    import prob_models
    return line_selection, market_goal_model, prob_models
ANCHORED_TOTAL = 2.5
EPS = 1e-6


def _audit(data, market):
    for res in data.get("canonical_results", []):
        if (res.get("market") or {}).get("group") == market:
            return ((res.get("context") or {}).get("decision_audit") or {}), res.get("fixture") or {}
    return {}, {}


def read_fit(data):
    """M1 solved from one read's stored option lists, or None."""
    _, market_goal_model, _ = _engine()
    h2h, fixture = _audit(data, "moneyline")
    totals, fixture_goals = _audit(data, "goals")
    spreads, fixture_spreads = _audit(data, "spreads")
    fixture = fixture or fixture_goals or fixture_spreads
    if not fixture:
        return None
    return market_goal_model.solve(h2h.get("input_options"), totals.get("input_options"), spreads.get("input_options"),
                                   fixture.get("home_team"), fixture.get("away_team"))


def m1_probability(fit, line):
    """M1's no-push equivalent probability for one line's side, as stage 4 would price it."""
    line_selection, market_goal_model, prob_models = _engine()
    market, side, point = line["market"], line["side"], line["point"]
    if market == "goals":
        profile = prob_models.asian_total_profile_from_counts(market_goal_model.total_probabilities(fit), float(point), side)
        return prob_models.asian_total_equivalent_probability(profile)
    if market == "btts":
        yes = market_goal_model.btts_probability(fit)
        return yes if side == "yes" else 1.0 - yes
    if market == "spreads":
        home, away = line["home"], line["away"]
        profile = line_selection._score_matrix_spread_profile(
            home, away, "", side == norm(home), float(point), score_probs=market_goal_model.score_probabilities(fit))
        return prob_models.asian_total_equivalent_probability(profile)
    raise ValueError(market)


def _fitted_handicap(fit, line):
    """True for the handicap line (either side) M1 was fitted to."""
    if line["market"] != "spreads":
        return False
    home_line = float(line["point"]) if line["side"] == norm(line["home"]) else -float(line["point"])
    return any(abs(float(k) - home_line) < 1e-9 for k in (fit.get("targets") or {}).get("handicaps") or {})


def build(db_path):
    outcomes = fixture_outcomes(db_path)
    rows, unsolved = [], 0
    for read in latest_read_rows(db_path):
        outcome = outcomes.get(str(read["fixture_api_id"]))
        if not outcome:
            continue
        data = json.loads(read["read_json"]) if isinstance(read["read_json"], str) else read["read_json"]
        lines = [l for l in read_lines(read) if l["market"] in MARKETS
                 and not (l["market"] == "goals" and l["point"] is not None and abs(float(l["point"]) - ANCHORED_TOTAL) < 1e-9)]
        if not lines:
            continue
        fit = read_fit(data)
        if fit is None:
            unsolved += 1
            continue
        for line in lines:
            label = _label({"market": line["market"], "side": line["side"], "point": line["point"],
                            "home": line["home"], "away": line["away"]}, outcome)
            if label is None:
                continue
            y, w = outcome_weight(label)
            if w <= 0:
                continue
            rows.append({**line, "y": y, "w": w, "p_m1": m1_probability(fit, line),
                         "fitted_line": _fitted_handicap(fit, line), "fit_error": fit["error"]})
    return rows, unsolved


def log_loss(y, p):
    p = min(max(p, EPS), 1 - EPS)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))
