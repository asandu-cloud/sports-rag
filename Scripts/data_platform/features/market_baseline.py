"""The market as a research baseline (rework A1 step 6; protocol amendment market-baseline.v1).

Research candidates are compared with the frozen control; this adds the betting
market, so a candidate must also show whether it beats the market's own
probability. Market probabilities come from the football-data.co.uk history
(``Index/football_data.sqlite3``, built by ``Scripts.ops.football_data_import``)
for fixtures linked to our fixture ids with agreeing scores (2019/20 onward), with the same method as
the live reference price: each bookmaker's margin removed proportionally, the
median across bookmakers.

* Individual bookmakers only. Aggregates ("Market average", "BetBrain average")
  stand in only when no single bookmaker priced the market; maxima never do.
* Pinnacle prices flagged ``suspect`` (2025/26 onward) are left out.
* ``timing="closing"`` uses closing prices only; ``"best_available"`` falls back
  to pre-closing prices (the only kind before 2019/20), and says so in ``basis``.

Events with a market price in the history: the 1X2 result (``winner``) and
goals over/under 2.5 (``over2.5``). Corners, shots and BTTS have no market price
there; handicap forecasts exist only at fixed lines, unlike the market's line.
"""
from collections import defaultdict
from contextlib import closing
import math
from pathlib import Path
import sqlite3

import numpy as np

from Scripts.data_platform.reference_price import devig_consensus

VERSION = "market-baseline.v1"
DEFAULT_DB = Path(__file__).resolve().parents[3] / "Index/football_data.sqlite3"
EVENTS = {"winner": ("1x2", ("home", "draw", "away")), "over2.5": ("ou25", ("over", "under"))}
AVERAGES = ("Market average", "BetBrain average")
EPS = 1e-15


def _is_aggregate(name):
    return "maximum" in name.lower() or "average" in name.lower()


def _fair(book_prices, selections):
    """Median fair probability vector across books priced on every selection."""
    probabilities = []
    for side in selections:
        _, consensus, _ = devig_consensus(book_prices, side, None, spreads=False)
        if consensus is None:
            return None, 0
        probabilities.append(consensus["fair_probability"])
    total = sum(probabilities)  # medians per side need not sum exactly to one
    return [p / total for p in probabilities], len(book_prices)


def _pick(rows, closing_flag, selections):
    """(fair vector, books, kind) from one timing, individual books first, then averages."""
    books = defaultdict(list)
    for bookmaker, flag, selection, price in rows:
        if flag == closing_flag:
            books[bookmaker].append((selection, None, price))
    individual = {b: o for b, o in books.items() if not _is_aggregate(b) and len(o) == len(selections)}
    if individual:
        p, n = _fair(individual, selections)
        if p:
            return p, n, "individual"
    for name in AVERAGES:
        if len(books.get(name, ())) == len(selections):
            p, n = _fair({name: books[name]}, selections)
            if p:
                return p, n, "average"
    return None, 0, None


def market_probabilities(db_path=DEFAULT_DB, fixture_ids=None, *, timing="closing"):
    """{fixture_api_id: {event: {"p": [...] or float, "books": n, "basis": ...}}}."""
    if timing not in {"closing", "best_available"}:
        raise ValueError(timing)
    query = """SELECT m.fixture_api_id, o.market, o.bookmaker, o.closing, o.selection, o.price
               FROM odds o JOIN matches m ON m.id = o.match_id
               WHERE m.link_status = 'matched' AND o.market IN ('1x2', 'ou25') AND o.suspect = 0"""
    grouped = defaultdict(lambda: defaultdict(list))
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        for fid, market, bookmaker, flag, selection, price in db.execute(query):
            grouped[fid][market].append((bookmaker, flag, selection, price))
    wanted = None if fixture_ids is None else {int(f) for f in fixture_ids}
    result = {}
    for fid, markets in grouped.items():
        if wanted is not None and fid not in wanted:
            continue
        for event, (market, selections) in EVENTS.items():
            rows = markets.get(market, [])
            for flag, label in ((1, "closing"), (0, "pre_closing")):
                if flag == 0 and timing == "closing":
                    break
                p, n, kind = _pick(rows, flag, selections)
                if p:
                    basis = label if kind == "individual" else f"{label}_average"
                    value = p if event == "winner" else p[0]
                    result.setdefault(fid, {})[event] = {"p": value, "books": n, "basis": basis}
                    break
    return result


def _outcome_probability(event, forecast_score, market_entry):
    """(forecast probability of the outcome, market probability of the same outcome)."""
    if event == "winner":
        winner = forecast_score["derived"]["winner"]
        return winner["p"][winner["y"]], market_entry["p"][winner["y"]]
    over = forecast_score["totals"]["2.5"]["binary"]
    return (over["p"], market_entry["p"]) if over["y"] else (1 - over["p"], 1 - market_entry["p"])


def compare_with_market(forecast_rows, market, event):
    """Log loss of a forecaster against the market on the same fixtures, with a paired week bootstrap.

    ``forecast_rows`` carry ``fixture_id``, ``week`` and ``scores`` (the research
    score dict). Negative ``mean_difference`` means the forecaster beat the market.
    """
    from Scripts.data_platform.features.phase4_qualification import paired_inference
    if event not in EVENTS:
        raise ValueError(event)
    paired, deltas, f_loss, m_loss = [], [], [], []
    for row in forecast_rows:
        entry = (market.get(int(row["fixture_id"])) or {}).get(event)
        if entry is None:
            continue
        pf, pm = _outcome_probability(event, row["scores"], entry)
        lf, lm = -math.log(max(pf, EPS)), -math.log(max(pm, EPS))
        paired.append(row)
        deltas.append(lf - lm)
        f_loss.append(lf)
        m_loss.append(lm)
    result = {"version": VERSION, "event": event, "forecasts": len(forecast_rows), "paired": len(paired),
              "weeks": len({r["week"] for r in paired})}
    if not paired:
        return {**result, "beats_market": False, "worse_than_market": False}
    inference = paired_inference(paired, np.array(deltas))
    low, high = inference["paired_week_delta95"]
    forecast_ll, market_ll = float(np.mean(f_loss)), float(np.mean(m_loss))
    return {**result, "forecast_logloss": forecast_ll, "market_logloss": market_ll,
            "relative_change": forecast_ll / market_ll - 1 if market_ll else None, **inference,
            "beats_market": bool(high < 0), "worse_than_market": bool(low > 0)}
