"""A5.1: is Spix's disagreement with the market information or error?

For every line the engine evaluated (latest pre-kickoff read per fixture and stage),
the market's fair probability is the median across bookmakers after power margin
removal (the A3.7 stage 2 method) of the prices that read saw. The closing
probability is the same consensus from API-Football's final pre-kickoff snapshot.

Each line is taken on the side Spix prefers (its probability above the market's),
which is what an edge-ranking selector considers; 1X2 keeps all three outcomes, with
Spix's probabilities from the result's projection (its candidates carry none). Two tests, both symmetric in the
side, so the choice of side does not bias them:

* outcome: logit P(win) = logit p_market + c * (logit p_spix - logit p_market).
  c = 0: Spix's deviation carries nothing; c = 1: trust Spix fully (D20's blend, live).
* closing: logit p_close - logit p_market = k * (logit p_spix - logit p_market).
  k > 0: the market moves toward Spix before kickoff.
"""
from __future__ import annotations

from contextlib import closing
import gzip
import json
from pathlib import Path
import sqlite3

import numpy as np

from .calibration_data import _label, latest_reads
from .reference_price import audit_books, devig_consensus, norm
from .scorecard import outcome_weight

MARKET_KEYS = {"goals": "totals", "corners": "totals_corners_over_under", "cards": "totals_cards_over_under",
               "btts": "btts", "spreads": "spreads", "moneyline": "h2h"}
TWO_WAY = {"goals": ("over", "under"), "corners": ("over", "under"), "cards": ("over", "under"),
           "btts": ("yes", "no")}
EPS = 1e-6
METHOD = "power"


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _other_side(market, side, point, home, away):
    """(side, point) of the complementary outcome on a two-way line, or None (1X2)."""
    if market in TWO_WAY:
        a, b = TWO_WAY[market]
        return (b if side == a else a, point) if side in (a, b) else None
    if market == "spreads" and point is not None and side in (home, away) and home != away:
        return (away if side == home else home, -float(point))
    return None


def _same_outcome(outcome, side, point):
    return outcome[0] == side and (outcome[1] == point or (outcome[1] is not None and point is not None
                                                          and abs(float(outcome[1]) - float(point)) < 1e-9))


def _best_price(books, market, side, point):
    """(best price, bookmaker offering it) for one side, or (None, None)."""
    prices = [(o[2], name) for name, outcomes in sorted(books.items()) for o in outcomes if _same_outcome(o, side, point)]
    return max(prices, key=lambda x: x[0]) if prices else (None, None)


def _consensus(books, market, side, point):
    if not books:
        return None
    _, consensus, _ = devig_consensus(books, side, point, spreads=market == "spreads", method=METHOD)
    return None if consensus is None else consensus


def read_lines(row, *, all_sides=False):
    """Every evaluated line in one read, on the side Spix prefers to the market (or every side)."""
    data = json.loads(row["read_json"]) if isinstance(row["read_json"], str) else row["read_json"]
    out = []
    for res in data.get("canonical_results", []):
        market = (res.get("market") or {}).get("group")
        key = MARKET_KEYS.get(market)
        audit = (res.get("context") or {}).get("decision_audit") or {}
        if key is None:
            continue
        fixture = res.get("fixture") or {}
        home, away = norm(fixture.get("home_team")), norm(fixture.get("away_team"))
        books = audit_books(audit.get("input_options"), key)
        sides = {}
        if market == "moneyline":  # 1X2 candidates carry no model probability; the projection holds all three
            components = (res.get("projection") or {}).get("components") or {}
            for outcome in ("home", "draw", "away"):
                if components.get(f"{outcome}_probability") is not None:
                    sides[(outcome, None)] = {"p": float(components[f"{outcome}_probability"]), "selected": False}
        for candidate in audit.get("candidates") or []:
            option = candidate.get("option") or {}
            raw = option.get("team") if market == "spreads" else option.get("side")
            if raw is None:
                continue
            point = option.get("point")
            ident = (norm(raw), None if point is None else float(point))
            if ident not in sides:
                if option.get("_model_prob") is None:
                    continue
                sides[ident] = {"p": float(option["_model_prob"]), "selected": False}
            sides[ident]["selected"] = sides[ident]["selected"] or bool(candidate.get("selector_selected"))
        done = set()
        for (side, point), entry in sides.items():
            if (side, point) in done:
                continue
            other = _other_side(market, side, point, home, away)
            done.update({(side, point), other} if other else {(side, point)})
            consensus = _consensus(books, market, side, point)
            if consensus is None:
                continue
            p_ref = consensus["fair_probability"]
            options = [(side, point, entry["p"], p_ref, entry["selected"])]
            if other is not None:
                partner = sides.get(other) or {}
                options.append((other[0], other[1], 1.0 - entry["p"], 1.0 - p_ref, bool(partner.get("selected"))))
            for s, pt, p_spix, p_mkt, selected in options:
                if all_sides or market == "moneyline" or p_spix >= p_mkt:
                    best_odds, best_book = _best_price(books, market, s, pt)
                    out.append({"fixture_id": str(row["fixture_api_id"]), "stage": row["stage"],
                                "kickoff": str(row["kickoff"]), "evaluated_at": str(row["evaluated_at"]),
                                "market": market, "market_key": key, "side": s, "point": pt,
                                "home": fixture.get("home_team"), "away": fixture.get("away_team"),
                                "p_spix": p_spix, "p_ref": p_mkt, "book_count": consensus["book_count"],
                                "best_odds": best_odds, "best_book": best_book, "selected": selected})
                    if not all_sides and market != "moneyline":
                        break
    return out


def _snapshot_books(payload, market_key, home, away):
    from .services.closing_prices import book_outcomes
    rows = (payload or {}).get("response") or []
    if not rows:
        return {}
    names = (home, away) if market_key == "spreads" else ("home", "away")
    return {b.get("name"): book_outcomes(b, market_key, *names) for b in rows[0].get("bookmakers", [])}


def closing_probability(payload, line):
    """Closing consensus fair probability of one line's side, or None."""
    return closing_prices(payload, line)[0]


def closing_prices(payload, line):
    """(closing consensus fair probability, closing price of the read's best-price bookmaker), each or None."""
    books = _snapshot_books(payload, line["market_key"], line["home"], line["away"])
    consensus = _consensus({k: v for k, v in books.items() if v}, line["market"], line["side"], line["point"])
    same_book = next((o[2] for o in books.get(line.get("best_book"), []) if _same_outcome(o, line["side"], line["point"])),
                     None)
    return (None if consensus is None else consensus["fair_probability"]), same_book


def closing_snapshots(db_path, backfill_dir=None):
    """fixture id -> API-Football final pre-kickoff /odds payload (database first, then saved files)."""
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)) as db:
        snaps = {str(e): (json.loads(p) if isinstance(p, str) else p) for e, p in db.execute(
            "SELECT event_id, odds_json FROM odds_snapshots WHERE kind = 'final_prematch' ORDER BY captured_at")}
    for path in sorted(Path(backfill_dir).glob("*.json.gz")) if backfill_dir else []:
        fid = path.name.split(".")[0]
        if fid not in snaps:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                snaps[fid] = json.load(handle)["body"]
    return snaps


def build(db_path, *, backfill_dir=None, all_sides=False):
    from .calibration_data import fixture_outcomes
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)) as db:
        reads = [dict(zip(("fixture_api_id", "stage", "evaluated_at", "kickoff", "read_json"), r)) for r in db.execute(
            "SELECT fixture_api_id, stage, evaluated_at, kickoff, read_json FROM match_reads")]
    outcomes = fixture_outcomes(db_path)
    snapshots = closing_snapshots(db_path, backfill_dir)
    dataset = []
    for row in latest_reads(reads):
        outcome = outcomes.get(str(row["fixture_api_id"]))
        if not outcome:
            continue
        for line in read_lines(row, all_sides=all_sides):
            label = _label({"market": line["market"], "side": line["side"], "point": line["point"],
                            "home": line["home"], "away": line["away"]}, outcome)
            if label is None:
                continue
            y, w = outcome_weight(label)
            if w <= 0:
                continue
            payload = snapshots.get(line["fixture_id"])
            p_close, best_book_close = closing_prices(payload, line) if payload else (None, None)
            dataset.append({**line, "y": y, "w": w, "p_close": p_close, "best_book_close_odds": best_book_close})
    return dataset


def fit_outcome_weight(rows, *, iterations=50):
    """Weighted maximum likelihood of c in logit P(win) = logit p_ref + c * (logit p_spix - logit p_ref)."""
    y = np.array([r["y"] for r in rows], dtype=float)
    w = np.array([r["w"] for r in rows], dtype=float)
    offset = _logit([r["p_ref"] for r in rows])
    x = _logit([r["p_spix"] for r in rows]) - offset
    c = 0.0
    for _ in range(iterations):
        q = 1 / (1 + np.exp(-(offset + c * x)))
        gradient, hessian = (w * (y - q) * x).sum(), (w * q * (1 - q) * x * x).sum()
        if hessian <= 1e-12:
            return None
        step = gradient / hessian
        c += step
        if abs(step) < 1e-10:
            break
    return float(c)


def fit_closing_move(rows):
    """Least squares through the origin of the closing move on Spix's deviation (k)."""
    priced = [r for r in rows if r.get("p_close") is not None]
    if not priced:
        return None
    ref = _logit([r["p_ref"] for r in priced])
    x = _logit([r["p_spix"] for r in priced]) - ref
    z = _logit([r["p_close"] for r in priced]) - ref
    w = np.array([r["w"] for r in priced], dtype=float)
    denominator = (w * x * x).sum()
    return None if denominator <= 0 else float((w * x * z).sum() / denominator)


def fixture_bootstrap(rows, statistic, *, resamples=1000, seed=20260928):
    """95% interval of ``statistic`` over resampled fixtures."""
    by_fixture = {}
    for r in rows:
        by_fixture.setdefault(r["fixture_id"], []).append(r)
    keys = sorted(by_fixture)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        sample = [r for k in rng.choice(keys, size=len(keys)) for r in by_fixture[k]]
        value = statistic(sample)
        if value is not None:
            values.append(value)
    return np.quantile(values, [.025, .975]).tolist() if values else [None, None]
