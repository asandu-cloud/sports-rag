"""Fair-price card numbers and Price Check (tracker D36): what a bet is worth, from the market's own prices.

For each main line of a fixture (1X2, goals over/under, BTTS, the main handicap, the main corners line) the card
gives the market's fair probability, the fair odds and the minimum odds worth taking (fair odds plus the 2% value
threshold, rounded up to the cent), with Spix's own probability beside it, labelled as Spix's. No bookmaker is named
(D37, D38).

The fair probability is the median across bookmakers of each one's price without its margin (power method), as
stage 3's market anchor computes it; the 1X2 is rescaled to sum to one. A line is on the card when at least
``MIN_BOOKS`` bookmakers price it on every side; the main line is the one most bookmakers price, then the most
balanced. The prices are the ones a read stored in its decision audit, so the card describes the read it ships with.

Price Check compares the odds a member sees with the fair price: a good price from the 2% threshold up, overpaying
from 2% below fair down (``FAIR_BAND``, owner to confirm), fair in between.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .confidence_labels import MIN_VALUE
from .reference_price import audit_books, consensus_spread, devig_consensus, norm

VERSION = "fair-prices.v1"
METHOD = "power"
MIN_BOOKS = 2
FAIR_BAND = 0.02  # within this much value either side of fair, a price is "fair" (owner to confirm, D36)
MARKETS = ("moneyline", "goals", "btts", "spreads", "corners")
MARKET_KEYS = {"moneyline": "h2h", "goals": "totals", "btts": "btts", "spreads": "spreads",
               "corners": "totals_corners_over_under"}
SIDES = {"moneyline": ("home", "draw", "away"), "goals": ("over", "under"), "btts": ("yes", "no"),
         "corners": ("over", "under")}


def minimum_odds(probability: float, *, min_value: float = MIN_VALUE) -> float:
    """The lowest odds, to the cent, at which the bet is worth at least ``min_value`` at this probability."""
    return math.ceil((1 + min_value) / probability * 100 - 1e-9) / 100


def price_check(probability: float, odds: float, *, min_value: float = MIN_VALUE, band: float = FAIR_BAND) -> dict:
    """"good_price", "fair" or "overpaying" for ``odds`` at the market's fair ``probability``."""
    value = probability * odds - 1
    verdict = "good_price" if value >= min_value else "overpaying" if value <= -band else "fair"
    return {"verdict": verdict, "value": value, "fair_odds": 1 / probability,
            "minimum_odds": minimum_odds(probability, min_value=min_value)}


def _results(canonical_results: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {str((r.get("market") or {}).get("group")): r for r in canonical_results or [] if isinstance(r, Mapping)}


def _books(result: Mapping[str, Any], market: str):
    options = ((result.get("context") or {}).get("decision_audit") or {}).get("input_options")
    return audit_books(options, MARKET_KEYS[market])


def _teams(result: Mapping[str, Any]):
    fixture = result.get("fixture") or {}
    return norm(fixture.get("home_team")), norm(fixture.get("away_team"))


def _sides(result: Mapping[str, Any], market: str, line):
    """[(side, line)] of every outcome of one line; a handicap's line is the home side's."""
    if market != "spreads":
        return [(side, line) for side in SIDES[market]]
    home, away = _teams(result)
    return [(home, line), (away, None if line is None else -line)]


def _consensus(books, market, side, line):
    _, consensus, _ = devig_consensus(books, side, line, spreads=market == "spreads", method=METHOD)
    return consensus


def _spix_probability(result: Mapping[str, Any], market: str, side: str, line) -> Optional[float]:
    """Spix's own probability for one outcome: the 1X2 projection, else the selector's candidates."""
    if market == "moneyline":
        spix = ((result.get("context") or {}).get("market_anchor") or {}).get("spix") or {}
        value = spix.get(side, ((result.get("projection") or {}).get("components") or {}).get(f"{side}_probability"))
        return float(value) if isinstance(value, (int, float)) else None
    for candidate in ((result.get("context") or {}).get("decision_audit") or {}).get("candidates") or []:
        opt = candidate.get("option") or {}
        raw = opt.get("team") if market == "spreads" else opt.get("side")
        point = opt.get("point")
        if norm(raw) == side and (point == line or (point is not None and line is not None
                                                    and abs(float(point) - float(line)) < 1e-9)):
            for key in ("_spix_prob", "_model_prob"):  # _spix_prob exists only where the market priced the line
                if isinstance(opt.get(key), (int, float)):
                    return float(opt[key])
    return None


def _entry(result: Mapping[str, Any], market: str, books, line) -> Optional[dict]:
    outcomes, counts, spread = [], [], None
    for side, side_line in _sides(result, market, line):
        consensus = _consensus(books, market, side, side_line)
        if consensus is None:
            return None
        counts.append(consensus["book_count"])
        spread = spread if spread is not None else consensus_spread(consensus)
        outcomes.append({"side": side, "line": side_line, "fair_probability": consensus["fair_probability"]})
    if min(counts) < MIN_BOOKS:
        return None
    if market == "moneyline":  # each book's three fair prices sum to one, their medians need not
        total = sum(o["fair_probability"] for o in outcomes)
        for o in outcomes:
            o["fair_probability"] /= total
    for o in outcomes:
        p = o["fair_probability"]
        o.update(fair_odds=1 / p, minimum_odds=minimum_odds(p),
                 spix_probability=_spix_probability(result, market, o["side"], o["line"]))
    return {"market": market, "line": line, "book_count": min(counts), "book_spread": spread, "outcomes": outcomes}


def _lines(result: Mapping[str, Any], market: str, books) -> List:
    if market in ("moneyline", "btts"):
        return [None]
    home, _ = _teams(result)
    if market == "spreads":  # the home side's handicaps
        return sorted({o[1] for outcomes in books.values() for o in outcomes if o[0] == home and o[1] is not None})
    return sorted({o[1] for outcomes in books.values() for o in outcomes if o[1] is not None})


def fair_card(canonical_results: Iterable[Mapping[str, Any]]) -> dict:
    """The fair-price card of one read: its main lines, each with every outcome's fair price and Spix's take."""
    results, lines = _results(canonical_results), []
    for market in MARKETS:
        result = results.get(market)
        if result is None:
            continue
        books = _books(result, market)
        entries = [e for line in _lines(result, market, books) if (e := _entry(result, market, books, line))]
        if entries:  # most books, then the most balanced
            lines.append(max(entries, key=lambda e: (e["book_count"],
                                                     -abs(e["outcomes"][0]["fair_probability"] - 0.5))))
    return {"version": VERSION, "method": METHOD, "min_books": MIN_BOOKS, "min_value": MIN_VALUE, "lines": lines}


def fair_outcome(canonical_results: Iterable[Mapping[str, Any]], market: str, side: str, line) -> Optional[dict]:
    """The fair price of any priced outcome of a read (Price Check), or None when no bookmaker prices it."""
    result = _results(canonical_results).get(market)
    if result is None or market not in MARKET_KEYS:
        return None
    side, line = norm(side), None if line is None else float(line)
    consensus = _consensus(_books(result, market), market, side, line)
    if consensus is None:
        return None
    probability = consensus["fair_probability"]
    if market == "moneyline":  # the same rescaling as the card
        others = [_consensus(_books(result, market), market, s, None) for s in SIDES["moneyline"]]
        if all(others):
            probability /= sum(c["fair_probability"] for c in others)
    return {"market": market, "side": side, "line": line, "fair_probability": probability,
            "book_count": consensus["book_count"], "spix_probability": _spix_probability(result, market, side, line)}
