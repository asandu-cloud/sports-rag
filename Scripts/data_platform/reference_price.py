"""Market reference price for one selection: each bookmaker's margin removed, median across books.

Shared by the publication reference (prices stored in the decision audit when a
pick is published) and the closing consensus (API-Football's final pre-kickoff
snapshot), so the two probabilities are always computed the same way. Rework
step A1.4 will replace the plain median with measured bookmaker weights; the
per-book prices are kept so the reference can be recomputed then.

Outcomes are ``(side, point, price)`` with sides normalised by ``norm``:
"over"/"under", "yes"/"no", "home"/"draw"/"away", or a team name for
handicaps (each team's own handicap, so the opposite side is the other team at
``-line``).
"""
import math
import statistics
from typing import Dict, List, Optional, Tuple

METHOD = "proportional_devig_median.v1"
# A3.1 (D17): power margin removal halves the 1X2 calibration error on closing history. It is an option
# until A1's gate is closed (D16); the default stays proportional so live results do not change.
METHOD_VERSIONS = {"proportional": METHOD, "power": "power_devig_median.v2",
                   "shin": "shin_devig_median.v2", "additive": "additive_devig_median.v2"}
METHOD_NAMES = {version: name for name, version in METHOD_VERSIONS.items()}
# The measurement yardstick: the publication reference, the closing consensus and the scorecard. Owner, 20 Oct:
# proportional through stages 1-3, so each is judged the same way; power from the Mon 2 Nov scorecard, with the
# history recomputed from the stored prices (``refair``).
MEASUREMENT_METHOD = "proportional"
Outcome = Tuple[str, Optional[float], float]


def norm(value) -> str:
    return str(value or "").strip().casefold().replace(":", "-")


def market_group(outcomes: List[Outcome], side: str, line, spreads: bool) -> List[Outcome]:
    """The pick's outcome plus the outcomes its margin is removed against."""
    if spreads:
        return [o for o in outcomes if (o[0] == side and o[1] == line)
                or (o[0] != side and line is not None and o[1] == -line)]
    return [o for o in outcomes if o[1] == line]


def devig_consensus(books: Dict[str, List[Outcome]], side: str, line, *, spreads: bool, wanted_book=None,
                    method="proportional"):
    """(same-book price or None, consensus or None, issue or None)."""
    from . import devig
    if method not in METHOD_VERSIONS:
        raise ValueError(f"unknown margin-removal method {method!r}")
    side, wanted = norm(side), norm(wanted_book)
    same, priced = None, []
    for name, outcomes in books.items():
        group = market_group(outcomes, side, line, spreads)
        picked = [o for o in group if o[0] == side]
        if len(picked) != 1 or len(group) < 2 or len({o[0] for o in group}) != len(group):
            continue  # missing, ambiguous or one-sided: cannot remove this book's margin
        inverse = sum(1 / o[2] for o in group)
        fair = devig.fair([o[2] for o in group], method)
        if fair is None:
            continue  # e.g. additive would make an outcome negative
        book_fair = fair[group.index(picked[0])]
        priced.append({"bookmaker": name, "odds": {o[0]: o[2] for o in group}, "margin": inverse - 1,
                       "fair": book_fair})
        if wanted and norm(name) == wanted:
            same = {"bookmaker": name, "odds": picked[0][2], "fair": book_fair}
    if not priced:
        return None, None, "selection_unavailable"
    consensus = {"method": METHOD_VERSIONS[method], "fair_probability": statistics.median(b["fair"] for b in priced),
                 "book_count": len(priced), "books": priced}
    return same, consensus, None if same or not wanted else "same_bookmaker_unavailable"


def refair(consensus, method) -> Optional[dict]:
    """A stored consensus with each book's margin removed by ``method`` instead, from the prices it kept.

    Each book's entry keeps its market group's prices and its fair probability for the pick under the stored
    method, which identifies the pick's outcome among them. None when the consensus kept no usable prices.
    """
    from . import devig
    if not consensus or method not in METHOD_VERSIONS:
        return None
    if consensus.get("method") == METHOD_VERSIONS[method]:
        return consensus
    stored = METHOD_NAMES.get(consensus.get("method"))
    books = []
    for book in consensus.get("books") or []:
        prices = list((book.get("odds") or {}).values())
        old = devig.fair(prices, stored) if stored and len(prices) >= 2 else None
        if not old or not isinstance(book.get("fair"), (int, float)):
            return None
        match = [i for i, p in enumerate(old) if p is not None and abs(p - book["fair"]) < 1e-9]
        if not match:
            return None
        new = devig.fair(prices, method)
        if new is not None:  # as ``devig_consensus``: a book this method cannot price is left out
            books.append({**book, "fair": new[match[0]]})
    if not books:
        return None
    again = {**consensus, "method": METHOD_VERSIONS[method], "book_count": len(books), "books": books,
             "fair_probability": statistics.median(b["fair"] for b in books)}
    same = consensus.get("same_book")
    if same:
        fair = next((b["fair"] for b in books if b.get("bookmaker") == same.get("bookmaker")), None)
        again["same_book"] = {**same, "fair": fair}
    return again


def consensus_spread(consensus) -> Optional[float]:
    """How far apart the bookmakers' fair probabilities are (M3): their interquartile range, or None under two books."""
    fairs = [b["fair"] for b in (consensus or {}).get("books") or []]
    if len(fairs) < 2:
        return None
    q1, _, q3 = statistics.quantiles(fairs, n=4, method="inclusive")
    return q3 - q1

def audit_books(options, market_key: str) -> Dict[str, List[Outcome]]:
    """Group decision-audit options for one market by bookmaker."""
    books: Dict[str, List[Outcome]] = {}
    for option in options or []:
        if option.get("market_key") != market_key or option.get("period", "regulation_time") != "regulation_time":
            continue
        price = option.get("odds")
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 1:
            continue
        side = norm(option.get("team") if market_key == "spreads" else option.get("side"))
        books.setdefault(str(option.get("bookmaker") or ""), []).append((side, option.get("point"), float(price)))
    return books


def publication_reference(result) -> dict:
    """The market's fair probability for a published pick, from the prices seen when it was published."""
    decision = result.get("decision") or {}
    quote = decision.get("quote") or {}
    market_key = quote.get("market_key") or (result.get("market") or {}).get("key")
    options = ((result.get("context") or {}).get("decision_audit") or {}).get("input_options") or []
    books = audit_books(options, market_key)
    method = METHOD_VERSIONS[MEASUREMENT_METHOD]
    if not books:
        return {"status": "unavailable", "reason": "no_publication_prices", "method": method}
    same, consensus, issue = devig_consensus(books, quote.get("side"), quote.get("line"),
                                             spreads=market_key == "spreads", wanted_book=quote.get("bookmaker"),
                                             method=MEASUREMENT_METHOD)
    if consensus is None:
        return {"status": "unavailable", "reason": issue, "method": method}
    return {"status": "available", "source": "decision_audit.input_options", **consensus,
            "same_book": same, **({"issue": issue} if issue else {})}
