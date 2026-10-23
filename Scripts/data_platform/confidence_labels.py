"""A5.3 confidence labels (tracker D44): how sure Spix is that a published pick is a good bet, high or low.

A pick is **high** when the market is firm and, in a market where Spix has earned a vote, Spix agrees:

* at least ``min_books`` bookmakers price the picked line two-sided, and their fair probabilities sit close
  together (interquartile range at most ``max_spread``: bookmaker disagreement, M3);
* the quoting bookmaker's price was seen at most ``fresh_minutes`` before publication (price confirmation's
  record, from stage 4); an unknown age is not fresh;
* the market's own probability gives at least ``MIN_VALUE`` at the published price;
* in a market listed in ``SPIX_VOTE_MARKETS``, Spix's own probability does too. A market joins that list only by an
  owner-approved tracker decision, after it passes the Spix check (``ops/confidence_study.py``).

Anything else is **low**: still a pick, just less sure. The label never changes a decision; only
``decision.confidence`` and ``context["confidence_evidence"]`` are written, and only for results evaluated under
``SPIX_CONFIDENCE_LABELS=evidence`` (``edge`` keeps ``line_selection.confidence_from_edge``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping, Optional

from .reference_price import audit_books, consensus_spread, devig_consensus, norm
from .settlement import utc_datetime

VERSION = "confidence-labels.v1"
MIN_VALUE = 0.02  # the market-value selection's threshold, line_selection.MARKET_VALUE_MIN_EV


@dataclass(frozen=True)
class Cutoffs:
    max_spread: Optional[float]
    min_books: int = 4
    fresh_minutes: float = 30.0


CUTOFFS = Cutoffs(max_spread=None)  # no spread cut-off has passed the firmness check yet, so every label is low
SPIX_VOTE_MARKETS: frozenset = frozenset()  # markets where Spix's agreement counts; owner-approved only (D44)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _same_point(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return _number(a) and _number(b) and abs(float(a) - float(b)) < 1e-9


def _spix_probability(result: Mapping, group: str, side: str, point, bookmaker) -> Optional[float]:
    """Spix's own probability for the picked side: the anchor record's for 1X2, else the selected candidate's."""
    context = result.get("context") or {}
    if group == "moneyline":
        spix = (context.get("market_anchor") or {}).get("spix") or {}
        if _number(spix.get(side)):
            return float(spix[side])
    for candidate in (context.get("decision_audit") or {}).get("candidates") or []:
        option = candidate.get("option") or {}
        raw = option.get("team") if group == "spreads" else option.get("side")
        if candidate.get("selector_selected") and norm(raw) == side and _same_point(option.get("point"), point) \
                and option.get("bookmaker") == bookmaker:
            for key in ("_spix_prob", "_model_prob"):  # _spix_prob only exists where the market anchor priced it
                if _number(option.get(key)):
                    return float(option[key])
    probability = (result.get("decision") or {}).get("model_probability")
    return float(probability) if _number(probability) else None


def line_evidence(result: Mapping) -> Optional[dict]:
    """Bookmaker agreement, the market's value and Spix's value for a recommended result's quote, or None."""
    decision = result.get("decision") or {}
    quote = decision.get("quote") or {}
    group, odds = (result.get("market") or {}).get("group"), quote.get("odds")
    if decision.get("status") != "recommended" or not quote.get("market_key") or not _number(odds):
        return None
    context = result.get("context") or {}
    method = (context.get("engine_switches") or {}).get("SPIX_DEVIG_METHOD") or "proportional"
    books = audit_books((context.get("decision_audit") or {}).get("input_options"), quote["market_key"])
    side, point = norm(quote.get("side")), quote.get("line")
    consensus = devig_consensus(books, side, point, spreads=group == "spreads", method=method)[1] if books else None
    if consensus is None:
        return None
    market_p = consensus["fair_probability"]
    spix_p = _spix_probability(result, group, side, point, quote.get("bookmaker"))
    return {"market": group, "book_count": consensus["book_count"], "book_spread": consensus_spread(consensus),
            "market_probability": market_p, "market_value": market_p * odds - 1,
            "spix_probability": spix_p, "spix_value": None if spix_p is None else spix_p * odds - 1, "method": method}


def price_age_minutes(result: Mapping) -> Optional[float]:
    """Minutes between the quoting bookmaker's last check and the price confirmation, for a confirmed price only."""
    record = (result.get("context") or {}).get("price_confirmation") or {}
    if record.get("status") not in ("confirmed", "repriced"):
        return None
    checked, seen = utc_datetime(record.get("checked_at")), utc_datetime(record.get("bookmaker_seen_at"))
    return None if checked is None or seen is None else (checked - seen).total_seconds() / 60


def label(evidence: Optional[Mapping], age_minutes: Optional[float], *, cutoffs: Optional[Cutoffs] = None,
          spix_vote: Optional[frozenset] = None, min_value: float = MIN_VALUE) -> str:
    """ "high" or "low"; the module's ``CUTOFFS`` and ``SPIX_VOTE_MARKETS`` unless given."""
    cutoffs = CUTOFFS if cutoffs is None else cutoffs
    spix_vote = SPIX_VOTE_MARKETS if spix_vote is None else spix_vote
    if not evidence or cutoffs.max_spread is None:
        return "low"
    spread = evidence.get("book_spread")
    firm = ((evidence.get("book_count") or 0) >= cutoffs.min_books and _number(spread) and spread <= cutoffs.max_spread
            and age_minutes is not None and age_minutes <= cutoffs.fresh_minutes)
    market_value = evidence.get("market_value")
    spix_value = evidence.get("spix_value")
    spix_agrees = evidence.get("market") not in spix_vote or (_number(spix_value) and spix_value >= min_value)
    return "high" if firm and _number(market_value) and market_value >= min_value and spix_agrees else "low"


def _as_dict(result):
    to_dict = getattr(result, "to_dict", None)
    return json.loads(json.dumps(to_dict() if callable(to_dict) else dict(result)))


def _rebuild(original, snapshot):
    return type(original).from_dict(snapshot) if callable(getattr(original, "to_dict", None)) else snapshot


def apply_label(result):
    """The result labelled from its evidence when it was evaluated under ``evidence``; anything else unchanged."""
    context = result.get("context") if isinstance(result, Mapping) else getattr(result, "context", None)
    if not isinstance(context, Mapping) or \
            (context.get("engine_switches") or {}).get("SPIX_CONFIDENCE_LABELS") != "evidence":
        return result
    snapshot = _as_dict(result)
    if (snapshot.get("decision") or {}).get("status") != "recommended":
        return result
    evidence, age = line_evidence(snapshot), price_age_minutes(snapshot)
    verdict = label(evidence, age)
    snapshot["decision"] = {**snapshot["decision"], "confidence": verdict}
    snapshot["context"] = {**snapshot["context"], "confidence_evidence": {
        **(evidence or {}), "price_age_minutes": age, "label": verdict, "version": VERSION,
        "cutoffs": asdict(CUTOFFS), "spix_vote": sorted(SPIX_VOTE_MARKETS)}}
    return _rebuild(result, snapshot)


def apply_labels(results) -> list:
    return [apply_label(result) for result in results]
