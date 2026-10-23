"""Presentation adapters for canonical :class:`core.market_result.MarketResult`.

The canonical market service deliberately knows nothing about Discord or text
formats.  This module is the narrow compatibility boundary for the remaining
text consumers while delivery surfaces are migrated to structured results.
"""

from __future__ import annotations

from typing import Iterable, List

try:
    from core.market_result import (
        DecisionStatus, MarketResult, probability_source, published_projection, published_result_probabilities,
    )
except ImportError:
    from Scripts.rag_ingest.core.market_result import (  # type: ignore[import]
        DecisionStatus, MarketResult, probability_source, published_projection, published_result_probabilities,
    )


_MARKET_LABELS = {
    "goals": "Goals Totals",
    "corners": "Corners Totals",
    "cards": "Cards Totals",
    "sot": "Shots on Target Totals",
    "btts": "Both Teams To Score (BTTS)",
    "moneyline": "Moneyline (1X2)",
    "spreads": "Score Handicap",
}


def market_label(result: MarketResult) -> str:
    """Return a stable user-facing label for a canonical market result."""
    return _MARKET_LABELS.get(result.market.group, result.market.group.replace("_", " ").title())


def result_pick_label(result: MarketResult) -> str:
    """Format the actual priced selection without inventing a model-only pick."""
    quote = result.decision.quote
    if quote is None:
        return ""
    side = quote.side.title()
    if result.market.group == "btts":
        return f"BTTS {side}"
    if quote.line is None:
        return side
    if result.market.group == "spreads":
        return f"{side} {quote.line:+g}"
    return f"{side} {quote.line:g}"


def projection_label(result: MarketResult) -> str:
    """Format the model output while retaining the measurement it represents."""
    value, source = published_projection(result)
    if value is None:
        return "Unavailable"
    group = result.market.group
    market = " (market)" if source == "market" else ""
    if result.projection.unit == "probability":
        if group == "moneyline":
            probabilities, source = published_result_probabilities(result)
            if probabilities:
                label = "1X2 (market)" if source == "market" else "1X2"
                return f"{label}: {probabilities['home']:.1%} / {probabilities['draw']:.1%} / {probabilities['away']:.1%}"
        if group == "btts":
            return f"BTTS Yes{market}: {value:.1%}"
        return f"{value:.1%}"
    if group == "spreads":
        return f"Goal difference{market}: {value:+.2f}"
    return f"Projected total{market}: {value:.2f} {result.projection.unit}"


def render_market_results_text(results: Iterable[MarketResult], *, notes: Iterable[str] = ()) -> str:
    """Render canonical results for legacy text-only callers.

    This is intentionally a one-way presentation adapter.  It does not parse
    text or re-run projection/selection logic.
    """
    ordered = list(results)
    if not ordered:
        return "No market results are available."

    lines: List[str] = [f"{market_label(ordered[0])} advice by fixture:"]
    for result in ordered:
        fixture = f"{result.fixture.home_team} vs {result.fixture.away_team}"
        lines.append(f"\n=== {fixture} ===")
        lines.append(f"  {projection_label(result)}")
        decision = result.decision
        if decision.status is DecisionStatus.RECOMMENDED and decision.quote is not None:
            quote = decision.quote
            price_source = ", ".join(
                item for item in (quote.bookmaker, quote.market_key) if item
            )
            source_suffix = f" ({price_source})" if price_source else ""
            lines.append(f"  Recommended: {result_pick_label(result)} @ {quote.odds:.2f}{source_suffix}.")
            if decision.model_probability is not None:
                label = ("Asian price-comparison probability" if decision.probability_basis == "asian_equivalent_non_push"
                         else "Market probability" if probability_source(result) == "market" else "Model probability")
                lines.append(f"  {label}: {decision.model_probability:.1%}")
            if decision.implied_probability is not None:
                lines.append(f"  Book implied probability: {decision.implied_probability:.1%}")
            if decision.value_edge is not None:
                lines.append(f"  Value edge: {decision.value_edge:+.1%}")
            if decision.expected_value is not None:
                lines.append(f"  Expected value: {decision.expected_value:+.3f} per unit")
            if decision.confidence:
                lines.append(f"  Confidence: {decision.confidence.title()}")
        elif decision.status is DecisionStatus.NO_BET:
            lines.append(f"  Verdict: No bet — {decision.reason}")
        else:
            lines.append(f"  Unavailable: {decision.reason}")

    for note in notes:
        lines.append(f"Note: {note}")
    return "\n".join(lines)
