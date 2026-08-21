"""Deterministic market rendering for Discord workflows.

This replaces prompt->stdout->regex paths for league-wide market workflows with
direct event fetching, enrichment, and renderer dispatch.
"""

from __future__ import annotations

from datetime import date
from typing import Callable, Dict, List, Optional, Set, Tuple

try:
    from core.events import fetch_events, filter_events_by_exact_date
except ImportError:
    from Scripts.rag_ingest.core.events import fetch_events, filter_events_by_exact_date  # type: ignore[import]

try:
    from core.parlay import enrich_events_for_groups
except ImportError:
    from Scripts.rag_ingest.core.parlay import enrich_events_for_groups  # type: ignore[import]

try:
    from core.market_result import MarketResult
    from core.market_service import evaluate_market
except ImportError:
    from Scripts.rag_ingest.core.market_result import MarketResult  # type: ignore[import]
    from Scripts.rag_ingest.core.market_service import evaluate_market  # type: ignore[import]

try:
    from rendering.market_results import render_market_results_text
except ImportError:
    from Scripts.rag_ingest.rendering.market_results import render_market_results_text  # type: ignore[import]

try:
    from rendering.btts import render_btts_answer
    from rendering.correct_score import render_correct_score_answer
    from rendering.moneyline import render_moneyline_answer
    from rendering.spreads import render_spreads_line_answer
    from rendering.team_totals import render_team_totals_answer
    from rendering.totals import render_goal_intervals, render_totals_line_answer
except ImportError:
    from Scripts.rag_ingest.rendering.btts import render_btts_answer  # type: ignore[import]
    from Scripts.rag_ingest.rendering.correct_score import render_correct_score_answer  # type: ignore[import]
    from Scripts.rag_ingest.rendering.moneyline import render_moneyline_answer  # type: ignore[import]
    from Scripts.rag_ingest.rendering.spreads import render_spreads_line_answer  # type: ignore[import]
    from Scripts.rag_ingest.rendering.team_totals import render_team_totals_answer  # type: ignore[import]
    from Scripts.rag_ingest.rendering.totals import render_goal_intervals, render_totals_line_answer  # type: ignore[import]


_Renderer = Callable[[str, str, List[Dict]], str]

# Public market names used by Discord/web delivery surfaces.  Only these have a
# complete MarketResult contract today.  Correct-score, intervals, and team
# totals retain their dedicated renderers until their structured contracts are
# introduced rather than being squeezed into an inaccurate generic shape.
_CANONICAL_MARKETS = {
    "totals": "goals",
    "goals": "goals",
    "corners": "corners",
    "cards": "cards",
    "sot": "sot",
    "btts": "btts",
    "moneyline": "moneyline",
    "spreads": "spreads",
}


def canonical_market_name(market: str) -> Optional[str]:
    """Return the canonical market-service name for a public market key."""
    return _CANONICAL_MARKETS.get((market or "").strip().lower())


def _canonical_groups(market: str) -> Set[str]:
    """Odds groups that must be present before canonical evaluation."""
    canonical = canonical_market_name(market)
    if canonical == "goals":
        return {"totals"}
    if canonical in {"corners", "cards", "sot", "btts", "moneyline", "spreads"}:
        return {canonical}
    return set()


def _market_spec(market: str) -> Tuple[str, Set[str], _Renderer]:
    market = (market or "").strip().lower()
    specs: Dict[str, Tuple[str, Set[str], _Renderer]] = {
        "totals": ("goals totals", {"totals"}, render_totals_line_answer),
        "corners": ("corners totals", {"corners"}, render_totals_line_answer),
        "cards": ("cards totals", {"cards"}, render_totals_line_answer),
        "sot": ("shots on target totals", {"sot"}, render_totals_line_answer),
        "btts": ("btts", {"btts"}, render_btts_answer),
        "moneyline": ("moneyline", {"moneyline"}, render_moneyline_answer),
        "spreads": ("spreads", {"spreads"}, render_spreads_line_answer),
        "correctscore": ("correct score", set(), render_correct_score_answer),
        "intervals": ("goal intervals", set(), render_goal_intervals),
        "teamlines": ("team lines", {"corners", "cards"}, render_team_totals_answer),
    }
    return specs.get(market, specs["totals"])


def evaluate_market_results_sync(
    league: str,
    market: str,
    target_date: date,
) -> Tuple[List[MarketResult], List[str]]:
    """Fetch, enrich, and evaluate one market without producing presentation text.

    This is the delivery boundary used by Discord and the website.  It keeps
    provider fetching outside the pure market service, but makes every
    supported consumer receive the same immutable decision objects.
    """
    canonical = canonical_market_name(market)
    if canonical is None:
        raise ValueError(f"Market {market!r} does not have a canonical result contract yet.")

    events, notes = fetch_events(league, target_date=target_date)
    day_events = filter_events_by_exact_date(events, target_date)
    if not day_events:
        return [], list(notes)

    desired_groups = _canonical_groups(market)
    if desired_groups:
        enriched, enrich_notes = enrich_events_for_groups(day_events, league, desired_groups)
        day_events = enriched or day_events
        notes.extend(enrich_notes)

    fixture_date = target_date.isoformat()
    results: List[MarketResult] = []
    for event in sorted(day_events, key=lambda ev: str(ev.get("commence_time") or "")):
        try:
            results.append(evaluate_market(
                event,
                league,
                canonical,
                fixture_date=fixture_date,
                context={"delivery_market": market},
            ))
        except Exception as exc:
            fixture = f"{event.get('home_team') or 'Home'} vs {event.get('away_team') or 'Away'}"
            notes.append(f"{fixture}: market evaluation failed ({exc}).")
    return results, list(notes)


def render_market_answer_sync(league: str, market: str, target_date: date) -> str:
    """Fetch events deterministically and render one league-wide market answer."""
    if canonical_market_name(market) is not None:
        results, notes = evaluate_market_results_sync(league, market, target_date)
        if not results:
            note_suffix = f" Notes: {'; '.join(notes[:2])}." if notes else ""
            return f"No fixtures found for {league} on {target_date.isoformat()}.{note_suffix}"
        return render_market_results_text(results, notes=notes[:5])

    user_q, desired_groups, renderer = _market_spec(market)
    events, notes = fetch_events(league, target_date=target_date)
    day_events = filter_events_by_exact_date(events, target_date)
    if not day_events:
        note_suffix = f" Notes: {'; '.join(notes[:2])}." if notes else ""
        return f"No fixtures found for {league} on {target_date.isoformat()}.{note_suffix}"

    if desired_groups:
        enriched, enrich_notes = enrich_events_for_groups(day_events, league, desired_groups)
        day_events = enriched or day_events
        if enrich_notes:
            notes.extend(enrich_notes)

    day_events = sorted(day_events, key=lambda ev: str(ev.get("commence_time") or ""))
    text = renderer(user_q, league, day_events)
    if notes:
        note_block = "\n".join(f"Note: {note}" for note in notes[:5])
        return f"{text}\n{note_block}"
    return text
