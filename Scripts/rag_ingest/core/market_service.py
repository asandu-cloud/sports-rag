"""Canonical pre-match market evaluation service.

The service assembles core projections, normalized odds, and shared line
selection into ``MarketResult`` objects.  It is intentionally presentation-free
and has no production callers yet; delivery surfaces migrate in Phase 4.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    from core.line_selection import (
        choose_best_moneyline_side,
        confidence_from_edge,
        select_best_btts_recommendation,
        select_best_spread_recommendation,
        select_best_total_recommendation,
    )
    from core.market_result import (
        Decision,
        DecisionStatus,
        EvidenceItem,
        FixtureRef,
        MarketRef,
        MarketResult,
        Projection,
        Provenance,
        decision_from_selector,
    )
    from core.odds_extraction import (
        extract_btts_odds,
        extract_moneyline_odds,
        extract_spread_line_options,
        extract_total_line_options,
    )
    from core.projections import (
        projected_btts_prob,
        projected_goal_difference,
        projected_moneyline_probs,
        projected_total_cards,
        projected_total_corners,
        projected_total_goals,
        projected_total_sot,
    )
except ImportError:
    from .line_selection import (  # type: ignore[no-redef]
        choose_best_moneyline_side,
        confidence_from_edge,
        select_best_btts_recommendation,
        select_best_spread_recommendation,
        select_best_total_recommendation,
    )
    from .market_result import (  # type: ignore[no-redef]
        Decision,
        DecisionStatus,
        EvidenceItem,
        FixtureRef,
        MarketRef,
        MarketResult,
        Projection,
        Provenance,
        decision_from_selector,
    )
    from .odds_extraction import (  # type: ignore[no-redef]
        extract_btts_odds,
        extract_moneyline_odds,
        extract_spread_line_options,
        extract_total_line_options,
    )
    from .projections import (  # type: ignore[no-redef]
        projected_btts_prob,
        projected_goal_difference,
        projected_moneyline_probs,
        projected_total_cards,
        projected_total_corners,
        projected_total_goals,
        projected_total_sot,
    )


CANONICAL_PIPELINE_VERSION = "core-market-service.v1"
SUPPORTED_MARKETS = ("goals", "corners", "cards", "sot", "btts", "moneyline", "spreads")
_TOTAL_MARKETS = {"goals", "corners", "cards", "sot"}


def _event_text(event: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = event.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _fixture_ref(event: Mapping[str, Any], league: str) -> FixtureRef:
    home = _event_text(event, "home_team", "home")
    away = _event_text(event, "away_team", "away")
    kickoff = _event_text(event, "commence_time", "fixture_date", "kickoff") or None
    event_id = _event_text(event, "id", "event_id", "fixture_id")
    if not event_id:
        event_id = hashlib.sha256(
            f"{league}|{home}|{away}|{kickoff or ''}".encode("utf-8")
        ).hexdigest()[:24]
    return FixtureRef(event_id=event_id, league=league, home_team=home, away_team=away, kickoff=kickoff)


def _snapshot_id(event: Mapping[str, Any], league: str, fixture_date: Optional[str]) -> str:
    try:
        encoded = json.dumps(
            {"league": league, "fixture_date": fixture_date, "event": dict(event)},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        encoded = f"{league}|{fixture_date or ''}|{event!r}"
    return f"event:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]}"


def _generated_at(value: Optional[str]) -> str:
    if value:
        return value
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _with_confidence(decision: Decision, stat_group: str) -> Decision:
    if not decision.is_recommended:
        return decision
    confidence = confidence_from_edge(
        abs(decision.value_edge or 0.0),
        stat_group=stat_group,
        model_prob=decision.model_probability,
        value_edge_pct=decision.value_edge,
    )
    return replace(decision, confidence=confidence)


def _unavailable(reason: str) -> Decision:
    return Decision(status=DecisionStatus.UNAVAILABLE, reason=reason)


def _components_evidence(season: Optional[float], recent: Optional[float]) -> List[EvidenceItem]:
    values: Dict[str, float] = {}
    if season is not None:
        values["season"] = season
    if recent is not None:
        values["recent"] = recent
    return [EvidenceItem("projection_components", values)] if values else []


def _build_result(
    *,
    fixture: FixtureRef,
    market: MarketRef,
    projection: Projection,
    decision: Decision,
    provenance: Provenance,
    evidence: Iterable[EvidenceItem] = (),
    context: Optional[Mapping[str, Any]] = None,
) -> MarketResult:
    return MarketResult(
        fixture=fixture,
        market=market,
        projection=projection,
        decision=decision,
        provenance=provenance,
        evidence=tuple(evidence),
        context=dict(context or {}),
    )


def evaluate_market(
    event: Mapping[str, Any],
    league: str,
    market_name: str,
    *,
    fixture_date: Optional[str] = None,
    league_ctx: Any = None,
    knockout_ctx: Any = None,
    lineup_ctx: Any = None,
    ref_mod: Any = None,
    input_snapshot_id: Optional[str] = None,
    generated_at: Optional[str] = None,
    model_version: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> MarketResult:
    """Evaluate one supported pre-match market for an already-fetched event.

    ``event`` is the normalized odds-provider payload.  Callers that have
    canonical storage should supply their own ``input_snapshot_id``; the local
    hash is a deterministic compatibility fallback during the migration.
    """

    market_name = str(market_name or "").lower().strip()
    if market_name not in SUPPORTED_MARKETS:
        raise ValueError(f"Unsupported canonical market: {market_name!r}")

    fixture = _fixture_ref(event, league)
    source_date = fixture_date or fixture.kickoff
    provenance = Provenance(
        pipeline_version=CANONICAL_PIPELINE_VERSION,
        input_snapshot_id=input_snapshot_id or _snapshot_id(event, league, source_date),
        generated_at=_generated_at(generated_at),
        model_version=model_version,
        as_of=source_date,
        sources=("core.projections", "core.line_selection", "core.odds_extraction"),
    )
    result_context = {"fixture_date": source_date} if source_date else {}
    result_context.update(dict(context or {}))
    home, away = fixture.home_team, fixture.away_team

    if market_name in _TOTAL_MARKETS:
        if market_name == "goals":
            value, season, recent = projected_total_goals(
                home, away, league, knockout_ctx=knockout_ctx,
                league_ctx=league_ctx, fixture_date=source_date,
            )
        elif market_name == "corners":
            value, season, recent = projected_total_corners(
                home, away, league, knockout_ctx=knockout_ctx,
                league_ctx=league_ctx, fixture_date=source_date,
            )
        elif market_name == "cards":
            value, season, recent, resolved_ref = projected_total_cards(
                home, away, league, knockout_ctx=knockout_ctx, ref_mod=ref_mod,
                lineup_ctx=lineup_ctx, league_ctx=league_ctx, fixture_date=source_date,
            )
            if resolved_ref is not None:
                result_context["referee_source"] = str(getattr(resolved_ref, "source", "unavailable"))
        else:
            value, season, recent = projected_total_sot(
                home, away, league, knockout_ctx=knockout_ctx,
                league_ctx=league_ctx, fixture_date=source_date,
            )

        projection = Projection(
            value=value, unit=market_name,
            season_component=season, recent_component=recent,
        )
        decision = _unavailable("Insufficient profile data to project this market.")
        if value is not None:
            selector_result = select_best_total_recommendation(
                extract_total_line_options(dict(event), market_name), value,
            )
            decision = _with_confidence(decision_from_selector(selector_result), market_name)
        return _build_result(
            fixture=fixture,
            market=MarketRef(key="totals", group=market_name, unit=market_name),
            projection=projection,
            decision=decision,
            provenance=provenance,
            evidence=_components_evidence(season, recent),
            context=result_context,
        )

    if market_name == "btts":
        p_yes, home_goals, away_goals, _, _ = projected_btts_prob(
            home, away, league, league_ctx=league_ctx, fixture_date=source_date,
        )
        projection = Projection(
            value=p_yes, unit="probability",
            components={"yes_probability": p_yes, "home_goals": home_goals, "away_goals": away_goals},
        )
        decision = _unavailable("Insufficient profile data to project BTTS.")
        if p_yes is not None:
            selector_result = select_best_btts_recommendation(extract_btts_odds(dict(event)), p_yes)
            decision = _with_confidence(decision_from_selector(selector_result), "goals")
            if decision.quote is not None:
                decision = replace(decision, quote=replace(
                    decision.quote, side=decision.quote.side.lower()
                ))
        return _build_result(
            fixture=fixture,
            market=MarketRef(key="btts", group="btts", unit="probability"),
            projection=projection,
            decision=decision,
            provenance=provenance,
            evidence=(EvidenceItem("btts_goal_split", {"home_goals": home_goals, "away_goals": away_goals}),),
            context=result_context,
        )

    if market_name == "moneyline":
        p_home, p_draw, p_away, home_goals, away_goals = projected_moneyline_probs(
            home, away, league, league_ctx=league_ctx, fixture_date=source_date,
        )
        projection = Projection(
            value=p_home, unit="probability",
            components={
                "home_probability": p_home,
                "draw_probability": p_draw,
                "away_probability": p_away,
                "home_goals": home_goals,
                "away_goals": away_goals,
            },
        )
        decision = _unavailable("Insufficient profile data to project moneyline.")
        if None not in (p_home, p_draw, p_away):
            options = extract_moneyline_odds(dict(event))
            if not options:
                decision = _unavailable("No market prices are available for this moneyline fixture.")
            else:
                selector_result = choose_best_moneyline_side(p_home, p_draw, p_away, options, home, away)
                decision = _with_confidence(decision_from_selector(selector_result), "goals")
        return _build_result(
            fixture=fixture,
            market=MarketRef(key="moneyline", group="moneyline", unit="probability"),
            projection=projection,
            decision=decision,
            provenance=provenance,
            evidence=(EvidenceItem("moneyline_probabilities", projection.components),),
            context=result_context,
        )

    projected_diff, season_diff, recent_diff = projected_goal_difference(
        home, away, league, league_ctx=league_ctx, fixture_date=source_date,
    )
    projection = Projection(
        value=projected_diff, unit="goals",
        season_component=season_diff, recent_component=recent_diff,
    )
    decision = _unavailable("Insufficient profile data to project the goal difference.")
    if projected_diff is not None:
        selector_result = select_best_spread_recommendation(
            extract_spread_line_options(dict(event)), projected_diff, home,
            away_team=away, league=league,
        )
        decision = _with_confidence(decision_from_selector(selector_result), "goals")
    return _build_result(
        fixture=fixture,
        market=MarketRef(key="spreads", group="spreads", unit="goals"),
        projection=projection,
        decision=decision,
        provenance=provenance,
        evidence=_components_evidence(season_diff, recent_diff),
        context=result_context,
    )


def evaluate_event(
    event: Mapping[str, Any],
    league: str,
    markets: Iterable[str] = SUPPORTED_MARKETS,
    **kwargs: Any,
) -> List[MarketResult]:
    """Evaluate requested canonical markets for one event in stable input order."""

    return [evaluate_market(event, league, market_name, **kwargs) for market_name in markets]
