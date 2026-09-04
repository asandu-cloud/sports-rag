"""Fixture-level Match Read generation boundary.

The legacy delivery flow fetches and evaluates each market separately.  This
module is the deliberately separate, fixture-first path: it fetches one
league/day slate, obtains every canonical odds group once, evaluates every
canonical market for each fixture once, and hands the resulting immutable
``MarketResult`` objects to the Match Read compiler.

It has no Discord, website, scheduler, or automatic database side effects.
Persistence is opt-in on each call so callers can use the exact same output
for shadow review before it becomes a delivery source.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# The Discord bot already inserts this directory before importing rendering
# modules, while ``python -m Scripts...`` does not.  Make this dispatcher
# usable from either entry point (including the operator CLI) without asking
# callers to recreate that fragile path setup.
_RAG_ROOT = Path(__file__).resolve().parents[1]
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))

try:
    from core.events import fetch_events, filter_events_by_exact_date
    from core.market_service import SUPPORTED_MARKETS, evaluate_event
    from core.parlay import enrich_events_for_groups
except ImportError:
    from Scripts.rag_ingest.core.events import fetch_events, filter_events_by_exact_date  # type: ignore[import]
    from Scripts.rag_ingest.core.market_service import SUPPORTED_MARKETS, evaluate_event  # type: ignore[import]
    from Scripts.rag_ingest.core.parlay import enrich_events_for_groups  # type: ignore[import]

try:
    from data_platform.services.match_read_compiler import (
        MatchReadDraft,
        compile_match_read,
        persist_match_read,
    )
except ImportError:
    from Scripts.data_platform.services.match_read_compiler import (  # type: ignore[import]
        MatchReadDraft,
        compile_match_read,
        persist_match_read,
    )


# ``goals`` is the canonical evaluator name but its provider odds group is
# named ``totals``.  The rest of the canonical evaluator names match the
# enrichment groups used by ``enrich_events_for_groups``.
CANONICAL_ODDS_GROUPS = frozenset({
    "totals",
    "corners",
    "cards",
    "sot",
    "btts",
    "moneyline",
    "spreads",
})


@dataclass(frozen=True)
class FixtureMatchReadGeneration:
    """One fixture's canonical evaluation, compiled read, and optional record.

    ``error`` is fixture-local.  A bad provider payload for one fixture must
    not prevent a reviewable Match Read slate from being produced for the
    others.  If an error occurs after compilation (during explicit
    persistence), ``draft`` remains available for inspection or retry.
    """

    event_id: str
    fixture_label: str
    canonical_results: Tuple[Any, ...]
    draft: Optional[MatchReadDraft]
    record: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def status(self) -> Optional[str]:
        """The compiled state, if this fixture was successfully compiled."""
        return self.draft.status if self.draft is not None else None


@dataclass(frozen=True)
class MatchReadGenerationRun:
    """The complete, deterministic result of generating one league/day slate."""

    league: str
    target_date: date
    stage: str
    fixtures: Tuple[FixtureMatchReadGeneration, ...]
    notes: Tuple[str, ...]

    @property
    def drafts(self) -> Tuple[MatchReadDraft, ...]:
        """All successfully compiled drafts, including no-bet/unavailable reads."""
        return tuple(item.draft for item in self.fixtures if item.draft is not None)

    @property
    def records(self) -> Tuple[Dict[str, Any], ...]:
        """Only records created through an explicit ``persist=True`` call."""
        return tuple(item.record for item in self.fixtures if item.record is not None)


def generate_match_reads_sync(
    league: str,
    target_date: date,
    *,
    stage: str = "pre_match",
    persist: bool = False,
    service: Any = None,
    fetcher: Optional[Callable[..., Tuple[Sequence[Mapping[str, Any]], Sequence[str]]]] = None,
    date_filter: Optional[Callable[..., Sequence[Mapping[str, Any]]]] = None,
    enricher: Optional[Callable[..., Tuple[Sequence[Mapping[str, Any]], Sequence[str]]]] = None,
    evaluator: Optional[Callable[..., Sequence[Any]]] = None,
    compiler: Optional[Callable[..., MatchReadDraft]] = None,
    persister: Optional[Callable[..., Dict[str, Any]]] = None,
    lineup_provider: Optional[Callable[[Mapping[str, Any], str, date], Any]] = None,
) -> MatchReadGenerationRun:
    """Generate one Match Read draft per exact-date fixture in a league.

    The default path performs exactly one fetch and one all-groups enrichment
    for the requested league/date.  Each remaining fixture is then passed to
    :func:`evaluate_event` once with all ``SUPPORTED_MARKETS`` in their stable
    canonical order.

    ``persist`` defaults to ``False``.  Setting it to ``True`` is an explicit
    caller decision and invokes the existing immutable Match Read service;
    this dispatcher never publishes a recommendation or sends a message.
    Dependency arguments are intentionally injectable for offline tests and
    controlled shadow runs.
    """
    normalised_league = str(league or "").strip()
    if not normalised_league:
        raise ValueError("league is required.")
    if not isinstance(target_date, date):
        raise TypeError("target_date must be a datetime.date.")
    normalised_stage = str(stage or "").strip().lower()
    if normalised_stage not in {"pre_match", "confirmed_lineups"}:
        raise ValueError("stage must be 'pre_match' or 'confirmed_lineups'.")
    if normalised_stage == "confirmed_lineups" and lineup_provider is None:
        # A stage label is a factual claim.  Do not allow a caller to create a
        # supposedly confirmed-lineups amendment merely by changing a CLI
        # argument while the evaluation still uses the pre-match inputs.
        return MatchReadGenerationRun(
            league=normalised_league,
            target_date=target_date,
            stage=normalised_stage,
            fixtures=(),
            notes=(
                f"{normalised_league}: confirmed-lineups generation requires a verified "
                "lineup provider; no Match Reads were generated.",
            ),
        )

    fetch = fetcher or fetch_events
    filter_for_date = date_filter or filter_events_by_exact_date
    enrich = enricher or enrich_events_for_groups
    evaluate = evaluator or evaluate_event
    compile_read = compiler or compile_match_read
    persist_read = persister or persist_match_read

    notes: List[str] = []
    try:
        fetched_events, fetch_notes = fetch(normalised_league, target_date=target_date)
    except Exception as exc:
        return MatchReadGenerationRun(
            league=normalised_league,
            target_date=target_date,
            stage=normalised_stage,
            fixtures=(),
            notes=(f"{normalised_league}: fixture fetch failed ({type(exc).__name__}: {exc}).",),
        )
    notes.extend(_normalise_notes(fetch_notes))

    try:
        day_events = list(filter_for_date(list(fetched_events or ()), target_date))
    except Exception as exc:
        return MatchReadGenerationRun(
            league=normalised_league,
            target_date=target_date,
            stage=normalised_stage,
            fixtures=(),
            notes=tuple(notes + [
                f"{normalised_league}: exact-date fixture filtering failed "
                f"({type(exc).__name__}: {exc})."
            ]),
        )
    if not day_events:
        return MatchReadGenerationRun(
            league=normalised_league,
            target_date=target_date,
            stage=normalised_stage,
            fixtures=(),
            notes=tuple(notes),
        )

    # Do this once across the full slate, rather than one provider round-trip
    # per market or fixture.  A failed/empty enrichment must not discard the
    # base prices we already fetched; those still produce auditable
    # unavailable/no-bet canonical results where appropriate.
    try:
        enriched_events, enrich_notes = enrich(day_events, normalised_league, set(CANONICAL_ODDS_GROUPS))
        notes.extend(_normalise_notes(enrich_notes))
        if enriched_events:
            day_events = list(enriched_events)
        else:
            notes.append(
                f"{normalised_league}: all-groups odds enrichment returned no events; "
                "evaluating the base fixture snapshots."
            )
    except Exception as exc:
        notes.append(
            f"{normalised_league}: all-groups odds enrichment failed "
            f"({type(exc).__name__}: {exc}); evaluating the base fixture snapshots."
        )

    fixture_rows: List[FixtureMatchReadGeneration] = []
    seen_fixture_ids: Set[str] = set()
    fixture_date = target_date.isoformat()
    for event in sorted(day_events, key=_event_sort_key):
        if not isinstance(event, Mapping):
            notes.append(f"{normalised_league}: ignored a non-mapping fixture payload.")
            continue
        event_id = _event_identity(event)
        fixture_label = _fixture_label(event)
        if event_id in seen_fixture_ids:
            notes.append(f"{fixture_label}: duplicate fixture payload ignored ({event_id}).")
            continue
        seen_fixture_ids.add(event_id)

        lineup_ctx = None
        if normalised_stage == "confirmed_lineups":
            try:
                lineup_ctx = lineup_provider(event, normalised_league, target_date)  # type: ignore[misc]
            except Exception as exc:
                message = f"{fixture_label}: confirmed-lineups lookup failed ({type(exc).__name__}: {exc})."
                notes.append(message)
                fixture_rows.append(FixtureMatchReadGeneration(
                    event_id=event_id,
                    fixture_label=fixture_label,
                    canonical_results=(),
                    draft=None,
                    error=message,
                ))
                continue
            if not _is_confirmed_lineup_context(lineup_ctx):
                message = (
                    f"{fixture_label}: confirmed lineups are not available; "
                    "no confirmed-lineups Match Read was generated."
                )
                notes.append(message)
                fixture_rows.append(FixtureMatchReadGeneration(
                    event_id=event_id,
                    fixture_label=fixture_label,
                    canonical_results=(),
                    draft=None,
                    error=message,
                ))
                continue

        try:
            evaluation_kwargs = {
                "markets": SUPPORTED_MARKETS,
                "fixture_date": fixture_date,
                "context": {
                    "delivery_surface": "match_read_shadow",
                    "match_read_stage": normalised_stage,
                },
            }
            if lineup_ctx is not None:
                evaluation_kwargs["lineup_ctx"] = lineup_ctx
            canonical_results = tuple(evaluate(event, normalised_league, **evaluation_kwargs))
        except Exception as exc:
            message = f"{fixture_label}: canonical fixture evaluation failed ({type(exc).__name__}: {exc})."
            notes.append(message)
            fixture_rows.append(FixtureMatchReadGeneration(
                event_id=event_id,
                fixture_label=fixture_label,
                canonical_results=(),
                draft=None,
                error=message,
            ))
            continue

        try:
            draft = compile_read(canonical_results, stage=normalised_stage)
        except Exception as exc:
            message = f"{fixture_label}: Match Read compilation failed ({type(exc).__name__}: {exc})."
            notes.append(message)
            fixture_rows.append(FixtureMatchReadGeneration(
                event_id=event_id,
                fixture_label=fixture_label,
                canonical_results=canonical_results,
                draft=None,
                error=message,
            ))
            continue

        record: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        if persist:
            try:
                record = persist_read(draft, service=service)
            except Exception as exc:
                error = f"{fixture_label}: Match Read persistence failed ({type(exc).__name__}: {exc})."
                notes.append(error)

        fixture_rows.append(FixtureMatchReadGeneration(
            event_id=event_id,
            fixture_label=fixture_label,
            canonical_results=canonical_results,
            draft=draft,
            record=record,
            error=error,
        ))

    return MatchReadGenerationRun(
        league=normalised_league,
        target_date=target_date,
        stage=normalised_stage,
        fixtures=tuple(fixture_rows),
        notes=tuple(notes),
    )


def _normalise_notes(values: Optional[Iterable[Any]]) -> List[str]:
    return [str(value) for value in values or () if str(value).strip()]


def _event_sort_key(event: Any) -> Tuple[str, str]:
    if not isinstance(event, Mapping):
        return ("", "")
    return (
        str(event.get("commence_time") or event.get("kickoff") or ""),
        _event_identity(event),
    )


def _event_identity(event: Mapping[str, Any]) -> str:
    """Use provider identity when supplied, with a stable defensive fallback."""
    for key in ("id", "event_id", "fixture_id"):
        value = str(event.get(key) or "").strip()
        if value:
            return value
    return "|".join((
        str(event.get("home_team") or event.get("home") or "").strip(),
        str(event.get("away_team") or event.get("away") or "").strip(),
        str(event.get("commence_time") or event.get("kickoff") or "").strip(),
    )) or "unknown-fixture"


def _fixture_label(event: Mapping[str, Any]) -> str:
    home = str(event.get("home_team") or event.get("home") or "Home").strip()
    away = str(event.get("away_team") or event.get("away") or "Away").strip()
    return f"{home} vs {away}"


def _is_confirmed_lineup_context(value: Any) -> bool:
    """Recognise a verified context without depending on a concrete class.

    The scorer currently uses ``LineupContext`` objects, but keeping this
    duck-typed lets a future platform-backed lineup snapshot pass through
    without a rendering-layer dependency.  Both flags are required so a
    neutral/unavailable context can never be labelled confirmed.
    """
    return (
        str(getattr(value, "source", "") or "").strip().lower() == "lineups"
        and bool(getattr(value, "is_available", False))
    )
