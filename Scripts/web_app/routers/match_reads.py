"""Public, persisted Match Read delivery endpoints.

The website must never re-run a model merely because a visitor loads a page.
These routes therefore only read immutable Match Reads already stored in the
platform database.  They expose the shared card DTO used by website and
Discord delivery, rather than legacy formatted text or the raw archival JSON.

An effective card is deliberately chosen here (at the presentation boundary):
the newest confirmed-lineups read wins; if none exists, the newest pre-match
read wins.  Historical versions remain available from the fixture detail
endpoint for transparent amendment history.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from fastapi import APIRouter, HTTPException, Query


logger = logging.getLogger("web_app.match_reads")

_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


router = APIRouter(prefix="/api/match-reads", tags=["match-reads"])

MATCH_READ_BOARD_SCHEMA_VERSION = "match-read-board.v1"
MATCH_READ_DETAIL_SCHEMA_VERSION = "match-read-detail.v1"
PUBLIC_MATCH_READ_LEAGUES = ("EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1", "UCL")
_EFFECTIVE_STAGES = ("confirmed_lineups", "pre_match")
_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


def _get_match_read_service():
    """Construct the read-only platform service lazily.

    Keeping this as a tiny dependency seam makes the router usable in an
    environment where the platform has not been bootstrapped yet, and lets
    tests supply an isolated database without starting the full website app.
    """
    from data_platform.services.match_reads import MatchReadService

    return MatchReadService()


def _parse_matchday(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="target_date must use YYYY-MM-DD format.",
        ) from exc
    if str(value) != parsed.isoformat():
        raise HTTPException(
            status_code=400,
            detail="target_date must use YYYY-MM-DD format.",
        )
    return parsed


def _timestamp_rank(value: Any) -> float:
    """Safely normalise an ISO timestamp for deterministic ordering."""
    raw = str(value or "").strip()
    if not raw:
        return float("-inf")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _revision_key(read: Mapping[str, Any]) -> tuple[int, float, float, int]:
    """Return a stable latest-version key within one fixture/stage stream."""
    try:
        version = int(read.get("version") or 0)
    except (TypeError, ValueError):
        version = 0
    try:
        record_id = int(read.get("id") or 0)
    except (TypeError, ValueError):
        record_id = 0
    provenance = read.get("provenance")
    evaluated_at = provenance.get("evaluated_at") if isinstance(provenance, Mapping) else None
    return (
        version,
        _timestamp_rank(evaluated_at),
        _timestamp_rank(read.get("created_at")),
        record_id,
    )


def select_effective_match_read(reads: Iterable[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Choose the approved visible revision without discarding history.

    The stage order is intentional product policy, not an inferred timestamp:
    any confirmed-lineups revision wins over every pre-match revision.  Within
    that stage, the immutable revision number is primary and timestamps/ID
    only resolve malformed or imported ties deterministically.
    """
    snapshots = [dict(read) for read in reads if isinstance(read, Mapping)]
    for stage in _EFFECTIVE_STAGES:
        candidates = [read for read in snapshots if read.get("stage") == stage]
        if candidates:
            return max(candidates, key=_revision_key)
    return None


def _history_key(read: Mapping[str, Any]) -> tuple[float, int, int, int]:
    """Order history newest first while retaining a stable tie-break."""
    provenance = read.get("provenance")
    evaluated_at = provenance.get("evaluated_at") if isinstance(provenance, Mapping) else None
    stage_rank = 1 if read.get("stage") == "confirmed_lineups" else 0
    version, _, _, record_id = _revision_key(read)
    return (_timestamp_rank(evaluated_at) or _timestamp_rank(read.get("created_at")), stage_rank, version, record_id)


def _update_metadata(read: Mapping[str, Any]) -> Dict[str, Any]:
    """Describe the visible amendment state without mutating the snapshot."""
    stage = str(read.get("stage") or "")
    supersedes = read.get("supersedes_id")
    try:
        version = int(read.get("version") or 0)
    except (TypeError, ValueError):
        version = 0

    if stage == "confirmed_lineups":
        return {
            "is_updated": True,
            "label": "Updated after confirmed lineups",
            "reason": "confirmed_lineups",
            "supersedes_id": supersedes,
        }
    if supersedes is not None or version > 1:
        return {
            "is_updated": True,
            "label": "Updated",
            "reason": "revised_pre_match",
            "supersedes_id": supersedes,
        }
    return {
        "is_updated": False,
        "label": None,
        "reason": None,
        "supersedes_id": None,
    }


def _card_from_read(read: Mapping[str, Any]) -> Dict[str, Any]:
    """Adapt a persisted record to the shared public card contract."""
    from data_platform.services.match_read_cards import (
        MatchReadCardError,
        build_match_read_card,
    )

    try:
        card = build_match_read_card(read)
    except MatchReadCardError as exc:
        raise ValueError(f"Stored Match Read {read.get('id')} is not deliverable: {exc}") from exc
    card["update"] = _update_metadata(read)
    return card


def _effective_cards(reads: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return one visible card per fixture, ordered by kickoff then fixture ID."""
    by_fixture: Dict[str, List[Mapping[str, Any]]] = {}
    for read in reads:
        fixture = read.get("fixture") if isinstance(read.get("fixture"), Mapping) else {}
        fixture_id = str(fixture.get("event_id") or "").strip()
        if not fixture_id:
            logger.warning("Skipping Match Read with no fixture event ID: %r", read.get("id"))
            continue
        by_fixture.setdefault(fixture_id, []).append(read)

    cards: List[Dict[str, Any]] = []
    for fixture_id, fixture_reads in by_fixture.items():
        effective = select_effective_match_read(fixture_reads)
        if effective is None:
            logger.warning("Skipping Match Read fixture %s with no delivery-supported stage", fixture_id)
            continue
        try:
            cards.append(_card_from_read(effective))
        except ValueError as exc:
            # A corrupt archival row should not hide the rest of a matchday
            # board.  It remains visible to operations through the log.
            logger.warning("Skipping unreadable Match Read fixture %s: %s", fixture_id, exc)

    return sorted(
        cards,
        key=lambda card: (
            str(card.get("fixture", {}).get("kickoff") or ""),
            str(card.get("fixture", {}).get("event_id") or ""),
        ),
    )


def _load_matchday_reads(service: Any, *, league: str, target_date: date) -> List[Dict[str, Any]]:
    """Load only cards explicitly released to the public website surface."""
    try:
        return service.list_delivered_for_matchday(
            league=league,
            target_date=target_date,
            surface="website",
        )
    except Exception as exc:
        logger.exception("Match Read store failed for %s on %s", league, target_date)
        raise HTTPException(
            status_code=503,
            detail="Persisted Match Reads are temporarily unavailable.",
        ) from exc


def _core_rank(card: Mapping[str, Any]) -> tuple[int, float, float, str]:
    """Rank a public fixture card for the five-card daily shortlist."""
    selections = card.get("selections")
    core = next(
        (
            item for item in selections
            if isinstance(item, Mapping) and item.get("role") == "core"
        ),
        {},
    ) if isinstance(selections, Sequence) else {}
    confidence = str(core.get("confidence") or "").lower()
    try:
        edge = float(core.get("value_edge"))
    except (TypeError, ValueError):
        edge = float("-inf")
    try:
        probability = float(core.get("model_probability"))
    except (TypeError, ValueError):
        probability = float("-inf")
    return (
        _CONFIDENCE_RANK.get(confidence, 0),
        edge,
        probability,
        str(card.get("id") or ""),
    )


@router.get("/best/{target_date}")
def get_best_match_reads(
    target_date: str,
    limit: int = Query(default=5, ge=1, le=5, description="Maximum fixture reads to return"),
) -> Dict[str, Any]:
    """Return the public, capped cross-league Match Read shortlist.

    Only already-persisted recommended cards participate.  No-bet and
    unavailable cards remain visible on their league boards, but are never
    disguised as a "best" opportunity.
    """
    matchday = _parse_matchday(target_date)
    service = _get_match_read_service()
    cards: List[Dict[str, Any]] = []
    for league in PUBLIC_MATCH_READ_LEAGUES:
        cards.extend(_effective_cards(_load_matchday_reads(
            service,
            league=league,
            target_date=matchday,
        )))
    recommended = [card for card in cards if card.get("status") == "recommended"]
    recommended.sort(key=_core_rank, reverse=True)
    return {
        "schema_version": MATCH_READ_BOARD_SCHEMA_VERSION,
        "date": matchday.isoformat(),
        "limit": limit,
        "available_count": len(recommended),
        "cards": recommended[:limit],
    }


@router.get("/fixtures/{fixture_id}")
def get_match_read_fixture(fixture_id: str) -> Dict[str, Any]:
    """Return the effective public card plus its immutable amendment history."""
    normalised_id = str(fixture_id or "").strip()
    if not normalised_id:
        raise HTTPException(status_code=400, detail="fixture_id is required.")
    service = _get_match_read_service()
    try:
        reads = service.list_delivered_for_fixture(
            normalised_id,
            surface="website",
            limit=100,
        )
    except Exception as exc:
        logger.exception("Match Read store failed for fixture %s", normalised_id)
        raise HTTPException(
            status_code=503,
            detail="Persisted Match Reads are temporarily unavailable.",
        ) from exc
    if not reads:
        raise HTTPException(status_code=404, detail="No persisted Match Read exists for this fixture.")

    effective = select_effective_match_read(reads)
    if effective is None:
        raise HTTPException(status_code=404, detail="No delivery-supported Match Read exists for this fixture.")
    try:
        effective_card = _card_from_read(effective)
        history = [_card_from_read(read) for read in sorted(reads, key=_history_key, reverse=True)]
    except ValueError as exc:
        logger.warning("Unreadable Match Read history for fixture %s: %s", normalised_id, exc)
        raise HTTPException(status_code=503, detail="Persisted Match Read is not deliverable.") from exc

    return {
        "schema_version": MATCH_READ_DETAIL_SCHEMA_VERSION,
        "fixture_id": normalised_id,
        "effective": effective_card,
        "history": history,
        "history_count": len(history),
    }


@router.get("/{league}/{target_date}")
def get_matchday_match_reads(league: str, target_date: str) -> Dict[str, Any]:
    """Return one public effective Match Read card per stored fixture.

    This is additive to the legacy live-calculation fixture endpoint.  An
    empty board is a valid response while the shadow pipeline has not yet
    persisted cards; GET never creates, updates, publishes, or delivers one.
    """
    league_code = str(league or "").strip()
    if not league_code:
        raise HTTPException(status_code=400, detail="league is required.")
    matchday = _parse_matchday(target_date)
    service = _get_match_read_service()
    reads = _load_matchday_reads(service, league=league_code, target_date=matchday)
    cards = _effective_cards(reads)
    return {
        "schema_version": MATCH_READ_BOARD_SCHEMA_VERSION,
        "league": league_code,
        "date": matchday.isoformat(),
        "cards": cards,
        "count": len(cards),
    }
