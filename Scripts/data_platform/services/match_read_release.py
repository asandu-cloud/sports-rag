"""Explicit public-release policy for persisted Match Reads.

Creating a Match Read is research/shadow work.  This service is the separate
operator boundary that promotes an already-persisted, current Match Read to a
public website card and the official published-selection cohort.  It is never
called by a website GET handler.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .match_read_delivery import MatchReadDeliveryService
from .match_reads import MatchReadService


DEFAULT_MATCH_READ_MAX_AGE_MINUTES = 120
_EFFECTIVE_STAGES = ("confirmed_lineups", "pre_match")


@dataclass(frozen=True)
class MatchReadReleaseAssessment:
    """Whether one immutable Match Read may be shown as a current card."""

    eligible: bool
    reason: Optional[str]
    evaluated_at: Optional[datetime]
    kickoff: Optional[datetime]


class MatchReadReleaseError(ValueError):
    """Raised for invalid release inputs rather than an ineligible card."""


def select_effective_match_read(reads: Iterable[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Apply the approved amendment policy to one fixture's history.

    A verified confirmed-lineups version takes precedence over a pre-match
    version.  Within a stage, the highest immutable version wins.  The caller
    decides whether it is considering all stored research rows or only rows
    already released to a particular surface.
    """
    snapshots = [dict(read) for read in reads if isinstance(read, Mapping)]
    for stage in _EFFECTIVE_STAGES:
        candidates = [read for read in snapshots if str(read.get("stage") or "") == stage]
        if candidates:
            return max(candidates, key=_revision_key)
    return None


def select_releasable_match_reads(
    reads: Iterable[Mapping[str, Any]],
    *,
    now: Optional[Any] = None,
    max_age_minutes: int = DEFAULT_MATCH_READ_MAX_AGE_MINUTES,
) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Choose current public reads from an arbitrary fixture-history set.

    This is the shared delivery safeguard for the website and Discord.  The
    caller can pass raw persisted history, a surface-specific subset, or an
    already-effective list; the result is always at most one *current* card
    per fixture.  ``skipped`` retains operational reasons without exposing a
    stale or started selection as an actionable card.
    """
    reference_now = _as_utc(now) if now is not None else datetime.now(timezone.utc)
    by_fixture: Dict[str, list[Mapping[str, Any]]] = {}
    skipped: list[Dict[str, Any]] = []
    for read in reads:
        if not isinstance(read, Mapping):
            continue
        fixture = read.get("fixture") if isinstance(read.get("fixture"), Mapping) else {}
        fixture_id = str(fixture.get("event_id") or "").strip()
        if not fixture_id:
            skipped.append({
                "fixture_id": None,
                "match_read_id": read.get("id"),
                "reason": "Fixture event ID is unavailable.",
            })
            continue
        by_fixture.setdefault(fixture_id, []).append(read)

    eligible: list[Dict[str, Any]] = []
    for fixture_id, fixture_reads in by_fixture.items():
        effective = select_effective_match_read(fixture_reads)
        if effective is None:
            skipped.append({"fixture_id": fixture_id, "reason": "No supported Match Read stage."})
            continue
        assessment = assess_match_read_release(
            effective,
            now=reference_now,
            max_age_minutes=max_age_minutes,
        )
        if not assessment.eligible:
            skipped.append({
                "fixture_id": fixture_id,
                "match_read_id": effective.get("id"),
                "reason": assessment.reason,
            })
            continue
        eligible.append(dict(effective))

    return sorted(eligible, key=_fixture_sort_key), skipped


def assess_match_read_release(
    read: Mapping[str, Any],
    *,
    now: Optional[Any] = None,
    max_age_minutes: int = DEFAULT_MATCH_READ_MAX_AGE_MINUTES,
) -> MatchReadReleaseAssessment:
    """Reject stale or already-started cards before a public release.

    A published price should never remain actionable after kickoff or after a
    long gap since the evaluated canonical snapshot.  This policy is kept
    independent of the compiler so historical/shadow records remain fully
    auditable even when they are no longer eligible to be public picks.
    """
    try:
        age_limit = int(max_age_minutes)
    except (TypeError, ValueError) as exc:
        raise MatchReadReleaseError("max_age_minutes must be an integer.") from exc
    if age_limit < 1:
        raise MatchReadReleaseError("max_age_minutes must be at least 1.")

    reference_now = _as_utc(now) if now is not None else datetime.now(timezone.utc)
    fixture = read.get("fixture") if isinstance(read.get("fixture"), Mapping) else {}
    kickoff = _parse_timestamp(fixture.get("kickoff"))
    evaluated = _read_evaluated_at(read)
    if kickoff is None:
        return MatchReadReleaseAssessment(False, "Fixture kickoff is unavailable.", evaluated, kickoff)
    if evaluated is None:
        return MatchReadReleaseAssessment(False, "Match Read evaluation time is unavailable.", evaluated, kickoff)
    if kickoff <= reference_now:
        return MatchReadReleaseAssessment(False, "Fixture has started or finished.", evaluated, kickoff)
    if evaluated > reference_now + timedelta(minutes=5):
        return MatchReadReleaseAssessment(False, "Match Read evaluation time is in the future.", evaluated, kickoff)
    if reference_now - evaluated > timedelta(minutes=age_limit):
        return MatchReadReleaseAssessment(
            False,
            f"Match Read is older than the {age_limit}-minute publication limit.",
            evaluated,
            kickoff,
        )
    return MatchReadReleaseAssessment(True, None, evaluated, kickoff)


class MatchReadReleaseService:
    """Promote selected, current Match Reads to the public website deliberately."""

    def __init__(
        self,
        *,
        match_reads: Optional[MatchReadService] = None,
        deliveries: Optional[MatchReadDeliveryService] = None,
    ) -> None:
        self._match_reads = match_reads or MatchReadService()
        self._deliveries = deliveries or MatchReadDeliveryService(match_reads=self._match_reads)

    def release_matchday_to_website(
        self,
        *,
        league: str,
        target_date: date | str,
        now: Optional[Any] = None,
        max_age_minutes: int = DEFAULT_MATCH_READ_MAX_AGE_MINUTES,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Release the effective read for each fixture in one league/day.

        This is intentionally explicit and idempotent.  It does not generate
        a new read; a caller must first create/review shadow records.  A
        website page load never invokes it.
        """
        normalised_league = str(league or "").strip()
        if not normalised_league:
            raise MatchReadReleaseError("league is required.")
        matchday = _matchday_date(target_date)
        stored = self._match_reads.list_for_matchday(
            league=normalised_league,
            target_date=matchday,
        )
        releasable, skipped = select_releasable_match_reads(
            stored,
            now=now,
            max_age_minutes=max_age_minutes,
        )

        released = []
        for effective in releasable:
            fixture = effective.get("fixture") if isinstance(effective.get("fixture"), Mapping) else {}
            fixture_id = str(fixture.get("event_id") or "").strip()
            # ``select_releasable_match_reads`` validates this identity, but
            # retain the guard here because this method creates side effects.
            if not fixture_id:
                skipped.append({
                    "fixture_id": None,
                    "match_read_id": effective.get("id"),
                    "reason": "Fixture event ID is unavailable.",
                })
                continue

            metadata = {
                "publication_kind": "match_read_website_board",
                "league": normalised_league,
                "target_date": matchday.isoformat(),
            }
            external_reference = (
                f"website:match-read:{normalised_league}:{matchday.isoformat()}:{fixture_id}"
            )
            if dry_run:
                released.append({
                    "fixture_id": fixture_id,
                    "match_read_id": int(effective["id"]),
                    "would_release": True,
                })
                continue
            result = self._deliveries.record_visible(
                int(effective["id"]),
                surface="website",
                external_reference=external_reference,
                delivered_at=now,
                metadata=metadata,
            )
            released.append({"fixture_id": fixture_id, **result})

        return {
            "league": normalised_league,
            "date": matchday.isoformat(),
            "dry_run": bool(dry_run),
            "stored_read_count": len(stored),
            "fixture_count": len({
                str((read.get("fixture") or {}).get("event_id") or "").strip()
                for read in stored
                if isinstance(read, Mapping) and isinstance(read.get("fixture"), Mapping)
                and str((read.get("fixture") or {}).get("event_id") or "").strip()
            }),
            "released": released,
            "skipped": skipped,
        }


def _revision_key(read: Mapping[str, Any]) -> tuple[int, float, float, int]:
    try:
        version = int(read.get("version") or 0)
    except (TypeError, ValueError):
        version = 0
    try:
        record_id = int(read.get("id") or 0)
    except (TypeError, ValueError):
        record_id = 0
    evaluated = _read_evaluated_at(read)
    created = _parse_timestamp(read.get("created_at"))
    return (
        version,
        evaluated.timestamp() if evaluated is not None else float("-inf"),
        created.timestamp() if created is not None else float("-inf"),
        record_id,
    )


def _read_evaluated_at(read: Mapping[str, Any]) -> Optional[datetime]:
    provenance = read.get("provenance") if isinstance(read.get("provenance"), Mapping) else {}
    return _parse_timestamp(provenance.get("evaluated_at")) or _parse_timestamp(read.get("created_at"))


def _fixture_sort_key(read: Mapping[str, Any]) -> tuple[float, str]:
    fixture = read.get("fixture") if isinstance(read.get("fixture"), Mapping) else {}
    kickoff = _parse_timestamp(fixture.get("kickoff"))
    return (
        kickoff.timestamp() if kickoff is not None else float("inf"),
        str(fixture.get("event_id") or ""),
    )


def _matchday_date(value: date | str) -> date:
    if isinstance(value, datetime):
        raise MatchReadReleaseError("target_date must be a date or YYYY-MM-DD string, not a datetime.")
    if isinstance(value, date):
        return value
    raw = str(value or "").strip()
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise MatchReadReleaseError("target_date must use YYYY-MM-DD format.") from exc
    if raw != parsed.isoformat():
        raise MatchReadReleaseError("target_date must use YYYY-MM-DD format.")
    return parsed


def _parse_timestamp(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _as_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    parsed = _parse_timestamp(value)
    if parsed is None:
        raise MatchReadReleaseError("now must be an ISO-8601 timestamp.")
    return parsed.astimezone(timezone.utc)
