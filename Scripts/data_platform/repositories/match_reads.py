"""Persistence for versioned fixture-level Match Reads.

Match Reads are intentionally separate from the legacy prediction/parlay
tables: a no-bet or an unpriced coherent build is still auditable, but it is
not silently turned into a settled betting recommendation.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError

from ..db import session_scope
from ..models import (
    MatchRead,
    MatchReadDelivery,
    MatchReadPackage,
    MatchReadPackageSelection,
    MatchReadSelection,
)
from ..outcomes import normalize_outcome


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _matchday_date(value: date | str) -> date:
    """Return a validated calendar date used to select fixture kickoffs.

    Match Read kickoffs are retained as the source ISO-8601 string, rather
    than converted to a database timestamp.  A board query therefore uses
    the calendar date encoded in that ISO value; it deliberately does not
    apply a new timezone conversion policy at read time.
    """
    if isinstance(value, datetime):
        raise ValueError("matchday date must be a date or YYYY-MM-DD string, not a datetime.")
    if isinstance(value, date):
        return value
    raw = str(value).strip()
    try:
        parsed = date.fromisoformat(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("matchday date must be a YYYY-MM-DD ISO date.") from exc
    if raw != parsed.isoformat():
        raise ValueError("matchday date must be a YYYY-MM-DD ISO date.")
    return parsed


def _kickoff_calendar_date(value: Optional[str]) -> Optional[date]:
    """Extract the calendar date written in a stored ISO-8601 kickoff.

    ``datetime.fromisoformat`` preserves the offset carried by the source
    value, so ``2026-09-07T00:30:00+02:00`` remains a September 7 fixture
    here rather than being silently converted to its UTC calendar date.
    Invalid legacy values are excluded from a matchday query rather than
    causing an otherwise valid board to fail.
    """
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None


class MatchReadRepository:
    """Create immutable Match Read versions and their user-visible deliveries."""

    def __init__(self, session_factory=session_scope):
        self._factory = session_factory

    def record(
        self,
        *,
        read_fields: Mapping[str, Any],
        selections: Sequence[Mapping[str, Any]],
        packages: Sequence[Mapping[str, Any]] = (),
    ) -> Dict[str, Any]:
        """Create or find an immutable Match Read snapshot.

        A retry of the exact payload returns the existing row. A changed
        payload becomes the next version in its fixture/stage stream and
        records the prior version as ``supersedes_id``.
        """
        fields = dict(read_fields)
        read_key = str(fields["read_key"])
        fixture_api_id = str(fields["fixture_api_id"])
        stage = str(fields["stage"])

        with self._factory() as session:
            existing = session.scalar(
                select(MatchRead).where(MatchRead.read_key == read_key)
            )
            if existing is not None:
                result = _read_to_dict(session, existing)
                result["created"] = False
                return result

            previous = session.scalar(
                select(MatchRead)
                .where(
                    MatchRead.fixture_api_id == fixture_api_id,
                    MatchRead.stage == stage,
                )
                .order_by(desc(MatchRead.version), desc(MatchRead.id))
                .limit(1)
            )
            fields["version"] = int(previous.version) + 1 if previous is not None else 1
            fields["supersedes_id"] = int(previous.id) if previous is not None else None
            read = MatchRead(**fields)
            session.add(read)
            try:
                session.flush()
            except IntegrityError:
                # A scheduler retry can race the first delivery attempt. The
                # deterministic read key lets us return the winning immutable
                # row instead of creating a duplicate version or failing a
                # harmless retry.
                session.rollback()
                existing = session.scalar(
                    select(MatchRead).where(MatchRead.read_key == read_key)
                )
                if existing is None:
                    raise
                result = _read_to_dict(session, existing)
                result["created"] = False
                return result

            selection_by_position: Dict[int, MatchReadSelection] = {}
            for selection_fields in selections:
                selection = MatchReadSelection(
                    match_read_id=int(read.id),
                    **dict(selection_fields),
                )
                session.add(selection)
                session.flush()
                selection_by_position[int(selection.position)] = selection

            for package_payload in packages:
                package_fields = dict(package_payload)
                selection_positions = list(package_fields.pop("selection_positions"))
                package = MatchReadPackage(
                    match_read_id=int(read.id),
                    **package_fields,
                )
                session.add(package)
                session.flush()
                for position, selection_position in enumerate(selection_positions, start=1):
                    selection = selection_by_position[int(selection_position)]
                    session.add(MatchReadPackageSelection(
                        package_id=int(package.id),
                        selection_id=int(selection.id),
                        position=position,
                    ))

            session.flush()
            result = _read_to_dict(session, read)
            result["created"] = True
            return result

    def get(self, match_read_id: int) -> Optional[Dict[str, Any]]:
        with self._factory() as session:
            read = session.get(MatchRead, match_read_id)
            return _read_to_dict(session, read) if read is not None else None

    def list_for_fixture(
        self,
        fixture_api_id: str,
        *,
        stage: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        with self._factory() as session:
            stmt = select(MatchRead).where(MatchRead.fixture_api_id == str(fixture_api_id))
            if stage is not None:
                stmt = stmt.where(MatchRead.stage == str(stage))
            rows = session.scalars(
                stmt.order_by(desc(MatchRead.evaluated_at), desc(MatchRead.id)).limit(limit)
            ).all()
            return [_read_to_dict(session, row) for row in rows]

    def list_for_matchday(
        self,
        *,
        league: str,
        target_date: date | str,
    ) -> List[Dict[str, Any]]:
        """Return every stored read/version/stage for one league calendar date.

        This is intentionally a raw historical query.  It does *not* choose
        an effective read when pre-match and confirmed-lineup versions both
        exist; the delivery layer must make that product decision explicitly.
        """
        league_code = str(league or "").strip()
        if not league_code:
            raise ValueError("league is required.")
        matchday = _matchday_date(target_date)
        matchday_text = matchday.isoformat()

        with self._factory() as session:
            # Valid ISO kickoffs begin with their own source calendar date.
            # Keep the indexed league/kickoff predicate narrow, then parse
            # the stored value below as a safeguard against malformed legacy
            # strings that merely share the same prefix.
            rows = session.scalars(
                select(MatchRead)
                .where(
                    MatchRead.league == league_code,
                    MatchRead.kickoff.startswith(matchday_text),
                )
                .order_by(
                    MatchRead.kickoff.asc(),
                    MatchRead.fixture_api_id.asc(),
                    MatchRead.stage.asc(),
                    MatchRead.version.asc(),
                    MatchRead.id.asc(),
                )
            ).all()
            return [
                _read_to_dict(session, row)
                for row in rows
                if _kickoff_calendar_date(row.kickoff) == matchday
            ]

    def list_delivered_for_matchday(
        self,
        *,
        league: str,
        target_date: date | str,
        surface: str,
    ) -> List[Dict[str, Any]]:
        """Return Match Reads released at least once on one named surface.

        A Match Read can have multiple delivery rows (for example, a retry or
        an amended Discord hub), but the returned read appears once.  This is
        the explicit public-release boundary: a persisted shadow candidate is
        not returned until a delivery adapter has recorded it as visible.
        """
        league_code = str(league or "").strip()
        if not league_code:
            raise ValueError("league is required.")
        surface_name = str(surface or "").strip().lower()
        if not surface_name:
            raise ValueError("surface is required.")
        matchday = _matchday_date(target_date)
        matchday_text = matchday.isoformat()

        with self._factory() as session:
            rows = session.scalars(
                select(MatchRead)
                .join(MatchReadDelivery, MatchReadDelivery.match_read_id == MatchRead.id)
                .where(
                    MatchRead.league == league_code,
                    MatchRead.kickoff.startswith(matchday_text),
                    MatchReadDelivery.surface == surface_name,
                )
                .distinct()
                .order_by(
                    MatchRead.kickoff.asc(),
                    MatchRead.fixture_api_id.asc(),
                    MatchRead.stage.asc(),
                    MatchRead.version.asc(),
                    MatchRead.id.asc(),
                )
            ).all()
            return [
                _read_to_dict(session, row)
                for row in rows
                if _kickoff_calendar_date(row.kickoff) == matchday
            ]

    def list_delivered_for_fixture(
        self,
        fixture_api_id: str,
        *,
        surface: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return only versions of one fixture released on ``surface``."""
        fixture_id = str(fixture_api_id or "").strip()
        if not fixture_id:
            raise ValueError("fixture_api_id is required.")
        surface_name = str(surface or "").strip().lower()
        if not surface_name:
            raise ValueError("surface is required.")
        try:
            row_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be an integer.") from exc
        if row_limit < 1:
            raise ValueError("limit must be at least 1.")

        with self._factory() as session:
            rows = session.scalars(
                select(MatchRead)
                .join(MatchReadDelivery, MatchReadDelivery.match_read_id == MatchRead.id)
                .where(
                    MatchRead.fixture_api_id == fixture_id,
                    MatchReadDelivery.surface == surface_name,
                )
                .distinct()
                .order_by(desc(MatchRead.evaluated_at), desc(MatchRead.id))
                .limit(row_limit)
            ).all()
            return [_read_to_dict(session, row) for row in rows]

    def record_delivery(
        self,
        match_read_id: int,
        *,
        delivery_key: str,
        surface: str,
        external_reference: Optional[str],
        delivered_at: datetime,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a public delivery without mutating the immutable Match Read."""
        with self._factory() as session:
            if session.get(MatchRead, match_read_id) is None:
                raise ValueError(f"Unknown Match Read {match_read_id}.")
            delivery = session.scalar(
                select(MatchReadDelivery).where(MatchReadDelivery.delivery_key == delivery_key)
            )
            created = delivery is None
            if delivery is None:
                delivery = MatchReadDelivery(
                    match_read_id=match_read_id,
                    delivery_key=delivery_key,
                    surface=surface,
                    external_reference=external_reference,
                    delivered_at=delivered_at,
                    metadata_json=dict(metadata) if metadata else None,
                )
                session.add(delivery)
                session.flush()
            return {
                "id": int(delivery.id),
                "match_read_id": int(delivery.match_read_id),
                "created": created,
            }

    def find_latest_delivery(
        self,
        *,
        surface: str,
        metadata: Mapping[str, Any],
        limit: int = 500,
    ) -> Optional[Dict[str, Any]]:
        """Find the newest delivery whose stored metadata contains ``metadata``.

        Delivery metadata deliberately remains schemaless because it describes
        an external surface, not the immutable Match Read itself.  Querying
        JSON directly would make this small operational lookup vary between
        SQLite and a future hosted database, so scan a bounded, newest-first
        set instead.  Matchday hubs have only a handful of Discord delivery
        rows; ``limit`` is a safety boundary, not a normal pagination API.
        """
        normalised_surface = str(surface or "").strip().lower()
        if not normalised_surface:
            raise ValueError("surface is required.")
        expected = dict(metadata or {})
        if not expected:
            raise ValueError("metadata is required.")
        try:
            row_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be an integer.") from exc
        if row_limit < 1:
            raise ValueError("limit must be at least 1.")

        with self._factory() as session:
            rows = session.scalars(
                select(MatchReadDelivery)
                .where(MatchReadDelivery.surface == normalised_surface)
                .order_by(
                    desc(MatchReadDelivery.delivered_at),
                    desc(MatchReadDelivery.id),
                )
                .limit(row_limit)
            ).all()
            for delivery in rows:
                stored = delivery.metadata_json if isinstance(delivery.metadata_json, Mapping) else {}
                if all(stored.get(key) == value for key, value in expected.items()):
                    return _delivery_to_dict(delivery)
        return None

    def link_published_recommendation(
        self,
        match_read_id: int,
        *,
        selection_position: int,
        published_recommendation_id: int,
    ) -> bool:
        """Attach a deliberately released leaf recommendation to a selection."""
        with self._factory() as session:
            selection = session.scalar(
                select(MatchReadSelection).where(
                    MatchReadSelection.match_read_id == match_read_id,
                    MatchReadSelection.position == selection_position,
                )
            )
            if selection is None:
                raise ValueError(
                    f"Match Read {match_read_id} has no selection at position {selection_position}."
                )
            if selection.published_recommendation_id is None:
                selection.published_recommendation_id = published_recommendation_id
                return True
            if int(selection.published_recommendation_id) == int(published_recommendation_id):
                return False
            raise ValueError(
                "An immutable Match Read selection cannot be linked to a different publication."
            )

    def mark_package_outcome(
        self,
        package_id: int,
        *,
        outcome: str,
        settled_at: Optional[datetime] = None,
        profit_units: Optional[float] = None,
    ) -> bool:
        """Store an independently tracked package result.

        Auto-settlement is intentionally deferred until actual same-book
        package quotes and their settlement rules are wired into delivery.
        """
        canonical = normalize_outcome(outcome)
        if canonical is None:
            raise ValueError(f"Unsupported package outcome: {outcome!r}.")
        with self._factory() as session:
            package = session.get(MatchReadPackage, package_id)
            if package is None:
                return False
            package.outcome = canonical
            package.settled_at = settled_at or datetime.now(timezone.utc)
            if profit_units is not None:
                package.profit_units = profit_units
            return True


def _read_to_dict(session, read: MatchRead) -> Dict[str, Any]:
    selections = session.scalars(
        select(MatchReadSelection)
        .where(MatchReadSelection.match_read_id == read.id)
        .order_by(MatchReadSelection.position)
    ).all()
    selection_dicts = [_selection_to_dict(item) for item in selections]
    selection_by_id = {int(item.id): item for item in selections}

    packages = session.scalars(
        select(MatchReadPackage)
        .where(MatchReadPackage.match_read_id == read.id)
        .order_by(MatchReadPackage.position)
    ).all()
    package_dicts = []
    for package in packages:
        links = session.scalars(
            select(MatchReadPackageSelection)
            .where(MatchReadPackageSelection.package_id == package.id)
            .order_by(MatchReadPackageSelection.position)
        ).all()
        package_dicts.append({
            "id": int(package.id),
            "package_key": package.package_key,
            "kind": package.kind,
            "status": package.status,
            "label": package.label,
            "rationale": package.rationale,
            "position": int(package.position),
            "quoted_odds": _number(package.quoted_odds),
            "bookmaker": package.bookmaker,
            "joint_model_probability": _number(package.joint_model_probability),
            "expected_value": _number(package.expected_value),
            "stake_units": _number(package.stake_units),
            "profit_units": _number(package.profit_units),
            "outcome": package.outcome,
            "settled_at": _iso(package.settled_at),
            "data": package.package_json,
            "selection_positions": [
                int(selection_by_id[int(link.selection_id)].position)
                for link in links
                if int(link.selection_id) in selection_by_id
            ],
        })

    return {
        "id": int(read.id),
        "read_key": read.read_key,
        "fixture": {
            "event_id": read.fixture_api_id,
            "league": read.league,
            "home_team": read.home_team,
            "away_team": read.away_team,
            "kickoff": read.kickoff,
        },
        "stage": read.stage,
        "version": int(read.version),
        "supersedes_id": int(read.supersedes_id) if read.supersedes_id is not None else None,
        "status": read.status,
        "thesis": read.thesis,
        "game_script": read.game_script_json,
        "provenance": {
            "input_snapshot_id": read.input_snapshot_id,
            "pipeline_version": read.pipeline_version,
            "model_version": read.model_version,
            "evaluated_at": _iso(read.evaluated_at),
        },
        "data": read.read_json,
        "selections": selection_dicts,
        "packages": package_dicts,
        "created_at": _iso(read.created_at),
    }


def _selection_to_dict(selection: MatchReadSelection) -> Dict[str, Any]:
    return {
        "id": int(selection.id),
        "published_recommendation_id": (
            int(selection.published_recommendation_id)
            if selection.published_recommendation_id is not None else None
        ),
        "role": selection.role,
        "position": int(selection.position),
        "market": {"group": selection.market_group, "key": selection.market_key},
        "pick": selection.pick,
        "quote_odds": _number(selection.quote_odds),
        "bookmaker": selection.bookmaker,
        "model_probability": _number(selection.model_probability),
        "value_edge": _number(selection.value_edge),
        "data": selection.selection_json,
    }


def _delivery_to_dict(delivery: MatchReadDelivery) -> Dict[str, Any]:
    """Project a delivery lookup row without loading its full Match Read."""
    return {
        "id": int(delivery.id),
        "match_read_id": int(delivery.match_read_id),
        "delivery_key": delivery.delivery_key,
        "surface": delivery.surface,
        "external_reference": delivery.external_reference,
        "delivered_at": _iso(delivery.delivered_at),
        "metadata": dict(delivery.metadata_json) if isinstance(delivery.metadata_json, Mapping) else {},
        "created_at": _iso(delivery.created_at),
    }
