"""Persistence for versioned fixture-level Match Reads.

Match Reads are intentionally separate from the legacy prediction/parlay
tables: a no-bet or an unpriced coherent build is still auditable, but it is
not silently turned into a settled betting recommendation.
"""

from __future__ import annotations

from datetime import datetime, timezone
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
