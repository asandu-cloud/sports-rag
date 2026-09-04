"""Explicit publication bridge for an already delivered Match Read.

Generation and persistence are deliberately not publication.  A Match Read is
only entered into the official tracked cohort after a delivery surface has
actually rendered it to users and calls this bridge with that immutable
external reference.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .match_reads import MatchReadService, MatchReadValidationError
from .publications import PublicationService


class MatchReadDeliveryError(ValueError):
    """Raised when a visible Match Read cannot be linked to tracking safely."""


class MatchReadDeliveryService:
    """Record a real delivery and link its visible selections to tracking.

    The caller must invoke this *after* Discord/API/web delivery succeeds.  It
    is intentionally not called by compilation, persistence, page reads, or
    slash-command previews.  That keeps shadow candidates and a user's page
    refresh out of the official ROI cohort.
    """

    def __init__(
        self,
        *,
        match_reads: Optional[MatchReadService] = None,
        publications: Optional[PublicationService] = None,
    ) -> None:
        self._match_reads = match_reads or MatchReadService()
        self._publications = publications or PublicationService()

    def record_visible(
        self,
        match_read_id: int,
        *,
        surface: str,
        external_reference: Optional[str],
        delivered_at: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a Match Read delivery and link its official leaf releases.

        No-bet and unavailable reads are still recorded as visible cards but
        have no underlying published recommendation.  Recommended Match Reads
        publish their explicitly selected leaves through ``PublicationService``
        and attach the resulting immutable recommendation IDs back to the
        selections.  All operations are idempotent, so a retry after an
        intermittent platform failure is safe.
        """
        read = self._match_reads.get(int(match_read_id))
        if read is None:
            raise MatchReadDeliveryError(f"Unknown Match Read {match_read_id}.")

        delivery_metadata = {
            "match_read_id": int(read["id"]),
            "match_read_version": int(read["version"]),
            "match_read_stage": read["stage"],
            "match_read_status": read["status"],
        }
        if metadata:
            delivery_metadata.update(dict(metadata))
        try:
            delivery = self._match_reads.record_delivery(
                int(read["id"]),
                surface=surface,
                external_reference=external_reference,
                delivered_at=delivered_at,
                metadata=delivery_metadata,
            )
        except MatchReadValidationError as exc:
            raise MatchReadDeliveryError(str(exc)) from exc

        publications_created = 0
        publication_deliveries_created = 0
        links_created = 0
        for selection in read["selections"]:
            # A read's immutable validation already guarantees a selected leaf
            # is a priced canonical recommendation.  Retain this explicit
            # check for a clear error if a legacy/corrupt row is ever loaded.
            decision = _decision(selection.get("data"))
            if decision.get("status") != "recommended":
                raise MatchReadDeliveryError(
                    f"Match Read {read['id']} selection {selection['position']} is not publishable."
                )
            published = self._publications.publish(
                selection["data"],
                surface=surface,
                external_reference=external_reference,
                published_at=delivered_at,
                delivery_metadata={
                    **delivery_metadata,
                    "match_read_selection_position": int(selection["position"]),
                    "match_read_selection_role": selection["role"],
                },
            )
            publications_created += int(bool(published.get("created")))
            publication_deliveries_created += int(bool(published.get("delivery_created")))
            try:
                linked = self._match_reads.link_published_recommendation(
                    int(read["id"]),
                    selection_position=int(selection["position"]),
                    published_recommendation_id=int(published["recommendation_id"]),
                )
            except ValueError as exc:
                raise MatchReadDeliveryError(str(exc)) from exc
            links_created += int(bool(linked))

        return {
            "match_read_id": int(read["id"]),
            "match_read_delivery_id": int(delivery["id"]),
            "delivery_created": bool(delivery["created"]),
            "recommendations_created": publications_created,
            "recommendation_deliveries_created": publication_deliveries_created,
            "selection_links_created": links_created,
        }


def _decision(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    decision = value.get("decision")
    return decision if isinstance(decision, Mapping) else {}
