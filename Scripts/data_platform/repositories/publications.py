"""Persistence for immutable canonical recommendation publication records."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Optional

from sqlalchemy import select

from ..db import session_scope
from ..models import Prediction, PublishedRecommendation, RecommendationDelivery


class PublicationRepository:
    """Create publication records without ever rewriting a released decision."""

    def __init__(self, session_factory=session_scope):
        self._factory = session_factory

    def record(
        self,
        *,
        recommendation_key: str,
        prediction_fields: Mapping[str, Any],
        released_at: datetime,
        input_snapshot_id: str,
        pipeline_version: str,
        model_version: Optional[str],
        decision_json: Mapping[str, Any],
        delivery_key: str,
        surface: str,
        external_reference: Optional[str],
        delivered_at: datetime,
        delivery_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Atomically create-or-find a release and create-or-find its delivery.

        A retry with the same keys returns the original rows unchanged.  In
        particular, it does not overwrite the price, probability, evidence,
        or input snapshot that was actually published.
        """
        with self._factory() as session:
            recommendation = session.scalar(
                select(PublishedRecommendation).where(
                    PublishedRecommendation.recommendation_key == recommendation_key
                )
            )
            created = recommendation is None
            if recommendation is None:
                prediction = Prediction(**dict(prediction_fields))
                session.add(prediction)
                session.flush()
                recommendation = PublishedRecommendation(
                    prediction_id=int(prediction.id),
                    recommendation_key=recommendation_key,
                    released_at=released_at,
                    input_snapshot_id=input_snapshot_id,
                    pipeline_version=pipeline_version,
                    model_version=model_version,
                    decision_json=dict(decision_json),
                )
                session.add(recommendation)
                session.flush()

            delivery = session.scalar(
                select(RecommendationDelivery).where(
                    RecommendationDelivery.delivery_key == delivery_key
                )
            )
            delivery_created = delivery is None
            if delivery is None:
                delivery = RecommendationDelivery(
                    recommendation_id=int(recommendation.id),
                    delivery_key=delivery_key,
                    surface=surface,
                    external_reference=external_reference,
                    delivered_at=delivered_at,
                    metadata_json=dict(delivery_metadata) if delivery_metadata else None,
                )
                session.add(delivery)
                session.flush()

            return {
                "recommendation_id": int(recommendation.id),
                "prediction_id": int(recommendation.prediction_id),
                "recommendation_key": recommendation.recommendation_key,
                "delivery_id": int(delivery.id),
                "created": created,
                "delivery_created": delivery_created,
            }
