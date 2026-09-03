"""Immutable records for recommendations that were actually released.

``Prediction`` is the settlement record for one betting leg.  These models
add the missing publication boundary: a canonical recommendation is recorded
once with its exact pre-match snapshot, and every Discord/website delivery is
then attached as a separate immutable event.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin
from .json_type import JSONB


class PublishedRecommendation(Base, TimestampMixin):
    """The single authoritative record of a released canonical decision."""

    __tablename__ = "published_recommendations"
    __table_args__ = (
        UniqueConstraint("prediction_id", name="uq_published_recommendation_prediction"),
        UniqueConstraint("recommendation_key", name="uq_published_recommendation_key"),
        Index("ix_published_recommendations_released_at", "released_at"),
        Index("ix_published_recommendations_snapshot", "input_snapshot_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id", ondelete="RESTRICT"), nullable=False
    )
    # Deterministic from the fixture, priced selection, and immutable input
    # snapshot. It makes retries and cross-surface publication idempotent.
    recommendation_key: Mapped[str] = mapped_column(String(64), nullable=False)
    released_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    input_snapshot_id: Mapped[str] = mapped_column(String(256), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(128))
    decision_json: Mapped[dict] = mapped_column(JSONB, nullable=False)


class RecommendationDelivery(Base, TimestampMixin):
    """One immutable delivery of a published recommendation to a surface."""

    __tablename__ = "recommendation_deliveries"
    __table_args__ = (
        UniqueConstraint("delivery_key", name="uq_recommendation_delivery_key"),
        Index("ix_recommendation_deliveries_recommendation", "recommendation_id"),
        Index("ix_recommendation_deliveries_surface", "surface"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recommendation_id: Mapped[int] = mapped_column(
        ForeignKey("published_recommendations.id", ondelete="RESTRICT"), nullable=False
    )
    # The delivery key is deterministic when an external message/page ID is
    # known, otherwise it is one idempotent delivery per recommendation/surface.
    delivery_key: Mapped[str] = mapped_column(String(64), nullable=False)
    surface: Mapped[str] = mapped_column(String(32), nullable=False)
    external_reference: Mapped[Optional[str]] = mapped_column(String(256))
    delivered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB)
