"""V6: immutable canonical recommendation publication records.

Revision ID: 0006_published_recommendations
Revises: 0005_feature_lineage
Create Date: 2026-09-03 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from data_platform.models.json_type import JSONB


revision = "0006_published_recommendations"
down_revision = "0005_feature_lineage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "published_recommendations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("prediction_id", sa.Integer, sa.ForeignKey("predictions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("recommendation_key", sa.String(64), nullable=False),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("input_snapshot_id", sa.String(256), nullable=False),
        sa.Column("pipeline_version", sa.String(128), nullable=False),
        sa.Column("model_version", sa.String(128)),
        sa.Column("decision_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("prediction_id", name="uq_published_recommendation_prediction"),
        sa.UniqueConstraint("recommendation_key", name="uq_published_recommendation_key"),
    )
    op.create_index("ix_published_recommendations_released_at", "published_recommendations", ["released_at"])
    op.create_index("ix_published_recommendations_snapshot", "published_recommendations", ["input_snapshot_id"])

    op.create_table(
        "recommendation_deliveries",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("recommendation_id", sa.Integer, sa.ForeignKey("published_recommendations.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("delivery_key", sa.String(64), nullable=False),
        sa.Column("surface", sa.String(32), nullable=False),
        sa.Column("external_reference", sa.String(256)),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata_json", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("delivery_key", name="uq_recommendation_delivery_key"),
    )
    op.create_index("ix_recommendation_deliveries_recommendation", "recommendation_deliveries", ["recommendation_id"])
    op.create_index("ix_recommendation_deliveries_surface", "recommendation_deliveries", ["surface"])


def downgrade() -> None:
    op.drop_table("recommendation_deliveries")
    op.drop_table("published_recommendations")
