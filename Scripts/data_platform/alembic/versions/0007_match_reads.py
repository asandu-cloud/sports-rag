"""V7: versioned fixture-level Match Read records and package tracking.

Revision ID: 0007_match_reads
Revises: 0006_published_recommendations
Create Date: 2026-09-03 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from data_platform.models.json_type import JSONB


revision = "0007_match_reads"
down_revision = "0006_published_recommendations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "match_reads",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("read_key", sa.String(64), nullable=False),
        sa.Column("fixture_api_id", sa.String(64), nullable=False),
        sa.Column("league", sa.String(32), nullable=False),
        sa.Column("home_team", sa.String(128), nullable=False),
        sa.Column("away_team", sa.String(128), nullable=False),
        sa.Column("kickoff", sa.String(64)),
        sa.Column("stage", sa.String(32), nullable=False, server_default="pre_match"),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("supersedes_id", sa.Integer, sa.ForeignKey("match_reads.id", ondelete="RESTRICT")),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("thesis", sa.Text, nullable=False),
        sa.Column("game_script_json", JSONB),
        sa.Column("input_snapshot_id", sa.String(256), nullable=False),
        sa.Column("pipeline_version", sa.String(128), nullable=False),
        sa.Column("model_version", sa.String(128)),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("read_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("read_key", name="uq_match_reads_key"),
        sa.UniqueConstraint("fixture_api_id", "stage", "version", name="uq_match_reads_fixture_stage_version"),
    )
    op.create_index("ix_match_reads_fixture_stage", "match_reads", ["fixture_api_id", "stage", "version"])
    op.create_index("ix_match_reads_league_kickoff", "match_reads", ["league", "kickoff"])
    op.create_index("ix_match_reads_status", "match_reads", ["status"])

    op.create_table(
        "match_read_selections",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("match_read_id", sa.Integer, sa.ForeignKey("match_reads.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "published_recommendation_id",
            sa.Integer,
            sa.ForeignKey("published_recommendations.id", ondelete="RESTRICT"),
        ),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("market_group", sa.String(32), nullable=False),
        sa.Column("market_key", sa.String(128), nullable=False),
        sa.Column("pick", sa.String(255), nullable=False),
        sa.Column("quote_odds", sa.Numeric(10, 4), nullable=False),
        sa.Column("bookmaker", sa.String(64)),
        sa.Column("model_probability", sa.Numeric(10, 6)),
        sa.Column("value_edge", sa.Numeric(10, 6)),
        sa.Column("selection_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("match_read_id", "position", name="uq_match_read_selection_position"),
    )
    op.create_index("ix_match_read_selections_read", "match_read_selections", ["match_read_id", "position"])
    op.create_index("ix_match_read_selections_published", "match_read_selections", ["published_recommendation_id"])
    op.create_index("ix_match_read_selections_role", "match_read_selections", ["role"])
    op.create_index("ix_match_read_selections_market", "match_read_selections", ["market_group", "market_key"])

    op.create_table(
        "match_read_packages",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("match_read_id", sa.Integer, sa.ForeignKey("match_reads.id", ondelete="CASCADE"), nullable=False),
        sa.Column("package_key", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("label", sa.String(128), nullable=False),
        sa.Column("rationale", sa.Text),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("quoted_odds", sa.Numeric(10, 4), nullable=False),
        sa.Column("bookmaker", sa.String(64), nullable=False),
        sa.Column("joint_model_probability", sa.Numeric(10, 6)),
        sa.Column("expected_value", sa.Numeric(10, 6)),
        sa.Column("stake_units", sa.Numeric(10, 4), nullable=False, server_default="1.0"),
        sa.Column("profit_units", sa.Numeric(10, 4)),
        sa.Column("outcome", sa.String(16)),
        sa.Column("settled_at", sa.DateTime(timezone=True)),
        sa.Column("package_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("package_key", name="uq_match_read_packages_key"),
        sa.UniqueConstraint("match_read_id", "position", name="uq_match_read_package_position"),
    )
    op.create_index("ix_match_read_packages_read", "match_read_packages", ["match_read_id", "position"])
    op.create_index("ix_match_read_packages_status", "match_read_packages", ["status"])

    op.create_table(
        "match_read_package_selections",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("package_id", sa.Integer, sa.ForeignKey("match_read_packages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("selection_id", sa.Integer, sa.ForeignKey("match_read_selections.id", ondelete="CASCADE"), nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("package_id", "selection_id", name="uq_match_read_package_selection"),
        sa.UniqueConstraint("package_id", "position", name="uq_match_read_package_selection_position"),
    )
    op.create_index("ix_match_read_package_selections_package", "match_read_package_selections", ["package_id", "position"])
    op.create_index("ix_match_read_package_selections_selection", "match_read_package_selections", ["selection_id"])

    op.create_table(
        "match_read_deliveries",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("match_read_id", sa.Integer, sa.ForeignKey("match_reads.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("delivery_key", sa.String(64), nullable=False),
        sa.Column("surface", sa.String(32), nullable=False),
        sa.Column("external_reference", sa.String(256)),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata_json", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("delivery_key", name="uq_match_read_delivery_key"),
    )
    op.create_index("ix_match_read_deliveries_read", "match_read_deliveries", ["match_read_id"])
    op.create_index("ix_match_read_deliveries_surface", "match_read_deliveries", ["surface"])


def downgrade() -> None:
    op.drop_table("match_read_deliveries")
    op.drop_table("match_read_package_selections")
    op.drop_table("match_read_packages")
    op.drop_table("match_read_selections")
    op.drop_table("match_reads")
