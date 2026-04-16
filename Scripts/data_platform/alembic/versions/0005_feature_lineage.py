"""V3: feature_builds table + lineage columns on feature snapshot tables

Revision ID: 0005_feature_lineage
Revises: 0004_kb_odds
Create Date: 2026-04-14 04:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from data_platform.models.json_type import JSONB


revision = "0005_feature_lineage"
down_revision = "0004_kb_odds"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "feature_builds",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("feature_version", sa.String(32), nullable=False),
        sa.Column("scope", sa.String(128), nullable=False),
        sa.Column("as_of_date", sa.Date, nullable=False),
        sa.Column("pipeline", sa.String(64), nullable=False, server_default="default"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(16), nullable=False, server_default="running"),
        sa.Column("sync_run_id", sa.Integer, sa.ForeignKey("sync_runs.id", ondelete="SET NULL")),
        sa.Column("upstream_payload_ids", JSONB),
        sa.Column("inputs_digest", sa.String(64)),
        sa.Column("row_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("scope", "as_of_date", "feature_version", name="uq_feature_builds_scope_date_version"),
    )
    op.create_index("ix_feature_builds_scope_date", "feature_builds", ["scope", "as_of_date"])
    op.create_index("ix_feature_builds_version", "feature_builds", ["feature_version"])

    with op.batch_alter_table("team_feature_snapshots") as batch:
        batch.add_column(sa.Column("feature_version", sa.String(32)))
        batch.add_column(sa.Column(
            "feature_build_id", sa.Integer,
            sa.ForeignKey("feature_builds.id", ondelete="SET NULL",
                          name="fk_team_feature_snapshots_feature_build_id"),
        ))
        batch.add_column(sa.Column("as_of_fixture_api_id", sa.Integer))
        batch.add_column(sa.Column("built_at", sa.DateTime(timezone=True)))

    with op.batch_alter_table("player_feature_snapshots") as batch:
        batch.add_column(sa.Column("feature_version", sa.String(32)))
        batch.add_column(sa.Column(
            "feature_build_id", sa.Integer,
            sa.ForeignKey("feature_builds.id", ondelete="SET NULL",
                          name="fk_player_feature_snapshots_feature_build_id"),
        ))
        batch.add_column(sa.Column("as_of_fixture_api_id", sa.Integer))
        batch.add_column(sa.Column("built_at", sa.DateTime(timezone=True)))


def downgrade() -> None:
    with op.batch_alter_table("player_feature_snapshots") as batch:
        batch.drop_column("built_at")
        batch.drop_column("as_of_fixture_api_id")
        batch.drop_column("feature_build_id")
        batch.drop_column("feature_version")
    with op.batch_alter_table("team_feature_snapshots") as batch:
        batch.drop_column("built_at")
        batch.drop_column("as_of_fixture_api_id")
        batch.drop_column("feature_build_id")
        batch.drop_column("feature_version")
    op.drop_table("feature_builds")
