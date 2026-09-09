"""V8: scheduled Match Read worker leases and observation ledger.

Revision ID: 0008_match_read_cycle
Revises: 0007_match_reads
Create Date: 2026-09-09 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from data_platform.models.json_type import JSONB


revision = "0008_match_read_cycle"
down_revision = "0007_match_reads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "worker_leases",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("lease_key", sa.String(128), nullable=False),
        sa.Column("owner_id", sa.String(96), nullable=False),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata_json", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("lease_key", name="uq_worker_leases_key"),
    )
    op.create_index("ix_worker_leases_expires_at", "worker_leases", ["expires_at"])

    op.create_table(
        "match_read_observations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("sync_run_id", sa.Integer, sa.ForeignKey("sync_runs.id", ondelete="SET NULL")),
        sa.Column("match_read_id", sa.Integer, sa.ForeignKey("match_reads.id", ondelete="SET NULL")),
        sa.Column("fixture_api_id", sa.String(64), nullable=False),
        sa.Column("league", sa.String(32), nullable=False),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("source", sa.String(32), nullable=False, server_default="api_football"),
        sa.Column("input_snapshot_id", sa.String(256)),
        sa.Column("read_key", sa.String(64)),
        sa.Column("provider_calls_json", JSONB),
        sa.Column("detail_json", JSONB),
        sa.Column("error_text", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_match_read_observations_fixture_stage_checked",
        "match_read_observations",
        ["fixture_api_id", "stage", "checked_at"],
    )
    op.create_index(
        "ix_match_read_observations_read_checked",
        "match_read_observations",
        ["match_read_id", "checked_at"],
    )
    op.create_index("ix_match_read_observations_run", "match_read_observations", ["sync_run_id"])
    op.create_index(
        "ix_match_read_observations_status_checked",
        "match_read_observations",
        ["status", "checked_at"],
    )


def downgrade() -> None:
    op.drop_table("match_read_observations")
    op.drop_table("worker_leases")
