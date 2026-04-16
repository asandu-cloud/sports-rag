"""V2: kb_documents, kb_refresh_queue, odds_snapshots

Revision ID: 0004_kb_odds
Revises: 0003_live
Create Date: 2026-04-14 03:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from data_platform.models.json_type import JSONB


revision = "0004_kb_odds"
down_revision = "0003_live"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "kb_documents",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("doc_id", sa.String(64), nullable=False),
        sa.Column("entity_type", sa.String(32), nullable=False),
        sa.Column("entity_id", sa.String(128)),
        sa.Column("league", sa.String(32)),
        sa.Column("season", sa.String(16)),
        sa.Column("doc_type", sa.String(32), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("doc_metadata", JSONB, nullable=False),
        sa.Column("chroma_synced_at", sa.DateTime(timezone=True)),
        sa.Column("chroma_synced_hash", sa.String(64)),
        sa.Column("last_source_run_id", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("doc_id", name="uq_kb_documents_doc_id"),
    )
    op.create_index("ix_kb_documents_entity", "kb_documents", ["entity_type", "entity_id"])
    op.create_index("ix_kb_documents_league_season", "kb_documents", ["league", "season", "doc_type"])
    op.create_index("ix_kb_documents_hash", "kb_documents", ["content_hash"])
    op.create_index("ix_kb_documents_synced", "kb_documents", ["chroma_synced_at"])

    op.create_table(
        "kb_refresh_queue",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("entity_type", sa.String(32), nullable=False),
        sa.Column("entity_key", sa.String(128), nullable=False),
        sa.Column("reason", sa.String(64)),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("error_text", sa.Text),
        sa.Column("meta", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_kb_queue_entity_status", "kb_refresh_queue", ["entity_type", "status"])
    op.create_index("ix_kb_queue_enqueued", "kb_refresh_queue", ["enqueued_at"])

    op.create_table(
        "odds_snapshots",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("event_id", sa.String(64), nullable=False),
        sa.Column("league", sa.String(32)),
        sa.Column("home_team", sa.String(128)),
        sa.Column("away_team", sa.String(128)),
        sa.Column("kickoff_utc", sa.DateTime(timezone=True)),
        sa.Column("kickoff_date", sa.String(16)),
        sa.Column("kind", sa.String(16), nullable=False, server_default="prematch"),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("odds_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("event_id", "captured_at", "kind", name="uq_odds_snapshot_event_time_kind"),
    )
    op.create_index("ix_odds_snapshots_event", "odds_snapshots", ["event_id", "captured_at"])
    op.create_index("ix_odds_snapshots_league_date", "odds_snapshots", ["league", "kickoff_date"])


def downgrade() -> None:
    for table in ("odds_snapshots", "kb_refresh_queue", "kb_documents"):
        op.drop_table(table)
