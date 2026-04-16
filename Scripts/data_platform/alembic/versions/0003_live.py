"""V2: live_fixtures, live_state_snapshots, live_model_snapshots, live_odds_snapshots, live_alerts

Revision ID: 0003_live
Revises: 0002_predictions_parlays
Create Date: 2026-04-14 02:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from data_platform.models.json_type import JSONB


revision = "0003_live"
down_revision = "0002_predictions_parlays"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "live_fixtures",
        sa.Column("fixture_id", sa.Integer, primary_key=True),
        sa.Column("league", sa.String(64), nullable=False),
        sa.Column("home_team", sa.String(128), nullable=False),
        sa.Column("away_team", sa.String(128), nullable=False),
        sa.Column("kickoff_utc", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "live_state_snapshots",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("fixture_id", sa.Integer, nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(16)),
        sa.Column("elapsed_min", sa.Numeric(6, 2)),
        sa.Column("state_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_live_state_fixture_time", "live_state_snapshots", ["fixture_id", "captured_at"])

    op.create_table(
        "live_model_snapshots",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("fixture_id", sa.Integer, nullable=False),
        sa.Column("snapshot_ts", sa.String(64), nullable=False),
        sa.Column("elapsed_min", sa.Numeric(6, 2)),
        sa.Column("score", sa.String(16)),
        sa.Column("snapshot_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_live_model_fixture_time", "live_model_snapshots", ["fixture_id", "snapshot_ts"])

    op.create_table(
        "live_odds_snapshots",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("fixture_id", sa.Integer, nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("odds_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_live_odds_fixture_time", "live_odds_snapshots", ["fixture_id", "captured_at"])

    op.create_table(
        "live_alerts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("fixture_id", sa.Integer, nullable=False),
        sa.Column("snapshot_ts", sa.String(64), nullable=False),
        sa.Column("emitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("alert_type", sa.String(64)),
        sa.Column("severity", sa.String(16)),
        sa.Column("message", sa.Text),
        sa.Column("alert_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_live_alerts_fixture_time", "live_alerts", ["fixture_id", "snapshot_ts"])


def downgrade() -> None:
    for table in ("live_alerts", "live_odds_snapshots", "live_model_snapshots", "live_state_snapshots", "live_fixtures"):
        op.drop_table(table)
