"""V2: predictions, trial_users, trial_parlays, parlay_sessions

Revision ID: 0002_predictions_parlays
Revises: 0001_initial
Create Date: 2026-04-14 01:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from data_platform.models.json_type import JSONB


revision = "0002_predictions_parlays"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("fixture_api_id", sa.String(64)),
        sa.Column("home_team", sa.String(128), nullable=False),
        sa.Column("away_team", sa.String(128), nullable=False),
        sa.Column("league", sa.String(32), nullable=False),
        sa.Column("kickoff", sa.String(64)),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("pick", sa.String(255), nullable=False),
        sa.Column("side", sa.String(32)),
        sa.Column("line", sa.Numeric(10, 4)),
        sa.Column("odds", sa.Numeric(10, 4)),
        sa.Column("bookmaker", sa.String(64)),
        sa.Column("model_prob", sa.Numeric(10, 6)),
        sa.Column("implied_prob", sa.Numeric(10, 6)),
        sa.Column("value_edge", sa.Numeric(10, 6)),
        sa.Column("confidence", sa.String(16)),
        sa.Column("projected_total", sa.Numeric(10, 4)),
        sa.Column("source", sa.String(32), nullable=False, server_default="standalone"),
        sa.Column("actual_result", sa.Numeric(10, 4)),
        sa.Column("outcome", sa.String(16)),
        sa.Column("graded_at", sa.DateTime(timezone=True)),
        sa.Column("season", sa.String(16)),
        sa.Column("prediction_date", sa.Date),
        sa.Column("closing_odds", sa.Numeric(10, 4)),
        sa.Column("closing_implied_prob", sa.Numeric(10, 6)),
        sa.Column("clv", sa.Numeric(10, 6)),
        sa.Column("extras", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "prediction_date", "home_team", "away_team", "league",
            "market", "pick", "source",
            name="uq_predictions_dedup",
        ),
    )
    op.create_index("ix_predictions_date", "predictions", ["prediction_date"])
    op.create_index("ix_predictions_market", "predictions", ["market"])
    op.create_index("ix_predictions_confidence", "predictions", ["confidence"])
    op.create_index("ix_predictions_outcome", "predictions", ["outcome"])
    op.create_index("ix_predictions_league", "predictions", ["league"])
    op.create_index("ix_predictions_fixture_api_id", "predictions", ["fixture_api_id"])

    op.create_table(
        "trial_users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("discord_id", sa.String(64), nullable=False),
        sa.Column("discord_username", sa.String(128)),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("hits", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_parlays_seen", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_profit", sa.Numeric(10, 4), nullable=False, server_default="0"),
        sa.Column("graduated", sa.Integer, nullable=False, server_default="0"),
        sa.Column("graduated_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("discord_id", name="uq_trial_users_discord_id"),
    )

    op.create_table(
        "trial_parlays",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("posted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("matchday_date", sa.Date, nullable=False),
        sa.Column("legs", JSONB, nullable=False),
        sa.Column("combined_odds", sa.Numeric(10, 4), nullable=False),
        sa.Column("outcome", sa.String(16)),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.Column("stake", sa.Numeric(10, 4), nullable=False, server_default="10.0"),
        sa.Column("profit", sa.Numeric(10, 4)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_trial_parlays_matchday", "trial_parlays", ["matchday_date"])

    op.create_table(
        "parlay_sessions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("parent_session_id", sa.Integer),
        sa.Column("discord_user_id", sa.Integer),
        sa.Column("guild_id", sa.Integer),
        sa.Column("channel_id", sa.Integer),
        sa.Column("thread_id", sa.Integer),
        sa.Column("message_id", sa.Integer),
        sa.Column("source_command", sa.String(64), nullable=False),
        sa.Column("action_type", sa.String(64), nullable=False),
        sa.Column("request_json", JSONB, nullable=False),
        sa.Column("result_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_parlay_sessions_user", "parlay_sessions", ["discord_user_id"])
    op.create_index("ix_parlay_sessions_thread", "parlay_sessions", ["thread_id"])
    op.create_index("ix_parlay_sessions_parent", "parlay_sessions", ["parent_session_id"])
    op.create_index("ix_parlay_sessions_message", "parlay_sessions", ["message_id"])


def downgrade() -> None:
    for table in ("parlay_sessions", "trial_parlays", "trial_users", "predictions"):
        op.drop_table(table)
