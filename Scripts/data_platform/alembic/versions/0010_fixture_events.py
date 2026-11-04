"""Match events per finished fixture (cards plan task 6, D54 a).

Revision ID: 0010_fixture_events
Revises: 0009_research_profiles
"""
from alembic import op
import sqlalchemy as sa

from data_platform.models.json_type import JSONB

revision = "0010_fixture_events"
down_revision = "0009_research_profiles"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "fixture_match_events",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("fixture_id", sa.Integer, sa.ForeignKey("fixtures.id", ondelete="CASCADE"), nullable=False),
        sa.Column("event_index", sa.Integer, nullable=False),
        sa.Column("team_api_id", sa.Integer),
        sa.Column("player_api_id", sa.Integer),
        sa.Column("assist_api_id", sa.Integer),
        sa.Column("elapsed", sa.Integer),
        sa.Column("extra", sa.Integer),
        sa.Column("event_type", sa.String(32)),
        sa.Column("detail", sa.String(128)),
        sa.Column("comments", sa.Text),
        sa.Column("raw_json", JSONB, nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("fixture_id", "event_index", name="uq_fixture_match_events_fixture_index"),
    )


def downgrade():
    op.drop_table("fixture_match_events")
