"""Research profile peer ranks and lookup indexes.

Revision ID: 0009_research_profiles
Revises: 0008_match_read_cycle
"""
from alembic import op
import sqlalchemy as sa

revision = "0009_research_profiles"
down_revision = "0008_match_read_cycle"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "research_profile_builds",
        sa.Column("scope_key", sa.String(96), primary_key=True),
        sa.Column("competition_id", sa.Integer, sa.ForeignKey("competitions.id")),
        sa.Column("season_id", sa.Integer, sa.ForeignKey("seasons.id")),
        sa.Column("season_year", sa.Integer, nullable=False),
        sa.Column("build_id", sa.String(64), nullable=False),
        sa.Column("built_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "research_peer_stats",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("scope_key", sa.String(96), sa.ForeignKey("research_profile_builds.scope_key"), nullable=False),
        sa.Column("subject_type", sa.String(16), nullable=False),
        sa.Column("subject_key", sa.String(128), nullable=False),
        sa.Column("competition_id", sa.Integer, sa.ForeignKey("competitions.id")),
        sa.Column("season_id", sa.Integer, sa.ForeignKey("seasons.id")),
        sa.Column("season_year", sa.Integer, nullable=False),
        sa.Column("peer_group", sa.String(32), nullable=False),
        sa.Column("metric", sa.String(48), nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("percentile", sa.Integer, nullable=False),
        sa.Column("rank", sa.Integer, nullable=False),
        sa.Column("peer_count", sa.Integer, nullable=False),
        sa.Column("league_average", sa.Float),
        sa.Column("sample_minutes", sa.Integer),
        sa.Column("sample_matches", sa.Integer, nullable=False),
        sa.Column("build_id", sa.String(64), nullable=False),
        sa.Column("built_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("scope_key", "subject_type", "subject_key", "peer_group", "metric",
                            name="uq_research_peer_metric"),
    )
    op.create_index("ix_research_peer_subject", "research_peer_stats", ["subject_type", "subject_key", "season_year"])
    op.create_table(
        "research_referee_aliases",
        sa.Column("raw_name", sa.String(128), primary_key=True),
        sa.Column("referee_key", sa.String(128), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
    )
    op.create_index("ix_research_referee_aliases_referee_key", "research_referee_aliases", ["referee_key"])
    op.create_index("ix_research_player_fixture", "fixture_player_stats", ["player_id", "fixture_id"])
    op.create_index("ix_research_fixture_referee", "fixtures", ["referee", "season_id"])


def downgrade():
    op.drop_index("ix_research_fixture_referee", table_name="fixtures")
    op.drop_index("ix_research_player_fixture", table_name="fixture_player_stats")
    op.drop_table("research_referee_aliases")
    op.drop_table("research_peer_stats")
    op.drop_table("research_profile_builds")
