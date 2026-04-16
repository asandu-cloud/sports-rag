"""initial platform schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-14 00:00:00.000000

Creates the V1 canonical schema: competitions, seasons, teams, players,
fixtures, per-fixture team/player stats, sync bookkeeping, raw payload
archive, feature snapshots, and canonical app user tables.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from data_platform.models.json_type import JSONB


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "competitions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("code", sa.String(32), nullable=False, unique=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("country", sa.String(64)),
        sa.Column("api_football_id", sa.Integer, nullable=False, unique=True),
        sa.Column("competition_type", sa.String(32), nullable=False, server_default="domestic_league"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_competitions_code", "competitions", ["code"])

    op.create_table(
        "seasons",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("competition_id", sa.Integer, sa.ForeignKey("competitions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("year", sa.Integer, nullable=False),
        sa.Column("label", sa.String(16), nullable=False),
        sa.Column("start_date", sa.Date),
        sa.Column("end_date", sa.Date),
        sa.Column("is_current", sa.Boolean, nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("competition_id", "year", name="uq_seasons_competition_year"),
    )
    op.create_index("ix_seasons_competition_current", "seasons", ["competition_id", "is_current"])

    op.create_table(
        "teams",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("api_football_id", sa.Integer, nullable=False, unique=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("canonical_name", sa.String(128)),
        sa.Column("short_code", sa.String(16)),
        sa.Column("country", sa.String(64)),
        sa.Column("founded", sa.Integer),
        sa.Column("logo_url", sa.String(512)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "players",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("api_football_id", sa.Integer, nullable=False, unique=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("position", sa.String(32)),
        sa.Column("nationality", sa.String(64)),
        sa.Column("birth_date", sa.Date),
        sa.Column("height_cm", sa.Integer),
        sa.Column("weight_kg", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "fixtures",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("api_football_id", sa.Integer, nullable=False, unique=True),
        sa.Column("competition_id", sa.Integer, sa.ForeignKey("competitions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("season_id", sa.Integer, sa.ForeignKey("seasons.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("home_team_id", sa.Integer, sa.ForeignKey("teams.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("away_team_id", sa.Integer, sa.ForeignKey("teams.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("kickoff_utc", sa.DateTime(timezone=True)),
        sa.Column("venue", sa.String(128)),
        sa.Column("city", sa.String(64)),
        sa.Column("status", sa.String(16)),
        sa.Column("elapsed", sa.Integer),
        sa.Column("home_goals", sa.Integer),
        sa.Column("away_goals", sa.Integer),
        sa.Column("round", sa.String(64)),
        sa.Column("referee", sa.String(128)),
        sa.Column("last_fetched_at", sa.DateTime(timezone=True)),
        sa.Column("last_api_updated_at", sa.DateTime(timezone=True)),
        sa.Column("payload_digest", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_fixtures_season_kickoff", "fixtures", ["season_id", "kickoff_utc"])
    op.create_index("ix_fixtures_competition_status", "fixtures", ["competition_id", "status"])
    op.create_index("ix_fixtures_home_team", "fixtures", ["home_team_id"])
    op.create_index("ix_fixtures_away_team", "fixtures", ["away_team_id"])

    op.create_table(
        "fixture_team_stats",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("fixture_id", sa.Integer, sa.ForeignKey("fixtures.id", ondelete="CASCADE"), nullable=False),
        sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("opponent_team_id", sa.Integer, sa.ForeignKey("teams.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("is_home", sa.Boolean, nullable=False),
        sa.Column("goals", sa.Integer),
        sa.Column("shots_total", sa.Integer),
        sa.Column("shots_on", sa.Integer),
        sa.Column("shots_off", sa.Integer),
        sa.Column("shots_blocked", sa.Integer),
        sa.Column("shots_inside_box", sa.Integer),
        sa.Column("shots_outside_box", sa.Integer),
        sa.Column("fouls_committed", sa.Integer),
        sa.Column("corners", sa.Integer),
        sa.Column("offsides", sa.Integer),
        sa.Column("possession", sa.Numeric(6, 4)),
        sa.Column("yellow_cards", sa.Integer),
        sa.Column("red_cards", sa.Integer),
        sa.Column("goalkeeper_saves", sa.Integer),
        sa.Column("passes_total", sa.Integer),
        sa.Column("passes_accurate", sa.Integer),
        sa.Column("pass_accuracy", sa.Numeric(6, 4)),
        sa.Column("expected_goals", sa.Numeric(7, 4)),
        sa.Column("goals_prevented", sa.Numeric(7, 4)),
        sa.Column("raw_payload_digest", sa.String(64)),
        sa.Column("stats_json", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("fixture_id", "team_id", name="uq_team_stats_fixture_team"),
    )
    op.create_index("ix_team_stats_team", "fixture_team_stats", ["team_id"])

    op.create_table(
        "fixture_player_stats",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("fixture_id", sa.Integer, sa.ForeignKey("fixtures.id", ondelete="CASCADE"), nullable=False),
        sa.Column("player_id", sa.Integer, sa.ForeignKey("players.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("position", sa.String(32)),
        sa.Column("minutes", sa.Integer),
        sa.Column("rating", sa.Numeric(4, 2)),
        sa.Column("captain", sa.Boolean),
        sa.Column("substitute", sa.Boolean),
        sa.Column("goals", sa.Integer),
        sa.Column("assists", sa.Integer),
        sa.Column("shots_total", sa.Integer),
        sa.Column("shots_on", sa.Integer),
        sa.Column("passes_total", sa.Integer),
        sa.Column("passes_accurate", sa.Integer),
        sa.Column("pass_accuracy", sa.Numeric(6, 4)),
        sa.Column("tackles", sa.Integer),
        sa.Column("interceptions", sa.Integer),
        sa.Column("duels_won", sa.Integer),
        sa.Column("duels_total", sa.Integer),
        sa.Column("yellow_cards", sa.Integer),
        sa.Column("red_cards", sa.Integer),
        sa.Column("fouls_committed", sa.Integer),
        sa.Column("fouls_drawn", sa.Integer),
        sa.Column("raw_payload_digest", sa.String(64)),
        sa.Column("stats_json", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("fixture_id", "player_id", name="uq_player_stats_fixture_player"),
    )
    op.create_index("ix_player_stats_player", "fixture_player_stats", ["player_id"])
    op.create_index("ix_player_stats_team", "fixture_player_stats", ["team_id"])

    op.create_table(
        "sync_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("run_kind", sa.String(32), nullable=False),
        sa.Column("scope", sa.String(128)),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(16), nullable=False, server_default="running"),
        sa.Column("stats", JSONB),
        sa.Column("error_text", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_sync_runs_kind_started", "sync_runs", ["run_kind", "started_at"])

    op.create_table(
        "sync_watermarks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("scope", sa.String(128), nullable=False),
        sa.Column("last_successful_at", sa.DateTime(timezone=True)),
        sa.Column("last_cursor", sa.Text),
        sa.Column("meta", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("scope", name="uq_sync_watermarks_scope"),
    )

    op.create_table(
        "raw_payload_archive",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("endpoint", sa.String(128), nullable=False),
        sa.Column("params_digest", sa.String(64), nullable=False),
        sa.Column("params", JSONB),
        sa.Column("storage_backend", sa.String(32), nullable=False),
        sa.Column("storage_uri", sa.Text, nullable=False),
        sa.Column("payload_digest", sa.String(64), nullable=False),
        sa.Column("bytes_size", sa.Integer),
        sa.Column("content_type", sa.String(64)),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sync_run_id", sa.Integer, sa.ForeignKey("sync_runs.id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_raw_archive_provider_endpoint", "raw_payload_archive", ["provider", "endpoint", "fetched_at"])
    op.create_index("ix_raw_archive_digest", "raw_payload_archive", ["payload_digest"])

    op.create_table(
        "team_feature_snapshots",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id", ondelete="CASCADE"), nullable=False),
        sa.Column("season_id", sa.Integer, sa.ForeignKey("seasons.id", ondelete="CASCADE"), nullable=False),
        sa.Column("as_of_date", sa.Date, nullable=False),
        sa.Column("matches_played", sa.Integer),
        sa.Column("goals_for_pm", sa.Numeric(8, 4)),
        sa.Column("goals_against_pm", sa.Numeric(8, 4)),
        sa.Column("assists_pm", sa.Numeric(8, 4)),
        sa.Column("corners_pm", sa.Numeric(8, 4)),
        sa.Column("corners_against_pm", sa.Numeric(8, 4)),
        sa.Column("sot_for_pm", sa.Numeric(8, 4)),
        sa.Column("sot_against_pm", sa.Numeric(8, 4)),
        sa.Column("shots_for_pm", sa.Numeric(8, 4)),
        sa.Column("cards_pm", sa.Numeric(8, 4)),
        sa.Column("yellows_pm", sa.Numeric(8, 4)),
        sa.Column("reds_pm", sa.Numeric(8, 4)),
        sa.Column("cards_per_90_team", sa.Numeric(8, 4)),
        sa.Column("fouls_per_90_team", sa.Numeric(8, 4)),
        sa.Column("cards_per_foul_team", sa.Numeric(8, 4)),
        sa.Column("aggression_index_norm", sa.Numeric(8, 4)),
        sa.Column("form_index_team", sa.Numeric(8, 4)),
        sa.Column("control_index", sa.Numeric(8, 4)),
        sa.Column("dominance_index", sa.Numeric(8, 4)),
        sa.Column("possession", sa.Numeric(6, 4)),
        sa.Column("expected_goals", sa.Numeric(8, 4)),
        sa.Column("pass_accuracy_pct", sa.Numeric(6, 4)),
        sa.Column("opp_cards_induced_pm", sa.Numeric(8, 4)),
        sa.Column("goals_home_pm", sa.Numeric(8, 4)),
        sa.Column("goals_away_pm", sa.Numeric(8, 4)),
        sa.Column("corners_home_pm", sa.Numeric(8, 4)),
        sa.Column("corners_away_pm", sa.Numeric(8, 4)),
        sa.Column("cards_home_pm", sa.Numeric(8, 4)),
        sa.Column("cards_away_pm", sa.Numeric(8, 4)),
        sa.Column("corners_against_home_pm", sa.Numeric(8, 4)),
        sa.Column("corners_against_away_pm", sa.Numeric(8, 4)),
        sa.Column("sot_against_home_pm", sa.Numeric(8, 4)),
        sa.Column("sot_against_away_pm", sa.Numeric(8, 4)),
        sa.Column("cards_induced_home_pm", sa.Numeric(8, 4)),
        sa.Column("cards_induced_away_pm", sa.Numeric(8, 4)),
        sa.Column("xg_home_pm", sa.Numeric(8, 4)),
        sa.Column("xg_away_pm", sa.Numeric(8, 4)),
        sa.Column("sot_home_pm", sa.Numeric(8, 4)),
        sa.Column("sot_away_pm", sa.Numeric(8, 4)),
        sa.Column("fouls_home_pm", sa.Numeric(8, 4)),
        sa.Column("fouls_away_pm", sa.Numeric(8, 4)),
        sa.Column("corners_var", sa.Numeric(10, 4)),
        sa.Column("goals_var", sa.Numeric(10, 4)),
        sa.Column("cards_var", sa.Numeric(10, 4)),
        sa.Column("sot_var", sa.Numeric(10, 4)),
        sa.Column("archetype", sa.String(64)),
        sa.Column("build_run_id", sa.Integer, sa.ForeignKey("sync_runs.id", ondelete="SET NULL")),
        sa.Column("extras", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("team_id", "season_id", "as_of_date", name="uq_team_feature_snapshot"),
    )
    op.create_index("ix_team_feature_season_date", "team_feature_snapshots", ["season_id", "as_of_date"])

    op.create_table(
        "player_feature_snapshots",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("player_id", sa.Integer, sa.ForeignKey("players.id", ondelete="CASCADE"), nullable=False),
        sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id", ondelete="SET NULL")),
        sa.Column("season_id", sa.Integer, sa.ForeignKey("seasons.id", ondelete="CASCADE"), nullable=False),
        sa.Column("as_of_date", sa.Date, nullable=False),
        sa.Column("appearances_total", sa.Integer),
        sa.Column("appearances_as_starter", sa.Integer),
        sa.Column("start_rate", sa.Numeric(6, 4)),
        sa.Column("avg_minutes_per_appearance", sa.Numeric(7, 2)),
        sa.Column("minutes_risk", sa.String(32)),
        sa.Column("recent_role_trend", sa.String(32)),
        sa.Column("goals_per_90", sa.Numeric(7, 4)),
        sa.Column("assists_per_90", sa.Numeric(7, 4)),
        sa.Column("sot_per_90", sa.Numeric(7, 4)),
        sa.Column("cards_per_90", sa.Numeric(7, 4)),
        sa.Column("starter_goals_per_90", sa.Numeric(7, 4)),
        sa.Column("starter_assists_per_90", sa.Numeric(7, 4)),
        sa.Column("starter_sot_per_90", sa.Numeric(7, 4)),
        sa.Column("starter_cards_per_90", sa.Numeric(7, 4)),
        sa.Column("form_index", sa.Numeric(8, 4)),
        sa.Column("aggression_index_norm", sa.Numeric(8, 4)),
        sa.Column("control_index", sa.Numeric(8, 4)),
        sa.Column("expected_card_risk", sa.Numeric(8, 4)),
        sa.Column("expected_foul_pressure", sa.Numeric(8, 4)),
        sa.Column("total_minutes", sa.Integer),
        sa.Column("build_run_id", sa.Integer, sa.ForeignKey("sync_runs.id", ondelete="SET NULL")),
        sa.Column("extras", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("player_id", "season_id", "as_of_date", name="uq_player_feature_snapshot"),
    )
    op.create_index("ix_player_feature_season_date", "player_feature_snapshots", ["season_id", "as_of_date"])

    op.create_table(
        "app_users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("discord_id", sa.String(64)),
        sa.Column("discord_username", sa.String(128)),
        sa.Column("discord_avatar_url", sa.Text),
        sa.Column("email", sa.String(255)),
        sa.Column("stripe_customer_id", sa.String(64)),
        sa.Column("stripe_subscription_id", sa.String(64)),
        sa.Column("tier", sa.String(32), nullable=False, server_default="free"),
        sa.Column("status", sa.String(32), nullable=False, server_default="inactive"),
        sa.Column("trial_start", sa.DateTime(timezone=True)),
        sa.Column("trial_end", sa.DateTime(timezone=True)),
        sa.Column("subscription_start", sa.DateTime(timezone=True)),
        sa.Column("subscription_end", sa.DateTime(timezone=True)),
        sa.Column("cancelled_at", sa.DateTime(timezone=True)),
        sa.Column("is_founding_member", sa.Boolean, nullable=False, server_default=sa.text("0")),
        sa.Column("referral_code", sa.String(64)),
        sa.Column("referred_by", sa.String(64)),
        sa.Column("referral_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("password_hash", sa.Text),
        sa.Column("auth_provider", sa.String(32), nullable=False, server_default="discord"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("discord_id", name="uq_app_users_discord_id"),
        sa.UniqueConstraint("stripe_customer_id", name="uq_app_users_stripe_customer"),
        sa.UniqueConstraint("referral_code", name="uq_app_users_referral_code"),
    )
    op.create_index("ix_app_users_email_provider", "app_users", ["email", "auth_provider"])

    op.create_table(
        "app_subscription_events",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("app_users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stripe_event_id", sa.String(128), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("tier", sa.String(32)),
        sa.Column("status", sa.String(32)),
        sa.Column("raw_json", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("stripe_event_id", name="uq_app_sub_events_stripe_id"),
    )

    op.create_table(
        "app_referrals",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("referrer_discord_id", sa.String(64), nullable=False),
        sa.Column("referred_discord_id", sa.String(64), nullable=False),
        sa.Column("referred_username", sa.String(128)),
        sa.Column("referral_code_used", sa.String(64), nullable=False),
        sa.Column("stripe_event_id", sa.String(128)),
        sa.Column("reward_type", sa.String(64)),
        sa.Column("reward_granted", sa.Boolean, nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_app_referrals_referrer", "app_referrals", ["referrer_discord_id"])


def downgrade() -> None:
    for table in (
        "app_referrals",
        "app_subscription_events",
        "app_users",
        "player_feature_snapshots",
        "team_feature_snapshots",
        "raw_payload_archive",
        "sync_watermarks",
        "sync_runs",
        "fixture_player_stats",
        "fixture_team_stats",
        "fixtures",
        "players",
        "teams",
        "seasons",
        "competitions",
    ):
        op.drop_table(table)
