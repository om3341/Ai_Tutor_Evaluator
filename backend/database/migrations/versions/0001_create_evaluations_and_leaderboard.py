"""create evaluations and leaderboard tables

Revision ID: 0001_step3_rankings
Revises:
Create Date: 2026-05-24
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_step3_rankings"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("student_level", sa.String(length=120), nullable=False),
        sa.Column("language", sa.String(length=120), nullable=False),
        sa.Column("model_a", sa.String(length=120), nullable=False),
        sa.Column("model_b", sa.String(length=120), nullable=False),
        sa.Column("response_a", sa.Text(), nullable=False),
        sa.Column("response_b", sa.Text(), nullable=False),
        sa.Column("winner", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("evaluation_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latency_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("elo_before", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("elo_after", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_evaluations_created_at", "evaluations", ["created_at"])
    op.create_index("ix_evaluations_model_a", "evaluations", ["model_a"])
    op.create_index("ix_evaluations_model_b", "evaluations", ["model_b"])
    op.create_index("ix_evaluations_model_pair", "evaluations", ["model_a", "model_b"])
    op.create_index("ix_evaluations_winner", "evaluations", ["winner"])

    op.create_table(
        "leaderboard",
        sa.Column("model_name", sa.String(length=120), nullable=False),
        sa.Column("elo_score", sa.Float(), server_default=sa.text("1000"), nullable=False),
        sa.Column("wins", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("losses", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("draws", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("matches_played", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("avg_correctness", sa.Float(), server_default=sa.text("0"), nullable=False),
        sa.Column("avg_teaching_quality", sa.Float(), server_default=sa.text("0"), nullable=False),
        sa.Column("avg_adaptation", sa.Float(), server_default=sa.text("0"), nullable=False),
        sa.Column("avg_emotional_intelligence", sa.Float(), server_default=sa.text("0"), nullable=False),
        sa.Column("avg_multilingual_quality", sa.Float(), server_default=sa.text("0"), nullable=False),
        sa.Column("avg_hallucination_risk", sa.Float(), server_default=sa.text("0"), nullable=False),
        sa.Column("avg_conversation_quality", sa.Float(), server_default=sa.text("0"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("model_name"),
    )
    op.create_index("ix_leaderboard_elo_score", "leaderboard", ["elo_score"])
    op.create_index("ix_leaderboard_matches_played", "leaderboard", ["matches_played"])
    op.create_index("ix_leaderboard_wins", "leaderboard", ["wins"])


def downgrade() -> None:
    op.drop_index("ix_leaderboard_wins", table_name="leaderboard")
    op.drop_index("ix_leaderboard_matches_played", table_name="leaderboard")
    op.drop_index("ix_leaderboard_elo_score", table_name="leaderboard")
    op.drop_table("leaderboard")
    op.drop_index("ix_evaluations_winner", table_name="evaluations")
    op.drop_index("ix_evaluations_model_pair", table_name="evaluations")
    op.drop_index("ix_evaluations_model_b", table_name="evaluations")
    op.drop_index("ix_evaluations_model_a", table_name="evaluations")
    op.drop_index("ix_evaluations_created_at", table_name="evaluations")
    op.drop_table("evaluations")
