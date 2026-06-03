"""add conversation simulation tables

Revision ID: 0004_conversation_sims
Revises: 0003_single_model_benchmarking
Create Date: 2026-06-01 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "0004_conversation_sims"
down_revision = "0003_single_model_benchmarking"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversation_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tutor_model", sa.String(length=120), nullable=False),
        sa.Column("student_model", sa.String(length=120), nullable=False),
        sa.Column("topic", sa.String(length=300), nullable=False),
        sa.Column("student_level", sa.String(length=120), nullable=False),
        sa.Column("language", sa.String(length=120), nullable=False),
        sa.Column("persona", sa.String(length=120), nullable=False),
        sa.Column("turns", sa.Integer(), nullable=False),
        sa.Column("transcript_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latency_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("benchmark_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("model_config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conversation_runs_created_at", "conversation_runs", ["created_at"], unique=False)
    op.create_index("ix_conversation_runs_language", "conversation_runs", ["language"], unique=False)
    op.create_index("ix_conversation_runs_models", "conversation_runs", ["tutor_model", "student_model"], unique=False)
    op.create_index("ix_conversation_runs_persona", "conversation_runs", ["persona"], unique=False)
    op.create_index("ix_conversation_runs_student_model", "conversation_runs", ["student_model"], unique=False)
    op.create_index("ix_conversation_runs_topic", "conversation_runs", ["topic"], unique=False)
    op.create_index("ix_conversation_runs_tutor_model", "conversation_runs", ["tutor_model"], unique=False)

    op.create_table(
        "evaluation_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("report_type", sa.String(length=80), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=False),
        sa.Column("failure_modes", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("judge_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_evaluation_reports_report_type", "evaluation_reports", ["report_type"], unique=False)
    op.create_index("ix_evaluation_reports_run_id", "evaluation_reports", ["run_id"], unique=False)
    op.create_index("ix_evaluation_reports_run_type", "evaluation_reports", ["run_id", "report_type"], unique=False)

    op.create_table(
        "benchmark_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("benchmark_type", sa.String(length=120), nullable=False),
        sa.Column("tutor_model", sa.String(length=120), nullable=False),
        sa.Column("student_model", sa.String(length=120), nullable=False),
        sa.Column("topic", sa.String(length=300), nullable=False),
        sa.Column("aggregate_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_benchmark_history_benchmark_type", "benchmark_history", ["benchmark_type"], unique=False)
    op.create_index("ix_benchmark_history_created_at", "benchmark_history", ["created_at"], unique=False)
    op.create_index("ix_benchmark_history_run_id", "benchmark_history", ["run_id"], unique=False)
    op.create_index("ix_benchmark_history_student_model", "benchmark_history", ["student_model"], unique=False)
    op.create_index("ix_benchmark_history_tutor_model", "benchmark_history", ["tutor_model"], unique=False)
    op.create_index("ix_benchmark_history_type_created", "benchmark_history", ["benchmark_type", "created_at"], unique=False)


def downgrade() -> None:
    op.drop_table("benchmark_history")
    op.drop_table("evaluation_reports")
    op.drop_table("conversation_runs")
