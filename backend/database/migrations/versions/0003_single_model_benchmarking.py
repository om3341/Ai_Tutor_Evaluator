"""single model benchmarking

Revision ID: 0003_single_model_benchmarking
Revises: 0002_step4_analytics
Create Date: 2026-05-26 12:30:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0003_single_model_benchmarking"
down_revision = "0002_step4_analytics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "benchmark_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_name", sa.String(length=120), nullable=False),
        sa.Column("dataset_name", sa.String(length=120), nullable=False),
        sa.Column("status", sa.String(length=32), server_default=sa.text("'completed'"), nullable=False),
        sa.Column("items_completed", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("aggregate_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latency_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("benchmark_report_markdown", sa.Text(), server_default="", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_benchmark_runs_created_at", "benchmark_runs", ["created_at"])
    op.create_index("ix_benchmark_runs_dataset_name", "benchmark_runs", ["dataset_name"])
    op.create_index("ix_benchmark_runs_model_dataset", "benchmark_runs", ["model_name", "dataset_name"])
    op.create_index("ix_benchmark_runs_model_name", "benchmark_runs", ["model_name"])

    op.create_table(
        "benchmark_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_item_id", sa.String(length=120), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("student_level", sa.String(length=120), nullable=False),
        sa.Column("language", sa.String(length=120), nullable=False),
        sa.Column("subject", sa.String(length=120), nullable=False),
        sa.Column("model_name", sa.String(length=120), nullable=False),
        sa.Column("response", sa.Text(), nullable=False),
        sa.Column("scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("overall_score", sa.Float(), nullable=False),
        sa.Column("learning_effectiveness", sa.String(length=32), nullable=False),
        sa.Column("safety_classification", sa.String(length=32), nullable=False),
        sa.Column("judge_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latency_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_benchmark_items_dataset_item_id", "benchmark_items", ["dataset_item_id"])
    op.create_index("ix_benchmark_items_model_created", "benchmark_items", ["model_name", "created_at"])
    op.create_index("ix_benchmark_items_model_name", "benchmark_items", ["model_name"])
    op.create_index("ix_benchmark_items_run_created", "benchmark_items", ["run_id", "created_at"])
    op.create_index("ix_benchmark_items_run_id", "benchmark_items", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_benchmark_items_run_id", table_name="benchmark_items")
    op.drop_index("ix_benchmark_items_run_created", table_name="benchmark_items")
    op.drop_index("ix_benchmark_items_model_name", table_name="benchmark_items")
    op.drop_index("ix_benchmark_items_model_created", table_name="benchmark_items")
    op.drop_index("ix_benchmark_items_dataset_item_id", table_name="benchmark_items")
    op.drop_table("benchmark_items")
    op.drop_index("ix_benchmark_runs_model_name", table_name="benchmark_runs")
    op.drop_index("ix_benchmark_runs_model_dataset", table_name="benchmark_runs")
    op.drop_index("ix_benchmark_runs_dataset_name", table_name="benchmark_runs")
    op.drop_index("ix_benchmark_runs_created_at", table_name="benchmark_runs")
    op.drop_table("benchmark_runs")
