"""add step 4 analytics metadata columns

Revision ID: 0002_step4_analytics
Revises: 0001_step3_rankings
Create Date: 2026-05-24
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0002_step4_analytics"
down_revision = "0001_step3_rankings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "evaluations",
        sa.Column("multilingual_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "evaluations",
        sa.Column("hallucination_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "evaluations",
        sa.Column("benchmark_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("evaluations", "benchmark_metadata")
    op.drop_column("evaluations", "hallucination_metrics")
    op.drop_column("evaluations", "multilingual_metrics")
