"""add frozen RAG retrieval artifacts to conversation runs

Revision ID: 0005_rag_grounding
Revises: 0004_conversation_sims
Create Date: 2026-06-01 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "0005_rag_grounding"
down_revision = "0004_conversation_sims"
branch_labels = None
depends_on = None


def upgrade() -> None:
    empty_list = sa.text("'[]'::jsonb")
    empty_object = sa.text("'{}'::jsonb")
    op.add_column(
        "conversation_runs",
        sa.Column("retrieved_chunks", postgresql.JSONB(astext_type=sa.Text()), server_default=empty_list, nullable=False),
    )
    op.add_column(
        "conversation_runs",
        sa.Column("retrieval_scores", postgresql.JSONB(astext_type=sa.Text()), server_default=empty_list, nullable=False),
    )
    op.add_column(
        "conversation_runs",
        sa.Column("chunk_ids", postgresql.JSONB(astext_type=sa.Text()), server_default=empty_list, nullable=False),
    )
    op.add_column(
        "conversation_runs",
        sa.Column("rag_context", postgresql.JSONB(astext_type=sa.Text()), server_default=empty_object, nullable=False),
    )
    op.add_column(
        "conversation_runs",
        sa.Column("retrieval_metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=empty_object, nullable=False),
    )


def downgrade() -> None:
    op.drop_column("conversation_runs", "retrieval_metadata")
    op.drop_column("conversation_runs", "rag_context")
    op.drop_column("conversation_runs", "chunk_ids")
    op.drop_column("conversation_runs", "retrieval_scores")
    op.drop_column("conversation_runs", "retrieved_chunks")
