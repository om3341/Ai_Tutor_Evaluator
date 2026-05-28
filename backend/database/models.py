from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    student_level: Mapped[str] = mapped_column(String(120), nullable=False)
    language: Mapped[str] = mapped_column(String(120), nullable=False)
    model_a: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    model_b: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    response_a: Mapped[str] = mapped_column(Text, nullable=False)
    response_b: Mapped[str] = mapped_column(Text, nullable=False)
    winner: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    scores: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    evaluation_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    latency_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    multilingual_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    hallucination_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    benchmark_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    elo_before: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False)
    elo_after: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_evaluations_created_at", "created_at"),
        Index("ix_evaluations_model_pair", "model_a", "model_b"),
    )


class Leaderboard(Base):
    __tablename__ = "leaderboard"

    model_name: Mapped[str] = mapped_column(String(120), primary_key=True)
    elo_score: Mapped[float] = mapped_column(Float, nullable=False, default=1000.0, server_default=text("1000"))
    wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    draws: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    matches_played: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    avg_correctness: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default=text("0"))
    avg_teaching_quality: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default=text("0"))
    avg_adaptation: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default=text("0"))
    avg_emotional_intelligence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default=text("0"))
    avg_multilingual_quality: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default=text("0"))
    avg_hallucination_risk: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default=text("0"))
    avg_conversation_quality: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default=text("0"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_leaderboard_elo_score", "elo_score"),
        Index("ix_leaderboard_wins", "wins"),
        Index("ix_leaderboard_matches_played", "matches_played"),
    )


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    dataset_name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed", server_default=text("'completed'"))
    items_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    aggregate_scores: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    latency_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    benchmark_report_markdown: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_benchmark_runs_created_at", "created_at"),
        Index("ix_benchmark_runs_model_dataset", "model_name", "dataset_name"),
    )


class BenchmarkItem(Base):
    __tablename__ = "benchmark_items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    dataset_item_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    student_level: Mapped[str] = mapped_column(String(120), nullable=False)
    language: Mapped[str] = mapped_column(String(120), nullable=False)
    subject: Mapped[str] = mapped_column(String(120), nullable=False)
    model_name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    scores: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    learning_effectiveness: Mapped[str] = mapped_column(String(32), nullable=False)
    safety_classification: Mapped[str] = mapped_column(String(32), nullable=False)
    judge_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    latency_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_benchmark_items_run_created", "run_id", "created_at"),
        Index("ix_benchmark_items_model_created", "model_name", "created_at"),
    )
