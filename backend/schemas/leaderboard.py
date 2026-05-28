from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class LeaderboardEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_name: str
    elo_score: float
    wins: int
    losses: int
    draws: int
    matches_played: int
    win_rate: float
    avg_correctness: float
    avg_teaching_quality: float
    avg_adaptation: float
    avg_emotional_intelligence: float
    avg_multilingual_quality: float
    avg_hallucination_risk: float
    avg_conversation_quality: float
    updated_at: datetime


class ModelStats(LeaderboardEntry):
    recent_evaluations: list["EvaluationHistoryItem"]


class EvaluationHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    prompt: str
    student_level: str
    language: str
    model_a: str
    model_b: str
    winner: str
    winner_model: str
    confidence: float
    scores: dict[str, Any]
    latency_metrics: dict[str, Any]
    elo_before: dict[str, float]
    elo_after: dict[str, float]
    created_at: datetime


class RecentPerformancePoint(BaseModel):
    evaluation_id: UUID
    model_name: str
    opponent: str
    result: str
    elo_before: float
    elo_after: float
    elo_delta: float
    confidence: float
    created_at: datetime
