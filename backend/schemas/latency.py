from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class ModelLatencyAnalytics(BaseModel):
    model_name: str
    samples: int
    avg_model_latency_ms: float
    p95_model_latency_ms: float
    avg_ttft_ms: float | None
    avg_tokens_per_second: float | None
    avg_judge_latency_ms: float
    avg_api_latency_ms: float
    speed_rank: int


class LatencyHistoryPoint(BaseModel):
    evaluation_id: UUID
    model_name: str
    language: str
    model_latency_ms: float | None
    ttft_ms: float | None
    tokens_per_second: float | None
    judge_latency_ms: float
    api_latency_ms: float
    created_at: datetime


class LatencyAnalyticsResponse(BaseModel):
    generated_at: datetime
    rows: list[ModelLatencyAnalytics]


class LatencyHistoryResponse(BaseModel):
    generated_at: datetime
    rows: list[LatencyHistoryPoint]
