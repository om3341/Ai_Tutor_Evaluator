from __future__ import annotations

from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Winner(str, Enum):
    """Allowed pairwise winner labels returned by the judge."""

    A = "A"
    B = "B"


class EvaluationRequest(BaseModel):
    """Payload accepted by POST /evaluate."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    student_prompt: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="Original student question or tutoring prompt.",
    )
    student_level: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Student grade, class, age band, or skill level.",
    )
    language: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Expected language or mix, for example Hindi, Marathi, Hinglish.",
    )
    model_a: str = Field(..., min_length=1, max_length=120)
    model_b: str = Field(..., min_length=1, max_length=120)
    response_a: str = Field(..., min_length=1, max_length=20000)
    response_b: str = Field(..., min_length=1, max_length=20000)
    latency_a_ms: float | None = Field(default=None, ge=0)
    latency_b_ms: float | None = Field(default=None, ge=0)
    ttft_a_ms: float | None = Field(default=None, ge=0)
    ttft_b_ms: float | None = Field(default=None, ge=0)
    tokens_per_second_a: float | None = Field(default=None, ge=0)
    tokens_per_second_b: float | None = Field(default=None, ge=0)

    @field_validator("model_b")
    @classmethod
    def models_must_be_distinct(cls, value: str, info) -> str:
        model_a = info.data.get("model_a")
        if model_a and value.casefold() == model_a.casefold():
            raise ValueError("model_a and model_b must be different models.")
        return value


class CriterionScores(BaseModel):
    """Scores use a 1-10 scale where higher is better.

    For hallucination_risk, higher means safer and lower hallucination risk.
    """

    model_config = ConfigDict(extra="forbid")

    correctness: int = Field(..., ge=1, le=10)
    teaching_quality: int = Field(..., ge=1, le=10)
    adaptation: int = Field(..., ge=1, le=10)
    emotional_intelligence: int = Field(..., ge=1, le=10)
    multilingual_quality: int = Field(..., ge=1, le=10)
    hallucination_risk: int = Field(..., ge=1, le=10)
    conversation_quality: int = Field(..., ge=1, le=10)


class PairwiseScores(BaseModel):
    """Scores for both candidate responses."""

    model_config = ConfigDict(extra="forbid")

    A: CriterionScores
    B: CriterionScores


class JudgeEvaluation(BaseModel):
    """Strict JSON shape that Gemini must return and the backend validates."""

    model_config = ConfigDict(extra="forbid")

    winner: Winner
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: PairwiseScores
    reasoning: str = Field(..., min_length=1, max_length=2000)
    benchmark_report_markdown: str | None = Field(default=None, max_length=12000)

    @field_validator("reasoning")
    @classmethod
    def reasoning_must_be_concise(cls, value: str) -> str:
        return " ".join(value.split())


class LatencyMetrics(BaseModel):
    """Server-side timing for the pairwise evaluation request."""

    model_config = ConfigDict(extra="forbid")

    judge_latency_ms: float = Field(..., ge=0)
    total_latency_ms: float = Field(..., ge=0)


class EvaluationResponse(JudgeEvaluation):
    """API response returned by POST /evaluate."""

    latency: LatencyMetrics
    evaluation_id: UUID | None = None
    model_a: str | None = None
    model_b: str | None = None


class ErrorResponse(BaseModel):
    """Consistent error payload for expected backend failures."""

    detail: str
    error_code: Literal[
        "GEMINI_CONFIGURATION_ERROR",
        "GEMINI_TIMEOUT",
        "GEMINI_RESPONSE_INVALID",
        "GEMINI_API_ERROR",
        "DATABASE_ERROR",
    ]
