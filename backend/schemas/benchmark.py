from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


SCORE_FIELDS = (
    "correctness",
    "teaching_quality",
    "adaptation",
    "emotional_intelligence",
    "multilingual_quality",
    "hallucination_risk",
    "conversation_quality",
)


class BenchmarkLoadRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    model_name: str = Field(..., min_length=1, max_length=120)


class BenchmarkRunRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    model_name: str = Field(..., min_length=1, max_length=120)
    dataset_name: str = Field(default="k12_teacher_core_v1", min_length=1, max_length=120)
    max_items: int | None = Field(default=None, ge=1, le=100)


class BenchmarkCollectedItem(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    item_id: str
    student_prompt: str
    student_level: str
    language: str
    subject: str
    rubric: str
    response: str
    generation_latency_ms: float = Field(default=0.0, ge=0)


class BenchmarkEvaluateCollectedRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    model_name: str = Field(..., min_length=1, max_length=120)
    dataset_name: str = Field(default="k12_teacher_core_v1", min_length=1, max_length=120)
    items: list[BenchmarkCollectedItem] = Field(..., min_length=1, max_length=100)


class BenchmarkJudgeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    student_prompt: str
    student_level: str
    language: str
    subject: str
    rubric: str
    model_name: str
    response: str


class BenchmarkScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correctness: int = Field(..., ge=1, le=10)
    teaching_quality: int = Field(..., ge=1, le=10)
    adaptation: int = Field(..., ge=1, le=10)
    emotional_intelligence: int = Field(..., ge=1, le=10)
    multilingual_quality: int = Field(..., ge=1, le=10)
    hallucination_risk: int = Field(..., ge=1, le=10)
    conversation_quality: int = Field(..., ge=1, le=10)


class SingleJudgeEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_score: int = Field(..., ge=1, le=100)
    scores: BenchmarkScores
    learning_effectiveness: Literal["Excellent", "Strong", "Moderate", "Weak", "Harmful"]
    safety_classification: Literal["Safe", "Minor Risk", "Unsafe"]
    strengths: list[str] = Field(default_factory=list, max_length=8)
    weaknesses: list[str] = Field(default_factory=list, max_length=8)
    reasoning: str = Field(..., min_length=1, max_length=800)


class BenchmarkItemResult(BaseModel):
    item_id: str
    student_prompt: str
    student_level: str
    language: str
    subject: str
    response: str
    scores: BenchmarkScores
    overall_score: int
    learning_effectiveness: str
    safety_classification: str
    reasoning: str
    generation_latency_ms: float
    judge_latency_ms: float


class BenchmarkAggregate(BaseModel):
    model_name: str
    dataset_name: str
    items_completed: int
    average_overall_score: float
    avg_correctness: float
    avg_teaching_quality: float
    avg_adaptation: float
    avg_emotional_intelligence: float
    avg_multilingual_quality: float
    avg_hallucination_risk: float
    avg_conversation_quality: float
    total_generation_latency_ms: float
    total_judge_latency_ms: float
    avg_generation_latency_ms: float
    avg_judge_latency_ms: float


class BenchmarkRunResponse(BaseModel):
    run_id: UUID
    model_name: str
    dataset_name: str
    status: str
    started_at: datetime
    completed_at: datetime | None
    aggregate: BenchmarkAggregate
    items: list[BenchmarkItemResult]
    benchmark_report_markdown: str


class BenchmarkRunSummary(BaseModel):
    run_id: UUID
    model_name: str
    dataset_name: str
    status: str
    items_completed: int
    average_overall_score: float
    avg_correctness: float
    avg_teaching_quality: float
    avg_adaptation: float
    avg_emotional_intelligence: float
    avg_multilingual_quality: float
    avg_hallucination_risk: float
    avg_conversation_quality: float
    created_at: datetime
    completed_at: datetime | None


class BenchmarkModelState(BaseModel):
    active_model: str | None = None
    loaded: bool = False


class BenchmarkDatasetItem(BaseModel):
    item_id: str
    student_prompt: str
    student_level: str
    language: str
    subject: str
    rubric: str
    tags: list[str] = Field(default_factory=list)
