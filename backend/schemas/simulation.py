from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from backend.schemas.rag import RagContext


PERSONAS = (
    "Curious Student",
    "Weak Student",
    "Average Student",
    "Advanced Student",
    "Distracted Student",
    "Exam-Anxious Student",
)

SIMULATION_SCORE_FIELDS = (
    "correctness",
    "teaching_quality",
    "adaptation",
    "emotional_intelligence",
    "multilingual_quality",
    "hallucination_risk",
    "conversation_quality",
    "learning_gain",
    "misconception_recovery",
    "scaffolding_effectiveness",
    "student_engagement",
    "retention_support",
    "knowledge_grounding",
    "context_faithfulness",
    "retrieval_utilization",
    "educational_grounding",
)


class SimulationLoadRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    tutor_model: str | None = Field(default=None, max_length=120)
    student_model: str | None = Field(default=None, max_length=120)


class SimulationRunRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    tutor_model: str = Field(..., min_length=1, max_length=120)
    student_model: str = Field(..., min_length=1, max_length=120)
    topic: str = Field(..., min_length=1, max_length=300)
    student_level: str = Field(..., min_length=1, max_length=120)
    language: str = Field(..., min_length=1, max_length=120)
    persona: Literal[
        "Curious Student",
        "Weak Student",
        "Average Student",
        "Advanced Student",
        "Distracted Student",
        "Exam-Anxious Student",
    ]
    tutor_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    student_temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    turns: int = Field(default=4, ge=1, le=12)


class SimulationMessage(BaseModel):
    role: Literal["student", "tutor"]
    content: str = Field(..., min_length=1)
    turn_index: int = Field(..., ge=1)
    latency_ms: float = Field(default=0.0, ge=0)


class SimulationScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correctness: int = Field(..., ge=1, le=10)
    teaching_quality: int = Field(..., ge=1, le=10)
    adaptation: int = Field(..., ge=1, le=10)
    emotional_intelligence: int = Field(..., ge=1, le=10)
    multilingual_quality: int = Field(..., ge=1, le=10)
    hallucination_risk: int = Field(..., ge=1, le=10)
    conversation_quality: int = Field(..., ge=1, le=10)
    learning_gain: int = Field(..., ge=1, le=10)
    misconception_recovery: int = Field(..., ge=1, le=10)
    scaffolding_effectiveness: int = Field(..., ge=1, le=10)
    student_engagement: int = Field(..., ge=1, le=10)
    retention_support: int = Field(..., ge=1, le=10)
    knowledge_grounding: int = Field(..., ge=1, le=10)
    context_faithfulness: int = Field(..., ge=1, le=10)
    retrieval_utilization: int = Field(..., ge=1, le=10)
    educational_grounding: int = Field(..., ge=1, le=10)


class ConversationJudgeEvaluation(SimulationScores):
    reasoning: str = Field(..., min_length=1, max_length=1200)
    failure_modes: list[str] = Field(default_factory=list, max_length=10)


class SimulationRunResponse(BaseModel):
    run_id: UUID
    tutor_model: str
    student_model: str
    topic: str
    student_level: str
    language: str
    persona: str
    turns: int
    transcript: list[SimulationMessage]
    retrieval: RagContext
    metrics: SimulationScores
    reasoning: str
    failure_modes: list[str]
    latency_metrics: dict[str, Any]
    created_at: datetime


class SimulationRunSummary(BaseModel):
    run_id: UUID
    tutor_model: str
    student_model: str
    topic: str
    student_level: str
    language: str
    persona: str
    turns: int
    metrics: dict[str, Any]
    created_at: datetime


class SimulationState(BaseModel):
    tutor_model: str | None = None
    student_model: str | None = None
    tutor_loaded: bool = False
    student_loaded: bool = False
    running: bool = False


class PersonaDescription(BaseModel):
    name: str
    learning_behavior: str
    curiosity_level: str
    misunderstanding_patterns: str
    follow_up_tendencies: str
    emotional_traits: str
