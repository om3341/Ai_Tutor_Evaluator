from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelGenerationRequest(BaseModel):
    """Payload for generating a candidate tutor response from a model server."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    student_prompt: str = Field(..., min_length=1, max_length=8000)
    student_level: str = Field(..., min_length=1, max_length=120)
    language: str = Field(..., min_length=1, max_length=120)
    model_name: str = Field(default="Qwen 3.8B", min_length=1, max_length=120)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)


class ModelGenerationResponse(BaseModel):
    """Generated candidate response plus basic inference latency."""

    model_name: str
    provider_model_name: str
    response: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
