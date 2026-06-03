from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RagChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    collection_name: str
    embedding_model: str
    top_k: int
    chunks: list[RagChunk] = Field(default_factory=list)


class RagRetrieveRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    topic: str = Field(..., min_length=1, max_length=300)
    student_level: str = Field(..., min_length=1, max_length=120)
    language: str = Field(..., min_length=1, max_length=120)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RagHealthResponse(BaseModel):
    reachable: bool
    collection_name: str
    collection_exists: bool
    detail: str
