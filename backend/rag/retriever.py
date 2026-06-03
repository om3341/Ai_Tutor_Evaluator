from __future__ import annotations

import re

from backend.config import Settings
from backend.rag.qdrant_client import QdrantKnowledgeClient
from backend.schemas.rag import RagContext


class EducationalRetriever:
    """Retrieves a fixed educational context for one benchmark scenario."""

    def __init__(self, settings: Settings, client: QdrantKnowledgeClient) -> None:
        self._settings = settings
        self._client = client

    async def retrieve_context(
        self,
        *,
        topic: str,
        student_level: str,
        language: str,
        top_k: int | None = None,
    ) -> RagContext:
        query = f"{topic} {student_level} {language}".strip()
        limit = top_k or self._settings.rag_top_k
        subject = self._infer_subject(topic)
        collection_name = self._collection_for_level(student_level)
        chunks = await self._client.query(
            collection_name=collection_name,
            query_text=query,
            embedding_model=self._settings.qdrant_embedding_model,
            text_payload_key=self._settings.qdrant_text_payload_key,
            top_k=limit,
            score_threshold=self._settings.rag_score_threshold,
            subject=subject,
        )
        return RagContext(
            query=query,
            collection_name=collection_name,
            embedding_model=self._settings.qdrant_embedding_model,
            top_k=limit,
            chunks=chunks,
        )

    def _collection_for_level(self, student_level: str) -> str:
        match = re.search(r"\bclass\s*(\d{1,2})\b", student_level, flags=re.IGNORECASE)
        if match and self._settings.qdrant_collection_name.startswith("ebuddy_class"):
            return f"ebuddy_class{int(match.group(1))}"
        if match and "{class}" in self._settings.qdrant_collection_name:
            return self._settings.qdrant_collection_name.replace("{class}", str(int(match.group(1))))
        return self._settings.qdrant_collection_name

    @staticmethod
    def _infer_subject(topic: str) -> str | None:
        normalized = topic.casefold()
        science_terms = (
            "photosynthesis",
            "plant",
            "oxygen",
            "carbon dioxide",
            "force",
            "energy",
            "magnet",
            "electric",
            "water cycle",
            "evaporation",
            "organ",
            "cell",
        )
        math_terms = ("fraction", "decimal", "geometry", "angle", "algebra", "equation", "ratio", "percentage")
        if any(term in normalized for term in science_terms):
            return "science"
        if any(term in normalized for term in math_terms):
            return "math"
        return None
