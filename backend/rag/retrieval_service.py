from __future__ import annotations

from backend.config import Settings
from backend.rag.context_builder import RagContextBuilder
from backend.rag.qdrant_client import QdrantKnowledgeClient, QdrantKnowledgeError
from backend.rag.retriever import EducationalRetriever
from backend.schemas.rag import RagContext
from backend.schemas.simulation import SimulationRunRequest


class RetrievalService:
    """Coordinates one-time retrieval and prompt-ready frozen context."""

    def __init__(
        self,
        settings: Settings,
        retriever: EducationalRetriever,
        context_builder: RagContextBuilder,
        client: QdrantKnowledgeClient,
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._context_builder = context_builder
        self._client = client

    async def retrieve_for_simulation(self, request: SimulationRunRequest) -> RagContext:
        context = await self.retrieve_context(
            topic=request.topic,
            student_level=request.student_level,
            language=request.language,
        )
        if self._settings.rag_require_context and not context.chunks:
            raise QdrantKnowledgeError(
                f"No educational context was retrieved for '{context.query}'. "
                "Populate the configured Qdrant collection or adjust the retrieval settings."
            )
        return context

    async def retrieve_context(
        self,
        *,
        topic: str,
        student_level: str,
        language: str,
        top_k: int | None = None,
    ) -> RagContext:
        return await self._retriever.retrieve_context(
            topic=topic,
            student_level=student_level,
            language=language,
            top_k=top_k,
        )

    def build_tutor_context(self, context: RagContext) -> str:
        return self._context_builder.build(context)

    @property
    def collection_name(self) -> str:
        return self._settings.qdrant_collection_name

    async def health(self) -> tuple[bool, bool, str]:
        return await self._client.health(self._settings.qdrant_collection_name)
