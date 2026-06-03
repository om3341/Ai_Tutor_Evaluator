from __future__ import annotations

import asyncio
from typing import Any

from backend.config import Settings
from backend.schemas.rag import RagChunk


class QdrantKnowledgeError(RuntimeError):
    """Raised when the educational knowledge collection cannot be queried."""


class QdrantKnowledgeClient:
    """Async facade over Qdrant's FastEmbed-enabled Python client."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise QdrantKnowledgeError(
                "Qdrant client is not installed. Run pip install -r requirements.txt."
            ) from exc
        self._client = QdrantClient(
            url=self._settings.qdrant_url,
            api_key=self._settings.qdrant_api_key or None,
            timeout=10,
        )
        return self._client

    async def collection_exists(self, collection_name: str) -> bool:
        try:
            return bool(await asyncio.to_thread(self._get_client().collection_exists, collection_name))
        except Exception as exc:
            raise QdrantKnowledgeError(f"Could not reach Qdrant at {self._settings.qdrant_url}: {exc}") from exc

    async def ensure_collection(self, collection_name: str, embedding_model: str) -> None:
        if await self.collection_exists(collection_name):
            return
        try:
            from qdrant_client import models
            client = self._get_client()
            await asyncio.to_thread(
                client.create_collection,
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=client.get_embedding_size(embedding_model),
                    distance=models.Distance.COSINE,
                ),
            )
        except Exception as exc:
            raise QdrantKnowledgeError(f"Could not create Qdrant collection '{collection_name}': {exc}") from exc

    async def query(
        self,
        *,
        collection_name: str,
        query_text: str,
        embedding_model: str,
        text_payload_key: str,
        top_k: int,
        score_threshold: float | None,
        subject: str | None = None,
    ) -> list[RagChunk]:
        if not await self.collection_exists(collection_name):
            raise QdrantKnowledgeError(
                f"Qdrant collection '{collection_name}' does not exist. "
                "Create and populate it before running a grounded simulation."
            )
        try:
            from qdrant_client import models
            query_filter = None
            if subject:
                query_filter = models.Filter(
                    must=[models.FieldCondition(key="subject", match=models.MatchValue(value=subject))]
                )
            response = await asyncio.to_thread(
                self._get_client().query_points,
                collection_name=collection_name,
                query=models.Document(text=query_text, model=embedding_model),
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            raise QdrantKnowledgeError(f"Qdrant retrieval failed for '{collection_name}': {exc}") from exc

        chunks: list[RagChunk] = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = payload.get(text_payload_key) or payload.get("text") or payload.get("content")
            if not isinstance(text, str) or not text.strip():
                continue
            chunks.append(
                RagChunk(
                    chunk_id=str(point.id),
                    text=text.strip(),
                    score=float(point.score),
                    metadata={key: value for key, value in payload.items() if key != text_payload_key},
                )
            )
        return chunks

    async def upload_documents(
        self,
        *,
        collection_name: str,
        embedding_model: str,
        text_payload_key: str,
        documents: list[str],
        metadata: list[dict[str, Any]],
        ids: list[str | int],
    ) -> None:
        if not (len(documents) == len(metadata) == len(ids)):
            raise ValueError("documents, metadata, and ids must have the same length.")
        await self.ensure_collection(collection_name, embedding_model)
        try:
            from qdrant_client import models
            payload = [
                {text_payload_key: document, **item_metadata}
                for document, item_metadata in zip(documents, metadata, strict=True)
            ]
            await asyncio.to_thread(
                self._get_client().upload_collection,
                collection_name=collection_name,
                vectors=[models.Document(text=document, model=embedding_model) for document in documents],
                payload=payload,
                ids=ids,
            )
        except Exception as exc:
            raise QdrantKnowledgeError(f"Could not upload documents to '{collection_name}': {exc}") from exc

    async def health(self, collection_name: str) -> tuple[bool, bool, str]:
        try:
            exists = await self.collection_exists(collection_name)
            detail = "Qdrant is reachable." if exists else f"Collection '{collection_name}' does not exist."
            return True, exists, detail
        except QdrantKnowledgeError as exc:
            return False, False, str(exc)
