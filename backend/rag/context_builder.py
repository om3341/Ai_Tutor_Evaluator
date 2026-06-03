from __future__ import annotations

from backend.schemas.rag import RagContext


class RagContextBuilder:
    """Formats frozen retrieval results for a grounded tutor prompt."""

    def build(self, context: RagContext) -> str:
        if not context.chunks:
            return "No retrieved knowledge chunks were available."
        sections = []
        for index, chunk in enumerate(context.chunks, start=1):
            sections.append(f"[Chunk {index} | id={chunk.chunk_id} | score={chunk.score:.4f}]\n{chunk.text}")
        return "\n\n".join(sections)
