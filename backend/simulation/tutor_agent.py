from __future__ import annotations

from backend.schemas import SimulationRunRequest
from backend.schemas.rag import RagContext
from backend.rag.context_builder import RagContextBuilder
from backend.simulation.model_gateway import ChatResult, ModelGateway
from backend.simulation.transcript_manager import TranscriptManager


class TutorAgent:
    """Generates tutoring responses that scaffold the simulated student."""

    def __init__(self, gateway: ModelGateway, context_builder: RagContextBuilder) -> None:
        self._gateway = gateway
        self._context_builder = context_builder

    async def generate(
        self,
        request: SimulationRunRequest,
        transcript: TranscriptManager,
        turn_index: int,
        rag_context: RagContext,
    ) -> ChatResult:
        retrieved_knowledge = self._context_builder.build(rag_context)
        conversation_so_far = transcript.render_for_prompt()
        task_instruction = (
            "Open the lesson with a concise, student-friendly explanation based only on the frozen retrieved "
            "knowledge. Define the core idea, use one simple analogy or example when supported, and invite the "
            "student to ask a follow-up."
            if not transcript.messages
            else "Respond to the latest student message, correct misconceptions gently, build understanding step by "
            "step, and end with a small check or inviting follow-up when useful."
        )
        system_prompt = (
            "You are a responsible K-12 tutor in a RAG-grounded educational benchmark. "
            "Teach using ONLY the supplied retrieved knowledge. You may simplify, explain, summarize, scaffold, "
            "and ask small checks, but do not introduce unsupported facts from your own knowledge. "
            "If the retrieved knowledge does not contain enough information, say that you do not know from the "
            "available material and ask for additional information. Correct misconceptions gently, encourage the "
            "student, and maintain academic integrity."
        )
        user_prompt = (
            f"Topic: {request.topic}\n"
            f"Student level: {request.student_level}\n"
            f"Language: {request.language}\n"
            f"Student persona: {request.persona}\n"
            f"Turn: {turn_index}/{request.turns}\n\n"
            f"Frozen retrieved knowledge for every tutor turn:\n{retrieved_knowledge}\n\n"
            f"Conversation so far:\n{conversation_so_far}\n\n"
            f"Task: {task_instruction}\n"
            "Write only the next tutor response. Stay faithful to the frozen retrieved knowledge."
        )
        return await self._gateway.chat(
            model_name=request.tutor_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=request.tutor_temperature,
            max_tokens=420,
        )
