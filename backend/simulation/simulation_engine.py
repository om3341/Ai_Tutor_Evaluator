from __future__ import annotations

import time

from collections.abc import AsyncIterator, Callable

from backend.schemas import SimulationMessage, SimulationRunRequest
from backend.schemas.rag import RagContext
from backend.simulation.student_agent import StudentAgent
from backend.simulation.transcript_manager import TranscriptManager
from backend.simulation.tutor_agent import TutorAgent


class SimulationEngine:
    """Runs a multi-turn student/tutor dialogue with transcript memory."""

    def __init__(self, student_agent: StudentAgent, tutor_agent: TutorAgent) -> None:
        self._student_agent = student_agent
        self._tutor_agent = tutor_agent

    async def run(
        self,
        request: SimulationRunRequest,
        rag_context: RagContext,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[TranscriptManager, dict[str, float]]:
        transcript = TranscriptManager()
        latency_metrics: dict[str, float] = {}
        async for event_type, payload in self.run_stream(request, rag_context, should_stop=should_stop):
            if event_type == "message":
                transcript.append(payload)
            elif event_type == "latency_metrics":
                latency_metrics = payload

        return transcript, latency_metrics

    async def run_stream(
        self,
        request: SimulationRunRequest,
        rag_context: RagContext,
        should_stop: Callable[[], bool] | None = None,
    ) -> AsyncIterator[tuple[str, SimulationMessage | dict[str, float]]]:
        transcript = TranscriptManager()
        started_at = time.perf_counter()
        student_latency = 0.0
        tutor_latency = 0.0

        for turn_index in range(1, request.turns + 1):
            if should_stop and should_stop():
                break
            tutor_result = await self._tutor_agent.generate(request, transcript, turn_index, rag_context)
            tutor_latency += tutor_result.latency_ms
            tutor_message = transcript.add(
                role="tutor",
                content=tutor_result.content,
                turn_index=turn_index,
                latency_ms=tutor_result.latency_ms,
            )
            yield "message", tutor_message

            if should_stop and should_stop():
                break
            student_result = await self._student_agent.generate(request, transcript, turn_index)
            student_latency += student_result.latency_ms
            student_message = transcript.add(
                role="student",
                content=student_result.content,
                turn_index=turn_index,
                latency_ms=student_result.latency_ms,
            )
            yield "message", student_message

        yield "latency_metrics", {
            "student_generation_latency_ms": round(student_latency, 2),
            "tutor_generation_latency_ms": round(tutor_latency, 2),
            "total_simulation_latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
