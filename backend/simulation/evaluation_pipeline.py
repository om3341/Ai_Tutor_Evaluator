from __future__ import annotations

import uuid
import time
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import BenchmarkHistory, ConversationRun, EvaluationReport
from backend.judge import GeminiJudge
from backend.rag.retrieval_service import RetrievalService
from backend.schemas import (
    ConversationJudgeEvaluation,
    SimulationRunRequest,
    SimulationRunResponse,
    SimulationRunSummary,
    SimulationScores,
)
from backend.schemas.rag import RagContext
from backend.simulation.simulation_engine import SimulationEngine
from backend.simulation.transcript_manager import TranscriptManager


class ConversationEvaluationPipeline:
    """Runs simulation, judges the full transcript, and persists research artifacts."""

    def __init__(self, engine: SimulationEngine, judge: GeminiJudge, retrieval_service: RetrievalService) -> None:
        self._engine = engine
        self._judge = judge
        self._retrieval_service = retrieval_service

    async def run(
        self,
        session: AsyncSession,
        request: SimulationRunRequest,
        should_stop: Callable[[], bool] | None = None,
    ) -> SimulationRunResponse:
        retrieval_started_at = time.perf_counter()
        rag_context = await self._retrieval_service.retrieve_for_simulation(request)
        retrieval_latency_ms = round((time.perf_counter() - retrieval_started_at) * 1000, 2)
        transcript, latency_metrics = await self._engine.run(request, rag_context, should_stop=should_stop)
        latency_metrics["retrieval_latency_ms"] = retrieval_latency_ms
        if not transcript.messages:
            raise ValueError("Simulation was stopped before any transcript messages were generated.")
        evaluation = await self._judge.evaluate_conversation(request, transcript.messages, rag_context)
        return await self._persist_response(session, request, transcript, rag_context, latency_metrics, evaluation)

    async def stream_run(
        self,
        session: AsyncSession,
        request: SimulationRunRequest,
        should_stop: Callable[[], bool] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        retrieval_started_at = time.perf_counter()
        rag_context = await self._retrieval_service.retrieve_for_simulation(request)
        retrieval_latency_ms = round((time.perf_counter() - retrieval_started_at) * 1000, 2)
        yield {"event": "retrieval", "retrieval": rag_context.model_dump(mode="json")}

        transcript = TranscriptManager()
        latency_metrics: dict[str, float] = {}
        async for event_type, payload in self._engine.run_stream(request, rag_context, should_stop=should_stop):
            if event_type == "message":
                transcript.append(payload)
                yield {"event": "message", "message": payload.model_dump(mode="json")}
            elif event_type == "latency_metrics":
                latency_metrics = payload

        latency_metrics["retrieval_latency_ms"] = retrieval_latency_ms
        if not transcript.messages:
            raise ValueError("Simulation was stopped before any transcript messages were generated.")

        yield {"event": "judging", "message": "Conversation complete. Running transcript judge..."}
        evaluation = await self._judge.evaluate_conversation(request, transcript.messages, rag_context)
        response = await self._persist_response(session, request, transcript, rag_context, latency_metrics, evaluation)
        yield {"event": "complete", "result": response.model_dump(mode="json")}

    async def _persist_response(
        self,
        session: AsyncSession,
        request: SimulationRunRequest,
        transcript: TranscriptManager,
        rag_context: RagContext,
        latency_metrics: dict[str, float],
        evaluation: ConversationJudgeEvaluation,
    ) -> SimulationRunResponse:
        now = datetime.now(timezone.utc)

        run = ConversationRun(
            tutor_model=request.tutor_model,
            student_model=request.student_model,
            topic=request.topic,
            student_level=request.student_level,
            language=request.language,
            persona=request.persona,
            turns=request.turns,
            transcript_json=transcript.model_dump(),
            retrieved_chunks=[chunk.model_dump() for chunk in rag_context.chunks],
            retrieval_scores=[chunk.score for chunk in rag_context.chunks],
            chunk_ids=[chunk.chunk_id for chunk in rag_context.chunks],
            rag_context=rag_context.model_dump(),
            retrieval_metadata={
                "query": rag_context.query,
                "collection_name": rag_context.collection_name,
                "embedding_model": rag_context.embedding_model,
                "top_k": rag_context.top_k,
                "chunk_count": len(rag_context.chunks),
            },
            model_config={
                "tutor_temperature": request.tutor_temperature,
                "student_temperature": request.student_temperature,
            },
            latency_metrics=latency_metrics,
            benchmark_metadata={
                "benchmark_type": "rag_grounded_student_tutor_simulation",
                "context_frozen": True,
            },
            created_at=now,
        )
        session.add(run)
        await session.flush()

        report = EvaluationReport(
            run_id=run.id,
            report_type="conversation_judge",
            metrics=evaluation.model_dump(exclude={"reasoning", "failure_modes"}),
            reasoning=evaluation.reasoning,
            failure_modes=evaluation.failure_modes,
            judge_json=evaluation.model_dump(),
            created_at=now,
        )
        session.add(report)
        session.add(
            BenchmarkHistory(
                run_id=run.id,
                benchmark_type="rag_grounded_student_tutor_simulation",
                tutor_model=request.tutor_model,
                student_model=request.student_model,
                topic=request.topic,
                aggregate_metrics=report.metrics,
                created_at=now,
            )
        )
        await session.commit()
        await session.refresh(run)

        return self._to_response(run, evaluation)

    async def list_runs(self, session: AsyncSession, limit: int = 25) -> list[SimulationRunSummary]:
        result = await session.execute(
            select(ConversationRun).order_by(ConversationRun.created_at.desc()).limit(limit)
        )
        runs = result.scalars().all()
        summaries: list[SimulationRunSummary] = []
        for run in runs:
            report_result = await session.execute(select(EvaluationReport).where(EvaluationReport.run_id == run.id))
            report = report_result.scalars().first()
            summaries.append(self._summary_from_run(run, report.metrics if report else {}))
        return summaries

    async def get_run(self, session: AsyncSession, run_id: uuid.UUID) -> SimulationRunResponse | None:
        run = await session.get(ConversationRun, run_id)
        if run is None:
            return None
        result = await session.execute(select(EvaluationReport).where(EvaluationReport.run_id == run.id))
        report = result.scalars().first()
        if report is None:
            return None
        evaluation = ConversationJudgeEvaluation.model_validate(
            {**report.metrics, "reasoning": report.reasoning, "failure_modes": report.failure_modes}
        )
        return self._to_response(run, evaluation)

    def _to_response(self, run: ConversationRun, evaluation: ConversationJudgeEvaluation) -> SimulationRunResponse:
        retrieval = run.rag_context or {
            "query": f"{run.topic} {run.student_level} {run.language}",
            "collection_name": "legacy_unrecorded",
            "embedding_model": "legacy_unrecorded",
            "top_k": 0,
            "chunks": [],
        }
        return SimulationRunResponse(
            run_id=run.id,
            tutor_model=run.tutor_model,
            student_model=run.student_model,
            topic=run.topic,
            student_level=run.student_level,
            language=run.language,
            persona=run.persona,
            turns=run.turns,
            transcript=run.transcript_json,
            retrieval=RagContext.model_validate(retrieval),
            metrics=SimulationScores.model_validate(evaluation.model_dump(exclude={"reasoning", "failure_modes"})),
            reasoning=evaluation.reasoning,
            failure_modes=evaluation.failure_modes,
            latency_metrics=run.latency_metrics,
            created_at=run.created_at,
        )

    @staticmethod
    def _summary_from_run(run: ConversationRun, metrics: dict[str, Any]) -> SimulationRunSummary:
        return SimulationRunSummary(
            run_id=run.id,
            tutor_model=run.tutor_model,
            student_model=run.student_model,
            topic=run.topic,
            student_level=run.student_level,
            language=run.language,
            persona=run.persona,
            turns=run.turns,
            metrics=metrics,
            created_at=run.created_at,
        )
