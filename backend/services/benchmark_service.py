from __future__ import annotations

import asyncio
import re
import time
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from backend.benchmark_dataset import get_dataset
from backend.config import Settings
from backend.database.models import BenchmarkItem, BenchmarkRun
from backend.judge import GeminiAPIError, GeminiJudge
from backend.models.gemma import GemmaClient, LlamaClient
from backend.models.qwen import QwenClient
from backend.schemas import (
    BenchmarkAggregate,
    BenchmarkCollectedItem,
    BenchmarkEvaluateCollectedRequest,
    BenchmarkItemResult,
    BenchmarkJudgeRequest,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    BenchmarkRunSummary,
    ModelGenerationRequest,
    SCORE_FIELDS,
)
from backend.text_cleaning import strip_thinking_tags


class BenchmarkService:
    """Runs full single-model educational benchmarks sequentially."""

    def __init__(
        self,
        settings: Settings,
        judge: GeminiJudge,
        qwen_client: QwenClient,
        gemma_client: GemmaClient,
        llama_client: LlamaClient,
    ) -> None:
        self._settings = settings
        self._judge = judge
        self._qwen_client = qwen_client
        self._gemma_client = gemma_client
        self._llama_client = llama_client

    async def run_benchmark(
        self,
        session: AsyncSession,
        request: BenchmarkRunRequest,
    ) -> BenchmarkRunResponse:
        dataset = get_dataset(request.dataset_name, request.max_items)
        item_results: list[BenchmarkItemResult] = []
        total_generation_latency_ms = 0.0
        total_judge_latency_ms = 0.0

        for item in dataset:
            generation_started = time.perf_counter()
            generation = await self._generate_response(
                model_name=request.model_name,
                student_prompt=item.student_prompt,
                student_level=item.student_level,
                language=item.language,
            )
            generation_latency_ms = generation.latency_ms or round((time.perf_counter() - generation_started) * 1000, 2)
            total_generation_latency_ms += generation_latency_ms

            response_text = strip_thinking_tags(generation.response)
            judge_request = BenchmarkJudgeRequest(
                student_prompt=item.student_prompt,
                student_level=item.student_level,
                language=item.language,
                subject=item.subject,
                rubric=item.rubric,
                model_name=request.model_name,
                response=response_text,
            )
            judge_started = time.perf_counter()
            judgment = await self._judge.evaluate_single(judge_request)
            judge_latency_ms = round((time.perf_counter() - judge_started) * 1000, 2)
            total_judge_latency_ms += judge_latency_ms

            item_results.append(
                BenchmarkItemResult(
                    item_id=item.item_id,
                    student_prompt=item.student_prompt,
                    student_level=item.student_level,
                    language=item.language,
                    subject=item.subject,
                    response=response_text,
                    scores=judgment.scores,
                    overall_score=judgment.overall_score,
                    learning_effectiveness=judgment.learning_effectiveness,
                    safety_classification=judgment.safety_classification,
                    reasoning=judgment.reasoning,
                    generation_latency_ms=generation_latency_ms,
                    judge_latency_ms=judge_latency_ms,
                )
            )

        aggregate = self._aggregate(request.model_name, request.dataset_name, item_results, total_generation_latency_ms, total_judge_latency_ms)
        report = self._render_report(aggregate, item_results)
        completed_at = datetime.now(UTC)
        run = await self._persist_run(
            session,
            model_name=request.model_name,
            dataset_name=request.dataset_name,
            item_results=item_results,
            aggregate=aggregate,
            report=report,
            completed_at=completed_at,
            total_generation_latency_ms=total_generation_latency_ms,
            total_judge_latency_ms=total_judge_latency_ms,
        )

        return BenchmarkRunResponse(
            run_id=run.id,
            model_name=request.model_name,
            dataset_name=request.dataset_name,
            status="completed",
            started_at=run.created_at,
            completed_at=completed_at,
            aggregate=aggregate,
            items=item_results,
            benchmark_report_markdown=report,
        )

    async def _persist_run(
        self,
        session: AsyncSession,
        *,
        model_name: str,
        dataset_name: str,
        item_results: list[BenchmarkItemResult],
        aggregate: BenchmarkAggregate,
        report: str,
        completed_at: datetime,
        total_generation_latency_ms: float,
        total_judge_latency_ms: float,
    ) -> BenchmarkRun:
        async with session.begin():
            run = BenchmarkRun(
                model_name=model_name,
                dataset_name=dataset_name,
                status="completed",
                items_completed=len(item_results),
                aggregate_scores=aggregate.model_dump(mode="json"),
                latency_metrics={
                    "total_generation_latency_ms": total_generation_latency_ms,
                    "total_judge_latency_ms": total_judge_latency_ms,
                    "avg_generation_latency_ms": aggregate.avg_generation_latency_ms,
                    "avg_judge_latency_ms": aggregate.avg_judge_latency_ms,
                },
                benchmark_report_markdown=report,
                completed_at=completed_at,
            )
            session.add(run)
            await session.flush()

            for result in item_results:
                session.add(
                    BenchmarkItem(
                        run_id=run.id,
                        dataset_item_id=result.item_id,
                        prompt=result.student_prompt,
                        student_level=result.student_level,
                        language=result.language,
                        subject=result.subject,
                        model_name=model_name,
                        response=result.response,
                        scores=result.scores.model_dump(mode="json"),
                        overall_score=result.overall_score,
                        learning_effectiveness=result.learning_effectiveness,
                        safety_classification=result.safety_classification,
                        judge_json=result.model_dump(mode="json"),
                        latency_metrics={
                            "generation_latency_ms": result.generation_latency_ms,
                            "judge_latency_ms": result.judge_latency_ms,
                        },
                    )
                )
            await session.flush()
            await session.refresh(run)
        return run

    async def evaluate_collected_responses(
        self,
        session: AsyncSession,
        request: BenchmarkEvaluateCollectedRequest,
    ) -> BenchmarkRunResponse:
        item_results: list[BenchmarkItemResult] = []
        total_generation_latency_ms = 0.0
        total_judge_latency_ms = 0.0

        for index, item in enumerate(request.items):
            await self._pace_judge(index)
            result, judge_latency_ms = await self._judge_collected_item(request.model_name, item)
            item_results.append(result)
            total_generation_latency_ms += item.generation_latency_ms
            total_judge_latency_ms += judge_latency_ms

        aggregate = self._aggregate(
            request.model_name,
            request.dataset_name,
            item_results,
            total_generation_latency_ms,
            total_judge_latency_ms,
        )
        report = self._render_report(aggregate, item_results)
        completed_at = datetime.now(UTC)
        run = await self._persist_run(
            session,
            model_name=request.model_name,
            dataset_name=request.dataset_name,
            item_results=item_results,
            aggregate=aggregate,
            report=report,
            completed_at=completed_at,
            total_generation_latency_ms=total_generation_latency_ms,
            total_judge_latency_ms=total_judge_latency_ms,
        )

        return BenchmarkRunResponse(
            run_id=run.id,
            model_name=request.model_name,
            dataset_name=request.dataset_name,
            status="completed",
            started_at=run.created_at,
            completed_at=completed_at,
            aggregate=aggregate,
            items=item_results,
            benchmark_report_markdown=report,
        )

    async def _judge_collected_item(
        self,
        model_name: str,
        item: BenchmarkCollectedItem,
    ) -> tuple[BenchmarkItemResult, float]:
        response_text = strip_thinking_tags(item.response)
        judge_request = BenchmarkJudgeRequest(
            student_prompt=item.student_prompt,
            student_level=item.student_level,
            language=item.language,
            subject=item.subject,
            rubric=item.rubric,
            model_name=model_name,
            response=response_text,
        )
        judge_started = time.perf_counter()
        judgment = await self._evaluate_single_with_quota_retry(judge_request)
        judge_latency_ms = round((time.perf_counter() - judge_started) * 1000, 2)
        return (
            BenchmarkItemResult(
                item_id=item.item_id,
                student_prompt=item.student_prompt,
                student_level=item.student_level,
                language=item.language,
                subject=item.subject,
                response=response_text,
                scores=judgment.scores,
                overall_score=judgment.overall_score,
                learning_effectiveness=judgment.learning_effectiveness,
                safety_classification=judgment.safety_classification,
                reasoning=judgment.reasoning,
                generation_latency_ms=item.generation_latency_ms,
                judge_latency_ms=judge_latency_ms,
            ),
            judge_latency_ms,
        )

    async def _evaluate_single_with_quota_retry(self, judge_request: BenchmarkJudgeRequest):
        try:
            return await self._judge.evaluate_single(judge_request)
        except GeminiAPIError as exc:
            retry_after = self._retry_delay_from_error(str(exc))
            if retry_after is None:
                raise
            await asyncio.sleep(retry_after + 1.0)
            return await self._judge.evaluate_single(judge_request)

    async def _pace_judge(self, item_index: int) -> None:
        if item_index == 0:
            return
        delay = self._settings.benchmark_judge_delay_seconds
        if delay > 0:
            await asyncio.sleep(delay)

    @staticmethod
    def _retry_delay_from_error(message: str) -> float | None:
        match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", message)
        if match:
            return float(match.group(1))
        match = re.search(r"Please retry in\s+([0-9.]+)s", message)
        if match:
            return float(match.group(1))
        if "429" in message or "quota" in message.casefold():
            return 60.0
        return None

    async def _generate_response(self, *, model_name: str, student_prompt: str, student_level: str, language: str):
        payload = ModelGenerationRequest(
            model_name=model_name,
            student_prompt=student_prompt,
            student_level=student_level,
            language=language,
        )
        normalized = model_name.casefold()
        if "qwen" in normalized:
            return await self._qwen_client.generate(payload)
        if "gemma" in normalized:
            return await self._gemma_client.generate(payload)
        if "llama" in normalized:
            return await self._llama_client.generate(payload)
        raise ValueError(f"Model generation is not connected for '{model_name}'.")

    @staticmethod
    def _aggregate(
        model_name: str,
        dataset_name: str,
        items: list[BenchmarkItemResult],
        total_generation_latency_ms: float,
        total_judge_latency_ms: float,
    ) -> BenchmarkAggregate:
        count = max(len(items), 1)
        averages = {
            field: round(sum(float(getattr(item.scores, field)) for item in items) / count, 2)
            for field in SCORE_FIELDS
        }
        return BenchmarkAggregate(
            model_name=model_name,
            dataset_name=dataset_name,
            items_completed=len(items),
            average_overall_score=round(sum(item.overall_score for item in items) / count, 2),
            avg_correctness=averages["correctness"],
            avg_teaching_quality=averages["teaching_quality"],
            avg_adaptation=averages["adaptation"],
            avg_emotional_intelligence=averages["emotional_intelligence"],
            avg_multilingual_quality=averages["multilingual_quality"],
            avg_hallucination_risk=averages["hallucination_risk"],
            avg_conversation_quality=averages["conversation_quality"],
            total_generation_latency_ms=round(total_generation_latency_ms, 2),
            total_judge_latency_ms=round(total_judge_latency_ms, 2),
            avg_generation_latency_ms=round(total_generation_latency_ms / count, 2),
            avg_judge_latency_ms=round(total_judge_latency_ms / count, 2),
        )

    @staticmethod
    def _render_report(aggregate: BenchmarkAggregate, items: list[BenchmarkItemResult]) -> str:
        rows = "\n".join(
            f"| {item.item_id} | {item.subject} | {item.language} | {item.overall_score} | "
            f"{item.learning_effectiveness} | {item.safety_classification} |"
            for item in items
        )
        return (
            f"# Benchmark Report: {aggregate.model_name}\n\n"
            f"Dataset: `{aggregate.dataset_name}`\n\n"
            "## Aggregate Scores\n\n"
            "| Metric | Score |\n| --- | ---: |\n"
            f"| Overall | {aggregate.average_overall_score} |\n"
            f"| Correctness | {aggregate.avg_correctness} |\n"
            f"| Teaching Quality | {aggregate.avg_teaching_quality} |\n"
            f"| Adaptation | {aggregate.avg_adaptation} |\n"
            f"| Emotional Intelligence | {aggregate.avg_emotional_intelligence} |\n"
            f"| Multilingual Quality | {aggregate.avg_multilingual_quality} |\n"
            f"| Low Hallucination / Safety | {aggregate.avg_hallucination_risk} |\n"
            f"| Conversation Quality | {aggregate.avg_conversation_quality} |\n\n"
            "## Item Results\n\n"
            "| Item | Subject | Language | Overall | Learning | Safety |\n| --- | --- | --- | ---: | --- | --- |\n"
            f"{rows}\n"
        )


def run_summary_from_model(run: BenchmarkRun) -> BenchmarkRunSummary:
    scores = run.aggregate_scores or {}
    return BenchmarkRunSummary(
        run_id=run.id,
        model_name=run.model_name,
        dataset_name=run.dataset_name,
        status=run.status,
        items_completed=run.items_completed,
        average_overall_score=float(scores.get("average_overall_score", 0.0)),
        avg_correctness=float(scores.get("avg_correctness", 0.0)),
        avg_teaching_quality=float(scores.get("avg_teaching_quality", 0.0)),
        avg_adaptation=float(scores.get("avg_adaptation", 0.0)),
        avg_emotional_intelligence=float(scores.get("avg_emotional_intelligence", 0.0)),
        avg_multilingual_quality=float(scores.get("avg_multilingual_quality", 0.0)),
        avg_hallucination_risk=float(scores.get("avg_hallucination_risk", 0.0)),
        avg_conversation_quality=float(scores.get("avg_conversation_quality", 0.0)),
        created_at=run.created_at,
        completed_at=run.completed_at,
    )
