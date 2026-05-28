from __future__ import annotations

import time

from loguru import logger

from backend.judge import GeminiJudge
from backend.schemas import EvaluationRequest, EvaluationResponse, LatencyMetrics


class PairwiseBenchmarkRunner:
    """Coordinates one pairwise benchmark evaluation.

    This class is intentionally thin. It gives the API layer a stable service
    boundary and leaves room for later model-generation orchestration without
    coupling that future work to FastAPI route code.
    """

    def __init__(self, judge: GeminiJudge) -> None:
        self._judge = judge

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        total_start = time.perf_counter()

        judge_start = time.perf_counter()
        evaluation = await self._judge.evaluate_pairwise(request)
        judge_latency_ms = (time.perf_counter() - judge_start) * 1000
        total_latency_ms = (time.perf_counter() - total_start) * 1000

        logger.info(
            "Pairwise evaluation complete winner={} judge_latency_ms={:.2f} total_latency_ms={:.2f}",
            evaluation.winner.value,
            judge_latency_ms,
            total_latency_ms,
        )

        return EvaluationResponse(
            **evaluation.model_dump(),
            model_a=request.model_a,
            model_b=request.model_b,
            latency=LatencyMetrics(
                judge_latency_ms=round(judge_latency_ms, 2),
                total_latency_ms=round(total_latency_ms, 2),
            ),
        )
