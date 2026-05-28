from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from backend.analytics.hallucination import hallucination_analytics
from backend.analytics.insights import generate_insights
from backend.analytics.latency import latency_analytics, latency_history
from backend.analytics.multilingual import multilingual_analytics
from backend.database import crud
from backend.schemas.analytics import (
    AnalyticsOverviewResponse,
    HallucinationAnalyticsResponse,
    InsightsResponse,
    MultilingualAnalyticsResponse,
)
from backend.schemas.latency import LatencyAnalyticsResponse, LatencyHistoryResponse


class AnalyticsService:
    def __init__(self, default_limit: int = 1000) -> None:
        self._default_limit = default_limit

    async def overview(self, session: AsyncSession, limit: int | None = None) -> AnalyticsOverviewResponse:
        evaluations = list(await crud.list_analytics_evaluations(session, limit=limit or self._default_limit))
        return AnalyticsOverviewResponse(
            generated_at=datetime.now(UTC),
            total_evaluations=len(evaluations),
            total_model_samples=len(evaluations) * 2,
            languages=sorted({evaluation.language for evaluation in evaluations}),
            models=sorted({model for evaluation in evaluations for model in (evaluation.model_a, evaluation.model_b)}),
        )

    async def multilingual(
        self,
        session: AsyncSession,
        limit: int | None = None,
    ) -> MultilingualAnalyticsResponse:
        evaluations = list(await crud.list_analytics_evaluations(session, limit=limit or self._default_limit))
        return multilingual_analytics(evaluations)

    async def hallucinations(
        self,
        session: AsyncSession,
        limit: int | None = None,
    ) -> HallucinationAnalyticsResponse:
        evaluations = list(await crud.list_analytics_evaluations(session, limit=limit or self._default_limit))
        return hallucination_analytics(evaluations)

    async def latency(self, session: AsyncSession, limit: int | None = None) -> LatencyAnalyticsResponse:
        evaluations = list(await crud.list_analytics_evaluations(session, limit=limit or self._default_limit))
        return latency_analytics(evaluations)

    async def latency_history(self, session: AsyncSession, limit: int = 100) -> LatencyHistoryResponse:
        evaluations = list(await crud.list_analytics_evaluations(session, limit=limit))
        return LatencyHistoryResponse(generated_at=datetime.now(UTC), rows=latency_history(evaluations))

    async def insights(self, session: AsyncSession, limit: int | None = None) -> InsightsResponse:
        evaluations = list(await crud.list_analytics_evaluations(session, limit=limit or self._default_limit))
        multilingual = multilingual_analytics(evaluations)
        latency = latency_analytics(evaluations)
        hallucinations = hallucination_analytics(evaluations)
        return generate_insights(multilingual, latency, hallucinations)
