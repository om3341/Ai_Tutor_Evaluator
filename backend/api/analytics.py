from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db_session
from backend.schemas.analytics import (
    AnalyticsOverviewResponse,
    HallucinationAnalyticsResponse,
    InsightsResponse,
    MultilingualAnalyticsResponse,
)
from backend.schemas.latency import LatencyAnalyticsResponse, LatencyHistoryResponse
from backend.services import AnalyticsService


router = APIRouter(prefix="/analytics", tags=["analytics"])


def get_analytics_service(request: Request) -> AnalyticsService:
    return request.app.state.analytics_service


@router.get("", response_model=AnalyticsOverviewResponse)
async def analytics_overview(
    limit: int = Query(default=1000, ge=1, le=5000),
    session: AsyncSession = Depends(get_db_session),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> AnalyticsOverviewResponse:
    try:
        return await analytics_service.overview(session, limit=limit)
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        raise database_unavailable(exc) from exc


@router.get("/multilingual", response_model=MultilingualAnalyticsResponse)
async def multilingual_analytics(
    limit: int = Query(default=1000, ge=1, le=5000),
    session: AsyncSession = Depends(get_db_session),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> MultilingualAnalyticsResponse:
    try:
        return await analytics_service.multilingual(session, limit=limit)
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        raise database_unavailable(exc) from exc


@router.get("/hallucinations", response_model=HallucinationAnalyticsResponse)
async def hallucination_analytics(
    limit: int = Query(default=1000, ge=1, le=5000),
    session: AsyncSession = Depends(get_db_session),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> HallucinationAnalyticsResponse:
    try:
        return await analytics_service.hallucinations(session, limit=limit)
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        raise database_unavailable(exc) from exc


@router.get("/latency", response_model=LatencyAnalyticsResponse)
async def latency_analytics(
    limit: int = Query(default=1000, ge=1, le=5000),
    session: AsyncSession = Depends(get_db_session),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> LatencyAnalyticsResponse:
    try:
        return await analytics_service.latency(session, limit=limit)
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        raise database_unavailable(exc) from exc


@router.get("/latency/models", response_model=LatencyAnalyticsResponse)
async def latency_models(
    limit: int = Query(default=1000, ge=1, le=5000),
    session: AsyncSession = Depends(get_db_session),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> LatencyAnalyticsResponse:
    try:
        return await analytics_service.latency(session, limit=limit)
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        raise database_unavailable(exc) from exc


@router.get("/latency/history", response_model=LatencyHistoryResponse)
async def latency_history(
    limit: int = Query(default=100, ge=1, le=1000),
    session: AsyncSession = Depends(get_db_session),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> LatencyHistoryResponse:
    try:
        return await analytics_service.latency_history(session, limit=limit)
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        raise database_unavailable(exc) from exc


@router.get("/insights", response_model=InsightsResponse)
async def benchmark_insights(
    limit: int = Query(default=1000, ge=1, le=5000),
    session: AsyncSession = Depends(get_db_session),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> InsightsResponse:
    try:
        return await analytics_service.insights(session, limit=limit)
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        raise database_unavailable(exc) from exc


def database_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"Database is not reachable: {exc}",
    )
