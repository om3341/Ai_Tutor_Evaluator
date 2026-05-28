from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import crud
from backend.database.connection import get_db_session
from backend.judge import GeminiAPIError, GeminiConfigurationError, GeminiInvalidResponseError, GeminiJudge, GeminiTimeoutError
from backend.schemas.leaderboard import EvaluationHistoryItem, LeaderboardEntry, ModelStats, RecentPerformancePoint
from backend.services import RankingService


router = APIRouter(tags=["leaderboard"])


def get_ranking_service(request: Request) -> RankingService:
    return request.app.state.ranking_service


def get_judge(request: Request) -> GeminiJudge:
    return request.app.state.judge


@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def leaderboard(
    limit: int | None = Query(default=None, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
    ranking_service: RankingService = Depends(get_ranking_service),
) -> list[LeaderboardEntry]:
    try:
        return await ranking_service.get_leaderboard(session, limit=limit)
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        raise database_unavailable(exc) from exc


@router.get("/leaderboard/top", response_model=list[LeaderboardEntry])
async def top_leaderboard(
    limit: int = Query(default=10, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
    ranking_service: RankingService = Depends(get_ranking_service),
) -> list[LeaderboardEntry]:
    try:
        return await ranking_service.get_leaderboard(session, limit=limit)
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        raise database_unavailable(exc) from exc


@router.delete("/leaderboard/reset")
async def reset_leaderboard(
    confirm: bool = Query(default=False),
    session: AsyncSession = Depends(get_db_session),
    ranking_service: RankingService = Depends(get_ranking_service),
) -> dict[str, int | str]:
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pass confirm=true to clear testing leaderboard and evaluation data.",
        )

    try:
        deleted = await ranking_service.clear_testing_data(session)
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        raise database_unavailable(exc) from exc
    return {"status": "cleared", **deleted}


@router.get("/models/{model_name}", response_model=ModelStats)
async def model_stats(
    model_name: str,
    recent_limit: int = Query(default=10, ge=1, le=50),
    session: AsyncSession = Depends(get_db_session),
    ranking_service: RankingService = Depends(get_ranking_service),
) -> ModelStats:
    try:
        stats = await ranking_service.get_model_stats(session, model_name, recent_limit=recent_limit)
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        raise database_unavailable(exc) from exc
    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' has no leaderboard entry.",
        )
    return stats


@router.get("/models/{model_name}/recent-performance", response_model=list[RecentPerformancePoint])
async def model_recent_performance(
    model_name: str,
    limit: int = Query(default=25, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
    ranking_service: RankingService = Depends(get_ranking_service),
) -> list[RecentPerformancePoint]:
    try:
        return await ranking_service.get_recent_performance(session, model_name, limit=limit)
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        raise database_unavailable(exc) from exc


@router.get("/evaluations/history", response_model=list[EvaluationHistoryItem])
async def evaluation_history(
    limit: int = Query(default=50, ge=1, le=200),
    model_name: str | None = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
    ranking_service: RankingService = Depends(get_ranking_service),
) -> list[EvaluationHistoryItem]:
    try:
        return await ranking_service.get_history(session, limit=limit, model_name=model_name)
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        raise database_unavailable(exc) from exc


@router.post("/evaluations/{evaluation_id}/benchmark-report")
async def generate_benchmark_report(
    evaluation_id: UUID,
    force: bool = Query(default=False),
    session: AsyncSession = Depends(get_db_session),
    judge: GeminiJudge = Depends(get_judge),
) -> dict[str, str]:
    try:
        evaluation = await crud.get_evaluation(session, evaluation_id)
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        raise database_unavailable(exc) from exc

    if evaluation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation not found.")

    existing_report = (evaluation.evaluation_json or {}).get("benchmark_report_markdown")
    if existing_report and not force:
        return {"benchmark_report_markdown": existing_report, "source": "cached"}

    try:
        report = await judge.generate_benchmark_report(evaluation)
        await crud.save_benchmark_report(session, evaluation, report)
        await session.commit()
    except GeminiConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except GeminiTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except (GeminiInvalidResponseError, GeminiAPIError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except (SQLAlchemyError, OSError, ConnectionError) as exc:
        raise database_unavailable(exc) from exc

    return {"benchmark_report_markdown": report, "source": "generated"}


def database_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"Database is not reachable: {exc}",
    )
