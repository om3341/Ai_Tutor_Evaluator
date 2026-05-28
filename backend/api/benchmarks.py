from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.benchmark_dataset import get_dataset
from backend.database.connection import get_db_session
from backend.database.models import BenchmarkRun
from backend.judge import GeminiAPIError, GeminiConfigurationError, GeminiInvalidResponseError, GeminiTimeoutError
from backend.models.gemma import GemmaGenerationError
from backend.models.qwen import QwenGenerationError
from backend.schemas import (
    BenchmarkDatasetItem,
    BenchmarkEvaluateCollectedRequest,
    BenchmarkLoadRequest,
    BenchmarkModelState,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    BenchmarkRunSummary,
)
from backend.services.benchmark_service import BenchmarkService, run_summary_from_model
from backend.services.model_process_manager import ModelLoadError, ModelProcessManager

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("/health")
async def benchmark_health() -> dict[str, str]:
    return {"status": "ok", "service": "benchmark analytics"}


def get_benchmark_service(request: Request) -> BenchmarkService:
    return request.app.state.benchmark_service


def get_model_process_manager(request: Request) -> ModelProcessManager:
    return request.app.state.model_process_manager


@router.get("/dataset", response_model=list[BenchmarkDatasetItem])
async def dataset(name: str = "k12_teacher_core_v1", max_items: int | None = None) -> list[BenchmarkDatasetItem]:
    try:
        return get_dataset(name, max_items)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/model", response_model=BenchmarkModelState)
async def active_model(request: Request) -> BenchmarkModelState:
    model_name = getattr(request.app.state, "active_benchmark_model", None)
    return BenchmarkModelState(active_model=model_name, loaded=bool(model_name))


@router.post("/model/load", response_model=BenchmarkModelState)
async def load_model(
    payload: BenchmarkLoadRequest,
    request: Request,
    process_manager: ModelProcessManager = Depends(get_model_process_manager),
) -> BenchmarkModelState:
    try:
        await process_manager.load(payload.model_name)
    except ModelLoadError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    request.app.state.active_benchmark_model = payload.model_name
    return BenchmarkModelState(active_model=payload.model_name, loaded=True)


@router.post("/model/unload", response_model=BenchmarkModelState)
async def unload_model(
    request: Request,
    process_manager: ModelProcessManager = Depends(get_model_process_manager),
) -> BenchmarkModelState:
    process_manager.unload(getattr(request.app.state, "active_benchmark_model", None))
    request.app.state.active_benchmark_model = None
    return BenchmarkModelState(active_model=None, loaded=False)


@router.post("/run", response_model=BenchmarkRunResponse)
async def run_benchmark(
    payload: BenchmarkRunRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkRunResponse:
    active = getattr(request.app.state, "active_benchmark_model", None)
    if active != payload.model_name:
        raise HTTPException(status_code=409, detail=f"Load '{payload.model_name}' before running the benchmark.")
    try:
        return await benchmark_service.run_benchmark(session, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (GemmaGenerationError, QwenGenerationError) as exc:
        logger.error("Benchmark generation error: {}", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except GeminiConfigurationError as exc:
        logger.error("Benchmark judge configuration error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except GeminiTimeoutError as exc:
        logger.error("Benchmark judge timeout: {}", exc)
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except GeminiInvalidResponseError as exc:
        logger.error("Benchmark judge invalid response: {}", exc)
        raise HTTPException(status_code=502, detail=f"Gemini judge returned invalid benchmark JSON: {exc}") from exc
    except GeminiAPIError as exc:
        logger.error("Benchmark judge API error: {}", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        logger.exception("Benchmark database error: {}", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark could not be saved. Did you run `alembic upgrade head`? {exc}",
        ) from exc


@router.post("/evaluate-collected", response_model=BenchmarkRunResponse)
async def evaluate_collected(
    payload: BenchmarkEvaluateCollectedRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),
) -> BenchmarkRunResponse:
    active = getattr(request.app.state, "active_benchmark_model", None)
    if active != payload.model_name:
        raise HTTPException(status_code=409, detail=f"Load '{payload.model_name}' before evaluating the benchmark.")
    try:
        return await benchmark_service.evaluate_collected_responses(session, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GeminiConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except GeminiTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except GeminiInvalidResponseError as exc:
        raise HTTPException(status_code=502, detail=f"Gemini judge returned invalid benchmark JSON: {exc}") from exc
    except GeminiAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        logger.exception("Benchmark database error: {}", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark could not be saved. Did you run `alembic upgrade head`? {exc}",
        ) from exc


@router.get("/runs", response_model=list[BenchmarkRunSummary])
async def runs(
    limit: int = 25,
    session: AsyncSession = Depends(get_db_session),
) -> list[BenchmarkRunSummary]:
    result = await session.execute(select(BenchmarkRun).order_by(BenchmarkRun.created_at.desc()).limit(limit))
    return [run_summary_from_model(run) for run in result.scalars().all()]


@router.get("/runs/{run_id}", response_model=BenchmarkRunSummary)
async def run_detail(
    run_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> BenchmarkRunSummary:
    run = await session.get(BenchmarkRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Benchmark run not found.")
    return run_summary_from_model(run)
