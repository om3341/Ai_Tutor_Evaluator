from __future__ import annotations

import sys

import orjson
from fastapi import Depends, FastAPI, status
from fastapi.responses import JSONResponse, ORJSONResponse
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api import analytics_router, benchmarks_router, generation_router, leaderboard_router
from backend.benchmark_runner import PairwiseBenchmarkRunner
from backend.config import Settings, get_settings
from backend.database.connection import get_db_session
from backend.judge import (
    GeminiAPIError,
    GeminiConfigurationError,
    GeminiInvalidResponseError,
    GeminiJudge,
    GeminiTimeoutError,
)
from backend.models.gemma import GemmaClient, LlamaClient
from backend.models.qwen import QwenClient
from backend.schemas import ErrorResponse, EvaluationRequest, EvaluationResponse
from backend.services import AnalyticsService, RankingService
from backend.services.benchmark_service import BenchmarkService
from backend.services.model_process_manager import ModelProcessManager
from backend.text_cleaning import strip_thinking_tags


class PrettyORJSONResponse(ORJSONResponse):
    """Fast JSON response using orjson while keeping local output readable."""

    def render(self, content: object) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_NON_STR_KEYS)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        default_response_class=PrettyORJSONResponse,
    )

    judge = GeminiJudge(settings=settings)
    runner = PairwiseBenchmarkRunner(judge=judge)
    app.state.judge = judge
    app.state.qwen_client = QwenClient(settings=settings)
    app.state.gemma_client = GemmaClient(settings=settings)
    app.state.llama_client = LlamaClient(settings=settings)
    app.state.active_benchmark_model = None
    app.state.model_process_manager = ModelProcessManager(settings=settings)
    app.state.benchmark_service = BenchmarkService(
        settings=settings,
        judge=judge,
        qwen_client=app.state.qwen_client,
        gemma_client=app.state.gemma_client,
        llama_client=app.state.llama_client,
    )
    app.state.ranking_service = RankingService(settings=settings)
    app.state.analytics_service = AnalyticsService()
    app.include_router(analytics_router)
    app.include_router(benchmarks_router)
    app.include_router(generation_router)
    app.include_router(leaderboard_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name, "version": settings.app_version}

    def get_runner() -> PairwiseBenchmarkRunner:
        return runner

    @app.post(
        "/evaluate",
        response_model=EvaluationResponse,
        responses={
            500: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
    )
    async def evaluate(
        request: EvaluationRequest,
        session: AsyncSession = Depends(get_db_session),
        benchmark_runner: PairwiseBenchmarkRunner = Depends(get_runner),
    ) -> EvaluationResponse | JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_410_GONE,
            content={
                "detail": "Pairwise /evaluate is retired. Use /benchmarks/model/load and /benchmarks/run.",
            },
        )
        clean_response_a = strip_thinking_tags(request.response_a)
        clean_response_b = strip_thinking_tags(request.response_b)
        if not clean_response_a or not clean_response_b:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "detail": "Candidate responses must contain visible text after removing <think> tags.",
                },
            )

        clean_request = request.model_copy(
            update={
                "response_a": clean_response_a,
                "response_b": clean_response_b,
            }
        )
        try:
            evaluation = await benchmark_runner.evaluate(clean_request)
        except GeminiConfigurationError as exc:
            logger.error("Gemini configuration error: {}", exc)
            return error_response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_code="GEMINI_CONFIGURATION_ERROR",
                detail=str(exc),
            )
        except GeminiTimeoutError as exc:
            logger.error("Gemini timeout: {}", exc)
            return error_response(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                error_code="GEMINI_TIMEOUT",
                detail=str(exc),
            )
        except GeminiInvalidResponseError as exc:
            logger.error("Gemini invalid response: {}", exc)
            return error_response(
                status_code=status.HTTP_502_BAD_GATEWAY,
                error_code="GEMINI_RESPONSE_INVALID",
                detail=str(exc),
            )
        except GeminiAPIError as exc:
            logger.error("Gemini API error: {}", exc)
            return error_response(
                status_code=status.HTTP_502_BAD_GATEWAY,
                error_code="GEMINI_API_ERROR",
                detail=str(exc),
            )

        try:
            persisted = await app.state.ranking_service.record_evaluation(session, clean_request, evaluation)
            return evaluation.model_copy(update={"evaluation_id": persisted.id})
        except (SQLAlchemyError, OSError, ConnectionError) as exc:
            logger.exception("Database error while saving evaluation: {}", exc)
            return error_response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_code="DATABASE_ERROR",
                detail=f"Evaluation was judged but could not be saved to the database: {exc}",
            )

    return app


def configure_logging(settings: Settings) -> None:
    """Configure one structured logger sink for the API process."""

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        serialize=True,
        backtrace=False,
        diagnose=False,
    )


def error_response(status_code: int, error_code: str, detail: str) -> JSONResponse:
    """Return documented error shape instead of FastAPI's nested detail format."""

    return JSONResponse(
        status_code=status_code,
        content={"error_code": error_code, "detail": detail},
    )


app = create_app()
