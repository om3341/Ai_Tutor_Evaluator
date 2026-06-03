from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import AsyncSessionLocal, get_db_session
from backend.judge import GeminiAPIError, GeminiConfigurationError, GeminiInvalidResponseError, GeminiTimeoutError
from backend.rag import QdrantKnowledgeError, RetrievalService
from backend.schemas import (
    PersonaDescription,
    RagContext,
    RagHealthResponse,
    RagRetrieveRequest,
    SimulationLoadRequest,
    SimulationRunRequest,
    SimulationRunResponse,
    SimulationRunSummary,
    SimulationState,
)
from backend.services.model_process_manager import ModelLoadError, ModelProcessManager
from backend.simulation import PERSONA_PROFILES, ConversationEvaluationPipeline
from backend.simulation.model_gateway import SimulationGenerationError

router = APIRouter(prefix="/simulations", tags=["student-tutor-simulations"])


def get_process_manager(request: Request) -> ModelProcessManager:
    return request.app.state.model_process_manager


def get_pipeline(request: Request) -> ConversationEvaluationPipeline:
    return request.app.state.conversation_pipeline


def get_retrieval_service(request: Request) -> RetrievalService:
    return request.app.state.retrieval_service


@router.get("/personas", response_model=list[PersonaDescription])
async def personas() -> list[PersonaDescription]:
    return [PersonaDescription(**profile.__dict__) for profile in PERSONA_PROFILES.values()]


@router.get("/rag/health", response_model=RagHealthResponse)
async def rag_health(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> RagHealthResponse:
    reachable, collection_exists, detail = await retrieval_service.health()
    return RagHealthResponse(
        reachable=reachable,
        collection_name=retrieval_service.collection_name,
        collection_exists=collection_exists,
        detail=detail,
    )


@router.post("/rag/retrieve", response_model=RagContext)
async def retrieve_rag_context(
    payload: RagRetrieveRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> RagContext:
    try:
        return await retrieval_service.retrieve_context(
            topic=payload.topic,
            student_level=payload.student_level,
            language=payload.language,
            top_k=payload.top_k,
        )
    except QdrantKnowledgeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/state", response_model=SimulationState)
async def state(request: Request) -> SimulationState:
    return SimulationState(
        tutor_model=getattr(request.app.state, "active_tutor_model", None),
        student_model=getattr(request.app.state, "active_student_model", None),
        tutor_loaded=bool(getattr(request.app.state, "active_tutor_model", None)),
        student_loaded=bool(getattr(request.app.state, "active_student_model", None)),
        running=bool(getattr(request.app.state, "simulation_running", False)),
    )


@router.post("/load", response_model=SimulationState)
async def load_models(
    payload: SimulationLoadRequest,
    request: Request,
    process_manager: ModelProcessManager = Depends(get_process_manager),
) -> SimulationState:
    try:
        if payload.tutor_model:
            await process_manager.load(payload.tutor_model)
            request.app.state.active_tutor_model = payload.tutor_model
        if payload.student_model:
            await process_manager.load(payload.student_model)
            request.app.state.active_student_model = payload.student_model
    except ModelLoadError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return await state(request)


@router.post("/unload", response_model=SimulationState)
async def unload_models(
    request: Request,
    process_manager: ModelProcessManager = Depends(get_process_manager),
) -> SimulationState:
    tutor_model = getattr(request.app.state, "active_tutor_model", None)
    student_model = getattr(request.app.state, "active_student_model", None)
    if tutor_model:
        process_manager.unload(tutor_model)
    if student_model and student_model != tutor_model:
        process_manager.unload(student_model)
    request.app.state.active_tutor_model = None
    request.app.state.active_student_model = None
    request.app.state.simulation_running = False
    return await state(request)


@router.post("/tutor/unload", response_model=SimulationState)
async def unload_tutor_model(
    request: Request,
    process_manager: ModelProcessManager = Depends(get_process_manager),
) -> SimulationState:
    tutor_model = getattr(request.app.state, "active_tutor_model", None)
    student_model = getattr(request.app.state, "active_student_model", None)
    request.app.state.active_tutor_model = None
    if tutor_model and tutor_model != student_model:
        process_manager.unload(tutor_model)
    if not getattr(request.app.state, "active_student_model", None):
        request.app.state.simulation_running = False
    return await state(request)


@router.post("/student/unload", response_model=SimulationState)
async def unload_student_model(
    request: Request,
    process_manager: ModelProcessManager = Depends(get_process_manager),
) -> SimulationState:
    tutor_model = getattr(request.app.state, "active_tutor_model", None)
    student_model = getattr(request.app.state, "active_student_model", None)
    request.app.state.active_student_model = None
    if student_model and student_model != tutor_model:
        process_manager.unload(student_model)
    if not getattr(request.app.state, "active_tutor_model", None):
        request.app.state.simulation_running = False
    return await state(request)


@router.post("/stop", response_model=SimulationState)
async def stop_simulation(request: Request) -> SimulationState:
    request.app.state.simulation_running = False
    return await state(request)


@router.post("/run", response_model=SimulationRunResponse)
async def run_simulation(
    payload: SimulationRunRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    pipeline: ConversationEvaluationPipeline = Depends(get_pipeline),
) -> SimulationRunResponse:
    if getattr(request.app.state, "simulation_running", False):
        raise HTTPException(status_code=409, detail="A simulation is already running.")
    if getattr(request.app.state, "active_tutor_model", None) != payload.tutor_model:
        raise HTTPException(status_code=409, detail=f"Load tutor model '{payload.tutor_model}' before running.")
    if getattr(request.app.state, "active_student_model", None) != payload.student_model:
        raise HTTPException(status_code=409, detail=f"Load student model '{payload.student_model}' before running.")

    request.app.state.simulation_running = True
    try:
        return await pipeline.run(
            session,
            payload,
            should_stop=lambda: not bool(getattr(request.app.state, "simulation_running", False)),
        )
    except SimulationGenerationError as exc:
        logger.error("Simulation generation error: {}", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except QdrantKnowledgeError as exc:
        logger.error("Simulation retrieval error: {}", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GeminiConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except GeminiTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except GeminiInvalidResponseError as exc:
        raise HTTPException(status_code=502, detail=f"Gemini transcript judge returned invalid JSON: {exc}") from exc
    except GeminiAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        logger.exception("Simulation database error: {}", exc)
        raise HTTPException(status_code=500, detail=f"Simulation could not be saved. Run alembic upgrade head. {exc}") from exc
    finally:
        request.app.state.simulation_running = False


@router.post("/run-stream")
async def run_simulation_stream(
    payload: SimulationRunRequest,
    request: Request,
    pipeline: ConversationEvaluationPipeline = Depends(get_pipeline),
) -> StreamingResponse:
    if getattr(request.app.state, "simulation_running", False):
        raise HTTPException(status_code=409, detail="A simulation is already running.")
    if getattr(request.app.state, "active_tutor_model", None) != payload.tutor_model:
        raise HTTPException(status_code=409, detail=f"Load tutor model '{payload.tutor_model}' before running.")
    if getattr(request.app.state, "active_student_model", None) != payload.student_model:
        raise HTTPException(status_code=409, detail=f"Load student model '{payload.student_model}' before running.")

    request.app.state.simulation_running = True

    async def event_stream() -> AsyncIterator[str]:
        try:
            async with AsyncSessionLocal() as stream_session:
                async for event in pipeline.stream_run(
                    stream_session,
                    payload,
                    should_stop=lambda: not bool(getattr(request.app.state, "simulation_running", False)),
                ):
                    yield json.dumps(event, ensure_ascii=False) + "\n"
        except SimulationGenerationError as exc:
            logger.error("Simulation stream generation error: {}", exc)
            yield _stream_error(str(exc), status_code=502)
        except QdrantKnowledgeError as exc:
            logger.error("Simulation stream retrieval error: {}", exc)
            yield _stream_error(str(exc), status_code=502)
        except ValueError as exc:
            yield _stream_error(str(exc), status_code=409)
        except GeminiConfigurationError as exc:
            yield _stream_error(str(exc), status_code=500)
        except GeminiTimeoutError as exc:
            yield _stream_error(str(exc), status_code=504)
        except GeminiInvalidResponseError as exc:
            yield _stream_error(f"Gemini transcript judge returned invalid JSON: {exc}", status_code=502)
        except GeminiAPIError as exc:
            yield _stream_error(str(exc), status_code=502)
        except SQLAlchemyError as exc:
            logger.exception("Simulation stream database error: {}", exc)
            yield _stream_error(f"Simulation could not be saved. Run alembic upgrade head. {exc}", status_code=500)
        except Exception as exc:
            logger.exception("Unexpected simulation stream error: {}", exc)
            yield _stream_error(f"Unexpected simulation stream error: {exc}", status_code=500)
        finally:
            request.app.state.simulation_running = False

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


def _stream_error(detail: str, *, status_code: int) -> str:
    return json.dumps({"event": "error", "status_code": status_code, "detail": detail}, ensure_ascii=False) + "\n"


@router.get("/runs", response_model=list[SimulationRunSummary])
async def list_runs(
    limit: int = 25,
    session: AsyncSession = Depends(get_db_session),
    pipeline: ConversationEvaluationPipeline = Depends(get_pipeline),
) -> list[SimulationRunSummary]:
    return await pipeline.list_runs(session, limit)


@router.get("/runs/{run_id}", response_model=SimulationRunResponse)
async def get_run(
    run_id: UUID,
    session: AsyncSession = Depends(get_db_session),
    pipeline: ConversationEvaluationPipeline = Depends(get_pipeline),
) -> SimulationRunResponse:
    result = await pipeline.get_run(session, run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Simulation run not found.")
    return result
