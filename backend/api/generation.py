from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from backend.models.gemma import GemmaClient, GemmaGenerationError, LlamaClient, LlamaGenerationError
from backend.models.qwen import QwenClient, QwenGenerationError
from backend.schemas import ModelGenerationRequest, ModelGenerationResponse

router = APIRouter(tags=["model-generation"])


def get_qwen_client(request: Request) -> QwenClient:
    return request.app.state.qwen_client


def get_gemma_client(request: Request) -> GemmaClient:
    return request.app.state.gemma_client


def get_llama_client(request: Request) -> LlamaClient:
    return request.app.state.llama_client


@router.post("/models/qwen/generate", response_model=ModelGenerationResponse)
async def generate_qwen_response(
    payload: ModelGenerationRequest,
    request: Request,
) -> ModelGenerationResponse:
    try:
        return await get_qwen_client(request).generate(payload)
    except QwenGenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


@router.post("/models/llama/generate", response_model=ModelGenerationResponse)
async def generate_llama_response(
    payload: ModelGenerationRequest,
    request: Request,
) -> ModelGenerationResponse:
    try:
        return await get_llama_client(request).generate(payload)
    except LlamaGenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


@router.post("/models/gemma/generate", response_model=ModelGenerationResponse)
async def generate_gemma_response(
    payload: ModelGenerationRequest,
    request: Request,
) -> ModelGenerationResponse:
    try:
        return await get_gemma_client(request).generate(payload)
    except GemmaGenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
