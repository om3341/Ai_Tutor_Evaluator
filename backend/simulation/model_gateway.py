from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from backend.config import Settings
from backend.text_cleaning import strip_thinking_tags


class SimulationGenerationError(RuntimeError):
    """Raised when a simulation agent model cannot generate a usable message."""


@dataclass(frozen=True)
class ChatResult:
    content: str
    latency_ms: float
    provider_model_name: str


class ModelGateway:
    """Small role-aware chat client for vLLM/OpenAI-compatible and Ollama models."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def chat(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        normalized = model_name.casefold()
        if "qwen 3.5" in normalized or "qwen3.5" in normalized or "2b student" in normalized:
            return await self._openai_chat(
                base_url=self._settings.qwen_student_base_url.rstrip("/"),
                provider_model_name=self._settings.qwen_student_model_name,
                timeout_seconds=self._settings.qwen_student_timeout_seconds,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=min(max_tokens, self._settings.qwen_student_max_tokens),
            )
        if "qwen" in normalized:
            return await self._openai_chat(
                base_url=self._settings.qwen_base_url.rstrip("/"),
                provider_model_name=self._settings.qwen_model_name,
                timeout_seconds=self._settings.qwen_timeout_seconds,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if "sarvam" in normalized:
            return await self._openai_chat(
                base_url=getattr(self._settings, "sarvam_base_url", "http://127.0.0.1:8003/v1").rstrip("/"),
                provider_model_name=getattr(self._settings, "sarvam_model_name", "sarvamai/sarvam-1-v0.5"),
                timeout_seconds=getattr(self._settings, "sarvam_timeout_seconds", 120.0),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if "gemma" in normalized:
            return await self._ollama_chat(
                base_url=self._settings.gemma_base_url.rstrip("/"),
                provider_model_name=self._settings.gemma_model_name,
                timeout_seconds=self._settings.gemma_timeout_seconds,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if "llama" in normalized:
            return await self._ollama_chat(
                base_url=self._settings.llama_base_url.rstrip("/"),
                provider_model_name=self._settings.llama_model_name,
                timeout_seconds=self._settings.llama_timeout_seconds,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        raise SimulationGenerationError(f"No simulation chat adapter configured for {model_name}.")

    async def _openai_chat(
        self,
        *,
        base_url: str,
        provider_model_name: str,
        timeout_seconds: float,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        payload = {
            "model": provider_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started_at = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(f"{base_url}/chat/completions", json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:1000]
            raise SimulationGenerationError(
                f"OpenAI-compatible model call failed for {provider_model_name}: "
                f"{exc.response.status_code} {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise SimulationGenerationError(f"OpenAI-compatible model call failed for {provider_model_name}: {exc}") from exc
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SimulationGenerationError(f"{provider_model_name} returned an invalid chat payload.") from exc
        return ChatResult(content=self._clean(content), latency_ms=latency_ms, provider_model_name=provider_model_name)

    async def _ollama_chat(
        self,
        *,
        base_url: str,
        provider_model_name: str,
        timeout_seconds: float,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        payload = {
            "model": provider_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        started_at = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(f"{base_url}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:1000]
            raise SimulationGenerationError(
                f"Ollama model call failed for {provider_model_name}: {exc.response.status_code} {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise SimulationGenerationError(f"Ollama model call failed for {provider_model_name}: {exc}") from exc
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        data: dict[str, Any] = response.json()
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise SimulationGenerationError(f"{provider_model_name} returned an invalid Ollama chat payload.") from exc
        return ChatResult(content=self._clean(content), latency_ms=latency_ms, provider_model_name=provider_model_name)

    @staticmethod
    def _clean(content: Any) -> str:
        if not isinstance(content, str):
            raise SimulationGenerationError("Model response content was not text.")
        clean = strip_thinking_tags(content).strip()
        if not clean:
            raise SimulationGenerationError("Model returned no visible text after cleaning.")
        return clean
