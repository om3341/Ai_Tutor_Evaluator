from __future__ import annotations

import asyncio
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import google.generativeai as genai
from loguru import logger
from pydantic import ValidationError

from backend.config import Settings
from backend.database.models import Evaluation
from backend.schemas import BenchmarkJudgeRequest, EvaluationRequest, JudgeEvaluation, SingleJudgeEvaluation


class GeminiJudgeError(RuntimeError):
    """Base class for expected Gemini judge failures."""


class GeminiConfigurationError(GeminiJudgeError):
    """Raised when required Gemini configuration is missing."""


class GeminiTimeoutError(GeminiJudgeError):
    """Raised when the judge call exceeds the configured timeout."""


class GeminiInvalidResponseError(GeminiJudgeError):
    """Raised when Gemini returns malformed or schema-invalid JSON."""


class GeminiAPIError(GeminiJudgeError):
    """Raised when Gemini fails after retry attempts."""


class GeminiJudge:
    """Gemini-backed pairwise evaluator for AI teacher responses.

    The Google Generative AI SDK is synchronous today, so this service keeps an
    async public API and runs the blocking SDK call in a worker thread. That
    preserves FastAPI responsiveness while avoiding extra infrastructure.
    """

    def __init__(self, settings: Settings, prompt_path: Path | None = None) -> None:
        self._settings = settings
        self._prompt_path = prompt_path or Path(__file__).resolve().parent / "prompts" / "judge_prompt.txt"
        self._report_prompt_path = Path(__file__).resolve().parent / "prompts" / "benchmark_report_prompt.txt"
        self._system_prompt = self._prompt_path.read_text(encoding="utf-8")
        self._report_system_prompt = self._report_prompt_path.read_text(encoding="utf-8")
        self._model: genai.GenerativeModel | None = None
        self._report_model: genai.GenerativeModel | None = None

    def _get_model(self) -> genai.GenerativeModel:
        if not self._settings.gemini_api_key:
            raise GeminiConfigurationError("GEMINI_API_KEY is not configured.")

        if self._model is None:
            genai.configure(api_key=self._settings.gemini_api_key)
            self._model = genai.GenerativeModel(
                model_name=self._settings.gemini_model_name,
                system_instruction=self._system_prompt,
            )
        return self._model

    def _get_report_model(self) -> genai.GenerativeModel:
        if not self._settings.gemini_api_key:
            raise GeminiConfigurationError("GEMINI_API_KEY is not configured.")

        if self._report_model is None:
            genai.configure(api_key=self._settings.gemini_api_key)
            self._report_model = genai.GenerativeModel(
                model_name=self._settings.gemini_model_name,
                system_instruction=self._report_system_prompt,
            )
        return self._report_model

    async def evaluate_pairwise(self, request: EvaluationRequest) -> JudgeEvaluation:
        """Compare response A vs B and return a validated structured judgment."""

        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(self._settings.gemini_max_retries + 1):
            try:
                logger.info(
                    "Calling Gemini judge attempt={} model={}",
                    attempt + 1,
                    self._settings.gemini_model_name,
                )
                raw_text = await asyncio.wait_for(
                    asyncio.to_thread(self._call_gemini_sync, request),
                    timeout=self._settings.gemini_timeout_seconds,
                )
                evaluation = self._parse_and_validate(raw_text)
                logger.info(
                    "Gemini judge succeeded winner={} confidence={} elapsed_ms={:.2f}",
                    evaluation.winner.value,
                    evaluation.confidence,
                    (time.perf_counter() - started) * 1000,
                )
                return evaluation
            except asyncio.TimeoutError as exc:
                logger.warning("Gemini judge timed out after {}s", self._settings.gemini_timeout_seconds)
                raise GeminiTimeoutError("Gemini judge request timed out.") from exc
            except GeminiConfigurationError:
                raise
            except GeminiInvalidResponseError as exc:
                last_error = exc
                logger.warning("Gemini judge returned invalid JSON: {}", exc)
            except Exception as exc:  # SDK exceptions vary by transport/version.
                last_error = exc
                logger.warning("Gemini judge attempt failed: {}", exc)

            if attempt < self._settings.gemini_max_retries:
                await asyncio.sleep(self._retry_delay_seconds(attempt))

        if isinstance(last_error, GeminiInvalidResponseError):
            raise last_error
        detail = f" Last error: {last_error}" if last_error else ""
        raise GeminiAPIError(f"Gemini judge failed after retries.{detail}") from last_error

    async def evaluate_single(self, request: BenchmarkJudgeRequest) -> SingleJudgeEvaluation:
        """Evaluate one model response against one benchmark item."""

        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(self._settings.gemini_max_retries + 1):
            try:
                logger.info("Calling Gemini single-response judge attempt={}", attempt + 1)
                raw_text = await asyncio.wait_for(
                    asyncio.to_thread(self._call_single_sync, request),
                    timeout=self._settings.gemini_timeout_seconds,
                )
                evaluation = self._parse_and_validate_single(raw_text)
                logger.info(
                    "Gemini single judge succeeded overall_score={} elapsed_ms={:.2f}",
                    evaluation.overall_score,
                    (time.perf_counter() - started) * 1000,
                )
                return evaluation
            except asyncio.TimeoutError as exc:
                raise GeminiTimeoutError("Gemini judge request timed out.") from exc
            except GeminiConfigurationError:
                raise
            except GeminiInvalidResponseError as exc:
                last_error = exc
                logger.warning("Gemini single judge returned invalid JSON: {}", exc)
            except Exception as exc:
                last_error = exc
                logger.warning("Gemini single judge attempt failed: {}", exc)

            if attempt < self._settings.gemini_max_retries:
                await asyncio.sleep(self._retry_delay_seconds(attempt))

        if isinstance(last_error, GeminiInvalidResponseError):
            raise last_error
        raise GeminiAPIError(f"Gemini single judge failed after retries. Last error: {last_error}") from last_error

    def _call_gemini_sync(self, request: EvaluationRequest) -> str:
        model = self._get_model()
        prompt = self._build_user_prompt(request)

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
            request_options={"timeout": self._settings.gemini_timeout_seconds},
        )

        text = getattr(response, "text", None)
        if not text:
            raise GeminiInvalidResponseError("Gemini response did not contain text.")
        return text

    def _call_single_sync(self, request: BenchmarkJudgeRequest) -> str:
        model = self._get_model()
        prompt = (
            "Evaluate this single AI teacher response for the benchmark item. "
            "Return only the strict JSON object described in the system instructions.\n\n"
            f"{json.dumps(request.model_dump(), ensure_ascii=False, indent=2)}"
        )
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
            request_options={"timeout": self._settings.gemini_timeout_seconds},
        )
        text = getattr(response, "text", None)
        if not text:
            raise GeminiInvalidResponseError("Gemini response did not contain text.")
        return text

    async def generate_benchmark_report(self, evaluation: Evaluation) -> str:
        """Generate a detailed Markdown benchmark report from a stored evaluation."""

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._call_report_sync, evaluation),
                timeout=self._settings.gemini_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            logger.warning("Gemini benchmark report timed out after {}s", self._settings.gemini_timeout_seconds)
            raise GeminiTimeoutError("Gemini benchmark report request timed out.") from exc
        except GeminiConfigurationError:
            raise
        except Exception as exc:
            raise GeminiAPIError(f"Gemini benchmark report failed. Last error: {exc}") from exc

    def _call_report_sync(self, evaluation: Evaluation) -> str:
        model = self._get_report_model()
        prompt = self._build_report_prompt(evaluation)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0},
            request_options={"timeout": self._settings.gemini_timeout_seconds},
        )
        text = getattr(response, "text", None)
        if not text:
            raise GeminiInvalidResponseError("Gemini benchmark report did not contain text.")
        return text.strip()

    def _build_report_prompt(self, evaluation: Evaluation) -> str:
        payload = {
            "evaluation_id": str(evaluation.id),
            "student_prompt": evaluation.prompt,
            "student_level": evaluation.student_level,
            "language": evaluation.language,
            "model_a": evaluation.model_a,
            "model_b": evaluation.model_b,
            "response_a": evaluation.response_a,
            "response_b": evaluation.response_b,
            "winner": evaluation.winner,
            "confidence": evaluation.confidence,
            "scores": evaluation.scores,
            "reasoning": evaluation.evaluation_json.get("reasoning"),
            "latency_metrics": evaluation.latency_metrics,
            "multilingual_metrics": evaluation.multilingual_metrics,
            "hallucination_metrics": evaluation.hallucination_metrics,
            "benchmark_metadata": evaluation.benchmark_metadata,
        }
        return (
            "Generate the on-demand benchmark report from this stored evaluation data. "
            "Return strict Markdown only.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
        )

    def _build_user_prompt(self, request: EvaluationRequest) -> str:
        """Render candidate responses as JSON to avoid prompt formatting ambiguity."""

        payload = {
            "student_prompt": request.student_prompt,
            "student_level": request.student_level,
            "language": request.language,
            "model_a": request.model_a,
            "model_b": request.model_b,
            "response_a": request.response_a,
            "response_b": request.response_b,
        }
        return (
            "Evaluate these two AI teacher responses pairwise. "
            "Return only the strict JSON object described in the system instructions.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def _parse_and_validate(self, raw_text: str) -> JudgeEvaluation:
        json_text = self._extract_json(raw_text)
        try:
            payload: Any = json.loads(json_text)
            return JudgeEvaluation.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise GeminiInvalidResponseError(str(exc)) from exc

    def _parse_and_validate_single(self, raw_text: str) -> SingleJudgeEvaluation:
        json_text = self._extract_json(raw_text)
        try:
            payload: Any = json.loads(json_text)
            if isinstance(payload, dict) and isinstance(payload.get("reasoning"), str):
                payload["reasoning"] = self._compact_text(payload["reasoning"], max_chars=800)
            return SingleJudgeEvaluation.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise GeminiInvalidResponseError(str(exc)) from exc

    @staticmethod
    def _compact_text(value: str, max_chars: int) -> str:
        compact = " ".join(value.split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _extract_json(raw_text: str) -> str:
        """Accept pure JSON, or recover JSON if the model wrapped it in fences."""

        text = raw_text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text

        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced:
            return fenced.group(1).strip()

        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            return text[first : last + 1]

        raise GeminiInvalidResponseError("No JSON object found in Gemini response.")

    @staticmethod
    def _retry_delay_seconds(attempt: int) -> float:
        base_delay = 0.75 * (2**attempt)
        jitter = random.uniform(0.0, 0.25)
        return base_delay + jitter
