from __future__ import annotations

from typing import Any

import requests


class APIClientError(RuntimeError):
    """Raised when the Streamlit frontend cannot complete an API request."""


def evaluate_pairwise(
    *,
    base_url: str,
    student_prompt: str,
    student_level: str,
    language: str,
    model_a: str,
    model_b: str,
    response_a: str,
    response_b: str,
    latency_a_ms: float | None = None,
    latency_b_ms: float | None = None,
    timeout_seconds: float = 90.0,
) -> dict[str, Any]:
    payload = {
        "student_prompt": student_prompt,
        "student_level": student_level,
        "language": language,
        "model_a": model_a,
        "model_b": model_b,
        "response_a": response_a,
        "response_b": response_b,
        "latency_a_ms": latency_a_ms,
        "latency_b_ms": latency_b_ms,
    }

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/evaluate",
            json=payload,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise APIClientError(f"Could not reach FastAPI backend: {exc}") from exc

    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise APIClientError(f"Backend returned {response.status_code}: {detail}")

    try:
        return response.json()
    except ValueError as exc:
        raise APIClientError("Backend returned a non-JSON response.") from exc


def check_health(base_url: str, timeout_seconds: float = 5.0) -> bool:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout_seconds)
        return response.ok
    except requests.RequestException:
        return False


def get_leaderboard(base_url: str, limit: int = 25, timeout_seconds: float = 10.0) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/leaderboard",
            params={"limit": limit},
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise APIClientError(f"Could not load leaderboard: {exc}") from exc

    if response.status_code >= 400:
        raise APIClientError(f"Leaderboard returned {response.status_code}: {_response_detail(response)}")
    return response.json()


def get_evaluation_history(base_url: str, limit: int = 25, timeout_seconds: float = 10.0) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/evaluations/history",
            params={"limit": limit},
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise APIClientError(f"Could not load evaluation history: {exc}") from exc

    if response.status_code >= 400:
        raise APIClientError(f"Evaluation history returned {response.status_code}: {_response_detail(response)}")
    return response.json()


def clear_leaderboard(base_url: str, timeout_seconds: float = 10.0) -> dict[str, Any]:
    try:
        response = requests.delete(
            f"{base_url.rstrip('/')}/leaderboard/reset",
            params={"confirm": "true"},
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise APIClientError(f"Could not clear leaderboard: {exc}") from exc

    if response.status_code >= 400:
        raise APIClientError(f"Clear leaderboard returned {response.status_code}: {_response_detail(response)}")
    return response.json()


def generate_qwen_response(
    *,
    base_url: str,
    student_prompt: str,
    student_level: str,
    language: str,
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 500,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    payload = {
        "student_prompt": student_prompt,
        "student_level": student_level,
        "language": language,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/models/qwen/generate",
            json=payload,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise APIClientError(f"Could not generate Qwen response: {exc}") from exc

    if response.status_code >= 400:
        raise APIClientError(f"Qwen generation returned {response.status_code}: {_response_detail(response)}")
    return response.json()


def generate_gemma_response(
    *,
    base_url: str,
    student_prompt: str,
    student_level: str,
    language: str,
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 500,
    timeout_seconds: float = 150.0,
) -> dict[str, Any]:
    payload = {
        "student_prompt": student_prompt,
        "student_level": student_level,
        "language": language,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/models/gemma/generate",
            json=payload,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise APIClientError(f"Could not generate Gemma response: {exc}") from exc

    if response.status_code >= 400:
        raise APIClientError(f"Gemma generation returned {response.status_code}: {_response_detail(response)}")
    return response.json()


def generate_llama_response(
    *,
    base_url: str,
    student_prompt: str,
    student_level: str,
    language: str,
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 500,
    timeout_seconds: float = 150.0,
) -> dict[str, Any]:
    payload = {
        "student_prompt": student_prompt,
        "student_level": student_level,
        "language": language,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/models/llama/generate",
            json=payload,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise APIClientError(f"Could not generate Llama response: {exc}") from exc

    if response.status_code >= 400:
        raise APIClientError(f"Llama generation returned {response.status_code}: {_response_detail(response)}")
    return response.json()


def get_analytics(base_url: str, path: str, timeout_seconds: float = 10.0) -> dict[str, Any]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise APIClientError(f"Could not load analytics: {exc}") from exc

    if response.status_code >= 400:
        raise APIClientError(f"Analytics returned {response.status_code}: {_response_detail(response)}")
    return response.json()


def generate_benchmark_report(
    base_url: str,
    evaluation_id: str,
    *,
    force: bool = False,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/evaluations/{evaluation_id}/benchmark-report",
            params={"force": str(force).lower()},
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise APIClientError(f"Could not generate benchmark report: {exc}") from exc

    if response.status_code >= 400:
        raise APIClientError(f"Benchmark report returned {response.status_code}: {_response_detail(response)}")
    return response.json()


def load_benchmark_model(base_url: str, model_name: str, timeout_seconds: float = 180.0) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/benchmarks/model/load",
            json={"model_name": model_name},
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise APIClientError(f"Could not load benchmark model: {exc}") from exc
    if response.status_code >= 400:
        raise APIClientError(f"Load model returned {response.status_code}: {_response_detail(response)}")
    return response.json()


def unload_benchmark_model(base_url: str, timeout_seconds: float = 10.0) -> dict[str, Any]:
    try:
        response = requests.post(f"{base_url.rstrip('/')}/benchmarks/model/unload", timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise APIClientError(f"Could not unload benchmark model: {exc}") from exc
    if response.status_code >= 400:
        raise APIClientError(f"Unload model returned {response.status_code}: {_response_detail(response)}")
    return response.json()


def get_benchmark_model_state(base_url: str, timeout_seconds: float = 10.0) -> dict[str, Any]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/benchmarks/model", timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise APIClientError(f"Could not read benchmark model state: {exc}") from exc
    if response.status_code >= 400:
        raise APIClientError(f"Model state returned {response.status_code}: {_response_detail(response)}")
    return response.json()


def get_benchmark_dataset(base_url: str, max_items: int | None = None, timeout_seconds: float = 10.0) -> list[dict[str, Any]]:
    params: dict[str, Any] = {}
    if max_items:
        params["max_items"] = max_items
    try:
        response = requests.get(f"{base_url.rstrip('/')}/benchmarks/dataset", params=params, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise APIClientError(f"Could not load benchmark dataset: {exc}") from exc
    if response.status_code >= 400:
        raise APIClientError(f"Benchmark dataset returned {response.status_code}: {_response_detail(response)}")
    return response.json()


def run_model_benchmark(
    *,
    base_url: str,
    model_name: str,
    dataset_name: str = "k12_teacher_core_v1",
    max_items: int | None = None,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model_name": model_name, "dataset_name": dataset_name}
    if max_items:
        payload["max_items"] = max_items
    try:
        response = requests.post(f"{base_url.rstrip('/')}/benchmarks/run", json=payload, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise APIClientError(f"Could not run benchmark: {exc}") from exc
    if response.status_code >= 400:
        raise APIClientError(f"Benchmark run returned {response.status_code}: {_response_detail(response)}")
    return response.json()


def evaluate_collected_benchmark(
    *,
    base_url: str,
    model_name: str,
    items: list[dict[str, Any]],
    dataset_name: str = "k12_teacher_core_v1",
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    payload = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "items": items,
    }
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/benchmarks/evaluate-collected",
            json=payload,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise APIClientError(f"Could not evaluate collected benchmark: {exc}") from exc
    if response.status_code >= 400:
        raise APIClientError(f"Benchmark evaluation returned {response.status_code}: {_response_detail(response)}")
    return response.json()


def get_benchmark_runs(base_url: str, limit: int = 25, timeout_seconds: float = 10.0) -> list[dict[str, Any]]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/benchmarks/runs", params={"limit": limit}, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise APIClientError(f"Could not load benchmark runs: {exc}") from exc
    if response.status_code >= 400:
        raise APIClientError(f"Benchmark runs returned {response.status_code}: {_response_detail(response)}")
    return response.json()


def _response_detail(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text
