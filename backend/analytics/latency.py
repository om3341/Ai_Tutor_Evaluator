from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

from backend.database.models import Evaluation
from backend.schemas.latency import LatencyAnalyticsResponse, LatencyHistoryPoint, ModelLatencyAnalytics


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile_value / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def latency_history(evaluations: list[Evaluation]) -> list[LatencyHistoryPoint]:
    rows: list[LatencyHistoryPoint] = []
    for evaluation in evaluations:
        latency = evaluation.latency_metrics or {}
        metadata = evaluation.benchmark_metadata or {}
        for side, model_name in (("A", evaluation.model_a), ("B", evaluation.model_b)):
            suffix = "a" if side == "A" else "b"
            rows.append(
                LatencyHistoryPoint(
                    evaluation_id=evaluation.id,
                    model_name=model_name,
                    language=evaluation.language,
                    model_latency_ms=_optional_float(latency.get(f"model_{suffix}_latency_ms")),
                    ttft_ms=_optional_float(metadata.get(f"model_{suffix}_ttft_ms")),
                    tokens_per_second=_optional_float(metadata.get(f"model_{suffix}_tokens_per_second")),
                    judge_latency_ms=float(latency.get("judge_latency_ms", 0.0)),
                    api_latency_ms=float(latency.get("total_latency_ms", 0.0)),
                    created_at=evaluation.created_at,
                )
            )
    return rows


def latency_analytics(evaluations: list[Evaluation]) -> LatencyAnalyticsResponse:
    history = latency_history(evaluations)
    grouped: dict[str, list[LatencyHistoryPoint]] = defaultdict(list)
    for row in history:
        grouped[row.model_name].append(row)

    analytics_rows: list[ModelLatencyAnalytics] = []
    for model_name, rows in grouped.items():
        model_latencies = [row.model_latency_ms for row in rows if row.model_latency_ms is not None]
        ttfts = [row.ttft_ms for row in rows if row.ttft_ms is not None]
        token_rates = [row.tokens_per_second for row in rows if row.tokens_per_second is not None]
        avg_model_latency = sum(model_latencies) / len(model_latencies) if model_latencies else 0.0
        analytics_rows.append(
            ModelLatencyAnalytics(
                model_name=model_name,
                samples=len(rows),
                avg_model_latency_ms=round(avg_model_latency, 2),
                p95_model_latency_ms=round(percentile(model_latencies, 95), 2),
                avg_ttft_ms=round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
                avg_tokens_per_second=round(sum(token_rates) / len(token_rates), 2) if token_rates else None,
                avg_judge_latency_ms=round(sum(row.judge_latency_ms for row in rows) / len(rows), 2),
                avg_api_latency_ms=round(sum(row.api_latency_ms for row in rows) / len(rows), 2),
                speed_rank=0,
            )
        )

    ranked = sorted(analytics_rows, key=lambda row: (row.avg_model_latency_ms or 999999.0, row.model_name))
    for index, row in enumerate(ranked, start=1):
        row.speed_rank = index

    return LatencyAnalyticsResponse(generated_at=datetime.now(UTC), rows=ranked)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
