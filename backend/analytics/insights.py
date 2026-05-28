from __future__ import annotations

from datetime import UTC, datetime

from backend.schemas.analytics import BenchmarkInsight, HallucinationAnalyticsResponse, InsightsResponse, MultilingualAnalyticsResponse
from backend.schemas.latency import LatencyAnalyticsResponse


def generate_insights(
    multilingual: MultilingualAnalyticsResponse,
    latency: LatencyAnalyticsResponse,
    hallucinations: HallucinationAnalyticsResponse,
) -> InsightsResponse:
    insights: list[BenchmarkInsight] = []

    if multilingual.rows:
        strongest = max(multilingual.rows, key=lambda row: row.educational_clarity + row.avg_multilingual_quality)
        weakest = min(multilingual.rows, key=lambda row: row.educational_clarity + row.avg_multilingual_quality)
        insights.append(
            BenchmarkInsight(
                severity="positive",
                category="multilingual",
                message=(
                    f"{strongest.model_name} is strongest for {strongest.language}, with "
                    f"{strongest.avg_multilingual_quality:.1f}/10 multilingual quality and "
                    f"{strongest.educational_clarity:.1f}/10 educational clarity."
                ),
            )
        )
        insights.append(
            BenchmarkInsight(
                severity="watch",
                category="multilingual",
                message=(
                    f"{weakest.model_name} needs attention for {weakest.language}; "
                    f"educational clarity is {weakest.educational_clarity:.1f}/10."
                ),
            )
        )

    if latency.rows:
        fastest = min(latency.rows, key=lambda row: row.avg_model_latency_ms or 999999.0)
        slowest = max(latency.rows, key=lambda row: row.avg_model_latency_ms)
        insights.append(
            BenchmarkInsight(
                severity="neutral",
                category="latency",
                message=(
                    f"{fastest.model_name} is currently fastest at "
                    f"{fastest.avg_model_latency_ms:.0f} ms average model latency."
                ),
            )
        )
        if slowest.avg_model_latency_ms > fastest.avg_model_latency_ms * 1.5 and slowest.avg_model_latency_ms > 0:
            insights.append(
                BenchmarkInsight(
                    severity="watch",
                    category="latency",
                    message=(
                        f"{slowest.model_name} is the latency bottleneck at "
                        f"{slowest.avg_model_latency_ms:.0f} ms average latency."
                    ),
                )
            )

    if hallucinations.rows:
        safest = max(hallucinations.rows, key=lambda row: row.avg_hallucination_risk)
        riskiest = max(hallucinations.rows, key=lambda row: row.hallucination_rate)
        insights.append(
            BenchmarkInsight(
                severity="positive",
                category="hallucination",
                message=(
                    f"{safest.model_name} has the best hallucination safety score "
                    f"({safest.avg_hallucination_risk:.1f}/10)."
                ),
            )
        )
        if riskiest.hallucination_rate > 0:
            insights.append(
                BenchmarkInsight(
                    severity="critical" if riskiest.hallucination_rate >= 0.25 else "watch",
                    category="hallucination",
                    message=(
                        f"{riskiest.model_name} has a {riskiest.hallucination_rate:.0%} hallucination-risk rate "
                        f"using the current judge threshold."
                    ),
                )
            )

    if not insights:
        insights.append(
            BenchmarkInsight(
                severity="neutral",
                category="data",
                message="Run a few persisted evaluations to generate benchmark insights.",
            )
        )

    return InsightsResponse(generated_at=datetime.now(UTC), insights=insights)
