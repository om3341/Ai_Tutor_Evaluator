from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from backend.database.models import Evaluation
from backend.schemas.analytics import ModelMultilingualAnalytics, MultilingualAnalyticsResponse


REGIONAL_LANGUAGES = {"hindi", "hinglish", "marathi", "tamil", "telugu", "bengali"}


def evaluation_multilingual_metrics(evaluation: Evaluation) -> dict[str, dict[str, float]]:
    return {
        "A": side_multilingual_metrics(evaluation.language, evaluation.scores["A"]),
        "B": side_multilingual_metrics(evaluation.language, evaluation.scores["B"]),
    }


def side_multilingual_metrics(language: str, scores: dict[str, Any]) -> dict[str, float]:
    language_key = language.strip().lower()
    multilingual = float(scores["multilingual_quality"])
    teaching = float(scores["teaching_quality"])
    conversation = float(scores["conversation_quality"])
    adaptation = float(scores["adaptation"])

    return {
        "language_consistency": round(multilingual, 2),
        "grammar_quality": round((multilingual + conversation) / 2, 2),
        "code_switch_naturalness": round(multilingual if language_key == "hinglish" else (multilingual + adaptation) / 2, 2),
        "educational_clarity": round((teaching + adaptation) / 2, 2),
        "transliteration_handling": round(multilingual if language_key in {"hinglish", "hindi"} else (multilingual + teaching) / 2, 2),
        "regional_language_quality": round(multilingual if language_key in REGIONAL_LANGUAGES else 0.0, 2),
    }


def multilingual_analytics(evaluations: list[Evaluation]) -> MultilingualAnalyticsResponse:
    buckets: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[tuple[str, str], int] = defaultdict(int)

    for evaluation in evaluations:
        for side, model_name in (("A", evaluation.model_a), ("B", evaluation.model_b)):
            key = (model_name, evaluation.language)
            scores = evaluation.scores[side]
            derived = (evaluation.multilingual_metrics or {}).get(side) or side_multilingual_metrics(
                evaluation.language,
                scores,
            )
            counts[key] += 1
            buckets[key]["avg_multilingual_quality"] += float(scores["multilingual_quality"])
            for metric, value in derived.items():
                buckets[key][metric] += float(value)

    rows = []
    for key, count in counts.items():
        model_name, language = key
        values = buckets[key]
        rows.append(
            ModelMultilingualAnalytics(
                model_name=model_name,
                language=language,
                samples=count,
                avg_multilingual_quality=round(values["avg_multilingual_quality"] / count, 2),
                language_consistency=round(values["language_consistency"] / count, 2),
                grammar_quality=round(values["grammar_quality"] / count, 2),
                code_switch_naturalness=round(values["code_switch_naturalness"] / count, 2),
                educational_clarity=round(values["educational_clarity"] / count, 2),
                transliteration_handling=round(values["transliteration_handling"] / count, 2),
                regional_language_quality=round(values["regional_language_quality"] / count, 2),
            )
        )

    return MultilingualAnalyticsResponse(
        generated_at=datetime.now(UTC),
        languages=sorted({evaluation.language for evaluation in evaluations}),
        models=sorted({model for evaluation in evaluations for model in (evaluation.model_a, evaluation.model_b)}),
        rows=sorted(rows, key=lambda row: (row.language, row.model_name)),
    )
