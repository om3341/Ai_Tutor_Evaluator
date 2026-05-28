from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ModelMultilingualAnalytics(BaseModel):
    model_name: str
    language: str
    samples: int
    avg_multilingual_quality: float
    language_consistency: float
    grammar_quality: float
    code_switch_naturalness: float
    educational_clarity: float
    transliteration_handling: float
    regional_language_quality: float


class MultilingualAnalyticsResponse(BaseModel):
    generated_at: datetime
    languages: list[str]
    models: list[str]
    rows: list[ModelMultilingualAnalytics]


class HallucinationAnalyticsRow(BaseModel):
    model_name: str
    samples: int
    hallucination_rate: float
    avg_hallucination_risk: float
    unsafe_responses: int
    overconfident_risky_responses: int
    fabricated_fact_risk: float
    misleading_content_risk: float


class HallucinationAnalyticsResponse(BaseModel):
    generated_at: datetime
    unsafe_threshold: int
    hallucination_threshold: int
    rows: list[HallucinationAnalyticsRow]


class BenchmarkInsight(BaseModel):
    severity: str
    category: str
    message: str


class InsightsResponse(BaseModel):
    generated_at: datetime
    insights: list[BenchmarkInsight]


class AnalyticsOverviewResponse(BaseModel):
    generated_at: datetime
    total_evaluations: int
    total_model_samples: int
    languages: list[str]
    models: list[str]
