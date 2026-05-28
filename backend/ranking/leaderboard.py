from __future__ import annotations

from backend.database.models import Evaluation, Leaderboard
from backend.schemas.leaderboard import EvaluationHistoryItem, LeaderboardEntry, RecentPerformancePoint


def win_rate(entry: Leaderboard) -> float:
    if not entry.matches_played:
        return 0.0
    return round((entry.wins / entry.matches_played) * 100.0, 2)


def leaderboard_entry_schema(entry: Leaderboard) -> LeaderboardEntry:
    return LeaderboardEntry(
        model_name=entry.model_name,
        elo_score=round(entry.elo_score, 2),
        wins=entry.wins,
        losses=entry.losses,
        draws=entry.draws,
        matches_played=entry.matches_played,
        win_rate=win_rate(entry),
        avg_correctness=round(entry.avg_correctness, 2),
        avg_teaching_quality=round(entry.avg_teaching_quality, 2),
        avg_adaptation=round(entry.avg_adaptation, 2),
        avg_emotional_intelligence=round(entry.avg_emotional_intelligence, 2),
        avg_multilingual_quality=round(entry.avg_multilingual_quality, 2),
        avg_hallucination_risk=round(entry.avg_hallucination_risk, 2),
        avg_conversation_quality=round(entry.avg_conversation_quality, 2),
        updated_at=entry.updated_at,
    )


def history_item_schema(evaluation: Evaluation) -> EvaluationHistoryItem:
    winner_model = evaluation.model_a if evaluation.winner == "A" else evaluation.model_b
    return EvaluationHistoryItem(
        id=evaluation.id,
        prompt=evaluation.prompt,
        student_level=evaluation.student_level,
        language=evaluation.language,
        model_a=evaluation.model_a,
        model_b=evaluation.model_b,
        winner=evaluation.winner,
        winner_model=winner_model,
        confidence=evaluation.confidence,
        scores=evaluation.scores,
        latency_metrics=evaluation.latency_metrics,
        elo_before=evaluation.elo_before,
        elo_after=evaluation.elo_after,
        created_at=evaluation.created_at,
    )


def recent_performance_points(evaluations: list[Evaluation], model_name: str) -> list[RecentPerformancePoint]:
    points: list[RecentPerformancePoint] = []
    for evaluation in evaluations:
        if evaluation.model_a == model_name:
            side = "A"
            opponent = evaluation.model_b
        elif evaluation.model_b == model_name:
            side = "B"
            opponent = evaluation.model_a
        else:
            continue

        before = float(evaluation.elo_before.get(model_name, 0.0))
        after = float(evaluation.elo_after.get(model_name, before))
        if evaluation.winner == side:
            result = "win"
        elif evaluation.winner == "DRAW":
            result = "draw"
        else:
            result = "loss"

        points.append(
            RecentPerformancePoint(
                evaluation_id=evaluation.id,
                model_name=model_name,
                opponent=opponent,
                result=result,
                elo_before=before,
                elo_after=after,
                elo_delta=round(after - before, 2),
                confidence=evaluation.confidence,
                created_at=evaluation.created_at,
            )
        )
    return points
