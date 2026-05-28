from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from backend.analytics.hallucination import evaluation_hallucination_metrics
from backend.analytics.multilingual import evaluation_multilingual_metrics
from backend.config import Settings
from backend.database import crud
from backend.database.models import Evaluation, Leaderboard
from backend.ranking.elo import update_pairwise_elo
from backend.ranking.leaderboard import history_item_schema, leaderboard_entry_schema, recent_performance_points
from backend.schemas import EvaluationRequest, EvaluationResponse
from backend.schemas.leaderboard import EvaluationHistoryItem, LeaderboardEntry, ModelStats, RecentPerformancePoint


SCORE_FIELDS = (
    "correctness",
    "teaching_quality",
    "adaptation",
    "emotional_intelligence",
    "multilingual_quality",
    "hallucination_risk",
    "conversation_quality",
)


class RankingService:
    """Coordinates persistent evaluation history and leaderboard updates."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def record_evaluation(
        self,
        session: AsyncSession,
        request: EvaluationRequest,
        response: EvaluationResponse,
    ) -> Evaluation:
        async with session.begin():
            entry_a = await crud.get_or_create_leaderboard_entry(
                session,
                request.model_a,
                self._settings.elo_starting_score,
            )
            entry_b = await crud.get_or_create_leaderboard_entry(
                session,
                request.model_b,
                self._settings.elo_starting_score,
            )

            elo_before = {
                request.model_a: round(entry_a.elo_score, 2),
                request.model_b: round(entry_b.elo_score, 2),
            }

            result_a = self._result_for_a(response.winner.value)
            elo_result = update_pairwise_elo(
                rating_a=entry_a.elo_score,
                rating_b=entry_b.elo_score,
                result_a=result_a,
                k_factor=self._settings.elo_k_factor,
                confidence=response.confidence,
            )

            scores = response.scores.model_dump(mode="json")
            self._apply_match_update(entry_a, scores["A"], elo_result.rating_a, result_a)
            self._apply_match_update(entry_b, scores["B"], elo_result.rating_b, 1.0 - result_a)

            elo_after = {
                request.model_a: elo_result.rating_a,
                request.model_b: elo_result.rating_b,
            }
            evaluation_payload = response.model_dump(mode="json")
            latency_metrics = response.latency.model_dump(mode="json")
            latency_metrics.update(
                {
                    "model_a_latency_ms": request.latency_a_ms,
                    "model_b_latency_ms": request.latency_b_ms,
                }
            )
            benchmark_metadata = {
                "model_a_ttft_ms": request.ttft_a_ms,
                "model_b_ttft_ms": request.ttft_b_ms,
                "model_a_tokens_per_second": request.tokens_per_second_a,
                "model_b_tokens_per_second": request.tokens_per_second_b,
            }

            evaluation = Evaluation(
                prompt=request.student_prompt,
                student_level=request.student_level,
                language=request.language,
                model_a=request.model_a,
                model_b=request.model_b,
                response_a=request.response_a,
                response_b=request.response_b,
                winner=response.winner.value,
                confidence=response.confidence,
                scores=scores,
                evaluation_json=evaluation_payload,
                latency_metrics=latency_metrics,
                multilingual_metrics={},
                hallucination_metrics={},
                benchmark_metadata=benchmark_metadata,
                elo_before=elo_before,
                elo_after=elo_after,
            )
            evaluation.multilingual_metrics = evaluation_multilingual_metrics(evaluation)
            evaluation.hallucination_metrics = evaluation_hallucination_metrics(evaluation)
            persisted = await crud.create_evaluation(session, evaluation)

        await session.refresh(persisted)
        return persisted

    async def get_leaderboard(self, session: AsyncSession, limit: int | None = None) -> list[LeaderboardEntry]:
        entries = await crud.list_leaderboard(session, limit=limit)
        return [leaderboard_entry_schema(entry) for entry in entries]

    async def get_model_stats(
        self,
        session: AsyncSession,
        model_name: str,
        recent_limit: int = 10,
    ) -> ModelStats | None:
        entry = await crud.get_model_stats(session, model_name)
        if entry is None:
            return None

        recent = await crud.list_evaluation_history(session, limit=recent_limit, model_name=model_name)
        base = leaderboard_entry_schema(entry).model_dump()
        return ModelStats(
            **base,
            recent_evaluations=[history_item_schema(item) for item in recent],
        )

    async def get_history(
        self,
        session: AsyncSession,
        *,
        limit: int = 50,
        model_name: str | None = None,
    ) -> list[EvaluationHistoryItem]:
        evaluations = await crud.list_evaluation_history(session, limit=limit, model_name=model_name)
        return [history_item_schema(item) for item in evaluations]

    async def get_recent_performance(
        self,
        session: AsyncSession,
        model_name: str,
        limit: int = 25,
    ) -> list[RecentPerformancePoint]:
        evaluations = await crud.list_evaluation_history(session, limit=limit, model_name=model_name)
        return recent_performance_points(list(evaluations), model_name)

    async def clear_testing_data(self, session: AsyncSession) -> dict[str, int]:
        async with session.begin():
            return await crud.clear_rankings(session)

    @staticmethod
    def _result_for_a(winner: str) -> float:
        if winner == "A":
            return 1.0
        if winner == "B":
            return 0.0
        return 0.5

    @staticmethod
    def _apply_match_update(
        entry: Leaderboard,
        scores: dict[str, int],
        new_elo: float,
        result: float,
    ) -> None:
        previous_matches = entry.matches_played
        entry.matches_played += 1
        entry.elo_score = new_elo

        if result == 1.0:
            entry.wins += 1
        elif result == 0.0:
            entry.losses += 1
        else:
            entry.draws += 1

        for field in SCORE_FIELDS:
            current_average = getattr(entry, f"avg_{field}")
            next_average = ((current_average * previous_matches) + float(scores[field])) / entry.matches_played
            setattr(entry, f"avg_{field}", round(next_average, 4))
