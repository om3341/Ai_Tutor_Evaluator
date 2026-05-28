from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import Evaluation, Leaderboard


async def get_or_create_leaderboard_entry(
    session: AsyncSession,
    model_name: str,
    starting_elo: float,
) -> Leaderboard:
    result = await session.execute(
        select(Leaderboard)
        .where(Leaderboard.model_name == model_name)
        .with_for_update()
    )
    entry = result.scalar_one_or_none()
    if entry is not None:
        return entry

    entry = Leaderboard(model_name=model_name, elo_score=starting_elo)
    session.add(entry)
    await session.flush()
    return entry


async def create_evaluation(session: AsyncSession, evaluation: Evaluation) -> Evaluation:
    session.add(evaluation)
    await session.flush()
    await session.refresh(evaluation)
    return evaluation


async def list_leaderboard(session: AsyncSession, limit: int | None = None) -> Sequence[Leaderboard]:
    statement = select(Leaderboard).order_by(
        Leaderboard.elo_score.desc(),
        Leaderboard.wins.desc(),
        Leaderboard.matches_played.desc(),
    )
    if limit is not None:
        statement = statement.limit(limit)
    result = await session.execute(statement)
    return result.scalars().all()


async def get_model_stats(session: AsyncSession, model_name: str) -> Leaderboard | None:
    return await session.get(Leaderboard, model_name)


async def list_evaluation_history(
    session: AsyncSession,
    *,
    limit: int = 50,
    model_name: str | None = None,
) -> Sequence[Evaluation]:
    statement = select(Evaluation).order_by(Evaluation.created_at.desc()).limit(limit)
    if model_name:
        statement = statement.where(or_(Evaluation.model_a == model_name, Evaluation.model_b == model_name))
    result = await session.execute(statement)
    return result.scalars().all()


async def list_analytics_evaluations(session: AsyncSession, *, limit: int = 1000) -> Sequence[Evaluation]:
    statement = select(Evaluation).order_by(Evaluation.created_at.desc()).limit(limit)
    result = await session.execute(statement)
    return result.scalars().all()


async def get_evaluation(session: AsyncSession, evaluation_id: UUID) -> Evaluation | None:
    return await session.get(Evaluation, evaluation_id)


async def save_benchmark_report(
    session: AsyncSession,
    evaluation: Evaluation,
    benchmark_report_markdown: str,
) -> Evaluation:
    payload = dict(evaluation.evaluation_json or {})
    payload["benchmark_report_markdown"] = benchmark_report_markdown
    evaluation.evaluation_json = payload
    await session.flush()
    await session.refresh(evaluation)
    return evaluation


async def clear_rankings(session: AsyncSession) -> dict[str, int]:
    """Delete testing data from evaluations and leaderboard tables."""

    evaluations_result = await session.execute(delete(Evaluation))
    leaderboard_result = await session.execute(delete(Leaderboard))
    await session.flush()
    return {
        "evaluations_deleted": evaluations_result.rowcount or 0,
        "leaderboard_rows_deleted": leaderboard_result.rowcount or 0,
    }
