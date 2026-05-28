from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EloResult:
    rating_a: float
    rating_b: float
    delta_a: float
    delta_b: float


def expected_score(rating: float, opponent_rating: float) -> float:
    """Return the standard ELO expected score against an opponent."""

    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def update_pairwise_elo(
    *,
    rating_a: float,
    rating_b: float,
    result_a: float,
    k_factor: float = 32.0,
    confidence: float = 1.0,
) -> EloResult:
    """Update two ELO ratings.

    `result_a` is 1.0 for A win, 0.0 for A loss, and 0.5 for draw.
    Confidence scales the update but is clamped so low-confidence judgments
    still move ratings gently instead of freezing the board.
    """

    if result_a not in {0.0, 0.5, 1.0}:
        raise ValueError("result_a must be 0.0, 0.5, or 1.0")

    confidence_weight = min(1.0, max(0.25, confidence))
    weighted_k = k_factor * confidence_weight
    expected_a = expected_score(rating_a, rating_b)
    expected_b = expected_score(rating_b, rating_a)
    result_b = 1.0 - result_a

    delta_a = weighted_k * (result_a - expected_a)
    delta_b = weighted_k * (result_b - expected_b)

    return EloResult(
        rating_a=round(rating_a + delta_a, 2),
        rating_b=round(rating_b + delta_b, 2),
        delta_a=round(delta_a, 2),
        delta_b=round(delta_b, 2),
    )
