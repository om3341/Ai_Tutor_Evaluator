from __future__ import annotations

from typing import Any

import pandas as pd

from frontend.utils import CATEGORY_LABELS, RADAR_CATEGORY_ORDER, average_score, make_summary_rows


def summary_dataframe(
    evaluation: dict[str, Any],
    model_a: str,
    model_b: str,
    latency_a_ms: float,
    latency_b_ms: float,
) -> pd.DataFrame:
    return pd.DataFrame(make_summary_rows(evaluation, model_a, model_b, latency_a_ms, latency_b_ms))


def category_delta_dataframe(evaluation: dict[str, Any], model_a: str, model_b: str) -> pd.DataFrame:
    rows = []
    for key in RADAR_CATEGORY_ORDER:
        score_a = evaluation["scores"]["A"][key]
        score_b = evaluation["scores"]["B"][key]
        rows.append(
            {
                "Category": CATEGORY_LABELS[key],
                model_a: score_a,
                model_b: score_b,
                "Delta A-B": score_a - score_b,
                "Leader": model_a if score_a > score_b else model_b if score_b > score_a else "Tie",
            }
        )
    return pd.DataFrame(rows)


def append_history(
    history: list[dict[str, Any]],
    evaluation: dict[str, Any],
    model_a: str,
    model_b: str,
    prompt: str,
) -> list[dict[str, Any]]:
    scores_a = evaluation["scores"]["A"]
    scores_b = evaluation["scores"]["B"]
    history.append(
        {
            "Prompt": prompt[:120],
            "Model A": model_a,
            "Model B": model_b,
            "Winner": model_a if evaluation["winner"] == "A" else model_b,
            "Confidence": evaluation["confidence"],
            f"{model_a} Avg": average_score(scores_a),
            f"{model_b} Avg": average_score(scores_b),
        }
    )
    return history[-25:]
