from __future__ import annotations

from typing import Any

import pandas as pd

from frontend.utils import MODEL_OPTIONS


def leaderboard_from_history(history: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a lightweight leaderboard from current-session evaluations."""

    rows = []
    for model in MODEL_OPTIONS:
        wins = sum(1 for item in history if item.get("Winner") == model)
        appearances = sum(
            1
            for item in history
            if item.get("Model A") == model or item.get("Model B") == model
        )
        win_rate = round((wins / appearances) * 100, 1) if appearances else 0.0
        rows.append(
            {
                "Model": model,
                "Session Matches": appearances,
                "Wins": wins,
                "Win Rate %": win_rate,
            }
        )
    return pd.DataFrame(rows).sort_values(["Wins", "Win Rate %"], ascending=False)
