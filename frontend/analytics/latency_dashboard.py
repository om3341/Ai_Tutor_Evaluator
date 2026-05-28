from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.api_client import APIClientError, get_analytics


def render(backend_url: str) -> None:
    st.subheader("Latency Dashboard")
    try:
        latency = get_analytics(backend_url, "/analytics/latency")
        history = get_analytics(backend_url, "/analytics/latency/history")
    except APIClientError as exc:
        st.warning(str(exc))
        return

    rows = latency.get("rows", [])
    if not rows:
        st.info("No latency analytics yet.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.plotly_chart(
        px.bar(
            df,
            x="model_name",
            y=["avg_model_latency_ms", "p95_model_latency_ms", "avg_judge_latency_ms"],
            barmode="group",
            labels={"value": "Latency (ms)", "variable": "Metric"},
        ),
        use_container_width=True,
    )

    hist_df = pd.DataFrame(history.get("rows", []))
    if not hist_df.empty:
        hist_df["created_at"] = pd.to_datetime(hist_df["created_at"])
        st.plotly_chart(
            px.line(
                hist_df.sort_values("created_at"),
                x="created_at",
                y="model_latency_ms",
                color="model_name",
                markers=True,
                labels={"model_latency_ms": "Model latency (ms)", "created_at": "Evaluation time"},
            ),
            use_container_width=True,
        )
