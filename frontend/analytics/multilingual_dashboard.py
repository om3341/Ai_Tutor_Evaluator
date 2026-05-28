from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import APIClientError, get_analytics


METRICS = [
    "language_consistency",
    "grammar_quality",
    "code_switch_naturalness",
    "educational_clarity",
    "transliteration_handling",
    "regional_language_quality",
]


def render(backend_url: str) -> None:
    st.subheader("Multilingual Analytics")
    try:
        payload = get_analytics(backend_url, "/analytics/multilingual")
    except APIClientError as exc:
        st.warning(str(exc))
        return

    rows = payload.get("rows", [])
    if not rows:
        st.info("No multilingual analytics yet.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    heatmap_df = df.pivot_table(
        index="model_name",
        columns="language",
        values="avg_multilingual_quality",
        aggfunc="mean",
    )
    st.plotly_chart(
        px.imshow(
            heatmap_df,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="Viridis",
            labels={"color": "Multilingual Fidelity"},
        ),
        use_container_width=True,
    )

    radar_df = df.groupby("model_name", as_index=False)[METRICS].mean()
    fig = go.Figure()
    for _, row in radar_df.iterrows():
        values = [row[metric] for metric in METRICS]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=METRICS + [METRICS[0]],
                fill="toself",
                name=row["model_name"],
            )
        )
    fig.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 10]}}, height=460)
    st.plotly_chart(fig, use_container_width=True)
