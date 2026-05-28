from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.api_client import APIClientError, get_analytics


def render(backend_url: str) -> None:
    st.subheader("Hallucination Dashboard")
    try:
        payload = get_analytics(backend_url, "/analytics/hallucinations")
    except APIClientError as exc:
        st.warning(str(exc))
        return

    rows = payload.get("rows", [])
    if not rows:
        st.info("No hallucination analytics yet.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.plotly_chart(
        px.bar(
            df.sort_values("hallucination_rate", ascending=False),
            x="model_name",
            y="hallucination_rate",
            color="avg_hallucination_risk",
            color_continuous_scale="RdYlGn",
            labels={"hallucination_rate": "Hallucination-risk rate", "avg_hallucination_risk": "Safety score"},
        ),
        use_container_width=True,
    )

    risk_df = df.set_index("model_name")[
        ["fabricated_fact_risk", "misleading_content_risk", "hallucination_rate"]
    ]
    st.plotly_chart(
        px.imshow(
            risk_df,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Reds",
            labels={"color": "Risk"},
        ),
        use_container_width=True,
    )
