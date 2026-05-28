from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.api_client import APIClientError, get_analytics


def render(backend_url: str) -> None:
    st.subheader("Benchmark Insights")
    try:
        payload = get_analytics(backend_url, "/analytics/insights")
        overview = get_analytics(backend_url, "/analytics")
    except APIClientError as exc:
        st.warning(str(exc))
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Evaluations", overview.get("total_evaluations", 0))
    metric_cols[1].metric("Model Samples", overview.get("total_model_samples", 0))
    metric_cols[2].metric("Languages", len(overview.get("languages", [])))
    metric_cols[3].metric("Models", len(overview.get("models", [])))

    insights = payload.get("insights", [])
    if insights:
        for item in insights:
            severity = item.get("severity", "neutral")
            message = item.get("message", "")
            if severity == "critical":
                st.error(message)
            elif severity == "watch":
                st.warning(message)
            elif severity == "positive":
                st.success(message)
            else:
                st.info(message)
        st.dataframe(pd.DataFrame(insights), use_container_width=True, hide_index=True)
    else:
        st.info("No insights yet.")
