from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


def render_learning_gain_dashboard(runs: list[dict[str, Any]]) -> None:
    _render_metric_dashboard(runs, "learning_gain", "Learning Gain Dashboard")


def render_scaffolding_dashboard(runs: list[dict[str, Any]]) -> None:
    _render_metric_dashboard(runs, "scaffolding_effectiveness", "Scaffolding Dashboard")


def render_engagement_dashboard(runs: list[dict[str, Any]]) -> None:
    _render_metric_dashboard(runs, "student_engagement", "Engagement Dashboard")


def render_misconception_recovery_dashboard(runs: list[dict[str, Any]]) -> None:
    _render_metric_dashboard(runs, "misconception_recovery", "Misconception Recovery Dashboard")


def render_rag_analytics_dashboard(runs: list[dict[str, Any]]) -> None:
    st.subheader("RAG Analytics Dashboard")
    metrics = (
        ("knowledge_grounding", "Grounding"),
        ("context_faithfulness", "Faithfulness"),
        ("retrieval_utilization", "Retrieval Utilization"),
        ("educational_grounding", "Educational Grounding"),
        ("hallucination_risk", "Hallucination Resistance"),
    )
    rows = []
    for run in runs:
        scores = run.get("metrics") or {}
        for key, label in metrics:
            rows.append(
                {
                    "Run": run.get("run_id"),
                    "Tutor": run.get("tutor_model"),
                    "Metric": label,
                    "Score": scores.get(key),
                }
            )
    df = pd.DataFrame(rows).dropna(subset=["Score"]) if rows else pd.DataFrame()
    if df.empty:
        st.info("No RAG-grounded simulation scores available.")
        return
    st.plotly_chart(
        px.bar(df, x="Run", y="Score", color="Metric", barmode="group", range_y=[0, 10]),
        use_container_width=True,
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_conversation_replay(run: dict[str, Any]) -> None:
    retrieval = run.get("retrieval") or {}
    st.subheader("Retrieved Context")
    for index, chunk in enumerate(retrieval.get("chunks") or [], start=1):
        with st.expander(f"Chunk {index} · score {chunk.get('score', 0):.4f}"):
            st.write(chunk.get("text", ""))
    st.subheader("Conversation Replay")
    for message in run.get("transcript", []):
        with st.chat_message("user" if message.get("role") == "student" else "assistant"):
            st.markdown(f"**{message.get('role', '').title()} · Turn {message.get('turn_index')}**")
            st.write(message.get("content", ""))


def render_benchmark_history(runs: list[dict[str, Any]]) -> None:
    st.subheader("Conversation Benchmark History")
    if not runs:
        st.info("No conversation benchmark history yet.")
        return
    st.dataframe(pd.DataFrame(runs), use_container_width=True, hide_index=True)


def _render_metric_dashboard(runs: list[dict[str, Any]], metric: str, title: str) -> None:
    st.subheader(title)
    rows = []
    for run in runs:
        metrics = run.get("metrics") or {}
        rows.append(
            {
                "Run": run.get("run_id"),
                "Tutor": run.get("tutor_model"),
                "Student": run.get("student_model"),
                "Topic": run.get("topic"),
                "Persona": run.get("persona"),
                "Score": metrics.get(metric),
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["Score"]) if rows else pd.DataFrame()
    if df.empty:
        st.info("No scored simulation runs available.")
        return
    st.plotly_chart(px.bar(df, x="Run", y="Score", color="Tutor", range_y=[0, 10]), use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
