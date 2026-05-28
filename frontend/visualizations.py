from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from frontend.utils import (
    BAR_CATEGORY_ORDER,
    CATEGORY_LABELS,
    RADAR_CATEGORY_ORDER,
    average_score,
)


PLOTLY_TEMPLATE = "plotly_white"
COLOR_A = "#f97316"
COLOR_B = "#64748b"
ACCENT = "#ea580c"


def radar_chart(evaluation: dict[str, Any], model_a: str, model_b: str) -> go.Figure:
    labels = [CATEGORY_LABELS[key] for key in RADAR_CATEGORY_ORDER]
    values_a = [evaluation["scores"]["A"][key] for key in RADAR_CATEGORY_ORDER]
    values_b = [evaluation["scores"]["B"][key] for key in RADAR_CATEGORY_ORDER]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_a + [values_a[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=model_a,
            line_color=COLOR_A,
            opacity=0.82,
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=values_b + [values_b[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=model_b,
            line_color=COLOR_B,
            opacity=0.78,
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=430,
        margin=dict(l=24, r=24, t=42, b=24),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=10)),
            bgcolor="#ffffff",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title="Educational Rubric Score Radar",
    )
    return fig


def category_bar_chart(evaluation: dict[str, Any], model_a: str, model_b: str) -> go.Figure:
    rows = []
    for key in BAR_CATEGORY_ORDER:
        rows.append({"Metric": CATEGORY_LABELS[key], "Model": model_a, "Score": evaluation["scores"]["A"][key]})
        rows.append({"Metric": CATEGORY_LABELS[key], "Model": model_b, "Score": evaluation["scores"]["B"][key]})
    df = pd.DataFrame(rows)

    fig = px.bar(
        df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        range_y=[0, 10],
        color_discrete_sequence=[COLOR_A, COLOR_B],
        template=PLOTLY_TEMPLATE,
        title="Core Rubric Comparison",
    )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=54, b=24), xaxis_title=None)
    return fig


def latency_chart(model_a: str, model_b: str, latency_a_ms: float, latency_b_ms: float, judge_latency_ms: float) -> go.Figure:
    df = pd.DataFrame(
        [
            {"Metric": f"{model_a} response", "Latency (ms)": latency_a_ms},
            {"Metric": f"{model_b} response", "Latency (ms)": latency_b_ms},
            {"Metric": "Rubric judge", "Latency (ms)": judge_latency_ms},
        ]
    )
    fig = px.bar(
        df,
        x="Metric",
        y="Latency (ms)",
        color="Metric",
        color_discrete_sequence=[COLOR_A, COLOR_B, ACCENT],
        template=PLOTLY_TEMPLATE,
        title="Latency Breakdown",
    )
    fig.update_layout(height=320, showlegend=False, margin=dict(l=20, r=20, t=54, b=24), xaxis_title=None)
    return fig


def confidence_gauge(confidence: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(confidence * 100, 1),
            number={"suffix": "%", "font": {"size": 34}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": ACCENT},
                "bgcolor": "#ffffff",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#fee2e2"},
                    {"range": [50, 75], "color": "#ffedd5"},
                    {"range": [75, 100], "color": "#dcfce7"},
                ],
            },
            title={"text": "Rubric Confidence"},
        )
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=260, margin=dict(l=16, r=16, t=44, b=8))
    return fig


def average_score_chart(evaluation: dict[str, Any], model_a: str, model_b: str) -> go.Figure:
    avg_a = average_score(evaluation["scores"]["A"])
    avg_b = average_score(evaluation["scores"]["B"])
    fig = go.Figure(
        go.Bar(
            x=[model_a, model_b],
            y=[avg_a, avg_b],
            marker_color=[COLOR_A, COLOR_B],
            text=[avg_a, avg_b],
            textposition="outside",
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Average Rubric Score",
        height=260,
        yaxis=dict(range=[0, 10]),
        margin=dict(l=20, r=20, t=54, b=24),
    )
    return fig
