from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from frontend.utils import CATEGORY_LABELS, RADAR_CATEGORY_ORDER, average_score, score_delta, winner_model_name


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #ffffff;
            color: #1f2937;
        }
        section[data-testid="stSidebar"] {
            background: #fff7ed;
            border-right: 1px solid #fed7aa;
        }
        h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stText, label, p {
            color: #1f2937;
        }
        .arena-card {
            border: 1px solid #fed7aa;
            background: #ffffff;
            border-radius: 8px;
            padding: 1.05rem;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
        }
        .winner-card {
            border-color: #f97316;
            box-shadow: 0 0 0 1px rgba(249, 115, 22, 0.22), 0 14px 32px rgba(249, 115, 22, 0.14);
        }
        .metric-pill {
            display: inline-flex;
            gap: 0.35rem;
            align-items: center;
            padding: 0.24rem 0.55rem;
            border-radius: 999px;
            background: #fff7ed;
            border: 1px solid #fdba74;
            font-size: 0.84rem;
            color: #9a3412;
            margin: 0.18rem 0.18rem 0.18rem 0;
        }
        .small-muted {
            color: #9a3412;
            font-size: 0.9rem;
        }
        .response-box {
            white-space: pre-wrap;
            line-height: 1.55;
            color: #1f2937;
            min-height: 13rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #fed7aa;
            padding: 0.8rem;
            border-radius: 8px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #1f2937;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            border-bottom: 1px solid #fed7aa;
        }
        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border: 1px solid #fed7aa;
            border-bottom: none;
            border-radius: 8px 8px 0 0;
            color: #9a3412;
            font-weight: 600;
            padding: 0.55rem 0.9rem;
        }
        .stTabs [aria-selected="true"] {
            background: #f97316;
            color: #ffffff;
            border-color: #f97316;
        }
        .stButton > button {
            border-radius: 8px;
            font-weight: 700;
            min-height: 2.65rem;
            border: 1px solid #fdba74;
            background: #ffffff;
            color: #9a3412;
            box-shadow: 0 4px 12px rgba(249, 115, 22, 0.10);
        }
        .stButton > button:hover {
            border-color: #f97316;
            color: #7c2d12;
            background: #fff7ed;
        }
        .stButton > button[kind="primary"] {
            background: #f97316;
            color: #ffffff;
            border-color: #f97316;
            box-shadow: 0 10px 22px rgba(249, 115, 22, 0.25);
        }
        .stButton > button[kind="primary"]:hover {
            background: #ea580c;
            border-color: #ea580c;
            color: #ffffff;
        }
        .stButton > button:disabled,
        .stButton > button[disabled] {
            background: #f3f4f6;
            color: #9ca3af;
            border-color: #e5e7eb;
            box-shadow: none;
        }
        div[data-testid="stAlert"] {
            border-radius: 8px;
        }
        textarea, input, div[data-baseweb="select"] > div {
            border-color: #fed7aa;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div style="padding: 1.2rem 0 0.4rem 0;">
          <div class="small-muted">AI Teacher Benchmark Arena</div>
          <h1 style="margin: 0.2rem 0 0.35rem 0; letter-spacing: 0;">K-12 Tutor Response Evaluation</h1>
          <p style="max-width: 850px; color: #4b5563; font-size: 1.04rem;">
            Compare tutor responses with a rigorous educational rubric for conceptual quality,
            scaffolding, student adaptation, multilingual fidelity, safety, and learning impact.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_cards(evaluation: dict[str, Any], model_a: str, model_b: str) -> None:
    scores_a = evaluation["scores"]["A"]
    scores_b = evaluation["scores"]["B"]
    winner_name = winner_model_name(evaluation["winner"], model_a, model_b)
    avg_a = average_score(scores_a)
    avg_b = average_score(scores_b)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Winner", winner_name)
    col2.metric("Confidence", f"{evaluation['confidence'] * 100:.1f}%")
    col3.metric(f"{model_a} Avg", f"{avg_a:.2f}")
    col4.metric(f"{model_b} Avg", f"{avg_b:.2f}")


def comparison_cards(
    *,
    evaluation: dict[str, Any],
    model_a: str,
    model_b: str,
    response_a: str,
    response_b: str,
) -> None:
    winner = evaluation["winner"]
    col_a, col_b = st.columns(2)

    with col_a:
        _response_card("A", model_a, response_a, evaluation["scores"]["A"], winner == "A", evaluation["scores"]["B"])
    with col_b:
        _response_card("B", model_b, response_b, evaluation["scores"]["B"], winner == "B", evaluation["scores"]["A"])


def _response_card(
    side: str,
    model_name: str,
    response: str,
    scores: dict[str, int],
    is_winner: bool,
    opponent_scores: dict[str, int],
) -> None:
    card_class = "arena-card winner-card" if is_winner else "arena-card"
    badge = "Winner" if is_winner else "Candidate"
    safe_response = escape(response)
    strengths = [
        CATEGORY_LABELS[key]
        for key in RADAR_CATEGORY_ORDER
        if score_delta(scores, opponent_scores, key) > 0
    ][:3]

    st.markdown(
        f"""
        <div class="{card_class}">
          <div class="small-muted">Response {side} · {badge}</div>
          <h3 style="margin-top: 0.25rem;">{model_name}</h3>
          <div>
            <span class="metric-pill">Avg {average_score(scores):.2f}/10</span>
            <span class="metric-pill">Pedagogy {scores["teaching_quality"]}/10</span>
            <span class="metric-pill">Conceptual {scores["correctness"]}/10</span>
          </div>
          <hr style="border-color: #fed7aa; margin: 0.9rem 0;" />
          <div class="response-box">{safe_response}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if strengths:
        st.caption(f"Stronger in: {', '.join(strengths)}")
    else:
        st.caption("No clear category lead against the other response.")


def reasoning_panel(evaluation: dict[str, Any]) -> None:
    st.markdown("#### Educational Rubric Reasoning")
    st.info(evaluation["reasoning"])
