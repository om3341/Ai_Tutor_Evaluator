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
            background:
                radial-gradient(880px 420px at 88% -10%, #ffffff 0%, rgba(255,255,255,0) 58%),
                #f7fbff;
            color: #0f172a;
        }
        section[data-testid="stSidebar"] {
            background: #eef7ff;
            border-right: 1px solid #bfdbfe;
        }
        h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stText, label, p {
            color: #0f172a;
        }
        .arena-card {
            border: 1px solid #bfdbfe;
            background: #ffffff;
            border-radius: 8px;
            padding: 1.05rem;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
        }
        .winner-card {
            border-color: #0284c7;
            box-shadow: 0 0 0 1px rgba(2, 132, 199, 0.22), 0 14px 32px rgba(2, 132, 199, 0.14);
        }
        .metric-pill {
            display: inline-flex;
            gap: 0.35rem;
            align-items: center;
            padding: 0.24rem 0.55rem;
            border-radius: 999px;
            background: #e0f2fe;
            border: 1px solid #7dd3fc;
            font-size: 0.84rem;
            color: #075985;
            margin: 0.18rem 0.18rem 0.18rem 0;
        }
        .small-muted {
            color: #0369a1;
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
            border: 1px solid #bfdbfe;
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
            border-bottom: 1px solid #bfdbfe;
        }
        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border: 1px solid #bfdbfe;
            border-bottom: none;
            border-radius: 8px 8px 0 0;
            color: #075985;
            font-weight: 600;
            padding: 0.55rem 0.9rem;
        }
        .stTabs [aria-selected="true"] {
            background: #0284c7;
            color: #ffffff;
            border-color: #0284c7;
        }
        .stButton > button {
            border-radius: 8px;
            font-weight: 700;
            min-height: 2.65rem;
            border: 1px solid #7dd3fc;
            background: #ffffff;
            color: #075985;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.10);
        }
        .stButton > button:hover {
            border-color: #0284c7;
            color: #0c4a6e;
            background: #e0f2fe;
        }
        .stButton > button[kind="primary"] {
            background: #0284c7;
            color: #ffffff;
            border-color: #0284c7;
            box-shadow: 0 10px 22px rgba(2, 132, 199, 0.25);
        }
        .stButton > button[kind="primary"]:hover {
            background: #0369a1;
            border-color: #0369a1;
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
            border-color: #bfdbfe;
        }
        textarea:focus, input:focus {
            border-color: #0284c7 !important;
            box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.12) !important;
        }
        .simulation-heading {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 0.35rem 0 0.5rem 0;
        }
        .simulation-heading h2 {
            margin: 0;
            font-size: 1.55rem;
        }
        .simulation-heading p {
            margin: 0.35rem 0 0 0;
            color: #6b7280;
            font-size: 0.96rem;
        }
        .role-label {
            color: #0369a1;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            margin: 0 0 0.38rem 0;
            text-transform: uppercase;
        }
        .model-status {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            min-height: 3.15rem;
            margin: 0.55rem 0 0.65rem 0;
            padding: 0.65rem 0.75rem;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: #ffffff;
        }
        .model-status span:last-child {
            display: flex;
            flex-direction: column;
            gap: 0.08rem;
        }
        .model-status strong {
            color: #374151;
            font-size: 0.88rem;
        }
        .model-status small {
            color: #6b7280;
            font-size: 0.78rem;
        }
        .status-dot {
            display: inline-block;
            width: 0.62rem;
            height: 0.62rem;
            flex: 0 0 0.62rem;
            border-radius: 999px;
        }
        .status-dot.online {
            background: #0284c7;
            box-shadow: 0 0 0 3px #bae6fd;
        }
        .status-dot.offline {
            background: #9ca3af;
            box-shadow: 0 0 0 3px #f3f4f6;
        }
        .persona-summary {
            min-height: 4.7rem;
            margin-top: 0.55rem;
            padding: 0.7rem 0.8rem;
            border-left: 3px solid #0284c7;
            border-radius: 0 6px 6px 0;
            background: #e0f2fe;
        }
        .persona-summary strong {
            color: #075985;
            font-size: 0.82rem;
        }
        .persona-summary p {
            margin: 0.22rem 0 0 0;
            color: #4b5563;
            font-size: 0.86rem;
            line-height: 1.45;
        }
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.22rem;
            margin-top: 1.2rem;
            padding: 1.5rem;
            border: 1px dashed #7dd3fc;
            border-radius: 8px;
            background: #f8fcff;
            color: #4b5563;
            text-align: center;
        }
        .empty-state strong {
            color: #075985;
        }
        .empty-state span {
            font-size: 0.9rem;
        }
        .live-context {
            margin: 0.35rem 0 0.85rem 0;
            padding: 0.65rem 0.8rem;
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            background: #e0f2fe;
            color: #075985;
            font-size: 0.9rem;
        }
        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: 0.4rem 0 1.1rem 0;
            padding: 0.78rem 1rem;
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 10px 30px rgba(2, 132, 199, 0.08);
        }
        .brand-mark {
            font-size: 0.78rem;
            font-weight: 800;
            color: #0369a1;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .classroom-status {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin: 0.25rem 0 1rem 0;
            padding: 0.78rem 0.9rem;
            border: 1px solid #bfdbfe;
            border-radius: 11px;
            background: #ffffff;
        }
        .classroom-status .pulse-dot {
            width: 0.62rem;
            height: 0.62rem;
            border-radius: 999px;
            background: #94a3b8;
            flex: 0 0 0.62rem;
        }
        .classroom-status.live .pulse-dot {
            background: #0284c7;
            box-shadow: 0 0 0 4px rgba(2, 132, 199, 0.14);
        }
        .phase-pill {
            margin-left: auto;
            padding: 0.18rem 0.55rem;
            border: 1px solid #bfdbfe;
            border-radius: 7px;
            background: #f0f9ff;
            color: #0369a1;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .stage-card, .activity-card {
            border: 1px solid #bfdbfe;
            border-radius: 16px;
            background: #ffffff;
            padding: 1rem;
            box-shadow: 0 18px 45px -32px rgba(15, 23, 42, 0.45);
        }
        .stage-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.8rem;
        }
        .stage-title h3 {
            margin: 0;
            color: #0f172a;
            font-size: 1.05rem;
        }
        .lesson-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.24rem 0.55rem;
            border: 1px solid #bfdbfe;
            border-radius: 999px;
            background: #f0f9ff;
            color: #075985;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .stream {
            display: flex;
            flex-direction: column;
            gap: 0.7rem;
        }
        .msg {
            padding: 0.76rem 0.85rem;
            border-radius: 12px;
            border: 1px solid #dbeafe;
            line-height: 1.55;
            font-size: 0.92rem;
            white-space: pre-wrap;
        }
        .msg .who {
            color: #64748b;
            font-size: 0.64rem;
            font-weight: 900;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.22rem;
        }
        .msg.tut {
            background: #eff6ff;
            border-color: #bfdbfe;
        }
        .msg.stu {
            background: #ffffff;
            border-color: #bae6fd;
        }
        .activity-card h4 {
            margin: 0 0 0.75rem 0;
            color: #075985;
            font-size: 0.84rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .activity-line {
            display: flex;
            justify-content: space-between;
            gap: 0.7rem;
            padding: 0.52rem 0;
            border-bottom: 1px solid #e0f2fe;
            color: #475569;
            font-size: 0.88rem;
        }
        .activity-line:last-child {
            border-bottom: none;
        }
        .activity-line strong {
            color: #0f172a;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="topbar">
          <div>
            <div class="brand-mark">AI Teacher Benchmark Platform</div>
            <h1 style="margin: 0.12rem 0 0 0; letter-spacing: 0; font-size: 1.75rem;">TeachBench Classroom</h1>
          </div>
          <div class="lesson-chip">RAG-grounded simulation</div>
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
          <hr style="border-color: #bfdbfe; margin: 0.9rem 0;" />
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
