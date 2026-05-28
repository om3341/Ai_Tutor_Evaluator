from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.api_client import (
    APIClientError,
    check_health,
    evaluate_collected_benchmark,
    generate_gemma_response,
    generate_llama_response,
    generate_qwen_response,
    get_benchmark_dataset,
    get_benchmark_model_state,
    get_benchmark_runs,
    load_benchmark_model,
    unload_benchmark_model,
)
from frontend.components import hero, inject_theme
from frontend.utils import MODEL_OPTIONS


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def main() -> None:
    st.set_page_config(
        page_title="AI Teacher Benchmark Platform",
        page_icon="AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    st.session_state.setdefault("benchmark_result", None)
    st.session_state.setdefault("benchmark_dataset", [])
    st.session_state.setdefault("benchmark_index", 0)
    st.session_state.setdefault("collected_items", [])

    with st.sidebar:
        st.header("Benchmark Controls")
        backend_url = st.text_input("FastAPI backend URL", value=DEFAULT_BACKEND_URL)
        backend_ok = check_health(backend_url)
        st.caption("Backend status: online" if backend_ok else "Backend status: not reachable")

        st.divider()
        model_name = st.selectbox("Model", MODEL_OPTIONS, index=1 if len(MODEL_OPTIONS) > 1 else 0)
        max_items = st.number_input("Dataset items", min_value=1, max_value=100, value=5, step=1)

    hero()
    st.caption("Single-model educational benchmark. No A/B battles, no winner selection, one active model at a time.")

    run_tab, results_tab, history_tab = st.tabs(["Run Benchmark", "Results", "History"])

    with run_tab:
        render_model_console(backend_url, model_name, int(max_items))
        render_staged_benchmark(backend_url, model_name, int(max_items))

    with results_tab:
        render_benchmark_result(st.session_state.get("benchmark_result"))

    with history_tab:
        render_benchmark_history(backend_url)


def render_model_console(backend_url: str, model_name: str, max_items: int) -> None:
    st.subheader("Model Lifecycle")
    try:
        state = get_benchmark_model_state(backend_url)
    except APIClientError as exc:
        st.error(str(exc))
        state = {"loaded": False, "active_model": None}

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Model", state.get("active_model") or "None")
    col2.metric("Loaded", "Yes" if state.get("loaded") else "No")
    col3.metric("Dataset", "k12_teacher_core_v1")

    action_left, action_mid, action_right = st.columns(3)
    with action_left:
        if st.button("Load Model", type="primary", use_container_width=True):
            try:
                state = load_benchmark_model(backend_url, model_name)
                st.success(f"Loaded {state['active_model']}.")
            except APIClientError as exc:
                st.error(str(exc))

    with action_mid:
        if st.button("Reset Responses", use_container_width=True):
            st.session_state.benchmark_dataset = []
            st.session_state.benchmark_index = 0
            st.session_state.collected_items = []
            st.session_state.benchmark_result = None
            st.success("Benchmark workspace reset.")

    with action_right:
        if st.button("Unload Model", use_container_width=True):
            try:
                unload_benchmark_model(backend_url)
                st.success("Model unloaded.")
            except APIClientError as exc:
                st.error(str(exc))

    st.caption("Generate and save each dataset item first. The Gemini judge runs only when you click Evaluate Benchmark.")


def render_staged_benchmark(backend_url: str, model_name: str, max_items: int) -> None:
    st.subheader("Dataset Response Collection")
    if st.button("Load Dataset Items", use_container_width=True):
        try:
            st.session_state.benchmark_dataset = get_benchmark_dataset(backend_url, max_items=max_items)
            st.session_state.benchmark_index = 0
            st.session_state.collected_items = []
            st.success("Dataset loaded.")
        except APIClientError as exc:
            st.error(str(exc))
            return

    dataset = st.session_state.get("benchmark_dataset", [])
    if not dataset:
        st.info("Load dataset items to start collecting model responses.")
        return

    collected = st.session_state.get("collected_items", [])
    index = min(st.session_state.get("benchmark_index", 0), len(dataset) - 1)
    item = dataset[index]

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Item", f"{index + 1}/{len(dataset)}")
    c2.metric("Saved Responses", len(collected))
    c3.metric("Remaining", max(len(dataset) - len(collected), 0))

    st.markdown(f"#### {item['item_id']} · {item['subject']} · {item['language']}")
    edited_prompt = st.text_area(
        "Editable benchmark prompt",
        value=item["student_prompt"],
        height=120,
        key=f"prompt_{item['item_id']}",
    )
    st.caption(f"Rubric: {item['rubric']}")

    response_key = f"response_{item['item_id']}"
    pending_response_key = f"pending_{response_key}"
    if pending_response_key in st.session_state:
        st.session_state[response_key] = st.session_state.pop(pending_response_key)
    st.text_area("Model response", key=response_key, height=240)

    left, mid, right = st.columns(3)
    with left:
        if st.button("Generate Response", type="primary", use_container_width=True):
            try:
                current = get_benchmark_model_state(backend_url)
                if current.get("active_model") != model_name:
                    st.warning("Load the selected model before generating.")
                    return
                with st.spinner("Generating one response..."):
                    result = generate_model_response(
                        backend_url=backend_url,
                        model_name=model_name,
                        prompt=edited_prompt,
                        student_level=item["student_level"],
                        language=item["language"],
                    )
                st.session_state[pending_response_key] = result["response"]
                st.session_state[f"latency_{item['item_id']}"] = result.get("latency_ms", 0.0)
                st.rerun()
            except APIClientError as exc:
                st.error(str(exc))

    with mid:
        if st.button("Save Response & Next", use_container_width=True):
            response = st.session_state.get(response_key, "").strip()
            if not response:
                st.warning("Generate or type a response before saving.")
                return
            saved = {
                "item_id": item["item_id"],
                "student_prompt": edited_prompt,
                "student_level": item["student_level"],
                "language": item["language"],
                "subject": item["subject"],
                "rubric": item["rubric"],
                "response": response,
                "generation_latency_ms": float(st.session_state.get(f"latency_{item['item_id']}", 0.0)),
            }
            st.session_state.collected_items = [
                existing for existing in collected if existing["item_id"] != item["item_id"]
            ] + [saved]
            st.session_state.benchmark_index = min(index + 1, len(dataset) - 1)
            st.success("Response saved.")
            st.rerun()

    with right:
        ready = len(st.session_state.get("collected_items", [])) >= len(dataset)
        if st.button("Evaluate Benchmark", type="primary", disabled=not ready, use_container_width=True):
            try:
                with st.spinner("Evaluating saved responses with Gemini judge..."):
                    st.session_state.benchmark_result = evaluate_collected_benchmark(
                        base_url=backend_url,
                        model_name=model_name,
                        items=st.session_state.collected_items,
                    )
                st.success("Benchmark evaluation complete.")
            except APIClientError as exc:
                st.error(str(exc))

    render_collected_table(st.session_state.get("collected_items", []))


def generate_model_response(
    *,
    backend_url: str,
    model_name: str,
    prompt: str,
    student_level: str,
    language: str,
) -> dict[str, Any]:
    normalized = model_name.casefold()
    if "qwen" in normalized:
        return generate_qwen_response(
            base_url=backend_url,
            student_prompt=prompt,
            student_level=student_level,
            language=language,
            model_name=model_name,
        )
    if "gemma" in normalized:
        return generate_gemma_response(
            base_url=backend_url,
            student_prompt=prompt,
            student_level=student_level,
            language=language,
            model_name=model_name,
        )
    if "llama" in normalized:
        return generate_llama_response(
            base_url=backend_url,
            student_prompt=prompt,
            student_level=student_level,
            language=language,
            model_name=model_name,
        )
    raise APIClientError(f"Generation is not connected for {model_name}.")


def render_collected_table(items: list[dict[str, Any]]) -> None:
    if not items:
        return
    st.markdown("#### Saved Responses")
    rows = [
        {
            "Item": item["item_id"],
            "Subject": item["subject"],
            "Language": item["language"],
            "Latency (ms)": item["generation_latency_ms"],
            "Response Preview": item["response"][:160],
        }
        for item in items
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_dataset_preview(backend_url: str, max_items: int) -> None:
    st.subheader("Benchmark Dataset")
    try:
        dataset = get_benchmark_dataset(backend_url, max_items=max_items)
    except APIClientError as exc:
        st.error(str(exc))
        return

    rows = [
        {
            "Item": item["item_id"],
            "Subject": item["subject"],
            "Level": item["student_level"],
            "Language": item["language"],
            "Prompt": item["student_prompt"],
        }
        for item in dataset
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_benchmark_result(result: dict[str, Any] | None) -> None:
    if not result:
        st.info("Run a benchmark to see aggregate educational metrics.")
        return

    aggregate = result["aggregate"]
    st.subheader(f"Benchmark Results · {result['model_name']}")

    cols = st.columns(5)
    cols[0].metric("Overall", aggregate["average_overall_score"])
    cols[1].metric("Correctness", aggregate["avg_correctness"])
    cols[2].metric("Teaching", aggregate["avg_teaching_quality"])
    cols[3].metric("Multilingual", aggregate["avg_multilingual_quality"])
    cols[4].metric("Safety", aggregate["avg_hallucination_risk"])

    score_rows = [
        {"Metric": "Correctness", "Score": aggregate["avg_correctness"]},
        {"Metric": "Teaching Quality", "Score": aggregate["avg_teaching_quality"]},
        {"Metric": "Adaptation", "Score": aggregate["avg_adaptation"]},
        {"Metric": "Emotional Intelligence", "Score": aggregate["avg_emotional_intelligence"]},
        {"Metric": "Multilingual Quality", "Score": aggregate["avg_multilingual_quality"]},
        {"Metric": "Low Hallucination / Safety", "Score": aggregate["avg_hallucination_risk"]},
        {"Metric": "Conversation Quality", "Score": aggregate["avg_conversation_quality"]},
    ]
    fig = px.bar(pd.DataFrame(score_rows), x="Metric", y="Score", range_y=[0, 10], template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Item-Level Benchmark Table")
    item_rows = [
        {
            "Item": item["item_id"],
            "Subject": item["subject"],
            "Language": item["language"],
            "Overall": item["overall_score"],
            "Learning": item["learning_effectiveness"],
            "Safety": item["safety_classification"],
            "Generation Latency (ms)": item["generation_latency_ms"],
            "Judge Latency (ms)": item["judge_latency_ms"],
            "Reasoning": item["reasoning"],
        }
        for item in result["items"]
    ]
    st.dataframe(pd.DataFrame(item_rows), use_container_width=True, hide_index=True)

    with st.expander("Benchmark Report", expanded=True):
        st.markdown(result["benchmark_report_markdown"])

    with st.expander("Generated Responses"):
        for item in result["items"]:
            st.markdown(f"**{item['item_id']} · {item['subject']} · {item['language']}**")
            st.write(item["response"])


def render_benchmark_history(backend_url: str) -> None:
    st.subheader("Benchmark Run History")
    try:
        runs = get_benchmark_runs(backend_url)
    except APIClientError as exc:
        st.error(str(exc))
        return
    if not runs:
        st.info("No benchmark runs stored yet.")
        return
    st.dataframe(pd.DataFrame(runs), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
