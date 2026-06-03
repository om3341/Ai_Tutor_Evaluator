from __future__ import annotations

from html import escape
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
    get_simulation_personas,
    get_rag_health,
    get_simulation_run,
    get_simulation_runs,
    get_simulation_state,
    load_benchmark_model,
    load_simulation_models,
    stream_simulation,
    stop_simulation,
    unload_benchmark_model,
    unload_simulation_models,
    unload_simulation_student,
    unload_simulation_tutor,
)
from frontend.components import hero, inject_theme
from frontend.analytics.simulation_dashboard import render_rag_analytics_dashboard
from frontend.utils import MODEL_OPTIONS


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def main() -> None:
    st.set_page_config(
       
        page_icon="AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    st.session_state.setdefault("benchmark_result", None)
    st.session_state.setdefault("simulation_result", None)
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
    st.caption("")

    simulation_tab, run_tab, results_tab, history_tab = st.tabs(
        ["Student-Tutor Simulation", "Single Response Benchmark", "Results", "History"]
    )

    with simulation_tab:
        render_simulation_dashboard(backend_url)

    with run_tab:
        render_model_console(backend_url, model_name, int(max_items))
        render_staged_benchmark(backend_url, model_name, int(max_items))

    with results_tab:
        render_benchmark_result(st.session_state.get("benchmark_result"))

    with history_tab:
        render_benchmark_history(backend_url)
        render_simulation_history(backend_url)


def render_simulation_dashboard(backend_url: str) -> None:
    st.markdown(
        """
        <div class="simulation-heading">
          <div>
            <h2>Student-Tutor Simulation</h2>
            <p>Set up a lesson, load both roles, and watch the learning conversation unfold.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        personas = get_simulation_personas(backend_url)
        persona_names = [persona["name"] for persona in personas]
    except APIClientError:
        personas = []
        persona_names = [
            "Curious Student",
            "Weak Student",
            "Average Student",
            "Advanced Student",
            "Distracted Student",
            "Exam-Anxious Student",
        ]

    try:
        state = get_simulation_state(backend_url)
    except APIClientError as exc:
        st.error(str(exc))
        state = {"tutor_loaded": False, "student_loaded": False, "tutor_model": None, "student_model": None}

    with st.sidebar:
        st.divider()
        st.header("Simulation Setup")
        st.caption("Configure both roles and the lesson before starting.")
        try:
            rag_health = get_rag_health(backend_url)
            if rag_health.get("reachable") and rag_health.get("collection_exists"):
                st.success(f"Qdrant ready · {rag_health['collection_name']}")
            else:
                st.warning(rag_health.get("detail", "Qdrant context collection is not ready."))
        except APIClientError as exc:
            st.warning(str(exc))

        st.markdown('<div class="role-label">Tutor</div>', unsafe_allow_html=True)
        tutor_model = st.selectbox("Tutor model", MODEL_OPTIONS, index=0, key="sim_tutor", label_visibility="collapsed")
        tutor_loaded = state.get("tutor_loaded", False)
        st.markdown(_model_status("Tutor", state.get("tutor_model"), tutor_loaded), unsafe_allow_html=True)
        tutor_load, tutor_unload = st.columns([2, 1])
        if tutor_load.button("Load Tutor", type="primary", use_container_width=True):
            try:
                load_simulation_models(backend_url, tutor_model=tutor_model)
                st.success("Tutor loaded.")
                st.rerun()
            except APIClientError as exc:
                st.error(str(exc))
        if tutor_unload.button("Unload", use_container_width=True, key="unload_sim_tutor"):
            try:
                unload_simulation_tutor(backend_url)
                st.success("Tutor model unloaded.")
                st.rerun()
            except APIClientError as exc:
                st.error(str(exc))

        st.markdown('<div class="role-label">Student</div>', unsafe_allow_html=True)
        student_model = st.selectbox(
            "Student model",
            MODEL_OPTIONS,
            index=1 if len(MODEL_OPTIONS) > 1 else 0,
            key="sim_student",
            label_visibility="collapsed",
        )
        student_loaded = state.get("student_loaded", False)
        st.markdown(_model_status("Student", state.get("student_model"), student_loaded), unsafe_allow_html=True)
        student_load, student_unload = st.columns([2, 1])
        if student_load.button("Load Student", type="primary", use_container_width=True):
            try:
                load_simulation_models(backend_url, student_model=student_model)
                st.success("Student loaded.")
                st.rerun()
            except APIClientError as exc:
                st.error(str(exc))
        if student_unload.button("Unload", use_container_width=True, key="unload_sim_student"):
            try:
                unload_simulation_student(backend_url)
                st.success("Student model unloaded.")
                st.rerun()
            except APIClientError as exc:
                st.error(str(exc))

        st.markdown('<div class="role-label lesson-label">Lesson</div>', unsafe_allow_html=True)
        topic = st.text_input("Topic", value="Photosynthesis", placeholder="What should the tutor teach?")
        student_level = st.selectbox(
            "Student level",
            ["Class 4", "Class 6", "Class 8", "Class 10", "Class 12"],
            index=1,
        )
        language = st.selectbox("Language", ["English", "Hinglish", "Hindi", "Marathi"], index=1)
        persona = st.selectbox("Student persona", persona_names, index=1 if "Weak Student" in persona_names else 0)
        selected_persona = next((item for item in personas if item["name"] == persona), None)
        if selected_persona:
            st.markdown(_persona_summary(selected_persona), unsafe_allow_html=True)

        with st.expander("Advanced settings", expanded=False):
            tutor_temperature = st.slider("Tutor creativity", 0.0, 1.5, 0.2, 0.05)
            student_temperature = st.slider("Student variability", 0.0, 1.5, 0.6, 0.05)
            turns = st.slider("Conversation turns", 1, 12, 4)

    both_loaded = state.get("tutor_loaded", False) and state.get("student_loaded", False)
    st.markdown(
        _classroom_status(
            running=bool(state.get("running", False)),
            ready=both_loaded,
            topic=topic,
            student_level=student_level,
            language=language,
            persona=persona,
        ),
        unsafe_allow_html=True,
    )

    stage_col, activity_col = st.columns([2.35, 1], gap="large")
    with stage_col:
        st.markdown(
            f"""
            <div class="stage-card">
              <div class="stage-title">
                <h3>Live Classroom Stage</h3>
                <span class="lesson-chip">{escape(topic)} · {escape(student_level)}</span>
              </div>
            """,
            unsafe_allow_html=True,
        )
        status_placeholder = st.empty()
        live_placeholder = st.empty()
        render_simulation_result(st.session_state.get("simulation_result"))
        st.markdown("</div>", unsafe_allow_html=True)

        run_col, stop_col, clear_col = st.columns([3, 1, 1])
        with run_col:
            if st.button(
                "Start Session",
                type="primary",
                use_container_width=True,
                disabled=not both_loaded or state.get("running", False),
            ):
                try:
                    st.session_state.simulation_result = None
                    live_retrieval: dict[str, Any] | None = None
                    live_transcript: list[dict[str, Any]] = []
                    completed = False
                    status_placeholder.info("Starting simulation and retrieving lesson context...")
                    for event in stream_simulation(
                        base_url=backend_url,
                        tutor_model=tutor_model,
                        student_model=student_model,
                        topic=topic,
                        student_level=student_level,
                        language=language,
                        persona=persona,
                        tutor_temperature=tutor_temperature,
                        student_temperature=student_temperature,
                        turns=turns,
                    ):
                        event_type = event.get("event")
                        if event_type == "retrieval":
                            live_retrieval = event.get("retrieval") or {}
                            chunk_count = len(live_retrieval.get("chunks") or [])
                            status_placeholder.info(f"Retrieved {chunk_count} context chunks. Tutor is starting...")
                            render_live_simulation(live_placeholder, live_retrieval, live_transcript)
                        elif event_type == "message":
                            message = event.get("message") or {}
                            live_transcript.append(message)
                            speaker = "Student" if message.get("role") == "student" else "Tutor"
                            status_placeholder.info(f"{speaker} responded in turn {message.get('turn_index')}.")
                            render_live_simulation(live_placeholder, live_retrieval, live_transcript)
                        elif event_type == "judging":
                            status_placeholder.info(event.get("message", "Conversation complete. Running judge..."))
                        elif event_type == "complete":
                            st.session_state.simulation_result = event.get("result")
                            completed = True
                            status_placeholder.success("Simulation complete.")
                    if completed:
                        live_placeholder.empty()
                except APIClientError as exc:
                    st.error(str(exc))
        with stop_col:
            if st.button("Stop", use_container_width=True, disabled=not state.get("running", False)):
                try:
                    stop_simulation(backend_url)
                    st.info("Stop requested.")
                except APIClientError as exc:
                    st.error(str(exc))
        with clear_col:
            if st.button("Unload", use_container_width=True):
                try:
                    unload_simulation_models(backend_url)
                    st.success("Simulation models unloaded.")
                    st.rerun()
                except APIClientError as exc:
                    st.error(str(exc))

        if not both_loaded:
            st.caption("Load a tutor and a student to start the session.")

    with activity_col:
        _render_activity_panel(
            state=state,
            topic=topic,
            language=language,
            persona=persona,
            turns=turns,
            result=st.session_state.get("simulation_result"),
        )


def render_live_simulation(
    placeholder: st.delta_generator.DeltaGenerator,
    retrieval: dict[str, Any] | None,
    transcript: list[dict[str, Any]],
) -> None:
    with placeholder.container():
        st.markdown("### Live Conversation")
        st.caption("Messages appear here as soon as each model finishes its turn.")
        if retrieval:
            chunks = retrieval.get("chunks") or []
            st.markdown(
                f'<div class="live-context">Context frozen from '
                f'<strong>{escape(str(retrieval.get("collection_name", "")))}</strong> · '
                f'{len(chunks)} chunks</div>',
                unsafe_allow_html=True,
            )
        if not transcript:
            st.info("Waiting for the tutor's opening explanation...")
            return
        message_cards = []
        for message in transcript:
            role = message.get("role", "student")
            label = "Student" if role == "student" else "Tutor"
            card_class = "stu" if role == "student" else "tut"
            content = escape(str(message.get("content", "")))
            turn = escape(str(message.get("turn_index", "-")))
            message_cards.append(
                f'<div class="msg {card_class}"><div class="who">{label} · Turn {turn}</div>{content}</div>'
            )
        st.markdown(f'<div class="stream">{"".join(message_cards)}</div>', unsafe_allow_html=True)


def _classroom_status(
    *,
    running: bool,
    ready: bool,
    topic: str,
    student_level: str,
    language: str,
    persona: str,
) -> str:
    if running:
        phase = "live"
        status = "Simulation in progress"
    elif ready:
        phase = "ready"
        status = "Ready to start"
    else:
        phase = "setup"
        status = "Load both models"
    live_class = " live" if running else ""
    return (
        f'<div class="classroom-status{live_class}">'
        f'<span class="pulse-dot"></span>'
        f'<span><strong>{status}</strong><br />'
        f'<small>{escape(topic)} · {escape(student_level)} · {escape(language)} · {escape(persona)}</small></span>'
        f'<span class="phase-pill">{phase}</span>'
        f'</div>'
    )


def _render_activity_panel(
    *,
    state: dict[str, Any],
    topic: str,
    language: str,
    persona: str,
    turns: int,
    result: dict[str, Any] | None,
) -> None:
    metrics = (result or {}).get("metrics") or {}
    retrieval = (result or {}).get("retrieval") or {}
    chunks = retrieval.get("chunks") or []
    st.markdown(
        f"""
        <div class="activity-card">
          <h4>Activity</h4>
          <div class="activity-line"><span>Tutor</span><strong>{escape(str(state.get("tutor_model") or "Not loaded"))}</strong></div>
          <div class="activity-line"><span>Student</span><strong>{escape(str(state.get("student_model") or "Not loaded"))}</strong></div>
          <div class="activity-line"><span>Topic</span><strong>{escape(topic)}</strong></div>
          <div class="activity-line"><span>Language</span><strong>{escape(language)}</strong></div>
          <div class="activity-line"><span>Persona</span><strong>{escape(persona)}</strong></div>
          <div class="activity-line"><span>Turns</span><strong>{turns}</strong></div>
          <div class="activity-line"><span>RAG chunks</span><strong>{len(chunks) if result else "—"}</strong></div>
          <div class="activity-line"><span>Learning gain</span><strong>{metrics.get("learning_gain", "—")}</strong></div>
          <div class="activity-line"><span>Grounding</span><strong>{metrics.get("knowledge_grounding", "—")}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _model_status(role: str, model_name: str | None, loaded: bool) -> str:
    state_class = "online" if loaded else "offline"
    state_label = "Ready" if loaded else "Not loaded"
    model_label = model_name or f"No {role.lower()} model active"
    return (
        f'<div class="model-status"><span class="status-dot {state_class}"></span>'
        f'<span><strong>{state_label}</strong><small>{model_label}</small></span></div>'
    )


def _persona_summary(persona: dict[str, Any]) -> str:
    behavior = persona.get("learning_behavior") or persona.get("description") or "Learns through conversation."
    return f'<div class="persona-summary"><strong>Learning behavior</strong><p>{escape(str(behavior))}</p></div>'


def render_simulation_result(result: dict[str, Any] | None) -> None:
    if not result:
        st.markdown(
            '<div class="empty-state"><strong>Your conversation will appear here.</strong>'
            "<span>Run a simulation to review the transcript and learning metrics.</span></div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown("---")
    st.markdown("### Retrieved Context")
    st.caption("This context was retrieved once and frozen for every tutor turn.")
    retrieval = result.get("retrieval") or {}
    chunks = retrieval.get("chunks") or []
    if chunks:
        st.caption(
            f"Query: {retrieval.get('query', '')} · Collection: {retrieval.get('collection_name', '')} "
            f"· Chunks: {len(chunks)}"
        )
        for index, chunk in enumerate(chunks, start=1):
            with st.expander(f"Chunk {index} · score {chunk.get('score', 0):.4f}"):
                st.write(chunk.get("text", ""))
                metadata = chunk.get("metadata") or {}
                if metadata:
                    st.json(metadata)
    else:
        st.warning("No retrieved context was stored for this simulation.")

    st.markdown("### Conversation Replay")
    st.caption("Follow how the student asks, responds, and builds understanding over time.")
    for message in result.get("transcript", []):
        role = message["role"].title()
        with st.chat_message("user" if message["role"] == "student" else "assistant"):
            st.markdown(f"**{role} · Turn {message['turn_index']}**")
            st.write(message["content"])

    metrics = result.get("metrics", {})
    if metrics:
        st.markdown("### RAG Analytics")
        rag_col1, rag_col2, rag_col3, rag_col4 = st.columns(4)
        rag_col1.metric("Grounding", metrics.get("knowledge_grounding", "-"))
        rag_col2.metric("Faithfulness", metrics.get("context_faithfulness", "-"))
        rag_col3.metric("Retrieval Use", metrics.get("retrieval_utilization", "-"))
        rag_col4.metric("Educational Grounding", metrics.get("educational_grounding", "-"))

        st.markdown("### Learning Metrics")
        metric_rows = [{"Metric": key.replace("_", " ").title(), "Score": value} for key, value in metrics.items()]
        df = pd.DataFrame(metric_rows)
        fig = px.bar(df, x="Metric", y="Score", range_y=[0, 10], color_discrete_sequence=["#0284c7"])
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=12, b=10))
        
        st.plotly_chart(fig, 
                        use_container_width=True,
                         key="learning_gain_chart")
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Judge Notes")
    st.write(result.get("reasoning", ""))
    failure_modes = result.get("failure_modes", [])
    if failure_modes:
        st.markdown("#### Failure Modes")
        for mode in failure_modes:
            st.warning(mode)


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


def render_simulation_history(backend_url: str) -> None:
    st.subheader("Student-Tutor Simulation History")
    try:
        runs = get_simulation_runs(backend_url)
    except APIClientError as exc:
        st.error(str(exc))
        return
    if not runs:
        st.info("No conversation simulations stored yet.")
        return
    rows = []
    for run in runs:
        metrics = run.get("metrics") or {}
        rows.append(
            {
                "Run": run["run_id"],
                "Tutor": run["tutor_model"],
                "Student": run["student_model"],
                "Topic": run["topic"],
                "Persona": run["persona"],
                "Turns": run["turns"],
                "Learning Gain": metrics.get("learning_gain"),
                "Scaffolding": metrics.get("scaffolding_effectiveness"),
                "Engagement": metrics.get("student_engagement"),
                "Grounding": metrics.get("knowledge_grounding"),
                "Faithfulness": metrics.get("context_faithfulness"),
                "Retrieval Use": metrics.get("retrieval_utilization"),
                "Educational Grounding": metrics.get("educational_grounding"),
                "Created": run["created_at"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    render_rag_analytics_dashboard(runs)
    selected_run_id = st.selectbox("Replay a simulation", [run["run_id"] for run in runs])
    if st.button("Open Replay", use_container_width=True):
        try:
            render_simulation_result(get_simulation_run(backend_url, selected_run_id))
        except APIClientError as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
