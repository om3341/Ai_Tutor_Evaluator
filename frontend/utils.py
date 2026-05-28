from __future__ import annotations

import re
from typing import Any


MODEL_OPTIONS = ["Sarvam 30B", "Qwen 3.8B", "google/gemma-4-E4B", "Llama 3.1 8B"]

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think\b[^>]*>.*$", flags=re.IGNORECASE | re.DOTALL)
_STRAY_THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", flags=re.IGNORECASE)

CATEGORY_LABELS = {
    "correctness": "Conceptual Quality",
    "teaching_quality": "Pedagogical Quality",
    "adaptation": "Student Adaptation",
    "emotional_intelligence": "Emotional Support",
    "multilingual_quality": "Multilingual Fidelity",
    "hallucination_risk": "Safety / Low Hallucination",
    "conversation_quality": "Dialogue Quality",
}

RADAR_CATEGORY_ORDER = [
    "correctness",
    "teaching_quality",
    "adaptation",
    "emotional_intelligence",
    "multilingual_quality",
    "hallucination_risk",
    "conversation_quality",
]

BAR_CATEGORY_ORDER = [
    "correctness",
    "teaching_quality",
    "emotional_intelligence",
    "multilingual_quality",
]


def average_score(scores: dict[str, int]) -> float:
    return round(sum(scores.values()) / len(scores), 2)


def winner_model_name(winner: str, model_a: str, model_b: str) -> str:
    return model_a if winner == "A" else model_b


def score_delta(scores_a: dict[str, int], scores_b: dict[str, int], category: str) -> int:
    return int(scores_a[category]) - int(scores_b[category])


def strip_thinking_tags(text: str) -> str:
    """Remove hidden reasoning wrappers before displaying or submitting responses."""

    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _UNCLOSED_THINK_RE.sub("", cleaned)
    cleaned = _STRAY_THINK_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def make_summary_rows(
    evaluation: dict[str, Any],
    model_a: str,
    model_b: str,
    latency_a_ms: float,
    latency_b_ms: float,
) -> list[dict[str, Any]]:
    scores_a = evaluation["scores"]["A"]
    scores_b = evaluation["scores"]["B"]
    return [
        {
            "Model": model_a,
            "Side": "A",
            "Average": average_score(scores_a),
            "Conceptual Quality": scores_a["correctness"],
            "Pedagogical Quality": scores_a["teaching_quality"],
            "Emotional Support": scores_a["emotional_intelligence"],
            "Multilingual Fidelity": scores_a["multilingual_quality"],
            "Safety / Low Hallucination": scores_a["hallucination_risk"],
            "Dialogue Quality": scores_a["conversation_quality"],
            "Student Adaptation": scores_a["adaptation"],
            "Response Latency (ms)": latency_a_ms,
        },
        {
            "Model": model_b,
            "Side": "B",
            "Average": average_score(scores_b),
            "Conceptual Quality": scores_b["correctness"],
            "Pedagogical Quality": scores_b["teaching_quality"],
            "Emotional Support": scores_b["emotional_intelligence"],
            "Multilingual Fidelity": scores_b["multilingual_quality"],
            "Safety / Low Hallucination": scores_b["hallucination_risk"],
            "Dialogue Quality": scores_b["conversation_quality"],
            "Student Adaptation": scores_b["adaptation"],
            "Response Latency (ms)": latency_b_ms,
        },
    ]


def sample_response(model_name: str, prompt: str, student_level: str, language: str) -> str:
    """Create editable placeholder responses for local demos.

    Step 2 does not add model-provider integrations. These drafts make the UI
    usable immediately while keeping candidate responses transparent and editable.
    """

    topic = prompt.strip() or "the concept"
    normalized_model_name = model_name.lower()

    if "sarvam" in normalized_model_name:
        return (
            f"Chalo {student_level} level pe simple way mein samajhte hain. "
            f"{topic} ka core idea yeh hai ki pehle basic meaning clear karo, "
            "phir ek everyday example se connect karo. Agar tum confused ho, "
            "toh hum isse step by step tod sakte hain: definition, example, "
            "and one quick check question."
        )
    if "qwen" in normalized_model_name:
        return (
            f"For {student_level}, here is a structured explanation in {language}. "
            f"The question asks: {topic}. First, identify the main concept. Second, "
            "explain the cause-and-effect relationship. Third, verify understanding "
            "with a short example and a practice question."
        )
    if "gemma" in normalized_model_name:
        return (
            f"Great question. For a {student_level} student, I would explain it like this: "
            f"{topic} can be understood by breaking it into small parts. Start with the "
            "main idea, connect it to something familiar, and then check whether the "
            "student can explain it back in their own words."
        )

    if "llama" in normalized_model_name:
        return (
            f"For {student_level}, let's make this clear and practical. "
            f"{topic} is easiest to understand when we first name the key idea, "
            "then walk through one example, and finally ask a quick follow-up so "
            "the student can test their understanding."
        )

    return (
        f"Great question. For a {student_level} student, I would explain it like this: "
        f"{topic} can be understood by breaking it into small parts. Start with the "
        "main idea, connect it to something familiar, and then check whether the "
        "student can explain it back in their own words."
    )
