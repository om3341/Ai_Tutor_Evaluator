from __future__ import annotations

from dataclasses import dataclass

from backend.schemas import SimulationRunRequest
from backend.simulation.model_gateway import ChatResult, ModelGateway
from backend.simulation.transcript_manager import TranscriptManager


@dataclass(frozen=True)
class PersonaProfile:
    name: str
    learning_behavior: str
    curiosity_level: str
    misunderstanding_patterns: str
    follow_up_tendencies: str
    emotional_traits: str


PERSONA_PROFILES: dict[str, PersonaProfile] = {
    "Curious Student": PersonaProfile(
        name="Curious Student",
        learning_behavior="Actively connects ideas and asks why/how questions.",
        curiosity_level="High",
        misunderstanding_patterns="Occasional overgeneralization from examples.",
        follow_up_tendencies="Asks deeper follow-ups and requests examples.",
        emotional_traits="Positive, energetic, and comfortable admitting uncertainty.",
    ),
    "Weak Student": PersonaProfile(
        name="Weak Student",
        learning_behavior="Needs simple steps, repetition, and concrete analogies.",
        curiosity_level="Medium",
        misunderstanding_patterns="Confuses key vocabulary and causal relationships.",
        follow_up_tendencies="Asks for simpler explanations and makes partial mistakes.",
        emotional_traits="Hesitant, easily discouraged, but willing to try.",
    ),
    "Average Student": PersonaProfile(
        name="Average Student",
        learning_behavior="Understands familiar examples but needs guided practice.",
        curiosity_level="Medium",
        misunderstanding_patterns="Misses edge cases and intermediate reasoning.",
        follow_up_tendencies="Asks practical questions and checks understanding.",
        emotional_traits="Neutral and cooperative.",
    ),
    "Advanced Student": PersonaProfile(
        name="Advanced Student",
        learning_behavior="Learns quickly and asks for extensions or exceptions.",
        curiosity_level="High",
        misunderstanding_patterns="May jump ahead and skip foundational checks.",
        follow_up_tendencies="Requests challenge questions and conceptual depth.",
        emotional_traits="Confident and analytical.",
    ),
    "Distracted Student": PersonaProfile(
        name="Distracted Student",
        learning_behavior="Loses focus and needs redirection to the topic.",
        curiosity_level="Low to medium",
        misunderstanding_patterns="Forgets previous points and mixes unrelated ideas.",
        follow_up_tendencies="Gives short replies, drifts off-topic, and needs prompts.",
        emotional_traits="Restless but not hostile.",
    ),
    "Exam-Anxious Student": PersonaProfile(
        name="Exam-Anxious Student",
        learning_behavior="Wants exam-focused clarity and reassurance.",
        curiosity_level="Medium",
        misunderstanding_patterns="Panics, asks if answers are enough, and memorizes without understanding.",
        follow_up_tendencies="Requests quick checks, likely questions, and confirmation.",
        emotional_traits="Nervous, self-doubting, and sensitive to tone.",
    ),
}


class StudentAgent:
    """Generates realistic student turns with persona and transcript memory."""

    def __init__(self, gateway: ModelGateway) -> None:
        self._gateway = gateway

    async def generate(self, request: SimulationRunRequest, transcript: TranscriptManager, turn_index: int) -> ChatResult:
        profile = PERSONA_PROFILES[request.persona]
        system_prompt = (
            "You are simulating a K-12 student in an educational benchmark. "
            "You are not the tutor. Ask, react, misunderstand sometimes, and learn gradually. "
            "Do not immediately master the topic. Keep messages concise and realistic."
        )
        phase = (
            "reply to the tutor's opening explanation and ask your first follow-up question"
            if turn_index == 1
            else "reply to the tutor and ask a follow-up"
        )
        user_prompt = (
            f"Task: {phase}.\n"
            f"Topic: {request.topic}\n"
            f"Student level: {request.student_level}\n"
            f"Language: {request.language}\n"
            f"Persona: {profile.name}\n"
            f"Learning behavior: {profile.learning_behavior}\n"
            f"Curiosity level: {profile.curiosity_level}\n"
            f"Misunderstanding patterns: {profile.misunderstanding_patterns}\n"
            f"Follow-up tendencies: {profile.follow_up_tendencies}\n"
            f"Emotional traits: {profile.emotional_traits}\n\n"
            f"Conversation so far:\n{transcript.render_for_prompt()}\n\n"
            "Write only the next student message. Include confusion, a mistake, a check of understanding, "
            "or a follow-up when natural. Match the requested language."
        )
        return await self._gateway.chat(
            model_name=request.student_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=request.student_temperature,
            max_tokens=220,
        )
