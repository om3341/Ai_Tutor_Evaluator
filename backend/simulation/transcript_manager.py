from __future__ import annotations

from backend.schemas import SimulationMessage


class TranscriptManager:
    """Maintains ordered student/tutor messages for a simulation run."""

    def __init__(self) -> None:
        self._messages: list[SimulationMessage] = []

    @property
    def messages(self) -> list[SimulationMessage]:
        return list(self._messages)

    def add(self, *, role: str, content: str, turn_index: int, latency_ms: float) -> SimulationMessage:
        message = SimulationMessage(
            role=role,  # type: ignore[arg-type]
            content=content.strip(),
            turn_index=turn_index,
            latency_ms=latency_ms,
        )
        self._messages.append(message)
        return message

    def append(self, message: SimulationMessage) -> None:
        self._messages.append(message)

    def render_for_prompt(self, max_messages: int = 12) -> str:
        recent = self._messages[-max_messages:]
        if not recent:
            return "No previous messages."
        return "\n".join(f"{message.role.title()}: {message.content}" for message in recent)

    def model_dump(self) -> list[dict[str, object]]:
        return [message.model_dump() for message in self._messages]
