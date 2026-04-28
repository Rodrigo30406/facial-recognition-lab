from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eleccia_voice.assistant import VoiceAssistant, VoiceSettings


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    priority: str = "normal"
    context: dict[str, Any] = field(default_factory=dict)


class ElecciaVoiceService:
    """Reusable voice module for Eleccia skills beyond facial recognition."""

    def __init__(self, settings: VoiceSettings) -> None:
        self._assistant = VoiceAssistant(settings=settings)

    @property
    def backend_kind(self) -> str | None:
        return self._assistant.backend_kind

    @property
    def backend_error(self) -> str | None:
        return self._assistant.backend_error

    def speak(self, request: SpeechRequest | str) -> bool:
        if isinstance(request, str):
            text = request
        else:
            text = request.text

        message = str(text).strip()
        if not message:
            return False
        return self._assistant.speak(message)

    def close(self) -> None:
        self._assistant.close()
