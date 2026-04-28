from types import SimpleNamespace
from unittest.mock import patch

from eleccia_voice import ElecciaVoiceService, SpeechRequest, VoiceSettings


def test_eleccia_voice_service_speak_with_string() -> None:
    service = ElecciaVoiceService(
        VoiceSettings(
            enabled=True,
            backend="unsupported-backend",
        )
    )
    service._assistant._backend = SimpleNamespace(kind="stub", engine=None)  # type: ignore[attr-defined]

    with patch("eleccia_voice.assistant._speak_message", return_value=True):
        ok = service.speak("Hola desde eleccia")
    service.close()

    assert ok is True


def test_eleccia_voice_service_speak_with_request() -> None:
    service = ElecciaVoiceService(
        VoiceSettings(
            enabled=True,
            backend="unsupported-backend",
        )
    )
    service._assistant._backend = SimpleNamespace(kind="stub", engine=None)  # type: ignore[attr-defined]

    request = SpeechRequest(
        text="Agenda creada",
        priority="high",
        context={"source": "agenda"},
    )
    with patch("eleccia_voice.assistant._speak_message", return_value=True):
        ok = service.speak(request)
    service.close()

    assert ok is True
