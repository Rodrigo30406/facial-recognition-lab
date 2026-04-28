from types import SimpleNamespace
from unittest.mock import patch

from eleccia_vision.domain.entities import RecognitionCandidate, RecognitionResult
from eleccia_voice import VoiceAssistant, VoiceSettings, format_voice_message, resolve_welcome_word


def _known(person_id: str) -> RecognitionResult:
    return RecognitionResult(
        decision="known_person",
        matched=True,
        person_id=person_id,
        top1=None,
        top2=None,
    )


def _ambiguous(top1_person_id: str, score: float = 0.75) -> RecognitionResult:
    return RecognitionResult(
        decision="ambiguous_match",
        matched=False,
        person_id=None,
        top1=RecognitionCandidate(person_id=top1_person_id, score=score),
        top2=RecognitionCandidate(person_id="other", score=score - 0.02),
    )


def test_resolve_welcome_word_variants() -> None:
    assert resolve_welcome_word("female") == "bienvenida"
    assert resolve_welcome_word("male") == "bienvenido"
    assert resolve_welcome_word("other") == "bienvenido(a)"


def test_format_voice_message_uses_welcome_token() -> None:
    text = format_voice_message(
        template="Hola {name}, {welcome} al laboratorio de IA",
        name="Ana",
        person_id="ana01",
        sex="female",
    )
    assert text == "Hola Ana, bienvenida al laboratorio de IA"


def test_voice_assistant_warning_is_emitted_once_when_backend_missing() -> None:
    assistant = VoiceAssistant(
        VoiceSettings(
            enabled=True,
            backend="unsupported-backend",
            template="Hola {name}, {welcome}",
        )
    )
    msg1 = assistant.on_recognition(
        _known("alice"),
        resolve_person=lambda _person_id: ("Alice Doe", "female"),
        now=10.0,
    )
    msg2 = assistant.on_recognition(
        _known("alice"),
        resolve_person=lambda _person_id: ("Alice Doe", "female"),
        now=10.1,
    )
    assistant.close()

    assert msg1 is not None
    assert msg1.startswith("Voice greet:")
    assert msg2 is None


def test_voice_assistant_unknown_person_greeting() -> None:
    assistant = VoiceAssistant(
        VoiceSettings(
            enabled=True,
            backend="unsupported-backend",
            unknown_greeting="Hola, bienvenido al laboratorio de IA",
        )
    )
    assistant._backend = SimpleNamespace(kind="stub", engine=None)

    unknown = RecognitionResult(
        decision="unknown_person",
        matched=False,
        person_id=None,
        top1=RecognitionCandidate(person_id="known_ref", score=0.42),
        top2=None,
    )

    with patch("eleccia_voice.assistant._speak_message", return_value=True):
        msg1 = assistant.on_recognition(
            unknown,
            resolve_person=lambda _person_id: ("", None),
            now=10.0,
        )
        msg2 = assistant.on_recognition(
            unknown,
            resolve_person=lambda _person_id: ("", None),
            now=10.1,
        )
    assistant.close()

    assert msg1 == "Saludo: Hola, bienvenido al laboratorio de IA"
    assert msg2 is None


def test_voice_assistant_unknown_without_candidates_but_face_ratio_greets() -> None:
    assistant = VoiceAssistant(
        VoiceSettings(
            enabled=True,
            backend="unsupported-backend",
            unknown_greeting="Hola, bienvenido al laboratorio de IA",
        )
    )
    assistant._backend = SimpleNamespace(kind="stub", engine=None)

    unknown = RecognitionResult(
        decision="unknown_person",
        matched=False,
        person_id=None,
        top1=None,
        top2=None,
    )

    with patch("eleccia_voice.assistant._speak_message", return_value=True):
        msg = assistant.on_recognition(
            unknown,
            resolve_person=lambda _person_id: ("", None),
            face_ratio=0.12,
            now=10.0,
        )
    assistant.close()

    assert msg == "Saludo: Hola, bienvenido al laboratorio de IA"


def test_voice_assistant_proximity_filter_blocks_far_face() -> None:
    assistant = VoiceAssistant(
        VoiceSettings(
            enabled=True,
            backend="unsupported-backend",
            min_face_ratio_for_greeting=0.10,
            template="Hola {name}, {welcome}",
        )
    )
    assistant._backend = SimpleNamespace(kind="stub", engine=None)

    with patch("eleccia_voice.assistant._speak_message", return_value=True):
        far_msg = assistant.on_recognition(
            _known("alice"),
            resolve_person=lambda _person_id: ("Alice Doe", "female"),
            face_ratio=0.05,
            now=10.0,
        )
        near_msg = assistant.on_recognition(
            _known("alice"),
            resolve_person=lambda _person_id: ("Alice Doe", "female"),
            face_ratio=0.15,
            now=11.0,
        )
    assistant.close()

    assert far_msg is None
    assert near_msg == "Saludo: Hola Alice Doe, bienvenida"


def test_voice_assistant_pose_compensation_allows_side_face() -> None:
    assistant = VoiceAssistant(
        VoiceSettings(
            enabled=True,
            backend="unsupported-backend",
            min_face_ratio_for_greeting=0.10,
            template="Hola {name}, {welcome}",
        )
    )
    assistant._backend = SimpleNamespace(kind="stub", engine=None)

    with patch("eleccia_voice.assistant._speak_message", return_value=True):
        side_msg = assistant.on_recognition(
            _known("alice"),
            resolve_person=lambda _person_id: ("Alice Doe", "female"),
            face_ratio=0.08,
            pose_yaw=45.0,
            pose_pitch=0.0,
            now=12.0,
        )
    assistant.close()

    assert side_msg == "Saludo: Hola Alice Doe, bienvenida"


def test_voice_assistant_ambiguous_same_top1_keeps_presence() -> None:
    assistant = VoiceAssistant(
        VoiceSettings(
            enabled=True,
            backend="unsupported-backend",
            template="Hola {name}",
            absence_seconds=1.0,
            reentry_delay_seconds=0.0,
        )
    )
    assistant._backend = SimpleNamespace(kind="stub", engine=None)

    with patch("eleccia_voice.assistant._speak_message", return_value=True):
        msg1 = assistant.on_recognition(
            _known("alice"),
            resolve_person=lambda _person_id: ("Alice Doe", "female"),
            now=10.0,
        )
        msg2 = assistant.on_recognition(
            _ambiguous("alice"),
            resolve_person=lambda _person_id: ("Alice Doe", "female"),
            now=11.5,
        )
        msg3 = assistant.on_recognition(
            _known("alice"),
            resolve_person=lambda _person_id: ("Alice Doe", "female"),
            now=11.6,
        )
    assistant.close()

    assert msg1 == "Saludo: Hola Alice Doe"
    assert msg2 is None
    assert msg3 is None


def test_voice_assistant_ambiguous_top1_change_starts_absence_counter() -> None:
    assistant = VoiceAssistant(
        VoiceSettings(
            enabled=True,
            backend="unsupported-backend",
            template="Hola {name}",
            absence_seconds=1.0,
            reentry_delay_seconds=0.0,
        )
    )
    assistant._backend = SimpleNamespace(kind="stub", engine=None)

    with patch("eleccia_voice.assistant._speak_message", return_value=True):
        msg1 = assistant.on_recognition(
            _known("alice"),
            resolve_person=lambda _person_id: ("Alice Doe", "female"),
            now=20.0,
        )
        msg2 = assistant.on_recognition(
            _ambiguous("bob"),
            resolve_person=lambda _person_id: ("Alice Doe", "female"),
            now=21.5,
        )
        msg3 = assistant.on_recognition(
            _known("alice"),
            resolve_person=lambda _person_id: ("Alice Doe", "female"),
            now=21.6,
        )
    assistant.close()

    assert msg1 == "Saludo: Hola Alice Doe"
    assert msg2 is None
    assert msg3 == "Saludo: Hola Alice Doe"


def test_voice_assistant_unknown_after_known_starts_absence_counter() -> None:
    assistant = VoiceAssistant(
        VoiceSettings(
            enabled=True,
            backend="unsupported-backend",
            template="Hola {name}",
            unknown_greeting="Hola, bienvenido al laboratorio de IA",
            absence_seconds=1.0,
            reentry_delay_seconds=0.0,
        )
    )
    assistant._backend = SimpleNamespace(kind="stub", engine=None)

    below_threshold_unknown = RecognitionResult(
        decision="unknown_person",
        matched=False,
        person_id=None,
        top1=RecognitionCandidate(person_id="alice", score=0.42),
        top2=RecognitionCandidate(person_id="bob", score=0.40),
    )

    with patch("eleccia_voice.assistant._speak_message", return_value=True):
        msg1 = assistant.on_recognition(
            _known("alice"),
            resolve_person=lambda _person_id: ("Alice Doe", "female"),
            now=30.0,
        )
        msg2 = assistant.on_recognition(
            below_threshold_unknown,
            resolve_person=lambda _person_id: ("Alice Doe", "female"),
            face_ratio=0.12,
            now=31.5,
        )
        msg3 = assistant.on_recognition(
            _known("alice"),
            resolve_person=lambda _person_id: ("Alice Doe", "female"),
            now=31.6,
        )
    assistant.close()

    assert msg1 == "Saludo: Hola Alice Doe"
    assert msg2 is None
    assert msg3 == "Saludo: Hola Alice Doe"


def test_voice_assistant_unknown_is_tracked_independently_per_presence_id() -> None:
    assistant = VoiceAssistant(
        VoiceSettings(
            enabled=True,
            backend="unsupported-backend",
            unknown_greeting="Hola, bienvenido al laboratorio de IA",
            absence_seconds=1.0,
            reentry_delay_seconds=0.0,
        )
    )
    assistant._backend = SimpleNamespace(kind="stub", engine=None)
    unknown = RecognitionResult(
        decision="unknown_person",
        matched=False,
        person_id=None,
        top1=RecognitionCandidate(person_id="known_ref", score=0.42),
        top2=None,
    )

    with patch("eleccia_voice.assistant._speak_message", return_value=True):
        msg1 = assistant.on_recognition(
            unknown,
            resolve_person=lambda _person_id: ("", None),
            face_ratio=0.12,
            now=40.0,
            presence_id="track-a",
        )
        msg2 = assistant.on_recognition(
            unknown,
            resolve_person=lambda _person_id: ("", None),
            face_ratio=0.12,
            now=40.1,
            presence_id="track-a",
        )
        msg3 = assistant.on_recognition(
            unknown,
            resolve_person=lambda _person_id: ("", None),
            face_ratio=0.12,
            now=40.2,
            presence_id="track-b",
        )
    assistant.close()

    assert msg1 == "Saludo: Hola, bienvenido al laboratorio de IA"
    assert msg2 is None
    assert msg3 == "Saludo: Hola, bienvenido al laboratorio de IA"
