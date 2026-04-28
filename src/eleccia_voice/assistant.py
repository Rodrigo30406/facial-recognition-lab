from __future__ import annotations

import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from eleccia_vision.domain.entities import RecognitionResult

PersonMetadataResolver = Callable[[str], tuple[str, str | None]]
KNOWN_PRESENCE_ID = "__known_presence__"
UNKNOWN_PRESENCE_ID = "__unknown_presence__"
DEFAULT_UNKNOWN_GREETING = "Hola, bienvenido al laboratorio de IA"


@dataclass(frozen=True)
class VoiceBackend:
    kind: str
    engine: Any | None = None


@dataclass(frozen=True)
class VoiceSettings:
    enabled: bool
    backend: str = "auto"
    template: str = "Hola {name}"
    reentry_delay_seconds: float = 8.0
    absence_seconds: float = 1.2
    voice_rate: int | None = None
    voice_volume: float | None = None
    voice_id: str | None = None
    voice_lang: str | None = None
    melo_language: str | None = None
    melo_speaker: str | None = None
    melo_speed: float | None = None
    melo_device: str | None = None
    unknown_greeting: str = DEFAULT_UNKNOWN_GREETING
    min_face_ratio_for_greeting: float = 0.0


@dataclass
class GreetingState:
    known_presence_person_id: dict[str, str] = field(default_factory=dict)
    known_presence_greeted: dict[str, bool] = field(default_factory=dict)
    known_presence_last_seen_ts: dict[str, float] = field(default_factory=dict)
    known_presence_regreet_armed: dict[str, bool] = field(default_factory=dict)
    last_greet_ts_by_person: dict[str, float] = field(default_factory=dict)
    unknown_presence_last_seen_ts: dict[str, float] = field(default_factory=dict)
    unknown_presence_greeted: dict[str, bool] = field(default_factory=dict)
    person_name_cache: dict[str, str] = field(default_factory=dict)
    person_sex_cache: dict[str, str | None] = field(default_factory=dict)
    backend_warning_shown: bool = False


class VoiceAssistant:
    def __init__(self, settings: VoiceSettings) -> None:
        self._settings = settings
        self._state = GreetingState()
        self._backend: VoiceBackend | None = None
        self.backend_error: str | None = None

        if settings.enabled:
            self._backend, self.backend_error = _build_voice_backend(settings)

    @property
    def backend_kind(self) -> str | None:
        if self._backend is None:
            return None
        return self._backend.kind

    def close(self) -> None:
        _close_voice_backend(self._backend)

    def is_regreet_marker_active(self, presence_id: str | None) -> bool:
        known_key = _build_known_presence_key(presence_id)
        return bool(self._state.known_presence_regreet_armed.get(known_key, False))

    def speak(self, text: str) -> bool:
        message = str(text).strip()
        if not message:
            return False
        if self._backend is None:
            return False
        return _speak_message(self._backend, message)

    def on_recognition(
        self,
        result: RecognitionResult,
        resolve_person: PersonMetadataResolver,
        face_ratio: float | None = None,
        pose_yaw: float | None = None,
        pose_pitch: float | None = None,
        now: float | None = None,
        presence_id: str | None = None,
    ) -> str | None:
        if not self._settings.enabled:
            return None

        ts = time.time() if now is None else now
        known_presence_key = _build_known_presence_key(presence_id)
        self._cleanup_stale_known_presences(now_ts=ts, keep_presence_key=known_presence_key)
        self._cleanup_stale_unknown_presences(now_ts=ts)
        current_known_person_id = self._state.known_presence_person_id.get(known_presence_key)
        close_enough = _is_face_close_enough(
            face_ratio=face_ratio,
            pose_yaw=pose_yaw,
            pose_pitch=pose_pitch,
            min_ratio=self._settings.min_face_ratio_for_greeting,
        )

        if result.decision == "known_person" and result.person_id is not None and close_enough:
            person_id = result.person_id
            if current_known_person_id != person_id:
                self._state.known_presence_person_id[known_presence_key] = person_id
                self._state.known_presence_greeted[known_presence_key] = False
            self._state.known_presence_last_seen_ts[known_presence_key] = ts

            if self._state.known_presence_greeted.get(known_presence_key, False):
                return None

            min_delay = max(0.0, float(self._settings.reentry_delay_seconds))
            last_greet = self._state.last_greet_ts_by_person.get(person_id, 0.0)
            if (ts - last_greet) < min_delay:
                return None

            display_name, person_sex = self._resolve_person_metadata(person_id, resolve_person)
            message = format_voice_message(
                template=self._settings.template,
                name=display_name,
                person_id=person_id,
                sex=person_sex,
            )

            if self._backend is None:
                self._state.known_presence_greeted[known_presence_key] = True
                self._state.known_presence_regreet_armed[known_presence_key] = False
                self._state.last_greet_ts_by_person[person_id] = ts
                if not self._state.backend_warning_shown:
                    self._state.backend_warning_shown = True
                    return "Voice greet: instala MeloTTS, pyttsx3, espeak o spd-say"
                return None

            if _speak_message(self._backend, message):
                self._state.known_presence_greeted[known_presence_key] = True
                self._state.known_presence_regreet_armed[known_presence_key] = False
                self._state.last_greet_ts_by_person[person_id] = ts
                return f"Saludo: {message}"
            return None

        if _should_keep_known_presence_on_ambiguous(
            result=result,
            current_person_id=current_known_person_id,
            close_enough=close_enough,
        ):
            self._state.known_presence_last_seen_ts[known_presence_key] = ts
            return None

        if _is_unknown_face_detected(result, face_ratio=face_ratio) and close_enough:
            # If we were tracking a known person, unknown frames should not replace
            # that presence immediately; let absence timeout handle the reset.
            if current_known_person_id is not None:
                pass
            else:
                unknown_presence_key = _build_unknown_presence_key(presence_id)
                self._state.unknown_presence_last_seen_ts[unknown_presence_key] = ts
                greeted = self._state.unknown_presence_greeted.get(unknown_presence_key, False)

                if greeted:
                    return None

                min_delay = max(0.0, float(self._settings.reentry_delay_seconds))
                last_greet = self._state.last_greet_ts_by_person.get(unknown_presence_key, 0.0)
                if (ts - last_greet) < min_delay:
                    return None

                message = (self._settings.unknown_greeting or "").strip() or DEFAULT_UNKNOWN_GREETING

                if self._backend is None:
                    self._state.unknown_presence_greeted[unknown_presence_key] = True
                    self._state.last_greet_ts_by_person[unknown_presence_key] = ts
                    if not self._state.backend_warning_shown:
                        self._state.backend_warning_shown = True
                        return "Voice greet: instala MeloTTS, pyttsx3, espeak o spd-say"
                    return None

                if _speak_message(self._backend, message):
                    self._state.unknown_presence_greeted[unknown_presence_key] = True
                    self._state.last_greet_ts_by_person[unknown_presence_key] = ts
                    return f"Saludo: {message}"
                return None

        if current_known_person_id is None:
            return None

        absence_seconds = max(0.0, float(self._settings.absence_seconds))
        last_seen = self._state.known_presence_last_seen_ts.get(known_presence_key, 0.0)
        if (ts - last_seen) >= absence_seconds:
            self._state.known_presence_person_id.pop(known_presence_key, None)
            self._state.known_presence_greeted.pop(known_presence_key, None)
            self._state.known_presence_last_seen_ts.pop(known_presence_key, None)
            self._state.known_presence_regreet_armed[known_presence_key] = True
        return None

    def _cleanup_stale_known_presences(self, now_ts: float, keep_presence_key: str | None = None) -> None:
        absence_seconds = max(0.0, float(self._settings.absence_seconds))
        if absence_seconds <= 0.0:
            return

        stale_keys = [
            key
            for key, last_seen in self._state.known_presence_last_seen_ts.items()
            if key != keep_presence_key
            if (now_ts - last_seen) >= absence_seconds
        ]
        for key in stale_keys:
            self._state.known_presence_person_id.pop(key, None)
            self._state.known_presence_greeted.pop(key, None)
            self._state.known_presence_last_seen_ts.pop(key, None)
            self._state.known_presence_regreet_armed[key] = True

    def _cleanup_stale_unknown_presences(self, now_ts: float) -> None:
        absence_seconds = max(0.0, float(self._settings.absence_seconds))
        if absence_seconds <= 0.0:
            return

        stale_keys = [
            key
            for key, last_seen in self._state.unknown_presence_last_seen_ts.items()
            if (now_ts - last_seen) >= absence_seconds
        ]
        for key in stale_keys:
            self._state.unknown_presence_last_seen_ts.pop(key, None)
            self._state.unknown_presence_greeted.pop(key, None)

    def _resolve_person_metadata(
        self,
        person_id: str,
        resolve_person: PersonMetadataResolver,
    ) -> tuple[str, str | None]:
        cached_name = self._state.person_name_cache.get(person_id)
        cached_sex_exists = person_id in self._state.person_sex_cache
        if cached_name is not None and cached_sex_exists:
            return cached_name, self._state.person_sex_cache[person_id]

        name = person_id
        sex: str | None = None
        try:
            resolved_name, resolved_sex = resolve_person(person_id)
            if isinstance(resolved_name, str) and resolved_name.strip():
                name = resolved_name.strip()
            if resolved_sex is not None:
                raw_sex = str(resolved_sex).strip()
                sex = raw_sex if raw_sex else None
        except Exception:
            name = person_id
            sex = None

        self._state.person_name_cache[person_id] = name
        self._state.person_sex_cache[person_id] = sex
        return name, sex


def build_voice_settings_from_args(args: Any) -> VoiceSettings:
    return VoiceSettings(
        enabled=bool(getattr(args, "voice_greet", False)),
        backend=str(getattr(args, "voice_backend", "auto")),
        template=str(getattr(args, "voice_template", "Hola {name}")),
        reentry_delay_seconds=max(0.0, float(getattr(args, "voice_reentry_delay_seconds", 8.0))),
        absence_seconds=max(0.0, float(getattr(args, "voice_absence_seconds", 1.2))),
        voice_rate=getattr(args, "voice_rate", None),
        voice_volume=getattr(args, "voice_volume", None),
        voice_id=getattr(args, "voice_id", None),
        voice_lang=getattr(args, "voice_lang", None),
        melo_language=getattr(args, "melo_language", None),
        melo_speaker=getattr(args, "melo_speaker", None),
        melo_speed=getattr(args, "melo_speed", None),
        melo_device=getattr(args, "melo_device", None),
        min_face_ratio_for_greeting=max(0.0, float(getattr(args, "voice_min_face_ratio", 0.0))),
    )


def _is_unknown_face_detected(result: RecognitionResult, face_ratio: float | None = None) -> bool:
    if result.decision != "unknown_person":
        return False
    # Evidence source 1: ranked candidates exist (gallery had identities).
    if (result.top1 is not None) or (result.top2 is not None):
        return True
    # Evidence source 2: no gallery candidates but a face was detected in frame.
    if face_ratio is not None and float(face_ratio) > 0.0:
        return True
    return False


def _build_known_presence_key(presence_id: str | None) -> str:
    if presence_id is None:
        return KNOWN_PRESENCE_ID
    value = str(presence_id).strip()
    if not value:
        return KNOWN_PRESENCE_ID
    return f"{KNOWN_PRESENCE_ID}::{value}"


def _build_unknown_presence_key(presence_id: str | None) -> str:
    if presence_id is None:
        return UNKNOWN_PRESENCE_ID
    value = str(presence_id).strip()
    if not value:
        return UNKNOWN_PRESENCE_ID
    return f"{UNKNOWN_PRESENCE_ID}::{value}"


def _should_keep_known_presence_on_ambiguous(
    *,
    result: RecognitionResult,
    current_person_id: str | None,
    close_enough: bool,
) -> bool:
    if not close_enough:
        return False
    if current_person_id is None or current_person_id == UNKNOWN_PRESENCE_ID:
        return False
    if result.decision != "ambiguous_match" or result.top1 is None:
        return False
    return result.top1.person_id == current_person_id


def _is_face_close_enough(
    face_ratio: float | None,
    pose_yaw: float | None,
    pose_pitch: float | None,
    min_ratio: float,
) -> bool:
    threshold = max(0.0, float(min_ratio))
    if threshold <= 0.0:
        return True
    if face_ratio is None:
        return False
    adjusted_ratio = _pose_adjusted_face_ratio(
        face_ratio=float(face_ratio),
        pose_yaw=pose_yaw,
        pose_pitch=pose_pitch,
    )
    return adjusted_ratio >= threshold


def _pose_adjusted_face_ratio(
    face_ratio: float,
    pose_yaw: float | None,
    pose_pitch: float | None,
) -> float:
    base = max(0.0, float(face_ratio))

    # Projective compensation: side/tilted faces look smaller in 2D bbox.
    # We approximate frontal-equivalent area by dividing by cos(yaw)*cos(pitch),
    # clamped to avoid over-correction on noisy/extreme poses.
    yaw = _clamp_abs_angle(pose_yaw, max_abs=55.0)
    pitch = _clamp_abs_angle(pose_pitch, max_abs=40.0)
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)

    denom = max(0.40, math.cos(yaw_rad) * math.cos(pitch_rad))
    factor = min(1.8, 1.0 / denom)
    return base * factor


def _clamp_abs_angle(value: float | None, max_abs: float) -> float:
    if value is None:
        return 0.0
    v = abs(float(value))
    return min(v, float(max_abs))


def format_voice_message(template: str, name: str, person_id: str, sex: str | None) -> str:
    welcome = resolve_welcome_word(sex)

    try:
        rendered = template.format(name=name, person_id=person_id, sex=sex or "", welcome=welcome)
    except Exception:
        rendered = f"Hola {name}, {welcome}"

    rendered = re.sub(r"\bbienvenid[oa]\b", welcome, rendered, flags=re.IGNORECASE)
    return rendered


def resolve_welcome_word(sex: str | None) -> str:
    if sex is None:
        return "bienvenido(a)"

    value = str(sex).strip().lower()
    female_values = {"female", "f", "femenino", "mujer", "woman"}
    male_values = {"male", "m", "masculino", "hombre", "man"}
    other_values = {"other", "x", "otro", "no_binario", "no-binario", "nonbinary", "nb"}

    if value in female_values:
        return "bienvenida"
    if value in male_values:
        return "bienvenido"
    if value in other_values:
        return "bienvenido(a)"
    return "bienvenido(a)"


class _Pyttsx3Speaker:
    def __init__(
        self,
        rate: int | None = None,
        volume: float | None = None,
        voice_id: str | None = None,
        voice_lang: str | None = None,
    ) -> None:
        self._rate = rate
        self._volume = volume
        self._voice_id = voice_id
        self._voice_lang = voice_lang
        self._queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._ready = threading.Event()
        self._failed = False
        self._thread = threading.Thread(target=self._run, name="pyttsx3-speaker", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2.0)

    def _run(self) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            if self._rate is not None:
                engine.setProperty("rate", int(self._rate))
            if self._volume is not None:
                vol = min(1.0, max(0.0, float(self._volume)))
                engine.setProperty("volume", vol)
            if self._voice_id:
                engine.setProperty("voice", self._voice_id)
            elif self._voice_lang:
                voice = _select_pyttsx3_voice(engine, self._voice_lang)
                if voice is not None:
                    engine.setProperty("voice", voice)
        except Exception:
            self._failed = True
            self._ready.set()
            return

        self._ready.set()
        while True:
            message = self._queue.get()
            if message is None:
                break
            try:
                engine.say(message)
                engine.runAndWait()
            except Exception:
                continue

        try:
            engine.stop()
        except Exception:
            pass

    def enqueue(self, message: str) -> bool:
        if self._failed:
            return False
        self._queue.put(message)
        return True

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=1.0)


class _MeloTTSSpeaker:
    def __init__(
        self,
        language: str = "ES",
        speaker: str | None = None,
        speed: float = 1.0,
        device: str = "auto",
    ) -> None:
        self._language = language
        self._speaker = speaker
        self._speed = speed
        self._device = device
        self._queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._ready = threading.Event()
        self._failed = False
        self._last_error: str | None = None
        self._thread = threading.Thread(target=self._run, name="melotts-speaker", daemon=True)
        self._thread.start()
        ready = self._ready.wait(timeout=300.0)
        if not ready:
            self._failed = True
            self._last_error = "MeloTTS initialization timed out (download/loading took too long)"

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _run(self) -> None:
        try:
            from melo.api import TTS
        except Exception as exc:
            self._failed = True
            self._last_error = f"Could not import MeloTTS: {exc}"
            self._ready.set()
            return

        sounddevice = None
        try:
            import sounddevice as sd  # type: ignore

            sounddevice = sd
        except Exception:
            sounddevice = None

        player = None if sounddevice is not None else _detect_audio_player()
        if sounddevice is None and player is None:
            self._failed = True
            self._last_error = "No audio output backend available (sounddevice/paplay/aplay/ffplay)"
            self._ready.set()
            return

        try:
            model = TTS(language=self._language, device=self._device)
            spk2id = _extract_melo_speaker_map(model)
            speaker_value = _resolve_melo_speaker_value(spk2id, self._speaker)
            if speaker_value is None:
                self._failed = True
                self._last_error = (
                    "Could not resolve MeloTTS speaker (spk2id missing and default speaker_id=0 failed)"
                )
                self._ready.set()
                return
        except Exception as exc:
            self._failed = True
            self._last_error = f"MeloTTS model init failed: {exc}"
            self._ready.set()
            return

        self._ready.set()
        while True:
            message = self._queue.get()
            if message is None:
                break
            tmp_path = ""
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                model.tts_to_file(
                    message,
                    speaker_value,
                    tmp_path,
                    speed=float(self._speed),
                )
                _play_wav_path(tmp_path, sounddevice=sounddevice, player=player)
            except Exception as exc:
                self._last_error = f"MeloTTS synth/playback error: {exc}"
                continue
            finally:
                if tmp_path:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except Exception:
                        pass

    def enqueue(self, message: str) -> bool:
        if self._failed:
            return False
        self._queue.put(message)
        return True

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=1.0)


def _build_voice_backend(settings: VoiceSettings) -> tuple[VoiceBackend | None, str | None]:
    selected = settings.backend
    if selected == "auto":
        if _can_import_melotts():
            selected = "melotts"
        elif _can_import_pyttsx3():
            selected = "pyttsx3"
        elif shutil.which("spd-say"):
            selected = "spd-say"
        elif shutil.which("espeak"):
            selected = "espeak"
        else:
            return None, "No TTS backend found (melotts/pyttsx3/spd-say/espeak)"

    if selected == "melotts":
        try:
            speaker = _MeloTTSSpeaker(
                language=_resolve_melo_language(settings),
                speaker=settings.melo_speaker,
                speed=_resolve_melo_speed(settings),
                device=_resolve_melo_device(settings),
            )
            if speaker._failed:
                return None, speaker.last_error
            return VoiceBackend(kind="melotts", engine=speaker), None
        except Exception as exc:
            return None, f"MeloTTS init failed: {exc}"

    if selected == "pyttsx3":
        try:
            speaker = _Pyttsx3Speaker(
                rate=settings.voice_rate,
                volume=settings.voice_volume,
                voice_id=settings.voice_id,
                voice_lang=settings.voice_lang,
            )
            if speaker._failed:
                return None, "pyttsx3 init failed"
            return VoiceBackend(kind="pyttsx3", engine=speaker), None
        except Exception as exc:
            return None, f"pyttsx3 init failed: {exc}"

    if selected == "spd-say":
        if shutil.which("spd-say"):
            return VoiceBackend(kind="spd-say", engine={"lang": settings.voice_lang}), None
        return None, "spd-say executable not found"

    if selected == "espeak":
        if shutil.which("espeak"):
            return VoiceBackend(kind="espeak", engine={"lang": settings.voice_lang}), None
        return None, "espeak executable not found"

    return None, f"Unsupported voice backend '{selected}'"


def _can_import_pyttsx3() -> bool:
    try:
        import pyttsx3  # noqa: F401

        return True
    except Exception:
        return False


def _can_import_melotts() -> bool:
    try:
        from melo.api import TTS  # noqa: F401

        return True
    except Exception:
        return False


def _resolve_melo_language(settings: VoiceSettings) -> str:
    raw = settings.melo_language or settings.voice_lang or "ES"
    value = str(raw).strip().upper().replace("-", "_")
    mapping = {
        "EN": "EN",
        "ES": "ES",
        "FR": "FR",
        "ZH": "ZH",
        "JP": "JP",
        "JA": "JP",
        "KR": "KR",
        "KO": "KR",
    }
    return mapping.get(value, "ES")


def _resolve_melo_speed(settings: VoiceSettings) -> float:
    if settings.melo_speed is not None:
        return max(0.1, float(settings.melo_speed))
    if settings.voice_rate is not None:
        return max(0.6, min(1.6, float(settings.voice_rate) / 150.0))
    return 1.0


def _resolve_melo_device(settings: VoiceSettings) -> str:
    raw = settings.melo_device
    if raw is None:
        return "auto"
    value = str(raw).strip()
    return value if value else "auto"


def _speak_message(backend: VoiceBackend, message: str) -> bool:
    try:
        if backend.kind == "melotts":
            if not isinstance(backend.engine, _MeloTTSSpeaker):
                return False
            return backend.engine.enqueue(message)
        if backend.kind == "pyttsx3":
            if not isinstance(backend.engine, _Pyttsx3Speaker):
                return False
            return backend.engine.enqueue(message)
        if backend.kind == "spd-say":
            lang = None
            if isinstance(backend.engine, dict):
                lang = backend.engine.get("lang")
            cmd = ["spd-say"]
            if isinstance(lang, str) and lang.strip():
                cmd.extend(["-l", lang.strip()])
            cmd.append(message)
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        if backend.kind == "espeak":
            lang = None
            if isinstance(backend.engine, dict):
                lang = backend.engine.get("lang")
            cmd = ["espeak"]
            if isinstance(lang, str) and lang.strip():
                cmd.extend(["-v", lang.strip()])
            cmd.append(message)
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
    except Exception:
        return False
    return False


def _close_voice_backend(backend: VoiceBackend | None) -> None:
    if backend is None:
        return
    if backend.kind == "melotts" and isinstance(backend.engine, _MeloTTSSpeaker):
        backend.engine.close()
        return
    if backend.kind == "pyttsx3" and isinstance(backend.engine, _Pyttsx3Speaker):
        backend.engine.close()
        return
def _detect_audio_player() -> list[str] | None:
    if shutil.which("paplay"):
        return ["paplay"]
    if shutil.which("aplay"):
        return ["aplay"]
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
    return None


def _resolve_melo_speaker_key(spk2id: dict[str, Any], preferred: str | None) -> str | None:
    if preferred:
        if preferred in spk2id:
            return preferred
        pref = preferred.lower()
        for key in spk2id:
            if key.lower() == pref:
                return key
    return next(iter(spk2id.keys()), None)


def _resolve_melo_speaker_value(spk2id: dict[str, Any] | None, preferred: str | None) -> int | None:
    if isinstance(spk2id, dict) and spk2id:
        key = _resolve_melo_speaker_key(spk2id, preferred)
        if key is not None:
            try:
                return int(spk2id[key])
            except Exception:
                pass
    return 0


def _extract_melo_speaker_map(model: Any) -> dict[str, Any] | None:
    hps = getattr(model, "hps", None)
    data = None
    if isinstance(hps, dict):
        data = hps.get("data")
    else:
        data = getattr(hps, "data", None)

    if isinstance(data, dict):
        spk2id = data.get("spk2id")
        if isinstance(spk2id, dict):
            return spk2id
    else:
        spk2id = getattr(data, "spk2id", None)
        if isinstance(spk2id, dict):
            return spk2id

    direct = getattr(model, "spk2id", None)
    if isinstance(direct, dict):
        return direct
    return None


def _play_wav_path(path: str, sounddevice: Any | None, player: list[str] | None) -> None:
    if sounddevice is not None:
        try:
            with wave.open(path, "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())

            if sample_width == 2:
                data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            if n_channels > 1:
                data = data.reshape(-1, n_channels)

            sounddevice.play(data, framerate, blocking=True)
            sounddevice.stop()
            return
        except Exception:
            pass

    if player is None:
        return

    subprocess.run(
        [*player, path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _play_audio(
    audio: np.ndarray,
    sample_rate: int,
    sounddevice: Any | None,
    player: list[str] | None,
) -> None:
    if sounddevice is not None:
        try:
            sounddevice.play(audio, sample_rate, blocking=True)
            sounddevice.stop()
            return
        except Exception:
            pass

    if player is None:
        return

    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())

        subprocess.run(
            [*player, tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _select_pyttsx3_voice(engine: Any, language_hint: str) -> str | None:
    hint = language_hint.strip().lower().replace("-", "_")
    if not hint:
        return None

    try:
        voices = engine.getProperty("voices")
    except Exception:
        return None

    for voice in voices:
        if _voice_matches_hint(voice, hint):
            voice_id = getattr(voice, "id", None)
            if isinstance(voice_id, str) and voice_id.strip():
                return voice_id
    return None


def _voice_matches_hint(voice: Any, hint: str) -> bool:
    candidates: list[str] = []

    voice_id = getattr(voice, "id", None)
    if isinstance(voice_id, str):
        candidates.append(voice_id.lower())

    voice_name = getattr(voice, "name", None)
    if isinstance(voice_name, str):
        candidates.append(voice_name.lower())

    langs = getattr(voice, "languages", None)
    if isinstance(langs, (list, tuple)):
        for item in langs:
            if isinstance(item, bytes):
                try:
                    decoded = item.decode("utf-8", errors="ignore").lower()
                except Exception:
                    decoded = ""
                if decoded:
                    candidates.append(decoded)
            elif isinstance(item, str):
                candidates.append(item.lower())

    simple_hint = hint.split("_", 1)[0]
    for candidate in candidates:
        normalized = candidate.replace("-", "_")
        if hint in normalized:
            return True
        if simple_hint and simple_hint in normalized:
            return True
    return False
