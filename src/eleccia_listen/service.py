from __future__ import annotations

from collections import deque
from difflib import SequenceMatcher
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

from eleccia_audio import audio_io_lock

CommandHandler = Callable[["CommandEvent"], None]


@dataclass(frozen=True)
class ListenSettings:
    enabled: bool = False
    backend: str = "stdin"
    stdin_prompt: str = "eleccia> "
    wake_word: str = "eleccia"
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_language: str | None = "es"
    whisper_beam_size: int = 5
    whisper_vad_filter: bool = True
    whisper_chunk_seconds: float = 4.0
    whisper_sample_rate_hz: int = 16000
    whisper_input_device_index: int | None = None
    whisper_min_rms: float = 0.003
    whisper_endpointing_enabled: bool = True
    whisper_frame_seconds: float = 0.2
    whisper_speech_start_seconds: float = 0.2
    whisper_silence_stop_seconds: float = 2.0
    whisper_max_utterance_seconds: float = 8.0
    whisper_pre_roll_seconds: float = 0.3
    debug_timing: bool = False
    noise_filter_enabled: bool = False
    openwakeword_model_paths: tuple[str, ...] = ()
    openwakeword_inference_framework: str = "onnx"
    openwakeword_threshold: float = 0.5
    openwakeword_chunk_size: int = 1280
    openwakeword_cooldown_seconds: float = 1.5
    require_wake_word: bool = True
    wake_word_aliases: tuple[str, ...] = ()
    wake_word_fuzzy_threshold: float = 0.80
    wake_command_window_seconds: float = 6.0


@dataclass(frozen=True)
class CommandIntent:
    name: str
    confidence: float = 1.0
    slots: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandEvent:
    text: str
    normalized_text: str
    intent: CommandIntent
    ts: float


class ElecciaListenService:
    """Command listening module (stdin or Whisper via faster-whisper)."""

    def __init__(self, settings: ListenSettings, on_command: CommandHandler | None = None) -> None:
        self._settings = settings
        self._on_command = on_command
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if not self._settings.enabled:
            return
        if self._running:
            return

        backend = self._settings.backend.strip().lower()
        if backend not in {"stdin", "whisper", "openwakeword_whisper"}:
            raise ValueError(f"Unsupported listen backend '{self._settings.backend}'")

        self._stop_event.clear()
        if backend == "stdin":
            target = self._run_stdin
        elif backend == "whisper":
            target = self._run_whisper
        else:
            target = self._run_openwakeword_whisper
        self._thread = threading.Thread(target=target, name=f"eleccia-listen-{backend}", daemon=True)
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None
        self._running = False

    def _run_stdin(self) -> None:
        prompt = self._settings.stdin_prompt
        while not self._stop_event.is_set():
            try:
                text = input(prompt)
            except EOFError:
                break
            except Exception:
                continue

            if self._stop_event.is_set():
                break
            if text is None:
                continue
            message = str(text).strip()
            if not message:
                continue

            self._dispatch_text(message)

    def _run_whisper(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
            from faster_whisper import WhisperModel
        except Exception as exc:
            print(f"[eleccia][listen] whisper backend unavailable: {exc}")
            return

        model = WhisperModel(
            self._settings.whisper_model,
            device=self._settings.whisper_device,
            compute_type=self._settings.whisper_compute_type,
        )
        noise_filter = _AdaptiveNoiseFilter() if self._settings.noise_filter_enabled else None
        if self._settings.noise_filter_enabled:
            print("[eleccia][listen] noise filter enabled")
        if self._settings.whisper_endpointing_enabled:
            self._run_whisper_endpointing(model=model, np=np, sd=sd, noise_filter=noise_filter)
            return
        self._run_whisper_chunked(model=model, np=np, sd=sd, noise_filter=noise_filter)

    def _run_whisper_chunked(
        self,
        model: object,
        np: object,
        sd: object,
        noise_filter: "_AdaptiveNoiseFilter | None" = None,
    ) -> None:
        chunk_seconds = max(0.5, float(self._settings.whisper_chunk_seconds))
        sample_rate = max(8000, int(self._settings.whisper_sample_rate_hz))
        n_samples = int(chunk_seconds * sample_rate)
        min_rms = max(0.0, float(self._settings.whisper_min_rms))

        while not self._stop_event.is_set():
            try:
                with audio_io_lock():
                    audio = sd.rec(
                        n_samples,
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                        device=self._settings.whisper_input_device_index,
                    )
                    sd.wait()
            except Exception:
                time.sleep(0.2)
                continue

            if self._stop_event.is_set():
                break

            waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
            if noise_filter is not None:
                waveform = noise_filter.process(waveform, speaking=True)
            self._transcribe_and_dispatch(
                model=model,
                waveform=waveform,
                np=np,
                min_rms=min_rms,
                capture_reason="chunk",
            )

    def _run_whisper_endpointing(
        self,
        model: object,
        np: object,
        sd: object,
        noise_filter: "_AdaptiveNoiseFilter | None" = None,
    ) -> None:
        sample_rate = max(8000, int(self._settings.whisper_sample_rate_hz))
        min_rms = max(0.0, float(self._settings.whisper_min_rms))
        requested_frame_seconds = max(0.05, min(1.0, float(self._settings.whisper_frame_seconds)))
        requested_frame_samples = max(1, int(round(requested_frame_seconds * sample_rate)))
        frame_samples = _closest_power_of_two(requested_frame_samples, min_value=256, max_value=4096)
        frame_seconds = frame_samples / float(sample_rate)
        if frame_samples != requested_frame_samples:
            print(
                "[eleccia][listen] adjusted frame size for ALSA stability: "
                f"{requested_frame_samples} -> {frame_samples} samples"
            )
        speech_start_seconds = max(frame_seconds, float(self._settings.whisper_speech_start_seconds))
        silence_stop_seconds = max(frame_seconds, float(self._settings.whisper_silence_stop_seconds))
        max_utterance_seconds = max(1.0, float(self._settings.whisper_max_utterance_seconds))
        pre_roll_seconds = max(0.0, float(self._settings.whisper_pre_roll_seconds))

        required_speech_frames = max(1, int(round(speech_start_seconds / frame_seconds)))
        silence_stop_frames = max(1, int(round(silence_stop_seconds / frame_seconds)))
        pre_roll_frames = max(0, int(round(pre_roll_seconds / frame_seconds)))
        max_utterance_samples = int(max_utterance_seconds * sample_rate)

        pre_buffer: deque[object] = deque(maxlen=pre_roll_frames)
        speaking = False
        speech_frames = 0
        silence_frames = 0
        utterance_chunks: list[object] = []
        utterance_samples = 0
        capture_errors = 0

        while not self._stop_event.is_set():
            try:
                with audio_io_lock():
                    frame = sd.rec(
                        frame_samples,
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                        device=self._settings.whisper_input_device_index,
                    )
                    sd.wait()
            except Exception:
                capture_errors += 1
                if capture_errors in {1, 5, 20}:
                    print("[eleccia][listen] capture error (alsa/portaudio); retrying...")
                time.sleep(min(1.0, 0.05 * capture_errors))
                continue
            capture_errors = 0

            if self._stop_event.is_set():
                break

            waveform = np.asarray(frame, dtype=np.float32).reshape(-1)
            if waveform.size == 0:
                continue
            if noise_filter is not None:
                waveform = noise_filter.process(waveform, speaking=speaking)

            frame_rms = float(np.sqrt(np.mean(np.square(waveform))))

            if not speaking:
                if pre_roll_frames > 0:
                    pre_buffer.append(waveform.copy())
                if frame_rms >= min_rms:
                    speech_frames += 1
                else:
                    speech_frames = 0
                if speech_frames < required_speech_frames:
                    continue

                speaking = True
                silence_frames = 0
                utterance_chunks = list(pre_buffer)
                utterance_chunks.append(waveform)
                utterance_samples = int(sum(chunk.size for chunk in utterance_chunks))
                pre_buffer.clear()
                if self._settings.debug_timing:
                    print(
                        "[eleccia][timing] speech_start "
                        f"rms={frame_rms:.4f} threshold={min_rms:.4f}"
                    )
                continue

            utterance_chunks.append(waveform)
            utterance_samples += int(waveform.size)

            if frame_rms < min_rms:
                silence_frames += 1
            else:
                silence_frames = 0

            should_flush = silence_frames >= silence_stop_frames or utterance_samples >= max_utterance_samples
            if not should_flush:
                continue

            audio = np.concatenate(utterance_chunks).astype(np.float32, copy=False)
            flush_reason = "silence" if silence_frames >= silence_stop_frames else "max_utterance"
            if self._settings.debug_timing:
                duration_s = utterance_samples / float(sample_rate)
                print(
                    "[eleccia][timing] speech_stop "
                    f"reason={flush_reason} duration={duration_s:.2f}s samples={utterance_samples}"
                )
            self._transcribe_and_dispatch(
                model=model,
                waveform=audio,
                np=np,
                min_rms=min_rms,
                capture_reason=flush_reason,
            )

            speaking = False
            speech_frames = 0
            silence_frames = 0
            utterance_chunks = []
            utterance_samples = 0

    def _transcribe_and_dispatch(
        self,
        model: object,
        waveform: object,
        np: object,
        min_rms: float,
        capture_reason: str,
    ) -> None:
        if waveform.size == 0:
            return

        rms = float(np.sqrt(np.mean(np.square(waveform))))
        if rms < min_rms:
            return

        started = time.perf_counter()
        if self._settings.debug_timing:
            sample_count = int(getattr(waveform, "size", 0))
            print(
                "[eleccia][timing] stt_transcribe_start "
                f"reason={capture_reason} samples={sample_count} rms={rms:.4f}"
            )
        try:
            segments, _info = model.transcribe(
                waveform,
                beam_size=max(1, int(self._settings.whisper_beam_size)),
                language=self._settings.whisper_language,
                vad_filter=bool(self._settings.whisper_vad_filter),
            )
        except Exception:
            if self._settings.debug_timing:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                print(f"[eleccia][timing] stt_transcribe_error elapsed_ms={elapsed_ms:.1f}")
            return

        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        if self._settings.debug_timing:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            print(
                "[eleccia][timing] stt_transcribe_done "
                f"elapsed_ms={elapsed_ms:.1f} text_len={len(text)}"
            )
        if not text:
            return
        self._dispatch_text(text)

    def _run_openwakeword_whisper(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
            from faster_whisper import WhisperModel
            from openwakeword.model import Model as OpenWakeWordModel
        except Exception as exc:
            print(f"[eleccia][listen] openwakeword_whisper backend unavailable: {exc}")
            return

        wakeword_models = list(self._settings.openwakeword_model_paths)
        framework = self._settings.openwakeword_inference_framework.strip().lower() or "onnx"
        try:
            if wakeword_models:
                oww_model = OpenWakeWordModel(
                    wakeword_models=wakeword_models,
                    inference_framework=framework,
                )
            else:
                oww_model = OpenWakeWordModel(inference_framework=framework)
        except Exception as exc:
            print(f"[eleccia][listen] failed to initialize openWakeWord: {exc}")
            return

        whisper_model = WhisperModel(
            self._settings.whisper_model,
            device=self._settings.whisper_device,
            compute_type=self._settings.whisper_compute_type,
        )

        sample_rate = max(8000, int(self._settings.whisper_sample_rate_hz))
        if sample_rate != 16000:
            print("[eleccia][listen] openWakeWord requires 16kHz input; forcing sample rate to 16000")
            sample_rate = 16000

        chunk_size = max(160, int(self._settings.openwakeword_chunk_size))
        threshold = max(0.0, min(1.0, float(self._settings.openwakeword_threshold)))
        cooldown = max(0.0, float(self._settings.openwakeword_cooldown_seconds))
        command_seconds = max(0.5, float(self._settings.whisper_chunk_seconds))
        command_samples = int(command_seconds * sample_rate)
        last_activation_ts = 0.0

        print(
            "[eleccia][listen] openWakeWord ready "
            f"(models={len(getattr(oww_model, 'models', {})) or 'default'}, "
            f"threshold={threshold:.2f})"
        )

        while not self._stop_event.is_set():
            try:
                with audio_io_lock():
                    frame = sd.rec(
                        chunk_size,
                        samplerate=sample_rate,
                        channels=1,
                        dtype="int16",
                        device=self._settings.whisper_input_device_index,
                    )
                    sd.wait()
            except Exception:
                time.sleep(0.1)
                continue

            if self._stop_event.is_set():
                break

            pcm16 = np.asarray(frame, dtype=np.int16).reshape(-1)
            if pcm16.size == 0:
                continue

            try:
                prediction = oww_model.predict(pcm16)
            except Exception:
                continue

            best_model, best_score = _best_openwakeword_prediction(prediction)
            if best_score < threshold:
                continue

            now_ts = time.time()
            if (now_ts - last_activation_ts) < cooldown:
                continue
            last_activation_ts = now_ts
            label = best_model or "wakeword"
            print(f"[eleccia][listen] wake detected: {label} ({best_score:.3f})")

            try:
                with audio_io_lock():
                    audio = sd.rec(
                        command_samples,
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                        device=self._settings.whisper_input_device_index,
                    )
                    sd.wait()
            except Exception:
                continue

            waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
            if waveform.size == 0:
                continue

            rms = float(np.sqrt(np.mean(np.square(waveform))))
            if rms < max(0.0, float(self._settings.whisper_min_rms)):
                continue

            try:
                segments, _info = whisper_model.transcribe(
                    waveform,
                    beam_size=max(1, int(self._settings.whisper_beam_size)),
                    language=self._settings.whisper_language,
                    vad_filter=bool(self._settings.whisper_vad_filter),
                )
            except Exception:
                continue

            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
            if text:
                self._dispatch_text(text)

    def _dispatch_text(self, text: str) -> None:
        require_wake_word = bool(self._settings.require_wake_word)

        event = CommandEvent(
            text=text,
            normalized_text=_normalize_text(text),
            intent=parse_command_text(
                text,
                wake_word=self._settings.wake_word,
                wake_word_aliases=self._settings.wake_word_aliases,
                wake_word_fuzzy_threshold=self._settings.wake_word_fuzzy_threshold,
                require_wake_word=require_wake_word,
            ),
            ts=time.time(),
        )
        print(
            f"[eleccia][stt] text='{event.text}' normalized='{event.normalized_text}' "
            f"intent={event.intent.name} conf={event.intent.confidence:.3f}"
        )
        if self._settings.debug_timing and event.intent.slots.get("wake_detected") == "true":
            wake_match = event.intent.slots.get("wake_match", "")
            wake_score = event.intent.slots.get("wake_score", "")
            print(
                "[eleccia][timing] wake_detected "
                f"match={wake_match} score={wake_score} intent={event.intent.name}"
            )
        if event.intent.name == "no_wakeword":
            return
        if self._on_command is None:
            return
        try:
            self._on_command(event)
        except Exception:
            return


def parse_command_text(
    text: str,
    wake_word: str = "eleccia",
    wake_word_aliases: tuple[str, ...] = (),
    wake_word_fuzzy_threshold: float = 0.80,
    require_wake_word: bool = False,
) -> CommandIntent:
    normalized = _normalize_text(text)
    wake = _normalize_text(wake_word)
    detected, clean, matched, score = _extract_wakeword_activation(
        text=normalized,
        wake_word=wake,
        aliases=tuple(_normalize_text(alias) for alias in wake_word_aliases),
        threshold=max(0.0, min(1.0, float(wake_word_fuzzy_threshold))),
    )

    if require_wake_word and not detected:
        return CommandIntent(
            name="no_wakeword",
            confidence=0.0,
            slots={"wake_detected": "false"},
        )

    if _contains_phrase(
        clean,
        (
            "enciende la luz",
            "enciende las luces",
            "prende la luz",
            "prende las luces",
            "encender luz",
            "encender luces",
        ),
    ):
        return _intent_with_wake("lights_on", 0.98, detected, matched, score)
    if _contains_phrase(
        clean,
        (
            "apaga la luz",
            "apaga las luces",
            "apagar luz",
            "apagar luces",
        ),
    ):
        return _intent_with_wake("lights_off", 0.98, detected, matched, score)
    if _contains_phrase(
        clean,
        (
            "abre camara",
            "abre la camara",
            "inicia camara",
            "inicia la camara",
            "enciende camara",
            "enciende la camara",
            "activa camara",
            "activa la camara",
        ),
    ):
        return _intent_with_wake("camera_on", 0.95, detected, matched, score)
    if _contains_phrase(
        clean,
        (
            "deten camara",
            "deten la camara",
            "apaga camara",
            "apaga la camara",
            "desactiva camara",
            "desactiva la camara",
            "cierra camara",
            "cierra la camara",
        ),
    ):
        return _intent_with_wake("camera_off", 0.95, detected, matched, score)
    if _contains_phrase(clean, ("estado", "status", "como estas")):
        return _intent_with_wake("status", 0.90, detected, matched, score)
    if _contains_phrase(
        clean,
        (
            "hola",
            "buenos dias",
            "buenas tardes",
            "buenas noches",
            "que tal",
        ),
    ):
        return _intent_with_wake("greeting", 0.90, detected, matched, score)

    if detected:
        return _intent_with_wake("wake", max(0.75, score), detected, matched, score)
    return _intent_with_wake("unknown", 0.20, detected, matched, score)


def _intent_with_wake(
    name: str,
    confidence: float,
    wake_detected: bool,
    wake_match: str,
    wake_score: float,
) -> CommandIntent:
    slots: dict[str, str] = {"wake_detected": "true" if wake_detected else "false"}
    if wake_detected:
        slots["wake_match"] = wake_match
        slots["wake_score"] = f"{wake_score:.3f}"
    return CommandIntent(name=name, confidence=confidence, slots=slots)


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _remove_wake_prefix(text: str, wake_word: str) -> str:
    if not wake_word:
        return text
    patterns = (
        rf"^\s*{re.escape(wake_word)}[\s,:-]+",
        rf"^\s*hola\s+{re.escape(wake_word)}[\s,:-]*",
    )
    out = text
    for pattern in patterns:
        out = re.sub(pattern, "", out).strip()
    return out


def _extract_wakeword_activation(
    text: str,
    wake_word: str,
    aliases: tuple[str, ...],
    threshold: float,
) -> tuple[bool, str, str, float]:
    if not text:
        return False, "", "", 0.0

    candidates: list[str] = []
    if wake_word:
        candidates.append(wake_word)
    for alias in aliases:
        if alias and alias not in candidates:
            candidates.append(alias)
    if not candidates:
        return False, text, "", 0.0

    tokens = re.findall(r"[a-z0-9]+", text)
    if not tokens:
        return False, text, "", 0.0

    best_score = 0.0
    best_candidate = ""
    best_idx = -1

    for idx, token in enumerate(tokens):
        for candidate in candidates:
            score = 1.0 if token == candidate else SequenceMatcher(None, token, candidate).ratio()
            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_idx = idx

    if best_idx < 0 or best_score < threshold:
        return False, text, "", best_score

    remaining_tokens = tokens[:best_idx] + tokens[best_idx + 1 :]
    clean = " ".join(remaining_tokens).strip()
    if clean.startswith("hola "):
        clean = clean[5:].strip()
    return True, clean, best_candidate, best_score


def _normalize_text(text: str) -> str:
    lowered = str(text).strip().lower()
    if not lowered:
        return ""
    decomposed = unicodedata.normalize("NFD", lowered)
    no_accents = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    normalized = unicodedata.normalize("NFC", no_accents)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _best_openwakeword_prediction(prediction: object) -> tuple[str | None, float]:
    if not isinstance(prediction, dict):
        return None, 0.0

    best_model: str | None = None
    best_score = 0.0
    for key, value in prediction.items():
        try:
            score = float(value)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_model = str(key)
    return best_model, best_score


def _closest_power_of_two(value: int, min_value: int = 256, max_value: int = 4096) -> int:
    clamped = max(min_value, min(max_value, int(value)))
    lower = 1
    while (lower * 2) <= clamped:
        lower *= 2
    upper = min(max_value, lower * 2)
    if upper == lower:
        return lower
    if abs(clamped - lower) <= abs(upper - clamped):
        return lower
    return upper


class _AdaptiveNoiseFilter:
    """Lightweight adaptive noise reducer for real-time mic frames."""

    def __init__(self) -> None:
        self._noise_rms = 0.005
        self._noise_beta = 0.97
        self._gate_factor = 1.35
        self._low_gain = 0.18
        self._prev_x = 0.0
        self._prev_y = 0.0
        self._hp_alpha = 0.97

    def process(self, frame: object, speaking: bool) -> object:
        import numpy as np

        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x

        # Remove low-frequency ambient noise (AC hum, table vibrations).
        y = np.empty_like(x)
        prev_x = self._prev_x
        prev_y = self._prev_y
        a = self._hp_alpha
        for idx, sample in enumerate(x):
            out = float(sample) - prev_x + (a * prev_y)
            y[idx] = out
            prev_x = float(sample)
            prev_y = out
        self._prev_x = prev_x
        self._prev_y = prev_y

        rms = float(np.sqrt(np.mean(np.square(y))))
        if not speaking:
            self._noise_rms = (self._noise_beta * self._noise_rms) + ((1.0 - self._noise_beta) * rms)

        threshold = self._noise_rms * self._gate_factor
        if rms < threshold:
            return (y * self._low_gain).astype(np.float32, copy=False)

        return y
