#!/usr/bin/env python3
"""Quick voice backend tester (without camera loop)."""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VoiceBackend:
    kind: str
    engine: Any | None = None


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
        self._ready.wait(timeout=3.0)

    def _run(self) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            if self._rate is not None:
                engine.setProperty("rate", int(self._rate))
            if self._volume is not None:
                engine.setProperty("volume", min(1.0, max(0.0, float(self._volume))))
            if self._voice_id:
                engine.setProperty("voice", self._voice_id)
            elif self._voice_lang:
                picked = _select_pyttsx3_voice(engine, self._voice_lang)
                if picked is not None:
                    engine.setProperty("voice", picked)
        except Exception:
            self._failed = True
            self._ready.set()
            return

        self._ready.set()
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                continue

        try:
            engine.stop()
        except Exception:
            pass

    def enqueue(self, text: str) -> bool:
        if self._failed:
            return False
        self._queue.put(text)
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
        except Exception as exc:
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
            text = self._queue.get()
            if text is None:
                break
            tmp_path = ""
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                model.tts_to_file(text, speaker_value, tmp_path, speed=float(self._speed))
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

    def enqueue(self, text: str) -> bool:
        if self._failed:
            return False
        self._queue.put(text)
        return True

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=1.0)


def _read_env_file(path: str) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _strip_optional_quotes(value.strip())
    return values


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _env_lookup(key: str, file_values: dict[str, str]) -> str | None:
    prefixed = f"FACIAL_{key}"
    if key in os.environ:
        return os.environ[key]
    if prefixed in os.environ:
        return os.environ[prefixed]
    if key in file_values:
        return file_values[key]
    if prefixed in file_values:
        return file_values[prefixed]
    return None


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
    # Fallback for single-speaker checkpoints where map is not exposed.
    return 0


def _extract_melo_speaker_map(model: Any) -> dict[str, Any] | None:
    # Compatibility across MeloTTS versions where hps/data can be object or dict.
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

    subprocess.run([*player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def _play_audio(
    audio: Any,
    sample_rate: int,
    sounddevice: Any | None,
    player: list[str] | None,
) -> None:
    import numpy as np

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
        subprocess.run([*player, tmp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


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


def _select_pyttsx3_voice(engine: Any, language_hint: str) -> str | None:
    hint = language_hint.strip().lower().replace("-", "_")
    if not hint:
        return None
    try:
        voices = engine.getProperty("voices")
    except Exception:
        return None
    for voice in voices:
        voice_id = getattr(voice, "id", None)
        voice_name = getattr(voice, "name", None)
        items = []
        if isinstance(voice_id, str):
            items.append(voice_id.lower())
        if isinstance(voice_name, str):
            items.append(voice_name.lower())
        for item in items:
            normalized = item.replace("-", "_")
            if hint in normalized:
                return voice_id if isinstance(voice_id, str) else None
            if hint.startswith("es") and ("spanish" in normalized or "espanol" in normalized):
                return voice_id if isinstance(voice_id, str) else None
    return None


def _list_pyttsx3_voices() -> int:
    try:
        import pyttsx3
    except Exception as exc:
        print(f"pyttsx3 not available: {exc}")
        return 1

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    if not voices:
        print("No voices found.")
        return 1

    for idx, voice in enumerate(voices, start=1):
        voice_id = getattr(voice, "id", "")
        voice_name = getattr(voice, "name", "")
        langs = getattr(voice, "languages", [])
        print(f"{idx:02d}. id={voice_id} | name={voice_name} | languages={langs}")
    return 0


def _build_backend(args: argparse.Namespace) -> VoiceBackend | None:
    selected = args.backend
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
            return None

    if selected == "melotts":
        speaker = _MeloTTSSpeaker(
            language=args.melo_language,
            speaker=args.melo_speaker,
            speed=args.melo_speed,
            device=args.melo_device,
        )
        if speaker._failed:
            if speaker._last_error:
                print(f"MeloTTS error: {speaker._last_error}")
            return None
        return VoiceBackend(kind="melotts", engine=speaker)

    if selected == "pyttsx3":
        speaker = _Pyttsx3Speaker(
            rate=args.voice_rate,
            volume=args.voice_volume,
            voice_id=args.voice_id,
            voice_lang=args.voice_lang,
        )
        if speaker._failed:
            return None
        return VoiceBackend(kind="pyttsx3", engine=speaker)

    if selected == "spd-say":
        if shutil.which("spd-say"):
            return VoiceBackend(kind="spd-say", engine={"lang": args.voice_lang})
        return None

    if selected == "espeak":
        if shutil.which("espeak"):
            return VoiceBackend(kind="espeak", engine={"lang": args.voice_lang})
        return None

    return None


def _speak_once(backend: VoiceBackend, text: str) -> bool:
    try:
        if backend.kind == "melotts":
            if not isinstance(backend.engine, _MeloTTSSpeaker):
                return False
            return backend.engine.enqueue(text)
        if backend.kind == "pyttsx3":
            if not isinstance(backend.engine, _Pyttsx3Speaker):
                return False
            return backend.engine.enqueue(text)
        if backend.kind == "spd-say":
            lang = backend.engine.get("lang") if isinstance(backend.engine, dict) else None
            cmd = ["spd-say"]
            if isinstance(lang, str) and lang.strip():
                cmd.extend(["-l", lang.strip()])
            cmd.append(text)
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        if backend.kind == "espeak":
            lang = backend.engine.get("lang") if isinstance(backend.engine, dict) else None
            cmd = ["espeak"]
            if isinstance(lang, str) and lang.strip():
                cmd.extend(["-v", lang.strip()])
            cmd.append(text)
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    except Exception:
        return False
    return False


def _close_backend(backend: VoiceBackend | None) -> None:
    if backend is None:
        return
    if backend.kind == "melotts" and isinstance(backend.engine, _MeloTTSSpeaker):
        backend.engine.close()
        return
    if backend.kind == "pyttsx3" and isinstance(backend.engine, _Pyttsx3Speaker):
        backend.engine.close()
        return
def _default_texts(language: str | None) -> list[str]:
    if language and language.lower().startswith("es"):
        return [
            "Hola, esta es una prueba de voz.",
            "Bienvenido al laboratorio de reconocimiento facial.",
            "Estoy ajustando el tono para que suene mas natural.",
        ]
    return [
        "Hello, this is a voice test.",
        "Welcome to the facial recognition lab.",
        "I am checking audio quality and pronunciation.",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice backend tester")
    parser.add_argument("--backend", choices=("auto", "melotts", "pyttsx3", "spd-say", "espeak"), default="auto")
    parser.add_argument("--voice-lang", type=str, default=None)
    parser.add_argument("--voice-rate", type=int, default=None)
    parser.add_argument("--voice-volume", type=float, default=None)
    parser.add_argument("--voice-id", type=str, default=None)
    parser.add_argument("--melo-language", type=str, default=None)
    parser.add_argument("--melo-speaker", type=str, default=None)
    parser.add_argument("--melo-speed", type=float, default=None)
    parser.add_argument("--melo-device", type=str, default=None)
    parser.add_argument("--text", action="append", default=None, help="Repeatable. Add one sentence per --text")
    parser.add_argument("--delay-seconds", type=float, default=1.2)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--list-pyttsx3-voices", action="store_true")
    parser.add_argument("--env-file", type=str, default=".env")
    args = parser.parse_args()
    _apply_env_defaults(args)
    return args


def _apply_env_defaults(args: argparse.Namespace) -> None:
    file_values = _read_env_file(args.env_file)
    args.backend = args.backend or (_env_lookup("DEMO_VOICE_BACKEND", file_values) or "auto")
    if args.voice_lang is None:
        args.voice_lang = _env_lookup("DEMO_VOICE_LANG", file_values)
    if args.voice_rate is None:
        raw = _env_lookup("DEMO_VOICE_RATE", file_values)
        if raw is not None:
            args.voice_rate = int(raw.strip())
    if args.voice_volume is None:
        raw = _env_lookup("DEMO_VOICE_VOLUME", file_values)
        if raw is not None:
            args.voice_volume = float(raw.strip())
    if args.voice_id is None:
        args.voice_id = _env_lookup("DEMO_VOICE_ID", file_values)
    if args.melo_language is None:
        raw = _env_lookup("DEMO_MELO_LANGUAGE", file_values) or args.voice_lang or "ES"
        args.melo_language = _normalize_melo_language(raw)
    if args.melo_speaker is None:
        args.melo_speaker = _env_lookup("DEMO_MELO_SPEAKER", file_values)
    if args.melo_speed is None:
        raw = _env_lookup("DEMO_MELO_SPEED", file_values)
        if raw is not None:
            args.melo_speed = max(0.1, float(raw.strip()))
        elif args.voice_rate is not None:
            args.melo_speed = max(0.6, min(1.6, float(args.voice_rate) / 150.0))
        else:
            args.melo_speed = 1.0
    if args.melo_device is None:
        args.melo_device = _env_lookup("DEMO_MELO_DEVICE", file_values) or "auto"


def _normalize_melo_language(raw: str) -> str:
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


def main() -> int:
    args = parse_args()
    if args.list_pyttsx3_voices:
        return _list_pyttsx3_voices()

    backend = _build_backend(args)
    if backend is None:
        print("Could not initialize voice backend.")
        print("Try: --backend pyttsx3 or --backend espeak, and verify dependencies.")
        return 1

    print(f"Voice backend ready: {backend.kind}")
    try:
        if args.interactive:
            print("Interactive mode. Write text and press Enter. Empty line exits.")
            while True:
                text = input("> ").strip()
                if not text:
                    break
                ok = _speak_once(backend, text)
                if not ok:
                    print("Failed to enqueue/speak text.")
                time.sleep(max(0.0, args.delay_seconds))
            return 0

        texts = args.text or _default_texts(args.voice_lang)
        for idx, text in enumerate(texts, start=1):
            print(f"[{idx}/{len(texts)}] {text}")
            ok = _speak_once(backend, text)
            if not ok:
                print("Failed to enqueue/speak text.")
            time.sleep(max(0.0, args.delay_seconds))
        return 0
    finally:
        _close_backend(backend)


if __name__ == "__main__":
    raise SystemExit(main())
