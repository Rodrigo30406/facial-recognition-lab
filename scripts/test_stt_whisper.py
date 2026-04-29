#!/usr/bin/env python3
"""Standalone STT demo (voice to text) for Eleccia listen module."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

# Allow running from repo root without forcing PYTHONPATH=src.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from eleccia_listen import CommandEvent, ElecciaListenService, ListenSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whisper STT demo (mic -> text)")
    parser.add_argument(
        "--backend",
        type=str,
        default="whisper",
        choices=["whisper", "stdin", "openwakeword_whisper"],
    )
    parser.add_argument("--wake-word", type=str, default="eleccia")
    parser.add_argument(
        "--wake-word-aliases",
        type=str,
        default="elexia,eleksia,elecia",
        help="Comma-separated aliases for fuzzy wake word matching",
    )
    parser.add_argument("--wake-word-fuzzy-threshold", type=float, default=0.80)
    parser.add_argument("--wake-command-window-seconds", type=float, default=6.0)
    parser.add_argument("--require-wake-word", action="store_true")
    parser.add_argument("--no-require-wake-word", dest="require_wake_word", action="store_false")
    parser.set_defaults(require_wake_word=True)
    parser.add_argument("--stdin-prompt", type=str, default="eleccia> ")
    parser.add_argument("--whisper-model", type=str, default="large-v3")
    parser.add_argument("--whisper-device", type=str, default="cuda")
    parser.add_argument("--whisper-compute-type", type=str, default="float16")
    parser.add_argument("--whisper-language", type=str, default="es")
    parser.add_argument("--whisper-beam-size", type=int, default=5)
    parser.add_argument("--whisper-vad-filter", action="store_true")
    parser.add_argument("--no-whisper-vad-filter", dest="whisper_vad_filter", action="store_false")
    parser.set_defaults(whisper_vad_filter=True)
    parser.add_argument("--whisper-chunk-seconds", type=float, default=4.0)
    parser.add_argument("--whisper-sample-rate-hz", type=int, default=16000)
    parser.add_argument("--whisper-input-device-index", type=int, default=None)
    parser.add_argument("--whisper-min-rms", type=float, default=0.003)
    parser.add_argument(
        "--openwakeword-model-paths",
        type=str,
        default="",
        help="Comma-separated .onnx/.tflite model paths for openWakeWord custom models",
    )
    parser.add_argument("--openwakeword-inference-framework", type=str, default="onnx", choices=["onnx", "tflite"])
    parser.add_argument("--openwakeword-threshold", type=float, default=0.5)
    parser.add_argument("--openwakeword-chunk-size", type=int, default=1280)
    parser.add_argument("--openwakeword-cooldown-seconds", type=float, default=1.5)
    parser.add_argument("--debug-timing", action="store_true", help="Print STT timing checkpoints")
    parser.add_argument("--noise-filter", action="store_true", help="Enable adaptive noise filter")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    settings = ListenSettings(
        enabled=True,
        backend=args.backend,
        stdin_prompt=args.stdin_prompt,
        wake_word=args.wake_word,
        wake_word_aliases=tuple(part.strip() for part in str(args.wake_word_aliases).split(",") if part.strip()),
        wake_word_fuzzy_threshold=max(0.0, min(1.0, float(args.wake_word_fuzzy_threshold))),
        wake_command_window_seconds=max(0.5, float(args.wake_command_window_seconds)),
        require_wake_word=bool(args.require_wake_word),
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        whisper_compute_type=args.whisper_compute_type,
        whisper_language=args.whisper_language,
        whisper_beam_size=max(1, int(args.whisper_beam_size)),
        whisper_vad_filter=bool(args.whisper_vad_filter),
        whisper_chunk_seconds=max(0.5, float(args.whisper_chunk_seconds)),
        whisper_sample_rate_hz=max(8000, int(args.whisper_sample_rate_hz)),
        whisper_input_device_index=args.whisper_input_device_index,
        whisper_min_rms=max(0.0, float(args.whisper_min_rms)),
        openwakeword_model_paths=tuple(
            part.strip() for part in str(args.openwakeword_model_paths).split(",") if part.strip()
        ),
        openwakeword_inference_framework=str(args.openwakeword_inference_framework),
        openwakeword_threshold=max(0.0, min(1.0, float(args.openwakeword_threshold))),
        openwakeword_chunk_size=max(160, int(args.openwakeword_chunk_size)),
        openwakeword_cooldown_seconds=max(0.0, float(args.openwakeword_cooldown_seconds)),
        debug_timing=bool(args.debug_timing),
        noise_filter_enabled=bool(args.noise_filter),
    )

    def on_command(event: CommandEvent) -> None:
        print(f"[text] {event.text}")
        print(f"[norm] {event.normalized_text}")
        print(f"[intent] {event.intent.name} ({event.intent.confidence:.2f})")
        print("-")

    service = ElecciaListenService(settings=settings, on_command=on_command)

    should_exit = False

    def _on_signal(_signum, _frame) -> None:
        nonlocal should_exit
        should_exit = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    service.start()
    print("[stt] running. Press Ctrl+C to stop.")
    print(f"[stt] backend={settings.backend} model={settings.whisper_model} device={settings.whisper_device}")

    try:
        while not should_exit:
            time.sleep(0.25)
    finally:
        service.stop()
        print("[stt] stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
