#!/usr/bin/env python3
"""Run a local webcam demo for facial recognition.

Usage:
  python3 scripts/run_camera_demo.py --camera-index 0 --camera-id cam-01

Controls:
  q: quit
  e: enroll current frame (requires --enroll-person-id)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Allow running from repo root without forcing PYTHONPATH=src.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from eleccia_vision.application.enrollment import InvalidImageError, PersonNotFoundError
from eleccia_vision.application.quality_gate import (
    AngleBucket,
    FaceObservation,
    QualityGateThresholds,
    build_angle_plan,
    bucket_instruction,
    evaluate_quality_gate,
)
from eleccia_vision.bootstrap import build_services
from eleccia_vision.domain.entities import RecognitionResult
from eleccia_vision.infrastructure.insightface_encoder import DetectedFace
from eleccia_voice import VoiceAssistant, build_voice_settings_from_args


@dataclass
class DisplayState:
    result: RecognitionResult | None = None
    latency_ms: float | None = None
    message: str = ""
    message_until_ts: float = 0.0
    landmarks: list[tuple[int, int]] = field(default_factory=list)
    landmarks_warning_shown: bool = False
    gate_status: str | None = None
    gate_reason: str = ""
    gate_progress: str = ""
    gate_pose: str = ""
    gate_current_bucket: AngleBucket | None = None
    gate_target_bucket: AngleBucket | None = None


@dataclass
class GuidedEnrollState:
    target_samples: int
    hold_frames: int
    cooldown_ms: int
    plan_by_bucket: dict[AngleBucket, int]
    captured_by_bucket: dict[AngleBucket, int]
    consecutive_green: int = 0
    captured_total: int = 0
    last_capture_ts_ms: float = 0.0
    completed: bool = False


@dataclass(frozen=True)
class GuidedPreset:
    target_samples: int
    hold_frames: int
    cooldown_ms: int
    landmarks_max_points: int
    landmarks_every: int
    min_det_score: float
    min_face_ratio: float
    min_sharpness: float
    min_brightness: float
    max_brightness: float
    max_abs_yaw: float
    max_abs_pitch: float
    max_abs_roll: float


DEFAULT_GUIDED_PRESET = GuidedPreset(
    target_samples=12,
    hold_frames=3,
    cooldown_ms=900,
    landmarks_max_points=20,
    landmarks_every=2,
    min_det_score=0.60,
    min_face_ratio=0.08,
    min_sharpness=90.0,
    min_brightness=50.0,
    max_brightness=210.0,
    max_abs_yaw=55.0,
    max_abs_pitch=40.0,
    max_abs_roll=40.0,
)

GUIDED_PRESETS: dict[str, GuidedPreset] = {
    "fast": GuidedPreset(
        target_samples=10,
        hold_frames=2,
        cooldown_ms=300,
        landmarks_max_points=50,
        landmarks_every=1,
        min_det_score=0.55,
        min_face_ratio=0.08,
        min_sharpness=90.0,
        min_brightness=45.0,
        max_brightness=220.0,
        max_abs_yaw=55.0,
        max_abs_pitch=40.0,
        max_abs_roll=40.0,
    ),
    "strict": GuidedPreset(
        target_samples=20,
        hold_frames=4,
        cooldown_ms=700,
        landmarks_max_points=50,
        landmarks_every=1,
        min_det_score=0.70,
        min_face_ratio=0.12,
        min_sharpness=140.0,
        min_brightness=60.0,
        max_brightness=190.0,
        max_abs_yaw=50.0,
        max_abs_pitch=35.0,
        max_abs_roll=30.0,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam demo for facial recognition")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--camera-id", type=str, default="cam-01", help="Camera identifier")
    parser.add_argument(
        "--recognize-every",
        type=int,
        default=5,
        help="Run recognition every N frames",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="Facial Recognition Demo",
        help="Display window title",
    )
    parser.add_argument(
        "--enroll-person-id",
        type=str,
        default=None,
        help="If provided, press 'e' to enroll current frame for this person",
    )
    parser.add_argument(
        "--show-landmarks",
        action="store_true",
        help="Draw facial landmarks overlay (requires InsightFace encoder)",
    )
    parser.add_argument(
        "--landmarks-max-points",
        type=int,
        default=None,
        help="Maximum number of landmarks to draw (minimum effective value: 10)",
    )
    parser.add_argument(
        "--landmarks-every",
        type=int,
        default=None,
        help="Update landmarks every N frames",
    )
    parser.add_argument(
        "--guided-enroll",
        action="store_true",
        help="Enable quality-gated guided enrollment with angle coverage",
    )
    parser.add_argument(
        "--guided-target-samples",
        type=int,
        default=None,
        help="Target number of auto-captured enrollment samples",
    )
    parser.add_argument(
        "--guided-hold-frames",
        type=int,
        default=None,
        help="Green frames required before auto-capturing",
    )
    parser.add_argument(
        "--guided-cooldown-ms",
        type=int,
        default=None,
        help="Cooldown between automatic captures in milliseconds",
    )
    parser.add_argument(
        "--guided-preset",
        choices=tuple(sorted(GUIDED_PRESETS)),
        default=None,
        help="Apply a predefined guided enrollment profile",
    )
    parser.add_argument("--guided-min-det-score", type=float, default=None)
    parser.add_argument("--guided-min-face-ratio", type=float, default=None)
    parser.add_argument("--guided-min-sharpness", type=float, default=None)
    parser.add_argument("--guided-min-brightness", type=float, default=None)
    parser.add_argument("--guided-max-brightness", type=float, default=None)
    parser.add_argument("--guided-max-abs-yaw", type=float, default=None)
    parser.add_argument("--guided-max-abs-pitch", type=float, default=None)
    parser.add_argument("--guided-max-abs-roll", type=float, default=None)
    parser.add_argument(
        "--voice-greet",
        action="store_true",
        help="Enable voice greeting for first known-person detection",
    )
    parser.add_argument(
        "--voice-backend",
        choices=("auto", "melotts", "chattts", "pyttsx3", "spd-say", "espeak"),
        default="auto",
        help="TTS backend to use when --voice-greet is enabled",
    )
    parser.add_argument(
        "--voice-template",
        type=str,
        default="Hola {name}",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--voice-reentry-delay-seconds",
        type=float,
        default=8.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--voice-absence-seconds",
        type=float,
        default=1.2,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--voice-min-face-ratio",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--voice-rate",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--voice-volume",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--voice-id",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--voice-lang",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--melo-language",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--melo-speaker",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--melo-speed",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--melo-device",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    argv = sys.argv[1:]
    args = parser.parse_args()
    _apply_demo_env_defaults(args, argv)
    _apply_guided_preset(args)
    return args


def _apply_demo_env_defaults(args: argparse.Namespace, argv: list[str]) -> None:
    file_values = _read_env_file()

    _apply_env_value(
        args=args,
        attr="camera_index",
        env_key="DEMO_CAMERA_INDEX",
        parser=_to_int,
        argv=argv,
        flag="--camera-index",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="camera_id",
        env_key="DEMO_CAMERA_ID",
        parser=_to_str,
        argv=argv,
        flag="--camera-id",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="recognize_every",
        env_key="DEMO_RECOGNIZE_EVERY",
        parser=_to_int,
        argv=argv,
        flag="--recognize-every",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="window_name",
        env_key="DEMO_WINDOW_NAME",
        parser=_to_str,
        argv=argv,
        flag="--window-name",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="enroll_person_id",
        env_key="DEMO_ENROLL_PERSON_ID",
        parser=_to_optional_str,
        argv=argv,
        flag="--enroll-person-id",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="show_landmarks",
        env_key="DEMO_SHOW_LANDMARKS",
        parser=_to_bool,
        argv=argv,
        flag="--show-landmarks",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="landmarks_max_points",
        env_key="DEMO_LANDMARKS_MAX_POINTS",
        parser=_to_int,
        argv=argv,
        flag="--landmarks-max-points",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="landmarks_every",
        env_key="DEMO_LANDMARKS_EVERY",
        parser=_to_int,
        argv=argv,
        flag="--landmarks-every",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="guided_enroll",
        env_key="DEMO_GUIDED_ENROLL",
        parser=_to_bool,
        argv=argv,
        flag="--guided-enroll",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="guided_target_samples",
        env_key="DEMO_GUIDED_TARGET_SAMPLES",
        parser=_to_int,
        argv=argv,
        flag="--guided-target-samples",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="guided_hold_frames",
        env_key="DEMO_GUIDED_HOLD_FRAMES",
        parser=_to_int,
        argv=argv,
        flag="--guided-hold-frames",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="guided_cooldown_ms",
        env_key="DEMO_GUIDED_COOLDOWN_MS",
        parser=_to_int,
        argv=argv,
        flag="--guided-cooldown-ms",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="guided_preset",
        env_key="DEMO_GUIDED_PRESET",
        parser=_to_guided_preset,
        argv=argv,
        flag="--guided-preset",
        file_values=file_values,
    )

    _apply_env_value(
        args=args,
        attr="voice_greet",
        env_key="DEMO_VOICE_GREET",
        parser=_to_bool,
        argv=argv,
        flag="--voice-greet",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_backend",
        env_key="DEMO_VOICE_BACKEND",
        parser=_to_voice_backend,
        argv=argv,
        flag="--voice-backend",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_template",
        env_key="DEMO_VOICE_TEMPLATE",
        parser=_to_str,
        argv=argv,
        flag="--voice-template",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_reentry_delay_seconds",
        env_key="DEMO_VOICE_REENTRY_DELAY_SECONDS",
        parser=_to_float,
        argv=argv,
        flag="--voice-reentry-delay-seconds",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_absence_seconds",
        env_key="DEMO_VOICE_ABSENCE_SECONDS",
        parser=_to_float,
        argv=argv,
        flag="--voice-absence-seconds",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_min_face_ratio",
        env_key="DEMO_VOICE_MIN_FACE_RATIO",
        parser=_to_float,
        argv=argv,
        flag="--voice-min-face-ratio",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_rate",
        env_key="DEMO_VOICE_RATE",
        parser=_to_int,
        argv=argv,
        flag="--voice-rate",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_volume",
        env_key="DEMO_VOICE_VOLUME",
        parser=_to_float,
        argv=argv,
        flag="--voice-volume",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_id",
        env_key="DEMO_VOICE_ID",
        parser=_to_optional_str,
        argv=argv,
        flag="--voice-id",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_lang",
        env_key="DEMO_VOICE_LANG",
        parser=_to_optional_str,
        argv=argv,
        flag="--voice-lang",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="melo_language",
        env_key="DEMO_MELO_LANGUAGE",
        parser=_to_optional_str,
        argv=argv,
        flag="--melo-language",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="melo_speaker",
        env_key="DEMO_MELO_SPEAKER",
        parser=_to_optional_str,
        argv=argv,
        flag="--melo-speaker",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="melo_speed",
        env_key="DEMO_MELO_SPEED",
        parser=_to_float,
        argv=argv,
        flag="--melo-speed",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="melo_device",
        env_key="DEMO_MELO_DEVICE",
        parser=_to_optional_str,
        argv=argv,
        flag="--melo-device",
        file_values=file_values,
    )


def _apply_env_value(
    args: argparse.Namespace,
    attr: str,
    env_key: str,
    parser,
    argv: list[str],
    flag: str,
    file_values: dict[str, str],
) -> None:
    if _flag_present(argv, flag):
        return
    raw = _env_lookup(env_key, file_values)
    if raw is None:
        return
    setattr(args, attr, parser(raw))


def _flag_present(argv: list[str], flag: str) -> bool:
    for token in argv:
        if token == flag or token.startswith(f"{flag}="):
            return True
    return False


def _read_env_file() -> dict[str, str]:
    env_file = os.getenv("FACIAL_ENV_FILE", ".env")
    path = Path(env_file)
    if not path.exists() or not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
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


def _to_int(raw: str) -> int:
    return int(raw.strip())


def _to_float(raw: str) -> float:
    return float(raw.strip())


def _to_str(raw: str) -> str:
    return raw.strip()


def _to_optional_str(raw: str) -> str | None:
    value = raw.strip()
    if not value:
        return None
    return value


def _to_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "Boolean value must be one of: 1/0, true/false, yes/no, on/off"
    )


def _to_guided_preset(raw: str) -> str:
    value = raw.strip().lower()
    if value not in GUIDED_PRESETS:
        allowed = ", ".join(sorted(GUIDED_PRESETS))
        raise ValueError(f"DEMO_GUIDED_PRESET must be one of: {allowed}")
    return value


def _to_voice_backend(raw: str) -> str:
    value = raw.strip().lower()
    allowed = {"auto", "melotts", "chattts", "pyttsx3", "spd-say", "espeak"}
    if value not in allowed:
        raise ValueError(
            "DEMO_VOICE_BACKEND must be one of: auto, melotts, chattts, pyttsx3, spd-say, espeak"
        )
    return value


def _apply_guided_preset(args: argparse.Namespace) -> None:
    base = DEFAULT_GUIDED_PRESET
    selected = GUIDED_PRESETS.get(args.guided_preset, base)

    args.guided_target_samples = _pick_arg(args.guided_target_samples, selected.target_samples)
    args.guided_hold_frames = _pick_arg(args.guided_hold_frames, selected.hold_frames)
    args.guided_cooldown_ms = _pick_arg(args.guided_cooldown_ms, selected.cooldown_ms)
    args.landmarks_max_points = _pick_arg(args.landmarks_max_points, selected.landmarks_max_points)
    args.landmarks_every = _pick_arg(args.landmarks_every, selected.landmarks_every)
    args.guided_min_det_score = _pick_arg(args.guided_min_det_score, selected.min_det_score)
    args.guided_min_face_ratio = _pick_arg(args.guided_min_face_ratio, selected.min_face_ratio)
    args.guided_min_sharpness = _pick_arg(args.guided_min_sharpness, selected.min_sharpness)
    args.guided_min_brightness = _pick_arg(args.guided_min_brightness, selected.min_brightness)
    args.guided_max_brightness = _pick_arg(args.guided_max_brightness, selected.max_brightness)
    args.guided_max_abs_yaw = _pick_arg(args.guided_max_abs_yaw, selected.max_abs_yaw)
    args.guided_max_abs_pitch = _pick_arg(args.guided_max_abs_pitch, selected.max_abs_pitch)
    args.guided_max_abs_roll = _pick_arg(args.guided_max_abs_roll, selected.max_abs_roll)


def _pick_arg(current, fallback):
    if current is None:
        return fallback
    return current


def main() -> None:
    args = parse_args()
    services = build_services()
    guided_state = _build_guided_enroll_state(args)
    gate_thresholds = _build_gate_thresholds(args)
    voice_assistant = VoiceAssistant(build_voice_settings_from_args(args))

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera_index}")

    print("Camera demo started")
    print("- Press 'q' to quit")
    if args.enroll_person_id:
        print(f"- Press 'e' to enroll current frame as '{args.enroll_person_id}'")
    if args.show_landmarks:
        print("- Landmark overlay enabled")
    if args.guided_enroll:
        print(f"- Guided enrollment enabled ({args.guided_target_samples} samples target)")
        if args.guided_preset:
            print(f"- Guided preset: {args.guided_preset}")
            print(
                f"- Guided landmarks defaults: points={args.landmarks_max_points} every={args.landmarks_every}"
            )
        if not args.enroll_person_id:
            print("- Warning: guided mode needs --enroll-person-id to auto-capture")
    if args.voice_greet:
        if voice_assistant.backend_kind is not None:
            print(f"- Voice greet enabled ({voice_assistant.backend_kind})")
        else:
            print("- Voice greet enabled but no TTS backend found (melotts/chattts/pyttsx3/spd-say/espeak)")
            if voice_assistant.backend_error:
                print(f"- Voice backend detail: {voice_assistant.backend_error}")

    state = DisplayState()
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                state.message = "Failed to read frame"
                state.message_until_ts = time.time() + 2.0
                continue

            frame_idx += 1
            if frame_idx % max(1, args.recognize_every) == 0:
                _run_recognition(
                    frame=frame,
                    services=services,
                    state=state,
                    camera_id=args.camera_id,
                    voice_assistant=voice_assistant,
                )
            if args.guided_enroll:
                _guided_enroll_step(
                    frame=frame,
                    services=services,
                    display_state=state,
                    guided_state=guided_state,
                    thresholds=gate_thresholds,
                    args=args,
                )
            elif args.show_landmarks and frame_idx % max(1, args.landmarks_every) == 0:
                _update_landmarks(frame, services, state, args.landmarks_max_points)

            _draw_overlay(frame, state)
            cv2.imshow(args.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("e") and args.enroll_person_id:
                _enroll_current_frame(frame, services, state, args.enroll_person_id, args.camera_id)
    finally:
        voice_assistant.close()
        cap.release()
        cv2.destroyAllWindows()


def _run_recognition(
    frame,
    services,
    state: DisplayState,
    camera_id: str,
    voice_assistant: VoiceAssistant,
) -> None:
    payload = _frame_to_jpeg_bytes(frame)
    if payload is None:
        state.message = "Failed to encode frame"
        state.message_until_ts = time.time() + 2.0
        return

    t0 = time.perf_counter()
    raw = services.recognition_service.recognize(payload)
    result = services.recognition_consistency_service.stabilize(raw, stream_id=camera_id)
    elapsed = (time.perf_counter() - t0) * 1000.0

    services.recognition_event_service.record_from_result(
        result=result,
        camera_id=camera_id,
    )

    state.result = result
    state.latency_ms = elapsed
    face_ratio, pose_yaw, pose_pitch = _estimate_voice_face_observation(frame=frame, services=services)
    voice_message = voice_assistant.on_recognition(
        result=result,
        resolve_person=lambda person_id: _resolve_person_metadata(services, person_id),
        face_ratio=face_ratio,
        pose_yaw=pose_yaw,
        pose_pitch=pose_pitch,
    )
    if voice_message:
        state.message = voice_message
        if voice_message.startswith("Saludo:"):
            state.message_until_ts = time.time() + 2.0
        else:
            state.message_until_ts = time.time() + 3.0


def _resolve_person_metadata(services, person_id: str) -> tuple[str, str | None]:
    name = person_id
    sex: str | None = None

    try:
        person = services.person_service.get_person(person_id)
    except Exception:
        return name, sex

    if person is not None and isinstance(person.full_name, str) and person.full_name.strip():
        name = person.full_name.strip()
    if person is not None and person.sex is not None:
        raw = str(person.sex).strip()
        sex = raw if raw else None
    return name, sex


def _estimate_voice_face_observation(frame, services) -> tuple[float | None, float | None, float | None]:
    encoder = getattr(services.recognition_service, "_encoder", None)
    analyze = getattr(encoder, "analyze_face", None)
    if not callable(analyze):
        return None, None, None

    try:
        detected = analyze(frame, max_points=10)
    except Exception:
        return None, None, None
    if detected is None:
        return None, None, None

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return None, None, None

    x1, y1, x2, y2 = detected.bbox
    face_w = max(0.0, float(x2) - float(x1))
    face_h = max(0.0, float(y2) - float(y1))
    frame_area = float(h * w)
    if frame_area <= 0.0:
        return None, None, None

    ratio = (face_w * face_h) / frame_area
    face_ratio = max(0.0, min(1.0, float(ratio)))
    return face_ratio, detected.yaw, detected.pitch


def _enroll_current_frame(
    frame,
    services,
    state: DisplayState,
    person_id: str,
    camera_id: str,
    capture_type: str = "operational",
) -> bool:
    payload = _frame_to_jpeg_bytes(frame)
    if payload is None:
        state.message = "Failed to encode frame for enrollment"
        state.message_until_ts = time.time() + 2.5
        return False

    try:
        sample = services.enrollment_service.enroll_image(
            person_id=person_id,
            image_bytes=payload,
            capture_type=capture_type,
            camera_id=camera_id,
        )
        state.message = f"Enrolled {person_id} (q={sample.quality_score:.2f})"
        state.message_until_ts = time.time() + 2.5
        return True
    except PersonNotFoundError:
        state.message = f"Person '{person_id}' not found"
    except InvalidImageError:
        state.message = "Invalid image for enrollment"

    state.message_until_ts = time.time() + 2.5
    return False


def _frame_to_jpeg_bytes(frame) -> bytes | None:
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return encoded.tobytes()


def _build_guided_enroll_state(args: argparse.Namespace) -> GuidedEnrollState:
    target = max(1, int(args.guided_target_samples))
    plan = build_angle_plan(target)
    captured: dict[AngleBucket, int] = {bucket: 0 for bucket in plan}
    return GuidedEnrollState(
        target_samples=target,
        hold_frames=max(1, int(args.guided_hold_frames)),
        cooldown_ms=max(0, int(args.guided_cooldown_ms)),
        plan_by_bucket=plan,
        captured_by_bucket=captured,
    )


def _build_gate_thresholds(args: argparse.Namespace) -> QualityGateThresholds:
    return QualityGateThresholds(
        min_det_score=float(args.guided_min_det_score),
        min_face_ratio=float(args.guided_min_face_ratio),
        min_sharpness=float(args.guided_min_sharpness),
        min_brightness=float(args.guided_min_brightness),
        max_brightness=float(args.guided_max_brightness),
        max_abs_yaw=float(args.guided_max_abs_yaw),
        max_abs_pitch=float(args.guided_max_abs_pitch),
        max_abs_roll=float(args.guided_max_abs_roll),
    )


def _guided_enroll_step(
    frame,
    services,
    display_state: DisplayState,
    guided_state: GuidedEnrollState,
    thresholds: QualityGateThresholds,
    args: argparse.Namespace,
) -> None:
    detected = _extract_face_observation(
        frame=frame,
        services=services,
        display_state=display_state,
        max_points=max(20, args.landmarks_max_points),
    )
    display_state.landmarks = detected.landmarks if (detected and args.show_landmarks) else []

    observation = _to_gate_observation(detected)
    assessment = evaluate_quality_gate(
        frame=frame,
        observation=observation,
        thresholds=thresholds,
        captured_by_bucket=guided_state.captured_by_bucket,
        plan_by_bucket=guided_state.plan_by_bucket,
    )

    if args.enroll_person_id is None:
        assessment = assessment.__class__(
            status="red",
            reason="Usa --enroll-person-id para capturar",
            current_bucket=assessment.current_bucket,
            target_bucket=assessment.target_bucket,
            face_ratio=assessment.face_ratio,
            sharpness=assessment.sharpness,
            brightness=assessment.brightness,
            yaw=assessment.yaw,
            pitch=assessment.pitch,
            roll=assessment.roll,
        )

    display_state.gate_status = assessment.status
    display_state.gate_reason = assessment.reason
    display_state.gate_current_bucket = assessment.current_bucket
    display_state.gate_target_bucket = assessment.target_bucket
    display_state.gate_pose = (
        f"yaw={_fmt_num(assessment.yaw)} pitch={_fmt_num(assessment.pitch)} roll={_fmt_num(assessment.roll)}"
    )
    display_state.gate_progress = _gate_progress(guided_state)

    if guided_state.completed:
        display_state.gate_status = "green"
        display_state.gate_reason = "Objetivo completado"
        return

    if assessment.status == "green" and args.enroll_person_id is not None:
        guided_state.consecutive_green += 1
    else:
        guided_state.consecutive_green = 0
        return

    now_ms = time.time() * 1000.0
    if guided_state.consecutive_green < guided_state.hold_frames:
        return
    if (now_ms - guided_state.last_capture_ts_ms) < guided_state.cooldown_ms:
        return

    ok = _enroll_current_frame(
        frame=frame,
        services=services,
        state=display_state,
        person_id=args.enroll_person_id,
        camera_id=args.camera_id,
        capture_type="guided_operational",
    )
    guided_state.last_capture_ts_ms = now_ms
    guided_state.consecutive_green = 0

    if not ok:
        return

    bucket = assessment.current_bucket or "center"
    if bucket in guided_state.captured_by_bucket:
        guided_state.captured_by_bucket[bucket] += 1
    guided_state.captured_total += 1
    display_state.gate_progress = _gate_progress(guided_state)

    if guided_state.captured_total >= guided_state.target_samples:
        guided_state.completed = True
        display_state.gate_status = "green"
        display_state.gate_reason = "Objetivo completado"


def _extract_face_observation(
    frame,
    services,
    display_state: DisplayState,
    max_points: int,
) -> DetectedFace | None:
    encoder = getattr(services.recognition_service, "_encoder", None)
    analyze = getattr(encoder, "analyze_face", None)
    if not callable(analyze):
        if not display_state.landmarks_warning_shown:
            display_state.landmarks_warning_shown = True
            display_state.message = "Guided mode requires ENCODER_BACKEND=insightface"
            display_state.message_until_ts = time.time() + 3.0
        return None
    try:
        return analyze(frame, max_points=max_points)
    except Exception:
        display_state.message = "Could not analyze face"
        display_state.message_until_ts = time.time() + 2.0
        return None


def _to_gate_observation(detected: DetectedFace | None) -> FaceObservation | None:
    if detected is None:
        return None
    return FaceObservation(
        bbox=detected.bbox,
        det_score=detected.det_score,
        yaw=detected.yaw,
        pitch=detected.pitch,
        roll=detected.roll,
    )


def _gate_progress(guided_state: GuidedEnrollState) -> str:
    c = guided_state.captured_by_bucket
    return (
        f"{guided_state.captured_total}/{guided_state.target_samples} "
        f"C{c['center']} L{c['left']} R{c['right']} U{c['up']} D{c['down']}"
    )


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.1f}"


def _draw_overlay(frame, state: DisplayState) -> None:
    _draw_landmarks(frame, state.landmarks)
    _draw_gate_border(frame, state.gate_status)
    h = frame.shape[0]

    line1 = "Decision: N/A"
    line2 = "Top1: N/A"
    line3 = "Latency: N/A"

    if state.result is not None:
        line1 = f"Decision: {state.result.decision}"
        if state.result.top1 is not None:
            line2 = f"Top1: {state.result.top1.person_id} ({state.result.top1.score:.3f})"
        if state.latency_ms is not None:
            line3 = f"Latency: {state.latency_ms:.1f} ms"

    cv2.putText(frame, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, line2, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    cv2.putText(frame, line3, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    if state.gate_status is not None:
        gate = state.gate_status.upper()
        current = bucket_instruction(state.gate_current_bucket)
        target = bucket_instruction(state.gate_target_bucket)
        cv2.putText(
            frame,
            f"Gate: {gate} | now:{current} -> target:{target}",
            (10, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 255, 80),
            2,
        )
        cv2.putText(
            frame,
            state.gate_reason,
            (10, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 255, 160),
            2,
        )
        cv2.putText(
            frame,
            f"{state.gate_progress} | {state.gate_pose}",
            (10, 164),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (190, 240, 190),
            1,
        )

    cv2.putText(frame, "q: quit | e: enroll", (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    if state.message and time.time() <= state.message_until_ts:
        cv2.putText(frame, state.message, (10, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)


def _draw_gate_border(frame, gate_status: str | None) -> None:
    if gate_status is None:
        return
    h, w = frame.shape[:2]
    color_map = {
        "red": (0, 0, 255),
        "yellow": (0, 220, 255),
        "green": (0, 200, 0),
    }
    color = color_map.get(gate_status, (180, 180, 180))
    cv2.rectangle(frame, (2, 2), (w - 3, h - 3), color, 3, lineType=cv2.LINE_AA)


def _update_landmarks(frame, services, state: DisplayState, requested_points: int) -> None:
    max_points = max(10, requested_points)
    encoder = getattr(services.recognition_service, "_encoder", None)
    extract = getattr(encoder, "extract_landmarks", None)
    if not callable(extract):
        state.landmarks = []
        if not state.landmarks_warning_shown:
            state.landmarks_warning_shown = True
            state.message = "Landmarks require ENCODER_BACKEND=insightface"
            state.message_until_ts = time.time() + 3.0
        return

    try:
        state.landmarks = extract(frame, max_points=max_points)
    except Exception:
        state.landmarks = []
        state.message = "Could not compute landmarks"
        state.message_until_ts = time.time() + 2.0


def _draw_landmarks(frame, points: list[tuple[int, int]]) -> None:
    if not points:
        return

    h, w = frame.shape[:2]
    arr = np.asarray(points, dtype=np.float32)
    mask = (
        (arr[:, 0] >= 0)
        & (arr[:, 0] < w)
        & (arr[:, 1] >= 0)
        & (arr[:, 1] < h)
    )
    arr = arr[mask]
    if arr.shape[0] == 0:
        return

    x_min, y_min = arr.min(axis=0)
    x_max, y_max = arr.max(axis=0)
    dx = max(1.0, float(x_max - x_min))
    dy = max(1.0, float(y_max - y_min))
    nx = (arr[:, 0] - x_min) / dx
    ny = (arr[:, 1] - y_min) / dy

    contour = _region(arr, (ny >= 0.10))
    brow_left = _region(arr, (nx <= 0.48) & (ny >= 0.10) & (ny <= 0.42))
    brow_right = _region(arr, (nx >= 0.52) & (ny >= 0.10) & (ny <= 0.42))
    eye_left = _region(arr, (nx <= 0.50) & (ny >= 0.25) & (ny <= 0.58))
    eye_right = _region(arr, (nx >= 0.50) & (ny >= 0.25) & (ny <= 0.58))
    nose = _region(arr, (nx >= 0.33) & (nx <= 0.67) & (ny >= 0.30) & (ny <= 0.82))
    mouth = _region(arr, (nx >= 0.20) & (nx <= 0.80) & (ny >= 0.60))

    contour_poly = _convex_hull(contour)
    _draw_polyline(frame, contour_poly, closed=True)
    _draw_polyline(frame, _sorted_by_x(brow_left), closed=False)
    _draw_polyline(frame, _sorted_by_x(brow_right), closed=False)
    _draw_polyline(frame, _ordered_loop(eye_left), closed=True)
    _draw_polyline(frame, _ordered_loop(eye_right), closed=True)
    _draw_polyline(frame, _sorted_by_y(nose), closed=False)
    _draw_polyline(frame, _ordered_loop(mouth), closed=True)

    for x, y in arr:
        cv2.circle(
            frame,
            (int(round(float(x))), int(round(float(y)))),
            1,
            (0, 255, 0),
            -1,
            lineType=cv2.LINE_AA,
        )


def _region(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return points[mask]


def _sorted_by_x(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return points
    order = np.argsort(points[:, 0])
    return points[order]


def _sorted_by_y(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return points
    order = np.argsort(points[:, 1])
    return points[order]


def _ordered_loop(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        return points
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    return points[order]


def _convex_hull(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        return points
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    return _ordered_loop(hull)


def _draw_polyline(frame, points: np.ndarray, closed: bool) -> None:
    if points.shape[0] < 2:
        return
    max_jump = _max_segment_distance(points)
    if not np.isfinite(max_jump):
        return

    pts = np.round(points).astype(np.int32)
    for i in range(pts.shape[0] - 1):
        p1 = (int(pts[i, 0]), int(pts[i, 1]))
        p2 = (int(pts[i + 1, 0]), int(pts[i + 1, 1]))
        if np.hypot(float(p2[0] - p1[0]), float(p2[1] - p1[1])) <= max_jump:
            cv2.line(frame, p1, p2, (0, 200, 0), 1, lineType=cv2.LINE_AA)

    if closed and pts.shape[0] > 2:
        p1 = (int(pts[-1, 0]), int(pts[-1, 1]))
        p2 = (int(pts[0, 0]), int(pts[0, 1]))
        if np.hypot(float(p2[0] - p1[0]), float(p2[1] - p1[1])) <= max_jump:
            cv2.line(frame, p1, p2, (0, 200, 0), 1, lineType=cv2.LINE_AA)


def _max_segment_distance(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return float("nan")

    diff = points[:, None, :] - points[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    nearest = np.min(d2, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return float("nan")

    base = float(np.median(np.sqrt(nearest)))
    return max(3.0, base * 2.2)


if __name__ == "__main__":
    main()
