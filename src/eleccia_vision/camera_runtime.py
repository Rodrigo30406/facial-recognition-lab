#!/usr/bin/env python3
"""Run the local webcam runtime for facial recognition.

Usage:
  python3 scripts/run_vision_runtime.py --camera-index 0 --camera-id cam-01

Controls:
  q: quit
  e: enroll current frame (requires --enroll-person-id)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
import numpy as np

from eleccia_vision.application.enrollment import InvalidImageError, PersonNotFoundError
from eleccia_vision.application.quality_gate import (
    AngleBucket,
    FaceObservation,
    QualityGateThresholds,
    build_angle_plan,
    evaluate_quality_gate,
)
from eleccia_core.bootstrap import build_services
from eleccia_vision.domain.entities import RecognitionCandidate, RecognitionResult
from eleccia_vision.infrastructure.insightface_encoder import DetectedFace
from eleccia_voice import VoiceAssistant, build_voice_settings_from_args


@dataclass
class DisplayState:
    result: RecognitionResult | None = None
    latency_ms: float | None = None
    fps: float | None = None
    message: str = ""
    message_until_ts: float = 0.0
    enroll_enabled: bool = False
    landmarks: list[tuple[int, int]] = field(default_factory=list)
    landmarks_warning_shown: bool = False
    gate_status: str | None = None
    gate_reason: str = ""
    gate_progress: str = ""
    gate_pose: str = ""
    gate_current_bucket: AngleBucket | None = None
    gate_target_bucket: AngleBucket | None = None
    gate_buckets: dict[str, tuple[int, int]] = field(default_factory=dict)
    face_overlays: list["FaceOverlay"] = field(default_factory=list)
    unknown_label_by_track: dict[str, int] = field(default_factory=dict)
    unknown_last_seen_ts_by_track: dict[str, float] = field(default_factory=dict)
    next_unknown_label_id: int = 1
    face_track_centers: dict[str, tuple[float, float]] = field(default_factory=dict)
    face_track_last_seen_ts: dict[str, float] = field(default_factory=dict)
    next_face_track_id: int = 1
    person_metadata_cache: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    last_event_signature_by_track: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    last_event_ts_by_track: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FaceOverlay:
    bbox: tuple[int, int, int, int]
    label: str
    in_range: bool
    regreet_armed: bool
    face_ratio: float | None
    landmarks: list[tuple[int, int]]


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
    # Turning the head to cover left/right/up/down poses shrinks the face box and
    # lowers detection confidence. These defaults leave enough headroom so a
    # turned face still clears the quality gate; otherwise the proximity/detection
    # checks reject the very pose the guide is asking for ("acercate/mira a camara"
    # while it wants you to turn away). Tighten via the "strict" preset or the
    # ELECCIA_GUIDED_* env vars when capture quality matters more than ease.
    min_det_score=0.45,
    min_face_ratio=0.045,
    min_sharpness=55.0,
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

TRACKING_MIN_FACE_RATIO = 0.0035

_IP_SOURCE_ALIASES = {"ip", "network", "rtsp"}


def _inject_credentials(url: str, user: str, password: str) -> str:
    """Insert user:password into a URL that doesn't already carry credentials."""
    if not user:
        return url
    try:
        parts = urlsplit(url)
        if parts.username or parts.password:
            return url
        host = parts.hostname or ""
        portpart = f":{parts.port}" if parts.port else ""
        auth = f"{quote(user, safe='')}:{quote(password, safe='')}@" if password else f"{quote(user, safe='')}@"
        return urlunsplit((parts.scheme, f"{auth}{host}{portpart}", parts.path, parts.query, parts.fragment))
    except Exception:
        return url


def _build_camera_source(args: argparse.Namespace) -> int | str:
    """Return an int local index or a URL string for the IP camera."""
    source = str(getattr(args, "camera_source", "local") or "local").strip().lower()
    if source not in _IP_SOURCE_ALIASES:
        return int(args.camera_index)

    user = (getattr(args, "camera_user", None) or "").strip()
    password = getattr(args, "camera_password", None) or ""

    url = (getattr(args, "camera_url", None) or "").strip()
    if url:
        return _inject_credentials(url, user, password)

    host = (getattr(args, "camera_ip", None) or "").strip()
    if not host:
        raise SystemExit(
            "Camera source 'ip' requires ELECCIA_CAMERA_IP (host/IP) or ELECCIA_CAMERA_URL"
        )
    scheme = (getattr(args, "camera_scheme", None) or "rtsp").strip() or "rtsp"
    port = int(getattr(args, "camera_port", 0) or 0)
    path = (getattr(args, "camera_rtsp_path", None) or "").strip()
    if path and not path.startswith("/"):
        path = "/" + path

    auth = ""
    if user:
        auth = f"{quote(user, safe='')}:{quote(password, safe='')}@" if password else f"{quote(user, safe='')}@"
    portpart = f":{port}" if port else ""
    return f"{scheme}://{auth}{host}{portpart}{path}"


def _mask_source(source: int | str) -> str:
    """Human-readable source label with credentials hidden."""
    if not isinstance(source, str):
        return f"index {source}"
    try:
        parts = urlsplit(source)
        if parts.username or parts.password:
            netloc = parts.hostname or ""
            if parts.port:
                netloc += f":{parts.port}"
            masked = f"{parts.username}:***@{netloc}" if parts.username else f"***@{netloc}"
            return urlunsplit((parts.scheme, masked, parts.path, parts.query, ""))
    except Exception:
        pass
    return source


class LatestFrameGrabber:
    """Continuously captures frames and keeps only the newest one."""

    def __init__(self, source: int | str, width: int = 0, height: int = 0, fps: int = 0) -> None:
        self._source: int | str = source if isinstance(source, str) else int(source)
        self._is_network = isinstance(self._source, str)
        self._width = int(width or 0)
        self._height = int(height or 0)
        self._fps = int(fps or 0)
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_id = 0
        self._latest_error_ts = 0.0
        self._consecutive_failures = 0

    @property
    def source_label(self) -> str:
        return _mask_source(self._source)

    def _open_capture(self) -> cv2.VideoCapture:
        if self._is_network:
            # FFMPEG backend handles RTSP/HTTP streams.
            return cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)
        return cv2.VideoCapture(self._source)

    def _configure(self, cap: cv2.VideoCapture) -> None:
        # Keep the buffer small so we always process the freshest frame.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if self._is_network:
            # Resolution/FPS/FOURCC come from the IP camera's own config; the
            # capture props below only apply to local USB devices.
            return
        # MJPG lets most USB webcams stream 720p+ at full frame rate; without it
        # they fall back to raw YUY2 and throttle to ~10 FPS over USB2 bandwidth.
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        if self._width and self._height:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
            except Exception:
                pass
        if self._fps:
            try:
                cap.set(cv2.CAP_PROP_FPS, float(self._fps))
            except Exception:
                pass

    def start(self) -> None:
        cap = self._open_capture()
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source {self.source_label}")

        self._configure(cap)

        self._cap = cap
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="camera-grabber", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        cap = self._cap
        self._cap = None
        if cap is not None:
            cap.release()

    def get_latest(self) -> tuple[int, np.ndarray] | None:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame_id, self._latest_frame.copy()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            cap = self._cap
            if cap is None:
                time.sleep(0.05)
                continue
            ok, frame = cap.read()
            if not ok:
                self._consecutive_failures += 1
                now = time.time()
                if (now - self._latest_error_ts) > 2.0:
                    self._latest_error_ts = now
                    print("[camera] frame read failed; retrying...")
                if self._consecutive_failures >= 50:
                    self._try_reopen_capture()
                    self._consecutive_failures = 0
                time.sleep(0.02)
                continue

            self._consecutive_failures = 0
            with self._lock:
                self._latest_frame = frame
                self._latest_frame_id += 1

    def _try_reopen_capture(self) -> None:
        old = self._cap
        self._cap = None
        if old is not None:
            try:
                old.release()
            except Exception:
                pass
        try:
            cap = self._open_capture()
            if not cap.isOpened():
                print(f"[camera] reopen failed for source {self.source_label}")
                return
            self._configure(cap)
            self._cap = cap
            print(f"[camera] reopened camera source {self.source_label}")
        except Exception as exc:
            print(f"[camera] reopen exception: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam runtime for facial recognition")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--camera-id", type=str, default="cam-01", help="Camera identifier")
    parser.add_argument(
        "--capture-width",
        type=int,
        default=1280,
        help="Requested camera capture width in pixels (0 = camera default)",
    )
    parser.add_argument(
        "--capture-height",
        type=int,
        default=720,
        help="Requested camera capture height in pixels (0 = camera default)",
    )
    parser.add_argument(
        "--capture-fps",
        type=int,
        default=30,
        help="Requested camera frame rate (0 = camera default)",
    )
    parser.add_argument(
        "--camera-source",
        choices=("local", "ip"),
        default="local",
        help="Camera source: 'local' USB index or 'ip' network camera (RTSP/HTTP)",
    )
    parser.add_argument("--camera-ip", type=str, default=None, help="IP camera host/IP (source=ip)")
    parser.add_argument("--camera-user", type=str, default=None, help="IP camera username")
    parser.add_argument("--camera-password", type=str, default=None, help="IP camera password")
    parser.add_argument("--camera-port", type=int, default=554, help="IP camera port (default 554 for RTSP)")
    parser.add_argument(
        "--camera-rtsp-path",
        type=str,
        default="",
        help="RTSP stream path, e.g. /Streaming/Channels/101 (Hikvision) or /cam/realmonitor?channel=1&subtype=0 (Dahua)",
    )
    parser.add_argument("--camera-scheme", type=str, default="rtsp", help="IP camera URL scheme (rtsp/http)")
    parser.add_argument(
        "--camera-url",
        type=str,
        default=None,
        help="Full IP camera URL (overrides ip/port/path; credentials injected if missing)",
    )
    parser.add_argument(
        "--recognize-every",
        type=int,
        default=5,
        help="Run recognition every N frames",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="Facial Recognition Runtime",
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
        choices=("auto", "melotts", "pyttsx3", "spd-say", "espeak"),
        default="auto",
        help="TTS backend to use when --voice-greet is enabled",
    )
    parser.add_argument(
        "--voice-template",
        type=str,
        default="Hola {name}, {welcome} al laboratorio de IA, Eleccia",
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
        "--voice-greet-unknown",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(parsed_argv)
    _apply_runtime_env_defaults(args, parsed_argv)
    _apply_guided_preset(args)
    return args


def _apply_runtime_env_defaults(args: argparse.Namespace, argv: list[str]) -> None:
    file_values = _read_env_file()

    _apply_env_value(
        args=args,
        attr="camera_index",
        env_key="ELECCIA_CAMERA_INDEX",
        parser=_to_int,
        argv=argv,
        flag="--camera-index",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="camera_id",
        env_key="ELECCIA_CAMERA_ID",
        parser=_to_str,
        argv=argv,
        flag="--camera-id",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="capture_width",
        env_key="ELECCIA_CAMERA_WIDTH",
        parser=_to_int,
        argv=argv,
        flag="--capture-width",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="capture_height",
        env_key="ELECCIA_CAMERA_HEIGHT",
        parser=_to_int,
        argv=argv,
        flag="--capture-height",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="capture_fps",
        env_key="ELECCIA_CAMERA_FPS",
        parser=_to_int,
        argv=argv,
        flag="--capture-fps",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="camera_source",
        env_key="ELECCIA_CAMERA_SOURCE",
        parser=_to_str,
        argv=argv,
        flag="--camera-source",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="camera_ip",
        env_key="ELECCIA_CAMERA_IP",
        parser=_to_optional_str,
        argv=argv,
        flag="--camera-ip",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="camera_user",
        env_key="ELECCIA_CAMERA_USER",
        parser=_to_optional_str,
        argv=argv,
        flag="--camera-user",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="camera_password",
        env_key="ELECCIA_CAMERA_PASSWORD",
        parser=_to_optional_str,
        argv=argv,
        flag="--camera-password",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="camera_port",
        env_key="ELECCIA_CAMERA_PORT",
        parser=_to_int,
        argv=argv,
        flag="--camera-port",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="camera_rtsp_path",
        env_key="ELECCIA_CAMERA_RTSP_PATH",
        parser=_to_str,
        argv=argv,
        flag="--camera-rtsp-path",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="camera_scheme",
        env_key="ELECCIA_CAMERA_SCHEME",
        parser=_to_str,
        argv=argv,
        flag="--camera-scheme",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="camera_url",
        env_key="ELECCIA_CAMERA_URL",
        parser=_to_optional_str,
        argv=argv,
        flag="--camera-url",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="recognize_every",
        env_key="ELECCIA_RECOGNIZE_EVERY",
        parser=_to_int,
        argv=argv,
        flag="--recognize-every",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="window_name",
        env_key="ELECCIA_WINDOW_NAME",
        parser=_to_str,
        argv=argv,
        flag="--window-name",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="enroll_person_id",
        env_key="ELECCIA_ENROLL_PERSON_ID",
        parser=_to_optional_str,
        argv=argv,
        flag="--enroll-person-id",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="show_landmarks",
        env_key="ELECCIA_SHOW_LANDMARKS",
        parser=_to_bool,
        argv=argv,
        flag="--show-landmarks",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="landmarks_max_points",
        env_key="ELECCIA_LANDMARKS_MAX_POINTS",
        parser=_to_int,
        argv=argv,
        flag="--landmarks-max-points",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="landmarks_every",
        env_key="ELECCIA_LANDMARKS_EVERY",
        parser=_to_int,
        argv=argv,
        flag="--landmarks-every",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="guided_enroll",
        env_key="ELECCIA_GUIDED_ENROLL",
        parser=_to_bool,
        argv=argv,
        flag="--guided-enroll",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="guided_target_samples",
        env_key="ELECCIA_GUIDED_TARGET_SAMPLES",
        parser=_to_int,
        argv=argv,
        flag="--guided-target-samples",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="guided_hold_frames",
        env_key="ELECCIA_GUIDED_HOLD_FRAMES",
        parser=_to_int,
        argv=argv,
        flag="--guided-hold-frames",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="guided_cooldown_ms",
        env_key="ELECCIA_GUIDED_COOLDOWN_MS",
        parser=_to_int,
        argv=argv,
        flag="--guided-cooldown-ms",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="guided_preset",
        env_key="ELECCIA_GUIDED_PRESET",
        parser=_to_guided_preset,
        argv=argv,
        flag="--guided-preset",
        file_values=file_values,
    )

    _apply_env_value(
        args=args,
        attr="voice_greet",
        env_key="ELECCIA_VISION_VOICE_GREET",
        parser=_to_bool,
        argv=argv,
        flag="--voice-greet",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_backend",
        env_key="ELECCIA_VOICE_BACKEND",
        parser=_to_voice_backend,
        argv=argv,
        flag="--voice-backend",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_template",
        env_key="ELECCIA_VOICE_TEMPLATE",
        parser=_to_str,
        argv=argv,
        flag="--voice-template",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_reentry_delay_seconds",
        env_key="ELECCIA_VOICE_REENTRY_DELAY_SECONDS",
        parser=_to_float,
        argv=argv,
        flag="--voice-reentry-delay-seconds",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_absence_seconds",
        env_key="ELECCIA_VOICE_ABSENCE_SECONDS",
        parser=_to_float,
        argv=argv,
        flag="--voice-absence-seconds",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_min_face_ratio",
        env_key="ELECCIA_VOICE_MIN_FACE_RATIO",
        parser=_to_float,
        argv=argv,
        flag="--voice-min-face-ratio",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_greet_unknown",
        env_key="ELECCIA_VOICE_GREET_UNKNOWN",
        parser=_to_bool,
        argv=argv,
        flag="--voice-greet-unknown",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_rate",
        env_key="ELECCIA_VOICE_RATE",
        parser=_to_int,
        argv=argv,
        flag="--voice-rate",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_volume",
        env_key="ELECCIA_VOICE_VOLUME",
        parser=_to_float,
        argv=argv,
        flag="--voice-volume",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_id",
        env_key="ELECCIA_VOICE_ID",
        parser=_to_optional_str,
        argv=argv,
        flag="--voice-id",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="voice_lang",
        env_key="ELECCIA_VOICE_LANG",
        parser=_to_optional_str,
        argv=argv,
        flag="--voice-lang",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="melo_language",
        env_key="ELECCIA_MELO_LANGUAGE",
        parser=_to_optional_str,
        argv=argv,
        flag="--melo-language",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="melo_speaker",
        env_key="ELECCIA_MELO_SPEAKER",
        parser=_to_optional_str,
        argv=argv,
        flag="--melo-speaker",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="melo_speed",
        env_key="ELECCIA_MELO_SPEED",
        parser=_to_float,
        argv=argv,
        flag="--melo-speed",
        file_values=file_values,
    )
    _apply_env_value(
        args=args,
        attr="melo_device",
        env_key="ELECCIA_MELO_DEVICE",
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
    if not raw.strip():
        return
    setattr(args, attr, parser(raw))


def _flag_present(argv: list[str], flag: str) -> bool:
    negative_flag = f"--no-{flag[2:]}" if flag.startswith("--") else ""
    for token in argv:
        if token == flag or token.startswith(f"{flag}="):
            return True
        if negative_flag and (token == negative_flag or token.startswith(f"{negative_flag}=")):
            return True
    return False


def _read_env_file() -> dict[str, str]:
    env_file = os.getenv("ELECCIA_ENV_FILE", ".env")
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
    if key in os.environ:
        return os.environ[key]
    if key in file_values:
        return file_values[key]
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
        raise ValueError(f"ELECCIA_GUIDED_PRESET must be one of: {allowed}")
    return value


def _to_voice_backend(raw: str) -> str:
    value = raw.strip().lower()
    allowed = {"auto", "melotts", "pyttsx3", "spd-say", "espeak"}
    if value not in allowed:
        raise ValueError(
            "ELECCIA_VOICE_BACKEND must be one of: auto, melotts, pyttsx3, spd-say, espeak"
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


def run_camera_runtime(args: argparse.Namespace, stop_event: threading.Event | None = None) -> None:
    if args.enroll_person_id:
        # Enrollment is data capture, not identification; greetings only belong
        # to detection mode even if ELECCIA_VISION_VOICE_GREET=true globally.
        args.voice_greet = False

    services = build_services()
    guided_state = _build_guided_enroll_state(args)
    gate_thresholds = _build_gate_thresholds(args)
    voice_assistant = VoiceAssistant(build_voice_settings_from_args(args))

    camera_source = _build_camera_source(args)
    if isinstance(camera_source, str):
        # Prefer TCP for RTSP: fewer artifacts/drops than the default UDP.
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    frame_grabber = LatestFrameGrabber(
        camera_source,
        width=int(getattr(args, "capture_width", 0) or 0),
        height=int(getattr(args, "capture_height", 0) or 0),
        fps=int(getattr(args, "capture_fps", 0) or 0),
    )
    try:
        frame_grabber.start()
    except Exception as exc:
        raise SystemExit(str(exc))

    print("Camera runtime started")
    print(f"- Camera source: {frame_grabber.source_label}")
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
            print("- Voice greet enabled but no TTS backend found (melotts/pyttsx3/spd-say/espeak)")
            if voice_assistant.backend_error:
                print(f"- Voice backend detail: {voice_assistant.backend_error}")

    # WINDOW_NORMAL makes the window user-resizable (the default AUTOSIZE locks it
    # to the frame size); KEEPRATIO avoids stretching the video when resized.
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    init_w = int(getattr(args, "capture_width", 0) or 1280)
    init_h = int(getattr(args, "capture_height", 0) or 720)
    if init_w > 1280:  # keep the initial window reasonable on smaller screens
        init_h = int(init_h * (1280 / init_w))
        init_w = 1280
    cv2.resizeWindow(args.window_name, init_w, init_h)

    state = DisplayState()
    state.enroll_enabled = bool(args.enroll_person_id)
    frame_idx = 0
    last_frame_id = 0
    landmarks_every = max(1, int(args.landmarks_every or 1))
    prev_frame_ts = time.perf_counter()

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            latest = frame_grabber.get_latest()
            if latest is None:
                state.message = "Failed to read frame"
                state.message_until_ts = time.time() + 2.0
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if stop_event is not None and stop_event.is_set():
                    break
                time.sleep(0.01)
                continue
            frame_id, frame = latest
            if frame_id == last_frame_id:
                # No new frame available yet; avoid re-processing stale data.
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if stop_event is not None and stop_event.is_set():
                    break
                time.sleep(0.002)
                continue
            last_frame_id = frame_id
            now_ts = time.perf_counter()
            dt = now_ts - prev_frame_ts
            prev_frame_ts = now_ts
            if dt > 1e-6:
                instant_fps = 1.0 / dt
                if state.fps is None:
                    state.fps = instant_fps
                else:
                    state.fps = (state.fps * 0.85) + (instant_fps * 0.15)

            frame_idx += 1
            show_landmarks_now = bool(args.show_landmarks) and (frame_idx % landmarks_every == 0)
            if frame_idx % max(1, args.recognize_every) == 0:
                _run_recognition(
                    frame=frame,
                    services=services,
                    state=state,
                    camera_id=args.camera_id,
                    voice_assistant=voice_assistant,
                    min_face_ratio_for_label=max(0.0, float(args.voice_min_face_ratio)),
                    unknown_label_ttl_seconds=max(0.0, float(args.voice_absence_seconds)),
                    show_landmarks=show_landmarks_now,
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

            _draw_overlay(frame, state, show_landmarks=show_landmarks_now)
            cv2.imshow(args.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("e") and args.enroll_person_id:
                _enroll_current_frame(frame, services, state, args.enroll_person_id, args.camera_id)
    finally:
        voice_assistant.close()
        frame_grabber.stop()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None, stop_event: threading.Event | None = None) -> None:
    args = parse_args(argv)
    run_camera_runtime(args=args, stop_event=stop_event)


def _run_recognition(
    frame,
    services,
    state: DisplayState,
    camera_id: str,
    voice_assistant: VoiceAssistant,
    min_face_ratio_for_label: float,
    unknown_label_ttl_seconds: float,
    show_landmarks: bool,
) -> None:
    now_ts = time.time()
    _cleanup_stale_face_tracks(
        state=state,
        now_ts=now_ts,
        ttl_seconds=unknown_label_ttl_seconds,
    )
    _cleanup_unknown_track_labels(
        state=state,
        now_ts=now_ts,
        ttl_seconds=unknown_label_ttl_seconds,
        active_unknown_track_ids=set(),
    )

    t0 = time.perf_counter()
    gallery = _load_gallery_candidates(services)
    detected_faces = _analyze_detected_faces(
        frame=frame,
        services=services,
        max_points=(50 if show_landmarks else 1),
    )
    state.face_overlays = []
    state.landmarks = []

    if not detected_faces:
        state.result = RecognitionResult(
            decision="unknown_person",
            matched=False,
            person_id=None,
            top1=None,
            top2=None,
        )
        state.latency_ms = (time.perf_counter() - t0) * 1000.0
        return

    detected_faces = sorted(detected_faces, key=lambda face: (face.bbox[0], face.bbox[1]))
    primary_choice: tuple[float, RecognitionResult, DetectedFace] | None = None
    first_voice_message: str | None = None
    active_unknown_track_ids: set[str] = set()
    tracked_faces = _assign_track_ids_for_faces(
        state=state,
        faces=[
            face
            for face in detected_faces
            if _is_trackable_face(frame=frame, bbox=face.bbox, min_ratio=TRACKING_MIN_FACE_RATIO)
        ],
        now_ts=now_ts,
    )
    tracked_by_bbox = {tuple(face.bbox): track_id for track_id, face in tracked_faces}

    for idx, face in enumerate(detected_faces):
        track_id = tracked_by_bbox.get(tuple(face.bbox), f"ephemeral-{idx}")
        raw = _recognize_face_from_detection(
            frame=frame,
            face=face,
            services=services,
            gallery=gallery,
        )
        result = services.recognition_consistency_service.stabilize(
            raw,
            stream_id=f"{camera_id}::{track_id}",
        )
        if _should_record_event(state=state, track_id=track_id, result=result, now_ts=now_ts):
            services.recognition_event_service.record_from_result(
                result=result,
                camera_id=camera_id,
                track_id=track_id,
            )

        is_trackable = not track_id.startswith("ephemeral-")
        if result.decision == "unknown_person" and is_trackable:
            unknown_idx = _assign_unknown_label_id(
                state=state,
                track_id=track_id,
                now_ts=now_ts,
            )
            active_unknown_track_ids.add(track_id)
            label = _build_unknown_face_label(result=result, unknown_index=unknown_idx)
        elif result.decision == "unknown_person":
            label = _build_unknown_face_label(result=result, unknown_index=0)
        else:
            label = _build_face_label(services=services, state=state, result=result)
        bbox_int = _bbox_to_int(face.bbox, frame_shape=frame.shape)
        face_ratio = _face_ratio_from_bbox(frame=frame, bbox=face.bbox)
        in_range = _is_in_face_ratio_range(
            face_ratio=face_ratio,
            min_face_ratio=min_face_ratio_for_label,
        )
        regreet_armed = False

        if is_trackable:
            voice_message = voice_assistant.on_recognition(
                result=result,
                resolve_person=lambda person_id: _resolve_person_metadata(services, person_id),
                face_ratio=face_ratio,
                pose_yaw=face.yaw,
                pose_pitch=face.pitch,
                presence_id=track_id,
            )
            regreet_armed = voice_assistant.is_regreet_marker_active(track_id)
            if first_voice_message is None and voice_message is not None:
                first_voice_message = voice_message

        state.face_overlays.append(
            FaceOverlay(
                bbox=bbox_int,
                label=label,
                in_range=in_range,
                regreet_armed=regreet_armed,
                face_ratio=face_ratio,
                landmarks=(face.landmarks if show_landmarks else []),
            )
        )

        area = _bbox_area(face.bbox)
        if primary_choice is None or area > primary_choice[0]:
            primary_choice = (area, result, face)

    _cleanup_unknown_track_labels(
        state=state,
        now_ts=now_ts,
        ttl_seconds=unknown_label_ttl_seconds,
        active_unknown_track_ids=active_unknown_track_ids,
    )

    elapsed = (time.perf_counter() - t0) * 1000.0
    state.latency_ms = elapsed

    if primary_choice is None:
        state.result = RecognitionResult(
            decision="unknown_person",
            matched=False,
            person_id=None,
            top1=None,
            top2=None,
        )
        return

    _, primary_result, _primary_face = primary_choice
    state.result = primary_result

    voice_message = first_voice_message
    if voice_message:
        state.message = voice_message
        if voice_message.startswith("Saludo:"):
            state.message_until_ts = time.time() + 2.0
        else:
            state.message_until_ts = time.time() + 3.0


def _recognize_face_from_detection(
    frame,
    face: DetectedFace,
    services,
    gallery,
) -> RecognitionResult:
    if face.embedding:
        return _recognize_from_probe_embedding(
            services=services,
            probe=face.embedding,
            gallery=gallery,
        )

    face_payload = _crop_face_to_jpeg_bytes(frame=frame, bbox=face.bbox)
    if face_payload is None:
        return RecognitionResult(
            decision="unknown_person",
            matched=False,
            person_id=None,
            top1=None,
            top2=None,
        )
    return services.recognition_service.recognize(face_payload)


def _recognize_from_probe_embedding(services, probe: list[float], gallery) -> RecognitionResult:
    recognition_service = services.recognition_service
    if not gallery:
        return RecognitionResult(
            decision="unknown_person",
            matched=False,
            person_id=None,
            top1=None,
            top2=None,
        )

    ranked = recognition_service._searcher.search(
        probe_embedding=probe,
        candidates=gallery,
        top_k=recognition_service._settings.recognition_top_k,
    )
    if not ranked:
        return RecognitionResult(
            decision="unknown_person",
            matched=False,
            person_id=None,
            top1=None,
            top2=None,
        )

    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else None

    if top1.score < recognition_service._settings.recognition_threshold:
        return RecognitionResult(
            decision="unknown_person",
            matched=False,
            person_id=None,
            top1=top1,
            top2=top2,
        )

    if _is_ambiguous_candidates(top1=top1, top2=top2, margin=recognition_service._settings.recognition_margin):
        return RecognitionResult(
            decision="ambiguous_match",
            matched=False,
            person_id=None,
            top1=top1,
            top2=top2,
        )

    return RecognitionResult(
        decision="known_person",
        matched=True,
        person_id=top1.person_id,
        top1=top1,
        top2=top2,
    )


def _load_gallery_candidates(services):
    recognition_service = services.recognition_service
    return recognition_service._face_repository.list_all()


def _is_ambiguous_candidates(
    top1: RecognitionCandidate,
    top2: RecognitionCandidate | None,
    margin: float,
) -> bool:
    if top2 is None:
        return False
    return (top1.score - top2.score) < margin


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


def _resolve_person_metadata_cached(
    services,
    state: DisplayState,
    person_id: str,
) -> tuple[str, str | None]:
    cached = state.person_metadata_cache.get(person_id)
    if cached is not None:
        return cached
    resolved = _resolve_person_metadata(services=services, person_id=person_id)
    state.person_metadata_cache[person_id] = resolved
    return resolved


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


def _analyze_detected_faces(frame, services, max_points: int) -> list[DetectedFace]:
    encoder = getattr(services.recognition_service, "_encoder", None)
    analyze_many = getattr(encoder, "analyze_faces", None)
    if callable(analyze_many):
        try:
            faces = analyze_many(frame, max_points=max_points)
            if isinstance(faces, list):
                return faces
        except Exception:
            return []

    analyze_one = getattr(encoder, "analyze_face", None)
    if not callable(analyze_one):
        return []
    try:
        face = analyze_one(frame, max_points=max_points)
    except Exception:
        return []
    if face is None:
        return []
    return [face]


def _crop_face_to_jpeg_bytes(frame, bbox: tuple[float, float, float, float]) -> bytes | None:
    x1, y1, x2, y2 = _bbox_to_int(bbox, frame.shape)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return _frame_to_jpeg_bytes(crop)


def _bbox_to_int(
    bbox: tuple[float, float, float, float],
    frame_shape: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    ix1 = int(max(0, min(w - 1, round(float(x1)))))
    iy1 = int(max(0, min(h - 1, round(float(y1)))))
    ix2 = int(max(0, min(w, round(float(x2)))))
    iy2 = int(max(0, min(h, round(float(y2)))))
    return ix1, iy1, ix2, iy2


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    cx = (float(x1) + float(x2)) * 0.5
    cy = (float(y1) + float(y2)) * 0.5
    return cx, cy


def _assign_track_ids_for_faces(
    state: DisplayState,
    faces: list[DetectedFace],
    now_ts: float,
) -> list[tuple[str, DetectedFace]]:
    if not faces:
        return []

    available_tracks = list(state.face_track_centers.keys())
    assigned_tracks: set[str] = set()
    assignments: list[tuple[str, DetectedFace]] = []

    for face in faces:
        track_id = _match_face_to_existing_track(
            face=face,
            available_tracks=available_tracks,
            assigned_tracks=assigned_tracks,
            state=state,
        )
        if track_id is None:
            track_id = f"face-{state.next_face_track_id}"
            state.next_face_track_id += 1

        state.face_track_centers[track_id] = _bbox_center(face.bbox)
        state.face_track_last_seen_ts[track_id] = now_ts
        assigned_tracks.add(track_id)
        assignments.append((track_id, face))

    return assignments


def _match_face_to_existing_track(
    face: DetectedFace,
    available_tracks: list[str],
    assigned_tracks: set[str],
    state: DisplayState,
) -> str | None:
    face_center = _bbox_center(face.bbox)
    face_area = max(1.0, _bbox_area(face.bbox))
    max_dist = max(60.0, min(260.0, 0.8 * float(np.sqrt(face_area))))

    best_track: str | None = None
    best_distance = float("inf")
    for track_id in available_tracks:
        if track_id in assigned_tracks:
            continue
        center = state.face_track_centers.get(track_id)
        if center is None:
            continue
        dx = float(face_center[0] - center[0])
        dy = float(face_center[1] - center[1])
        dist = float(np.hypot(dx, dy))
        if dist < best_distance:
            best_distance = dist
            best_track = track_id

    if best_track is None:
        return None
    if best_distance > max_dist:
        return None
    return best_track


def _cleanup_stale_face_tracks(state: DisplayState, now_ts: float, ttl_seconds: float) -> None:
    if ttl_seconds <= 0.0:
        return

    stale_tracks = [
        track_id
        for track_id, last_seen in state.face_track_last_seen_ts.items()
        if (now_ts - float(last_seen)) >= ttl_seconds
    ]
    for track_id in stale_tracks:
        state.face_track_last_seen_ts.pop(track_id, None)
        state.face_track_centers.pop(track_id, None)
        state.last_event_signature_by_track.pop(track_id, None)
        state.last_event_ts_by_track.pop(track_id, None)


def _face_ratio_from_bbox(frame, bbox: tuple[float, float, float, float]) -> float | None:
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return None
    frame_area = float(h * w)
    if frame_area <= 0.0:
        return None
    ratio = _bbox_area(bbox) / frame_area
    return max(0.0, min(1.0, float(ratio)))


def _is_trackable_face(
    frame,
    bbox: tuple[float, float, float, float],
    min_ratio: float,
) -> bool:
    ratio = _face_ratio_from_bbox(frame=frame, bbox=bbox)
    if ratio is None:
        return False
    return float(ratio) >= max(0.0, float(min_ratio))


def _build_face_label(services, state: DisplayState, result: RecognitionResult) -> str:
    if result.decision == "known_person" and result.person_id is not None:
        name, _sex = _resolve_person_metadata_cached(services, state, result.person_id)
        if result.top1 is not None:
            return f"{name} ({result.top1.score:.2f})"
        return name

    if result.decision == "ambiguous_match" and result.top1 is not None:
        return f"Ambiguo: {result.top1.person_id} ({result.top1.score:.2f})"

    if result.top1 is not None:
        return f"Desconocido ({result.top1.score:.2f})"
    return "Desconocido"


def _build_unknown_face_label(result: RecognitionResult, unknown_index: int) -> str:
    base = "Desconocido"
    if unknown_index > 0:
        base = f"Desconocido {unknown_index}"
    if result.top1 is not None:
        return f"{base} ({result.top1.score:.2f})"
    return base


def _assign_unknown_label_id(state: DisplayState, track_id: str, now_ts: float) -> int:
    existing = state.unknown_label_by_track.get(track_id)
    if existing is not None:
        state.unknown_last_seen_ts_by_track[track_id] = now_ts
        return existing

    idx = max(1, int(state.next_unknown_label_id))
    state.next_unknown_label_id = idx + 1
    state.unknown_label_by_track[track_id] = idx
    state.unknown_last_seen_ts_by_track[track_id] = now_ts
    return idx


def _cleanup_unknown_track_labels(
    state: DisplayState,
    now_ts: float,
    ttl_seconds: float,
    active_unknown_track_ids: set[str],
) -> None:
    if ttl_seconds <= 0.0:
        return

    stale_tracks: list[str] = []
    for track_id, last_seen in state.unknown_last_seen_ts_by_track.items():
        if track_id in active_unknown_track_ids:
            continue
        if (now_ts - float(last_seen)) >= ttl_seconds:
            stale_tracks.append(track_id)

    for track_id in stale_tracks:
        state.unknown_last_seen_ts_by_track.pop(track_id, None)
        state.unknown_label_by_track.pop(track_id, None)


def _event_signature(result: RecognitionResult) -> tuple[str, str | None]:
    top1_person = result.top1.person_id if result.top1 is not None else None
    return result.decision, top1_person


def _should_record_event(
    state: DisplayState,
    track_id: str,
    result: RecognitionResult,
    now_ts: float,
    min_interval_seconds: float = 0.8,
) -> bool:
    signature = _event_signature(result)
    prev_signature = state.last_event_signature_by_track.get(track_id)
    prev_ts = state.last_event_ts_by_track.get(track_id, 0.0)

    if prev_signature == signature and (now_ts - prev_ts) < max(0.0, min_interval_seconds):
        return False

    state.last_event_signature_by_track[track_id] = signature
    state.last_event_ts_by_track[track_id] = now_ts
    return True


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
    display_state.face_overlays = []
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
    display_state.gate_buckets = {
        bucket: (
            guided_state.captured_by_bucket.get(bucket, 0),
            guided_state.plan_by_bucket.get(bucket, 0),
        )
        for bucket in ("center", "left", "right", "up", "down")
    }

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
            display_state.message = "Guided mode requires ELECCIA_ENCODER_BACKEND=insightface"
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


# ---------------------------------------------------------------------------
# HUD renderer
#
# Overlays are drawn straight onto the BGR camera frame. OpenCV's Hershey fonts
# are ASCII-only, so avoid accented literals here (they render as boxes);
# Spanish text produced by other modules is passed through as-is.
# ---------------------------------------------------------------------------

# Institutional JNE palette (BGR). Accent is the JNE guinda (deep crimson,
# ~#B01E3C); the surfaces are a warm charcoal with a faint guinda tint. Status
# colors stay semantic (green/amber/red) so they read regardless of branding.
_HEADER_H = 46
_C_INK = (242, 240, 242)
_C_MUTED = (168, 162, 170)
_C_PANEL = (30, 24, 36)
_C_HEADER = (20, 14, 26)
_C_ACCENT = (60, 30, 176)      # JNE guinda #B01E3C
_C_ACCENT_SOFT = (110, 92, 216)
_C_GREEN = (96, 210, 122)
_C_AMBER = (60, 190, 250)
_C_RED = (84, 84, 240)
_C_TRACK = (58, 52, 62)

_GATE_COLORS = {"red": _C_RED, "yellow": _C_AMBER, "green": _C_GREEN}

# Lazy, path-keyed cache for the optional header logo so it is decoded once
# instead of every frame.
_LOGO_CACHE: dict[str, object] = {"path": None, "loaded": False, "bgr": None, "alpha": None}


def _get_logo(height: int):
    path = os.getenv("ELECCIA_LOGO_PATH", "asset/logo.png")
    if _LOGO_CACHE["loaded"] and _LOGO_CACHE["path"] == path:
        return _LOGO_CACHE["bgr"], _LOGO_CACHE["alpha"]

    _LOGO_CACHE["loaded"] = True
    _LOGO_CACHE["path"] = path
    _LOGO_CACHE["bgr"] = None
    _LOGO_CACHE["alpha"] = None

    try:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        raw = None
    if raw is None or raw.ndim < 2:
        return None, None

    h0, w0 = raw.shape[:2]
    if h0 <= 0 or w0 <= 0:
        return None, None
    target_w = max(1, int(round(w0 * (height / float(h0)))))
    resized = cv2.resize(raw, (target_w, height), interpolation=cv2.INTER_AREA)

    if resized.ndim == 3 and resized.shape[2] == 4:
        bgr = resized[:, :, :3].copy()
        alpha = resized[:, :, 3].astype(np.float32) / 255.0
    else:
        bgr = resized[:, :, :3].copy() if resized.ndim == 3 else cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        alpha = np.ones((height, target_w), np.float32)

    _LOGO_CACHE["bgr"] = bgr
    _LOGO_CACHE["alpha"] = alpha
    return bgr, alpha


def _blit_logo(frame, bgr, alpha, x, y) -> None:
    fh, fw = frame.shape[:2]
    lh, lw = bgr.shape[:2]
    x, y = int(x), int(y)
    w = min(lw, fw - x)
    h = min(lh, fh - y)
    if w <= 0 or h <= 0:
        return
    roi = frame[y:y + h, x:x + w].astype(np.float32)
    a = alpha[:h, :w, None]
    blended = bgr[:h, :w].astype(np.float32) * a + roi * (1.0 - a)
    frame[y:y + h, x:x + w] = blended.astype(np.uint8)


def _fill_rounded(img, x1, y1, x2, y2, color, radius) -> None:
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    r = int(min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if r < 1:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1, lineType=cv2.LINE_AA)
        return
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.circle(img, (x1 + r, y1 + r), r, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x2 - r, y1 + r), r, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x1 + r, y2 - r), r, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x2 - r, y2 - r), r, color, -1, lineType=cv2.LINE_AA)


def _panel(frame, x1, y1, x2, y2, color=_C_PANEL, alpha=0.62, radius=14) -> None:
    h, w = frame.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    _fill_rounded(overlay, 0, 0, x2 - x1 - 1, y2 - y1 - 1, color, radius)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi)


def _tsize(text, scale, thickness, font=cv2.FONT_HERSHEY_SIMPLEX):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    return tw, th


def _text(frame, text, org, scale=0.5, color=_C_INK, thickness=1,
          font=cv2.FONT_HERSHEY_SIMPLEX, shadow=True) -> None:
    if shadow:
        cv2.putText(frame, text, (org[0] + 1, org[1] + 1), font, scale,
                    (0, 0, 0), thickness + 1, lineType=cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, lineType=cv2.LINE_AA)


def _corner_brackets(frame, bbox, color, thickness=2, length_frac=0.24) -> None:
    x1, y1, x2, y2 = bbox
    span = min(x2 - x1, y2 - y1)
    length = int(max(10, span * length_frac))
    for cx, cy, sx, sy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + sx * length, cy), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + sy * length), color, thickness, lineType=cv2.LINE_AA)


def _face_color(overlay: "FaceOverlay") -> tuple[int, int, int]:
    label = overlay.label or ""
    if label.startswith("Ambiguo"):
        return _C_AMBER
    if label.startswith("Desconocido"):
        return _C_RED
    if overlay.in_range and not overlay.regreet_armed:
        return _C_GREEN
    return _C_AMBER


def _label_chip(frame, anchor_x, anchor_y, text, dot_color) -> None:
    scale, thickness = 0.5, 1
    tw, th = _tsize(text, scale, thickness, cv2.FONT_HERSHEY_DUPLEX)
    pad, dot, gap = 10, 4, 8
    cw = pad + dot * 2 + gap + tw + pad
    ch = th + 14
    w = frame.shape[1]
    x1 = int(max(4, min(anchor_x, w - cw - 4)))
    y1 = int(anchor_y - ch)
    if y1 < _HEADER_H + 4:
        y1 = _HEADER_H + 4
    y2 = y1 + ch
    x2 = x1 + cw
    _panel(frame, x1, y1, x2, y2, color=(18, 16, 14), alpha=0.82, radius=ch // 2)
    cy = (y1 + y2) // 2
    cv2.circle(frame, (x1 + pad + dot, cy), dot, dot_color, -1, lineType=cv2.LINE_AA)
    _text(frame, text, (x1 + pad + dot * 2 + gap, cy + th // 2), scale, _C_INK,
          thickness, cv2.FONT_HERSHEY_DUPLEX, shadow=False)


def _chip_right(frame, right_x, cy, text, accent) -> int:
    scale, thickness = 0.5, 1
    tw, th = _tsize(text, scale, thickness)
    pad, dot, gap = 9, 4, 7
    cw = pad + dot * 2 + gap + tw + pad
    ch = 26
    x2, x1 = int(right_x), int(right_x) - cw
    y1, y2 = cy - ch // 2, cy + ch // 2
    _panel(frame, x1, y1, x2, y2, color=_C_PANEL, alpha=0.7, radius=ch // 2)
    cv2.circle(frame, (x1 + pad + dot, cy), dot, accent, -1, lineType=cv2.LINE_AA)
    _text(frame, text, (x1 + pad + dot * 2 + gap, cy + th // 2), scale, _C_INK,
          thickness, shadow=False)
    return x1


def _draw_viewfinder(frame, color) -> None:
    h, w = frame.shape[:2]
    m, length, t = 16, 40, 2
    for cx, cy, sx, sy in ((m, m, 1, 1), (w - m, m, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + sx * length, cy), color, t, lineType=cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + sy * length), color, t, lineType=cv2.LINE_AA)


def _draw_header(frame, state: DisplayState) -> None:
    w = frame.shape[1]
    _panel(frame, 0, 0, w, _HEADER_H, color=_C_HEADER, alpha=0.58, radius=0)
    cv2.line(frame, (0, _HEADER_H), (w, _HEADER_H), _C_ACCENT, 1, lineType=cv2.LINE_AA)

    x = 16
    logo_bgr, logo_alpha = _get_logo(_HEADER_H - 16)
    if logo_bgr is not None:
        _blit_logo(frame, logo_bgr, logo_alpha, x, 8)
        x += logo_bgr.shape[1] + 12
    else:
        cv2.circle(frame, (x + 10, 23), 7, _C_ACCENT, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (x + 10, 23), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        x += 28

    # Wordmark "EleccIA": the "IA" is drawn in the accent to highlight the AI.
    _text(frame, "Elecc", (x, 30), 0.72, _C_INK, 1, cv2.FONT_HERSHEY_DUPLEX)
    ew, _ = _tsize("Elecc", 0.72, 1, cv2.FONT_HERSHEY_DUPLEX)
    _text(frame, "IA", (x + ew, 30), 0.72, _C_ACCENT_SOFT, 1, cv2.FONT_HERSHEY_DUPLEX)
    iw, _ = _tsize("IA", 0.72, 1, cv2.FONT_HERSHEY_DUPLEX)
    brand_end = x + ew + iw

    # Telemetry chips, drawn right-to-left. On narrow windows, stop before they
    # would collide with the wordmark instead of overlapping the subtitle.
    cy = 23
    chips = [("EN LINEA", _C_GREEN)]
    if state.fps is not None:
        chips.append((f"{state.fps:.0f} FPS", _C_ACCENT))
    if state.latency_ms is not None:
        chips.append((f"{state.latency_ms:.0f} ms", _C_ACCENT))
    chips.append((f"{len(state.face_overlays)} rostro(s)", _C_ACCENT))

    left_limit = brand_end + 16
    rx = w - 14
    chips_left = w
    for text, accent in chips:
        tw, _ = _tsize(text, 0.5, 1)
        chip_w = tw + 33  # pad + dot + gap + text + pad, matches _chip_right
        if rx - chip_w < left_limit:
            break
        rx = _chip_right(frame, rx, cy, text, accent)
        chips_left = rx
        rx -= 8

    # Subtitle only when it fits between the wordmark and the chips.
    subtitle = "Reconocimiento facial"
    sw, _ = _tsize(subtitle, 0.46, 1)
    if brand_end + 14 + sw + 12 <= chips_left:
        _text(frame, subtitle, (brand_end + 14, 30), 0.46, _C_MUTED, 1)


def _parse_progress(text: str) -> tuple[int, int, dict[str, int]]:
    done = total = 0
    buckets: dict[str, int] = {}
    try:
        parts = text.split()
        if parts and "/" in parts[0]:
            a, b = parts[0].split("/", 1)
            done, total = int(a), int(b)
        for token in parts[1:]:
            if len(token) >= 2 and token[0] in "CLRUD":
                buckets[token[0]] = int(token[1:])
    except (ValueError, IndexError):
        pass
    return done, total, buckets


# Short, imperative head-pose instructions (ASCII for the Hershey font). The
# words are what the user should DO, not the internal bucket name.
_POSE_INSTRUCTION = {
    "center": "Mira al frente",
    "left": "Voltea a tu izquierda",
    "right": "Voltea a tu derecha",
    "up": "Sube la cabeza",
    "down": "Baja la cabeza",
}

# Pose-map dots laid out spatially so their position mirrors the head movement.
_POSE_MAP = {"center": (0, 0), "left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}


def _fit_scale(text, max_w, start=0.6, min_scale=0.42, thickness=1,
               font=cv2.FONT_HERSHEY_DUPLEX) -> float:
    scale = start
    while scale > min_scale:
        tw, _ = _tsize(text, scale, thickness, font)
        if tw <= max_w:
            return scale
        scale -= 0.02
    return min_scale


def _enroll_instruction(state: DisplayState) -> str:
    reason = state.gate_reason or ""
    if reason.startswith("Objetivo completado"):
        return "Listo! Rostro registrado"
    if state.gate_status == "green":
        return "Perfecto, no te muevas"
    if state.gate_status == "yellow" and state.gate_target_bucket in _POSE_INSTRUCTION:
        return _POSE_INSTRUCTION[state.gate_target_bucket]
    return reason or "Acomodate frente a la camara"


def _enroll_counts(state: DisplayState) -> tuple[int, int]:
    if state.gate_buckets:
        done = sum(cap for cap, _ in state.gate_buckets.values())
        total = sum(plan for _, plan in state.gate_buckets.values())
        if total > 0:
            return done, total
    done, total, _ = _parse_progress(state.gate_progress)
    return done, total


def _draw_direction_cue(frame, bucket, color) -> None:
    """Big arrow on the frame edge pointing where the user should turn."""
    h, w = frame.shape[:2]
    xc, yc = w // 2, h // 2
    arrows = {
        "left": ((int(w * 0.17), yc), (int(w * 0.09), yc)),
        "right": ((int(w * 0.83), yc), (int(w * 0.91), yc)),
        "up": ((xc, int(h * 0.32)), (xc, int(h * 0.22))),
        "down": ((xc, int(h * 0.72)), (xc, int(h * 0.82))),
    }
    if bucket not in arrows:
        return
    p1, p2 = arrows[bucket]
    cv2.arrowedLine(frame, p1, p2, (0, 0, 0), 11, cv2.LINE_AA, 0, 0.45)
    cv2.arrowedLine(frame, p1, p2, color, 6, cv2.LINE_AA, 0, 0.45)


def _draw_pose_map(frame, cx, cy, state: DisplayState, accent) -> None:
    spacing, radius = 20, 7
    pending = (98, 92, 102)
    # Faint cross so the layout reads as directions.
    cv2.line(frame, (cx - spacing, cy), (cx + spacing, cy), _C_TRACK, 1, lineType=cv2.LINE_AA)
    cv2.line(frame, (cx, cy - spacing), (cx, cy + spacing), _C_TRACK, 1, lineType=cv2.LINE_AA)
    for bucket, (dx, dy) in _POSE_MAP.items():
        px, py = cx + dx * spacing, cy + dy * spacing
        cap, plan = state.gate_buckets.get(bucket, (0, 0))
        done = plan > 0 and cap >= plan
        is_target = bucket == state.gate_target_bucket
        if is_target and not done:
            cv2.circle(frame, (px, py), radius + 3, accent, 2, lineType=cv2.LINE_AA)
        color = _C_GREEN if done else (accent if is_target else pending)
        cv2.circle(frame, (px, py), radius, color, -1, lineType=cv2.LINE_AA)


def _draw_enroll_panel(frame, state: DisplayState) -> None:
    if state.gate_status is None:
        return

    h, w = frame.shape[:2]
    accent = _GATE_COLORS.get(state.gate_status, _C_MUTED)
    pw, ph = 430, 132
    x1, y2 = 16, h - 16
    x2, y1 = x1 + pw, y2 - ph
    _panel(frame, x1, y1, x2, y2, alpha=0.62, radius=16)
    _fill_rounded(frame, x1 + 10, y1 + 14, x1 + 14, y2 - 14, accent, 2)

    tx = x1 + 26
    col_right = x2 - 96  # reserve the right block for the pose map

    _text(frame, "ENROLAMIENTO GUIADO", (tx, y1 + 24), 0.44, _C_MUTED, 1,
          cv2.FONT_HERSHEY_DUPLEX, shadow=False)

    done, total = _enroll_counts(state)
    count = f"Captura {done}/{total}" if total else "Captura 0/0"
    cw, _ = _tsize(count, 0.44, 1)
    _text(frame, count, (col_right - cw, y1 + 24), 0.44, _C_INK, 1)

    # Big, plain-language instruction — the one thing the user must read.
    cv2.circle(frame, (tx + 6, y1 + 52), 6, accent, -1, lineType=cv2.LINE_AA)
    instr = _enroll_instruction(state)
    instr_scale = _fit_scale(instr, max_w=col_right - (tx + 20), start=0.6, min_scale=0.44)
    _text(frame, instr, (tx + 20, y1 + 58), instr_scale, _C_INK, 1, cv2.FONT_HERSHEY_DUPLEX)

    bar_x1, bar_x2, bar_y, bar_h = tx, col_right, y1 + 74, 8
    _fill_rounded(frame, bar_x1, bar_y, bar_x2, bar_y + bar_h, _C_TRACK, bar_h // 2)
    if total > 0:
        frac = max(0.0, min(1.0, done / total))
        fill_x = bar_x1 + int((bar_x2 - bar_x1) * frac)
        if fill_x > bar_x1 + bar_h:
            _fill_rounded(frame, bar_x1, bar_y, fill_x, bar_y + bar_h, _C_GREEN, bar_h // 2)

    completed = (state.gate_reason or "").startswith("Objetivo completado")
    hint = "Puedes cerrar con Q" if completed else "La captura es automatica"
    _text(frame, hint, (tx, y1 + 104), 0.4, _C_MUTED, 1, shadow=False)

    _draw_pose_map(frame, cx=x2 - 50, cy=y1 + 62, state=state, accent=accent)


def _draw_toast(frame, state: DisplayState) -> None:
    if not (state.message and time.time() <= state.message_until_ts):
        return
    h, w = frame.shape[:2]
    msg = state.message
    accent = _C_GREEN if msg.startswith("Saludo:") else _C_ACCENT
    scale, thickness = 0.58, 1
    tw, th = _tsize(msg, scale, thickness, cv2.FONT_HERSHEY_DUPLEX)
    pad, dot, gap = 16, 5, 10
    bw = pad + dot * 2 + gap + tw + pad
    x1 = (w - bw) // 2
    x2 = x1 + bw
    y2 = h - 64
    y1 = y2 - 38
    _panel(frame, x1, y1, x2, y2, color=(18, 16, 14), alpha=0.82, radius=19)
    cv2.rectangle(frame, (x1, y1), (x1 + 3, y2), accent, -1)
    cy = (y1 + y2) // 2
    cv2.circle(frame, (x1 + pad + dot, cy), dot, accent, -1, lineType=cv2.LINE_AA)
    _text(frame, msg, (x1 + pad + dot * 2 + gap, cy + th // 2), scale, _C_INK,
          thickness, cv2.FONT_HERSHEY_DUPLEX, shadow=False)


def _draw_controls(frame, state: DisplayState) -> None:
    h, w = frame.shape[:2]
    # "E enrolar" is only shown when the runtime was started with an enrollment
    # target; otherwise the key is inert and the hint would be misleading.
    text = "Q  salir      E  enrolar" if state.enroll_enabled else "Q  salir"
    tw, _ = _tsize(text, 0.5, 1)
    _text(frame, text, (w - tw - 16, h - 16), 0.5, _C_MUTED, 1)


def _draw_overlay(frame, state: DisplayState, show_landmarks: bool) -> None:
    _draw_face_overlays(frame, state.face_overlays, show_landmarks=show_landmarks)
    _draw_landmarks(frame, state.landmarks)

    viewfinder_color = _C_ACCENT
    if state.gate_status is not None:
        viewfinder_color = _GATE_COLORS.get(state.gate_status, _C_ACCENT)
    _draw_viewfinder(frame, viewfinder_color)

    if state.gate_status == "yellow" and state.gate_target_bucket in ("left", "right", "up", "down"):
        _draw_direction_cue(frame, state.gate_target_bucket, viewfinder_color)

    _draw_header(frame, state)
    _draw_enroll_panel(frame, state)
    _draw_toast(frame, state)
    _draw_controls(frame, state)


def _draw_face_overlays(frame, overlays: list[FaceOverlay], show_landmarks: bool) -> None:
    for overlay in overlays:
        if show_landmarks:
            _draw_landmarks(frame, overlay.landmarks)
        x1, y1, x2, y2 = overlay.bbox
        color = _face_color(overlay)
        dim = tuple(int(c * 0.55) for c in color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), dim, 1, lineType=cv2.LINE_AA)
        _corner_brackets(frame, (x1, y1, x2, y2), color, thickness=2)
        _label_chip(frame, x1, y1 - 8, overlay.label, color)


def _is_in_face_ratio_range(face_ratio: float | None, min_face_ratio: float) -> bool:
    if min_face_ratio <= 0.0:
        return True
    if face_ratio is None:
        return False
    return float(face_ratio) >= float(min_face_ratio)


def _update_landmarks(frame, services, state: DisplayState, requested_points: int) -> None:
    max_points = max(10, requested_points)
    encoder = getattr(services.recognition_service, "_encoder", None)
    extract = getattr(encoder, "extract_landmarks", None)
    if not callable(extract):
        state.landmarks = []
        if not state.landmarks_warning_shown:
            state.landmarks_warning_shown = True
            state.message = "Landmarks require ELECCIA_ENCODER_BACKEND=insightface"
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
