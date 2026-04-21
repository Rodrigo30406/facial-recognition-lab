#!/usr/bin/env python3
"""Run a local webcam demo for facial recognition.

Usage:
  PYTHONPATH=src python scripts/run_camera_demo.py --camera-index 0 --camera-id cam-01

Controls:
  q: quit
  e: enroll current frame (requires --enroll-person-id)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from facial_recognition.application.enrollment import InvalidImageError, PersonNotFoundError
from facial_recognition.bootstrap import build_services
from facial_recognition.domain.entities import RecognitionResult


@dataclass
class DisplayState:
    result: RecognitionResult | None = None
    latency_ms: float | None = None
    message: str = ""
    message_until_ts: float = 0.0
    landmarks: list[tuple[int, int]] = field(default_factory=list)
    landmarks_warning_shown: bool = False


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
        default=20,
        help="Maximum number of landmarks to draw (minimum effective value: 10)",
    )
    parser.add_argument(
        "--landmarks-every",
        type=int,
        default=2,
        help="Update landmarks every N frames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    services = build_services()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera_index}")

    print("Camera demo started")
    print("- Press 'q' to quit")
    if args.enroll_person_id:
        print(f"- Press 'e' to enroll current frame as '{args.enroll_person_id}'")
    if args.show_landmarks:
        print("- Landmark overlay enabled")

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
                _run_recognition(frame, services, state, args.camera_id)
            if args.show_landmarks and frame_idx % max(1, args.landmarks_every) == 0:
                _update_landmarks(frame, services, state, args.landmarks_max_points)

            _draw_overlay(frame, state)
            cv2.imshow(args.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("e") and args.enroll_person_id:
                _enroll_current_frame(frame, services, state, args.enroll_person_id, args.camera_id)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _run_recognition(frame, services, state: DisplayState, camera_id: str) -> None:
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


def _enroll_current_frame(frame, services, state: DisplayState, person_id: str, camera_id: str) -> None:
    payload = _frame_to_jpeg_bytes(frame)
    if payload is None:
        state.message = "Failed to encode frame for enrollment"
        state.message_until_ts = time.time() + 2.5
        return

    try:
        sample = services.enrollment_service.enroll_image(
            person_id=person_id,
            image_bytes=payload,
            capture_type="operational",
            camera_id=camera_id,
        )
        state.message = f"Enrolled {person_id} (q={sample.quality_score:.2f})"
    except PersonNotFoundError:
        state.message = f"Person '{person_id}' not found"
    except InvalidImageError:
        state.message = "Invalid image for enrollment"

    state.message_until_ts = time.time() + 2.5


def _frame_to_jpeg_bytes(frame) -> bytes | None:
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return encoded.tobytes()


def _draw_overlay(frame, state: DisplayState) -> None:
    _draw_landmarks(frame, state.landmarks)
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
    cv2.putText(frame, "q: quit | e: enroll", (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    if state.message and time.time() <= state.message_until_ts:
        cv2.putText(frame, state.message, (10, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)


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
