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
from dataclasses import dataclass

import cv2

from facial_recognition.application.enrollment import InvalidImageError, PersonNotFoundError
from facial_recognition.bootstrap import build_services
from facial_recognition.domain.entities import RecognitionResult


@dataclass
class DisplayState:
    result: RecognitionResult | None = None
    latency_ms: float | None = None
    message: str = ""
    message_until_ts: float = 0.0


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
    result = services.recognition_service.recognize(payload)
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


if __name__ == "__main__":
    main()
