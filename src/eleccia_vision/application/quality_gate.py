from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

AngleBucket = Literal["center", "left", "right", "up", "down"]

_ANGLE_ORDER: tuple[AngleBucket, ...] = ("center", "left", "right", "up", "down")
_ANGLE_LABELS: dict[AngleBucket, str] = {
    "center": "frente",
    "left": "izquierda",
    "right": "derecha",
    "up": "arriba",
    "down": "abajo",
}


@dataclass(frozen=True)
class FaceObservation:
    bbox: tuple[float, float, float, float]
    det_score: float
    yaw: float | None
    pitch: float | None
    roll: float | None


@dataclass(frozen=True)
class QualityGateThresholds:
    min_det_score: float = 0.60
    min_face_ratio: float = 0.08
    min_sharpness: float = 90.0
    min_brightness: float = 50.0
    max_brightness: float = 210.0
    max_abs_yaw: float = 55.0
    max_abs_pitch: float = 40.0
    max_abs_roll: float = 40.0


@dataclass(frozen=True)
class QualityGateAssessment:
    status: Literal["red", "yellow", "green"]
    reason: str
    current_bucket: AngleBucket | None
    target_bucket: AngleBucket | None
    face_ratio: float | None
    sharpness: float | None
    brightness: float | None
    yaw: float | None
    pitch: float | None
    roll: float | None


def build_angle_plan(target_samples: int) -> dict[AngleBucket, int]:
    if target_samples < 1:
        raise ValueError("target_samples must be >= 1")

    plan: dict[AngleBucket, int] = {bucket: 0 for bucket in _ANGLE_ORDER}
    for bucket in _ANGLE_ORDER:
        if sum(plan.values()) >= target_samples:
            break
        plan[bucket] += 1

    i = 0
    while sum(plan.values()) < target_samples:
        plan[_ANGLE_ORDER[i % len(_ANGLE_ORDER)]] += 1
        i += 1
    return plan


def next_target_bucket(
    captured_by_bucket: dict[AngleBucket, int], plan_by_bucket: dict[AngleBucket, int]
) -> AngleBucket | None:
    remaining: list[tuple[int, int, AngleBucket]] = []
    for idx, bucket in enumerate(_ANGLE_ORDER):
        deficit = plan_by_bucket.get(bucket, 0) - captured_by_bucket.get(bucket, 0)
        if deficit > 0:
            remaining.append((deficit, -idx, bucket))
    if not remaining:
        return None
    remaining.sort(reverse=True)
    return remaining[0][2]


def classify_bucket(yaw: float | None, pitch: float | None) -> AngleBucket:
    if pitch is not None:
        if pitch <= -12.0:
            return "up"
        if pitch >= 12.0:
            return "down"
    if yaw is not None:
        if yaw <= -15.0:
            return "left"
        if yaw >= 15.0:
            return "right"
    return "center"


def bucket_instruction(bucket: AngleBucket | None) -> str:
    if bucket is None:
        return "completado"
    return _ANGLE_LABELS[bucket]


def evaluate_quality_gate(
    frame: np.ndarray,
    observation: FaceObservation | None,
    thresholds: QualityGateThresholds,
    captured_by_bucket: dict[AngleBucket, int],
    plan_by_bucket: dict[AngleBucket, int],
) -> QualityGateAssessment:
    target_bucket = next_target_bucket(captured_by_bucket, plan_by_bucket)
    if observation is None:
        return QualityGateAssessment(
            status="red",
            reason="No se detecta rostro",
            current_bucket=None,
            target_bucket=target_bucket,
            face_ratio=None,
            sharpness=None,
            brightness=None,
            yaw=None,
            pitch=None,
            roll=None,
        )

    x1, y1, x2, y2 = observation.bbox
    h, w = frame.shape[:2]
    face_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(max(1, w * h))
    crop = _safe_face_crop(frame, observation.bbox)
    sharpness = _sharpness(crop) if crop is not None else 0.0
    brightness = _brightness(crop) if crop is not None else 0.0
    yaw = observation.yaw
    pitch = observation.pitch
    roll = observation.roll
    current_bucket = classify_bucket(yaw=yaw, pitch=pitch)

    if observation.det_score < thresholds.min_det_score:
        return _reject(
            "Acercate y mira a camara (deteccion baja)",
            target_bucket,
            current_bucket,
            face_ratio,
            sharpness,
            brightness,
            yaw,
            pitch,
            roll,
        )
    if face_ratio < thresholds.min_face_ratio:
        return _reject(
            "Acercate un poco mas",
            target_bucket,
            current_bucket,
            face_ratio,
            sharpness,
            brightness,
            yaw,
            pitch,
            roll,
        )
    if sharpness < thresholds.min_sharpness:
        return _reject(
            "Quedate quieto (imagen borrosa)",
            target_bucket,
            current_bucket,
            face_ratio,
            sharpness,
            brightness,
            yaw,
            pitch,
            roll,
        )
    if brightness < thresholds.min_brightness:
        return _reject(
            "Falta luz",
            target_bucket,
            current_bucket,
            face_ratio,
            sharpness,
            brightness,
            yaw,
            pitch,
            roll,
        )
    if brightness > thresholds.max_brightness:
        return _reject(
            "Demasiada luz",
            target_bucket,
            current_bucket,
            face_ratio,
            sharpness,
            brightness,
            yaw,
            pitch,
            roll,
        )

    if yaw is not None and abs(yaw) > thresholds.max_abs_yaw:
        return _reject(
            "Giro extremo, vuelve un poco",
            target_bucket,
            current_bucket,
            face_ratio,
            sharpness,
            brightness,
            yaw,
            pitch,
            roll,
        )
    if pitch is not None and abs(pitch) > thresholds.max_abs_pitch:
        return _reject(
            "Inclinacion extrema, vuelve un poco",
            target_bucket,
            current_bucket,
            face_ratio,
            sharpness,
            brightness,
            yaw,
            pitch,
            roll,
        )
    if roll is not None and abs(roll) > thresholds.max_abs_roll:
        return _reject(
            "Endereza un poco la cabeza",
            target_bucket,
            current_bucket,
            face_ratio,
            sharpness,
            brightness,
            yaw,
            pitch,
            roll,
        )

    if target_bucket is not None and current_bucket != target_bucket:
        return QualityGateAssessment(
            status="yellow",
            reason=f"Mueve a: {bucket_instruction(target_bucket)}",
            current_bucket=current_bucket,
            target_bucket=target_bucket,
            face_ratio=face_ratio,
            sharpness=sharpness,
            brightness=brightness,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )

    return QualityGateAssessment(
        status="green",
        reason="Toma valida",
        current_bucket=current_bucket,
        target_bucket=target_bucket,
        face_ratio=face_ratio,
        sharpness=sharpness,
        brightness=brightness,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
    )


def _reject(
    reason: str,
    target_bucket: AngleBucket | None,
    current_bucket: AngleBucket | None,
    face_ratio: float | None,
    sharpness: float | None,
    brightness: float | None,
    yaw: float | None,
    pitch: float | None,
    roll: float | None,
) -> QualityGateAssessment:
    return QualityGateAssessment(
        status="red",
        reason=reason,
        current_bucket=current_bucket,
        target_bucket=target_bucket,
        face_ratio=face_ratio,
        sharpness=sharpness,
        brightness=brightness,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
    )


def _safe_face_crop(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())
