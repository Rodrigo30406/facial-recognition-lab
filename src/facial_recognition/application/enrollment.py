from __future__ import annotations

from pathlib import Path
from time import time_ns

import cv2
import numpy as np

from facial_recognition.domain.entities import FaceRecord, FaceSampleRecord
from facial_recognition.domain.interfaces import (
    FaceEncoder,
    FaceRepository,
    FaceSampleRepository,
    PersonRepository,
)


class PersonNotFoundError(ValueError):
    pass


class InvalidImageError(ValueError):
    pass


class EnrollmentService:
    def __init__(
        self,
        person_repository: PersonRepository,
        face_repository: FaceRepository,
        sample_repository: FaceSampleRepository,
        encoder: FaceEncoder,
        sample_storage_dir: str,
    ) -> None:
        self._person_repository = person_repository
        self._face_repository = face_repository
        self._sample_repository = sample_repository
        self._encoder = encoder
        self._sample_storage_dir = Path(sample_storage_dir)

    def enroll_image(
        self,
        person_id: str,
        image_bytes: bytes,
        capture_type: str = "operational",
        camera_id: str | None = None,
    ) -> FaceSampleRecord:
        person = self._person_repository.get(person_id)
        if person is None:
            raise PersonNotFoundError(f"Person '{person_id}' does not exist")

        image = _decode_image(image_bytes)
        quality_score = _quality_score(image)

        image_path = self._save_sample_image(person_id=person_id, image=image)
        sample = FaceSampleRecord(
            person_id=person_id,
            image_path=str(image_path),
            capture_type=capture_type,
            camera_id=camera_id,
            quality_score=quality_score,
            pose_yaw=None,
            pose_pitch=None,
            pose_roll=None,
        )
        self._sample_repository.create(sample)

        embedding = self._encoder.encode(image_bytes)
        self._face_repository.upsert(FaceRecord(person_id=person_id, embedding=embedding))
        return sample

    def _save_sample_image(self, person_id: str, image: np.ndarray) -> Path:
        person_dir = self._sample_storage_dir / person_id
        person_dir.mkdir(parents=True, exist_ok=True)

        file_path = person_dir / f"sample_{time_ns()}.jpg"
        ok = cv2.imwrite(file_path.as_posix(), image)
        if not ok:
            raise InvalidImageError("Could not save sample image")
        return file_path


def _decode_image(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("Uploaded payload is not a valid image")
    return image


def _quality_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    sharp_norm = min(sharpness / 1000.0, 1.0)
    bright_norm = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    return round((0.7 * sharp_norm) + (0.3 * bright_norm), 4)
