from pathlib import Path

import cv2
import numpy as np

from facial_recognition.application.enrollment import (
    EnrollmentService,
    InvalidImageError,
    PersonNotFoundError,
)
from facial_recognition.domain.entities import PersonRecord
from facial_recognition.infrastructure.dummy_encoder import DummyFaceEncoder
from facial_recognition.infrastructure.inmemory_person_repo import InMemoryPersonRepository
from facial_recognition.infrastructure.inmemory_repo import InMemoryFaceRepository
from facial_recognition.infrastructure.inmemory_sample_repo import InMemoryFaceSampleRepository


def _make_image_bytes() -> bytes:
    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("could not encode test image")
    return encoded.tobytes()


def test_enroll_image_persists_sample_and_embedding(tmp_path: Path) -> None:
    person_repo = InMemoryPersonRepository()
    person_repo.create(PersonRecord(person_id="alice", full_name="Alice Doe"))

    face_repo = InMemoryFaceRepository()
    sample_repo = InMemoryFaceSampleRepository()

    service = EnrollmentService(
        person_repository=person_repo,
        face_repository=face_repo,
        sample_repository=sample_repo,
        encoder=DummyFaceEncoder(),
        sample_storage_dir=tmp_path.as_posix(),
    )

    sample = service.enroll_image(person_id="alice", image_bytes=_make_image_bytes())

    assert sample.person_id == "alice"
    assert Path(sample.image_path).exists()
    assert 0.0 <= sample.quality_score <= 1.0
    assert len(face_repo.list_all()) == 1


def test_enroll_image_fails_for_unknown_person(tmp_path: Path) -> None:
    service = EnrollmentService(
        person_repository=InMemoryPersonRepository(),
        face_repository=InMemoryFaceRepository(),
        sample_repository=InMemoryFaceSampleRepository(),
        encoder=DummyFaceEncoder(),
        sample_storage_dir=tmp_path.as_posix(),
    )

    try:
        service.enroll_image(person_id="ghost", image_bytes=_make_image_bytes())
    except PersonNotFoundError:
        return

    raise AssertionError("Expected PersonNotFoundError")


def test_enroll_image_fails_for_invalid_payload(tmp_path: Path) -> None:
    person_repo = InMemoryPersonRepository()
    person_repo.create(PersonRecord(person_id="alice", full_name="Alice Doe"))

    service = EnrollmentService(
        person_repository=person_repo,
        face_repository=InMemoryFaceRepository(),
        sample_repository=InMemoryFaceSampleRepository(),
        encoder=DummyFaceEncoder(),
        sample_storage_dir=tmp_path.as_posix(),
    )

    try:
        service.enroll_image(person_id="alice", image_bytes=b"not-an-image")
    except InvalidImageError:
        return

    raise AssertionError("Expected InvalidImageError")
