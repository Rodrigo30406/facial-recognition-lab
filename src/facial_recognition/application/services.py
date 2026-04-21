from math import sqrt

from facial_recognition.config import Settings
from facial_recognition.domain.entities import FaceRecord, MatchResult
from facial_recognition.domain.interfaces import FaceEncoder, FaceRepository


class FaceRecognitionService:
    """Use-case service independent of concrete frameworks."""

    def __init__(
        self,
        encoder: FaceEncoder,
        repository: FaceRepository,
        settings: Settings,
    ) -> None:
        self._encoder = encoder
        self._repository = repository
        self._settings = settings

    def enroll(self, person_id: str, image_bytes: bytes) -> None:
        embedding = self._encoder.encode(image_bytes)
        self._repository.upsert(FaceRecord(person_id=person_id, embedding=embedding))

    def match(self, image_bytes: bytes) -> MatchResult:
        probe = self._encoder.encode(image_bytes)
        candidates = self._repository.list_all()
        if not candidates:
            return MatchResult(matched=False, person_id=None, distance=1.0)

        best_person = None
        best_distance = float("inf")
        for record in candidates:
            dist = _l2_distance(probe, record.embedding)
            if dist < best_distance:
                best_distance = dist
                best_person = record.person_id

        matched = best_distance <= self._settings.similarity_threshold
        return MatchResult(
            matched=matched,
            person_id=best_person if matched else None,
            distance=best_distance,
        )


def _l2_distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding sizes must match")
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))
