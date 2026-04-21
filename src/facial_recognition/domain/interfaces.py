from datetime import datetime
from typing import Protocol

from facial_recognition.domain.entities import (
    FaceRecord,
    FaceSampleRecord,
    PersonRecord,
    RecognitionEventRecord,
)


class FaceEncoder(Protocol):
    """Converts an image payload into an embedding vector."""

    def encode(self, image_bytes: bytes) -> list[float]:
        ...


class FaceRepository(Protocol):
    """Stores and retrieves embeddings."""

    def upsert(self, record: FaceRecord) -> None:
        ...

    def list_all(self) -> list[FaceRecord]:
        ...


class PersonRepository(Protocol):
    """Stores and retrieves people metadata."""

    def create(self, person: PersonRecord) -> bool:
        ...

    def get(self, person_id: str) -> PersonRecord | None:
        ...

    def list_all(self) -> list[PersonRecord]:
        ...


class FaceSampleRepository(Protocol):
    """Stores face sample metadata for enrollment traceability."""

    def create(self, sample: FaceSampleRecord) -> None:
        ...


class RecognitionEventRepository(Protocol):
    """Stores and retrieves recognition event history."""

    def create(self, event: RecognitionEventRecord) -> RecognitionEventRecord:
        ...

    def list_recent(
        self,
        limit: int = 100,
        decision: str | None = None,
        camera_id: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[RecognitionEventRecord]:
        ...
