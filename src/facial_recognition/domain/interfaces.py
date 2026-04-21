from typing import Protocol

from facial_recognition.domain.entities import FaceRecord


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
