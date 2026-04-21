from facial_recognition.domain.entities import FaceRecord
from facial_recognition.domain.interfaces import FaceRepository


class InMemoryFaceRepository(FaceRepository):
    def __init__(self) -> None:
        self._records: dict[str, FaceRecord] = {}

    def upsert(self, record: FaceRecord) -> None:
        self._records[record.person_id] = record

    def list_all(self) -> list[FaceRecord]:
        return list(self._records.values())
