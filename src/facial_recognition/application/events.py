from datetime import datetime

from facial_recognition.domain.entities import RecognitionEventRecord, RecognitionResult
from facial_recognition.domain.interfaces import RecognitionEventRepository


class RecognitionEventService:
    def __init__(self, repository: RecognitionEventRepository) -> None:
        self._repository = repository

    def record_from_result(
        self,
        result: RecognitionResult,
        camera_id: str | None = None,
        track_id: str | None = None,
        snapshot_path: str | None = None,
        notes: str | None = None,
    ) -> RecognitionEventRecord:
        event = RecognitionEventRecord(
            camera_id=camera_id,
            track_id=track_id,
            decision=result.decision,
            top1_person_id=result.top1.person_id if result.top1 is not None else None,
            top1_score=result.top1.score if result.top1 is not None else None,
            top2_person_id=result.top2.person_id if result.top2 is not None else None,
            top2_score=result.top2.score if result.top2 is not None else None,
            snapshot_path=snapshot_path,
            notes=notes,
        )
        return self._repository.create(event)

    def list_events(
        self,
        limit: int = 100,
        decision: str | None = None,
        camera_id: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[RecognitionEventRecord]:
        return self._repository.list_recent(
            limit=limit,
            decision=decision,
            camera_id=camera_id,
            date_from=date_from,
            date_to=date_to,
        )
