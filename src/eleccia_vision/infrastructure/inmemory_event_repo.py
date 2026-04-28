from datetime import datetime, timezone

from eleccia_vision.domain.entities import RecognitionEventRecord
from eleccia_vision.domain.interfaces import RecognitionEventRepository


class InMemoryRecognitionEventRepository(RecognitionEventRepository):
    def __init__(self) -> None:
        self._events: list[RecognitionEventRecord] = []
        self._next_id = 1

    def create(self, event: RecognitionEventRecord) -> RecognitionEventRecord:
        saved = RecognitionEventRecord(
            camera_id=event.camera_id,
            track_id=event.track_id,
            decision=event.decision,
            top1_person_id=event.top1_person_id,
            top1_score=event.top1_score,
            top2_person_id=event.top2_person_id,
            top2_score=event.top2_score,
            snapshot_path=event.snapshot_path,
            notes=event.notes,
            event_id=self._next_id,
            created_at=datetime.now(timezone.utc),
        )
        self._events.append(saved)
        self._next_id += 1
        return saved

    def list_recent(
        self,
        limit: int = 100,
        decision: str | None = None,
        camera_id: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[RecognitionEventRecord]:
        events = list(reversed(self._events))

        def match(event: RecognitionEventRecord) -> bool:
            if decision is not None and event.decision != decision:
                return False
            if camera_id is not None and event.camera_id != camera_id:
                return False
            if date_from is not None and event.created_at is not None and event.created_at < date_from:
                return False
            if date_to is not None and event.created_at is not None and event.created_at > date_to:
                return False
            return True

        filtered = [event for event in events if match(event)]
        return filtered[:limit]
