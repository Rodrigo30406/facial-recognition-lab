import json
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from facial_recognition.domain.entities import (
    FaceRecord,
    FaceSampleRecord,
    PersonRecord,
    RecognitionEventRecord,
)
from facial_recognition.domain.interfaces import (
    FaceRepository,
    FaceSampleRepository,
    PersonRepository,
    RecognitionEventRepository,
)
from facial_recognition.infrastructure.sqlalchemy_models import (
    FaceEmbeddingModel,
    FaceSampleModel,
    PersonModel,
    RecognitionEventModel,
)


class SQLitePersonRepository(PersonRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(self, person: PersonRecord) -> bool:
        with self._session_factory() as session:
            existing = session.get(PersonModel, person.person_id)
            if existing is not None:
                return False
            session.add(
                PersonModel(
                    person_id=person.person_id,
                    full_name=person.full_name,
                    sex=person.sex,
                )
            )
            session.commit()
            return True

    def get(self, person_id: str) -> PersonRecord | None:
        with self._session_factory() as session:
            row = session.get(PersonModel, person_id)
            if row is None:
                return None
            return PersonRecord(person_id=row.person_id, full_name=row.full_name, sex=row.sex)

    def list_all(self) -> list[PersonRecord]:
        with self._session_factory() as session:
            rows = session.scalars(select(PersonModel).order_by(PersonModel.created_at.asc())).all()
            return [PersonRecord(person_id=r.person_id, full_name=r.full_name, sex=r.sex) for r in rows]


class SQLiteFaceRepository(FaceRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def upsert(self, record: FaceRecord) -> None:
        with self._session_factory() as session:
            session.execute(
                delete(FaceEmbeddingModel).where(FaceEmbeddingModel.person_id == record.person_id)
            )
            session.add(
                FaceEmbeddingModel(
                    person_id=record.person_id,
                    embedding_json=json.dumps(record.embedding),
                )
            )
            session.commit()

    def list_all(self) -> list[FaceRecord]:
        with self._session_factory() as session:
            rows = session.scalars(select(FaceEmbeddingModel)).all()
            return [
                FaceRecord(person_id=r.person_id, embedding=json.loads(r.embedding_json)) for r in rows
            ]


class SQLiteFaceSampleRepository(FaceSampleRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(self, sample: FaceSampleRecord) -> None:
        with self._session_factory() as session:
            session.add(
                FaceSampleModel(
                    person_id=sample.person_id,
                    image_path=sample.image_path,
                    capture_type=sample.capture_type,
                    camera_id=sample.camera_id,
                    quality_score=sample.quality_score,
                    pose_yaw=sample.pose_yaw,
                    pose_pitch=sample.pose_pitch,
                    pose_roll=sample.pose_roll,
                )
            )
            session.commit()


class SQLiteRecognitionEventRepository(RecognitionEventRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(self, event: RecognitionEventRecord) -> RecognitionEventRecord:
        with self._session_factory() as session:
            row = RecognitionEventModel(
                camera_id=event.camera_id,
                track_id=event.track_id,
                decision=event.decision,
                top1_person_id=event.top1_person_id,
                top1_score=event.top1_score,
                top2_person_id=event.top2_person_id,
                top2_score=event.top2_score,
                snapshot_path=event.snapshot_path,
                notes=event.notes,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return _to_event_record(row)

    def list_recent(
        self,
        limit: int = 100,
        decision: str | None = None,
        camera_id: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[RecognitionEventRecord]:
        with self._session_factory() as session:
            stmt = select(RecognitionEventModel)
            if decision is not None:
                stmt = stmt.where(RecognitionEventModel.decision == decision)
            if camera_id is not None:
                stmt = stmt.where(RecognitionEventModel.camera_id == camera_id)
            if date_from is not None:
                stmt = stmt.where(RecognitionEventModel.created_at >= date_from)
            if date_to is not None:
                stmt = stmt.where(RecognitionEventModel.created_at <= date_to)

            rows = session.scalars(
                stmt.order_by(RecognitionEventModel.created_at.desc()).limit(limit)
            ).all()
            return [_to_event_record(row) for row in rows]


def _to_event_record(row: RecognitionEventModel) -> RecognitionEventRecord:
    return RecognitionEventRecord(
        camera_id=row.camera_id,
        track_id=row.track_id,
        decision=row.decision,
        top1_person_id=row.top1_person_id,
        top1_score=row.top1_score,
        top2_person_id=row.top2_person_id,
        top2_score=row.top2_score,
        snapshot_path=row.snapshot_path,
        notes=row.notes,
        event_id=row.event_id,
        created_at=row.created_at,
    )
