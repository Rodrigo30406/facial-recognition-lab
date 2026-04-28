from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from facial_recognition.application.consistency import RecognitionConsistencyService
from facial_recognition.application.enrollment import EnrollmentService
from facial_recognition.application.events import RecognitionEventService
from facial_recognition.application.persons import PersonService
from facial_recognition.application.recognition import RecognitionService
from facial_recognition.application.services import FaceRecognitionService
from facial_recognition.config import Settings
from facial_recognition.infrastructure.dummy_encoder import DummyFaceEncoder
from facial_recognition.infrastructure.insightface_encoder import InsightFaceEncoder
from facial_recognition.infrastructure.faiss_search import FaissSearcher
from facial_recognition.infrastructure.sqlalchemy_models import Base
from facial_recognition.infrastructure.sqlite_repos import (
    SQLiteFaceRepository,
    SQLiteFaceSampleRepository,
    SQLitePersonRepository,
    SQLiteRecognitionEventRepository,
)


@dataclass(frozen=True)
class ServiceContainer:
    face_service: FaceRecognitionService
    person_service: PersonService
    enrollment_service: EnrollmentService
    recognition_service: RecognitionService
    recognition_consistency_service: RecognitionConsistencyService
    recognition_event_service: RecognitionEventService


def build_services(settings: Settings | None = None) -> ServiceContainer:
    cfg = settings or Settings.from_env()
    _prepare_sqlite_path(cfg.database_url)
    Path(cfg.sample_storage_dir).mkdir(parents=True, exist_ok=True)

    engine = create_engine(cfg.database_url, future=True)
    Base.metadata.create_all(engine)
    _apply_sqlite_compat_migrations(engine=engine, database_url=cfg.database_url)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    encoder = _build_encoder(cfg)
    person_repo = SQLitePersonRepository(session_factory)
    face_repo = SQLiteFaceRepository(session_factory)
    sample_repo = SQLiteFaceSampleRepository(session_factory)
    event_repo = SQLiteRecognitionEventRepository(session_factory)

    face_service = FaceRecognitionService(encoder=encoder, repository=face_repo, settings=cfg)
    person_service = PersonService(repository=person_repo)
    enrollment_service = EnrollmentService(
        person_repository=person_repo,
        face_repository=face_repo,
        sample_repository=sample_repo,
        encoder=encoder,
        sample_storage_dir=cfg.sample_storage_dir,
    )
    recognition_service = RecognitionService(
        encoder=encoder,
        face_repository=face_repo,
        searcher=FaissSearcher(),
        settings=cfg,
    )
    recognition_consistency_service = RecognitionConsistencyService(
        enabled=cfg.temporal_consistency_enabled,
        min_consistent_frames=cfg.temporal_min_consistent_frames,
    )
    recognition_event_service = RecognitionEventService(repository=event_repo)
    return ServiceContainer(
        face_service=face_service,
        person_service=person_service,
        enrollment_service=enrollment_service,
        recognition_service=recognition_service,
        recognition_consistency_service=recognition_consistency_service,
        recognition_event_service=recognition_event_service,
    )


def _prepare_sqlite_path(database_url: str) -> None:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return
    raw_path = database_url[len(prefix) :]
    if raw_path == ":memory:":
        return
    db_path = Path(raw_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)


def _apply_sqlite_compat_migrations(*, engine, database_url: str) -> None:
    if not database_url.startswith("sqlite:///"):
        return

    with engine.begin() as conn:
        rows = conn.exec_driver_sql("PRAGMA table_info(persons)").fetchall()
        if not rows:
            return

        columns = {str(row[1]) for row in rows if len(row) > 1}
        if "sex" not in columns:
            conn.exec_driver_sql("ALTER TABLE persons ADD COLUMN sex VARCHAR(16)")


def _build_encoder(cfg: Settings) -> DummyFaceEncoder | InsightFaceEncoder:
    backend = cfg.encoder_backend.strip().lower()
    if backend == "dummy":
        return DummyFaceEncoder()
    if backend == "insightface":
        return InsightFaceEncoder(
            model_name=cfg.insightface_model_name,
            providers=cfg.insightface_providers,
            ctx_id=cfg.insightface_ctx_id,
            det_size=cfg.insightface_det_size,
        )
    raise ValueError(f"Unsupported encoder_backend '{cfg.encoder_backend}'")
