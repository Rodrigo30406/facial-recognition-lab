import pytest

from eleccia_vision.application.recognition import RecognitionService
from eleccia_vision.config import Settings
from eleccia_vision.domain.entities import FaceRecord
from eleccia_vision.infrastructure.faiss_search import FaissSearcher
from eleccia_vision.infrastructure.inmemory_repo import InMemoryFaceRepository


class MapEncoder:
    def __init__(self, mapping: dict[bytes, list[float]]) -> None:
        self._mapping = mapping

    def encode(self, image_bytes: bytes) -> list[float]:
        return self._mapping[image_bytes]


class FailingEncoder:
    def encode(self, image_bytes: bytes) -> list[float]:
        raise ValueError("No face detected in image payload")


@pytest.fixture()
def require_faiss() -> None:
    pytest.importorskip("faiss")


def test_recognition_known_person(require_faiss: None) -> None:
    repo = InMemoryFaceRepository()
    repo.upsert(FaceRecord(person_id="alice", embedding=[1.0, 0.0, 0.0]))
    repo.upsert(FaceRecord(person_id="bob", embedding=[0.0, 1.0, 0.0]))

    service = RecognitionService(
        encoder=MapEncoder({b"probe": [0.98, 0.02, 0.0]}),
        face_repository=repo,
        searcher=FaissSearcher(),
        settings=Settings(recognition_threshold=0.5, recognition_margin=0.05, recognition_top_k=2),
    )

    result = service.recognize(b"probe")

    assert result.decision == "known_person"
    assert result.matched is True
    assert result.person_id == "alice"
    assert result.top1 is not None
    assert result.top1.score >= 0.5


def test_recognition_ambiguous_match(require_faiss: None) -> None:
    repo = InMemoryFaceRepository()
    repo.upsert(FaceRecord(person_id="alice", embedding=[1.0, 0.0]))
    repo.upsert(FaceRecord(person_id="bob", embedding=[0.95, 0.05]))

    service = RecognitionService(
        encoder=MapEncoder({b"probe": [1.0, 0.0]}),
        face_repository=repo,
        searcher=FaissSearcher(),
        settings=Settings(recognition_threshold=0.5, recognition_margin=0.1, recognition_top_k=2),
    )

    result = service.recognize(b"probe")

    assert result.decision == "ambiguous_match"
    assert result.matched is False
    assert result.person_id is None
    assert result.top1 is not None and result.top2 is not None


def test_recognition_unknown_when_below_threshold(require_faiss: None) -> None:
    repo = InMemoryFaceRepository()
    repo.upsert(FaceRecord(person_id="alice", embedding=[1.0, 0.0, 0.0]))

    service = RecognitionService(
        encoder=MapEncoder({b"probe": [0.0, 1.0, 0.0]}),
        face_repository=repo,
        searcher=FaissSearcher(),
        settings=Settings(recognition_threshold=0.8, recognition_margin=0.05, recognition_top_k=2),
    )

    result = service.recognize(b"probe")

    assert result.decision == "unknown_person"
    assert result.matched is False
    assert result.person_id is None


def test_recognition_unknown_when_no_face_detected(require_faiss: None) -> None:
    repo = InMemoryFaceRepository()
    repo.upsert(FaceRecord(person_id="alice", embedding=[1.0, 0.0, 0.0]))

    service = RecognitionService(
        encoder=FailingEncoder(),
        face_repository=repo,
        searcher=FaissSearcher(),
        settings=Settings(recognition_threshold=0.5, recognition_margin=0.05, recognition_top_k=2),
    )

    result = service.recognize(b"probe")

    assert result.decision == "unknown_person"
    assert result.matched is False
    assert result.person_id is None
