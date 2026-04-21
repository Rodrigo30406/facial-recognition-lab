from facial_recognition.application.services import FaceRecognitionService
from facial_recognition.config import Settings
from facial_recognition.infrastructure.dummy_encoder import DummyFaceEncoder
from facial_recognition.infrastructure.inmemory_repo import InMemoryFaceRepository


def test_enroll_and_match_same_image() -> None:
    service = FaceRecognitionService(
        encoder=DummyFaceEncoder(),
        repository=InMemoryFaceRepository(),
        settings=Settings(similarity_threshold=0.0001),
    )

    image = b"fake-image-content"
    service.enroll(person_id="alice", image_bytes=image)
    result = service.match(image_bytes=image)

    assert result.matched is True
    assert result.person_id == "alice"
    assert result.distance == 0.0
