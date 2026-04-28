from eleccia_vision.application.persons import (
    InvalidPersonSexError,
    PersonAlreadyExistsError,
    PersonService,
)
from eleccia_vision.application.services import FaceRecognitionService
from eleccia_vision.config import Settings
from eleccia_vision.infrastructure.dummy_encoder import DummyFaceEncoder
from eleccia_vision.infrastructure.inmemory_person_repo import InMemoryPersonRepository
from eleccia_vision.infrastructure.inmemory_repo import InMemoryFaceRepository


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


def test_person_service_create_and_list() -> None:
    service = PersonService(repository=InMemoryPersonRepository())

    service.create_person("alice", "Alice Doe")
    service.create_person("bob", "Bob Roe", sex="femenino")
    people = service.list_people()

    assert [p.person_id for p in people] == ["alice", "bob"]
    assert [p.sex for p in people] == [None, "female"]


def test_person_service_duplicate_person() -> None:
    service = PersonService(repository=InMemoryPersonRepository())
    service.create_person("alice", "Alice Doe")

    try:
        service.create_person("alice", "Alice Clone")
    except PersonAlreadyExistsError:
        return

    raise AssertionError("Expected PersonAlreadyExistsError")


def test_person_service_invalid_sex() -> None:
    service = PersonService(repository=InMemoryPersonRepository())

    try:
        service.create_person("alice", "Alice Doe", sex="desconocido")
    except InvalidPersonSexError:
        return

    raise AssertionError("Expected InvalidPersonSexError")
