from facial_recognition.application.services import FaceRecognitionService
from facial_recognition.config import Settings
from facial_recognition.infrastructure.dummy_encoder import DummyFaceEncoder
from facial_recognition.infrastructure.inmemory_repo import InMemoryFaceRepository


def build_service() -> FaceRecognitionService:
    settings = Settings()
    encoder = DummyFaceEncoder()
    repository = InMemoryFaceRepository()
    return FaceRecognitionService(encoder=encoder, repository=repository, settings=settings)
