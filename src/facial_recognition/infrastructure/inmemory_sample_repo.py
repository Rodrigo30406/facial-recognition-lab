from facial_recognition.domain.entities import FaceSampleRecord
from facial_recognition.domain.interfaces import FaceSampleRepository


class InMemoryFaceSampleRepository(FaceSampleRepository):
    def __init__(self) -> None:
        self._samples: list[FaceSampleRecord] = []

    def create(self, sample: FaceSampleRecord) -> None:
        self._samples.append(sample)
