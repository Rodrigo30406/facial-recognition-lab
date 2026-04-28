from eleccia_vision.domain.entities import FaceSampleRecord
from eleccia_vision.domain.interfaces import FaceSampleRepository


class InMemoryFaceSampleRepository(FaceSampleRepository):
    def __init__(self) -> None:
        self._samples: list[FaceSampleRecord] = []

    def create(self, sample: FaceSampleRecord) -> None:
        self._samples.append(sample)
