from eleccia_vision.domain.entities import PersonRecord
from eleccia_vision.domain.interfaces import PersonRepository


class InMemoryPersonRepository(PersonRepository):
    def __init__(self) -> None:
        self._people: dict[str, PersonRecord] = {}

    def create(self, person: PersonRecord) -> bool:
        if person.person_id in self._people:
            return False
        self._people[person.person_id] = person
        return True

    def get(self, person_id: str) -> PersonRecord | None:
        return self._people.get(person_id)

    def list_all(self) -> list[PersonRecord]:
        return list(self._people.values())
