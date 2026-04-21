from facial_recognition.domain.entities import PersonRecord
from facial_recognition.domain.interfaces import PersonRepository


class PersonAlreadyExistsError(ValueError):
    pass


class PersonService:
    def __init__(self, repository: PersonRepository) -> None:
        self._repository = repository

    def create_person(self, person_id: str, full_name: str) -> PersonRecord:
        person = PersonRecord(person_id=person_id, full_name=full_name)
        created = self._repository.create(person)
        if not created:
            raise PersonAlreadyExistsError(f"Person '{person_id}' already exists")
        return person

    def get_person(self, person_id: str) -> PersonRecord | None:
        return self._repository.get(person_id)

    def list_people(self) -> list[PersonRecord]:
        return self._repository.list_all()
