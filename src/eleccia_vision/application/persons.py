from eleccia_vision.domain.entities import PersonRecord
from eleccia_vision.domain.interfaces import PersonRepository


class PersonAlreadyExistsError(ValueError):
    pass


class InvalidPersonSexError(ValueError):
    pass


class PersonService:
    def __init__(self, repository: PersonRepository) -> None:
        self._repository = repository

    def create_person(
        self,
        person_id: str,
        full_name: str,
        sex: str | None = None,
    ) -> PersonRecord:
        person = PersonRecord(
            person_id=person_id,
            full_name=full_name,
            sex=_normalize_person_sex(sex),
        )
        created = self._repository.create(person)
        if not created:
            raise PersonAlreadyExistsError(f"Person '{person_id}' already exists")
        return person

    def get_person(self, person_id: str) -> PersonRecord | None:
        return self._repository.get(person_id)

    def list_people(self) -> list[PersonRecord]:
        return self._repository.list_all()


def _normalize_person_sex(value: str | None) -> str | None:
    if value is None:
        return None

    raw = value.strip().lower()
    if not raw:
        return None

    male_values = {"male", "m", "man", "masculino", "hombre"}
    female_values = {"female", "f", "woman", "femenino", "mujer"}
    other_values = {"other", "x", "nb", "nonbinary", "no_binario", "no-binario", "otro"}

    if raw in male_values:
        return "male"
    if raw in female_values:
        return "female"
    if raw in other_values:
        return "other"

    raise InvalidPersonSexError("Invalid sex value. Use: male, female, other")
