from dataclasses import dataclass


@dataclass(frozen=True)
class FaceRecord:
    person_id: str
    embedding: list[float]


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    person_id: str | None
    distance: float
