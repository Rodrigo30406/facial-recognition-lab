from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FaceRecord:
    person_id: str
    embedding: list[float]


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    person_id: str | None
    distance: float


@dataclass(frozen=True)
class PersonRecord:
    person_id: str
    full_name: str


@dataclass(frozen=True)
class FaceSampleRecord:
    person_id: str
    image_path: str
    capture_type: str
    camera_id: str | None
    quality_score: float
    pose_yaw: float | None
    pose_pitch: float | None
    pose_roll: float | None


@dataclass(frozen=True)
class RecognitionCandidate:
    person_id: str
    score: float


@dataclass(frozen=True)
class RecognitionResult:
    decision: str
    matched: bool
    person_id: str | None
    top1: RecognitionCandidate | None
    top2: RecognitionCandidate | None


@dataclass(frozen=True)
class RecognitionEventRecord:
    camera_id: str | None
    track_id: str | None
    decision: str
    top1_person_id: str | None
    top1_score: float | None
    top2_person_id: str | None
    top2_score: float | None
    snapshot_path: str | None = None
    notes: str | None = None
    event_id: int | None = None
    created_at: datetime | None = None
