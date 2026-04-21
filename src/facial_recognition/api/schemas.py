from datetime import datetime

from pydantic import BaseModel, Field


class PersonCreateRequest(BaseModel):
    person_id: str
    full_name: str


class PersonResponse(BaseModel):
    person_id: str
    full_name: str


class EnrollResponse(BaseModel):
    status: str = Field(default="ok")


class EnrollmentImageResponse(BaseModel):
    status: str = Field(default="ok")
    person_id: str
    image_path: str
    capture_type: str
    camera_id: str | None
    quality_score: float


class MatchResponse(BaseModel):
    matched: bool
    person_id: str | None
    distance: float


class CandidateResponse(BaseModel):
    person_id: str
    score: float


class RecognitionMatchResponse(BaseModel):
    decision: str
    matched: bool
    person_id: str | None
    top1: CandidateResponse | None
    top2: CandidateResponse | None


class RecognitionEventResponse(BaseModel):
    event_id: int | None
    created_at: datetime | None
    camera_id: str | None
    track_id: str | None
    decision: str
    top1_person_id: str | None
    top1_score: float | None
    top2_person_id: str | None
    top2_score: float | None
    snapshot_path: str | None
    notes: str | None
