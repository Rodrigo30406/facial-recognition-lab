from pydantic import BaseModel, Field


class EnrollResponse(BaseModel):
    status: str = Field(default="ok")


class MatchResponse(BaseModel):
    matched: bool
    person_id: str | None
    distance: float
