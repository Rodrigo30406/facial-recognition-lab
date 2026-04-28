from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from facial_recognition.api.schemas import (
    CandidateResponse,
    EnrollResponse,
    EnrollmentImageResponse,
    MatchResponse,
    PersonCreateRequest,
    PersonResponse,
    RecognitionEventResponse,
    RecognitionMatchResponse,
)
from facial_recognition.application.enrollment import InvalidImageError, PersonNotFoundError
from facial_recognition.application.persons import InvalidPersonSexError, PersonAlreadyExistsError
from facial_recognition.bootstrap import build_services

app = FastAPI(title="Facial Recognition API", version="0.1.0")
services = build_services()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/persons", response_model=PersonResponse)
def create_person(payload: PersonCreateRequest) -> PersonResponse:
    try:
        person = services.person_service.create_person(
            person_id=payload.person_id,
            full_name=payload.full_name,
            sex=payload.sex,
        )
    except PersonAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidPersonSexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PersonResponse(person_id=person.person_id, full_name=person.full_name, sex=person.sex)


@app.get("/v1/persons", response_model=list[PersonResponse])
def list_people() -> list[PersonResponse]:
    people = services.person_service.list_people()
    return [PersonResponse(person_id=p.person_id, full_name=p.full_name, sex=p.sex) for p in people]


@app.post("/v1/enrollments/image", response_model=EnrollmentImageResponse)
async def enroll_image(
    person_id: str = Form(...),
    image: UploadFile = File(...),
    capture_type: str = Form("operational"),
    camera_id: str | None = Form(default=None),
) -> EnrollmentImageResponse:
    payload = await image.read()
    try:
        sample = services.enrollment_service.enroll_image(
            person_id=person_id,
            image_bytes=payload,
            capture_type=capture_type,
            camera_id=camera_id,
        )
    except PersonNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return EnrollmentImageResponse(
        person_id=sample.person_id,
        image_path=sample.image_path,
        capture_type=sample.capture_type,
        camera_id=sample.camera_id,
        quality_score=sample.quality_score,
    )


@app.post("/v1/faces/enroll", response_model=EnrollResponse)
async def enroll_face(person_id: str = Form(...), image: UploadFile = File(...)) -> EnrollResponse:
    person = services.person_service.get_person(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' does not exist")

    payload = await image.read()
    services.face_service.enroll(person_id=person_id, image_bytes=payload)
    return EnrollResponse()


@app.post("/v1/recognitions/match", response_model=RecognitionMatchResponse)
async def recognize_face(
    image: UploadFile = File(...),
    camera_id: str | None = Form(default=None),
    track_id: str | None = Form(default=None),
    snapshot_path: str | None = Form(default=None),
    notes: str | None = Form(default=None),
) -> RecognitionMatchResponse:
    payload = await image.read()
    raw = services.recognition_service.recognize(payload)
    stream_id = _recognition_stream_id(camera_id=camera_id, track_id=track_id)
    result = services.recognition_consistency_service.stabilize(raw, stream_id=stream_id)
    services.recognition_event_service.record_from_result(
        result=result,
        camera_id=camera_id,
        track_id=track_id,
        snapshot_path=snapshot_path,
        notes=notes,
    )

    def as_candidate(candidate: object) -> CandidateResponse | None:
        if candidate is None:
            return None
        return CandidateResponse(person_id=candidate.person_id, score=candidate.score)

    return RecognitionMatchResponse(
        decision=result.decision,
        matched=result.matched,
        person_id=result.person_id,
        top1=as_candidate(result.top1),
        top2=as_candidate(result.top2),
    )


def _recognition_stream_id(camera_id: str | None, track_id: str | None) -> str:
    if camera_id and track_id:
        return f"{camera_id}::{track_id}"
    if camera_id:
        return f"{camera_id}::_"
    if track_id:
        return f"_::{track_id}"
    return "_default"


@app.get("/v1/events", response_model=list[RecognitionEventResponse])
def list_events(
    limit: int = Query(default=100, ge=1, le=500),
    decision: str | None = Query(default=None),
    camera_id: str | None = Query(default=None),
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
) -> list[RecognitionEventResponse]:
    events = services.recognition_event_service.list_events(
        limit=limit,
        decision=decision,
        camera_id=camera_id,
        date_from=date_from,
        date_to=date_to,
    )
    return [
        RecognitionEventResponse(
            event_id=e.event_id,
            created_at=e.created_at,
            camera_id=e.camera_id,
            track_id=e.track_id,
            decision=e.decision,
            top1_person_id=e.top1_person_id,
            top1_score=e.top1_score,
            top2_person_id=e.top2_person_id,
            top2_score=e.top2_score,
            snapshot_path=e.snapshot_path,
            notes=e.notes,
        )
        for e in events
    ]


@app.post("/v1/faces/match", response_model=MatchResponse)
async def match_face(image: UploadFile = File(...)) -> MatchResponse:
    payload = await image.read()
    result = services.face_service.match(payload)
    return MatchResponse(
        matched=result.matched,
        person_id=result.person_id,
        distance=result.distance,
    )
