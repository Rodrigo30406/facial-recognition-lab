from fastapi import FastAPI, File, Form, UploadFile

from facial_recognition.api.schemas import EnrollResponse, MatchResponse
from facial_recognition.bootstrap import build_service

app = FastAPI(title="Facial Recognition API", version="0.1.0")
service = build_service()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/faces/enroll", response_model=EnrollResponse)
async def enroll_face(person_id: str = Form(...), image: UploadFile = File(...)) -> EnrollResponse:
    payload = await image.read()
    service.enroll(person_id=person_id, image_bytes=payload)
    return EnrollResponse()


@app.post("/v1/faces/match", response_model=MatchResponse)
async def match_face(image: UploadFile = File(...)) -> MatchResponse:
    payload = await image.read()
    result = service.match(payload)
    return MatchResponse(
        matched=result.matched,
        person_id=result.person_id,
        distance=result.distance,
    )
