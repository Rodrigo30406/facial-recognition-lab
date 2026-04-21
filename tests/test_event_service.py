from datetime import timedelta

from facial_recognition.application.events import RecognitionEventService
from facial_recognition.domain.entities import RecognitionCandidate, RecognitionResult
from facial_recognition.infrastructure.inmemory_event_repo import InMemoryRecognitionEventRepository


def test_record_and_list_events() -> None:
    service = RecognitionEventService(repository=InMemoryRecognitionEventRepository())

    result = RecognitionResult(
        decision="known_person",
        matched=True,
        person_id="alice",
        top1=RecognitionCandidate(person_id="alice", score=0.92),
        top2=RecognitionCandidate(person_id="bob", score=0.61),
    )

    saved = service.record_from_result(result=result, camera_id="cam-01", track_id="t-123")

    assert saved.event_id is not None
    assert saved.created_at is not None
    assert saved.decision == "known_person"
    assert saved.top1_person_id == "alice"
    assert saved.top2_person_id == "bob"

    events = service.list_events(limit=10)
    assert len(events) == 1
    assert events[0].event_id == saved.event_id


def test_record_unknown_event() -> None:
    service = RecognitionEventService(repository=InMemoryRecognitionEventRepository())

    result = RecognitionResult(
        decision="unknown_person",
        matched=False,
        person_id=None,
        top1=None,
        top2=None,
    )

    saved = service.record_from_result(result=result)

    assert saved.decision == "unknown_person"
    assert saved.top1_person_id is None
    assert saved.top1_score is None


def test_event_filters_by_decision_camera_and_date_range() -> None:
    service = RecognitionEventService(repository=InMemoryRecognitionEventRepository())

    known = RecognitionResult(
        decision="known_person",
        matched=True,
        person_id="alice",
        top1=RecognitionCandidate(person_id="alice", score=0.93),
        top2=None,
    )
    unknown = RecognitionResult(
        decision="unknown_person",
        matched=False,
        person_id=None,
        top1=None,
        top2=None,
    )

    first = service.record_from_result(result=known, camera_id="cam-01")
    second = service.record_from_result(result=unknown, camera_id="cam-02")

    filtered_decision = service.list_events(limit=10, decision="known_person")
    assert len(filtered_decision) == 1
    assert filtered_decision[0].event_id == first.event_id

    filtered_camera = service.list_events(limit=10, camera_id="cam-02")
    assert len(filtered_camera) == 1
    assert filtered_camera[0].event_id == second.event_id

    assert first.created_at is not None
    assert second.created_at is not None
    start = first.created_at - timedelta(seconds=1)
    end = first.created_at + timedelta(seconds=1)

    filtered_date = service.list_events(limit=10, date_from=start, date_to=end)
    assert any(event.event_id == first.event_id for event in filtered_date)
