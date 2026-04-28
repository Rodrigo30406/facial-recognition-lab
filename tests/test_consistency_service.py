from eleccia_vision.application.consistency import RecognitionConsistencyService
from eleccia_vision.domain.entities import RecognitionCandidate, RecognitionResult


def _known(person_id: str, score: float = 0.9) -> RecognitionResult:
    return RecognitionResult(
        decision="known_person",
        matched=True,
        person_id=person_id,
        top1=RecognitionCandidate(person_id=person_id, score=score),
        top2=None,
    )


def _unknown() -> RecognitionResult:
    return RecognitionResult(
        decision="unknown_person",
        matched=False,
        person_id=None,
        top1=None,
        top2=None,
    )


def test_consistency_requires_minimum_known_streak() -> None:
    service = RecognitionConsistencyService(enabled=True, min_consistent_frames=3)

    first = service.stabilize(_known("alice"), stream_id="cam-01")
    second = service.stabilize(_known("alice"), stream_id="cam-01")
    third = service.stabilize(_known("alice"), stream_id="cam-01")

    assert first.decision == "ambiguous_match"
    assert second.decision == "ambiguous_match"
    assert third.decision == "known_person"
    assert third.person_id == "alice"


def test_consistency_resets_after_unknown() -> None:
    service = RecognitionConsistencyService(enabled=True, min_consistent_frames=2)

    first = service.stabilize(_known("alice"), stream_id="cam-01")
    mid = service.stabilize(_unknown(), stream_id="cam-01")
    second = service.stabilize(_known("alice"), stream_id="cam-01")
    third = service.stabilize(_known("alice"), stream_id="cam-01")

    assert first.decision == "ambiguous_match"
    assert mid.decision == "unknown_person"
    assert second.decision == "ambiguous_match"
    assert third.decision == "known_person"


def test_consistency_keeps_streams_isolated() -> None:
    service = RecognitionConsistencyService(enabled=True, min_consistent_frames=2)

    a1 = service.stabilize(_known("alice"), stream_id="cam-a")
    b1 = service.stabilize(_known("alice"), stream_id="cam-b")
    a2 = service.stabilize(_known("alice"), stream_id="cam-a")

    assert a1.decision == "ambiguous_match"
    assert b1.decision == "ambiguous_match"
    assert a2.decision == "known_person"


def test_consistency_can_be_disabled() -> None:
    service = RecognitionConsistencyService(enabled=False, min_consistent_frames=99)
    out = service.stabilize(_known("alice"), stream_id="cam-01")
    assert out.decision == "known_person"
