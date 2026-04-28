from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from eleccia_vision.domain.entities import RecognitionResult


@dataclass
class _ConsistencyState:
    pending_person_id: str | None = None
    pending_count: int = 0


class RecognitionConsistencyService:
    """Stabilizes recognition results using per-stream temporal consistency."""

    def __init__(self, enabled: bool = True, min_consistent_frames: int = 3) -> None:
        if min_consistent_frames < 1:
            raise ValueError("min_consistent_frames must be >= 1")
        self._enabled = enabled
        self._min_consistent_frames = min_consistent_frames
        self._states: dict[str, _ConsistencyState] = {}
        self._lock = Lock()

    def stabilize(self, result: RecognitionResult, stream_id: str | None = None) -> RecognitionResult:
        if not self._enabled:
            return result

        key = stream_id or "_default"
        with self._lock:
            state = self._states.setdefault(key, _ConsistencyState())

            if result.decision == "known_person" and result.person_id is not None:
                if state.pending_person_id == result.person_id:
                    state.pending_count += 1
                else:
                    state.pending_person_id = result.person_id
                    state.pending_count = 1

                if state.pending_count >= self._min_consistent_frames:
                    return result

                # Conservative policy: known is emitted only after consistency window.
                return RecognitionResult(
                    decision="ambiguous_match",
                    matched=False,
                    person_id=None,
                    top1=result.top1,
                    top2=result.top2,
                )

            # Unknown/ambiguous/not-usable resets pending known streak.
            state.pending_person_id = None
            state.pending_count = 0
            return result

    def reset(self, stream_id: str | None = None) -> None:
        with self._lock:
            if stream_id is None:
                self._states.clear()
                return
            self._states.pop(stream_id, None)
