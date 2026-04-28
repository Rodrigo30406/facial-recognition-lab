from __future__ import annotations

from eleccia_vision.config import Settings
from eleccia_vision.domain.entities import RecognitionCandidate, RecognitionResult
from eleccia_vision.domain.interfaces import FaceEncoder, FaceRepository
from eleccia_vision.infrastructure.faiss_search import FaissSearcher


class RecognitionService:
    def __init__(
        self,
        encoder: FaceEncoder,
        face_repository: FaceRepository,
        searcher: FaissSearcher,
        settings: Settings,
    ) -> None:
        self._encoder = encoder
        self._face_repository = face_repository
        self._searcher = searcher
        self._settings = settings

    def recognize(self, image_bytes: bytes) -> RecognitionResult:
        try:
            probe = self._encoder.encode(image_bytes)
        except ValueError:
            # Common operational case: no face found in the frame.
            # Treat as unknown instead of crashing the caller loop.
            return RecognitionResult(
                decision="unknown_person",
                matched=False,
                person_id=None,
                top1=None,
                top2=None,
            )
        gallery = self._face_repository.list_all()
        if not gallery:
            return RecognitionResult(
                decision="unknown_person",
                matched=False,
                person_id=None,
                top1=None,
                top2=None,
            )

        ranked = self._searcher.search(
            probe_embedding=probe,
            candidates=gallery,
            top_k=self._settings.recognition_top_k,
        )
        if not ranked:
            return RecognitionResult(
                decision="unknown_person",
                matched=False,
                person_id=None,
                top1=None,
                top2=None,
            )

        top1 = ranked[0]
        top2 = ranked[1] if len(ranked) > 1 else None

        if top1.score < self._settings.recognition_threshold:
            return RecognitionResult(
                decision="unknown_person",
                matched=False,
                person_id=None,
                top1=top1,
                top2=top2,
            )

        if _is_ambiguous(top1=top1, top2=top2, margin=self._settings.recognition_margin):
            return RecognitionResult(
                decision="ambiguous_match",
                matched=False,
                person_id=None,
                top1=top1,
                top2=top2,
            )

        return RecognitionResult(
            decision="known_person",
            matched=True,
            person_id=top1.person_id,
            top1=top1,
            top2=top2,
        )


def _is_ambiguous(top1: RecognitionCandidate, top2: RecognitionCandidate | None, margin: float) -> bool:
    if top2 is None:
        return False
    return (top1.score - top2.score) < margin
