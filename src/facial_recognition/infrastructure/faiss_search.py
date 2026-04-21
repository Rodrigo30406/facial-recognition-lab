from __future__ import annotations

import numpy as np

from facial_recognition.domain.entities import FaceRecord, RecognitionCandidate


class FaissSearcher:
    """FAISS cosine similarity search over normalized embeddings."""

    def search(
        self,
        probe_embedding: list[float],
        candidates: list[FaceRecord],
        top_k: int,
    ) -> list[RecognitionCandidate]:
        if not candidates:
            return []

        import faiss

        matrix = np.asarray([c.embedding for c in candidates], dtype=np.float32)
        probe = np.asarray(probe_embedding, dtype=np.float32)

        if matrix.ndim != 2 or probe.ndim != 1:
            raise ValueError("Invalid embedding dimensions")
        if matrix.shape[1] != probe.shape[0]:
            raise ValueError("Probe and gallery dimensions must match")

        matrix = _normalize_rows(matrix)
        probe = _normalize_row(probe)

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        k = min(top_k, len(candidates))
        scores, positions = index.search(probe.reshape(1, -1), k)

        out: list[RecognitionCandidate] = []
        for score, pos in zip(scores[0], positions[0], strict=True):
            if pos < 0:
                continue
            out.append(RecognitionCandidate(person_id=candidates[int(pos)].person_id, score=float(score)))
        return out


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return arr / norms


def _normalize_row(arr: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("Probe embedding norm is zero")
    return arr / norm
