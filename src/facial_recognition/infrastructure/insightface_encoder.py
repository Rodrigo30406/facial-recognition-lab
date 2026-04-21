from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import cv2
import numpy as np

from facial_recognition.domain.interfaces import FaceEncoder


class InsightFaceEncoder(FaceEncoder):
    """Face encoder backed by InsightFace FaceAnalysis."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: Sequence[str] | None = None,
        ctx_id: int = 0,
        det_size: tuple[int, int] = (640, 640),
        face_analysis_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._providers = list(providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])

        factory = face_analysis_factory
        if factory is None:
            from insightface.app import FaceAnalysis

            factory = FaceAnalysis

        self._app = factory(name=model_name, providers=self._providers)
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    def encode(self, image_bytes: bytes) -> list[float]:
        image = _decode_image(image_bytes)
        faces = self._app.get(image)
        if not faces:
            raise ValueError("No face detected in image payload")

        best_face = _pick_best_face(faces)
        embedding = _extract_embedding(best_face)
        return embedding.tolist()

    def extract_landmarks(self, image: np.ndarray, max_points: int = 20) -> list[tuple[int, int]]:
        faces = self._app.get(image)
        if not faces:
            return []

        best_face = _pick_best_face(faces)
        points = _read_attr(best_face, "landmark_2d_106")
        if points is None:
            points = _read_attr(best_face, "kps")
        if points is None:
            return []

        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
            return []

        limit = max(1, int(max_points))
        if arr.shape[0] > limit:
            indices = np.linspace(0, arr.shape[0] - 1, num=limit, dtype=int)
            arr = arr[indices]

        return [(int(round(x)), int(round(y))) for x, y in arr[:, :2]]


def _decode_image(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image payload")
    return image


def _pick_best_face(faces: Sequence[Any]) -> Any:
    def score(face: Any) -> tuple[float, float]:
        det_score = float(_read_attr(face, "det_score", 0.0))
        bbox = _read_attr(face, "bbox", [0.0, 0.0, 0.0, 0.0])
        x1, y1, x2, y2 = [float(v) for v in list(bbox)[:4]]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return det_score, area

    return max(faces, key=score)


def _extract_embedding(face: Any) -> np.ndarray:
    normed_embedding = _read_attr(face, "normed_embedding")
    if normed_embedding is not None:
        vector = np.asarray(normed_embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("Empty embedding produced by InsightFace")
        return _l2_normalize(vector)

    raw_embedding = _read_attr(face, "embedding")
    if raw_embedding is None:
        raise ValueError("InsightFace face object does not provide embedding vectors")

    vector = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        raise ValueError("Empty embedding produced by InsightFace")
    return _l2_normalize(vector)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Embedding norm is zero")
    return vector / norm


def _read_attr(face: Any, key: str, default: Any = None) -> Any:
    if isinstance(face, dict):
        return face.get(key, default)
    return getattr(face, key, default)
