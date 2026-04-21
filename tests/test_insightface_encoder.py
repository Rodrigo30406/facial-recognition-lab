from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from facial_recognition.infrastructure.insightface_encoder import InsightFaceEncoder


@dataclass
class FakeFace:
    det_score: float
    bbox: list[float]
    normed_embedding: list[float] | None = None
    embedding: list[float] | None = None
    landmark_2d_106: list[list[float]] | None = None
    kps: list[list[float]] | None = None


def _make_image_bytes() -> bytes:
    image = np.full((48, 48, 3), 140, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("could not encode test image")
    return encoded.tobytes()


def test_encoder_calls_prepare_with_runtime_params() -> None:
    captured: dict[str, object] = {}

    class FakeFaceAnalysis:
        def __init__(self, name: str, providers: list[str]) -> None:
            captured["name"] = name
            captured["providers"] = providers

        def prepare(self, ctx_id: int, det_size: tuple[int, int]) -> None:
            captured["ctx_id"] = ctx_id
            captured["det_size"] = det_size

        def get(self, image: np.ndarray) -> list[FakeFace]:
            return [FakeFace(det_score=0.9, bbox=[0, 0, 10, 10], normed_embedding=[1.0, 0.0])]

    encoder = InsightFaceEncoder(
        model_name="buffalo_l",
        providers=["CPUExecutionProvider"],
        ctx_id=-1,
        det_size=(320, 320),
        face_analysis_factory=FakeFaceAnalysis,
    )

    assert captured["name"] == "buffalo_l"
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["ctx_id"] == -1
    assert captured["det_size"] == (320, 320)

    emb = encoder.encode(_make_image_bytes())
    assert emb == pytest.approx([1.0, 0.0], rel=1e-6, abs=1e-6)


def test_encoder_picks_best_face_by_detection_score() -> None:
    class FakeFaceAnalysis:
        def __init__(self, name: str, providers: list[str]) -> None:
            del name, providers

        def prepare(self, ctx_id: int, det_size: tuple[int, int]) -> None:
            del ctx_id, det_size

        def get(self, image: np.ndarray) -> list[FakeFace]:
            del image
            return [
                FakeFace(det_score=0.50, bbox=[0, 0, 200, 200], normed_embedding=[1.0, 0.0]),
                FakeFace(det_score=0.97, bbox=[0, 0, 50, 50], normed_embedding=[0.0, 1.0]),
            ]

    encoder = InsightFaceEncoder(face_analysis_factory=FakeFaceAnalysis)
    emb = encoder.encode(_make_image_bytes())

    assert emb == pytest.approx([0.0, 1.0], rel=1e-6, abs=1e-6)


def test_encoder_normalizes_raw_embedding_when_normed_not_present() -> None:
    class FakeFaceAnalysis:
        def __init__(self, name: str, providers: list[str]) -> None:
            del name, providers

        def prepare(self, ctx_id: int, det_size: tuple[int, int]) -> None:
            del ctx_id, det_size

        def get(self, image: np.ndarray) -> list[FakeFace]:
            del image
            return [FakeFace(det_score=0.9, bbox=[0, 0, 20, 20], embedding=[3.0, 4.0])]

    encoder = InsightFaceEncoder(face_analysis_factory=FakeFaceAnalysis)
    emb = encoder.encode(_make_image_bytes())

    assert emb == pytest.approx([0.6, 0.8], rel=1e-6, abs=1e-6)


def test_encoder_raises_when_no_face_detected() -> None:
    class FakeFaceAnalysis:
        def __init__(self, name: str, providers: list[str]) -> None:
            del name, providers

        def prepare(self, ctx_id: int, det_size: tuple[int, int]) -> None:
            del ctx_id, det_size

        def get(self, image: np.ndarray) -> list[FakeFace]:
            del image
            return []

    encoder = InsightFaceEncoder(face_analysis_factory=FakeFaceAnalysis)

    with pytest.raises(ValueError, match="No face detected"):
        encoder.encode(_make_image_bytes())


def test_encoder_raises_when_payload_is_not_image() -> None:
    class FakeFaceAnalysis:
        def __init__(self, name: str, providers: list[str]) -> None:
            del name, providers

        def prepare(self, ctx_id: int, det_size: tuple[int, int]) -> None:
            del ctx_id, det_size

        def get(self, image: np.ndarray) -> list[FakeFace]:
            del image
            return [FakeFace(det_score=0.9, bbox=[0, 0, 10, 10], normed_embedding=[1.0, 0.0])]

    encoder = InsightFaceEncoder(face_analysis_factory=FakeFaceAnalysis)

    with pytest.raises(ValueError, match="Invalid image payload"):
        encoder.encode(b"not-an-image")


def test_extract_landmarks_uses_106_and_limits_count() -> None:
    class FakeFaceAnalysis:
        def __init__(self, name: str, providers: list[str]) -> None:
            del name, providers

        def prepare(self, ctx_id: int, det_size: tuple[int, int]) -> None:
            del ctx_id, det_size

        def get(self, image: np.ndarray) -> list[FakeFace]:
            del image
            points = [[float(i), float(i + 1)] for i in range(30)]
            return [FakeFace(det_score=0.9, bbox=[0, 0, 10, 10], landmark_2d_106=points)]

    encoder = InsightFaceEncoder(face_analysis_factory=FakeFaceAnalysis)
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    pts = encoder.extract_landmarks(image, max_points=10)

    assert len(pts) == 10
    assert pts[0] == (0, 1)
    assert pts[-1] == (29, 30)


def test_extract_landmarks_falls_back_to_kps() -> None:
    class FakeFaceAnalysis:
        def __init__(self, name: str, providers: list[str]) -> None:
            del name, providers

        def prepare(self, ctx_id: int, det_size: tuple[int, int]) -> None:
            del ctx_id, det_size

        def get(self, image: np.ndarray) -> list[FakeFace]:
            del image
            return [
                FakeFace(
                    det_score=0.9,
                    bbox=[0, 0, 10, 10],
                    kps=[[10.2, 12.8], [14.0, 18.0], [20.0, 21.0], [8.0, 16.0], [18.0, 17.0]],
                )
            ]

    encoder = InsightFaceEncoder(face_analysis_factory=FakeFaceAnalysis)
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    pts = encoder.extract_landmarks(image, max_points=20)

    assert len(pts) == 5
    assert pts[0] == (10, 13)
