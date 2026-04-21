import hashlib

from facial_recognition.domain.interfaces import FaceEncoder


class DummyFaceEncoder(FaceEncoder):
    """Deterministic fake encoder for local development/testing."""

    def encode(self, image_bytes: bytes) -> list[float]:
        digest = hashlib.sha256(image_bytes).digest()
        # 16-dim vector normalized to 0..1
        return [b / 255.0 for b in digest[:16]]
