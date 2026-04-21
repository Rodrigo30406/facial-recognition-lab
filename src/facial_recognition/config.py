from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Application settings container."""

    similarity_threshold: float = 0.45
    database_url: str = "sqlite:///data/db/facial_recognition.db"
    sample_storage_dir: str = "data/samples"

    # Cosine similarity policy for FAISS-based recognition.
    recognition_threshold: float = 0.5
    recognition_margin: float = 0.08
    recognition_top_k: int = 5
