from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Application settings container."""

    similarity_threshold: float = 0.45
