from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application settings container."""

    similarity_threshold: float = 0.45
    database_url: str = "sqlite:///data/db/eleccia_vision.db"
    sample_storage_dir: str = "data/samples"

    # Cosine similarity policy for FAISS-based recognition.
    recognition_threshold: float = 0.5
    recognition_margin: float = 0.08
    recognition_top_k: int = 5

    # Encoder backend policy.
    encoder_backend: str = "dummy"
    insightface_model_name: str = "buffalo_l"
    insightface_providers: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")
    insightface_ctx_id: int = 0
    insightface_det_size: tuple[int, int] = (640, 640)

    # Temporal consistency policy.
    temporal_consistency_enabled: bool = True
    temporal_min_consistent_frames: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        file_values = _read_env_file()
        return cls(
            similarity_threshold=_env_float(
                "ELECCIA_SIMILARITY_THRESHOLD", cls.similarity_threshold, file_values
            ),
            database_url=_env_str("ELECCIA_DATABASE_URL", cls.database_url, file_values),
            sample_storage_dir=_env_str("ELECCIA_SAMPLE_STORAGE_DIR", cls.sample_storage_dir, file_values),
            recognition_threshold=_env_float(
                "ELECCIA_RECOGNITION_THRESHOLD", cls.recognition_threshold, file_values
            ),
            recognition_margin=_env_float(
                "ELECCIA_RECOGNITION_MARGIN",
                cls.recognition_margin,
                file_values,
            ),
            recognition_top_k=_env_int("ELECCIA_RECOGNITION_TOP_K", cls.recognition_top_k, file_values),
            encoder_backend=_env_str("ELECCIA_ENCODER_BACKEND", cls.encoder_backend, file_values),
            insightface_model_name=_env_str(
                "ELECCIA_INSIGHTFACE_MODEL_NAME", cls.insightface_model_name, file_values
            ),
            insightface_providers=_env_providers(
                "ELECCIA_INSIGHTFACE_PROVIDERS", cls.insightface_providers, file_values
            ),
            insightface_ctx_id=_env_int("ELECCIA_INSIGHTFACE_CTX_ID", cls.insightface_ctx_id, file_values),
            insightface_det_size=_env_det_size(
                "ELECCIA_INSIGHTFACE_DET_SIZE", cls.insightface_det_size, file_values
            ),
            temporal_consistency_enabled=_env_bool(
                "ELECCIA_TEMPORAL_CONSISTENCY_ENABLED",
                cls.temporal_consistency_enabled,
                file_values,
            ),
            temporal_min_consistent_frames=_env_int(
                "ELECCIA_TEMPORAL_MIN_CONSISTENT_FRAMES",
                cls.temporal_min_consistent_frames,
                file_values,
            ),
        )


def _read_env_file() -> dict[str, str]:
    env_file = os.getenv("ELECCIA_ENV_FILE", ".env")
    path = Path(env_file)
    if not path.exists() or not path.is_file():
        return {}

    values: dict[str, str] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = _strip_optional_quotes(value.strip())
        values[key] = value
    return values


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _env_lookup(key: str, file_values: dict[str, str]) -> str | None:
    if key in os.environ:
        return os.environ[key]
    if key in file_values:
        return file_values[key]
    return None


def _env_str(key: str, default: str, file_values: dict[str, str]) -> str:
    raw = _env_lookup(key, file_values)
    if raw is None:
        return default
    return raw.strip()


def _env_int(key: str, default: int, file_values: dict[str, str]) -> int:
    raw = _env_lookup(key, file_values)
    if raw is None:
        return default
    return int(raw.strip())


def _env_float(key: str, default: float, file_values: dict[str, str]) -> float:
    raw = _env_lookup(key, file_values)
    if raw is None:
        return default
    return float(raw.strip())


def _env_bool(key: str, default: bool, file_values: dict[str, str]) -> bool:
    raw = _env_lookup(key, file_values)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Environment variable '{key}' must be one of: "
        "1/0, true/false, yes/no, on/off"
    )


def _env_providers(
    key: str, default: tuple[str, ...], file_values: dict[str, str]
) -> tuple[str, ...]:
    raw = _env_lookup(key, file_values)
    if raw is None:
        return default

    parsed = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not parsed:
        raise ValueError(
            f"Environment variable '{key}' must contain at least one provider"
        )
    return parsed


def _env_det_size(
    key: str, default: tuple[int, int], file_values: dict[str, str]
) -> tuple[int, int]:
    raw = _env_lookup(key, file_values)
    if raw is None:
        return default

    normalized = raw.lower().replace("x", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"Environment variable '{key}' must be '<width>,<height>' or "
            f"'<width>x<height>'"
        )

    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Environment variable '{key}' must use positive integers"
        )
    return (width, height)
