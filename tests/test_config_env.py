import pytest

from eleccia_vision.config import Settings


@pytest.fixture(autouse=True)
def isolate_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("FACIAL_ENV_FILE", (tmp_path / "missing.env").as_posix())


def test_settings_from_dotenv_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "ENCODER_BACKEND",
        "INSIGHTFACE_MODEL_NAME",
        "INSIGHTFACE_PROVIDERS",
        "INSIGHTFACE_CTX_ID",
        "INSIGHTFACE_DET_SIZE",
        "FACIAL_ENCODER_BACKEND",
    ):
        monkeypatch.delenv(key, raising=False)

    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "ENCODER_BACKEND=insightface",
                "INSIGHTFACE_MODEL_NAME=buffalo_s",
                "INSIGHTFACE_PROVIDERS=CPUExecutionProvider",
                "INSIGHTFACE_CTX_ID=-1",
                "INSIGHTFACE_DET_SIZE=320x320",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FACIAL_ENV_FILE", (tmp_path / ".env").as_posix())

    cfg = Settings.from_env()

    assert cfg.encoder_backend == "insightface"
    assert cfg.insightface_model_name == "buffalo_s"
    assert cfg.insightface_providers == ("CPUExecutionProvider",)
    assert cfg.insightface_ctx_id == -1
    assert cfg.insightface_det_size == (320, 320)


def test_real_env_overrides_dotenv(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENCODER_BACKEND", raising=False)
    (tmp_path / ".env").write_text("ENCODER_BACKEND=dummy\n", encoding="utf-8")
    monkeypatch.setenv("FACIAL_ENV_FILE", (tmp_path / ".env").as_posix())
    monkeypatch.setenv("ENCODER_BACKEND", "insightface")

    cfg = Settings.from_env()

    assert cfg.encoder_backend == "insightface"


def test_settings_from_env_unprefixed_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENCODER_BACKEND", "insightface")
    monkeypatch.setenv("INSIGHTFACE_MODEL_NAME", "buffalo_s")
    monkeypatch.setenv("INSIGHTFACE_PROVIDERS", "CPUExecutionProvider")
    monkeypatch.setenv("INSIGHTFACE_CTX_ID", "-1")
    monkeypatch.setenv("INSIGHTFACE_DET_SIZE", "320x320")
    monkeypatch.setenv("RECOGNITION_TOP_K", "7")
    monkeypatch.setenv("RECOGNITION_THRESHOLD", "0.61")
    monkeypatch.setenv("TEMPORAL_CONSISTENCY_ENABLED", "true")
    monkeypatch.setenv("TEMPORAL_MIN_CONSISTENT_FRAMES", "4")

    cfg = Settings.from_env()

    assert cfg.encoder_backend == "insightface"
    assert cfg.insightface_model_name == "buffalo_s"
    assert cfg.insightface_providers == ("CPUExecutionProvider",)
    assert cfg.insightface_ctx_id == -1
    assert cfg.insightface_det_size == (320, 320)
    assert cfg.recognition_top_k == 7
    assert cfg.recognition_threshold == pytest.approx(0.61)
    assert cfg.temporal_consistency_enabled is True
    assert cfg.temporal_min_consistent_frames == 4


def test_settings_from_env_prefixed_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACIAL_ENCODER_BACKEND", "insightface")

    cfg = Settings.from_env()

    assert cfg.encoder_backend == "insightface"


def test_settings_from_env_invalid_det_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSIGHTFACE_DET_SIZE", "broken")

    with pytest.raises(ValueError, match="INSIGHTFACE_DET_SIZE"):
        Settings.from_env()


def test_settings_from_env_invalid_boolean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEMPORAL_CONSISTENCY_ENABLED", "maybe")

    with pytest.raises(ValueError, match="TEMPORAL_CONSISTENCY_ENABLED"):
        Settings.from_env()
