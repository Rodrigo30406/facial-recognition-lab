import pytest

from eleccia_vision.config import Settings


@pytest.fixture(autouse=True)
def isolate_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("ELECCIA_ENV_FILE", (tmp_path / "missing.env").as_posix())


def test_settings_from_dotenv_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "ELECCIA_ENCODER_BACKEND",
        "ELECCIA_INSIGHTFACE_MODEL_NAME",
        "ELECCIA_INSIGHTFACE_PROVIDERS",
        "ELECCIA_INSIGHTFACE_CTX_ID",
        "ELECCIA_INSIGHTFACE_DET_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)

    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "ELECCIA_ENCODER_BACKEND=insightface",
                "ELECCIA_INSIGHTFACE_MODEL_NAME=buffalo_s",
                "ELECCIA_INSIGHTFACE_PROVIDERS=CPUExecutionProvider",
                "ELECCIA_INSIGHTFACE_CTX_ID=-1",
                "ELECCIA_INSIGHTFACE_DET_SIZE=320x320",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ELECCIA_ENV_FILE", (tmp_path / ".env").as_posix())

    cfg = Settings.from_env()

    assert cfg.encoder_backend == "insightface"
    assert cfg.insightface_model_name == "buffalo_s"
    assert cfg.insightface_providers == ("CPUExecutionProvider",)
    assert cfg.insightface_ctx_id == -1
    assert cfg.insightface_det_size == (320, 320)


def test_real_env_overrides_dotenv(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELECCIA_ENCODER_BACKEND", raising=False)
    (tmp_path / ".env").write_text("ELECCIA_ENCODER_BACKEND=dummy\n", encoding="utf-8")
    monkeypatch.setenv("ELECCIA_ENV_FILE", (tmp_path / ".env").as_posix())
    monkeypatch.setenv("ELECCIA_ENCODER_BACKEND", "insightface")

    cfg = Settings.from_env()

    assert cfg.encoder_backend == "insightface"


def test_settings_from_env_eleccia_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELECCIA_ENCODER_BACKEND", "insightface")
    monkeypatch.setenv("ELECCIA_INSIGHTFACE_MODEL_NAME", "buffalo_s")
    monkeypatch.setenv("ELECCIA_INSIGHTFACE_PROVIDERS", "CPUExecutionProvider")
    monkeypatch.setenv("ELECCIA_INSIGHTFACE_CTX_ID", "-1")
    monkeypatch.setenv("ELECCIA_INSIGHTFACE_DET_SIZE", "320x320")
    monkeypatch.setenv("ELECCIA_RECOGNITION_TOP_K", "7")
    monkeypatch.setenv("ELECCIA_RECOGNITION_THRESHOLD", "0.61")
    monkeypatch.setenv("ELECCIA_TEMPORAL_CONSISTENCY_ENABLED", "true")
    monkeypatch.setenv("ELECCIA_TEMPORAL_MIN_CONSISTENT_FRAMES", "4")

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


def test_settings_from_env_invalid_det_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELECCIA_INSIGHTFACE_DET_SIZE", "broken")

    with pytest.raises(ValueError, match="ELECCIA_INSIGHTFACE_DET_SIZE"):
        Settings.from_env()


def test_settings_from_env_invalid_boolean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELECCIA_TEMPORAL_CONSISTENCY_ENABLED", "maybe")

    with pytest.raises(ValueError, match="ELECCIA_TEMPORAL_CONSISTENCY_ENABLED"):
        Settings.from_env()
