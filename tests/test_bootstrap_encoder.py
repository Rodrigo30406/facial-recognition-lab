import pytest

import facial_recognition.bootstrap as bootstrap_module
from facial_recognition.config import Settings
from facial_recognition.infrastructure.dummy_encoder import DummyFaceEncoder


def test_build_encoder_dummy_backend() -> None:
    encoder = bootstrap_module._build_encoder(Settings(encoder_backend="dummy"))
    assert isinstance(encoder, DummyFaceEncoder)


def test_build_encoder_insightface_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class StubInsightFaceEncoder:
        def __init__(
            self,
            model_name: str,
            providers: tuple[str, ...],
            ctx_id: int,
            det_size: tuple[int, int],
        ) -> None:
            captured["model_name"] = model_name
            captured["providers"] = providers
            captured["ctx_id"] = ctx_id
            captured["det_size"] = det_size

    monkeypatch.setattr(bootstrap_module, "InsightFaceEncoder", StubInsightFaceEncoder)

    settings = Settings(
        encoder_backend="insightface",
        insightface_model_name="buffalo_s",
        insightface_providers=("CPUExecutionProvider",),
        insightface_ctx_id=-1,
        insightface_det_size=(320, 320),
    )
    encoder = bootstrap_module._build_encoder(settings)

    assert isinstance(encoder, StubInsightFaceEncoder)
    assert captured == {
        "model_name": "buffalo_s",
        "providers": ("CPUExecutionProvider",),
        "ctx_id": -1,
        "det_size": (320, 320),
    }


def test_build_encoder_invalid_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported encoder_backend"):
        bootstrap_module._build_encoder(Settings(encoder_backend="unknown_backend"))
