from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from eleccia_listen import CommandEvent, ElecciaListenService, ListenSettings
from eleccia_mqtt import ElecciaMqttService, MqttSettings
from eleccia_vision import ElecciaVisionService, VisionSettings
from eleccia_voice import ElecciaVoiceService, VoiceSettings


@dataclass(frozen=True)
class RuntimeSettings:
    modules: tuple[str, ...] = ("vision",)
    identification_args: str = ""
    voice_enabled: bool = False
    voice_backend: str = "auto"
    voice_lang: str | None = None
    voice_rate: int | None = None
    voice_volume: float | None = None
    voice_id: str | None = None
    melo_language: str | None = None
    melo_speaker: str | None = None
    melo_speed: float | None = None
    melo_device: str | None = None
    listen_enabled: bool = False
    listen_backend: str = "stdin"
    listen_stdin_prompt: str = "eleccia> "
    listen_wake_word: str = "eleccia"
    listen_whisper_model: str = "large-v3"
    listen_whisper_device: str = "cuda"
    listen_whisper_compute_type: str = "float16"
    listen_whisper_language: str | None = "es"
    listen_whisper_beam_size: int = 5
    listen_whisper_vad_filter: bool = True
    listen_whisper_chunk_seconds: float = 4.0
    listen_whisper_sample_rate_hz: int = 16000
    listen_whisper_input_device_index: int | None = None
    listen_whisper_min_rms: float = 0.003
    listen_whisper_endpointing_enabled: bool = True
    listen_whisper_frame_seconds: float = 0.2
    listen_whisper_speech_start_seconds: float = 0.2
    listen_whisper_silence_stop_seconds: float = 2.0
    listen_whisper_max_utterance_seconds: float = 8.0
    listen_whisper_pre_roll_seconds: float = 0.3
    listen_debug_timing: bool = False
    listen_noise_filter_enabled: bool = False
    listen_openwakeword_model_paths: tuple[str, ...] = ()
    listen_openwakeword_inference_framework: str = "onnx"
    listen_openwakeword_threshold: float = 0.5
    listen_openwakeword_chunk_size: int = 1280
    listen_openwakeword_cooldown_seconds: float = 1.5
    listen_require_wake_word: bool = True
    listen_wake_word_aliases: tuple[str, ...] = ()
    listen_wake_word_fuzzy_threshold: float = 0.80
    listen_wake_command_window_seconds: float = 6.0
    mqtt_enabled: bool = False
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_username: str | None = None
    mqtt_password: str | None = None
    mqtt_client_id: str | None = None
    mqtt_topic_prefix: str = "eleccia"
    mqtt_qos: int = 0
    mqtt_retain: bool = False

    @classmethod
    def from_env(cls) -> RuntimeSettings:
        file_values = read_env_file()
        modules_raw = _env_lookup("ELECCIA_CORE_MODULES", file_values) or "vision"
        modules = tuple(part.strip().lower() for part in modules_raw.split(",") if part.strip())

        return cls(
            modules=modules or ("vision",),
            identification_args=_env_lookup("ELECCIA_IDENTIFICATION_ARGS", file_values) or "",
            voice_enabled=_env_bool("ELECCIA_VOICE_ENABLED", False, file_values),
            voice_backend=_env_lookup("ELECCIA_VOICE_BACKEND", file_values) or "auto",
            voice_lang=_env_lookup("ELECCIA_VOICE_LANG", file_values),
            voice_rate=_env_int_optional("ELECCIA_VOICE_RATE", file_values),
            voice_volume=_env_float_optional("ELECCIA_VOICE_VOLUME", file_values),
            voice_id=_env_lookup("ELECCIA_VOICE_ID", file_values),
            melo_language=_env_lookup("ELECCIA_MELO_LANGUAGE", file_values),
            melo_speaker=_env_lookup("ELECCIA_MELO_SPEAKER", file_values),
            melo_speed=_env_float_optional("ELECCIA_MELO_SPEED", file_values),
            melo_device=_env_lookup("ELECCIA_MELO_DEVICE", file_values),
            listen_enabled=_env_bool("ELECCIA_LISTEN_ENABLED", False, file_values),
            listen_backend=_env_lookup("ELECCIA_LISTEN_BACKEND", file_values) or "stdin",
            listen_stdin_prompt=_env_lookup("ELECCIA_LISTEN_STDIN_PROMPT", file_values) or "eleccia> ",
            listen_wake_word=_env_lookup("ELECCIA_LISTEN_WAKE_WORD", file_values) or "eleccia",
            listen_whisper_model=_env_lookup("ELECCIA_LISTEN_WHISPER_MODEL", file_values) or "large-v3",
            listen_whisper_device=_env_lookup("ELECCIA_LISTEN_WHISPER_DEVICE", file_values) or "cuda",
            listen_whisper_compute_type=(
                _env_lookup("ELECCIA_LISTEN_WHISPER_COMPUTE_TYPE", file_values) or "float16"
            ),
            listen_whisper_language=(
                _env_lookup("ELECCIA_LISTEN_WHISPER_LANGUAGE", file_values) or "es"
            ),
            listen_whisper_beam_size=_env_int("ELECCIA_LISTEN_WHISPER_BEAM_SIZE", 5, file_values),
            listen_whisper_vad_filter=_env_bool("ELECCIA_LISTEN_WHISPER_VAD_FILTER", True, file_values),
            listen_whisper_chunk_seconds=_env_float("ELECCIA_LISTEN_WHISPER_CHUNK_SECONDS", 4.0, file_values),
            listen_whisper_sample_rate_hz=_env_int("ELECCIA_LISTEN_WHISPER_SAMPLE_RATE_HZ", 16000, file_values),
            listen_whisper_input_device_index=_env_int_optional(
                "ELECCIA_LISTEN_WHISPER_INPUT_DEVICE_INDEX",
                file_values,
            ),
            listen_whisper_min_rms=_env_float("ELECCIA_LISTEN_WHISPER_MIN_RMS", 0.003, file_values),
            listen_whisper_endpointing_enabled=_env_bool(
                "ELECCIA_LISTEN_WHISPER_ENDPOINTING_ENABLED",
                True,
                file_values,
            ),
            listen_whisper_frame_seconds=_env_float(
                "ELECCIA_LISTEN_WHISPER_FRAME_SECONDS",
                0.2,
                file_values,
            ),
            listen_whisper_speech_start_seconds=_env_float(
                "ELECCIA_LISTEN_WHISPER_SPEECH_START_SECONDS",
                0.2,
                file_values,
            ),
            listen_whisper_silence_stop_seconds=_env_float(
                "ELECCIA_LISTEN_WHISPER_SILENCE_STOP_SECONDS",
                2.0,
                file_values,
            ),
            listen_whisper_max_utterance_seconds=_env_float(
                "ELECCIA_LISTEN_WHISPER_MAX_UTTERANCE_SECONDS",
                8.0,
                file_values,
            ),
            listen_whisper_pre_roll_seconds=_env_float(
                "ELECCIA_LISTEN_WHISPER_PRE_ROLL_SECONDS",
                0.3,
                file_values,
            ),
            listen_debug_timing=_env_bool(
                "ELECCIA_LISTEN_DEBUG_TIMING",
                False,
                file_values,
            ),
            listen_noise_filter_enabled=_env_bool(
                "ELECCIA_LISTEN_NOISE_FILTER_ENABLED",
                False,
                file_values,
            ),
            listen_openwakeword_model_paths=_env_tuple_str("ELECCIA_LISTEN_OPENWAKEWORD_MODEL_PATHS", file_values),
            listen_openwakeword_inference_framework=(
                _env_lookup("ELECCIA_LISTEN_OPENWAKEWORD_INFERENCE_FRAMEWORK", file_values) or "onnx"
            ),
            listen_openwakeword_threshold=_env_float(
                "ELECCIA_LISTEN_OPENWAKEWORD_THRESHOLD",
                0.5,
                file_values,
            ),
            listen_openwakeword_chunk_size=_env_int("ELECCIA_LISTEN_OPENWAKEWORD_CHUNK_SIZE", 1280, file_values),
            listen_openwakeword_cooldown_seconds=_env_float(
                "ELECCIA_LISTEN_OPENWAKEWORD_COOLDOWN_SECONDS",
                1.5,
                file_values,
            ),
            listen_require_wake_word=_env_bool("ELECCIA_LISTEN_REQUIRE_WAKE_WORD", True, file_values),
            listen_wake_word_aliases=_env_tuple_str("ELECCIA_LISTEN_WAKE_WORD_ALIASES", file_values),
            listen_wake_word_fuzzy_threshold=_env_float(
                "ELECCIA_LISTEN_WAKE_WORD_FUZZY_THRESHOLD",
                0.80,
                file_values,
            ),
            listen_wake_command_window_seconds=_env_float(
                "ELECCIA_LISTEN_WAKE_COMMAND_WINDOW_SECONDS",
                6.0,
                file_values,
            ),
            mqtt_enabled=_env_bool("ELECCIA_MQTT_ENABLED", False, file_values),
            mqtt_host=_env_lookup("ELECCIA_MQTT_HOST", file_values) or "127.0.0.1",
            mqtt_port=_env_int("ELECCIA_MQTT_PORT", 1883, file_values),
            mqtt_username=_env_lookup("ELECCIA_MQTT_USERNAME", file_values),
            mqtt_password=_env_lookup("ELECCIA_MQTT_PASSWORD", file_values),
            mqtt_client_id=_env_lookup("ELECCIA_MQTT_CLIENT_ID", file_values),
            mqtt_topic_prefix=_env_lookup("ELECCIA_MQTT_TOPIC_PREFIX", file_values) or "eleccia",
            mqtt_qos=_env_int("ELECCIA_MQTT_QOS", 0, file_values),
            mqtt_retain=_env_bool("ELECCIA_MQTT_RETAIN", False, file_values),
        )


class RuntimeModule(Protocol):
    name: str

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    @property
    def is_running(self) -> bool:
        ...


class VisionIdentificationModule:
    name = "vision"

    def __init__(self, settings: RuntimeSettings) -> None:
        self._settings = settings
        self._service: ElecciaVisionService | None = None

    @property
    def is_running(self) -> bool:
        service = self._service
        return bool(service is not None and service.is_running)

    def start(self) -> None:
        if self.is_running:
            return
        service = ElecciaVisionService(
            settings=VisionSettings(
                enabled=True,
                identification_args=self._settings.identification_args,
            )
        )
        service.start()
        self._service = service
        print("[eleccia] vision module started (subprocess service)")
        if service.last_error:
            print(f"[eleccia] vision module detail: {service.last_error}")

    def stop(self) -> None:
        service = self._service
        if service is not None:
            service.stop()
        self._service = None


class VoiceModule:
    name = "voice"

    def __init__(self, settings: RuntimeSettings) -> None:
        self._settings = settings
        self._service: ElecciaVoiceService | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return

        voice_settings = VoiceSettings(
            enabled=bool(self._settings.voice_enabled),
            backend=str(self._settings.voice_backend or "auto"),
            voice_lang=self._settings.voice_lang,
            voice_rate=self._settings.voice_rate,
            voice_volume=self._settings.voice_volume,
            voice_id=self._settings.voice_id,
            melo_language=self._settings.melo_language,
            melo_speaker=self._settings.melo_speaker,
            melo_speed=self._settings.melo_speed,
            melo_device=self._settings.melo_device,
        )
        self._service = ElecciaVoiceService(settings=voice_settings)
        self._running = True

        backend = self._service.backend_kind
        backend_error = self._service.backend_error
        if backend is None:
            print("[eleccia] voice module ready: disabled/no backend")
            if backend_error:
                print(f"[eleccia] voice backend detail: {backend_error}")
            return

        print(f"[eleccia] voice module ready: {backend}")
        if backend_error:
            print(f"[eleccia] voice backend detail: {backend_error}")

    def stop(self) -> None:
        if self._service is not None:
            self._service.close()
        self._running = False

    def speak_text(self, text: str) -> bool:
        service = self._service
        if service is None:
            return False
        return service.speak(text)


class ListenModule:
    name = "listen"

    def __init__(
        self,
        settings: RuntimeSettings,
        on_command: Callable[[CommandEvent], None] | None = None,
    ) -> None:
        self._settings = settings
        self._on_command = on_command
        self._service: ElecciaListenService | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        listen_settings = ListenSettings(
            enabled=bool(self._settings.listen_enabled),
            backend=self._settings.listen_backend,
            stdin_prompt=self._settings.listen_stdin_prompt,
            wake_word=self._settings.listen_wake_word,
            whisper_model=self._settings.listen_whisper_model,
            whisper_device=self._settings.listen_whisper_device,
            whisper_compute_type=self._settings.listen_whisper_compute_type,
            whisper_language=self._settings.listen_whisper_language,
            whisper_beam_size=self._settings.listen_whisper_beam_size,
            whisper_vad_filter=self._settings.listen_whisper_vad_filter,
            whisper_chunk_seconds=self._settings.listen_whisper_chunk_seconds,
            whisper_sample_rate_hz=self._settings.listen_whisper_sample_rate_hz,
            whisper_input_device_index=self._settings.listen_whisper_input_device_index,
            whisper_min_rms=self._settings.listen_whisper_min_rms,
            whisper_endpointing_enabled=self._settings.listen_whisper_endpointing_enabled,
            whisper_frame_seconds=self._settings.listen_whisper_frame_seconds,
            whisper_speech_start_seconds=self._settings.listen_whisper_speech_start_seconds,
            whisper_silence_stop_seconds=self._settings.listen_whisper_silence_stop_seconds,
            whisper_max_utterance_seconds=self._settings.listen_whisper_max_utterance_seconds,
            whisper_pre_roll_seconds=self._settings.listen_whisper_pre_roll_seconds,
            debug_timing=self._settings.listen_debug_timing,
            noise_filter_enabled=self._settings.listen_noise_filter_enabled,
            openwakeword_model_paths=self._settings.listen_openwakeword_model_paths,
            openwakeword_inference_framework=self._settings.listen_openwakeword_inference_framework,
            openwakeword_threshold=self._settings.listen_openwakeword_threshold,
            openwakeword_chunk_size=self._settings.listen_openwakeword_chunk_size,
            openwakeword_cooldown_seconds=self._settings.listen_openwakeword_cooldown_seconds,
            require_wake_word=self._settings.listen_require_wake_word,
            wake_word_aliases=self._settings.listen_wake_word_aliases,
            wake_word_fuzzy_threshold=self._settings.listen_wake_word_fuzzy_threshold,
            wake_command_window_seconds=self._settings.listen_wake_command_window_seconds,
        )
        self._service = ElecciaListenService(settings=listen_settings, on_command=self._on_command)
        self._service.start()
        self._running = True
        state = "enabled" if self._settings.listen_enabled else "disabled"
        print(f"[eleccia] listen module ready: {state} ({self._settings.listen_backend})")

    def stop(self) -> None:
        if self._service is not None:
            self._service.stop()
        self._running = False


class MqttModule:
    name = "mqtt"

    def __init__(self, settings: RuntimeSettings) -> None:
        self._settings = settings
        self._service: ElecciaMqttService | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        service = ElecciaMqttService(
            settings=MqttSettings(
                enabled=self._settings.mqtt_enabled,
                host=self._settings.mqtt_host,
                port=self._settings.mqtt_port,
                username=self._settings.mqtt_username,
                password=self._settings.mqtt_password,
                client_id=self._settings.mqtt_client_id,
                topic_prefix=self._settings.mqtt_topic_prefix,
                qos=self._settings.mqtt_qos,
                retain=self._settings.mqtt_retain,
            )
        )
        service.start()
        self._service = service
        self._running = True

        if not self._settings.mqtt_enabled:
            print("[eleccia] mqtt module ready: disabled")
            return
        if service.is_running:
            print(
                f"[eleccia] mqtt module ready: {self._settings.mqtt_host}:{self._settings.mqtt_port}"
            )
            return
        print("[eleccia] mqtt module failed to connect")
        if service.last_error:
            print(f"[eleccia] mqtt detail: {service.last_error}")

    def stop(self) -> None:
        if self._service is not None:
            self._service.stop()
        self._running = False

    def publish_intent(self, event: CommandEvent) -> None:
        service = self._service
        if service is None:
            return
        ok = service.publish_intent(
            text=event.text,
            intent=event.intent.name,
            confidence=event.intent.confidence,
            slots=dict(event.intent.slots),
        )
        if not ok and self._settings.mqtt_enabled and service.last_error:
            print(f"[eleccia] mqtt publish failed: {service.last_error}")


class ElecciaRuntime:
    def __init__(self, settings: RuntimeSettings, repo_root: Path | None = None) -> None:
        self._settings = settings
        self._repo_root = repo_root or _repo_root()
        self._modules: list[RuntimeModule] = self._build_modules()

    @property
    def modules(self) -> tuple[RuntimeModule, ...]:
        return tuple(self._modules)

    def start(self) -> None:
        for module in self._modules:
            try:
                module.start()
            except Exception as exc:
                print(f"[eleccia] failed to start module '{module.name}': {exc}")

    def stop(self) -> None:
        for module in reversed(self._modules):
            try:
                module.stop()
            except Exception as exc:
                print(f"[eleccia] failed to stop module '{module.name}': {exc}")

    def _build_modules(self) -> list[RuntimeModule]:
        modules: list[RuntimeModule] = []
        requested = set(self._settings.modules)
        vision_module: VisionIdentificationModule | None = None
        voice_module: VoiceModule | None = None
        mqtt_module: MqttModule | None = None

        if "vision" in requested:
            vision_module = VisionIdentificationModule(settings=self._settings)
            modules.append(vision_module)

        if "voice" in requested:
            voice_module = VoiceModule(settings=self._settings)
            modules.append(voice_module)

        if "mqtt" in requested or self._settings.mqtt_enabled:
            mqtt_module = MqttModule(settings=self._settings)
            modules.append(mqtt_module)

        if "listen" in requested:
            modules.append(
                ListenModule(
                    settings=self._settings,
                    on_command=self._build_listen_handler(
                        vision_module=vision_module,
                        voice_module=voice_module,
                        mqtt_module=mqtt_module,
                    ),
                )
            )

        unknown = sorted(requested - {"vision", "voice", "listen", "mqtt"})
        for name in unknown:
            print(f"[eleccia] unknown module '{name}' (ignored)")

        return modules

    def _build_listen_handler(
        self,
        vision_module: VisionIdentificationModule | None,
        voice_module: VoiceModule | None,
        mqtt_module: MqttModule | None,
    ) -> Callable[[CommandEvent], None]:
        def _handler(event: CommandEvent) -> None:
            intent = event.intent.name
            print(f"[eleccia][listen] '{event.text}' -> intent={intent}")

            _apply_internal_intent(intent_name=intent, vision_module=vision_module)

            if mqtt_module is not None:
                mqtt_module.publish_intent(event)

            if voice_module is not None:
                response = _intent_response(intent)
                if response is not None:
                    voice_module.speak_text(response)

        return _handler


def load_env_file_into_environ() -> None:
    env_values = read_env_file()
    for key, value in env_values.items():
        os.environ.setdefault(key, value)


def read_env_file() -> dict[str, str]:
    env_file = os.getenv("ELECCIA_ENV_FILE", ".env")
    path = Path(env_file)
    if not path.is_absolute():
        path = _repo_root() / env_file
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
        values[key] = _strip_optional_quotes(value.strip())
    return values


def build_runtime_from_env() -> ElecciaRuntime:
    load_env_file_into_environ()
    settings = RuntimeSettings.from_env()
    return ElecciaRuntime(settings=settings, repo_root=_repo_root())


def build_identification_runtime_for_api() -> ElecciaRuntime | None:
    load_env_file_into_environ()
    file_values = read_env_file()
    enabled = _env_bool("ELECCIA_AUTO_START_IDENTIFICATION", False, file_values)
    if not enabled:
        return None

    settings = RuntimeSettings.from_env()
    if "vision" not in settings.modules:
        settings = RuntimeSettings(
            modules=("vision",),
            identification_args=settings.identification_args,
            voice_enabled=settings.voice_enabled,
            voice_backend=settings.voice_backend,
            voice_lang=settings.voice_lang,
            voice_rate=settings.voice_rate,
            voice_volume=settings.voice_volume,
            voice_id=settings.voice_id,
            melo_language=settings.melo_language,
            melo_speaker=settings.melo_speaker,
            melo_speed=settings.melo_speed,
            melo_device=settings.melo_device,
            listen_enabled=settings.listen_enabled,
            listen_backend=settings.listen_backend,
            listen_stdin_prompt=settings.listen_stdin_prompt,
            listen_wake_word=settings.listen_wake_word,
            listen_whisper_model=settings.listen_whisper_model,
            listen_whisper_device=settings.listen_whisper_device,
            listen_whisper_compute_type=settings.listen_whisper_compute_type,
            listen_whisper_language=settings.listen_whisper_language,
            listen_whisper_beam_size=settings.listen_whisper_beam_size,
            listen_whisper_vad_filter=settings.listen_whisper_vad_filter,
            listen_whisper_chunk_seconds=settings.listen_whisper_chunk_seconds,
            listen_whisper_sample_rate_hz=settings.listen_whisper_sample_rate_hz,
            listen_whisper_input_device_index=settings.listen_whisper_input_device_index,
            listen_whisper_min_rms=settings.listen_whisper_min_rms,
            listen_whisper_endpointing_enabled=settings.listen_whisper_endpointing_enabled,
            listen_whisper_frame_seconds=settings.listen_whisper_frame_seconds,
            listen_whisper_speech_start_seconds=settings.listen_whisper_speech_start_seconds,
            listen_whisper_silence_stop_seconds=settings.listen_whisper_silence_stop_seconds,
            listen_whisper_max_utterance_seconds=settings.listen_whisper_max_utterance_seconds,
            listen_whisper_pre_roll_seconds=settings.listen_whisper_pre_roll_seconds,
            listen_debug_timing=settings.listen_debug_timing,
            listen_noise_filter_enabled=settings.listen_noise_filter_enabled,
            listen_openwakeword_model_paths=settings.listen_openwakeword_model_paths,
            listen_openwakeword_inference_framework=settings.listen_openwakeword_inference_framework,
            listen_openwakeword_threshold=settings.listen_openwakeword_threshold,
            listen_openwakeword_chunk_size=settings.listen_openwakeword_chunk_size,
            listen_openwakeword_cooldown_seconds=settings.listen_openwakeword_cooldown_seconds,
            listen_require_wake_word=settings.listen_require_wake_word,
            listen_wake_word_aliases=settings.listen_wake_word_aliases,
            listen_wake_word_fuzzy_threshold=settings.listen_wake_word_fuzzy_threshold,
            listen_wake_command_window_seconds=settings.listen_wake_command_window_seconds,
        )
    return ElecciaRuntime(settings=settings, repo_root=_repo_root())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _env_bool(key: str, default: bool, file_values: dict[str, str]) -> bool:
    raw = _env_lookup(key, file_values)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int_optional(key: str, file_values: dict[str, str]) -> int | None:
    raw = _env_lookup(key, file_values)
    if raw is None or not raw.strip():
        return None
    return int(raw.strip())


def _env_int(key: str, default: int, file_values: dict[str, str]) -> int:
    raw = _env_lookup(key, file_values)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _env_float_optional(key: str, file_values: dict[str, str]) -> float | None:
    raw = _env_lookup(key, file_values)
    if raw is None or not raw.strip():
        return None
    return float(raw.strip())


def _env_float(key: str, default: float, file_values: dict[str, str]) -> float:
    raw = _env_lookup(key, file_values)
    if raw is None or not raw.strip():
        return default
    return float(raw.strip())


def _env_tuple_str(key: str, file_values: dict[str, str]) -> tuple[str, ...]:
    raw = _env_lookup(key, file_values)
    if raw is None:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _apply_internal_intent(
    *,
    intent_name: str,
    vision_module: VisionIdentificationModule | None,
) -> None:
    if vision_module is None:
        return
    if intent_name == "camera_on":
        vision_module.start()
        return
    if intent_name == "camera_off":
        vision_module.stop()


def _intent_response(intent_name: str) -> str | None:
    if intent_name == "greeting":
        return "Hola, en que te ayudo?"
    if intent_name == "wake":
        return "Hola, en que te ayudo?"
    if intent_name == "lights_on":
        return "Encendiendo luces."
    if intent_name == "lights_off":
        return "Apagando luces."
    if intent_name == "camera_on":
        return "Activando la camara."
    if intent_name == "camera_off":
        return "Deteniendo la camara."
    if intent_name == "status":
        return "Todos los modulos operativos."
    return None
