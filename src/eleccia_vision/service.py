from __future__ import annotations

import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VisionDetectionSettings:
    enabled: bool = True
    detection_args: str = ""


@dataclass(frozen=True)
class VisionEnrollSettings:
    enabled: bool = True
    person_id: str = ""
    enroll_args: str = ""
    guided_enroll: bool = True
    guided_preset: str | None = None


@dataclass(frozen=True)
class VisionSettings:
    """Backwards-compatible alias settings used by ElecciaVisionService."""

    enabled: bool = True
    identification_args: str = ""


class _VisionProcessService:
    """Base vision service backed by a dedicated camera subprocess."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._lifecycle_lock = threading.Lock()
        self._last_error: str | None = None
        self._repo_root = Path(__file__).resolve().parents[2]
        self._script_path = self._repo_root / "scripts" / "run_vision_runtime.py"

    @property
    def is_running(self) -> bool:
        process = self._process
        return bool(process is not None and process.poll() is None)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> None:
        with self._lifecycle_lock:
            process = self._process
            if process is not None and process.poll() is None:
                return

            self._last_error = None
            if not self._script_path.exists():
                self._last_error = f"camera runtime script not found: {self._script_path}"
                print(f"[eleccia] vision module failed: {self._last_error}")
                return

            cmd = [sys.executable, str(self._script_path), *self._build_argv()]
            try:
                self._process = subprocess.Popen(cmd, cwd=str(self._repo_root))
            except Exception as exc:
                self._process = None
                self._last_error = str(exc)
                print(f"[eleccia] vision module failed to start subprocess: {exc}")

    def stop(self) -> None:
        with self._lifecycle_lock:
            process = self._process
            self._process = None
        if process is None:
            return
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)

    def _build_argv(self) -> list[str]:
        raise NotImplementedError


class ElecciaVisionDetectionService(_VisionProcessService):
    """Detection mode service (camera recognition runtime)."""

    def __init__(self, settings: VisionDetectionSettings) -> None:
        super().__init__()
        self._settings = settings

    def start(self) -> None:
        if not self._settings.enabled:
            return
        super().start()

    def _build_argv(self) -> list[str]:
        raw = self._settings.detection_args.strip()
        if not raw:
            return []
        argv = shlex.split(raw)
        # Prevent accidental enrollment mode when running detection service.
        blocked = {"--enroll-person-id", "--guided-enroll"}
        filtered: list[str] = []
        skip_next = False
        for token in argv:
            if skip_next:
                skip_next = False
                continue
            if token in blocked:
                if token == "--enroll-person-id":
                    skip_next = True
                continue
            if token.startswith("--enroll-person-id="):
                continue
            filtered.append(token)
        return filtered


class ElecciaVisionEnrollService(_VisionProcessService):
    """Enrollment mode service (camera + capture samples for one person)."""

    def __init__(self, settings: VisionEnrollSettings) -> None:
        super().__init__()
        self._settings = settings

    def start(self) -> None:
        if not self._settings.enabled:
            return
        if not self._settings.person_id.strip():
            self._last_error = "Vision enroll requires non-empty person_id"
            print(f"[eleccia] vision enroll failed: {self._last_error}")
            return
        super().start()

    def _build_argv(self) -> list[str]:
        argv = shlex.split(self._settings.enroll_args.strip()) if self._settings.enroll_args.strip() else []
        person_id = self._settings.person_id.strip()
        argv.extend(["--enroll-person-id", person_id])
        if self._settings.guided_enroll and "--guided-enroll" not in argv:
            argv.append("--guided-enroll")
        guided_preset = (self._settings.guided_preset or "").strip()
        if guided_preset and "--guided-preset" not in argv:
            argv.extend(["--guided-preset", guided_preset])
        return argv


class ElecciaVisionService(ElecciaVisionDetectionService):
    """Backwards-compatible detection service used by Eleccia Core."""

    def __init__(self, settings: VisionSettings) -> None:
        super().__init__(
            VisionDetectionSettings(
                enabled=settings.enabled,
                detection_args=settings.identification_args,
            )
        )
