from __future__ import annotations

import shlex
import threading
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class VisionSettings:
    enabled: bool = True
    identification_args: str = ""


class ElecciaVisionService:
    """In-process vision module service backed by camera demo runtime."""

    def __init__(self, settings: VisionSettings) -> None:
        self._settings = settings
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._last_error: str | None = None

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return bool(self._running and thread is not None and thread.is_alive())

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> None:
        if not self._settings.enabled:
            return
        if self._running:
            return

        argv = shlex.split(self._settings.identification_args.strip()) if self._settings.identification_args else []
        self._stop_event.clear()
        self._last_error = None

        def _runner() -> None:
            try:
                camera_demo = import_module("eleccia_vision.camera_demo")
                camera_demo.main(argv=argv, stop_event=self._stop_event)
            except Exception as exc:
                self._last_error = str(exc)
                print(f"[eleccia] vision module failed (inprocess): {exc}")
            finally:
                self._running = False

        self._thread = threading.Thread(target=_runner, name="eleccia-vision-inprocess", daemon=True)
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._thread = None
        self._running = False
