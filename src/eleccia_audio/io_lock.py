from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import threading
import time
from typing import Iterator

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None


_THREAD_AUDIO_MUTEX = threading.RLock()


def _lock_file_path() -> Path:
    raw = os.getenv("ELECCIA_AUDIO_LOCK_FILE", "/tmp/eleccia_audio_io.lock").strip()
    if not raw:
        raw = "/tmp/eleccia_audio_io.lock"
    return Path(raw)


def _lock_timeout_seconds() -> float:
    raw = os.getenv("ELECCIA_AUDIO_LOCK_TIMEOUT_SECONDS", "10.0").strip()
    try:
        value = float(raw)
    except Exception:
        value = 10.0
    return max(0.1, value)


def _lock_strict() -> bool:
    raw = os.getenv("ELECCIA_AUDIO_LOCK_STRICT", "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@contextmanager
def audio_io_lock(timeout_seconds: float | None = None) -> Iterator[None]:
    timeout = _lock_timeout_seconds() if timeout_seconds is None else max(0.1, float(timeout_seconds))
    strict = _lock_strict()
    lock_path = _lock_file_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+", encoding="utf-8")
    started = time.monotonic()

    with _THREAD_AUDIO_MUTEX:
        try:
            if fcntl is not None:
                if strict:
                    # Strict mode: never continue without inter-process lock.
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                else:
                    while True:
                        try:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if (time.monotonic() - started) >= timeout:
                                # Non-strict fallback: proceed with thread lock only.
                                break
                            time.sleep(0.01)
            yield
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            try:
                lock_file.close()
            except Exception:
                pass
