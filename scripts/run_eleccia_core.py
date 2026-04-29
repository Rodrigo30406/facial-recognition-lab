#!/usr/bin/env python3
"""Run Eleccia Core runtime and autostart enabled modules."""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

# Allow running from repo root without forcing PYTHONPATH=src.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from eleccia_core.runtime import build_runtime_from_env


def main() -> int:
    runtime = build_runtime_from_env()
    runtime.start()

    started = [module.name for module in runtime.modules]
    print(f"[eleccia] runtime active modules: {', '.join(started) if started else 'none'}")
    print("[eleccia] press Ctrl+C to stop")

    should_exit = False

    def _on_signal(_signum, _frame) -> None:
        nonlocal should_exit
        should_exit = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        while not should_exit:
            time.sleep(0.25)
    finally:
        runtime.stop()
        print("[eleccia] runtime stopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
