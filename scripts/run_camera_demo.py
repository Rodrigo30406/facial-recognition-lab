#!/usr/bin/env python3
"""Run camera demo from the Eleccia Vision module.

Usage:
  python3 scripts/run_camera_demo.py --camera-index 0 --camera-id cam-01
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without forcing PYTHONPATH=src.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from eleccia_vision.camera_demo import main


if __name__ == "__main__":
    main()
