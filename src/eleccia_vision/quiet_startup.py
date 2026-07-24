"""Optional suppression of third-party startup noise.

InsightFace/onnxruntime print directly to stdout while loading models, and
librosa/huggingface_hub/torch emit deprecation warnings on first import. None of
it is actionable for the operator, so it is silenced by default. Set
``ELECCIA_QUIET_STARTUP=0`` (or false/no/off) to restore the full diagnostic
output, e.g. to confirm the CUDA provider or debug a model-load failure.
"""

from __future__ import annotations

import contextlib
import os
import sys
import warnings

_DISABLED = {"0", "false", "no", "off"}


def quiet_enabled() -> bool:
    return os.getenv("ELECCIA_QUIET_STARTUP", "1").strip().lower() not in _DISABLED


def silence_startup_warnings() -> None:
    """Filter chatty deprecation/future warnings and quiet child processes.

    Call this BEFORE importing the heavy modules (insightface, torch, librosa)
    so the warnings raised during their import are filtered.
    """
    if not quiet_enabled():
        return
    for category in (FutureWarning, DeprecationWarning, UserWarning):
        warnings.filterwarnings("ignore", category=category)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # The vision subprocess inherits this env and starts already quiet.
    os.environ.setdefault("PYTHONWARNINGS", "ignore")


@contextlib.contextmanager
def suppress_native_stdout():
    """Silence direct print()s (InsightFace/onnxruntime) for the wrapped block.

    Errors still surface: stdout/stderr are restored in ``finally`` before any
    exception propagates, so tracebacks remain visible.
    """
    if not quiet_enabled():
        yield
        return
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err
