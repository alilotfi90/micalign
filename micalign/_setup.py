"""
micalign._setup
Runtime environment fixes that MUST run before the vendored MicaSense
library is imported, plus a one-time importer for that library.

Why this exists
---------------
* NumPy 2.0 removed ``np.mat``; the bundled MicaSense source still uses it.
  ``np.asmatrix`` is semantically identical for their usage, so we alias it.
* OpenCV logs noisy TIFF warnings for MicaSense-specific EXIF tags
  (48020/48021/48022/51022 — rig/vignette metadata libtiff doesn't know).
  They are harmless; we lower the log level.
* The MicaSense library is imported exactly once via a module-level
  singleton to avoid class-identity mismatches from re-importing.
"""

import os
import sys

_MICASENSE_CAPTURE = None
_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")


def _apply_runtime_patches():
    import numpy as np
    if not hasattr(np, "mat"):
        np.mat = np.asmatrix          # NumPy 2.0 shim for vendored MicaSense
    try:
        import cv2
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass


def load_micasense_capture(vendor_dir: str = None):
    """
    Import and return ``micasense.capture`` exactly once.

    Raises a clear, actionable error if the vendored library or its
    system dependencies (exiftool) are missing.
    """
    global _MICASENSE_CAPTURE
    if _MICASENSE_CAPTURE is not None:
        return _MICASENSE_CAPTURE

    _apply_runtime_patches()

    vendor_dir = vendor_dir or _VENDOR_DIR
    capture_py = os.path.join(vendor_dir, "micasense", "capture.py")
    if not os.path.exists(capture_py):
        raise RuntimeError(
            f"Vendored MicaSense library not found at {vendor_dir}/micasense/.\n"
            f"Run `python -m micalign.bootstrap` to download it, or copy the\n"
            f"`micasense/` folder from github.com/micasense/imageprocessing there."
        )

    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)

    try:
        import micasense.capture as capture_mod
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import the vendored MicaSense library: {e}\n"
            f"Python deps: pip install numpy opencv-python tifffile imageio "
            f"scikit-image matplotlib pyexiftool pytz pysolar pyzbar packaging\n"
            f"System deps: exiftool (e.g. `apt install libimage-exiftool-perl`)\n"
            f"             zbar    (e.g. `apt install libzbar0`) — needed for panel detection."
        ) from e

    _MICASENSE_CAPTURE = capture_mod
    return _MICASENSE_CAPTURE
