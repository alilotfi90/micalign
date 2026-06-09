"""
micalign.calibration
Load a MicaSense reflectance-panel capture and extract per-band irradiance.

The calibration capture is a single set of band TIFFs imaging the grey
reflectance panel (typically the first capture of a flight, e.g.
``IMG_0000_*.tif``). Its panel-derived irradiance converts every flight
capture from radiance to reflectance.
"""

import os
import glob
import re
from typing import List, Tuple

from ._setup import load_micasense_capture


def _resolve_capture_files(path_or_glob: str) -> List[str]:
    """
    Resolve a calibration argument to the list of band files (1..6) of one
    capture. Accepts:
      * a directory containing exactly one IMG_xxxx capture
      * a glob like '/path/IMG_0000_*.tif'
      * a single band file '/path/IMG_0000_3.tif' (siblings are globbed)
    """
    if os.path.isdir(path_or_glob):
        files = sorted(glob.glob(os.path.join(path_or_glob, "IMG_*.tif")))
        if not files:
            raise FileNotFoundError(
                f"No IMG_*.tif files found in calibration directory: {path_or_glob}"
            )
        prefixes = sorted({_capture_prefix(f) for f in files})
        if len(prefixes) != 1:
            raise ValueError(
                f"Calibration directory contains multiple captures {prefixes}; "
                f"point to one capture (a folder/glob with a single IMG_xxxx set)."
            )
        return [f for f in files if _capture_prefix(f) == prefixes[0]]

    matches = sorted(glob.glob(path_or_glob))
    if matches:
        prefix = _capture_prefix(matches[0])
        base = os.path.dirname(path_or_glob) or "."
        return sorted(glob.glob(os.path.join(base, f"{prefix}_*.tif")))

    raise FileNotFoundError(f"Calibration path matched no files: {path_or_glob}")


def _capture_prefix(path: str) -> str:
    """'/x/IMG_0000_3.tif' -> 'IMG_0000'."""
    base = os.path.basename(path)
    m = re.match(r"(IMG_\d{4})_\d+\.tif$", base, re.IGNORECASE)
    if m:
        return m.group(1)
    return os.path.splitext(base)[0]


def load_panel_irradiance(calibration: str,
                          vendor_dir: str = None,
                          verbose: bool = True) -> Tuple[List[float], str]:
    """
    Detect the reflectance panel in the calibration capture and return its
    per-band irradiance.

    Returns
    -------
    irradiance : list of 6 floats (Blue, Green, Red, NIR, RedEdge, Pan order)
    cal_name   : capture id of the calibration set (e.g. 'IMG_0000')
    """
    capture_mod = load_micasense_capture(vendor_dir)
    files = _resolve_capture_files(calibration)
    optical = [f for f in files if _band_suffix(f) in range(1, 7)]
    optical = sorted(optical, key=_band_suffix)
    if len(optical) < 6:
        raise ValueError(
            f"Calibration capture needs 6 optical bands (suffixes _1.._6); "
            f"found {len(optical)} in {calibration}."
        )

    cal_name = _capture_prefix(optical[0])
    panel = capture_mod.Capture.from_filelist(optical)

    if not panel.detect_panels():
        raise RuntimeError(
            f"Could not detect the reflectance panel in {cal_name}. Check that the "
            f"calibration capture actually images the panel, and that the zbar "
            f"library is installed (panel QR detection needs it)."
        )

    irradiance = panel.panel_irradiance()
    if verbose:
        vals = ", ".join(f"{v:.4f}" for v in irradiance)
        print(f"  Calibration {cal_name}: panel irradiance = [{vals}]")
    return irradiance, cal_name


def _band_suffix(path: str) -> int:
    """'/x/IMG_0000_3.tif' -> 3 (or -1 if not matched)."""
    m = re.search(r"_(\d+)\.tif$", os.path.basename(path), re.IGNORECASE)
    return int(m.group(1)) if m else -1
