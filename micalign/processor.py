"""
micalign.processor
Folder-level driver: point it at a folder of MicaSense captures and a
calibration capture; it writes an aligned multi-band reflectance TIFF
(+ QC JSON) for every capture in the folder.

Example
-------
    from micalign import AlignmentProcessor

    proc = AlignmentProcessor(
        input_dir   = "sample images 4/sample images",   # IMG_0010_1..7.tif (+ more)
        calibration = "sample images 4/caliberation",    # IMG_0000_1..7.tif
        output_dir  = "aligned_out",
    )
    proc.run()
"""

import os
import re
import glob
import time
import gc
from collections import defaultdict
from typing import List, Optional

import numpy as np
import cv2

from ._setup import load_micasense_capture
from .calibration import load_panel_irradiance
from .aligner import align_capture
from .utils import crop_to_valid
from . import io as _io


_INTERP = {
    "cubic": cv2.INTER_CUBIC,
    "linear": cv2.INTER_LINEAR,
    "lanczos": cv2.INTER_LANCZOS4,
}


def _group_captures(input_dir: str):
    """
    Group IMG_xxxx_*.tif files in a folder by capture id.
    Returns dict {capture_name: [sorted band files (suffix 1..6)]}.
    Band 7 (thermal LWIR) is excluded from alignment.
    """
    files = sorted(glob.glob(os.path.join(input_dir, "IMG_*.tif")))
    groups = defaultdict(list)
    for f in files:
        m = re.match(r"(IMG_\d{4})_(\d+)\.tif$", os.path.basename(f), re.IGNORECASE)
        if not m:
            continue
        cap_id, suffix = m.group(1), int(m.group(2))
        if 1 <= suffix <= 6:                 # optical bands only
            groups[cap_id].append((suffix, f))
    out = {}
    for cap_id, items in groups.items():
        items.sort(key=lambda t: t[0])
        out[cap_id] = [f for _, f in items]
    return dict(sorted(out.items()))


class AlignmentProcessor:
    """
    Parameters
    ----------
    input_dir       : folder of captures (each IMG_xxxx_1..7.tif). All captures
                      found are processed.
    calibration     : the reflectance-panel capture (folder, glob, or one band
                      file). Its panel irradiance is shared by every capture.
    output_dir      : where aligned TIFFs and QC JSONs are written.
    reference_index : band aligned to (default 5 = panchromatic; highest
                      resolution and best texture for robust alignment).
    use_ecc         : try ECC refinement (default True). ECC is used only when
                      it converges, is plausible, and scores better than the
                      phase-correlation fallback; otherwise the validated phase
                      translation (or factory) is used. Set False for a faster,
                      phase-only run.
    interpolation   : 'cubic' (default, best for quantitative reflectance),
                      'linear', or 'lanczos'.
    crop            : crop each output to the region where all bands overlap
                      (default True).
    save_rgb        : write the stretched 8-bit RGB PNG (default True) — the
                      YOLO/annotation-ready image, using the same global stretch
                      as the original micasharpen RGB output.
    save_tif        : also write the float32 reflectance multi-band TIFF
                      (default False — only needed for a reflectance/NDVI/SR
                      pipeline, not for YOLO).
    cir_preview     : also write a colour-infrared PNG (NIR/Red/Green) (default
                      False).
    save_qc         : write the per-capture alignment QC JSON (default True;
                      tiny, lets you skip any frame that didn't align cleanly).
    vendor_dir      : override the vendored MicaSense location (optional).
    """

    def __init__(self,
                 input_dir: str,
                 calibration: str,
                 output_dir: str,
                 reference_index: int = 5,
                 use_ecc: bool = True,
                 interpolation: str = "cubic",
                 crop: bool = True,
                 save_rgb: bool = True,
                 save_tif: bool = False,
                 cir_preview: bool = False,
                 save_qc: bool = True,
                 vendor_dir: Optional[str] = None,
                 verbose: bool = True):
        self.input_dir = input_dir
        self.calibration = calibration
        self.output_dir = output_dir
        self.reference_index = reference_index
        self.use_ecc = use_ecc
        if interpolation not in _INTERP:
            raise ValueError(f"interpolation must be one of {list(_INTERP)}")
        self.interpolation_name = interpolation
        self.interpolation = _INTERP[interpolation]
        self.crop = crop
        self.save_rgb = save_rgb
        self.save_tif = save_tif
        self.cir_preview = cir_preview
        self.save_qc = save_qc
        self.vendor_dir = vendor_dir
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def run(self) -> dict:
        """Process every capture in ``input_dir``. Returns a summary dict."""
        t0 = time.time()
        capture_mod = load_micasense_capture(self.vendor_dir)

        if not os.path.isdir(self.input_dir):
            raise NotADirectoryError(f"input_dir is not a folder: {self.input_dir}")
        groups = _group_captures(self.input_dir)
        if not groups:
            raise FileNotFoundError(
                f"No IMG_xxxx_*.tif captures found in {self.input_dir}")

        self._log(f"Loading calibration from: {self.calibration}")
        irradiance, cal_name = load_panel_irradiance(
            self.calibration, self.vendor_dir, verbose=self.verbose)

        self._log(f"\nFound {len(groups)} capture(s) in {self.input_dir}\n")
        summary = {"calibration": cal_name, "captures": [], "output_dir": self.output_dir}

        for cap_id, files in groups.items():
            t = time.time()
            self._log(f"[{cap_id}] {len(files)} bands")
            try:
                if len(files) < 6:
                    self._log(f"  SKIP — only {len(files)} optical bands (<6)")
                    summary["captures"].append({"capture": cap_id, "status": "skipped",
                                                 "reason": "fewer than 6 optical bands"})
                    continue

                cap = capture_mod.Capture.from_filelist(files)
                stack, names, results = align_capture(
                    cap, irradiance,
                    reference_index=self.reference_index,
                    use_ecc=self.use_ecc,
                    interpolation=self.interpolation,
                    verbose=self.verbose,
                )

                if self.crop:
                    stack, (r0, c0) = crop_to_valid(stack)

                n_deg = sum(1 for r in results if getattr(r, "degraded", False))
                cap_summary = {
                    "capture": cap_id, "status": "ok",
                    "shape": list(stack.shape), "degraded_bands": n_deg,
                }

                if self.save_rgb:
                    cap_summary["rgb_png"] = _io.save_rgb_preview(
                        stack, self.output_dir, cap_id)
                if self.cir_preview:
                    cap_summary["cir_png"] = _io.save_rgb_preview(
                        stack, self.output_dir, cap_id, cir=True)
                if self.save_tif:
                    cap_summary["aligned_tif"] = _io.save_aligned_stack(
                        stack, names, self.output_dir, cap_id)
                if self.save_qc:
                    cap_summary["qc_json"] = _io.save_qc(
                        results, self.output_dir, cap_id, names,
                        output_shape=tuple(np.moveaxis(stack, 2, 0).shape),
                        reference_index=self.reference_index,
                        interpolation_name=self.interpolation_name,
                        calibration_name=cal_name,
                    )

                made = [k.split("_")[-1] for k in cap_summary if k.endswith(("_png", "_tif"))]
                self._log(f"  -> {cap_id}: {', '.join(made) or 'nothing'}  "
                          f"shape={stack.shape}  degraded_bands={n_deg}  "
                          f"({time.time()-t:.1f}s)\n")
                summary["captures"].append(cap_summary)
                del cap, stack, results
                gc.collect()
            except Exception as e:
                self._log(f"  ERROR — {type(e).__name__}: {e}\n")
                summary["captures"].append({"capture": cap_id, "status": "error",
                                            "error": f"{type(e).__name__}: {e}"})
                gc.collect()

        ok = sum(1 for c in summary["captures"] if c["status"] == "ok")
        self._log(f"Done: {ok}/{len(groups)} captures aligned in {time.time()-t0:.1f}s "
                  f"-> {self.output_dir}")
        summary["elapsed_s"] = round(time.time() - t0, 1)
        return summary


def align_folder(input_dir: str, calibration: str, output_dir: str, **kwargs) -> dict:
    """Convenience wrapper around AlignmentProcessor(...).run()."""
    return AlignmentProcessor(input_dir, calibration, output_dir, **kwargs).run()
