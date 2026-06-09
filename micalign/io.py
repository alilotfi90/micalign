"""
micalign.io
Write the aligned multi-band reflectance stack and a QC sidecar.

Output per capture:
  * {capture}_aligned.tif  — float32, band-sequential multi-band raster
                             (planarconfig='separate' so GDAL/QGIS open it as
                             ONE N-band image, not N single-band images), with
                             band descriptions in MS→reference order.
  * {capture}_align_qc.json — per-band alignment method, cc, phase response,
                             residual shift, score, and degraded flags.
"""

import os
import json
import numpy as np
import tifffile
import imageio.v2 as imageio

from . import utils as _utils


def save_rgb_preview(stack: np.ndarray,
                     output_dir: str,
                     capture_name: str,
                     cir: bool = False) -> str:
    """
    Write a stretched 8-bit quick-look PNG next to the float TIFF, using the
    same global p2-p98 stretch as the original micasharpen RGB output. This is
    for visual inspection only — the quantitative product is the float TIFF.

    cir=False -> natural colour (Red/Green/Blue);  cir=True -> NIR/Red/Green.
    """
    os.makedirs(output_dir, exist_ok=True)
    suffix = "cir" if cir else "rgb"
    path = os.path.join(output_dir, f"{capture_name}_{suffix}.png")
    img = _utils.to_cir(stack) if cir else _utils.to_rgb(stack)
    imageio.imwrite(path, img)
    return path



def save_aligned_stack(stack: np.ndarray,
                       band_names,
                       output_dir: str,
                       capture_name: str) -> str:
    """Write the (H, W, C) stack as a single multi-band float32 GeoTIFF-style TIFF."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{capture_name}_aligned.tif")
    band_seq = np.moveaxis(stack, 2, 0).astype(np.float32)   # (C, H, W)
    tifffile.imwrite(
        path,
        band_seq,
        photometric="minisblack",
        planarconfig="separate",
        metadata={"axes": "CYX", "bands": list(band_names)},
    )
    return path


def save_qc(results,
            output_dir: str,
            capture_name: str,
            band_names,
            output_shape,
            reference_index: int,
            interpolation_name: str,
            calibration_name: str) -> str:
    """Write the per-capture alignment QC JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{capture_name}_align_qc.json")
    bands = []
    n_degraded = 0
    for r in results:
        if getattr(r, "degraded", False):
            n_degraded += 1
        bands.append({
            "band_index": r.band_index,
            "band_name": r.band_name,
            "method": r.method,
            "alignment_score_ncc": r.score,
            "score_factory_ncc": r.score_factory,
            "ecc_cc": r.cc,
            "phase_response": r.phase_response,
            "residual_shift_px": [r.residual_tx, r.residual_ty],
            "degraded": getattr(r, "degraded", False),
            "notes": getattr(r, "notes", ""),
        })
    doc = {
        "capture": capture_name,
        "calibration": calibration_name,
        "reference_band_index": reference_index,
        "output_shape_CHW": list(output_shape),
        "interpolation": interpolation_name,
        "band_order": list(band_names),
        "n_bands_degraded": n_degraded,
        "bands": bands,
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return path
