"""
micalign
========
Robust band-to-reference alignment for MicaSense Altum-PT / RedEdge-P captures.

Point it at a folder of 7-band captures and a reflectance-panel calibration
capture; it writes a co-registered, radiometrically-calibrated multi-band
reflectance TIFF (+ alignment QC JSON) for every capture.

    from micalign import AlignmentProcessor
    AlignmentProcessor(
        input_dir   = "sample images/samples",
        calibration = "sample images/caliberation",
        output_dir  = "aligned_out",
    ).run()

Key design points
-----------------
* Single composed resample (lens undistortion + RigRelatives + residual) per
  band, INTER_CUBIC by default — removes the cumulative blur of the original
  three-stage linear resampling.
* Robust residual alignment: ECC refinement when it converges and helps,
  otherwise a validated phase-correlation translation; the best candidate is
  chosen by an alignment score (gradient NCC vs the reference). Failures are
  recorded, never silent.
"""

from .processor import AlignmentProcessor, align_folder

__version__ = "0.1.0"
__all__ = ["AlignmentProcessor", "align_folder"]
