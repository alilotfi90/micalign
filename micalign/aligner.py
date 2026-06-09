"""
micalign.aligner
Robust band-to-reference alignment for a single MicaSense capture.

Per band the pipeline is:
  1. RigRelatives (factory) warp seeds a coarse registration to the reference.
  2. The residual is estimated by candidate methods and the BEST is chosen by
     an alignment score (gradient NCC vs the reference), not by a single
     fragile threshold:
        - factory   : residual = identity (RigRelatives only)
        - phase     : phase-correlation translation (robust, always returns)
        - ecc       : ECC homography, seeded from the phase translation
                      (used only if it converges, is plausible, and scores best)
  3. Lens undistortion + RigRelatives + the chosen residual are composed and
     applied in a SINGLE cubic resample (avoids the cumulative blur of the
     three separate linear resamples in the original pipeline).

Failures are never silent: every band records its method, cc, phase response,
residual shift, alignment score, and a 'degraded' flag in the QC dict.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional

from . import utils


# ── tuning defaults ──────────────────────────────────────────────────────────
DEFAULT_ECC_SCALE      = 0.25     # ECC solved at 1/4 resolution (speed)
DEFAULT_ECC_ITERS      = 200
DEFAULT_ECC_EPS        = 1e-6
DEFAULT_GAUSS_FILT      = 5
MIN_PHASE_RESPONSE      = 0.03    # below this, phase peak is untrustworthy
MIN_SCORE_IMPROVEMENT   = 0.005   # residual must beat factory by at least this NCC
MAX_TRANSLATION_FRAC    = 0.15    # residual translation cap, as frac of min(H,W)


@dataclass
class BandAlignResult:
    band_index: int
    band_name: str
    method: str                       # 'ecc' | 'phase' | 'factory'
    score: float                      # gradient NCC vs reference (chosen transform)
    score_factory: float
    cc: Optional[float] = None        # ECC correlation coefficient (if ECC tried)
    phase_response: Optional[float] = None
    residual_tx: float = 0.0
    residual_ty: float = 0.0
    degraded: bool = False            # True if no residual helped but a shift was expected
    notes: str = ""
    _H_residual: np.ndarray = field(default=None, repr=False)


def _try_ecc(ref_grad_small, mov_grad_small, scale,
             seed_dx, seed_dy, iters, eps, gauss):
    """
    Run ECC (homography) on downsampled gradient maps, seeded with the
    phase-correlation translation. Returns (cc, H_full_3x3) or (None, None)
    on non-convergence. Never raises.
    """
    W = np.eye(3, 3, dtype=np.float32)
    W[0, 2] = seed_dx * scale
    W[1, 2] = seed_dy * scale
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        cc, W = cv2.findTransformECC(
            ref_grad_small, mov_grad_small, W,
            cv2.MOTION_HOMOGRAPHY, criteria, None, gauss,
        )
    except cv2.error:
        return None, None
    return float(cc), utils.rescale_homography(W, scale)


def align_band(ms_native: np.ndarray,
               rig_matrix: np.ndarray,
               undist_maps,
               ref_grad: np.ndarray,
               ref_norm_small: np.ndarray,
               out_hw,
               grid_xy,
               hann_window,
               band_index: int,
               band_name: str,
               use_ecc: bool = True,
               ecc_scale: float = DEFAULT_ECC_SCALE,
               interpolation: int = cv2.INTER_CUBIC,
               verbose: bool = True) -> (np.ndarray, BandAlignResult):
    """
    Align one native-resolution band to the reference grid and return the
    aligned band plus a BandAlignResult. See module docstring for the method.

    Parameters
    ----------
    ms_native   : (h_ms, w_ms) float32 — calibrated reflectance, BEFORE undistortion
    rig_matrix  : (3,3) RigRelatives warp matrix for this band (MS→ref grid)
    undist_maps : (map1, map2) lens-undistortion maps for this band
    ref_grad    : (H_ref, W_ref) float32 — reference gradient map (full res)
    ref_norm_small : downsampled, normalised reference (for ECC feature map)
    out_hw      : (H_ref, W_ref)
    grid_xy     : (xs, ys) output meshgrids (reused across bands of a capture)
    hann_window : Hanning window matching the reference size (for phaseCorrelate)
    """
    map1, map2 = undist_maps
    h_ref, w_ref = out_hw
    max_translation = MAX_TRANSLATION_FRAC * min(h_ref, w_ref)

    # factory-warped band (RigRelatives only) — for residual estimation + scoring
    und = cv2.remap(ms_native, map1, map2, cv2.INTER_LINEAR)
    ms_up = cv2.warpPerspective(
        und, np.linalg.inv(rig_matrix.astype(np.float64)).astype(np.float32),
        (w_ref, h_ref), flags=cv2.INTER_LINEAR,
    )
    ms_up_grad = utils.gradient_magnitude(utils.normalize01(ms_up))

    def score(grad_img):
        return utils.ncc(utils.central_crop(grad_img), utils.central_crop(ref_grad))

    # ── candidate: factory (identity residual) ──────────────────────────────
    score_factory = score(ms_up_grad)
    candidates = [("factory", np.eye(3, dtype=np.float32), score_factory, None, None)]

    # ── candidate: phase-correlation translation ────────────────────────────
    (dx, dy), phase_resp = cv2.phaseCorrelate(
        ref_grad.astype(np.float64) * hann_window,
        ms_up_grad.astype(np.float64) * hann_window,
    )
    H_phase = utils.translation_homography(dx, dy)
    if phase_resp >= MIN_PHASE_RESPONSE and (dx * dx + dy * dy) ** 0.5 <= max_translation:
        score_phase = score(utils.gradient_magnitude(
            utils.normalize01(utils.warp_residual(ms_up, H_phase))))
        candidates.append(("phase", H_phase, score_phase, None, phase_resp))

    # ── candidate: ECC (seeded from phase), if enabled ───────────────────────
    cc_val = None
    if use_ecc:
        hs = max(1, int(h_ref * ecc_scale))
        ws = max(1, int(w_ref * ecc_scale))
        ref_small = utils.gradient_magnitude(ref_norm_small)
        mov_small = utils.gradient_magnitude(
            cv2.resize(utils.normalize01(ms_up), (ws, hs), interpolation=cv2.INTER_AREA))
        cc_val, H_ecc = _try_ecc(ref_small, mov_small, ecc_scale,
                                 dx, dy, DEFAULT_ECC_ITERS, DEFAULT_ECC_EPS,
                                 DEFAULT_GAUSS_FILT)
        if (H_ecc is not None
                and utils.homography_is_plausible(H_ecc, max_translation)):
            score_ecc = score(utils.gradient_magnitude(
                utils.normalize01(utils.warp_residual(ms_up, H_ecc))))
            candidates.append(("ecc", H_ecc, score_ecc, cc_val, None))

    # ── choose the best-scoring candidate ────────────────────────────────────
    method, H_res, best_score, cc_used, resp_used = max(candidates, key=lambda c: c[2])

    # Guard: a residual must beat factory by a margin; otherwise keep factory.
    if method != "factory" and best_score < score_factory + MIN_SCORE_IMPROVEMENT:
        method, H_res, best_score = "factory", np.eye(3, dtype=np.float32), score_factory

    tx, ty = float(H_res[0, 2]), float(H_res[1, 2])

    # Degraded if we fell back to factory yet phase says the band is well off:
    degraded = (method == "factory"
                and phase_resp >= MIN_PHASE_RESPONSE
                and (dx * dx + dy * dy) ** 0.5 > 2.0)
    notes = ""
    if degraded:
        notes = (f"residual not refined (factory kept), but phase suggests "
                 f"~{(dx*dx+dy*dy)**0.5:.1f}px offset — alignment may be degraded")

    # ── final output: ONE composed cubic resample from native band ───────────
    aligned = utils.compose_single_remap(
        ms_native, rig_matrix, H_res, map1, map2, out_hw, grid_xy,
        interpolation=interpolation,
    )

    res = BandAlignResult(
        band_index=band_index, band_name=band_name, method=method,
        score=round(best_score, 4), score_factory=round(score_factory, 4),
        cc=(None if cc_val is None else round(cc_val, 4)),
        phase_response=round(float(phase_resp), 4),
        residual_tx=round(tx, 3), residual_ty=round(ty, 3),
        degraded=degraded, notes=notes, _H_residual=H_res,
    )
    if verbose:
        flag = "  [DEGRADED]" if degraded else ""
        cc_s = f"cc={cc_val:.3f} " if cc_val is not None else "cc=--- "
        print(f"    band {band_index} {band_name:9s}: method={method:7s} "
              f"score={best_score:.3f} {cc_s}phase_resp={phase_resp:.3f} "
              f"shift=({tx:+.1f},{ty:+.1f})px{flag}")
    return aligned, res


def align_capture(cap,
                  panel_irradiance: List[float],
                  reference_index: int = 5,
                  use_ecc: bool = True,
                  ecc_scale: float = DEFAULT_ECC_SCALE,
                  interpolation: int = cv2.INTER_CUBIC,
                  verbose: bool = True):
    """
    Align all optical bands of a capture to the reference band.

    Returns
    -------
    stack   : (H_ref, W_ref, n_bands) float32 reflectance, reference band last
    names   : list of band names in stack order
    results : list of BandAlignResult (reference band's entry is the identity)
    """
    n = len(cap.images)
    if n < 6:
        raise ValueError(
            f"Capture has {n} bands; need at least 6 (5 MS + panchromatic). "
            f"Supported cameras: Altum-PT, RedEdge-P."
        )
    band_names = [str(b) for b in cap.band_names()]

    # reference band: undistorted reflectance + its gradient map
    ref_img = cap.images[reference_index]
    ref_refl = ref_img.undistorted(
        ref_img.reflectance(panel_irradiance[reference_index])).astype(np.float32)
    h_ref, w_ref = ref_refl.shape
    ref_norm = utils.normalize01(ref_refl)
    ref_grad = utils.gradient_magnitude(ref_norm)

    hs = max(1, int(h_ref * ecc_scale))
    ws = max(1, int(w_ref * ecc_scale))
    ref_norm_small = cv2.resize(ref_norm, (ws, hs), interpolation=cv2.INTER_AREA)
    hann = cv2.createHanningWindow((w_ref, h_ref), cv2.CV_32F)

    wm = cap.get_warp_matrices(ref_index=reference_index)
    ys, xs = np.mgrid[0:h_ref, 0:w_ref].astype(np.float32)
    grid_xy = (xs, ys)

    ms_indices = [i for i in range(min(n, 6)) if i != reference_index]
    stack_bands = []
    names = []
    results = []

    for i in ms_indices:
        ms_native = cap.images[i].reflectance(panel_irradiance[i]).astype(np.float32)
        undist = utils.undistortion_maps(cap.images[i])
        aligned, res = align_band(
            ms_native=ms_native,
            rig_matrix=np.array(wm[i], dtype=np.float64),
            undist_maps=undist,
            ref_grad=ref_grad,
            ref_norm_small=ref_norm_small,
            out_hw=(h_ref, w_ref),
            grid_xy=grid_xy,
            hann_window=hann,
            band_index=i,
            band_name=band_names[i],
            use_ecc=use_ecc,
            ecc_scale=ecc_scale,
            interpolation=interpolation,
            verbose=verbose,
        )
        stack_bands.append(aligned)
        names.append(band_names[i])
        results.append(res)
        del ms_native, aligned

    # reference band, undistorted (one resample), placed last
    ref_out = cv2.remap(
        ref_img.reflectance(panel_irradiance[reference_index]).astype(np.float32),
        *utils.undistortion_maps(ref_img), interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    stack_bands.append(ref_out)
    names.append(band_names[reference_index])
    results.append(BandAlignResult(
        band_index=reference_index, band_name=band_names[reference_index],
        method="reference", score=1.0, score_factory=1.0,
        residual_tx=0.0, residual_ty=0.0, _H_residual=np.eye(3, dtype=np.float32)))

    stack = np.stack(stack_bands, axis=2).astype(np.float32)
    return stack, names, results
