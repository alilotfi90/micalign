"""
micalign.utils
Shared helpers: normalisation, gradients, alignment scoring, homography
rescaling, and the single-pass composed-remap that does undistortion +
RigRelatives + residual alignment in ONE resample.
"""

import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Basic image helpers
# ─────────────────────────────────────────────────────────────────────────────
def normalize01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise a float array to [0, 1] (safe on flat inputs)."""
    arr = arr.astype(np.float32)
    return cv2.normalize(arr, None, 0.0, 1.0, cv2.NORM_MINMAX)


def gradient_magnitude(img_f32: np.ndarray) -> np.ndarray:
    """
    Sobel gradient magnitude, normalised to [0, 1].

    Edges (leaf boundaries, plot edges, soil/canopy transitions) sit at the
    same pixel location in every spectral band regardless of band intensity,
    so aligning on gradient maps removes the spectral dependence that makes
    raw-intensity matching unreliable across bands.
    """
    gx = cv2.Sobel(img_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f32, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)


def central_crop(img: np.ndarray, frac: float = 0.6) -> np.ndarray:
    """Return the centre `frac` of an image (avoids warp-border effects in scoring)."""
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    return img[r0:r0 + ch, c0:c0 + cw]


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two equal-shape arrays (in [-1, 1])."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Homography rescaling (ECC solved at reduced scale → full resolution)
# ─────────────────────────────────────────────────────────────────────────────
def rescale_homography(H_small: np.ndarray, scale: float) -> np.ndarray:
    """
    Convert a homography estimated at a downsampled resolution back to
    full-resolution coordinates.

        x_small = S @ x_full,   S = diag(scale, scale, 1)
        H_full  = S^-1 @ H_small @ S
    """
    S = np.diag([scale, scale, 1.0]).astype(np.float64)
    S_inv = np.diag([1.0 / scale, 1.0 / scale, 1.0]).astype(np.float64)
    return (S_inv @ H_small.astype(np.float64) @ S).astype(np.float32)


def translation_homography(dx: float, dy: float) -> np.ndarray:
    """3x3 homography for a pure translation (output→source convention)."""
    H = np.eye(3, dtype=np.float32)
    H[0, 2] = dx
    H[1, 2] = dy
    return H


def homography_is_plausible(H: np.ndarray,
                            max_translation: float,
                            scale_tol: float = 0.2,
                            shear_tol: float = 0.1) -> bool:
    """
    Sanity-check a residual homography so a wild ECC/optimiser solution is
    never accepted. Residual (post-RigRelatives) transforms should be close
    to a small translation: near-unit scale, tiny rotation/shear, bounded
    translation, negligible perspective.
    """
    H = np.asarray(H, dtype=np.float64)
    tx, ty = H[0, 2], H[1, 2]
    if (tx * tx + ty * ty) ** 0.5 > max_translation:
        return False
    # scale of the linear part
    sx = (H[0, 0] ** 2 + H[1, 0] ** 2) ** 0.5
    sy = (H[0, 1] ** 2 + H[1, 1] ** 2) ** 0.5
    if not (1 - scale_tol <= sx <= 1 + scale_tol):
        return False
    if not (1 - scale_tol <= sy <= 1 + scale_tol):
        return False
    # off-diagonal (rotation/shear) should be small for a residual
    if abs(H[0, 1]) > shear_tol or abs(H[1, 0]) > shear_tol:
        return False
    # perspective terms negligible
    if abs(H[2, 0]) > 1e-4 or abs(H[2, 1]) > 1e-4:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Single-pass composed resample
# ─────────────────────────────────────────────────────────────────────────────
def undistortion_maps(image_obj):
    """
    Build the (map1, map2) lens-undistortion maps for a MicaSense Image,
    exactly as the vendored library does in ``Image.undistorted``.
    """
    ncm, _ = cv2.getOptimalNewCameraMatrix(
        image_obj.cv2_camera_matrix(),
        image_obj.cv2_distortion_coeff(),
        image_obj.size(),
        1,
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        image_obj.cv2_camera_matrix(),
        image_obj.cv2_distortion_coeff(),
        np.eye(3),
        ncm,
        image_obj.size(),
        cv2.CV_32F,
    )
    return map1, map2


def warp_residual(ms_up: np.ndarray, H_residual: np.ndarray) -> np.ndarray:
    """
    Apply a residual homography (output→source convention) to an already
    factory-warped band. Used only for scoring candidates; the final output
    uses ``compose_single_remap`` instead.
    """
    h, w = ms_up.shape
    if np.allclose(H_residual, np.eye(3)):
        return ms_up
    return cv2.warpPerspective(
        ms_up, H_residual.astype(np.float32), (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )


def compose_single_remap(src_native: np.ndarray,
                         rig_matrix: np.ndarray,
                         H_residual: np.ndarray,
                         map1: np.ndarray,
                         map2: np.ndarray,
                         out_hw,
                         grid_xy,
                         interpolation: int = cv2.INTER_CUBIC,
                         border_value: float = 0.0) -> np.ndarray:
    """
    Resample a native-resolution band to the reference grid in ONE pass,
    composing lens undistortion + RigRelatives + the residual alignment.

    Geometry (all in output→source direction):
        output pixel p
          --H_residual-->  RigRelatives-warped (ms_up) coords
          --rig_matrix-->  undistorted-MS coords        (ms_up(q) = und(rig·q))
          --(map1,map2)--> raw/native pixel coords
        out(p) = src_native(map[ rig · H_residual · p ])

    Composing the two homographies as M = rig_matrix @ H_residual lets us
    evaluate the undistortion lookup once and resample the source exactly
    once, avoiding the cumulative blur of three separate linear resamples.

    Parameters
    ----------
    src_native : (h_ms, w_ms) float32 — native band BEFORE undistortion
    rig_matrix : (3, 3)               — cap.get_warp_matrices()[i] (MS→ref grid)
    H_residual : (3, 3)               — residual alignment (output→ms_up)
    map1, map2 : undistortion maps for this band (undistorted→native)
    out_hw     : (H_ref, W_ref) output grid size
    grid_xy    : (xs, ys) float32 meshgrids of the output grid (reused per capture)
    """
    h_ref, w_ref = out_hw
    xs, ys = grid_xy                                              # float32 HxW meshgrids
    M = (np.asarray(rig_matrix, np.float64) @ np.asarray(H_residual, np.float64))

    # Apply the composed homography directly on the meshgrids (output→undistorted-MS),
    # avoiding a 3xN coordinate stack + matmul (much lower peak memory per band).
    den = M[2, 0] * xs + M[2, 1] * ys + M[2, 2]
    ux = ((M[0, 0] * xs + M[0, 1] * ys + M[0, 2]) / den).astype(np.float32)
    uy = ((M[1, 0] * xs + M[1, 1] * ys + M[1, 2]) / den).astype(np.float32)
    del den

    # undistorted-MS coords → native (raw) coords via the undistortion maps
    raw_x = cv2.remap(map1, ux, uy, cv2.INTER_LINEAR)
    raw_y = cv2.remap(map2, ux, uy, cv2.INTER_LINEAR)

    out = cv2.remap(
        src_native.astype(np.float32), raw_x, raw_y,
        interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cropping
# ─────────────────────────────────────────────────────────────────────────────
def crop_to_valid(stack: np.ndarray):
    """
    Crop a (H, W, C) stack to the bounding box where ALL bands are non-zero,
    removing the zero border introduced by warping. Returns (cropped, (r0, c0)).
    """
    valid = np.all(stack > 0, axis=2)
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    if rows.sum() < 2 or cols.sum() < 2:
        return stack, (0, 0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return stack[r0:r1 + 1, c0:c1 + 1, :], (int(r0), int(c0))


# ─────────────────────────────────────────────────────────────────────────────
# Display stretch (quick-look previews only — NOT applied to the float TIFF)
# ─────────────────────────────────────────────────────────────────────────────
def stretch(arr: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear global stretch to uint8 [0, 255] — ported verbatim from
    the original micasharpen pipeline (the stretch confirmed most faithful).

    Maps:  [0,   p2]  -> [0,    5.1]    (shadow detail preserved)
           [p2,  p98] -> [5.1,  249.9]  (main range, linear)
           [p98, max] -> [249.9, 255]   (highlight detail preserved)

    p2/p98 are computed jointly over the non-zero pixels of the whole array
    (one shared range across all channels). Zero pixels map to 0 (no-data).
    """
    v = arr[arr > 0]
    if len(v) == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    p2, p98 = np.percentile(v, 2), np.percentile(v, 98)
    o2, o98 = (2 / 100) * 255, (98 / 100) * 255
    if (p98 - p2) < 1e-8:
        vmax = float(arr.max()) if arr.max() > 0 else 1.0
        return np.clip(arr / vmax * 255, 0, 255).astype(np.uint8)
    r = np.zeros_like(arr, dtype=np.float32)
    r[arr <= p2] = (arr[arr <= p2] / (p2 + 1e-8)) * o2
    m = (arr > p2) & (arr <= p98)
    r[m] = o2 + (arr[m] - p2) / (p98 - p2) * (o98 - o2)
    vmax = max(float(arr.max()), p98 + 1e-8)
    r[arr > p98] = o98 + (arr[arr > p98] - p98) / (vmax - p98) * (255 - o98)
    return np.clip(r, 0, 255).astype(np.uint8)


def to_rgb(stack: np.ndarray) -> np.ndarray:
    """Stretched uint8 natural-colour RGB from an aligned stack (bands R=2,G=1,B=0)."""
    return stretch(np.stack([stack[:, :, 2], stack[:, :, 1], stack[:, :, 0]], axis=2))


def to_cir(stack: np.ndarray) -> np.ndarray:
    """Stretched uint8 colour-infrared (NIR/Red/Green); fringing is obvious here."""
    return stretch(np.stack([stack[:, :, 3], stack[:, :, 2], stack[:, :, 1]], axis=2))
