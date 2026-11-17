"""Stage 1 - image preparation.

Goals
-----
* Make the image resolution predictable (detectors are tuned for ~1500-2000 px).
* Reduce the influence of uneven lighting, so a hold in a shadow and the same
  hold in a bright spot produce similar descriptors.
* Suppress chalk speckle / texture without blurring real boundaries.
"""
from __future__ import annotations

import cv2
import numpy as np


def resize_long_side(bgr: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is `long_side`. Never upscales.
    Returns (image, scale) where scale = new_size / old_size."""
    h, w = bgr.shape[:2]
    scale = min(1.0, long_side / max(h, w))
    if scale == 1.0:
        return bgr, 1.0
    out = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return out, scale


def normalize_illumination(bgr: np.ndarray, clip_limit: float = 2.0, flatten_sigma_frac: float = 0.15) -> np.ndarray:
    """Illumination normalisation that leaves chroma alone.

    1. Estimate the slowly varying illumination of the L channel with a very
       wide Gaussian and divide it out (a cheap homomorphic / retinex step).
       The sigma is large (15% of the image) so individual holds are NOT
       flattened, only wall-scale gradients such as a lamp falling off.
    2. Apply CLAHE to the corrected L channel to lift shadowed holds.
    Colour (a, b) channels are untouched, so hue stays a usable cue.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    sigma = max(bgr.shape[:2]) * flatten_sigma_frac
    illum = cv2.GaussianBlur(L, (0, 0), sigma)
    flat = L / np.maximum(illum, 1.0) * float(illum.mean())
    flat = np.clip(flat, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(flat)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def edge_preserving_smooth(bgr: np.ndarray, spatial: int = 9, color: int = 16) -> np.ndarray:
    """Mean-shift filtering flattens chalk dust, plastic texture and the
    speckle on glossy holds into near-constant colour patches, while keeping
    real hold boundaries sharp."""
    return cv2.pyrMeanShiftFiltering(bgr, sp=spatial, sr=color, maxLevel=1)


# --------------------------------------------------------------------------
# Wall (plywood) colour model
# --------------------------------------------------------------------------
def estimate_wall_ab(bgr: np.ndarray) -> tuple[int, int]:
    """Estimate the (a, b) Lab colour of the bare wall surface.

    Bare plywood is the only large-ish population of pixels that is both
    mid-saturated beige and consistent in hue across the image, so we look for
    the peak of a 2-D (a, b) histogram restricted to plausible wall tones.
    Print / log the result and override it with --wall-ab if your wall is
    painted or a different material.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), 2), cv2.COLOR_BGR2LAB).astype(np.int32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    m = (L > 140) & (a >= 126) & (a <= 146) & (b >= 138) & (b <= 172)
    if m.sum() < 500:
        return 133, 148  # generic birch-ply default
    hist, _, _ = np.histogram2d(a[m], b[m], bins=[np.arange(126, 148, 2), np.arange(138, 174, 2)])
    i, j = np.unravel_index(hist.argmax(), hist.shape)
    return 126 + 2 * i + 1, 138 + 2 * j + 1


def wall_fraction(lab_img: np.ndarray, mask: np.ndarray, wall_ab: tuple[int, int], tol: int = 8) -> float:
    """Fraction of `mask` pixels whose chroma is within `tol` of the wall colour."""
    px = lab_img[mask]
    if len(px) == 0:
        return 0.0
    d = np.hypot(px[:, 1].astype(np.int32) - wall_ab[0], px[:, 2].astype(np.int32) - wall_ab[1])
    return float(((d <= tol) & (px[:, 0] > 130)).mean())
