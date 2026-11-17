"""Stage 2b - classical fallback detector (no GPU, no model weights).

This is a baseline, not the recommended production path (see sam_backend.py).
It over-segments the image with a marker-controlled watershed on a colour
gradient, then merges neighbouring regions whose shared boundary is weak.

Cues used
---------
* Chroma gradient (a, b of Lab) - robust to shadows and highlights.
* Lightness gradient, down-weighted - so a specular highlight or a chalk
  patch does not look like a hold boundary on its own.
* Dark contact lines - holds cast thin shadows on their neighbours; a
  black-hat transform turns these into extra boundary evidence.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.graph import merge_hierarchical, rag_boundary
from skimage.morphology import h_minima
from skimage.segmentation import watershed
from scipy import ndimage as ndi

from skimage.segmentation import relabel_sequential

from .preprocess import (edge_preserving_smooth, estimate_wall_ab, normalize_illumination,
                         resize_long_side, wall_fraction)


def boundary_strength(lab: np.ndarray, w_light: float = 0.5, w_dark: float = 1.0) -> np.ndarray:
    """Per-pixel boundary evidence in [0, ~1]."""
    lab_f = lab.astype(np.float32)
    L, a, b = lab_f[..., 0], lab_f[..., 1] - 128.0, lab_f[..., 2] - 128.0

    def sobel_mag(ch):
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        return np.hypot(gx, gy)

    g_chroma = np.hypot(sobel_mag(a), sobel_mag(b))
    g_light = sobel_mag(L) * w_light

    # thin dark lines (contact shadows between holds)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dark = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, k) * w_dark

    g = np.sqrt(g_chroma ** 2 + g_light ** 2) + dark
    g = cv2.GaussianBlur(g, (0, 0), 0.8)
    return g / (np.percentile(g, 99) + 1e-6)


def _weight_boundary(graph, src, dst, n):
    default = {"weight": 0.0, "count": 0}
    cs = graph[src].get(n, default)["count"]
    cd = graph[dst].get(n, default)["count"]
    ws = graph[src].get(n, default)["weight"]
    wd = graph[dst].get(n, default)["weight"]
    count = cs + cd
    return {"count": count, "weight": (cs * ws + cd * wd) / count}


def _merge_boundary(graph, src, dst):
    pass


def remove_wall(labels: np.ndarray, lab_img: np.ndarray, wall_ab, frac: float = 0.55) -> np.ndarray:
    """Set regions that are mostly bare-wall coloured to background (0)."""
    out = labels.copy()
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        m = labels[sl] == i
        if wall_fraction(lab_img[sl], m, wall_ab) >= frac:
            out[sl][m] = 0
    return out


def merge_enclosed_chalk(labels: np.ndarray, lab_img: np.ndarray, enclosed_frac: float = 0.75,
                         max_rel_area: float = 0.6, chalk_chroma: float = 24.0,
                         chalk_L: float = 150.0) -> np.ndarray:
    """Absorb chalk / gloss patches into the hold that surrounds them.

    A region is treated as a chalk patch when it is (a) bright and nearly
    colourless and (b) mostly enclosed by ONE other region that is clearly
    larger. Enclosed regions that are *coloured* or dark (small screw-on
    chips, bolt-on feet) are kept as independent holds.
    """
    labels, _, _ = relabel_sequential(labels)
    k = np.ones((3, 3), np.uint8)
    for _ in range(3):  # a couple of passes handles chalk islands inside chalk islands
        changed = False
        areas = np.bincount(labels.ravel())
        for i, sl in enumerate(ndi.find_objects(labels), start=1):
            if sl is None or areas[i] == 0:
                continue
            pad = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in sl)
            lab_c, crop = lab_img[pad], labels[pad]
            m = crop == i
            if not m.any():
                continue
            px = lab_c[m].astype(np.float32)
            chroma = float(np.hypot(px[:, 1] - 128, px[:, 2] - 128).mean())
            if chroma > chalk_chroma or px[:, 0].mean() < chalk_L:
                continue
            ring = cv2.dilate(m.astype(np.uint8), k, iterations=2).astype(bool) & ~m
            nb = crop[ring]
            if len(nb) == 0:
                continue
            counts = np.bincount(nb[nb > 0], minlength=1) if (nb > 0).any() else np.zeros(1, int)
            top = int(counts.argmax())
            if top == 0 or counts[top] / len(nb) < enclosed_frac:
                continue
            if areas[i] > max_rel_area * areas[top]:
                continue
            labels[labels == i] = top
            areas[top] += areas[i]
            areas[i] = 0
            changed = True
        if not changed:
            break
    return labels


def detect_classical(
    bgr: np.ndarray,
    work_long_side: int = 1600,
    h: float = 0.10,
    merge_thresh: float = 0.50,
    min_area_frac: float = 2e-4,
    return_debug: bool = False,
):
    """Returns (label_map at original resolution, scores). Label 0 = background.

    h             - depth of gradient minima used as watershed seeds (lower = more seeds)
    merge_thresh  - merge neighbours while their mean shared-boundary strength is below this
    min_area_frac - discard regions smaller than this fraction of the image
    """
    small, scale = resize_long_side(bgr, work_long_side)
    norm = normalize_illumination(small)
    smooth = edge_preserving_smooth(norm)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)

    grad = boundary_strength(lab)

    seeds = h_minima(cv2.GaussianBlur(grad, (0, 0), 1.5), h)
    markers, _ = ndi.label(seeds)
    over = watershed(grad, markers)

    rag = rag_boundary(over, grad, connectivity=2)
    merged = merge_hierarchical(
        over, rag, thresh=merge_thresh, rag_copy=False, in_place_merge=True,
        merge_func=_merge_boundary, weight_func=_weight_boundary,
    )

    wall_ab = estimate_wall_ab(small)
    merged = remove_wall(merged, lab, wall_ab)
    merged = merge_enclosed_chalk(merged, lab)

    # drop tiny regions
    min_area = int(min_area_frac * merged.size)
    ids, counts = np.unique(merged, return_counts=True)
    small_ids = ids[counts < min_area]
    merged = np.where(np.isin(merged, small_ids), 0, merged)

    if scale != 1.0:
        merged = cv2.resize(merged.astype(np.int32), (bgr.shape[1], bgr.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    if return_debug:
        return merged, {"grad": grad, "over": over, "smooth": smooth, "scale": scale, "wall_ab": wall_ab}
    return merged
