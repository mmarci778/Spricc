"""Stage 3 - turn raw masks from ANY backend into clean, stable `holds`.

Every backend (classical, SAM, YOLO) emits a list of `Candidate` masks. This
module is the single place that decides which of them are real holds:

* removes bare wall, tiny specks and giant wall-sized masks
* removes duplicates (same hold found twice)
* resolves parent/child masks:
    - a child that is a *sub-part* of a hold (chalk patch, glossy highlight,
      second colour of a two-tone hold)  ->  dropped, parent kept
    - a child that is a *separate object sitting on top* (screw-on chip,
      foot hold bolted onto a volume)    ->  kept as its own hold
* smooths contours into simplified polygons
* computes a label anchor inside the *visible* part of each hold
* assigns deterministic ids (top-to-bottom, left-to-right)

Masks are stored cropped to their bounding box so that ~1000 SAM proposals fit
in memory.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import wall_fraction


@dataclass
class Candidate:
    x0: int
    y0: int
    mask: np.ndarray          # bool, cropped to bbox
    score: float = 1.0

    @property
    def x1(self) -> int:
        return self.x0 + self.mask.shape[1]

    @property
    def y1(self) -> int:
        return self.y0 + self.mask.shape[0]

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def window(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        """Return this mask restricted to the window (x0,y0,x1,y1), as bool array."""
        out = np.zeros((y1 - y0, x1 - x0), bool)
        ix0, iy0, ix1, iy1 = max(x0, self.x0), max(y0, self.y0), min(x1, self.x1), min(y1, self.y1)
        if ix1 > ix0 and iy1 > iy0:
            out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = self.mask[iy0 - self.y0:iy1 - self.y0,
                                                                   ix0 - self.x0:ix1 - self.x0]
        return out

    def full(self, shape) -> np.ndarray:
        return self.window(0, 0, shape[1], shape[0])


def candidate_from_full(mask: np.ndarray, score: float = 1.0) -> Candidate | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return Candidate(int(x0), int(y0), mask[y0:y1, x0:x1].copy(), score)


def labels_to_candidates(labels: np.ndarray) -> list[Candidate]:
    from scipy import ndimage as ndi
    out = []
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        m = labels[sl] == i
        if m.any():
            out.append(Candidate(sl[1].start, sl[0].start, m, 1.0))
    return out


# --------------------------------------------------------------------------
# Hierarchy / duplicate resolution (needed for SAM & YOLO tiles, which return
# overlapping masks at several granularities)
# --------------------------------------------------------------------------
def _child_contrast(lab_img: np.ndarray, P: Candidate, C: Candidate, grow: int = 5) -> float | None:
    """Lab distance between child C and the ring of parent P that surrounds it."""
    H, W = lab_img.shape[:2]
    X0, Y0 = max(C.x0 - grow - 1, 0), max(C.y0 - grow - 1, 0)
    X1, Y1 = min(C.x1 + grow + 1, W), min(C.y1 + grow + 1, H)
    cm = C.window(X0, Y0, X1, Y1)
    ring = cv2.dilate(cm.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=grow).astype(bool)
    ring &= ~cm & P.window(X0, Y0, X1, Y1)
    if cm.sum() == 0 or ring.sum() < 20:
        return None
    sub = lab_img[Y0:Y1, X0:X1].astype(np.float32)
    return float(np.linalg.norm(sub[cm].mean(0) - sub[ring].mean(0)))


def _is_soft_chalk(lab_img: np.ndarray, C: Candidate, soft_ratio: float = 0.16,
                   chalk_chroma: float = 26.0, chalk_L: float = 150.0) -> bool:
    """True if C looks like a chalk / gloss patch rather than a separate object.

    Chalk is (a) bright and almost colourless and (b) *feathered*: its lightness
    changes gradually across many pixels. A bolted-on chip has a crisp edge.
    Edge sharpness = mean |dL/dx| along the mask boundary divided by the total
    lightness step: ~0.5 for a 1-2 px edge, <0.15 for a soft ramp.
    """
    H, W = lab_img.shape[:2]
    X0, Y0, X1, Y1 = max(C.x0 - 8, 0), max(C.y0 - 8, 0), min(C.x1 + 8, W), min(C.y1 + 8, H)
    cm = C.window(X0, Y0, X1, Y1)
    sub = lab_img[Y0:Y1, X0:X1].astype(np.float32)
    inside = sub[cm]
    if len(inside) < 20:
        return False
    chroma = float(np.hypot(inside[:, 1] - 128, inside[:, 2] - 128).mean())
    if chroma > chalk_chroma or float(inside[:, 0].mean()) < chalk_L:
        return False
    k3 = np.ones((3, 3), np.uint8)
    cmu = cm.astype(np.uint8)
    band = cv2.dilate(cmu, k3, iterations=2).astype(bool) & ~cv2.erode(cmu, k3, iterations=2).astype(bool)
    outside = cv2.dilate(cmu, k3, iterations=8).astype(bool) & ~cv2.dilate(cmu, k3, iterations=4).astype(bool)
    if outside.sum() < 10:
        return False
    L = cv2.GaussianBlur(sub[..., 0], (0, 0), 1.0)
    g = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1)) / 8.0   # per-pixel slope
    step = abs(float(inside[:, 0].mean()) - float(sub[..., 0][outside].mean()))
    if step < 4:
        return True                                   # no real edge at all
    return float(g[band].mean()) / step < soft_ratio


def resolve_hierarchy(
    cands: list[Candidate],
    lab_img: np.ndarray,
    dup_iou: float = 0.85,
    contain_frac: float = 0.90,
    chip_max_rel_area: float = 0.35,
    chip_min_contrast: float = 32.0,
) -> list[Candidate]:
    """Keep the right level of granularity from overlapping proposals.

    A contained child is kept as its own hold only if it is small relative to
    the parent, differs clearly in colour from the ring of parent around it
    (chip_min_contrast, a Lab distance) and is NOT a soft-edged chalk patch.
    These are zero-shot heuristics. Two-tone holds are the known failure case
    (the second colour can look like a chip); a model fine-tuned on your gym's
    holds learns this distinction directly and should replace them.
    """
    order = sorted(range(len(cands)), key=lambda i: -cands[i].score)
    alive = np.ones(len(cands), bool)
    bb = np.array([[c.x0, c.y0, c.x1, c.y1] for c in cands]) if cands else np.zeros((0, 4))
    areas = np.array([c.area for c in cands])

    for ii, i in enumerate(order):
        if not alive[i]:
            continue
        A = cands[i]
        for j in order[ii + 1:]:
            if not alive[j] or not alive[i]:
                continue
            B = cands[j]
            x0, y0 = max(bb[i, 0], bb[j, 0]), max(bb[i, 1], bb[j, 1])
            x1, y1 = min(bb[i, 2], bb[j, 2]), min(bb[i, 3], bb[j, 3])
            if x1 <= x0 or y1 <= y0:
                continue
            wa, wb = A.window(x0, y0, x1, y1), B.window(x0, y0, x1, y1)
            inter = int((wa & wb).sum())
            if inter == 0:
                continue
            union = areas[i] + areas[j] - inter
            if inter / union >= dup_iou:                      # same object found twice
                alive[j] = False
                continue
            big, small = (i, j) if areas[i] >= areas[j] else (j, i)
            if inter / areas[small] >= contain_frac:          # one inside the other
                P, C = cands[big], cands[small]
                contrast = None
                if areas[small] / areas[big] <= chip_max_rel_area:
                    contrast = _child_contrast(lab_img, P, C)
                if contrast is None or contrast < chip_min_contrast or _is_soft_chalk(lab_img, C):
                    alive[small] = False                      # sub-part of the parent: drop it
                    if small == i:
                        break
            elif inter / areas[small] > 0.5:                  # heavy partial overlap: keep higher score
                alive[j] = False
            # light overlap (<50 %): kept; the visible-area step gives the overlap to the smaller hold
    return [c for c, a in zip(cands, alive) if a]


def filter_candidates(cands, lab_img, wall_ab, min_area_frac=2e-4, max_area_frac=0.12, wall_frac=0.55):
    H, W = lab_img.shape[:2]
    lo, hi = min_area_frac * H * W, max_area_frac * H * W
    out = []
    for c in cands:
        if not (lo <= c.area <= hi):
            continue
        sub = lab_img[c.y0:c.y1, c.x0:c.x1]
        if wall_fraction(sub, c.mask, wall_ab) >= wall_frac:
            continue
        out.append(c)
    return out


# --------------------------------------------------------------------------
# Candidates -> holds
# --------------------------------------------------------------------------
def _clean_polygon(c: Candidate, simplify: float, smooth: int = 3):
    """Smooth the mask, fill holes, return simplified outer polygon in full-image coords."""
    pad = smooth + 2
    m = np.pad(c.mask.astype(np.uint8), pad)
    if smooth > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * smooth + 1, 2 * smooth + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # outer contour only => holes filled
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    cnt = cv2.approxPolyDP(cnt, simplify, True)
    if len(cnt) < 3:
        return None
    pts = cnt[:, 0, :].astype(np.int32) - pad + np.array([c.x0, c.y0])
    return pts


def build_holds(cands: list[Candidate], bgr: np.ndarray, simplify: float | None = None) -> list[dict]:
    H, W = bgr.shape[:2]
    simplify = simplify if simplify is not None else max(1.0, 0.0008 * max(H, W))
    polys = []
    for c in cands:
        p = _clean_polygon(c, simplify)
        if p is not None:
            polys.append(p)
    # larger first; smaller (nested) holds painted later so they own their pixels
    polys.sort(key=lambda p: -cv2.contourArea(p.astype(np.float32)))
    owner = np.zeros((H, W), np.int32)
    for idx, p in enumerate(polys, start=1):
        cv2.fillPoly(owner, [p.reshape(-1, 1, 2)], idx)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    holds = []
    for idx, p in enumerate(polys, start=1):
        x, y, w, h = cv2.boundingRect(p)
        vis = owner[y:y + h, x:x + w] == idx
        if vis.sum() < 30:                       # completely covered by nested holds
            continue
        dt = cv2.distanceTransform(np.pad(vis.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        ay, ax = np.unravel_index(dt.argmax(), dt.shape)
        col = np.median(rgb[y:y + h, x:x + w][vis], axis=0).astype(int)
        m = cv2.moments(p.reshape(-1, 1, 2).astype(np.float32))
        cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (x + w / 2, y + h / 2)
        holds.append({
            "polygon": p.tolist(),
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": int(cv2.contourArea(p.astype(np.float32))),
            "centroid": [round(cx, 1), round(cy, 1)],
            "anchor": [int(x + ax), int(y + ay)],
            "color": "#%02x%02x%02x" % tuple(col),
        })
    # deterministic ids: reading order by rows of ~ the median hold height
    band = max(40, int(np.median([hd["bbox"][3] for hd in holds]))) if holds else 40
    holds.sort(key=lambda hd: (hd["anchor"][1] // band, hd["anchor"][0]))
    for i, hd in enumerate(holds, start=1):
        hd["id"] = i
    return holds
