"""Stage 2c - fine-tuned instance-segmentation model (Ultralytics YOLO-seg).

This is the production target once you have labelled data from your gym
(see train/README.md). It learns what *a hold* is, including the cases the
zero-shot heuristics get wrong: two-tone holds, chalk, chips, volumes.

Large photos are processed as overlapping tiles plus one downscaled full pass
(so very large volumes, bigger than the tile overlap, are still found whole).
Masks that touch an *interior* tile border are discarded because the
neighbouring tile sees that hold uncut.

NOTE: not executed in the authoring environment (no ultralytics / weights).
"""
from __future__ import annotations

import cv2
import numpy as np

from .postprocess import Candidate, candidate_from_full


def propose_yolo(bgr: np.ndarray, weights: str, tile: int = 1024, overlap: float = 0.3,
                 conf: float = 0.25, iou: float = 0.6, device: str | None = None) -> list[Candidate]:
    from ultralytics import YOLO
    model = YOLO(weights)
    H, W = bgr.shape[:2]
    out: list[Candidate] = []

    def run(img, x_off, y_off, sx, sy, drop_interior_edges, tile_box):
        r = model.predict(img, imgsz=tile, conf=conf, iou=iou, retina_masks=True,
                          max_det=1000, verbose=False, device=device)[0]
        if r.masks is None:
            return
        th, tw = img.shape[:2]
        for poly, c in zip(r.masks.xy, r.boxes.conf.cpu().numpy()):
            if len(poly) < 3:
                continue
            px0, py0 = poly.min(0)
            px1, py1 = poly.max(0)
            if drop_interior_edges:
                m = 3
                if (px0 < m and tile_box[0] > 0) or (py0 < m and tile_box[1] > 0) or \
                   (px1 > tw - m and tile_box[2] < W) or (py1 > th - m and tile_box[3] < H):
                    continue
            full = np.zeros((H, W), np.uint8)
            pts = (poly * np.array([sx, sy]) + np.array([x_off, y_off])).astype(np.int32)
            cv2.fillPoly(full, [pts.reshape(-1, 1, 2)], 1)
            cand = candidate_from_full(full.astype(bool), float(c))
            if cand is not None:
                out.append(cand)

    # 1) one downscaled pass over the whole image (catches very large holds)
    s = min(1.0, tile / max(H, W))
    small = cv2.resize(bgr, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    run(small, 0, 0, 1 / s, 1 / s, False, (0, 0, W, H))

    # 2) overlapping full-resolution tiles
    if max(H, W) > tile:
        step = int(tile * (1 - overlap))
        for y in list(range(0, max(H - tile, 0) + 1, step)) + ([H - tile] if H > tile and (H - tile) % step else []):
            for x in list(range(0, max(W - tile, 0) + 1, step)) + ([W - tile] if W > tile and (W - tile) % step else []):
                box = (x, y, min(x + tile, W), min(y + tile, H))
                run(bgr[box[1]:box[3], box[0]:box[2]], x, y, 1.0, 1.0, True, box)
    return out
