"""Orchestration: image in -> holds.json dict out."""
from __future__ import annotations

import time

import cv2
import numpy as np

from .classical import detect_classical
from .postprocess import (build_holds, filter_candidates, labels_to_candidates,
                          resolve_hierarchy)
from .preprocess import estimate_wall_ab

SCHEMA_VERSION = 1


def detect_holds(bgr: np.ndarray, backend: str = "classical", image_name: str = "wall.jpg",
                 wall_ab: tuple[int, int] | None = None, **opts) -> dict:
    """backend: classical | sam | sam2 | yolo.  opts are forwarded to the backend."""
    t0 = time.time()
    H, W = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    wall_ab = wall_ab or tuple(int(v) for v in estimate_wall_ab(bgr))

    if backend == "classical":
        labels = detect_classical(bgr, **opts)
        cands = labels_to_candidates(labels)
        cands = filter_candidates(cands, lab, wall_ab)
    elif backend in ("sam", "sam2"):
        from .sam_backend import propose_sam
        cands = propose_sam(bgr, version=2 if backend == "sam2" else 1, **opts)
        cands = filter_candidates(cands, lab, wall_ab)
        cands = resolve_hierarchy(cands, lab)
    elif backend == "yolo":
        from .yolo_backend import propose_yolo
        cands = propose_yolo(bgr, **opts)
        cands = filter_candidates(cands, lab, wall_ab)
        cands = resolve_hierarchy(cands, lab)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    holds = build_holds(cands, bgr)
    return {
        "version": SCHEMA_VERSION,
        "image": {"file": image_name, "width": W, "height": H},
        "detector": {"backend": backend, "wall_ab": list(wall_ab), "seconds": round(time.time() - t0, 1)},
        "holds": holds,
    }


def render_preview(bgr: np.ndarray, data: dict) -> np.ndarray:
    """Outline every hold (white keyline + black line) and print its id at the anchor."""
    out = bgr.copy()
    for h in data["holds"]:
        poly = np.array(h["polygon"], np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [poly], True, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.polylines(out, [poly], True, (0, 0, 0), 1, cv2.LINE_AA)
    scale = max(bgr.shape[:2]) / 2000
    for h in data["holds"]:
        x, y = h["anchor"]
        txt = str(h["id"])
        cv2.putText(out, txt, (x - 8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * max(scale, 0.8), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (x - 8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * max(scale, 0.8), (0, 0, 0), 1, cv2.LINE_AA)
    return out


def add_backend_args(ap):
    ap.add_argument("--backend", default="classical", choices=["classical", "sam", "sam2", "yolo"])
    ap.add_argument("--wall-ab", help="override wall colour as 'a,b' in OpenCV Lab (e.g. 133,148)")
    ap.add_argument("--checkpoint", help="SAM / SAM2 weights file")
    ap.add_argument("--model-type", default="vit_h", help="SAM 1 only: vit_b | vit_l | vit_h")
    ap.add_argument("--weights", help="YOLO-seg weights (best.pt)")
    ap.add_argument("--device", help="cuda | cpu (default: auto)")
    ap.add_argument("--points-per-side", type=int, default=48, help="SAM: prompt grid density")
    ap.add_argument("--merge-thresh", type=float, default=0.5, help="classical: higher merges more")


def backend_opts(a) -> tuple[dict, tuple | None]:
    """argparse namespace -> (kwargs for detect_holds, wall_ab)."""
    opts: dict = {}
    if a.backend == "classical":
        opts["merge_thresh"] = a.merge_thresh
    elif a.backend in ("sam", "sam2"):
        if not a.checkpoint:
            raise SystemExit("--checkpoint is required for the sam / sam2 backends")
        opts.update(checkpoint=a.checkpoint, device=a.device, points_per_side=a.points_per_side)
        if a.backend == "sam":
            opts["model_type"] = a.model_type
    else:
        if not a.weights:
            raise SystemExit("--weights is required for the yolo backend")
        opts.update(weights=a.weights, device=a.device)
    wall_ab = tuple(int(v) for v in a.wall_ab.split(",")) if a.wall_ab else None
    return opts, wall_ab
