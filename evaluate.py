#!/usr/bin/env python3
"""Score a detector against hand-corrected ground truth.

  python evaluate.py pred.holds.json gt.holds.json            # gt from the UI via apply_overrides.py
  python evaluate.py pred.holds.json gt.coco.json             # or a COCO file with ONE image

Instance-level metrics (every hold is one object; a box around a cluster counts for nothing):
  precision / recall / F1 at IoU 0.5 and 0.75   (one-to-one Hungarian matching)
  mean IoU of matched holds
  splits  : ground-truth holds that only exist as 2+ unmatched predictions   (one hold shows up as several)
  merges  : unmatched predictions that swallow 2+ ground-truth holds        (clicking selects the wrong hold)
Splits and merges matter most for route setting; F1 can hide them.
"""
import json, sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from holdroute.postprocess import Candidate


def load_instances(path):
    d = json.load(open(path))
    if "holds" in d:
        return [[np.array(p, np.int32) for p in [h["polygon"]] + h.get("parts", [])] for h in d["holds"]], d["image"]
    img = d["images"][0]                                   # COCO: first image only
    out = []
    for a in d["annotations"]:
        if a["image_id"] == img["id"]:
            out.append([np.array(s, np.float32).reshape(-1, 2).astype(np.int32) for s in a["segmentation"]])
    return out, img


def to_cand(polys):
    pts = np.concatenate(polys)
    x0, y0 = pts.min(0); x1, y1 = pts.max(0) + 1
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [(p - [x0, y0]).reshape(-1, 1, 2)], 1)
    return Candidate(int(x0), int(y0), m.astype(bool))


def inter_matrix(P, G):
    I = np.zeros((len(P), len(G)), np.int64)
    for i, p in enumerate(P):
        for j, g in enumerate(G):
            x0, y0, x1, y1 = max(p.x0, g.x0), max(p.y0, g.y0), min(p.x1, g.x1), min(p.y1, g.y1)
            if x1 > x0 and y1 > y0:
                I[i, j] = (p.window(x0, y0, x1, y1) & g.window(x0, y0, x1, y1)).sum()
    return I


def main(pred_path, gt_path):
    pp, _ = load_instances(pred_path); gp, _ = load_instances(gt_path)
    P, G = [to_cand(x) for x in pp], [to_cand(x) for x in gp]
    pa, ga = np.array([c.area for c in P]), np.array([c.area for c in G])
    I = inter_matrix(P, G); U = pa[:, None] + ga[None, :] - I; IoU = I / np.maximum(U, 1)

    print(f"predicted {len(P)}   ground truth {len(G)}")
    r, c = linear_sum_assignment(-IoU)
    for thr in (0.5, 0.75):
        tp = int(((IoU[r, c]) >= thr).sum()); fp, fn = len(P) - tp, len(G) - tp
        pr, rc = tp / max(len(P), 1), tp / max(len(G), 1); f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        print(f"IoU>={thr:<4}  TP {tp:4d}  FP {fp:4d}  FN {fn:4d}   precision {pr:.3f}  recall {rc:.3f}  F1 {f1:.3f}")
    ok = IoU[r, c] >= 0.5
    print(f"mean IoU of matched (>=0.5): {IoU[r, c][ok].mean() if ok.any() else 0:.3f}")

    # splits / merges are only counted among holds that did NOT already match one-to-one,
    # otherwise a chip sitting on a volume (legitimately nested) would look like a split/merge.
    mp = {int(i) for i, j in zip(r, c) if IoU[i, j] >= 0.5}; mg = {int(j) for i, j in zip(r, c) if IoU[i, j] >= 0.5}
    free_p = [i for i in range(len(P)) if i not in mp]; free_g = [j for j in range(len(G)) if j not in mg]
    splits = merges = 0
    for j in free_g:      # a GT hold covered by 2+ leftover predictions
        parts = [i for i in free_p if I[i, j] / max(pa[i], 1) >= 0.5]
        if len(parts) >= 2 and sum(I[i, j] for i in parts) >= 0.6 * ga[j]:
            splits += 1
    for i in free_p:      # a leftover prediction covering 2+ leftover GT holds
        parts = [j for j in free_g if I[i, j] / max(ga[j], 1) >= 0.5]
        if len(parts) >= 2 and sum(I[i, j] for j in parts) >= 0.6 * pa[i]:
            merges += 1
    print(f"splits (GT hold broken into 2+ pieces): {splits}\nmerges (prediction covers 2+ GT holds): {merges}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
