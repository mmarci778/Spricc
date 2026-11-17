#!/usr/bin/env python3
"""COCO (one class: hold) -> Ultralytics YOLO-seg dataset made of overlapping tiles.

  python train/coco_to_yolo_tiles.py --coco all.coco.json --images photos/ --out dataset --val-frac 0.2

* Walls are huge and dense (300+ holds). Tiles of 1024 px keep small holds large enough to learn.
* The train/val split is made by IMAGE (ideally by wall + session), never by tile: neighbouring
  tiles of one photo share holds, and a random tile split would leak and inflate the score.
* A hold is kept in a tile only if >= 60 % of it lies inside; otherwise it is left unlabelled in
  that tile (the neighbouring tile, which overlaps, sees it whole).
"""
import argparse, json, pathlib, random

import cv2
import numpy as np


def tiles(w, h, tile, overlap):
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(w - tile, 0) + 1, step)); ys = list(range(0, max(h - tile, 0) + 1, step))
    if w > tile and xs[-1] != w - tile: xs.append(w - tile)
    if h > tile and ys[-1] != h - tile: ys.append(h - tile)
    return [(x, y, min(x + tile, w), min(y + tile, h)) for y in ys for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True); ap.add_argument("--images", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=1024); ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.2); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    coco = json.load(open(a.coco)); out = pathlib.Path(a.out)
    by_img = {}
    for an in coco["annotations"]:
        by_img.setdefault(an["image_id"], []).append(an)
    imgs = sorted(coco["images"], key=lambda i: i["file_name"]); random.Random(a.seed).shuffle(imgs)
    n_val = 0 if len(imgs) < 2 else max(1, round(a.val_frac * len(imgs)))
    split = {i["id"]: ("val" if k < n_val else "train") for k, i in enumerate(imgs)}
    if n_val == 0: print("only one image: it is used for train AND val - fine for a smoke test, meaningless for evaluation")

    counts = {"train": [0, 0], "val": [0, 0]}
    for im in imgs:
        bgr = cv2.imread(str(pathlib.Path(a.images) / im["file_name"]))
        if bgr is None: print("skip (unreadable):", im["file_name"]); continue
        H, W = bgr.shape[:2]
        polys = {an["id"]: [np.array(s, np.float32).reshape(-1, 2).astype(np.int32) for s in an["segmentation"]] for an in by_img.get(im["id"], [])}
        total = {k: sum(cv2.contourArea(p) for p in v) for k, v in polys.items()}
        for sp in ([split[im["id"]]] if n_val else ["train", "val"]):
            for (x0, y0, x1, y1) in tiles(W, H, a.tile, a.overlap):
                tw, th = x1 - x0, y1 - y0; lines = []
                for k, ps in polys.items():
                    m = np.zeros((th, tw), np.uint8)
                    for p in ps: cv2.fillPoly(m, [(p - [x0, y0]).reshape(-1, 1, 2)], 1)
                    vis = int(m.sum())
                    if vis < 40 or vis < 0.6 * max(total[k], 1): continue
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    c = max(cnts, key=cv2.contourArea)
                    if len(c) < 3: continue
                    pts = c[:, 0, :] / np.array([tw, th])
                    lines.append("0 " + " ".join(f"{v:.5f}" for v in pts.ravel()))
                stem = f"{pathlib.Path(im['file_name']).stem}_{x0}_{y0}"
                (out / "images" / sp).mkdir(parents=True, exist_ok=True); (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / "images" / sp / f"{stem}.jpg"), bgr[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines))
                counts[sp][0] += 1; counts[sp][1] += len(lines)
    (out / "holds.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: hold\n")
    print(f"train: {counts['train'][0]} tiles / {counts['train'][1]} instances   val: {counts['val'][0]} tiles / {counts['val'][1]} instances")
    print(f"dataset config: {out / 'holds.yaml'}")


if __name__ == "__main__":
    main()
