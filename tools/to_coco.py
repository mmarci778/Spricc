#!/usr/bin/env python3
"""holds.json (+ image) -> COCO instance-segmentation JSON, for CVAT / Label Studio / training.

  python tools/to_coco.py wall.holds.json -o wall.coco.json
Import the result in CVAT (Upload annotations -> COCO 1.0) to correct it with
polygon tools, then export COCO again as ground truth. Pass several holds files
to build one dataset.
"""
import argparse, json, pathlib


def polys(h):
    return [h["polygon"]] + h.get("parts", [])


def to_coco(files):
    images, anns = [], []
    for img_id, f in enumerate(files, start=1):
        d = json.load(open(f))
        images.append({"id": img_id, "file_name": d["image"]["file"], "width": d["image"]["width"], "height": d["image"]["height"]})
        for h in d["holds"]:
            seg = [[c for pt in p for c in pt] for p in polys(h)]
            xs = [pt[0] for p in polys(h) for pt in p]; ys = [pt[1] for p in polys(h) for pt in p]
            area = h.get("area") or 0
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": 1, "iscrowd": 0, "segmentation": seg,
                         "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], "area": area})
    return {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "hold"}]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("holds", nargs="+"); ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    c = to_coco(a.holds); json.dump(c, open(a.out, "w"))
    print(f"{len(c['images'])} image(s), {len(c['annotations'])} instances -> {a.out}")
