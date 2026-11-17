#!/usr/bin/env python3
"""Detect climbing holds once; reuse the resulting holds.json for any number of routes.

  python3 detect.py wall.jpg                              # classical fallback, no GPU needed
  python3 detect.py wall.jpg --backend sam  --checkpoint sam_vit_h_4b8939.pth --model-type vit_h
  python3 detect.py wall.jpg --backend sam2 --checkpoint sam2.1_hiera_large.pt
  python3 detect.py wall.jpg --backend yolo --weights runs/segment/train/weights/best.pt
"""
import argparse, json, pathlib, sys

import cv2

from holdroute.pipeline import add_backend_args, backend_opts, detect_holds, render_preview


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", help="holds.json path (default: <image>.holds.json)")
    ap.add_argument("--preview", help="preview jpg path (default: <image>.preview.jpg)")
    add_backend_args(ap)
    a = ap.parse_args()

    bgr = cv2.imread(a.image)
    if bgr is None:
        sys.exit(f"cannot read {a.image}")
    opts, wall_ab = backend_opts(a)
    data = detect_holds(bgr, a.backend, image_name=pathlib.Path(a.image).name, wall_ab=wall_ab, **opts)

    stem = pathlib.Path(a.image).with_suffix("")
    out = a.out or f"{stem}.holds.json"
    json.dump(data, open(out, "w"), separators=(",", ":"))
    prev = a.preview or f"{stem}.preview.jpg"
    cv2.imwrite(prev, render_preview(bgr, data), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(data['holds'])} holds  ({data['detector']['seconds']} s, wall colour a,b = {data['detector']['wall_ab']})")
    print(f"  holds   -> {out}\n  preview -> {prev}")


if __name__ == "__main__":
    main()
