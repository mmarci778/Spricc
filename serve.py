#!/usr/bin/env python3
"""Local server for the route builder. Optionally runs detection when you drop in a raw photo.

  python serve.py                                   # classical backend
  python serve.py --backend sam --checkpoint sam_vit_h_4b8939.pth
then open http://127.0.0.1:8000/

Detection results are cached by image hash under data/<hash>/, so the same photo
is never analysed twice.
"""
import argparse, hashlib, json, pathlib

import cv2
import numpy as np
from flask import Flask, Response, abort, redirect, request, send_from_directory

from holdroute.pipeline import add_backend_args, backend_opts, detect_holds, render_preview

ROOT = pathlib.Path(__file__).resolve().parent
PUBLIC = {"app", "examples", "data"}          # only these folders are served

ap = argparse.ArgumentParser()
add_backend_args(ap)
ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
ARGS = ap.parse_args() if __name__ == "__main__" else ap.parse_args([])

app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    demo = ROOT / "examples" / "wall.holds.json"
    q = "?image=/examples/wall.jpg&holds=/examples/wall.holds.json" if demo.exists() else ""
    return redirect("/app/index.html" + q)


@app.get("/api/ping")
def ping():
    return {"ok": True, "backend": ARGS.backend}


@app.post("/api/detect")
def detect():
    f = request.files.get("image")
    if not f:
        abort(400, "no image uploaded")
    raw = f.read()
    folder = ROOT / "data" / hashlib.sha1(raw).hexdigest()[:12]
    cached = folder / "holds.json"
    if cached.exists():
        return Response(cached.read_text(), mimetype="application/json")
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        abort(400, "could not decode image")
    opts, wall_ab = backend_opts(ARGS)
    try:
        data = detect_holds(bgr, ARGS.backend, image_name=f.filename or "wall.jpg", wall_ab=wall_ab, **opts)
    except Exception as e:                                   # e.g. missing weights / out of memory
        abort(500, f"{type(e).__name__}: {e}")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "wall.jpg").write_bytes(raw)
    cv2.imwrite(str(folder / "preview.jpg"), render_preview(bgr, data))
    text = json.dumps(data, separators=(",", ":"))
    cached.write_text(text)
    return Response(text, mimetype="application/json")


@app.get("/<path:p>")
def static_files(p):
    if p.split("/")[0] not in PUBLIC:
        abort(404)
    return send_from_directory(ROOT, p)


if __name__ == "__main__":
    print(f"Route builder: http://{ARGS.host}:{ARGS.port}/   (detection backend: {ARGS.backend})")
    app.run(host=ARGS.host, port=ARGS.port, threaded=True)
