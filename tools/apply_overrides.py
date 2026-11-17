#!/usr/bin/env python3
"""Bake the corrections you made in the UI (hide / merge / draw) into a corrected holds.json.

  python tools/apply_overrides.py wall.holds.json wall.routes.json -o wall.corrected.holds.json

The corrected file is your ground truth: feed it to evaluate.py and tools/to_coco.py.
A merged hold keeps its largest polygon in "polygon" and the rest in "parts".
"""
import argparse, json


def apply(holds: dict, routes: dict) -> dict:
    ov = routes.get("overrides", {"hidden": [], "merged": {}, "added": []})
    hidden = set(ov.get("hidden", []))
    merged = {int(k): int(v) for k, v in ov.get("merged", {}).items()}
    def root(i):
        while i in merged: i = merged[i]
        return i
    base = {h["id"]: dict(h) for h in holds["holds"]}
    for a in ov.get("added", []):
        base[a["id"]] = {"id": a["id"], "polygon": a["polygon"], "anchor": a["anchor"], "color": a.get("color", "#888888")}
    out = {i: h for i, h in base.items() if i not in hidden and i not in merged}
    for src, dst in merged.items():
        t, s = out.get(root(dst)), base.get(src)
        if t is not None and s is not None:
            t.setdefault("parts", []).append(s["polygon"])
    holds = dict(holds)
    holds["holds"] = sorted(out.values(), key=lambda h: h["id"])
    holds["detector"] = {**holds.get("detector", {}), "corrected": True}
    return holds


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("holds"); ap.add_argument("routes"); ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    res = apply(json.load(open(a.holds)), json.load(open(a.routes)))
    json.dump(res, open(a.out, "w"), separators=(",", ":"))
    print(f"{len(res['holds'])} holds written to {a.out}")
