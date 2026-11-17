"""Synthetic checks for the hierarchy rules. Run: python -m pytest tests -q   (or python tests/test_postprocess.py)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import cv2, numpy as np
from holdroute.postprocess import Candidate, candidate_from_full, resolve_hierarchy, build_holds

H, W = 300, 400

def disk(cx, cy, r):
    m = np.zeros((H, W), np.uint8); cv2.circle(m, (cx, cy), r, 1, -1); return m.astype(bool)

def scene():
    """red hold (big), blue chip on it, white chalk patch on it, plus a duplicate of the red hold."""
    bgr = np.full((H, W, 3), (140, 180, 215), np.uint8)             # plywood-ish
    big = disk(150, 150, 90)
    bgr[big] = (40, 40, 200)                                        # red
    chip = disk(120, 130, 14); bgr[chip] = (200, 60, 30)            # blue chip: strong contrast
    chalk = disk(190, 180, 25)                                      # feathered white chalk patch
    soft = cv2.GaussianBlur(chalk.astype(np.float32), (0, 0), 7)[..., None]
    bgr = (bgr * (1 - soft) + np.array([235, 235, 235]) * soft).astype(np.uint8)
    wchip = disk(210, 110, 12); bgr[wchip] = (240, 240, 240)        # crisp white chip (kept)
    return bgr, big, chip, chalk, wchip

def test_rules():
    bgr, big, chip, chalk, wchip = scene()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    cands = [candidate_from_full(big, 0.9), candidate_from_full(big, 0.8),      # duplicate
             candidate_from_full(chip, 0.8), candidate_from_full(chalk, 0.8),
             candidate_from_full(wchip, 0.8)]
    out = resolve_hierarchy(cands, lab)
    areas = sorted(c.area for c in out)
    assert len(out) == 3, f"expected parent + 2 chips, got {len(out)}: {areas}"
    holds = build_holds(out, bgr)
    assert len(holds) == 3 and sorted(h["id"] for h in holds) == [1, 2, 3]
    # nested hold must own its pixels: anchor of the parent is outside the chip
    parent = max(holds, key=lambda h: h["area"]); child = min(holds, key=lambda h: h["area"])
    assert np.hypot(parent["anchor"][0] - 120, parent["anchor"][1] - 130) > 14
    print("ok", [(h["id"], h["area"], h["anchor"]) for h in holds])

if __name__ == "__main__":
    test_rules()
