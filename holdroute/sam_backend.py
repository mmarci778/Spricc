"""Stage 2a - Segment Anything as a class-agnostic *mask proposal* engine.

Why SAM: it was trained on >1B masks and cuts objects out along real
boundaries regardless of colour, gloss or chalk. It does not know what a
climbing hold is, so it also returns wall, shadows and sub-parts. The shared
post-processing (postprocess.py) selects the right level of granularity.

Weights are NOT bundled. Download once:
  SAM 1   https://github.com/facebookresearch/segment-anything#model-checkpoints
          (vit_b ~375 MB, vit_l, vit_h ~2.5 GB)
  SAM 2.1 https://github.com/facebookresearch/sam2#download-checkpoints
          (sam2.1_hiera_large.pt etc.)
A CUDA GPU is strongly recommended (vit_h on CPU: minutes per image).

NOTE: this module was written against the public APIs of both packages but
could not be executed in the environment this project was authored in (no
network / no GPU). Run `python detect.py wall.jpg --backend sam ...` and
inspect the preview first.
"""
from __future__ import annotations

import cv2
import numpy as np

from .postprocess import Candidate


def _rle_to_candidate(rle, score: float) -> Candidate | None:
    from pycocotools import mask as mask_utils
    m = mask_utils.decode(rle).astype(bool)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return Candidate(int(x0), int(y0), m[y0:y1, x0:x1].copy(), score)


def propose_sam(
    bgr: np.ndarray,
    checkpoint: str,
    version: int = 1,
    model_type: str = "vit_h",          # SAM1 only: vit_b | vit_l | vit_h
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str | None = None,
    points_per_side: int = 48,          # dense wall => dense point grid (default 32 misses small holds)
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.92,
    crop_n_layers: int = 1,             # re-run on crops so small holds get more pixels
    min_mask_region_area: int = 150,
) -> list[Candidate]:
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    kw = dict(points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh,
              stability_score_thresh=stability_score_thresh, crop_n_layers=crop_n_layers,
              crop_n_points_downscale_factor=2, min_mask_region_area=min_mask_region_area,
              output_mode="coco_rle")   # RLE keeps ~1000 full-res masks out of RAM

    if version == 2:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        model = build_sam2(sam2_config, checkpoint, device=device, apply_postprocessing=False)
        gen = SAM2AutomaticMaskGenerator(model, **kw)
    else:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
        gen = SamAutomaticMaskGenerator(model, **kw)

    out = []
    for r in gen.generate(rgb):
        c = _rle_to_candidate(r["segmentation"], float(r["predicted_iou"]) * float(r["stability_score"]))
        if c is not None:
            out.append(c)
    return out
