#!/usr/bin/env python3
"""Fine-tune an instance-segmentation model on your gym's holds.

  pip install ultralytics
  python train/train_yolo.py --data dataset/holds.yaml

Starting point, not a recipe: look at the validation predictions after every run
(runs/segment/*/val_batch*_pred.jpg) before changing hyper-parameters.
"""
import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="yolo11m-seg.pt", help="start from COCO-pretrained weights; try -s for speed, -l if you have >1k labelled walls")
ap.add_argument("--epochs", type=int, default=150); ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--device", default=None)
a = ap.parse_args()

YOLO(a.model).train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
    max_det=1000,                    # dense walls: the default 300 instances is too low
    hsv_h=0.5, hsv_s=0.6, hsv_v=0.4, # colour is NOT a hold cue: aggressive hue jitter stops the model memorising colours
    degrees=8, scale=0.4, fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=15,
    patience=30,
)
# evaluate on the held-out walls with:  YOLO('runs/segment/train/weights/best.pt').val(data=a.data)
# then run the real test, instance quality on whole photos:  python evaluate.py pred.holds.json gt.holds.json
