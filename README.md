# Holdroute: climbing-hold detection and route builder

Detect every hold on a wall photo **once**, then build and edit any number of routes on top of it.

```
photo ──► detect.py ──► wall.holds.json ──► app/index.html (route builder)
           │                  ▲                    │
           │                  └── corrections ◄────┘   hide / merge / draw
           └─ backends: classical (no GPU) | sam | sam2 | yolo (fine-tuned)
```

## What has and has not been verified

| Part | Status |
|---|---|
| Classical backend, post-processing, JSON output | **Run** on the example wall: 361 holds in ~8 s on one CPU core |
| Route builder UI | **Run** in headless Chromium: 20 automated checks (`tests/ui_test.py`) |
| Hierarchy rules (chips vs chalk vs duplicates) | **Unit-tested** on synthetic shapes (`tests/test_postprocess.py`) |
| `evaluate.py`, `tools/*`, tiling for training | **Run** end to end on the example |
| SAM / SAM 2 / YOLO backends and `train_yolo.py` | **Written against the public APIs but never executed**: the authoring environment had no GPU, no network and no weights. Run them on your photo and look at the preview before trusting them |

No ground-truth labels existed for the example photo, so **no accuracy figure is claimed anywhere**. Producing one is what `evaluate.py` is for.

## Quick start

```bash
pip install -r requirements.txt
python3 serve.py                 # then open http://127.0.0.1:8000/  (loads the bundled example)
```

Detect your own wall (writes `wall.holds.json` and `wall.preview.jpg` with hold ids drawn on):

```bash
python detect.py photos/wall.jpg                                   # classical baseline, CPU only
python detect.py photos/wall.jpg --backend sam  --checkpoint sam_vit_h_4b8939.pth --model-type vit_h
python detect.py photos/wall.jpg --backend sam2 --checkpoint sam2.1_hiera_large.pt
python detect.py photos/wall.jpg --backend yolo --weights best.pt  # after fine-tuning, see below
```

`pip install -r requirements-ml.txt` first for the last three. Or run `python serve.py --backend sam --checkpoint ...`
and drop a raw photo into the app: it is detected on the server and cached by image hash, so the same photo is never analysed twice.
Without the server, open `app/index.html` directly and drop the photo and its `.holds.json` onto the page.

## Using the route builder

| Do this | To get this |
|---|---|
| Click a hold | add it to the active route (a route is created if none exists); holds are numbered in click order |
| Click a numbered hold again | remove it |
| **New route**, colour dots, name field | create a route, choose its tape colour, rename it |
| Click a route / keys `1`-`9` | switch routes; nothing is re-detected |
| Clear, Duplicate, Delete | edit routes; Undo/Redo = Ctrl/Cmd+Z, +Shift |
| View options | focus (dim everything but the route), outline all holds, dots for other routes, line between holds |
| Wheel / pinch, drag | zoom, pan (a drag never selects) |
| **Fix detections** → Hide, Merge, Draw | delete a false hold; merge two pieces of one hold (click the keeper, then the piece); trace a missing hold (click corners, Enter or double-click to close) |
| Export / Import | routes + corrections as `*.routes.json`. Routes are also auto-saved in the browser |

Any hold of any colour can be in any route, and in several routes at once. Routes store only hold ids; hold ids never change.

**Click accuracy.** A click selects the *smallest* hold polygon that contains it, so a chip bolted onto a volume, or a hold lying over its neighbour, is picked rather than the thing underneath it. A click in a gap snaps to the nearest outline within 6 screen pixels. Tested: clicking the anchor of all 361 detected holds returns exactly that hold.

## The example wall: what makes it hard

1. **Almost no wall is visible.** Holds touch and overlap; the only separation is often a thin contact shadow. Gap-based methods (thresholding, "find blobs on a plain background") fail by construction.
2. **Nested objects.** Small screw-on chips sit on large volumes. Both are holds, so the masks must nest, and a click must pick the chip.
3. **Two-tone holds.** The yellow/white target hold at top-left, the blue-and-white holds on the left. Colour changes *inside* an object, so colour segmentation splits it.
4. **Chalk and gloss.** Large white patches on the red bowl and the pink holds. They look exactly like a white hold, except that chalk has a soft edge and a bolted-on object has a crisp one.
5. **Colour overlap with the wall.** White, cream and yellow holds sit close to the plywood colour.
6. **Scale.** Hold area spans roughly two orders of magnitude (small foot chips to large volumes).
7. **A horizontal seam near y ≈ 875** where holds are cut off, which looks like a stitched photo. A seam creates a false boundary; shoot single, straight-on frames.

## Approaches considered

| Approach | Verdict |
|---|---|
| Colour thresholding / k-means | Rejected. Fails on points 1, 3, 4, 5 |
| Classical edges + watershed + region merging | Kept as the **no-GPU baseline**. Finds most outlines; over-splits chalky and multi-colour holds (see below) |
| **SAM / SAM 2 (zero-shot proposals)** + rule-based selection | **Recommended first step.** Cuts along real boundaries regardless of colour, no labels needed. It doesn't know what a "hold" is, so selection rules are needed |
| **Fine-tuned instance segmentation (YOLO-seg, Mask R-CNN, Mask2Former)** | **Recommended end state.** Learns your gym's holds, including two-tone, chalk and chips. Needs labels |
| Bounding-box detectors | Rejected: the brief needs masks, and boxes of dense holds overlap badly |

**Recommended path:** SAM for proposals → correct a handful of walls in the UI → use those corrections to fine-tune YOLO-seg → keep SAM as the labelling accelerator for each new wall.

## Pipeline

1. `preprocess.py`: resize; divide out wall-scale illumination and apply CLAHE to lightness only (chroma untouched); mean-shift smoothing flattens chalk speckle and texture while keeping edges; estimate the bare-wall colour.
2. Backend → masks: `classical.py` (gradient from chroma + down-weighted lightness + dark contact lines → marker watershed → merge regions with weak shared boundaries → remove wall-coloured regions → absorb enclosed chalk patches), `sam_backend.py`, or `yolo_backend.py` (tiled).
3. `postprocess.py`, shared by every backend: size and wall filters; duplicate removal; **parent/child resolution** (a contained mask is kept as a separate hold only if it is small, clearly different in colour from its surroundings **and** not soft-edged like chalk); smoothed polygons; a label anchor inside the *visible* part of each hold; deterministic ids in reading order.

`holds.json` (schema v1): `holds[] = {id, polygon, bbox, area, centroid, anchor, color}` plus `image` and `detector` blocks. Nested holds overlap; the UI resolves that with the smallest-wins rule.

## What the classical baseline gets wrong (observed on the example)

- Two-tone holds are often cut in two, and some chalk-heavy holds keep a stray fragment.
- Some outlines are ragged or wedge-shaped where colour is similar to a neighbour.
- It produced 361 regions where I count roughly 250-300 holds by eye, i.e. it over-splits. This is a visual estimate, not a measurement.

These are the cases the Fix tools exist for, and the cases a fine-tuned model should learn.

## Training, testing, improving

**Data.** 40-100 photos of your gym's walls are a realistic start (~10k hold instances), covering different walls, times of day, lamps, set/strip stages and cameras. Straight-on, ≥2000 px on the long side. Hold-out walls or sessions for validation, never random tiles.

**Annotate (semi-automatic).**
1. `python detect.py wall.jpg --backend sam ...`, open in the app, fix errors with Hide / Merge / Draw, Export.
2. `python tools/apply_overrides.py wall.holds.json wall.routes.json -o wall.gt.holds.json`
3. `python tools/to_coco.py *.gt.holds.json -o all.coco.json` (or import into CVAT for finer polygon edits).

Label rule to write down for annotators: *a hold is a separately bolted object; chalk, gloss and a second colour are not separate holds; a chip bolted onto a volume is.*

**Train.**
```bash
python train/coco_to_yolo_tiles.py --coco all.coco.json --images photos/ --out dataset --val-frac 0.2
python train/train_yolo.py --data dataset/holds.yaml
```
Aggressive hue augmentation is deliberate: colour is not what makes something a hold.

**Evaluate on whole photos** (not tiles):
```bash
python detect.py val/wall.jpg --backend yolo --weights runs/segment/train/weights/best.pt
python evaluate.py val/wall.holds.json val/wall.gt.holds.json
```
Reports precision/recall/F1 at IoU 0.5 and 0.75, mean IoU, and **splits / merges**: the errors that make a click select the wrong hold, which F1 can hide. Add the failures to the training set and repeat.

## Limitations and next steps

- Wall-colour estimation assumes bare plywood is visible somewhere; override with `--wall-ab a,b` for painted walls.
- Zero-shot parent/child rules can't tell a two-tone hold from a chip with certainty; fine-tuning fixes this.
- Steep or overhanging walls shot at an angle foreshorten holds; capture straight-on or rectify with a homography.
- Hold ids are stable within one `holds.json`, but a re-detection produces new ids. After the wall is reset, treat it as a new wall. To track holds across photos of the same wall, register the images (feature matching + homography) and match polygons by IoU.
- Not included: hold roles (start/finish/feet-only), grades, multi-user sync and a database. Routes are plain JSON, so these slot in cleanly.

## Layout

```
detect.py  serve.py  evaluate.py
holdroute/  preprocess.py  classical.py  sam_backend.py  yolo_backend.py  postprocess.py  pipeline.py
app/index.html          route builder (single file, no build step)
tools/                  apply_overrides.py  to_coco.py
train/                  coco_to_yolo_tiles.py  train_yolo.py
tests/                  test_postprocess.py  ui_test.py
examples/               wall.jpg  wall.holds.json  wall.preview.jpg
```
