# Phase 12: Trained glyph segmentation detector

## Context

The current segmentation stage (`ClassicalSegmenter`, `src/hieroglyph/segmentation/classical.py`)
is classical OpenCV contour detection: binarize → morphological close → contour
bounding boxes. Per the README's known limitations, this works on clean,
evenly-lit photos but struggles on heavily weathered or cluttered wall
photos. The `Segmenter` abstract interface (`src/hieroglyph/segmentation/base.py`)
was deliberately built to allow a second, trained implementation to be
dropped in without touching anything downstream (reading order, classifier,
lookup, inference wiring).

This spec covers replacing/supplementing `ClassicalSegmenter` with a trained
detector, as the first of three planned enhancements (repo priority order
established in brainstorming: ML enhancements → website → mobile app, done
as separate efforts/repos).

### Dataset research findings

- `data/raw/` only contains single-glyph crops (Glyphnet / Pyramid-of-Unas
  dataset, 171 classes, ~4,031 images) — no real multi-glyph annotated
  photos exist in this repo.
- No bulk downloadable real-photo dataset with bounding-box/segmentation
  annotations was found. The largest candidate,
  [EngAdhamTamer/hieroglyph-detection](https://github.com/EngAdhamTamer/hieroglyph-detection)
  (235,436 hieroglyph instances from temple walls, obelisks, papyrus,
  stelae; single-class "hieroglyph" YOLO segmentation labels), does **not**
  ship its images/labels (confirmed via its `dataset/data.yaml` and
  `SETUP.md` — "actual dataset images are not included ... due to size
  constraints").
- That same repo **does** ship a pretrained checkpoint: `best.pt`
  (YOLOv8-segmentation, MIT licensed, 6.8MB, 70.9% mAP@0.5 on real
  archaeological photos), downloadable from its
  [v1.0.0 release](https://github.com/EngAdhamTamer/hieroglyph-detection/releases/download/v1.0.0/best.pt).
  Its single-class convention ("hieroglyph", no per-sign label) matches
  what we need exactly, since sign classification is already a separate
  downstream stage in our pipeline.
- One small real-photo dataset is directly downloadable:
  [Roboflow "egyptian-hieroglyphs"](https://universe.roboflow.com/custom-yolov8-ljpde/egyptian-hieroglyphs)
  — 40 images, 19 Gardiner-coded classes, CC BY 4.0. Too small to train on,
  usable as a real-photo qualitative eval set.

### Decision

Fine-tune the pretrained YOLOv8-seg checkpoint (`best.pt` above) on
synthesized composite images built from our own `data/raw/` crops, rather
than training a detector from scratch. Starting from weights already
exposed to real archaeological photos should generalize better than
training purely on synthetic composites, at the cost of adding
`ultralytics` as a new dependency and a heavier CPU inference footprint
than a lighter option (e.g. torchvision SSDlite) would have had.

## Architecture

New `YoloSegmenter` class in `src/hieroglyph/segmentation/yolo.py`,
implementing the existing `Segmenter` ABC (`detect(image: np.ndarray) ->
list[BoundingBox]`) — the same interface `ClassicalSegmenter` implements.
Nothing in `pipeline/inference.py` needs to change structurally to support
it; `run_inference`'s `segmenter` parameter already accepts any `Segmenter`.

`YoloSegmenter` wraps `ultralytics.YOLO`: loads a checkpoint path at
construction, and in `detect()` runs the model on the input image and
converts `Results.boxes.xyxy` into our `BoundingBox` dataclass (x, y,
width, height). YOLOv8-seg also produces polygon masks; we ignore them —
our `Segmenter` interface only needs axis-aligned boxes.

## Data pipeline: composite synthesis

A new module/script (`src/hieroglyph/segmentation/synthesize.py`, driven
from a notebook `notebooks/04_train_segmenter.ipynb`) builds a labeled
training set from `data/raw/<gardiner_code>/*.png` crops:

- For each synthetic image: pick a random background, paste a random
  number of randomly-chosen crops onto it at random positions/scales/
  rotations, rejecting placements with heavy overlap (extending the same
  spirit as `test_segmentation.py`'s synthetic-shapes tests, but at
  training scale — thousands of composites, not 3).
- Write YOLO-format label files: one class (`0: hieroglyph`), one line per
  pasted crop with its resulting bounding box in the composite.
- Produce train/valid/test splits, holding out a slice of *unseen glyph
  crops* (not just unseen composite arrangements) for the test split, so
  test-set performance isn't inflated by the model having seen the same
  crop pixels during training.

Background source: plain/textured synthetic backgrounds (solid colors,
simple noise/gradient textures) generated in code — no external background
image dataset needed for a first version.

## Training

Fine-tune `best.pt` on the synthesized composites, on Colab (GPU) — same
convention as `notebooks/02_train_classifier.ipynb` (the setup doc already
notes Colab has its own preinstalled torch/torchvision and training
happens there, not locally/CPU-only). Save the fine-tuned weights to
`models/yolo_seg.pt`, alongside the existing `models/best_model.pt`
(classifier).

## Evaluation

`notebooks/05_evaluate_segmenter.ipynb`, mirroring
`03_evaluate_classifier.ipynb`'s style:

- Quantitative: precision/recall/mAP on the held-out synthetic composite
  test split (unseen crops).
- Qualitative: run both `ClassicalSegmenter` and `YoloSegmenter` against
  the 40-image real Roboflow set (CC BY 4.0) as a real-photo domain-gap
  sanity check — no formal metric required here since that set's
  annotations use a different class convention (19 Gardiner classes vs.
  our single-class detector), just visual/box-count comparison.

Resulting numbers get added to the README next to the existing classifier
accuracy line, following the same "known limitations" honesty the README
already models.

## Integration

- `run_inference`'s `segmenter` parameter keeps defaulting to
  `ClassicalSegmenter()` — no forced dependency on a multi-MB model file
  for existing tests/callers that don't pass one explicitly.
- `app/streamlit_app.py` is updated to explicitly construct
  `YoloSegmenter(YOLO_CHECKPOINT_PATH)` and pass it to `run_inference`,
  since the demo app is the surface that should show off the better
  segmenter.

## Dependencies

Add `ultralytics` to `requirements.txt`, following the existing comment
style that distinguishes local CPU installs from Colab's preinstalled
environment.

## Testing

- `tests/test_segmentation.py` gains a `YoloSegmenter` test mirroring the
  existing `ClassicalSegmenter` tests (deterministic synthetic image,
  assert detected boxes land near expected shape centers).
- Open question, to resolve during implementation: whether this test runs
  against the real fine-tuned checkpoint (requires the model file to be
  present/downloaded) or is skipped/marked when the checkpoint isn't
  available, so `pytest` doesn't hard-fail in environments without the
  model file.
- New unit tests for the composite synthesizer: given a fixed random seed
  and a small set of fixture crops, assert the right number of composites
  are generated and that written YOLO label files are internally
  consistent with the pasted crop positions.

## Out of scope (YAGNI)

- Per-sign-class detection (the detector stays single-class; classification
  remains the existing ResNet18 classifier's job).
- Sourcing/annotating a large real-photo dataset ourselves — not pursued
  given the effort/benefit tradeoff versus fine-tuning the pretrained
  checkpoint.
- Reading-order and classifier-coverage enhancements — separate,
  independently-scoped follow-up efforts (not part of this spec).
