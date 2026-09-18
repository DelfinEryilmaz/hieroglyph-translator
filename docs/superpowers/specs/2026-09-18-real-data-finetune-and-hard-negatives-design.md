# Real-data fine-tuning + hard-negative composites: design

## Context

The dense-text-detection branch (merged, see
`docs/superpowers/specs/2026-09-17-dense-text-detection-design.md`) fixed
two problems: destructive whole-image downscale (via `TiledYoloSegmenter`)
and sparse-only training data (via `generate_mixed_dataset`'s dense/column
composites). After retraining on the mixed dataset and re-testing against
the real Papyrus of Ani photo, detection on dense text columns improved
only modestly (318->341 boxes with tiling; the big jump, 14->318, already
came from tiling alone, before any retraining). Visually, the retrained
checkpoint still places boxes on the papyrus's human/god figure
illustrations (clothing folds, limbs) — false positives the model was
never taught to reject, since every synthetic composite so far (scatter,
dense, or mixed) contains *only* glyph crops on a noisy background, never
"here's some other kind of ink marking that isn't a sign."

A quick diagnostic (desaturating the real photo to match training's exact
grayscale statistics before inference) ruled out color-channel mismatch as
the cause: false-positive count barely changed (341->308) and the same
figure regions were still flagged. The gap is structural: the model has
never seen a negative example.

Separately, this repo already has 90 real, human-annotated hieroglyph
photos sitting unused: `data/real_eval_photos/train/` (83 image+label
pairs) and `data/real_eval_photos/valid/` (7 pairs), from the Roboflow
"egyptian-hieroglyphs" dataset (CC BY 4.0, downloaded per
`notebooks/05_evaluate_segmenter.ipynb` section 3's manual step). Phase
12's original spec
(`docs/superpowers/specs/2026-09-15-glyph-segmentation-detector-design.md`)
dismissed this set as "too small to train on," and it has only ever been
used for a qualitative box-count eyeball check. That conclusion was right
for training *from scratch* but wrong for a **fine-tuning** stage: a large
synthetic pretrain followed by a short fine-tune on a small amount of real
data is the standard way to close a sim2real gap that synthetic data alone
can't close (real ink texture, real lighting, real papyrus/stone surface,
real JPEG/scan artifacts) — none of which any synthesizer can fully fake.

This spec adds both, as two independent, additive levers:

1. **Hard-negative composites** — teach the detector what *isn't* a sign,
   attacking the false-positive-on-illustrations problem directly.
2. **Real-data fine-tuning stage** — teach the detector what a *real*
   photographed sign looks like, attacking the general sim2real texture
   gap.

Neither requires touching `Segmenter`, `BoundingBox`, or any downstream
pipeline code — both are training-data/training-recipe changes only,
matching this repo's established pattern of keeping the segmentation
interface stable while swapping what feeds it.

## 1. Hard-negative composites

### Problem
`generate_mixed_dataset` (`src/hieroglyph/segmentation/synthesize.py`,
already committed) mixes `synthesize_composite` (sparse scatter) and
`synthesize_column_composite` (dense column) by `dense_fraction` — but
*every* composite it ever produces has glyph crops pasted onto it. The
model has literally never been shown "here is dark, structured ink-like
content that is not a hieroglyph" during training.

### Design
New function `synthesize_negative_composite()` in `synthesize.py`: builds
a canvas with the same noisy background as the other synthesizers, then
draws a random number of procedural "distractor" shapes — thick curved
polylines and filled/outlined ellipses at varied position, size, and
stroke width, using the same dark ink tone range as real glyph strokes
(`cv2.polylines`/`cv2.ellipse`) — and returns a `SynthesizedComposite` with
an **empty `boxes` list**. These shapes approximate the general character
of figure/illustration line-art (continuous curved strokes, larger
connected shapes) as opposed to a real glyph's small, compact, mostly
self-contained ink blob — without needing to source or license a real
illustration image dataset.

`generate_mixed_dataset` gains a `negative_fraction: float = 0.15`
parameter: per composite index, `rng.random()` first checks against
`negative_fraction` (producing a negative composite if it falls below),
then against `negative_fraction + dense_fraction` (dense), else scatter —
purely additive on top of the existing two-way split, default parameter
values keep existing callers' behavior close to unchanged in proportion.

This is a training-data-only change — it does not claim to reproduce real
illustration content exactly, only to give the model *some* negative
signal it currently has none of. If it doesn't fully close the gap, the
real-data fine-tune (below) is the second, independent lever.

## 2. Real-data fine-tuning stage

### Data: commit the already-downloaded real train/valid split
`data/real_eval_photos/train/` (83 pairs, 2.1MB) and
`data/real_eval_photos/valid/` (7 pairs, 163KB) — currently covered by the
blanket-ignored `data/real_eval_photos/` `.gitignore` entry — get carved
out and committed to the repo (small enough: ~2.3MB total). This makes
them automatically available on Colab via the training notebook's existing
`git clone` step, with no new manual upload cell needed. `data/real_eval_photos/test/`
(inconsistent/incomplete leftover — loose label files at the top level
plus a near-empty `images/`/`labels/` pair, likely a partial re-export) and
`data/real_eval_photos/papyrus/` (re-downloadable via
`scripts/download_papyrus_eval_photo.py`, no need to commit) stay ignored.
License: CC BY 4.0 — provenance recorded in a new sourcing report matching
this repo's convention, alongside `reports/2026-09-15-segmentation-detector-sourcing.md`.

### Label remapping
The Roboflow export's labels are standard YOLO-*detection* format (`class
cx cy w h`, one line per box, 19 Gardiner classes by their own indexing) —
our detector is YOLOv8-*segmentation*, single-class, using a 4-corner
polygon-per-box label convention (`box_to_yolo_line`,
`src/hieroglyph/segmentation/synthesize.py`). A new module
`src/hieroglyph/segmentation/real_data.py` provides
`prepare_real_finetune_split(images_dir: Path, labels_dir: Path, output_dir: Path) -> int`:
for each image with a matching label file (by filename stem — raises
`ValueError` on a mismatch, matching this repo's fail-loud convention),
copies the image unchanged to `output_dir/images/` and rewrites every line
of its label file as a single-class (`0`) polygon line to
`output_dir/labels/`, reusing the same corner-ordering convention
`box_to_yolo_line` already uses (extracted into a small shared helper in
`synthesize.py` so both call sites can't drift apart on the exact corner
order ultralytics' segment-task loader expects). Returns the number of
image/label pairs processed.

### Training: a second stage, not a bigger first stage
`notebooks/04_train_segmenter.ipynb` gains a new section after the
existing synthetic-composite `model.train(...)` call: remap
`data/real_eval_photos/{train,valid}` via `prepare_real_finetune_split`
into a `data/real_finetune/{train,valid}` layout, build a second
`data.yaml`, then call `model.train(...)` **again on the same in-memory
`model` object** (ultralytics continues from the model's current weights
when `.train()` is called a second time on one `YOLO` instance, rather
than reinitializing) with a small epoch count and the real-only data — a
short, low-volume fine-tune on top of the synthetic pretrain, not a
from-scratch run on 90 images. The final checkpoint-download cell is
updated to pull this second run's weights (`yolo_seg_finetune_real/weights/best.pt`)
instead of the first stage's, since that's the model callers should
actually use.

## Out of scope (YAGNI)

- Tuning `negative_fraction`, distractor-shape parameters, or real-fine-tune
  epoch count/learning rate empirically — ship sensible defaults, let the
  qualitative papyrus check (already in `notebooks/05_evaluate_segmenter.ipynb`
  Section 5) be the judge; a numeric sweep is a separate future effort if
  this round doesn't move the needle enough.
- Fixing/using `data/real_eval_photos/test/`'s inconsistent leftover
  structure — noted, not addressed here.
- Sourcing additional real illustration/figure photos as literal hard
  negatives (vs. the procedural approximation above) — a stronger version
  of the same idea, left for later if procedural negatives prove
  insufficient.
- Any change to `Segmenter`, `BoundingBox`, `TiledYoloSegmenter`, or
  `YoloSegmenter` — this is purely a training-data/training-recipe change.
- Actually running the retrain on Colab — same convention as every prior
  phase, a manual human step after this plan's tasks land.
