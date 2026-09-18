# Dense/papyrus text detection: design

## Context

Phase 12 (`docs/superpowers/specs/2026-09-15-glyph-segmentation-detector-design.md`)
shipped `YoloSegmenter`, fine-tuned on synthetic composites built by
`synthesize_composite()`: 3-10 crops scattered sparsely across a 640x640
canvas via reject-sampling, each crop close to its native scale. The
qualitative check in `notebooks/05_evaluate_segmenter.ipynb` against 10 real
Roboflow photos already showed the trained detector finding noticeably
*fewer* boxes than `ClassicalSegmenter` per photo (README, "Model
baseline").

Testing against the Papyrus of Ani (a real papyrus scroll, cursive
hieroglyphic text in tight vertical columns) made the cause concrete: real
papyrus text looks nothing like the training distribution. Two distinct
problems compound:

1. **Training-data shape mismatch.** Real papyrus columns pack signs
   edge-to-edge, often overlapping, at a much smaller scale relative to the
   frame than `synthesize_composite`'s sparse scatter ever produces. A
   detector never shown that layout has no reason to recognize it.
2. **Inference-time downscale destroys small signs.** `YoloSegmenter.detect`
   (`src/hieroglyph/segmentation/yolo.py`) hands the *whole* image to
   `model.predict`, which internally resizes to `imgsz` (640). A
   high-resolution photo of a densely-packed column shrinks small
   individual signs to a handful of pixels before the model ever sees them,
   regardless of how well-trained the model is.

This spec covers both fixes. It's scoped narrower than Phase 12: no new
`Segmenter` ABC changes, no new external dependencies (`ultralytics` is
already a dependency), pure additions alongside the existing
`synthesize_composite` / `YoloSegmenter` code paths.

### Prior work already committed on this branch (`worktree-dense-text-detection`)

- `synthesize_column_composite()` (`src/hieroglyph/segmentation/synthesize.py`):
  deterministic cumulative stacking of crops into `n_columns` tight
  top-to-bottom stacks, small `scale_range`, optional negative
  `glyph_gap_range` for deliberate overlap — models a dense papyrus column
  instead of `synthesize_composite`'s sparse scatter.
- `generate_mixed_dataset()`: per-composite-index mix of
  `synthesize_column_composite` (dense) and `synthesize_composite` (sparse)
  by `dense_fraction`, additive alongside the existing `generate_dataset`
  (which stays available, unchanged, for callers that want scatter-only).

Both are covered by passing tests (`tests/test_synthesize.py`) but **not
yet used by anything** — `notebooks/04_train_segmenter.ipynb` still calls
`generate_dataset` (scatter-only), so the actual `models/yolo_seg.pt`
checkpoint has never been trained on dense/column data.

### Also already implemented, not yet committed

- `src/hieroglyph/segmentation/tiling.py` + `tests/test_tiling.py`
  (untracked): `TiledYoloSegmenter`, a `Segmenter` that splits a large image
  into overlapping tiles, runs YOLO on each tile at native resolution (no
  whole-image downscale), offsets detections back to global coordinates,
  and merges duplicates from signs straddling a tile boundary via greedy
  NMS (`merge_tiled_detections`). The "smaller than a tile" case is decided
  **per axis**: an axis no longer than its tile dimension gets a single
  origin `0` on that axis, while a longer axis is still tiled normally. So
  an image smaller than a tile in *both* dimensions degenerates to one
  `(0, 0)` origin and behaves exactly like plain `YoloSegmenter`, while a
  tall narrow column crop (small in one dimension only) is still tiled
  along its long axis rather than being silently truncated to the first
  tile. `TiledYoloSegmenter` is a drop-in replacement with no downside for
  small photos in either case.
  35/35 tests pass (`tests/test_tiling.py`, `tests/test_synthesize.py`).

## What this spec adds

### 1. Wire `TiledYoloSegmenter` into the app

`src/hieroglyph/segmentation/yolo.py`'s existing `load_or_fallback` stays
unchanged (still returns plain `YoloSegmenter`, used directly by
`notebooks/05_evaluate_segmenter.ipynb`'s side-by-side comparison cell).
A new `load_tiled_or_fallback(checkpoint_path) -> Segmenter` is added to
`tiling.py` itself (not `yolo.py`, to avoid a circular import —
`tiling.py` already imports `_xyxy_to_boxes` from `yolo.py`): returns
`TiledYoloSegmenter(checkpoint_path)` when the checkpoint exists, else
`ClassicalSegmenter()`, mirroring `load_or_fallback`'s existing contract
exactly. `app/streamlit_app.py` switches its import from
`hieroglyph.segmentation.yolo.load_or_fallback` to
`hieroglyph.segmentation.tiling.load_tiled_or_fallback` — the demo app is
the surface that should show off tiled inference on real (possibly large)
uploaded photos.

### 2. Retrain on the mixed dataset

`notebooks/04_train_segmenter.ipynb`'s composite-generation cell switches
from `generate_dataset` to `generate_mixed_dataset` for all three splits
(train/valid/test), keeping the same seeds (0/1/2) so the split-generation
convention documented in the notebook's own markdown stays true.
`notebooks/05_evaluate_segmenter.ipynb`'s held-out-test-split regeneration
cell makes the matching change (same `dense_fraction` default, same
`seed=2`) so it reproduces the *same* test composites the retrained model
was evaluated against, per the notebook's existing "same seeds" contract.

Actually re-running training on Colab (GPU) and downloading the new
`models/yolo_seg.pt` is a manual human step, same convention as Phase 12 —
not something this plan's tasks execute.

### 3. Real-papyrus qualitative eval

Add a `scripts/download_papyrus_eval_photo.py` (mirrors
`scripts/download_pretrained_yolo.py`'s pattern: thin `urllib` wrapper, no
API token) that downloads a public-domain, real photo of a dense
hieroglyphic-text papyrus column: "Papyrus of Ani BM Sheet 12" (British
Museum photographic reproduction of a papyrus dated ~1250 BCE — public
domain, copyright term expired), hosted on Wikimedia Commons at
`https://upload.wikimedia.org/wikipedia/commons/c/c9/Papyrus_of_Ani_BM_Sheet_12.jpg`
(2,414x1,498px, confirmed via Commons file-info license tag: "public domain
in its country of origin and other countries and areas where the copyright
term is the author's life plus 100 years or fewer"). Saved to
`data/real_eval_photos/papyrus/` (already covered by the existing
`data/real_eval_photos/` `.gitignore` entry). Provenance recorded in
`reports/2026-09-17-papyrus-eval-photo-sourcing.md`, matching this repo's
sourcing-report convention.

`notebooks/05_evaluate_segmenter.ipynb` gains a new section 5 running
`ClassicalSegmenter`, plain `YoloSegmenter`, and `TiledYoloSegmenter` side
by side against this photo, printing box counts for each — the qualitative
signal this whole effort is meant to move: `TiledYoloSegmenter` should find
meaningfully more boxes than plain `YoloSegmenter` on this specific
high-resolution, densely-packed image, even before retraining (tiling helps
regardless of what the model was trained on; retraining on mixed data is
the second, independent lever).

## Out of scope (YAGNI)

- Sign-facing / reading-order changes for right-to-left or top-to-bottom
  column reading order — separate, already-noted future work
  (`docs/technical-report/main.tex`, "Known Limitations and Future Work").
- Automatic tile-size/overlap tuning — `TiledYoloSegmenter`'s defaults
  (`tile_size=(640, 640)`, `overlap=0.2`) are used as-is; no sweep.
- Merging `worktree-dense-text-detection` back into `master` — a separate
  step after this plan's tasks land and the retrained checkpoint is
  evaluated, via `superpowers:finishing-a-development-branch`.
