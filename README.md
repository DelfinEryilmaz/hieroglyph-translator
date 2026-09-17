# Egyptian Hieroglyph → English Gloss Translator

Takes a photo of Egyptian hieroglyphs and outputs, per detected sign, its
Gardiner code, transliteration (when one exists), and English gloss.

This is **not** a fluent sentence translator — hieroglyphic translation
requires Middle Egyptian grammar knowledge that a simple sign-to-word mapping
can't provide, and no large aligned hieroglyph→English sentence corpus exists
to train on. Output is a per-sign gloss list, presented honestly as that.

## Pipeline
Photo → glyph segmentation (classical CV, or a fine-tuned YOLOv8 detector —
see below) → glyph classification (fine-tuned CNN) → sign lookup (Gardiner
list) → reading-order sort → gloss output.

## Known limitations
- Segmentation defaults to classical OpenCV (contour detection) — works
  best on clean/high-contrast photos, struggles on heavily weathered or
  cluttered wall photos. A trained alternative now exists
  (`hieroglyph.segmentation.yolo.YoloSegmenter`, a YOLOv8 detector
  fine-tuned on synthesized composite images — see
  `notebooks/04_train_segmenter.ipynb` and
  `reports/2026-09-15-segmentation-detector-sourcing.md`) but isn't trained
  by default; run the training notebook and drop the result at
  `models/yolo_seg.pt` to have the Streamlit demo pick it up automatically
  (falls back to classical CV if that file isn't present).
- Reading order is a simple row-major (top-to-bottom, left-to-right) sort —
  doesn't account for true Egyptian reading order (which depends on
  sign-facing direction).
- ~51 of 171 sign classes have too few images (fewer than 3) to have any
  held-out test data — the classifier trains on them but its accuracy on
  those specific signs is unverified.

## Model baseline
ResNet18 fine-tuned on the Glyphnet/Pyramid-of-Unas dataset (171 classes,
~4,031 images): **97.3% test accuracy** (620 test images, ~120 classes with
test coverage). Confusions are concentrated among visually similar sign
clusters (e.g. G1/G17/G35/G4; M1/M29/M17) rather than arbitrary errors — see
`notebooks/03_evaluate_classifier.ipynb` for the full per-class report.

`YoloSegmenter` (YOLOv8-segmentation fine-tuned on synthesized composite
images): **mAP50 0.995 / mAP50-95 0.823** on a held-out synthetic test set
(500 composites, 3,179 instances). This measures detection on our own
synthetic compositing style, not real photos — no ground-truth boxes exist
for real photos to compute mAP against; a qualitative check against 10 real
archaeological photos (`notebooks/05_evaluate_segmenter.ipynb`) shows the
trained detector finds noticeably fewer boxes than classical CV per photo,
consistent with classical contour detection over-segmenting texture/noise
that the trained detector correctly ignores — see the notebook for details.

## Setup

```
py -V:3.13 -m venv .venv
.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
```

The last step installs this repo's own `hieroglyph` package in editable mode,
so `from hieroglyph.data...` imports work from scripts, notebooks, and tests
without manual `sys.path` hacking.

## Project layout
- `src/hieroglyph/` — pipeline code (data, models, segmentation, lookup, inference)
- `notebooks/` — data exploration + Colab training/eval notebooks
- `data_tables/gardiner_signs.csv` — sign → transliteration/gloss lookup table
- `app/streamlit_app.py` — demo web app
- `tests/` — unit/integration tests
- `scripts/download_data.py` — dataset download helper
- `scripts/download_pretrained_yolo.py` — pretrained segmentation-detector
  checkpoint download helper
- `reports/` — reference material and notes collected while building this
  (dataset sourcing, Gardiner list sourcing, decisions) — read this for
  context if picking the project back up later
