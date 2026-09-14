# Egyptian Hieroglyph → English Gloss Translator

Takes a photo of Egyptian hieroglyphs and outputs, per detected sign, its
Gardiner code, transliteration (when one exists), and English gloss.

This is **not** a fluent sentence translator — hieroglyphic translation
requires Middle Egyptian grammar knowledge that a simple sign-to-word mapping
can't provide, and no large aligned hieroglyph→English sentence corpus exists
to train on. Output is a per-sign gloss list, presented honestly as that.

## Pipeline
Photo → glyph segmentation (classical CV) → glyph classification (fine-tuned
CNN) → sign lookup (Gardiner list) → reading-order sort → gloss output.

## Known limitations
- Segmentation is classical OpenCV (contour detection), not a trained
  detector — works best on clean/high-contrast photos, struggles on heavily
  weathered or cluttered wall photos. Built behind a swappable interface so a
  trained detector can replace it later.
- Reading order is a simple row-major (top-to-bottom, left-to-right) sort —
  doesn't account for true Egyptian reading order (which depends on
  sign-facing direction).

## Setup

```
py -V:3.13 -m venv .venv
.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv/Scripts/python -m pip install -r requirements.txt
```

## Project layout
- `src/hieroglyph/` — pipeline code (data, models, segmentation, lookup, inference)
- `notebooks/` — data exploration + Colab training/eval notebooks
- `data_tables/gardiner_signs.csv` — sign → transliteration/gloss lookup table
- `app/streamlit_app.py` — demo web app
- `tests/` — unit/integration tests
- `scripts/download_data.py` — dataset download helper
- `reports/` — reference material and notes collected while building this
  (dataset sourcing, Gardiner list sourcing, decisions) — read this for
  context if picking the project back up later
