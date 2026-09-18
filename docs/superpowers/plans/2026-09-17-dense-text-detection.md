# Dense/Papyrus Text Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the trained segmentation detector actually usable on dense, real-photo text (papyrus columns), by wiring in the already-implemented `TiledYoloSegmenter`, switching detector training to the already-implemented dense/column composite mix, and adding a real-papyrus qualitative eval.

**Architecture:** Two independent, additive fixes land on top of Phase 12's `YoloSegmenter`: (1) `TiledYoloSegmenter` (`src/hieroglyph/segmentation/tiling.py`, already written and tested but uncommitted) runs inference over overlapping tiles instead of one destructive whole-image resize, wired into the Streamlit app via a new `load_tiled_or_fallback`; (2) `notebooks/04_train_segmenter.ipynb` and `05_evaluate_segmenter.ipynb` switch from `generate_dataset` (sparse-only) to `generate_mixed_dataset` (already written, tested, and committed on this branch), so the model itself eventually sees dense/column training data. A real Papyrus of Ani photo, downloaded via a new script, gives a concrete qualitative check for both fixes.

**Tech Stack:** Python 3.13, OpenCV, NumPy, `ultralytics` (all existing deps — nothing new).

**Spec:** `docs/superpowers/specs/2026-09-17-dense-text-detection-design.md`

## Global Constraints

- Python `>=3.13` (per `pyproject.toml`).
- `BoundingBox` (`src/hieroglyph/segmentation/types.py`) stays `(x, y, width, height)` ints — do not change its shape.
- `Segmenter.detect(image: np.ndarray) -> list[BoundingBox]` (`src/hieroglyph/segmentation/base.py`) is the interface every segmentation implementation must satisfy — do not change its signature.
- `src/hieroglyph/segmentation/yolo.py`'s existing `load_or_fallback` and `YoloSegmenter` stay unchanged — `notebooks/05_evaluate_segmenter.ipynb`'s side-by-side comparison cell needs plain (non-tiled) `YoloSegmenter` to exist independently.
- No new PyPI dependencies — `ultralytics==8.4.152` (`requirements.txt`) already covers everything this plan needs.
- Tests follow this repo's existing convention: deterministic synthetic fixtures built inline in the test file — no external fixture files, no mocking of third-party classes.
- `pytest` config already sets `pythonpath = ["src"]` (`pyproject.toml`) — tests import `hieroglyph.*` directly, no path hacks needed.
- This repo is currently checked out at `.claude/worktrees/dense-text-detection` on branch `worktree-dense-text-detection`, diverged from `master` at commit `20a6113`. All tasks below operate on this worktree; merging back into `master` is out of scope for this plan.

---

## Task 1: Commit the already-implemented tiling module

**Files:**
- Commit (already written, untracked): `src/hieroglyph/segmentation/tiling.py`
- Commit (already written, untracked): `tests/test_tiling.py`

**Interfaces:**
- Produces (used by Task 2): `TiledYoloSegmenter(checkpoint_path: Path, tile_size=(640,640), overlap=0.2, confidence_threshold=0.25, iou_threshold=0.5)` — implements `Segmenter.detect`. `generate_tile_origins`, `merge_tiled_detections`, `_iou`, `_offset_box` (internal helpers, already covered by `tests/test_tiling.py`).

Both files already exist in the working tree, fully implemented, with all 15 tests in `tests/test_tiling.py` passing — this task is verify-then-commit, not new implementation.

- [ ] **Step 1: Verify the existing tests pass**

Run: `.venv/Scripts/python -m pytest tests/test_tiling.py -v`
Expected: PASS (15 tests)

- [ ] **Step 2: Commit**

```bash
git add src/hieroglyph/segmentation/tiling.py tests/test_tiling.py
git commit -m "feat: add TiledYoloSegmenter for sliding-window inference on large photos"
```

---

## Task 2: Wire `TiledYoloSegmenter` into the Streamlit app

**Depends on:** Task 1 (`TiledYoloSegmenter`, committed).

**Files:**
- Modify: `src/hieroglyph/segmentation/tiling.py`
- Modify: `tests/test_tiling.py`
- Modify: `app/streamlit_app.py`

**Interfaces:**
- Consumes: `hieroglyph.segmentation.base.Segmenter`, `hieroglyph.segmentation.classical.ClassicalSegmenter` (existing, no-arg constructor), `TiledYoloSegmenter` (Task 1).
- Produces (used by `app/streamlit_app.py`): `load_tiled_or_fallback(checkpoint_path: Path) -> Segmenter`.

- [ ] **Step 1: Write the failing test**

Add this import to the top of `tests/test_tiling.py` (alongside the existing `from hieroglyph.segmentation.tiling import (...)` block) and this function at the end of the file:

```python
from pathlib import Path

from hieroglyph.segmentation.classical import ClassicalSegmenter
from hieroglyph.segmentation.tiling import load_tiled_or_fallback


def test_load_tiled_or_fallback_returns_classical_when_no_checkpoint(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.pt"

    segmenter = load_tiled_or_fallback(missing_path)

    assert isinstance(segmenter, ClassicalSegmenter)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_tiling.py::test_load_tiled_or_fallback_returns_classical_when_no_checkpoint -v`
Expected: FAIL with `ImportError: cannot import name 'load_tiled_or_fallback'`

- [ ] **Step 3: Implement**

In `src/hieroglyph/segmentation/tiling.py`, add this import to the existing import block (alongside `from hieroglyph.segmentation.base import Segmenter`):

```python
from hieroglyph.segmentation.classical import ClassicalSegmenter
```

Then add this function at the end of the file, after `TiledYoloSegmenter`:

```python
def load_tiled_or_fallback(checkpoint_path: Path) -> Segmenter:
    """TiledYoloSegmenter if a fine-tuned checkpoint exists at checkpoint_path,
    otherwise ClassicalSegmenter -- mirrors yolo.py's load_or_fallback exactly,
    kept here (not there) to avoid a circular import, since this module
    already imports from yolo.py.
    """
    if checkpoint_path.exists():
        return TiledYoloSegmenter(checkpoint_path)
    return ClassicalSegmenter()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_tiling.py -v`
Expected: PASS (16 tests, old and new)

- [ ] **Step 5: Wire it into the Streamlit app**

In `app/streamlit_app.py`, find:

```python
from hieroglyph.segmentation.yolo import load_or_fallback
```

Replace with:

```python
from hieroglyph.segmentation.tiling import load_tiled_or_fallback
```

Then find:

```python
@st.cache_resource(show_spinner="Loading segmenter...")
def load_segmenter():
    return load_or_fallback(YOLO_CHECKPOINT_PATH)
```

Replace with:

```python
@st.cache_resource(show_spinner="Loading segmenter...")
def load_segmenter():
    return load_tiled_or_fallback(YOLO_CHECKPOINT_PATH)
```

- [ ] **Step 6: Verify the file is syntactically valid**

Run: `.venv/Scripts/python -m py_compile app/streamlit_app.py`
Expected: exits 0, no output.

- [ ] **Step 7: Commit**

```bash
git add src/hieroglyph/segmentation/tiling.py tests/test_tiling.py app/streamlit_app.py
git commit -m "feat: wire TiledYoloSegmenter into the Streamlit demo"
```

---

## Task 3: Real-papyrus eval photo download script + sourcing report

**Files:**
- Create: `scripts/download_papyrus_eval_photo.py`
- Create: `reports/2026-09-17-papyrus-eval-photo-sourcing.md`

**Interfaces:**
- Produces (used by Task 5's notebook): a CLI script, `python scripts/download_papyrus_eval_photo.py [--dest PATH]`, default dest `data/real_eval_photos/papyrus/papyrus_of_ani_bm_sheet_12.jpg`.

No automated test for this task — mirrors `scripts/download_pretrained_yolo.py`'s precedent (also untested, thin network-download wrapper).

- [ ] **Step 1: Create the download script**

Create `scripts/download_papyrus_eval_photo.py`:

```python
"""Download a real, densely-packed papyrus photo for qualitative
segmentation eval (see reports/2026-09-17-papyrus-eval-photo-sourcing.md
for what this image is and why).

No API token needed -- it's a public-domain file hosted on Wikimedia
Commons. Wikimedia's servers reject requests with no User-Agent header
(HTTP 403) -- plain urllib.request.urlretrieve sends none, so this uses an
explicit Request with a descriptive User-Agent instead, per Wikimedia's
own request policy (https://meta.wikimedia.org/wiki/User-Agent_policy).
"""

import argparse
import sys
import urllib.request
from pathlib import Path

IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/c/c9/Papyrus_of_Ani_BM_Sheet_12.jpg"
)
USER_AGENT = (
    "hieroglyph-translator-research-script/1.0 "
    "(https://github.com/DelfinEryilmaz/hieroglyph-translator)"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        default=str(PROJECT_ROOT / "data" / "real_eval_photos" / "papyrus" / "papyrus_of_ani_bm_sheet_12.jpg"),
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {IMAGE_URL} -> {dest} ...")
    try:
        request = urllib.request.Request(IMAGE_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request) as response, open(dest, "wb") as f:
            f.write(response.read())
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(f"You can also download it manually from {IMAGE_URL}", file=sys.stderr)
        return 1

    print(f"Done. Saved to {dest} ({dest.stat().st_size} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify it runs**

Run: `.venv/Scripts/python scripts/download_papyrus_eval_photo.py`
Expected: exits 0, prints a "Done." line, and `data/real_eval_photos/papyrus/papyrus_of_ani_bm_sheet_12.jpg` exists and is a few MB (~3MB).

- [ ] **Step 3: Write the sourcing report**

Create `reports/2026-09-17-papyrus-eval-photo-sourcing.md`:

```markdown
# Real-papyrus qualitative eval photo: source

**Chosen:** "Papyrus of Ani BM Sheet 12.jpg" on Wikimedia Commons
https://commons.wikimedia.org/wiki/File:Papyrus_of_Ani_BM_Sheet_12.jpg
(direct file: https://upload.wikimedia.org/wikipedia/commons/c/c9/Papyrus_of_Ani_BM_Sheet_12.jpg,
downloaded via `scripts/download_papyrus_eval_photo.py`)

## Why
`notebooks/05_evaluate_segmenter.ipynb`'s existing qualitative check
(Section 3) uses the Roboflow "egyptian-hieroglyphs" set -- real photos,
but none of them are actual papyrus columns of dense cursive text, which is
exactly the case that exposed the sim2real gap this effort addresses (see
`docs/superpowers/specs/2026-09-17-dense-text-detection-design.md`). This
photo is one sheet of the real Papyrus of Ani (Book of the Dead, ~1250 BCE,
British Museum EA10470), a photographic reproduction of a dense
hieroglyphic-text column -- exactly the layout `synthesize_composite`'s
sparse scatter never trained the detector on.

- 2,414x1,498px -- large enough to actually exercise
  `TiledYoloSegmenter`'s tiling (a single 640x640 tile could not cover it).
- Public domain: Wikimedia Commons' file-info page states it is "public
  domain in its country of origin and other countries and areas where the
  copyright term is the author's life plus 100 years or fewer" (a ~1250 BCE
  papyrus, photographed for the British Museum's own reproduction of a
  reproduction already over a century old).

## License note
Public domain, per Wikimedia Commons' license tag on the file page linked
above. We keep this file as the paper trail for that provenance, matching
this repo's convention (see `reports/dataset-source.md`,
`reports/2026-09-15-segmentation-detector-sourcing.md`).
```

- [ ] **Step 4: Commit**

```bash
git add scripts/download_papyrus_eval_photo.py reports/2026-09-17-papyrus-eval-photo-sourcing.md
git commit -m "feat: add real-papyrus eval photo download script + sourcing notes"
```

(Leave the downloaded `data/real_eval_photos/papyrus/*.jpg` file uncommitted — `.gitignore` already excludes `data/real_eval_photos/`.)

---

## Task 4: Switch training notebook to the mixed dense/sparse dataset

**Depends on:** `generate_mixed_dataset` (already committed on this branch, `src/hieroglyph/segmentation/synthesize.py`).

**Files:**
- Modify: `notebooks/04_train_segmenter.ipynb`

No pytest coverage applies to notebook cells (matches existing repo convention).

- [ ] **Step 1: Patch the notebook**

Run this one-off Python snippet (e.g. via `.venv/Scripts/python -c "..."` or a scratch `.py` file you delete afterward):

```python
import json
from pathlib import Path

NB_PATH = Path("notebooks/04_train_segmenter.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

OLD_MARKDOWN = """## 5. Split crop files, then generate synthetic composites

Splitting happens on individual *crop files* before compositing, not on
composites -- so no single glyph crop's pixels ever appear in both the
train and test splits (see `hieroglyph.segmentation.synthesize.split_crop_paths`'s
docstring)."""

NEW_MARKDOWN = """## 5. Split crop files, then generate synthetic composites

Splitting happens on individual *crop files* before compositing, not on
composites -- so no single glyph crop's pixels ever appear in both the
train and test splits (see `hieroglyph.segmentation.synthesize.split_crop_paths`'s
docstring).

Uses `generate_mixed_dataset`, which mixes dense/column composites (modeling
tightly-packed real papyrus text) with the original sparse/scatter
composites (modeling sparser wall-carving-style photos), so the detector
sees both real-photo layouts during training -- see
`docs/superpowers/specs/2026-09-17-dense-text-detection-design.md`."""

OLD_CODE = """from pathlib import Path
from hieroglyph.segmentation.synthesize import generate_dataset, list_crop_paths, split_crop_paths

RAW_DIR = Path("data/raw")
SYNTH_DIR = Path("data/synthetic_composites")

crop_paths = list_crop_paths(RAW_DIR)
train_crops, valid_crops, test_crops = split_crop_paths(crop_paths, seed=0)
print(f"{len(train_crops)} train crops, {len(valid_crops)} valid crops, {len(test_crops)} test crops")

generate_dataset(train_crops, SYNTH_DIR / "train", num_composites=4000, seed=0)
generate_dataset(valid_crops, SYNTH_DIR / "valid", num_composites=500, seed=1)
generate_dataset(test_crops, SYNTH_DIR / "test", num_composites=500, seed=2)
print("done generating composites")"""

NEW_CODE = """from pathlib import Path
from hieroglyph.segmentation.synthesize import generate_mixed_dataset, list_crop_paths, split_crop_paths

RAW_DIR = Path("data/raw")
SYNTH_DIR = Path("data/synthetic_composites")

crop_paths = list_crop_paths(RAW_DIR)
train_crops, valid_crops, test_crops = split_crop_paths(crop_paths, seed=0)
print(f"{len(train_crops)} train crops, {len(valid_crops)} valid crops, {len(test_crops)} test crops")

generate_mixed_dataset(train_crops, SYNTH_DIR / "train", num_composites=4000, seed=0)
generate_mixed_dataset(valid_crops, SYNTH_DIR / "valid", num_composites=500, seed=1)
generate_mixed_dataset(test_crops, SYNTH_DIR / "test", num_composites=500, seed=2)
print("done generating composites (mix of dense/column and sparse/scatter)")"""

replaced_markdown = replaced_code = False
for cell in nb["cells"]:
    source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if cell["cell_type"] == "markdown" and source == OLD_MARKDOWN:
        cell["source"] = NEW_MARKDOWN
        replaced_markdown = True
    elif cell["cell_type"] == "code" and source == OLD_CODE:
        cell["source"] = NEW_CODE
        replaced_code = True

assert replaced_markdown, "markdown cell to replace not found"
assert replaced_code, "code cell to replace not found"

NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("patched notebooks/04_train_segmenter.ipynb")
```

- [ ] **Step 2: Verify it's valid JSON**

Run: `.venv/Scripts/python -c "import json; json.load(open('notebooks/04_train_segmenter.ipynb', encoding='utf-8')); print('valid json')"`
Expected: prints `valid json`

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_train_segmenter.ipynb
git commit -m "feat: train the segmentation detector on the mixed dense/sparse dataset"
```

---

## Task 5: Evaluation notebook — matching test split + real-papyrus qualitative section

**Depends on:** Task 3 (`scripts/download_papyrus_eval_photo.py`), Task 2 (`TiledYoloSegmenter`), `generate_mixed_dataset` (already committed).

**Files:**
- Modify: `notebooks/05_evaluate_segmenter.ipynb`

No pytest coverage applies to notebook cells.

- [ ] **Step 1: Patch the notebook**

Run this one-off Python snippet:

```python
import json
from pathlib import Path

NB_PATH = Path("notebooks/05_evaluate_segmenter.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

OLD_MARKDOWN_1 = """## 1. Regenerate the held-out synthetic test split

Uses the same seeds as `notebooks/04_train_segmenter.ipynb` (`split_crop_paths(..., seed=0)`,
then `generate_dataset(test_crops, ..., seed=2)`), so this reproduces the
*exact same* test composites the training notebook held out -- no need to
download gigabytes of synthetic images from Colab, `data/raw/` is already
here locally."""

NEW_MARKDOWN_1 = """## 1. Regenerate the held-out synthetic test split

Uses the same seeds as `notebooks/04_train_segmenter.ipynb` (`split_crop_paths(..., seed=0)`,
then `generate_mixed_dataset(test_crops, ..., seed=2)`), so this reproduces
the *exact same* test composites the training notebook held out -- no need
to download gigabytes of synthetic images from Colab, `data/raw/` is
already here locally."""

OLD_CODE_1 = """from hieroglyph.segmentation.synthesize import generate_dataset, list_crop_paths, split_crop_paths

RAW_DIR = Path("..") / "data" / "raw"
SYNTH_DIR = Path("..") / "data" / "synthetic_composites"

crop_paths = list_crop_paths(RAW_DIR)
_, _, test_crops = split_crop_paths(crop_paths, seed=0)
generate_dataset(test_crops, SYNTH_DIR / "test", num_composites=500, seed=2)
print(f"{len(test_crops)} held-out test crops -> {SYNTH_DIR / 'test'}")"""

NEW_CODE_1 = """from hieroglyph.segmentation.synthesize import generate_mixed_dataset, list_crop_paths, split_crop_paths

RAW_DIR = Path("..") / "data" / "raw"
SYNTH_DIR = Path("..") / "data" / "synthetic_composites"

crop_paths = list_crop_paths(RAW_DIR)
_, _, test_crops = split_crop_paths(crop_paths, seed=0)
generate_mixed_dataset(test_crops, SYNTH_DIR / "test", num_composites=500, seed=2)
print(f"{len(test_crops)} held-out test crops -> {SYNTH_DIR / 'test'}")"""

NEW_MARKDOWN_5 = """## 5. Qualitative: dense papyrus text (tiled inference)

Run `scripts/download_papyrus_eval_photo.py` first (see
`reports/2026-09-17-papyrus-eval-photo-sourcing.md` for what this photo is
and why) to fetch a real, densely-packed papyrus photo into
`../data/real_eval_photos/papyrus/`. Compares classical CV, plain
`YoloSegmenter` (whole-image resize to 640px), and `TiledYoloSegmenter`
(overlapping tiles, no destructive resize) -- `TiledYoloSegmenter` finding
meaningfully more boxes here is the qualitative signal this section exists
to check, since tiling helps even before the checkpoint itself is retrained
on dense/column data (Section 1)."""

NEW_CODE_5 = """from hieroglyph.segmentation.tiling import TiledYoloSegmenter

PAPYRUS_DIR = Path("..") / "data" / "real_eval_photos" / "papyrus"
papyrus_paths = sorted(PAPYRUS_DIR.rglob("*.jpg")) + sorted(PAPYRUS_DIR.rglob("*.png"))
assert papyrus_paths, f"No papyrus eval images found under {PAPYRUS_DIR} -- run scripts/download_papyrus_eval_photo.py first."

tiled = TiledYoloSegmenter(CHECKPOINT_PATH)

for path in papyrus_paths:
    image = cv2.imread(str(path))
    classical_boxes = classical.detect(image)
    trained_boxes = trained.detect(image)
    tiled_boxes = tiled.detect(image)
    print(
        f"{path.name}: classical={len(classical_boxes)} boxes, "
        f"trained(whole-image)={len(trained_boxes)} boxes, "
        f"trained(tiled)={len(tiled_boxes)} boxes"
    )"""

replaced_markdown_1 = replaced_code_1 = False
for cell in nb["cells"]:
    source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if cell["cell_type"] == "markdown" and source == OLD_MARKDOWN_1:
        cell["source"] = NEW_MARKDOWN_1
        replaced_markdown_1 = True
    elif cell["cell_type"] == "code" and source == OLD_CODE_1:
        cell["source"] = NEW_CODE_1
        replaced_code_1 = True

assert replaced_markdown_1, "section-1 markdown cell to replace not found"
assert replaced_code_1, "section-1 code cell to replace not found"

nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": NEW_MARKDOWN_5})
new_code_cell = {"cell_type": "code", "metadata": {}, "source": NEW_CODE_5}
new_code_cell["execution_count"] = None
new_code_cell["outputs"] = []
nb["cells"].append(new_code_cell)

NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("patched notebooks/05_evaluate_segmenter.ipynb")
```

- [ ] **Step 2: Verify it's valid JSON**

Run: `.venv/Scripts/python -c "import json; json.load(open('notebooks/05_evaluate_segmenter.ipynb', encoding='utf-8')); print('valid json')"`
Expected: prints `valid json`

- [ ] **Step 3: Commit**

```bash
git add notebooks/05_evaluate_segmenter.ipynb
git commit -m "feat: add real-papyrus qualitative eval, match mixed-dataset test split"
```

---

## Task 6: README updates

**Depends on:** Tasks 1-5 (documents what they built).

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the segmentation limitation bullet**

Find:

```
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
```

Replace with:

```
- Segmentation defaults to classical OpenCV (contour detection) — works
  best on clean/high-contrast photos, struggles on heavily weathered or
  cluttered wall photos. A trained alternative now exists
  (`hieroglyph.segmentation.yolo.YoloSegmenter`/`TiledYoloSegmenter`, a
  YOLOv8 detector fine-tuned on synthesized composite images — see
  `notebooks/04_train_segmenter.ipynb` and
  `reports/2026-09-15-segmentation-detector-sourcing.md`) but isn't trained
  by default; run the training notebook and drop the result at
  `models/yolo_seg.pt` to have the Streamlit demo pick it up automatically
  (falls back to classical CV if that file isn't present).
- Dense, tightly-packed real text (e.g. an actual papyrus column, as
  opposed to a sparser wall-carving-style photo) is a known hard case: the
  original training data was sparse-scatter only. `TiledYoloSegmenter`
  (`src/hieroglyph/segmentation/tiling.py`) fixes the inference-time half
  of this (no destructive whole-image downscale) independently of
  retraining; `generate_mixed_dataset`
  (`src/hieroglyph/segmentation/synthesize.py`) fixes the training-data
  half, but the shipped `models/yolo_seg.pt` hasn't been retrained on it
  yet — see `docs/superpowers/specs/2026-09-17-dense-text-detection-design.md`
  and `notebooks/05_evaluate_segmenter.ipynb` Section 5 for the real-papyrus
  qualitative check.
```

- [ ] **Step 2: Add the new script to the project layout list**

Find:

```
- `scripts/download_pretrained_yolo.py` — pretrained segmentation-detector
  checkpoint download helper
```

Replace with:

```
- `scripts/download_pretrained_yolo.py` — pretrained segmentation-detector
  checkpoint download helper
- `scripts/download_papyrus_eval_photo.py` — real-papyrus qualitative-eval
  photo download helper
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document dense/papyrus text detection support"
```

---

## Suggested execution order

- **Wave A (sequential, Task 2 depends on Task 1):** Task 1, then Task 2
- **Wave B (parallel, independent of Wave A and each other):** Task 3, Task 4
- **Wave C (depends on Task 2 and Task 3):** Task 5
- **Wave D (sequential, after everything else):** Task 6

## Not automated by this plan (manual follow-up)

- Actually re-running `notebooks/04_train_segmenter.ipynb` on Colab (GPU)
  and downloading the retrained `models/yolo_seg.pt` — same convention as
  Phase 12, a human action.
- Re-running `notebooks/05_evaluate_segmenter.ipynb` locally against that
  retrained checkpoint to record new mAP/qualitative numbers in the README.
- Merging `worktree-dense-text-detection` back into `master` — via
  `superpowers:finishing-a-development-branch`, after the above.
