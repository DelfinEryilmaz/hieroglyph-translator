# Glyph Segmentation Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a trained YOLOv8-segmentation-based `Segmenter` (fine-tuned from a pretrained checkpoint on synthesized composite training images) as a swappable alternative to `ClassicalSegmenter`, wired into the Streamlit demo with automatic fallback.

**Architecture:** A new pure-Python composite-image synthesizer builds YOLO-format training data from the existing single-glyph crops in `data/raw/`. A new `YoloSegmenter` implements the existing `Segmenter` ABC by wrapping `ultralytics.YOLO`. Training happens on Colab (mirrors `notebooks/02_train_classifier.ipynb`); evaluation happens locally on CPU (mirrors `notebooks/03_evaluate_classifier.ipynb`). The Streamlit app picks up the trained checkpoint automatically if present, falling back to `ClassicalSegmenter` otherwise.

**Tech Stack:** Python 3.13, OpenCV, NumPy, PyTorch/torchvision (existing), `ultralytics` (new dependency, this plan).

**Spec:** `docs/superpowers/specs/2026-09-15-glyph-segmentation-detector-design.md`

## Global Constraints

- Python `>=3.13` (per `pyproject.toml`).
- Local inference is CPU-only; training happens on Colab (GPU) — same convention `requirements.txt` and `notebooks/02_train_classifier.ipynb` already use for the classifier.
- `BoundingBox` (`src/hieroglyph/segmentation/types.py`) stays `(x, y, width, height)` ints — do not change its shape.
- `Segmenter.detect(image: np.ndarray) -> list[BoundingBox]` (`src/hieroglyph/segmentation/base.py`) is the interface every segmentation implementation must satisfy — do not change its signature.
- The detector is single-class (`0: hieroglyph`) — per-sign identity stays the separate ResNet18 classifier stage (`src/hieroglyph/models/classifier.py`). Do not conflate the two.
- Pretrained checkpoint source: MIT-licensed `best.pt` from `EngAdhamTamer/hieroglyph-detection` (`https://github.com/EngAdhamTamer/hieroglyph-detection/releases/download/v1.0.0/best.pt`) — provenance recorded in `reports/2026-09-15-segmentation-detector-sourcing.md` (Task 3).
- Tests follow this repo's existing convention: deterministic synthetic fixtures built inline in the test file (see `tests/test_segmentation.py`, `tests/test_inference.py`) — no external fixture files, no mocking of third-party classes; extract pure logic into its own function instead so it's testable without the third-party object.
- `pytest` config already sets `pythonpath = ["src"]` (`pyproject.toml`) — tests import `hieroglyph.*` directly, no path hacks needed.

---

## Task 1: Composite image synthesizer

**Files:**
- Create: `src/hieroglyph/segmentation/synthesize.py`
- Create: `tests/test_synthesize.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `hieroglyph.segmentation.types.BoundingBox` (existing: `x, y, width, height` ints, plus `.x2`, `.y2`, `.area` properties).
- Produces (used by Task 4's notebook):
  - `SynthesizedComposite` dataclass: `image: np.ndarray`, `boxes: list[BoundingBox]`.
  - `list_crop_paths(raw_dir: Path) -> list[Path]`
  - `split_crop_paths(crop_paths: list[Path], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0) -> tuple[list[Path], list[Path], list[Path]]`
  - `synthesize_composite(crops: list[np.ndarray], canvas_size: tuple[int, int] = (640, 640), background_value: int = 210, max_overlap_fraction: float = 0.15, max_attempts_per_crop: int = 20, rng: random.Random | None = None) -> SynthesizedComposite`
  - `generate_dataset(crop_paths: list[Path], output_dir: Path, num_composites: int, crops_per_composite: tuple[int, int] = (3, 10), canvas_size: tuple[int, int] = (640, 640), seed: int = 0) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_synthesize.py`:

```python
import random
from pathlib import Path

import cv2
import numpy as np

from hieroglyph.segmentation.synthesize import (
    box_to_yolo_line,
    generate_dataset,
    list_crop_paths,
    paste_crop,
    split_crop_paths,
    synthesize_composite,
)
from hieroglyph.segmentation.types import BoundingBox


def _dark_square_crop(size=10, background=220, ink=50):
    crop = np.full((size, size), background, dtype=np.uint8)
    crop[2:8, 2:8] = ink
    return crop


def test_paste_crop_only_transfers_dark_pixels():
    canvas = np.full((20, 20), 200, dtype=np.uint8)
    crop = _dark_square_crop()

    paste_crop(canvas, crop, x=5, y=5)

    # Dark ink pixels landed on the canvas
    assert canvas[7, 7] == 50
    # The crop's own light background pixels did NOT overwrite the canvas
    assert canvas[5, 5] == 200


def test_synthesize_composite_places_one_box_per_crop_when_room_allows():
    crops = [_dark_square_crop(), _dark_square_crop(), _dark_square_crop()]
    rng = random.Random(0)

    composite = synthesize_composite(crops, canvas_size=(200, 200), rng=rng)

    assert len(composite.boxes) == 3
    assert composite.image.shape == (200, 200)


def test_synthesize_composite_is_deterministic_given_same_seed():
    crops = [_dark_square_crop(), _dark_square_crop()]

    result_a = synthesize_composite(crops, canvas_size=(100, 100), rng=random.Random(42))
    result_b = synthesize_composite(crops, canvas_size=(100, 100), rng=random.Random(42))

    assert [(b.x, b.y) for b in result_a.boxes] == [(b.x, b.y) for b in result_b.boxes]


def test_box_to_yolo_line_normalizes_to_unit_range():
    box = BoundingBox(x=10, y=20, width=30, height=40)

    line = box_to_yolo_line(box, canvas_w=100, canvas_h=200)

    cls, cx, cy, w, h = line.split()
    assert cls == "0"
    assert abs(float(cx) - 0.25) < 1e-6  # (10 + 15) / 100
    assert abs(float(cy) - 0.20) < 1e-6  # (20 + 20) / 200
    assert abs(float(w) - 0.30) < 1e-6
    assert abs(float(h) - 0.20) < 1e-6


def test_list_crop_paths_finds_all_pngs_under_class_folders(tmp_path: Path):
    for cls in ["A1", "B2"]:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        cv2.imwrite(str(cls_dir / "0.png"), _dark_square_crop())

    paths = list_crop_paths(tmp_path)

    assert len(paths) == 2


def test_split_crop_paths_is_disjoint_and_covers_everything():
    paths = [Path(f"crop_{i}.png") for i in range(100)]

    train, valid, test = split_crop_paths(paths, ratios=(0.8, 0.1, 0.1), seed=0)

    assert len(train) + len(valid) + len(test) == 100
    assert set(train) & set(valid) == set()
    assert set(train) & set(test) == set()
    assert set(valid) & set(test) == set()
    assert len(train) == 80


def test_generate_dataset_writes_matching_images_and_labels(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    for cls in ["A1", "B2"]:
        cls_dir = raw_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(3):
            cv2.imwrite(str(cls_dir / f"{i}.png"), _dark_square_crop())

    crop_paths = list_crop_paths(raw_dir)
    output_dir = tmp_path / "synthetic"
    generate_dataset(
        crop_paths, output_dir, num_composites=4, crops_per_composite=(1, 2), canvas_size=(100, 100), seed=1
    )

    images = sorted((output_dir / "images").glob("*.png"))
    labels = sorted((output_dir / "labels").glob("*.txt"))
    assert len(images) == 4
    assert len(labels) == 4

    for label_path in labels:
        lines = [l for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for line in lines:
            parts = line.split()
            assert parts[0] == "0"
            assert len(parts) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_synthesize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hieroglyph.segmentation.synthesize'`

- [ ] **Step 3: Write the implementation**

Create `src/hieroglyph/segmentation/synthesize.py`:

```python
"""Synthesize labeled multi-glyph training images from single-glyph crops.

The classifier's dataset (data/raw/<gardiner_code>/*.png) is per-glyph
crops -- no real photo has ever been annotated with per-glyph bounding
boxes. To train a detector we manufacture that annotation for free: paste
several crops onto a plain background at random positions, and the pasted
positions *are* the ground-truth boxes, no manual labeling needed.

Only the glyph's ink (darker-than-background pixels, matching
ClassicalSegmenter's own `invert=True` assumption) gets pasted -- pasting
the crop's own light background too would stamp a visible rectangle onto
the canvas, which no real photo looks like.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hieroglyph.segmentation.types import BoundingBox

FOREGROUND_THRESHOLD = 180  # pixel darker than this = glyph ink, not crop background


@dataclass
class SynthesizedComposite:
    image: np.ndarray
    boxes: list[BoundingBox]


def _foreground_mask(crop_gray: np.ndarray) -> np.ndarray:
    return crop_gray < FOREGROUND_THRESHOLD


def paste_crop(canvas: np.ndarray, crop_gray: np.ndarray, x: int, y: int) -> None:
    """Paste only crop_gray's dark (glyph) pixels onto canvas at (x, y), in place.

    x, y must place the crop fully inside canvas -- callers are responsible
    for choosing in-bounds positions.
    """
    h, w = crop_gray.shape
    mask = _foreground_mask(crop_gray)
    region = canvas[y : y + h, x : x + w]
    region[mask] = crop_gray[mask]


def _boxes_overlap_fraction(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection area as a fraction of the smaller box's area
    (0 = no overlap, 1 = one fully inside the other)."""
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    return intersection / min(a.area, b.area)


def synthesize_composite(
    crops: list[np.ndarray],
    canvas_size: tuple[int, int] = (640, 640),
    background_value: int = 210,
    max_overlap_fraction: float = 0.15,
    max_attempts_per_crop: int = 20,
    rng: random.Random | None = None,
) -> SynthesizedComposite:
    """Paste `crops` (grayscale numpy arrays) onto one synthetic canvas at
    random, non-overlapping-ish positions. Crops that can't be placed
    within max_attempts_per_crop tries (canvas too crowded) are skipped.
    """
    rng = rng or random.Random()
    canvas_w, canvas_h = canvas_size
    canvas = np.full((canvas_h, canvas_w), background_value, dtype=np.uint8)
    boxes: list[BoundingBox] = []

    for crop in crops:
        h, w = crop.shape
        if w >= canvas_w or h >= canvas_h:
            continue  # crop too big for this canvas, skip rather than crash

        for _ in range(max_attempts_per_crop):
            x = rng.randint(0, canvas_w - w)
            y = rng.randint(0, canvas_h - h)
            candidate = BoundingBox(x=x, y=y, width=w, height=h)
            if all(_boxes_overlap_fraction(candidate, placed) <= max_overlap_fraction for placed in boxes):
                paste_crop(canvas, crop, x, y)
                boxes.append(candidate)
                break

    return SynthesizedComposite(image=canvas, boxes=boxes)


def box_to_yolo_line(box: BoundingBox, canvas_w: int, canvas_h: int) -> str:
    """One YOLO-format label line: `class cx cy w h`, all normalized to [0, 1]."""
    cx = (box.x + box.width / 2) / canvas_w
    cy = (box.y + box.height / 2) / canvas_h
    w = box.width / canvas_w
    h = box.height / canvas_h
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def write_composite(composite: SynthesizedComposite, image_path: Path, label_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), composite.image)
    canvas_h, canvas_w = composite.image.shape
    lines = [box_to_yolo_line(box, canvas_w, canvas_h) for box in composite.boxes]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def list_crop_paths(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("*/*.png"))


def split_crop_paths(
    crop_paths: list[Path],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split crop files (not composites) into train/valid/test pools so no
    individual glyph crop ever appears in more than one split -- otherwise
    the detector could "pass" a test purely by having memorized that exact
    crop's pixels during training, rather than actually generalizing.
    """
    shuffled = crop_paths.copy()
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    return shuffled[:n_train], shuffled[n_train : n_train + n_valid], shuffled[n_train + n_valid :]


def generate_dataset(
    crop_paths: list[Path],
    output_dir: Path,
    num_composites: int,
    crops_per_composite: tuple[int, int] = (3, 10),
    canvas_size: tuple[int, int] = (640, 640),
    seed: int = 0,
) -> None:
    """Generate `num_composites` synthetic training images + YOLO labels
    by sampling from `crop_paths` (the caller's chosen pool -- e.g. a
    train/valid/test-specific subset from split_crop_paths, so callers
    control leakage across splits).

    Writes output_dir/images/composite_%04d.png and
    output_dir/labels/composite_%04d.txt.
    """
    if not crop_paths:
        raise ValueError("crop_paths is empty")

    rng = random.Random(seed)
    for i in range(num_composites):
        n_crops = rng.randint(*crops_per_composite)
        chosen_paths = rng.choices(crop_paths, k=n_crops)
        crops = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in chosen_paths]

        composite = synthesize_composite(crops, canvas_size=canvas_size, rng=rng)

        write_composite(
            composite,
            image_path=output_dir / "images" / f"composite_{i:04d}.png",
            label_path=output_dir / "labels" / f"composite_{i:04d}.txt",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_synthesize.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Ignore generated data directories**

`.gitignore` already excludes `data/raw/` and `data/processed/` but not
the new directories this synthesizer and the eval notebook generate/download
into. Add two lines to `.gitignore` (after the existing `data/processed/`
line):

```
data/synthetic_composites/
data/real_eval_photos/
```

- [ ] **Step 6: Commit**

```bash
git add src/hieroglyph/segmentation/synthesize.py tests/test_synthesize.py .gitignore
git commit -m "feat: add composite-image synthesizer for detector training data"
```

---

## Task 2: `YoloSegmenter`

**Files:**
- Modify: `requirements.txt`
- Create: `src/hieroglyph/segmentation/yolo.py`
- Modify: `tests/test_segmentation.py`

**Interfaces:**
- Consumes: `hieroglyph.segmentation.base.Segmenter` (ABC, existing), `hieroglyph.segmentation.classical.ClassicalSegmenter` (existing, no-arg constructor), `hieroglyph.segmentation.types.BoundingBox` (existing).
- Produces (used by Task 5's notebook and Task 6):
  - `YoloSegmenter(checkpoint_path: Path, confidence_threshold: float = 0.25)` — implements `Segmenter.detect(image: np.ndarray) -> list[BoundingBox]`.
  - `load_or_fallback(checkpoint_path: Path) -> Segmenter`

- [ ] **Step 1: Install ultralytics and record its resolved version**

Run:
```bash
.venv/Scripts/python -m pip install ultralytics
.venv/Scripts/python -m pip show ultralytics
```

Note the `Version:` line from the second command's output — use that exact value in Step 2 below (do not guess a version number).

- [ ] **Step 2: Pin the dependency**

Edit `requirements.txt`, adding this block after the existing `opencv-python==5.0.0.93` line (keep the file's existing per-package comment style):

```
# ultralytics wraps YOLOv8 (used for the trained segmentation detector).
# Regular pip install (no special CPU index needed, unlike torch above).
ultralytics==<version from Step 1>
```

- [ ] **Step 3: Write the failing tests**

Append to `tests/test_segmentation.py` (add these imports to the existing import block at the top, and these functions at the end of the file):

```python
from pathlib import Path

from hieroglyph.segmentation.types import BoundingBox
from hieroglyph.segmentation.yolo import _xyxy_to_boxes, load_or_fallback


def test_xyxy_to_boxes_converts_corner_format_to_xywh():
    xyxy = [(10.0, 20.0, 40.0, 50.0)]

    boxes = _xyxy_to_boxes(xyxy)

    assert boxes == [BoundingBox(x=10, y=20, width=30, height=30)]


def test_xyxy_to_boxes_rounds_fractional_coordinates():
    xyxy = [(10.4, 20.6, 40.2, 50.8)]

    boxes = _xyxy_to_boxes(xyxy)

    assert boxes == [BoundingBox(x=10, y=21, width=30, height=30)]


def test_load_or_fallback_returns_classical_when_no_checkpoint(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.pt"

    segmenter = load_or_fallback(missing_path)

    assert isinstance(segmenter, ClassicalSegmenter)
```

(`ClassicalSegmenter` is already imported at the top of this file.)

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_segmentation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hieroglyph.segmentation.yolo'`

- [ ] **Step 5: Write the implementation**

Create `src/hieroglyph/segmentation/yolo.py`:

```python
"""A trained-detector Segmenter: YOLOv8 instance segmentation, used only
for its bounding boxes (see segmentation/types.py -- our interface never
needed the polygon masks YOLOv8-seg also produces).

Fine-tuned from a pretrained checkpoint sourced in
reports/2026-09-15-segmentation-detector-sourcing.md -- see that file for
where the starting weights came from and why we didn't train from scratch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from hieroglyph.segmentation.base import Segmenter
from hieroglyph.segmentation.classical import ClassicalSegmenter
from hieroglyph.segmentation.types import BoundingBox


def _xyxy_to_boxes(xyxy: list[tuple[float, float, float, float]]) -> list[BoundingBox]:
    """Convert YOLO's (x1, y1, x2, y2) corner format into our BoundingBox
    (x, y, width, height) convention. Kept separate from YoloSegmenter.detect
    so this conversion logic is testable without loading a real model.
    """
    boxes = []
    for x1, y1, x2, y2 in xyxy:
        boxes.append(
            BoundingBox(x=round(x1), y=round(y1), width=round(x2 - x1), height=round(y2 - y1))
        )
    return boxes


class YoloSegmenter(Segmenter):
    def __init__(self, checkpoint_path: Path, confidence_threshold: float = 0.25) -> None:
        self.model = YOLO(str(checkpoint_path))
        self.confidence_threshold = confidence_threshold

    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        results = self.model.predict(image, conf=self.confidence_threshold, verbose=False)
        xyxy = results[0].boxes.xyxy.tolist()
        return _xyxy_to_boxes(xyxy)


def load_or_fallback(checkpoint_path: Path) -> Segmenter:
    """YoloSegmenter if a fine-tuned checkpoint exists at checkpoint_path,
    otherwise ClassicalSegmenter -- lets callers (e.g. the Streamlit demo)
    work before notebooks/04_train_segmenter.ipynb has ever been run.
    """
    if checkpoint_path.exists():
        return YoloSegmenter(checkpoint_path)
    return ClassicalSegmenter()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_segmentation.py -v`
Expected: PASS (all tests in the file, old and new)

- [ ] **Step 7: Commit**

```bash
git add requirements.txt src/hieroglyph/segmentation/yolo.py tests/test_segmentation.py
git commit -m "feat: add YoloSegmenter, a trained-detector Segmenter implementation"
```

---

## Task 3: Pretrained checkpoint download script + sourcing report

**Files:**
- Create: `scripts/download_pretrained_yolo.py`
- Create: `reports/2026-09-15-segmentation-detector-sourcing.md`

**Interfaces:**
- Produces (used by Task 4's notebook): a CLI script, `python scripts/download_pretrained_yolo.py [--dest PATH]`, default dest `models/yolo_seg_pretrained.pt`.

No automated test for this task — `scripts/download_data.py` (the existing precedent this mirrors) also has no test, since it's a thin network-download wrapper.

- [ ] **Step 1: Create the download script**

Create `scripts/download_pretrained_yolo.py`:

```python
"""Download the pretrained YOLOv8-segmentation checkpoint we fine-tune for
glyph detection (see reports/2026-09-15-segmentation-detector-sourcing.md
for where this comes from and why).

No API token needed -- it's a public GitHub release asset.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

CHECKPOINT_URL = (
    "https://github.com/EngAdhamTamer/hieroglyph-detection/releases/download/v1.0.0/best.pt"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=str(PROJECT_ROOT / "models" / "yolo_seg_pretrained.pt"))
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {CHECKPOINT_URL} -> {dest} ...")
    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, dest)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(f"You can also download it manually from {CHECKPOINT_URL}", file=sys.stderr)
        return 1

    print(f"Done. Saved to {dest} ({dest.stat().st_size} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify it runs**

Run: `.venv/Scripts/python scripts/download_pretrained_yolo.py --dest models/yolo_seg_pretrained.pt`
Expected: exits 0, prints a "Done." line, and `models/yolo_seg_pretrained.pt` exists and is a few MB.

- [ ] **Step 3: Write the sourcing report**

Create `reports/2026-09-15-segmentation-detector-sourcing.md`:

```markdown
# Segmentation detector: pretrained checkpoint source

**Chosen starting point:** `best.pt` from
https://github.com/EngAdhamTamer/hieroglyph-detection
(release: https://github.com/EngAdhamTamer/hieroglyph-detection/releases/tag/v1.0.0)

## Why
Research during Phase 12 planning found no bulk downloadable real-photo
dataset with glyph bounding-box/segmentation annotations (the closest
candidate, this same repo's own 235,436-instance training set, is
explicitly **not** distributed — confirmed by reading its
`dataset/data.yaml`, which says the images/labels aren't included "due to
size constraints"). Its trained checkpoint *is* distributed, though, and:

- MIT licensed
- YOLOv8-segmentation, single class ("hieroglyph") — matches our need
  exactly, since sign classification is already a separate downstream
  stage (the ResNet18 classifier) in our pipeline
- Reports 70.9% mAP@0.5 on real archaeological photos (temple walls,
  obelisks, papyrus, stelae)
- 6.8MB, downloaded via `scripts/download_pretrained_yolo.py`

We fine-tune this checkpoint on our own synthesized composite images
(`hieroglyph.segmentation.synthesize`, built from `data/raw/` single-glyph
crops) rather than training from scratch, so the model starts from weights
already exposed to real photos instead of only ever seeing our synthetic
compositing style.

## Other candidates considered, not used
- [Roboflow "egyptian-hieroglyphs"](https://universe.roboflow.com/custom-yolov8-ljpde/egyptian-hieroglyphs)
  — real photos, 40 images, 19 Gardiner-coded classes, CC BY 4.0. Too
  small to train on; used instead as a real-photo qualitative eval set in
  `notebooks/05_evaluate_segmenter.ipynb`.
- Roboflow "Hieroglyphic_3" (nadine-mostafa) — found during search, not
  independently verified (couldn't confirm image count/license without a
  Roboflow account); not pursued.

## License note
`best.pt`'s MIT license permits this reuse (including fine-tuning and
redistributing derivative weights). We keep this file as the paper trail
for that provenance, matching this repo's convention (see
`reports/dataset-source.md`, `reports/gardiner-lookup-table-sourcing.md`).
```

- [ ] **Step 4: Commit**

```bash
git add scripts/download_pretrained_yolo.py reports/2026-09-15-segmentation-detector-sourcing.md
git commit -m "feat: add pretrained detector checkpoint download script + sourcing notes"
```

(Leave the downloaded `models/yolo_seg_pretrained.pt` file uncommitted — `.gitignore` already excludes `models/*.pt`, so a plain `git add scripts/... reports/...` as above won't pick it up.)

---

## Task 4: Training notebook

**Depends on:** Task 1 (`generate_dataset`, `list_crop_paths`, `split_crop_paths`), Task 3 (`scripts/download_pretrained_yolo.py`).

**Files:**
- Create: `notebooks/04_train_segmenter.ipynb`

**Interfaces:**
- Consumes: `hieroglyph.segmentation.synthesize.{list_crop_paths, split_crop_paths, generate_dataset}` (Task 1), `scripts/download_pretrained_yolo.py` (Task 3).
- Produces: (when run manually on Colab by a human with a GPU) a fine-tuned checkpoint the user downloads and places at `models/yolo_seg.pt` locally — this is a human action, not something this task automates.

This is a Colab notebook (mirrors `notebooks/02_train_classifier.ipynb`'s structure exactly) — it is not executed as part of this task; only the `.ipynb` file is created. No pytest coverage applies to notebook cells (matches existing repo convention: notebooks 01-03 have none either).

- [ ] **Step 1: Build the notebook**

Run this one-off Python snippet (e.g. via `.venv/Scripts/python -c "..."` or a scratch `.py` file you delete afterward) to write `notebooks/04_train_segmenter.ipynb`. Match `notebooks/02_train_classifier.ipynb`'s nbformat metadata exactly (`nbformat: 4`, `nbformat_minor: 5`, `kernelspec` name `python3`).

```python
import json
from pathlib import Path

CELLS = [
    ("markdown", """# Train the glyph segmentation detector (Colab, GPU)

Run this notebook on Colab with a GPU runtime
(Runtime -> Change runtime type -> GPU).

This fine-tunes a pretrained YOLOv8-segmentation checkpoint (see
`reports/2026-09-15-segmentation-detector-sourcing.md`) on synthetic
composite images built from our own single-glyph crops
(`data/raw/`) -- see `docs/superpowers/specs/2026-09-15-glyph-segmentation-detector-design.md`
for the full design."""),
    ("markdown", "## 1. Clone the repo and install it"),
    ("code", """!git clone https://github.com/DelfinEryilmaz/hieroglyph-translator.git
%cd hieroglyph-translator
!pip install -e . -q
!pip install ultralytics -q"""),
    ("markdown", "## 2. Check GPU is available"),
    ("code", """import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU detected -- check Runtime > Change runtime type > GPU")"""),
    ("markdown", """## 3. Upload the glyph crop dataset

Same dataset the classifier trains on. Upload the original `archive.zip`
from Kaggle (or a zip of your local `data/raw/` folder) -- either works,
this cell handles both."""),
    ("code", """from google.colab import files

uploaded = files.upload()  # pick archive.zip (from Kaggle) or your own data_raw.zip
uploaded_filename = next(iter(uploaded))"""),
    ("code", """import shutil
import zipfile
from pathlib import Path

DATA_ROOT = Path("data/raw")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(uploaded_filename) as zf:
    zf.extractall(DATA_ROOT)


def class_folders(root):
    return [p for p in root.iterdir() if p.is_dir()]


top_level = class_folders(DATA_ROOT)
# If the zip had one extra wrapper folder (some Windows re-zips do this),
# flatten it so DATA_ROOT/<class>/*.png is the actual layout.
if len(top_level) == 1 and not list(top_level[0].glob("*.png")):
    inner = top_level[0]
    for child in inner.iterdir():
        shutil.move(str(child), str(DATA_ROOT / child.name))
    inner.rmdir()

print(f"{len(class_folders(DATA_ROOT))} class folders under {DATA_ROOT}")"""),
    ("markdown", """## 4. Split crop files, then generate synthetic composites

Splitting happens on individual *crop files* before compositing, not on
composites -- so no single glyph crop's pixels ever appear in both the
train and test splits (see `hieroglyph.segmentation.synthesize.split_crop_paths`'s
docstring)."""),
    ("code", """from pathlib import Path
from hieroglyph.segmentation.synthesize import generate_dataset, list_crop_paths, split_crop_paths

RAW_DIR = Path("data/raw")
SYNTH_DIR = Path("data/synthetic_composites")

crop_paths = list_crop_paths(RAW_DIR)
train_crops, valid_crops, test_crops = split_crop_paths(crop_paths, seed=0)
print(f"{len(train_crops)} train crops, {len(valid_crops)} valid crops, {len(test_crops)} test crops")

generate_dataset(train_crops, SYNTH_DIR / "train", num_composites=4000, seed=0)
generate_dataset(valid_crops, SYNTH_DIR / "valid", num_composites=500, seed=1)
generate_dataset(test_crops, SYNTH_DIR / "test", num_composites=500, seed=2)
print("done generating composites")"""),
    ("code", """import yaml

data_yaml = {
    "path": str(SYNTH_DIR.resolve()),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "names": {0: "hieroglyph"},
    "nc": 1,
}
(SYNTH_DIR / "data.yaml").write_text(yaml.dump(data_yaml), encoding="utf-8")
print((SYNTH_DIR / "data.yaml").read_text())"""),
    ("markdown", "## 5. Download the pretrained checkpoint we're fine-tuning from"),
    ("code", """!python scripts/download_pretrained_yolo.py --dest models/yolo_seg_pretrained.pt"""),
    ("markdown", "## 6. Fine-tune"),
    ("code", """from ultralytics import YOLO

model = YOLO("models/yolo_seg_pretrained.pt")
train_results = model.train(
    data=str(SYNTH_DIR / "data.yaml"),
    epochs=30,
    imgsz=640,
    device=0,
    project="runs",
    name="yolo_seg_finetune",
)"""),
    ("markdown", """## 7. Download the fine-tuned checkpoint

This is the file that goes into your local `models/yolo_seg.pt` for
`notebooks/05_evaluate_segmenter.ipynb` and the Streamlit demo (Phase 12).
**Rename it to `yolo_seg.pt` locally** after downloading -- both of those
expect that exact filename."""),
    ("code", """from google.colab import files

files.download("runs/yolo_seg_finetune/weights/best.pt")"""),
]

cells_json = []
for cell_type, source in CELLS:
    cell = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cells_json.append(cell)

notebook = {
    "cells": cells_json,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("notebooks/04_train_segmenter.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print("wrote notebooks/04_train_segmenter.ipynb")
```

- [ ] **Step 2: Verify it's valid JSON / valid notebook**

Run: `.venv/Scripts/python -c "import json; json.load(open('notebooks/04_train_segmenter.ipynb', encoding='utf-8')); print('valid json')"`
Expected: prints `valid json`

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_train_segmenter.ipynb
git commit -m "feat: add Colab notebook to fine-tune the glyph segmentation detector"
```

---

## Task 5: Evaluation notebook

**Depends on:** Task 1 (`list_crop_paths`, `split_crop_paths`, `generate_dataset` — must use the same seeds as Task 4 to regenerate an identical held-out test split), Task 2 (`YoloSegmenter`).

**Files:**
- Create: `notebooks/05_evaluate_segmenter.ipynb`

Runs locally (CPU is fine for evaluation, same reasoning as `notebooks/03_evaluate_classifier.ipynb`) after `models/yolo_seg.pt` has been downloaded from Task 4's notebook run. No pytest coverage applies (matches notebook 03's precedent).

- [ ] **Step 1: Build the notebook**

Same technique as Task 4 Step 1 — run a one-off Python snippet to write the file:

```python
import json
from pathlib import Path

CELLS = [
    ("markdown", """# Evaluate the glyph segmentation detector

Run this locally (CPU is fine) after downloading `models/yolo_seg.pt`
from `notebooks/04_train_segmenter.ipynb`'s Colab run."""),
    ("code", """from pathlib import Path

from ultralytics import YOLO

CHECKPOINT_PATH = Path("..") / "models" / "yolo_seg.pt"

assert CHECKPOINT_PATH.exists(), (
    f"No checkpoint found at {CHECKPOINT_PATH} -- run notebooks/04_train_segmenter.ipynb "
    "on Colab first, download its output, and save it here as yolo_seg.pt."
)"""),
    ("markdown", """## 1. Regenerate the held-out synthetic test split

Uses the same seeds as `notebooks/04_train_segmenter.ipynb` (`split_crop_paths(..., seed=0)`,
then `generate_dataset(test_crops, ..., seed=2)`), so this reproduces the
*exact same* test composites the training notebook held out -- no need to
download gigabytes of synthetic images from Colab, `data/raw/` is already
here locally."""),
    ("code", """from hieroglyph.segmentation.synthesize import generate_dataset, list_crop_paths, split_crop_paths

RAW_DIR = Path("..") / "data" / "raw"
SYNTH_DIR = Path("..") / "data" / "synthetic_composites"

crop_paths = list_crop_paths(RAW_DIR)
_, _, test_crops = split_crop_paths(crop_paths, seed=0)
generate_dataset(test_crops, SYNTH_DIR / "test", num_composites=500, seed=2)
print(f"{len(test_crops)} held-out test crops -> {SYNTH_DIR / 'test'}")"""),
    ("code", """import yaml

data_yaml = {
    "path": str(SYNTH_DIR.resolve()),
    "train": "test/images",  # unused by model.val(split="test"), but ultralytics requires the key
    "val": "test/images",
    "test": "test/images",
    "names": {0: "hieroglyph"},
    "nc": 1,
}
(SYNTH_DIR / "eval_data.yaml").write_text(yaml.dump(data_yaml), encoding="utf-8")"""),
    ("markdown", "## 2. Quantitative: mAP on the held-out synthetic test set"),
    ("code", """model = YOLO(str(CHECKPOINT_PATH))
metrics = model.val(data=str(SYNTH_DIR / "eval_data.yaml"), split="test")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")"""),
    ("markdown", """## 3. Qualitative: real photos

Manual step first: download the "egyptian-hieroglyphs" dataset from
https://universe.roboflow.com/custom-yolov8-ljpde/egyptian-hieroglyphs
(Export -> any format, e.g. YOLOv8 -> download zip), unzip into
`../data/real_eval_photos/` so the next cell can find the images."""),
    ("code", """import cv2

from hieroglyph.segmentation.classical import ClassicalSegmenter
from hieroglyph.segmentation.yolo import YoloSegmenter

REAL_EVAL_DIR = Path("..") / "data" / "real_eval_photos"
image_paths = sorted(REAL_EVAL_DIR.rglob("*.jpg")) + sorted(REAL_EVAL_DIR.rglob("*.png"))
assert image_paths, f"No real eval images found under {REAL_EVAL_DIR} -- see the download note above."

classical = ClassicalSegmenter()
trained = YoloSegmenter(CHECKPOINT_PATH)

for path in image_paths[:10]:
    image = cv2.imread(str(path))
    classical_boxes = classical.detect(image)
    trained_boxes = trained.detect(image)
    print(f"{path.name}: classical={len(classical_boxes)} boxes, trained={len(trained_boxes)} boxes")"""),
    ("markdown", """## 4. Record the baseline

Copy the mAP50 number into the README (`## Model baseline` section, next
to the classifier's accuracy) once you've reviewed the qualitative
comparison above and are ready to treat this checkpoint as the one
`app/streamlit_app.py` builds on."""),
]

cells_json = []
for cell_type, source in CELLS:
    cell = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cells_json.append(cell)

notebook = {
    "cells": cells_json,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("notebooks/05_evaluate_segmenter.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print("wrote notebooks/05_evaluate_segmenter.ipynb")
```

- [ ] **Step 2: Verify it's valid JSON**

Run: `.venv/Scripts/python -c "import json; json.load(open('notebooks/05_evaluate_segmenter.ipynb', encoding='utf-8')); print('valid json')"`
Expected: prints `valid json`

- [ ] **Step 3: Commit**

```bash
git add notebooks/05_evaluate_segmenter.ipynb
git commit -m "feat: add local evaluation notebook for the glyph segmentation detector"
```

---

## Task 6: Streamlit integration

**Depends on:** Task 2 (`YoloSegmenter`, `load_or_fallback`).

**Files:**
- Modify: `app/streamlit_app.py`

- [ ] **Step 1: Add the import**

In `app/streamlit_app.py`, find:

```python
from hieroglyph.lookup.gardiner_lookup import GardinerLookup
from hieroglyph.models.classifier import load_checkpoint
from hieroglyph.pipeline.inference import run_inference
from hieroglyph.utils.visualization import draw_annotated_image
```

Replace with:

```python
from hieroglyph.lookup.gardiner_lookup import GardinerLookup
from hieroglyph.models.classifier import load_checkpoint
from hieroglyph.pipeline.inference import run_inference
from hieroglyph.segmentation.yolo import load_or_fallback
from hieroglyph.utils.visualization import draw_annotated_image
```

- [ ] **Step 2: Add the checkpoint path constant**

Find:

```python
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pt"
```

Replace with:

```python
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pt"
YOLO_CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "yolo_seg.pt"
```

- [ ] **Step 3: Add a cached segmenter loader**

Find:

```python
@st.cache_resource(show_spinner="Loading model...")
def load_model_and_lookup():
    model, class_to_idx = load_checkpoint(CHECKPOINT_PATH)
    lookup = GardinerLookup()
    return model, class_to_idx, lookup
```

Replace with:

```python
@st.cache_resource(show_spinner="Loading model...")
def load_model_and_lookup():
    model, class_to_idx = load_checkpoint(CHECKPOINT_PATH)
    lookup = GardinerLookup()
    return model, class_to_idx, lookup


@st.cache_resource(show_spinner="Loading segmenter...")
def load_segmenter():
    return load_or_fallback(YOLO_CHECKPOINT_PATH)
```

- [ ] **Step 4: Make the sidebar note reflect which segmenter is active**

Find:

```python
    st.markdown(
        '<div class="limitation-note">Segmentation uses classical image '
        "processing, not a trained detector — works best on clean, "
        "high-contrast photos.</div>",
        unsafe_allow_html=True,
    )
```

Replace with:

```python
    if YOLO_CHECKPOINT_PATH.exists():
        st.markdown(
            '<div class="limitation-note">Segmentation uses a YOLOv8 '
            "detector fine-tuned on synthesized composite images — see "
            "notebooks/05_evaluate_segmenter.ipynb for its held-out test "
            "accuracy.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="limitation-note">Segmentation uses classical image '
            "processing, not a trained detector — works best on clean, "
            "high-contrast photos. Run notebooks/04_train_segmenter.ipynb "
            "to train a detector and unlock this upgrade.</div>",
            unsafe_allow_html=True,
        )
```

- [ ] **Step 5: Pass the segmenter into `run_inference`**

Find:

```python
model, class_to_idx, lookup = load_model_and_lookup()
```

Replace with:

```python
model, class_to_idx, lookup = load_model_and_lookup()
segmenter = load_segmenter()
```

Then find:

```python
with st.spinner("Detecting and classifying signs..."):
    result = run_inference(image_bgr, model, class_to_idx, lookup)
```

Replace with:

```python
with st.spinner("Detecting and classifying signs..."):
    result = run_inference(image_bgr, model, class_to_idx, lookup, segmenter=segmenter)
```

- [ ] **Step 6: Verify the file is syntactically valid**

Run: `.venv/Scripts/python -m py_compile app/streamlit_app.py`
Expected: exits 0, no output.

- [ ] **Step 7: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: wire the trained segmenter into the Streamlit demo with fallback"
```

---

## Task 7: README updates

**Depends on:** Tasks 1-6 (documents what they built).

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the pipeline line**

Find:

```
## Pipeline
Photo → glyph segmentation (classical CV) → glyph classification (fine-tuned
CNN) → sign lookup (Gardiner list) → reading-order sort → gloss output.
```

Replace with:

```
## Pipeline
Photo → glyph segmentation (classical CV, or a fine-tuned YOLOv8 detector —
see below) → glyph classification (fine-tuned CNN) → sign lookup (Gardiner
list) → reading-order sort → gloss output.
```

- [ ] **Step 2: Update the segmentation limitation bullet**

Find:

```
- Segmentation is classical OpenCV (contour detection), not a trained
  detector — works best on clean/high-contrast photos, struggles on heavily
  weathered or cluttered wall photos. Built behind a swappable interface so a
  trained detector can replace it later.
```

Replace with:

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

- [ ] **Step 3: Add the new script to the project layout list**

Find:

```
- `scripts/download_data.py` — dataset download helper
```

Replace with:

```
- `scripts/download_data.py` — dataset download helper
- `scripts/download_pretrained_yolo.py` — pretrained segmentation-detector
  checkpoint download helper
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document the trained segmentation detector"
```

---

## Suggested execution order

- **Wave A (parallel, no interdependencies):** Task 1, Task 2, Task 3
- **Wave B (parallel, each depends only on specific Wave A tasks):** Task 4, Task 5, Task 6
- **Wave C (sequential, after Wave B):** Task 7
