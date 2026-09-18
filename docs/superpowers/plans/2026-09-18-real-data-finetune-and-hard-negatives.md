# Real-Data Fine-Tuning + Hard-Negative Composites Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close two remaining gaps in the segmentation detector's real-photo performance: false positives on non-glyph illustration content (via synthetic hard-negative training composites), and general sim2real texture mismatch (via a short real-data fine-tuning stage using 90 already-downloaded, human-annotated real photos).

**Architecture:** Two independent, additive training-recipe changes, neither touching `Segmenter`, `BoundingBox`, or any downstream pipeline code. (1) `synthesize_negative_composite()` (new) + a `negative_fraction` parameter on the existing `generate_mixed_dataset()` inject procedural, glyph-free "distractor" composites with empty labels into training. (2) A new `real_data.py` module remaps the already-committed real train/valid split's detection-format labels into our detector's single-class segmentation-polygon format, feeding a second `model.train()` stage (continuing from the synthetic-pretrain weights) in `notebooks/04_train_segmenter.ipynb`.

**Tech Stack:** Python 3.13, OpenCV, NumPy — no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-18-real-data-finetune-and-hard-negatives-design.md`

## Global Constraints

- Python `>=3.13` (per `pyproject.toml`).
- `Segmenter.detect(image: np.ndarray) -> list[BoundingBox]` interface stays unchanged — this plan is training-data/training-recipe only.
- No new PyPI dependencies.
- Tests follow this repo's existing convention: deterministic synthetic fixtures built inline in the test file — no external fixture files, no mocking of third-party classes.
- `pytest` config already sets `pythonpath = ["src"]` (`pyproject.toml`) — tests import `hieroglyph.*` directly.
- `generate_mixed_dataset`'s label-line format is the 4-corner-polygon convention (`box_to_yolo_line`), NOT plain `class cx cy w h` — required because the checkpoint is a YOLOv8-*segmentation* model. Any new label-writing code in this plan (both `synthesize_negative_composite`'s empty labels and `real_data.py`'s remapped labels) must match this convention exactly.
- Changing `generate_mixed_dataset`'s default `negative_fraction` from "no negative branch at all" to `0.15` changes its RNG output stream vs. before at the same seed — same accepted tradeoff this repo already made once for the dense-composite crop-count fix; not a bug, no back-compat shim needed.
- This repo is currently checked out at `.claude/worktrees/real-data-and-negatives` on branch `worktree-real-data-and-negatives`, branched from `master` at `214c6e7` (post dense-text-detection merge). All tasks below operate on this worktree.

---

## Task 1: Hard-negative composites

**Files:**
- Modify: `src/hieroglyph/segmentation/synthesize.py`
- Modify: `tests/test_synthesize.py`

**Interfaces:**
- Consumes: `hieroglyph.segmentation.synthesize.SynthesizedComposite`, `FOREGROUND_THRESHOLD` (existing).
- Produces (used by Task 4's notebook and by `generate_mixed_dataset` itself):
  `synthesize_negative_composite(canvas_size: tuple[int,int] = (640,640), background_value: int = 210, background_noise_std: float = 10.0, n_shapes_range: tuple[int,int] = (3,8), ink_value_range: tuple[int,int] = (40,140), thickness_range: tuple[int,int] = (2,6), rng: random.Random | None = None) -> SynthesizedComposite`
  `generate_mixed_dataset(..., negative_fraction: float = 0.15, ...)` (new parameter on the existing function).

- [ ] **Step 1: Write the failing tests**

Add `synthesize_negative_composite` and `FOREGROUND_THRESHOLD` to the existing `from hieroglyph.segmentation.synthesize import (...)` block at the top of `tests/test_synthesize.py`, and append these functions to the end of the file:

```python
def test_synthesize_negative_composite_has_no_boxes():
    composite = synthesize_negative_composite(canvas_size=(100, 100), rng=random.Random(0))

    assert composite.boxes == []
    assert composite.image.shape == (100, 100)


def test_synthesize_negative_composite_is_deterministic_given_same_seed():
    a = synthesize_negative_composite(canvas_size=(100, 100), rng=random.Random(7))
    b = synthesize_negative_composite(canvas_size=(100, 100), rng=random.Random(7))

    assert np.array_equal(a.image, b.image)


def test_synthesize_negative_composite_draws_visible_distractor_content():
    composite = synthesize_negative_composite(canvas_size=(200, 200), rng=random.Random(1))

    dark_pixel_count = int((composite.image < FOREGROUND_THRESHOLD).sum())
    assert dark_pixel_count > 50


def test_generate_mixed_dataset_negative_fraction_one_writes_only_empty_labels(tmp_path: Path):
    crop_paths = _make_raw_crop_paths(tmp_path)

    output_dir = tmp_path / "synthetic"
    generate_mixed_dataset(crop_paths, output_dir, num_composites=5, negative_fraction=1.0, seed=3)

    labels = sorted((output_dir / "labels").glob("*.txt"))
    assert len(labels) == 5
    for label_path in labels:
        assert label_path.read_text(encoding="utf-8").strip() == ""


def test_generate_mixed_dataset_default_negative_fraction_produces_some_negatives(tmp_path: Path):
    crop_paths = _make_raw_crop_paths(tmp_path)

    output_dir = tmp_path / "synthetic"
    generate_mixed_dataset(crop_paths, output_dir, num_composites=40, seed=4)

    labels = sorted((output_dir / "labels").glob("*.txt"))
    empty_count = sum(1 for p in labels if not p.read_text(encoding="utf-8").strip())
    assert 0 < empty_count < len(labels)  # a real mix, not all-or-nothing
```

(`_make_raw_crop_paths` is an existing helper already defined in this file, above `test_generate_mixed_dataset_writes_matching_images_and_labels`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_synthesize.py -v -k "negative"`
Expected: FAIL with `ImportError: cannot import name 'synthesize_negative_composite'` (or `TypeError: generate_mixed_dataset() got an unexpected keyword argument 'negative_fraction'` for the two `generate_mixed_dataset` tests once the import is fixed manually — either way, fails before Step 3).

- [ ] **Step 3: Write the implementation**

In `src/hieroglyph/segmentation/synthesize.py`, add this function after `synthesize_column_composite` and before `box_to_yolo_line`:

```python
def synthesize_negative_composite(
    canvas_size: tuple[int, int] = (640, 640),
    background_value: int = 210,
    background_noise_std: float = 10.0,
    n_shapes_range: tuple[int, int] = (3, 8),
    ink_value_range: tuple[int, int] = (40, 140),
    thickness_range: tuple[int, int] = (2, 6),
    rng: random.Random | None = None,
) -> SynthesizedComposite:
    """Build a canvas with procedural "distractor" line-art -- no glyph
    crops, no boxes -- so the detector sees some "this is dark, structured
    ink, but not a sign" signal during training. Every composite
    synthesize_composite/synthesize_column_composite ever produce has
    glyph crops on it; the detector has never been shown a hard negative.

    Draws a random number of thick curved strokes (short random polylines)
    and ellipses (filled or outlined) at random position/size/thickness,
    in the same dark-ink tone range real glyph strokes use -- approximating
    the general character of figure/illustration line-art (continuous
    curved marks, larger connected shapes) as opposed to a glyph's small,
    compact, self-contained ink blob. This is a proxy, not real
    illustration content -- see
    docs/superpowers/specs/2026-09-18-real-data-finetune-and-hard-negatives-design.md
    for why a procedural approximation was chosen over sourcing a real
    illustration dataset.
    """
    rng = rng or random.Random()
    canvas_w, canvas_h = canvas_size

    rng_np = np.random.default_rng(rng.getrandbits(32))
    noise = rng_np.normal(loc=0.0, scale=background_noise_std, size=(canvas_h, canvas_w))
    canvas = np.clip(background_value + noise, 0, 255).astype(np.uint8)

    n_shapes = rng.randint(*n_shapes_range)
    for _ in range(n_shapes):
        ink_value = rng.randint(*ink_value_range)
        thickness = rng.randint(*thickness_range)

        if rng.random() < 0.5:
            n_points = rng.randint(3, 6)
            points = [
                (rng.randint(0, canvas_w - 1), rng.randint(0, canvas_h - 1)) for _ in range(n_points)
            ]
            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=False, color=int(ink_value), thickness=thickness)
        else:
            center = (rng.randint(0, canvas_w - 1), rng.randint(0, canvas_h - 1))
            axes = (rng.randint(10, max(11, canvas_w // 6)), rng.randint(10, max(11, canvas_h // 6)))
            angle = rng.uniform(0, 360)
            filled = rng.random() < 0.5
            cv2.ellipse(
                canvas,
                center,
                axes,
                angle,
                0,
                360,
                color=int(ink_value),
                thickness=-1 if filled else thickness,
            )

    return SynthesizedComposite(image=canvas, boxes=[])
```

Then find `generate_mixed_dataset`'s current body and signature:

```python
def generate_mixed_dataset(
    crop_paths: list[Path],
    output_dir: Path,
    num_composites: int,
    dense_fraction: float = 0.5,
    scatter_canvas_size: tuple[int, int] = (640, 640),
    column_canvas_sizes: list[tuple[int, int]] = [(200, 640), (640, 200), (300, 640)],
    seed: int = 0,
) -> None:
```

Replace the signature and its docstring's opening paragraph with:

```python
def generate_mixed_dataset(
    crop_paths: list[Path],
    output_dir: Path,
    num_composites: int,
    dense_fraction: float = 0.5,
    negative_fraction: float = 0.15,
    scatter_canvas_size: tuple[int, int] = (640, 640),
    column_canvas_sizes: list[tuple[int, int]] = [(200, 640), (640, 200), (300, 640)],
    seed: int = 0,
) -> None:
    """Generate `num_composites` synthetic training images + YOLO labels,
    mixing dense column-packed composites, sparse scatter composites, and
    hard-negative composites (no glyphs at all) -- a real photo can be a
    densely packed papyrus column, a sparser wall-carving-style
    inscription, or contain non-glyph illustration content the detector
    must learn to reject, so training data that's only glyphs-on-every-image
    leaves the detector with no negative signal. `generate_dataset`
    (scatter-only) stays the entry point for callers that don't need the
    mix; this is purely additive alongside it.

    A `negative_fraction` of composites (checked first, before the
    dense/scatter split) are hard-negative composites via
    `synthesize_negative_composite` -- procedural distractor line-art, no
    glyph crops, empty labels. See
    docs/superpowers/specs/2026-09-18-real-data-finetune-and-hard-negatives-design.md.
```

(Keep the rest of the existing docstring — the "Per composite index...", "The two paths deliberately use..." and "Writes output_dir/images/..." paragraphs — unchanged below what you just replaced; only the opening paragraph and the signature change.)

Then find the function body's loop:

```python
    rng = random.Random(seed)
    for i in range(num_composites):
        # The branch (and, for the dense branch, the canvas) is chosen first
        # because the crop count depends on it -- see the docstring.
        is_dense = rng.random() < dense_fraction
        if is_dense:
            canvas_size = rng.choice(column_canvas_sizes)
            budget = dense_crop_budget(canvas_size)
            n_crops = rng.randint(budget, 2 * budget)
        else:
            n_crops = rng.randint(3, 10)  # same default range as generate_dataset's crops_per_composite

        chosen_paths = rng.choices(crop_paths, k=n_crops)
        crops = []
        for p in chosen_paths:
            crop = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if crop is None:
                raise ValueError(f"Could not read crop image (missing, corrupt, or not an image): {p}")
            crops.append(crop)

        if is_dense:
            composite = synthesize_column_composite(crops, canvas_size=canvas_size, rng=rng)
        else:
            composite = synthesize_composite(crops, canvas_size=scatter_canvas_size, rng=rng)

        write_composite(
```

Replace with:

```python
    rng = random.Random(seed)
    for i in range(num_composites):
        # The branch (and, for the dense branch, the canvas) is chosen first
        # because the crop count depends on it -- see the docstring. One
        # roll decides negative vs. dense vs. scatter (additive thresholds),
        # not a separate roll per check.
        roll = rng.random()
        if roll < negative_fraction:
            composite = synthesize_negative_composite(canvas_size=scatter_canvas_size, rng=rng)
        else:
            is_dense = roll < negative_fraction + dense_fraction
            if is_dense:
                canvas_size = rng.choice(column_canvas_sizes)
                budget = dense_crop_budget(canvas_size)
                n_crops = rng.randint(budget, 2 * budget)
            else:
                n_crops = rng.randint(3, 10)  # same default range as generate_dataset's crops_per_composite

            chosen_paths = rng.choices(crop_paths, k=n_crops)
            crops = []
            for p in chosen_paths:
                crop = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if crop is None:
                    raise ValueError(f"Could not read crop image (missing, corrupt, or not an image): {p}")
                crops.append(crop)

            if is_dense:
                composite = synthesize_column_composite(crops, canvas_size=canvas_size, rng=rng)
            else:
                composite = synthesize_composite(crops, canvas_size=scatter_canvas_size, rng=rng)

        write_composite(
```

(The rest of the function — the `write_composite(...)` call and its arguments — is unchanged; only its indentation level and what precedes it changed.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_synthesize.py -v`
Expected: PASS (all tests in the file, old and new — 26 total)

- [ ] **Step 5: Commit**

```bash
git add src/hieroglyph/segmentation/synthesize.py tests/test_synthesize.py
git commit -m "feat: add hard-negative composites to teach the detector what isn't a sign"
```

---

## Task 2: Real-photo label remapping (`real_data.py`)

**Depends on:** none (independent of Task 1; touches a different part of `synthesize.py`).

**Files:**
- Modify: `src/hieroglyph/segmentation/synthesize.py` (small refactor: extract a shared polygon-line helper)
- Create: `src/hieroglyph/segmentation/real_data.py`
- Create: `tests/test_real_data.py`

**Interfaces:**
- Consumes: `hieroglyph.segmentation.synthesize._corner_polygon_line` (new, this task's refactor step).
- Produces (used by Task 4's notebook):
  `prepare_real_finetune_split(images_dir: Path, labels_dir: Path, output_dir: Path) -> int`

- [ ] **Step 1: Refactor — extract the shared polygon-line helper**

In `src/hieroglyph/segmentation/synthesize.py`, find:

```python
def box_to_yolo_line(box: BoundingBox, canvas_w: int, canvas_h: int) -> str:
    """One YOLO segmentation-format label line: class index followed by
    the box's own four corners as a degenerate polygon (top-left,
    top-right, bottom-right, bottom-left), each coordinate normalized to
    [0, 1].

    The fine-tuned checkpoint is a YOLOv8-*segmentation* model (see
    reports/2026-09-15-segmentation-detector-sourcing.md) -- ultralytics'
    segment-task dataset loader requires segment labels, not detection's
    `class cx cy w h`, and raises ValueError otherwise. A box's own four
    corners, traced in order, is exactly the box as a polygon: no
    information is lost, and ultralytics converts it back to the same
    bounding box internally (`segments2boxes`).
    """
    x1 = box.x / canvas_w
    y1 = box.y / canvas_h
    x2 = box.x2 / canvas_w
    y2 = box.y2 / canvas_h
    return f"0 {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"
```

Replace with:

```python
def _corner_polygon_line(cls: int, x1: float, y1: float, x2: float, y2: float) -> str:
    """One YOLO *segmentation*-format label line for an axis-aligned box,
    given already-normalized [0, 1] corner coordinates: class index
    followed by the box's own four corners as a degenerate polygon
    (top-left, top-right, bottom-right, bottom-left).

    Shared by box_to_yolo_line (this module, pixel-space BoundingBox
    input) and hieroglyph.segmentation.real_data.prepare_real_finetune_split
    (already-normalized real-photo boxes) so both stay identical on the
    exact corner order ultralytics' segment-task loader expects -- see
    box_to_yolo_line's docstring for why this format is needed at all.
    """
    return f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"


def box_to_yolo_line(box: BoundingBox, canvas_w: int, canvas_h: int) -> str:
    """One YOLO segmentation-format label line for a pixel-space
    BoundingBox -- see _corner_polygon_line for the format itself and why
    it's needed.
    """
    x1 = box.x / canvas_w
    y1 = box.y / canvas_h
    x2 = box.x2 / canvas_w
    y2 = box.y2 / canvas_h
    return _corner_polygon_line(0, x1, y1, x2, y2)
```

- [ ] **Step 2: Run existing tests to verify the refactor didn't change behavior**

Run: `.venv/Scripts/python -m pytest tests/test_synthesize.py -v -k "box_to_yolo_line"`
Expected: PASS (`test_box_to_yolo_line_normalizes_to_unit_range`) — output format is unchanged, only how it's built internally.

- [ ] **Step 3: Write the failing tests**

Create `tests/test_real_data.py`:

```python
from pathlib import Path

import cv2
import numpy as np
import pytest

from hieroglyph.segmentation.real_data import _remap_label_line, prepare_real_finetune_split


def _tiny_image(size=20):
    return np.full((size, size, 3), 200, dtype=np.uint8)


def test_remap_label_line_converts_center_box_to_single_class_polygon():
    line = "4 0.5 0.5 0.2 0.4"

    parts = _remap_label_line(line).split()

    assert parts[0] == "0"
    x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[6])
    assert abs(x1 - 0.4) < 1e-6
    assert abs(y1 - 0.3) < 1e-6
    assert abs(x2 - 0.6) < 1e-6
    assert abs(y2 - 0.7) < 1e-6


def test_remap_label_line_clips_out_of_bounds_box():
    line = "7 0.05 0.05 0.2 0.2"  # centered near the corner, would extend past [0,1]

    parts = _remap_label_line(line).split()

    coords = [float(v) for v in parts[1:]]
    assert all(0.0 <= c <= 1.0 for c in coords)


def test_prepare_real_finetune_split_copies_and_remaps_matching_pairs(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    for name, line in [("a", "0 0.5 0.5 0.2 0.2"), ("b", "3 0.4 0.6 0.1 0.3")]:
        cv2.imwrite(str(images_dir / f"{name}.jpg"), _tiny_image())
        (labels_dir / f"{name}.txt").write_text(line, encoding="utf-8")

    output_dir = tmp_path / "output"
    count = prepare_real_finetune_split(images_dir, labels_dir, output_dir)

    assert count == 2
    assert sorted(p.name for p in (output_dir / "images").glob("*.jpg")) == ["a.jpg", "b.jpg"]
    for name in ("a", "b"):
        label_text = (output_dir / "labels" / f"{name}.txt").read_text(encoding="utf-8")
        assert label_text.startswith("0 ")


def test_prepare_real_finetune_split_handles_multi_line_label_file(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    cv2.imwrite(str(images_dir / "multi.jpg"), _tiny_image())
    (labels_dir / "multi.txt").write_text("0 0.2 0.2 0.1 0.1\n5 0.7 0.7 0.1 0.1", encoding="utf-8")

    prepare_real_finetune_split(images_dir, labels_dir, tmp_path / "output")

    lines = (tmp_path / "output" / "labels" / "multi.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(line.startswith("0 ") for line in lines)


def test_prepare_real_finetune_split_raises_on_image_with_no_label(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    cv2.imwrite(str(images_dir / "orphan.jpg"), _tiny_image())

    with pytest.raises(ValueError, match="orphan"):
        prepare_real_finetune_split(images_dir, labels_dir, tmp_path / "output")


def test_prepare_real_finetune_split_raises_on_label_with_no_image(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    (labels_dir / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")

    with pytest.raises(ValueError, match="orphan"):
        prepare_real_finetune_split(images_dir, labels_dir, tmp_path / "output")
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_real_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hieroglyph.segmentation.real_data'`

- [ ] **Step 5: Write the implementation**

Create `src/hieroglyph/segmentation/real_data.py`:

```python
"""Adapt the real, human-annotated Roboflow "egyptian-hieroglyphs" photos
(data/real_eval_photos/{train,valid}/, see
reports/2026-09-18-real-finetune-data-sourcing.md for provenance) into the
single-class YOLOv8-segmentation label format our detector needs, so they
can be used for a short real-data fine-tune stage after synthetic
pretraining (notebooks/04_train_segmenter.ipynb) -- see
docs/superpowers/specs/2026-09-18-real-data-finetune-and-hard-negatives-design.md.

The source labels are standard YOLO-*detection* format (`class cx cy w h`,
19 Gardiner-code classes, one file per image) -- our detector cares only
about "is there a sign here", not which one (that's the separate ResNet18
classifier's job), so every box gets remapped to class 0 and re-encoded as
a degenerate polygon (segmentation task's label format, not detection's).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from hieroglyph.segmentation.synthesize import _corner_polygon_line


def _remap_label_line(line: str) -> str:
    """Convert one `class cx cy w h` (already-normalized) detection line
    into one single-class polygon line, per _corner_polygon_line."""
    parts = line.split()
    cx, cy, w, h = (float(v) for v in parts[1:5])
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    # Clip to [0, 1] -- some exports include boxes that run a hair past the
    # image edge; ultralytics' loader rejects out-of-range coordinates.
    x1, x2 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))
    y1, y2 = max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))
    return _corner_polygon_line(0, x1, y1, x2, y2)


def prepare_real_finetune_split(images_dir: Path, labels_dir: Path, output_dir: Path) -> int:
    """Copy every image in images_dir that has a matching (same filename
    stem) label file in labels_dir to output_dir/images/, and write its
    remapped, single-class label to output_dir/labels/. Raises ValueError
    if an image has no matching label file or vice versa -- a mismatch
    means the source export is corrupt/incomplete, not something to paper
    over silently. Returns the number of pairs processed.
    """
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    label_paths = sorted(labels_dir.glob("*.txt"))
    image_stems = {p.stem for p in image_paths}
    label_stems = {p.stem for p in label_paths}

    if image_stems != label_stems:
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)
        raise ValueError(
            "Mismatched image/label pairs -- "
            f"images with no label: {missing_labels}, labels with no image: {missing_images}"
        )

    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        shutil.copy2(image_path, output_images_dir / image_path.name)

        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        remapped = [_remap_label_line(line) for line in lines]
        (output_labels_dir / f"{image_path.stem}.txt").write_text("\n".join(remapped), encoding="utf-8")

    return len(image_paths)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_real_data.py -v`
Expected: PASS (7 tests)

- [ ] **Step 7: Commit**

```bash
git add src/hieroglyph/segmentation/synthesize.py src/hieroglyph/segmentation/real_data.py tests/test_real_data.py
git commit -m "feat: add real-photo label remapping for a real-data fine-tune stage"
```

---

## Task 3: Commit the real train/valid split + sourcing report

**Files:**
- Modify: `.gitignore`
- Add: `data/real_eval_photos/train/images/*.jpg` and `data/real_eval_photos/train/labels/*.txt` (83 pairs, already present locally, currently gitignored)
- Add: `data/real_eval_photos/valid/images/*.jpg` and `data/real_eval_photos/valid/labels/*.txt` (7 pairs, already present locally, currently gitignored)
- Create: `reports/2026-09-18-real-finetune-data-sourcing.md`

No code changes in this task — it commits data files already sitting in the working tree (downloaded per `notebooks/05_evaluate_segmenter.ipynb` section 3's existing manual step) plus a small `.gitignore` carve-out.

- [ ] **Step 1: Verify the data is present locally**

Run: `ls data/real_eval_photos/train/images | wc -l` and `ls data/real_eval_photos/valid/images | wc -l`
Expected: `83` and `7`. If either is missing or the counts don't match, STOP and report BLOCKED — do not re-download or fabricate data; this task only commits what Phase 12/dense-text-detection work already downloaded.

- [ ] **Step 2: Carve out train/ and valid/ from the blanket .gitignore**

Find, in `.gitignore`:

```
data/real_eval_photos/
```

Replace with:

```
data/real_eval_photos/*
!data/real_eval_photos/train/
!data/real_eval_photos/train/**
!data/real_eval_photos/valid/
!data/real_eval_photos/valid/**
```

(This un-ignores only `train/` and `valid/` — `test/` (inconsistent leftover structure, see spec) and `papyrus/` (re-downloadable, no need to commit) stay ignored.)

- [ ] **Step 3: Verify the carve-out works as intended**

Run: `git status --short data/real_eval_photos/`
Expected: shows `train/images/*.jpg`, `train/labels/*.txt`, `valid/images/*.jpg`, `valid/labels/*.txt` as untracked (`??`), and does NOT show anything under `data/real_eval_photos/test/` or `data/real_eval_photos/papyrus/`.

- [ ] **Step 4: Write the sourcing report**

Create `reports/2026-09-18-real-finetune-data-sourcing.md`:

```markdown
# Real-photo fine-tune data: what's committed and why

**Committed:** `data/real_eval_photos/train/` (83 image+label pairs) and
`data/real_eval_photos/valid/` (7 pairs), from the Roboflow
"egyptian-hieroglyphs" dataset
(https://universe.roboflow.com/custom-yolov8-ljpde/egyptian-hieroglyphs),
already referenced in
`reports/2026-09-15-segmentation-detector-sourcing.md` and downloaded per
`notebooks/05_evaluate_segmenter.ipynb` section 3's manual step.

## Why committed now (reversing Phase 12's original decision)
Phase 12's design spec
(`docs/superpowers/specs/2026-09-15-glyph-segmentation-detector-design.md`)
judged this set "too small to train on" and used it only for a qualitative
box-count eyeball check. That's the right call for training *from
scratch*, but not for a short **fine-tuning** stage after a large
synthetic pretrain -- see
`docs/superpowers/specs/2026-09-18-real-data-finetune-and-hard-negatives-design.md`
for the reasoning. At ~2.3MB total, committing these two splits is small
enough to keep in the repo, which also means `notebooks/04_train_segmenter.ipynb`'s
existing `git clone` step on Colab picks them up automatically -- no new
manual upload cell needed.

## What's NOT committed
- `data/real_eval_photos/test/` -- an inconsistent leftover from the
  original export (loose label files directly under `test/` plus a
  near-empty `images/`/`labels/` subdirectory pair, likely a partial
  re-export). Not used by anything; left as a future cleanup.
- `data/real_eval_photos/papyrus/` -- the Papyrus of Ani qualitative-eval
  photo, re-downloadable via `scripts/download_papyrus_eval_photo.py`; no
  need to duplicate it into git history.

## License
CC BY 4.0, per the Roboflow dataset page linked above --
attribution: "egyptian-hieroglyphs Dataset" by custom-yolov8-ljpde, made
available on Roboflow Universe under CC BY 4.0.
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore reports/2026-09-18-real-finetune-data-sourcing.md \
  data/real_eval_photos/train data/real_eval_photos/valid
git commit -m "feat: commit real annotated train/valid split for fine-tuning"
```

---

## Task 4: Wire the real-data fine-tune stage into the training notebook

**Depends on:** Task 2 (`prepare_real_finetune_split`), Task 3 (committed `data/real_eval_photos/{train,valid}`).

**Files:**
- Modify: `notebooks/04_train_segmenter.ipynb`

No pytest coverage applies to notebook cells (matches existing repo convention).

- [ ] **Step 1: Patch the notebook**

Run this one-off Python snippet:

```python
import json
from pathlib import Path

NB_PATH = Path("notebooks/04_train_segmenter.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

OLD_MARKDOWN_DOWNLOAD = """## 8. Download the fine-tuned checkpoint

The checkpoint is already saved in Google Drive (`DRIVE_RUNS_DIR`, mounted
in step 2) the moment training finishes, so it already survived this
session -- this cell just also pulls a copy to your local machine for
convenience. This is the file that goes into your local `models/yolo_seg.pt`
for `notebooks/05_evaluate_segmenter.ipynb` and the Streamlit demo
(Phase 12). **Rename it to `yolo_seg.pt` locally** after downloading --
both of those expect that exact filename."""

OLD_CODE_DOWNLOAD = """from google.colab import files

files.download(f"{DRIVE_RUNS_DIR}/yolo_seg_finetune/weights/best.pt")"""

NEW_MARKDOWN_REAL_FINETUNE = """## 8. Fine-tune further on real annotated photos

Synthetic composites (steps 5-7) teach volume and glyph shape, but no
synthesizer fully reproduces real ink texture, lighting, or surface noise
-- this stage continues training the *same* `model` object (its current
weights, not a fresh checkpoint) for a short run on
`data/real_eval_photos/{train,valid}` (83 + 7 real, human-annotated
photos, committed to the repo -- see
`reports/2026-09-18-real-finetune-data-sourcing.md`), remapped from their
original 19-Gardiner-class detection labels down to our single
"hieroglyph" class. See
`docs/superpowers/specs/2026-09-18-real-data-finetune-and-hard-negatives-design.md`
for why this small-but-real stage is expected to help where more
synthetic data alone can't."""

NEW_CODE_REAL_FINETUNE = """from hieroglyph.segmentation.real_data import prepare_real_finetune_split

REAL_DIR = Path("data/real_eval_photos")
REAL_FINETUNE_DIR = Path("data/real_finetune")

train_count = prepare_real_finetune_split(
    REAL_DIR / "train" / "images", REAL_DIR / "train" / "labels", REAL_FINETUNE_DIR / "train"
)
valid_count = prepare_real_finetune_split(
    REAL_DIR / "valid" / "images", REAL_DIR / "valid" / "labels", REAL_FINETUNE_DIR / "valid"
)
print(f"{train_count} real train pairs, {valid_count} real valid pairs -> {REAL_FINETUNE_DIR}")"""

NEW_CODE_REAL_YAML = """real_data_yaml = {
    "path": str(REAL_FINETUNE_DIR.resolve()),
    "train": "train/images",
    "val": "valid/images",
    "names": {0: "hieroglyph"},
    "nc": 1,
}
(REAL_FINETUNE_DIR / "data.yaml").write_text(yaml.dump(real_data_yaml), encoding="utf-8")
print((REAL_FINETUNE_DIR / "data.yaml").read_text())"""

NEW_CODE_REAL_TRAIN = """# Continues from the synthetic-pretrain weights already in `model` (calling
# .train() again on the same YOLO instance resumes from its current
# in-memory weights) -- a short, low-volume fine-tune, not a from-scratch
# run on 90 images.
real_train_results = model.train(
    data=str(REAL_FINETUNE_DIR / "data.yaml"),
    epochs=15,
    imgsz=640,
    device=0,
    project=DRIVE_RUNS_DIR,
    name="yolo_seg_finetune_real",
)"""

NEW_MARKDOWN_DOWNLOAD = """## 9. Download the fine-tuned checkpoint

The checkpoint is already saved in Google Drive (`DRIVE_RUNS_DIR`, mounted
in step 2) the moment training finishes, so it already survived this
session -- this cell just also pulls a copy to your local machine for
convenience. This is the *real-data-fine-tuned* run's weights (step 8),
the one that should actually be used -- not step 7's synthetic-only
checkpoint. This is the file that goes into your local `models/yolo_seg.pt`
for `notebooks/05_evaluate_segmenter.ipynb` and the Streamlit demo.
**Rename it to `yolo_seg.pt` locally** after downloading -- both of those
expect that exact filename."""

NEW_CODE_DOWNLOAD = """from google.colab import files

files.download(f"{DRIVE_RUNS_DIR}/yolo_seg_finetune_real/weights/best.pt")"""

replaced_markdown = replaced_code = False
new_cells = []
for cell in nb["cells"]:
    source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if cell["cell_type"] == "markdown" and source == OLD_MARKDOWN_DOWNLOAD:
        new_cells.append({"cell_type": "markdown", "metadata": {}, "source": NEW_MARKDOWN_REAL_FINETUNE})
        for src in (NEW_CODE_REAL_FINETUNE, NEW_CODE_REAL_YAML, NEW_CODE_REAL_TRAIN):
            new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})
        new_cells.append({"cell_type": "markdown", "metadata": {}, "source": NEW_MARKDOWN_DOWNLOAD})
        replaced_markdown = True
    elif cell["cell_type"] == "code" and source == OLD_CODE_DOWNLOAD:
        new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": NEW_CODE_DOWNLOAD})
        replaced_code = True
    else:
        new_cells.append(cell)

assert replaced_markdown, "download-section markdown cell to replace not found"
assert replaced_code, "download-section code cell to replace not found"

nb["cells"] = new_cells
NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("patched notebooks/04_train_segmenter.ipynb")
```

- [ ] **Step 2: Verify it's valid JSON**

Run: `.venv/Scripts/python -c "import json; json.load(open('notebooks/04_train_segmenter.ipynb', encoding='utf-8')); print('valid json')"`
Expected: prints `valid json`

- [ ] **Step 3: Verify the real-data pipeline actually runs locally (without Colab/GPU)**

Run this one-off snippet to confirm `prepare_real_finetune_split` works end-to-end against the real committed data (this does NOT run `model.train` — that step needs Colab/GPU and is a manual step per the spec):

```python
from pathlib import Path
from hieroglyph.segmentation.real_data import prepare_real_finetune_split

REAL_DIR = Path("data/real_eval_photos")
train_count = prepare_real_finetune_split(
    REAL_DIR / "train" / "images", REAL_DIR / "train" / "labels", Path("data/real_finetune/train")
)
valid_count = prepare_real_finetune_split(
    REAL_DIR / "valid" / "images", REAL_DIR / "valid" / "labels", Path("data/real_finetune/valid")
)
print(f"train_count={train_count} valid_count={valid_count}")
```

Expected: `train_count=83 valid_count=7`, no errors. Delete the resulting `data/real_finetune/` directory afterward (it's a local sanity check, not a committed artifact — already covered by `data/synthetic_composites/`'s sibling gitignore pattern if you check, but verify with `git status` that nothing under `data/real_finetune/` got staged).

- [ ] **Step 4: Commit**

```bash
git add notebooks/04_train_segmenter.ipynb
git commit -m "feat: add real-data fine-tune stage to the training notebook"
```

---

## Task 5: README updates

**Depends on:** Tasks 1-4 (documents what they built).

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Extend the segmentation limitation bullet**

Find (this exact paragraph, added by the dense-text-detection merge):

```
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

Replace with:

```
- Dense, tightly-packed real text (e.g. an actual papyrus column, as
  opposed to a sparser wall-carving-style photo) is a known hard case: the
  original training data was sparse-scatter only. `TiledYoloSegmenter`
  (`src/hieroglyph/segmentation/tiling.py`) fixes the inference-time half
  of this (no destructive whole-image downscale) independently of
  retraining; `generate_mixed_dataset`
  (`src/hieroglyph/segmentation/synthesize.py`) fixes the training-data
  half — see `docs/superpowers/specs/2026-09-17-dense-text-detection-design.md`
  and `notebooks/05_evaluate_segmenter.ipynb` Section 5 for the real-papyrus
  qualitative check.
- A retrained checkpoint on the mixed dense/sparse dataset still showed
  false positives on non-glyph illustration content (figures, clothing
  folds) in real photos — the model had never seen a negative example.
  Two further, independent levers now exist:
  `synthesize_negative_composite` (procedural distractor training
  composites with empty labels) and a real-data fine-tune stage using 90
  committed, human-annotated real photos
  (`data/real_eval_photos/{train,valid}/`) — see
  `docs/superpowers/specs/2026-09-18-real-data-finetune-and-hard-negatives-design.md`.
  As with the dense/sparse fix, `models/yolo_seg.pt` hasn't been retrained
  on this yet.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document hard-negative composites and the real-data fine-tune stage"
```

---

## Suggested execution order

- **Wave A (parallel, no interdependencies):** Task 1, Task 3
- **Wave B (sequential, after Task 1 since both touch synthesize.py):** Task 2
- **Wave C (after Task 2 and Task 3):** Task 4
- **Wave D (sequential, after everything else):** Task 5

## Not automated by this plan (manual follow-up)

- Actually re-running `notebooks/04_train_segmenter.ipynb` on Colab (GPU) — now a two-stage run (synthetic pretrain + real fine-tune) — and downloading the retrained `models/yolo_seg.pt`.
- Re-running `notebooks/05_evaluate_segmenter.ipynb` locally against that retrained checkpoint to record new mAP/qualitative numbers, including the real-papyrus Section 5 check, in the README.
- Merging `worktree-real-data-and-negatives` back into `master` — via `superpowers:finishing-a-development-branch`, after the above.
