"""Synthesize labeled multi-glyph training images from single-glyph crops.

The classifier's dataset (data/raw/<gardiner_code>/*.png) is per-glyph
crops -- no real photo has ever been annotated with per-glyph bounding
boxes. To train a detector we manufacture that annotation for free: paste
several crops onto a synthetic background at random positions and scales,
and the pasted positions *are* the ground-truth boxes, no manual labeling
needed. Backgrounds carry mild Gaussian noise rather than being a perfectly
flat color, since no real photo is a flat color either.

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
    background_noise_std: float = 10.0,
    scale_range: tuple[float, float] = (0.7, 1.3),
    max_overlap_fraction: float = 0.15,
    max_attempts_per_crop: int = 20,
    rng: random.Random | None = None,
) -> SynthesizedComposite:
    """Paste `crops` (grayscale numpy arrays) onto one synthetic canvas at
    random, non-overlapping-ish positions, with per-crop scale jitter and
    a mildly noisy (not perfectly flat) background -- real photos are
    never a flat color, and fine-tuning only on flat backgrounds risks
    pulling the pretrained checkpoint's real-photo-exposed weights toward
    a trivial synthetic domain instead. Crops that can't be placed within
    max_attempts_per_crop tries (canvas too crowded) are skipped.
    """
    rng = rng or random.Random()
    canvas_w, canvas_h = canvas_size

    # Vectorized (numpy) per-pixel noise around background_value -- a pure
    # Python per-pixel loop over a 640x640 canvas would be far too slow
    # across thousands of composites. Seeded from `rng` (not numpy's own
    # global state) so the whole composite stays deterministic given one
    # `random.Random` seed.
    rng_np = np.random.default_rng(rng.getrandbits(32))
    noise = rng_np.normal(loc=0.0, scale=background_noise_std, size=(canvas_h, canvas_w))
    canvas = np.clip(background_value + noise, 0, 255).astype(np.uint8)

    boxes: list[BoundingBox] = []

    for crop in crops:
        scale = rng.uniform(*scale_range)
        h0, w0 = crop.shape
        if scale != 1.0:
            new_w, new_h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            crop = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)
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
        crops = []
        for p in chosen_paths:
            crop = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if crop is None:
                raise ValueError(f"Could not read crop image (missing, corrupt, or not an image): {p}")
            crops.append(crop)

        composite = synthesize_composite(crops, canvas_size=canvas_size, rng=rng)

        write_composite(
            composite,
            image_path=output_dir / "images" / f"composite_{i:04d}.png",
            label_path=output_dir / "labels" / f"composite_{i:04d}.txt",
        )
