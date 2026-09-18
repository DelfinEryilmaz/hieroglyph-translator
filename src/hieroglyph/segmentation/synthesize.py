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

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hieroglyph.segmentation.types import BoundingBox

FOREGROUND_THRESHOLD = 180  # pixel darker than this = glyph ink, not crop background

# Defaults for the dense/column layout, named so `generate_mixed_dataset`'s
# estimate of "how many crops fill a column" can't drift away from the
# values `synthesize_column_composite` actually lays out with.
COLUMN_SCALE_RANGE = (0.2, 0.6)
COLUMN_GLYPH_GAP_RANGE = (-0.15, 0.08)

# Median height of the classifier's per-glyph crops (data/raw/*/*.png,
# measured over 311 sampled files: median 75x50 px, mean 75x50). Only used
# to *estimate* a crop budget -- oversupplying is harmless, since
# synthesize_column_composite skips crops that no longer fit.
TYPICAL_CROP_HEIGHT_PX = 75


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


def synthesize_column_composite(
    crops: list[np.ndarray],
    canvas_size: tuple[int, int] = (200, 640),
    n_columns: int = 2,
    column_gap_range: tuple[float, float] = (0.0, 0.15),
    glyph_gap_range: tuple[float, float] = COLUMN_GLYPH_GAP_RANGE,
    background_value: int = 210,
    background_noise_std: float = 10.0,
    scale_range: tuple[float, float] = COLUMN_SCALE_RANGE,
    rng: random.Random | None = None,
) -> SynthesizedComposite:
    """Paste `crops` onto one synthetic canvas as `n_columns` tightly packed,
    top-to-bottom stacks -- modeling a dense papyrus column of signs packed
    edge-to-edge, unlike `synthesize_composite`'s sparse random scatter.

    Crops are round-robin assigned to columns (crop i goes to column
    i % n_columns), then each column is laid out by **deterministic
    cumulative stacking**, not reject-sampling: a glyph's y advances by
    `glyph_h * (1 + gap_fraction)`, with gap_fraction drawn fresh per glyph
    from `glyph_gap_range` -- a negative value is a deliberate small
    overlap, matching how real carved/painted columns crowd signs edge to
    edge. `paste_crop`'s dark-pixel-only paste already gives correct
    partial occlusion where two glyphs overlap, so it's reused unchanged.
    `scale_range` defaults much smaller than the scatter function's,
    since a real column's individual signs are small relative to the
    frame -- this is the scale-gap fix that makes composites look like a
    dense real photo instead of a handful of oversized glyphs.

    A crop that doesn't fit its column -- either its scaled width overflows
    the column's horizontal band, or its scaled height would run past the
    canvas at the column's current y -- is skipped individually rather than
    abandoning the rest of the column: each crop is independently
    rescaled, so a later, smaller crop can still legitimately fit in the
    remaining room even after an earlier, larger one didn't. An empty
    `crops` list or non-positive `n_columns` yields a composite with no
    boxes.
    """
    rng = rng or random.Random()
    canvas_w, canvas_h = canvas_size

    rng_np = np.random.default_rng(rng.getrandbits(32))
    noise = rng_np.normal(loc=0.0, scale=background_noise_std, size=(canvas_h, canvas_w))
    canvas = np.clip(background_value + noise, 0, 255).astype(np.uint8)

    boxes: list[BoundingBox] = []
    if not crops or n_columns <= 0:
        return SynthesizedComposite(image=canvas, boxes=boxes)

    column_width = canvas_w / n_columns

    for col in range(n_columns):
        column_crops = crops[col::n_columns]
        if not column_crops:
            continue

        gap_frac = rng.uniform(*column_gap_range)
        x = round(col * column_width + gap_frac * column_width)

        y = 0
        for crop in column_crops:
            scale = rng.uniform(*scale_range)
            h0, w0 = crop.shape
            if scale != 1.0:
                new_w, new_h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                crop = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)
            h, w = crop.shape

            if x < 0 or x + w > canvas_w:
                continue  # this scaled crop doesn't fit this column's x band, skip only it

            if y + h > canvas_h:
                continue  # this scaled crop doesn't fit at the current y, skip only it

            paste_crop(canvas, crop, x, y)
            boxes.append(BoundingBox(x=x, y=y, width=w, height=h))

            gap_fraction = rng.uniform(*glyph_gap_range)
            next_y = y + round(h * (1 + gap_fraction))
            y = next_y if next_y > y else y + 1  # guard against a pathological gap_fraction <= -1

    return SynthesizedComposite(image=canvas, boxes=boxes)


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


def dense_crop_budget(canvas_size: tuple[int, int], n_columns: int = 2) -> int:
    """Roughly how many crops `synthesize_column_composite` needs to fill
    `n_columns` columns of a `canvas_size` canvas top to bottom.

    A stacked glyph advances y by `scaled_height * (1 + gap_fraction)`, so
    with a typical crop of TYPICAL_CROP_HEIGHT_PX, the mean of
    COLUMN_SCALE_RANGE and the mean of COLUMN_GLYPH_GAP_RANGE, one column
    of height `canvas_h` holds about `canvas_h / advance` glyphs. Callers
    should sample *above* this estimate: a column whose scales happen to
    land at the small end of the range needs more crops than the mean
    predicts, and oversupplying costs nothing but a few unused reads --
    `synthesize_column_composite` skips crops that no longer fit rather
    than overflowing the canvas.
    """
    canvas_h = canvas_size[1]
    mean_scale = sum(COLUMN_SCALE_RANGE) / 2
    mean_gap = sum(COLUMN_GLYPH_GAP_RANGE) / 2
    advance = max(1.0, TYPICAL_CROP_HEIGHT_PX * mean_scale * (1 + mean_gap))
    per_column = math.ceil(canvas_h / advance)
    return max(1, per_column * max(1, n_columns))


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

    Per composite index, `rng.random() < dense_fraction` picks
    `synthesize_column_composite` with its `canvas_size` sampled uniformly
    from `column_canvas_sizes` (covers both portrait single-column and
    landscape multi-column real-world shapes); otherwise falls back to
    `synthesize_composite`'s scatter path at `scatter_canvas_size`.

    The two paths deliberately use **different crop counts**. The scatter
    path keeps `generate_dataset`'s 3-10, which is what sparse scatter
    means. The dense path instead sizes its sample to the chosen canvas via
    `dense_crop_budget` (see there) and samples between 1x and 2x that
    estimate, because column stacking fills top-down and simply stops when
    it runs out of crops: at 3-10 crops a 640-tall canvas ended up ~16%
    filled with nothing at all below the top sixth -- sparser than the
    scatter composites it is supposed to contrast with, and a systematic
    "no signs in the bottom 80%" artifact for a detector to latch onto
    instead of generalizing.

    Writes output_dir/images/composite_%04d.png and
    output_dir/labels/composite_%04d.txt, same naming convention as
    `generate_dataset`.
    """
    if not crop_paths:
        raise ValueError("crop_paths is empty")

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
            composite,
            image_path=output_dir / "images" / f"composite_{i:04d}.png",
            label_path=output_dir / "labels" / f"composite_{i:04d}.txt",
        )
