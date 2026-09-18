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
