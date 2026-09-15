import random
from pathlib import Path

import cv2
import numpy as np
import pytest

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

    parts = line.split()
    assert len(parts) == 9  # class + 4 corners x 2 coords
    assert parts[0] == "0"
    values = [float(v) for v in parts[1:]]
    x1, y1, x2, y1b, x2b, y2, x1b, y2b = values
    # top-left
    assert abs(x1 - 0.10) < 1e-6  # 10 / 100
    assert abs(y1 - 0.10) < 1e-6  # 20 / 200
    # top-right
    assert abs(x2 - 0.40) < 1e-6  # (10+30) / 100
    assert abs(y1b - 0.10) < 1e-6
    # bottom-right
    assert abs(x2b - 0.40) < 1e-6
    assert abs(y2 - 0.30) < 1e-6  # (20+40) / 200
    # bottom-left
    assert abs(x1b - 0.10) < 1e-6
    assert abs(y2b - 0.30) < 1e-6


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
            assert len(parts) == 9


def test_generate_dataset_labels_match_placed_box_positions(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    cls_dir = raw_dir / "A1"
    cls_dir.mkdir(parents=True)
    cv2.imwrite(str(cls_dir / "0.png"), _dark_square_crop())

    crop_paths = list_crop_paths(raw_dir)
    output_dir = tmp_path / "synthetic"
    generate_dataset(crop_paths, output_dir, num_composites=1, crops_per_composite=(1, 1), canvas_size=(100, 100), seed=7)

    label_path = output_dir / "labels" / "composite_0000.txt"
    lines = [l for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1

    parts = lines[0].split()
    assert len(parts) == 9
    x1, y1, x2, y1b, x2b, y2, x1b, y2b = [float(v) for v in parts[1:]]
    # A rectangle's corners: top-left and bottom-left share x; top-left and top-right share y
    assert abs(x1 - x1b) < 1e-6
    assert abs(x2 - x2b) < 1e-6
    assert abs(y1 - y1b) < 1e-6
    assert abs(y2 - y2b) < 1e-6
    assert x1 < x2
    assert y1 < y2


def test_generate_dataset_raises_clear_error_on_unreadable_crop(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    cls_dir = raw_dir / "A1"
    cls_dir.mkdir(parents=True)
    (cls_dir / "not_an_image.png").write_bytes(b"this is not valid png data")

    crop_paths = list_crop_paths(raw_dir)
    output_dir = tmp_path / "synthetic"

    with pytest.raises(ValueError, match="Could not read crop image"):
        generate_dataset(crop_paths, output_dir, num_composites=1, canvas_size=(100, 100), seed=1)
