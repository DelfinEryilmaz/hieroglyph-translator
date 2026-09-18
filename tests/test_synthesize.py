import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from hieroglyph.segmentation.synthesize import (
    box_to_yolo_line,
    generate_dataset,
    generate_mixed_dataset,
    list_crop_paths,
    paste_crop,
    split_crop_paths,
    synthesize_column_composite,
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


def test_synthesize_column_composite_boxes_overlap_in_y_when_gap_negative():
    crops = [_dark_square_crop(size=20), _dark_square_crop(size=20)]
    rng = random.Random(1)

    composite = synthesize_column_composite(
        crops,
        canvas_size=(100, 300),
        n_columns=1,
        glyph_gap_range=(-0.5, -0.5),
        scale_range=(1.0, 1.0),
        rng=rng,
    )

    assert len(composite.boxes) == 2
    first, second = composite.boxes
    # Deliberate negative gap: the second glyph's top starts above the
    # first glyph's bottom edge, i.e. they overlap vertically.
    assert second.y < first.y2


def test_synthesize_column_composite_places_all_crops_when_canvas_tall_enough():
    crops = [_dark_square_crop(size=10) for _ in range(8)]
    rng = random.Random(2)

    composite = synthesize_column_composite(crops, canvas_size=(100, 2000), n_columns=2, rng=rng)

    assert len(composite.boxes) == 8


def test_synthesize_column_composite_places_later_smaller_crop_after_earlier_oversized_one():
    # Each crop is independently rescaled, so an earlier crop failing to fit
    # at the column's current y must not abandon the rest of the column --
    # a later, smaller crop can still legitimately fit at that same y.
    small = _dark_square_crop(size=5)
    big = _dark_square_crop(size=30)
    crops = [small] * 9 + [big] + [small] * 5
    rng = random.Random(5)

    composite = synthesize_column_composite(
        crops,
        canvas_size=(100, 60),
        n_columns=1,
        scale_range=(1.0, 1.0),
        glyph_gap_range=(0.0, 0.0),
        rng=rng,
    )

    # 9 leading small crops (y: 0 -> 45), the 30px-tall crop doesn't fit at
    # y=45 (45+30=75 > 60) and is skipped alone, then 3 of the 5 trailing
    # small crops still fit in the remaining 15px (y: 45 -> 60).
    assert len(composite.boxes) == 12
    assert all(b.height == 5 for b in composite.boxes)  # the oversized crop never got placed


def test_synthesize_column_composite_is_deterministic_given_same_seed():
    crops = [_dark_square_crop(size=10) for _ in range(5)]

    result_a = synthesize_column_composite(crops, canvas_size=(150, 600), rng=random.Random(42))
    result_b = synthesize_column_composite(crops, canvas_size=(150, 600), rng=random.Random(42))

    assert [(b.x, b.y, b.width, b.height) for b in result_a.boxes] == [
        (b.x, b.y, b.width, b.height) for b in result_b.boxes
    ]
    assert np.array_equal(result_a.image, result_b.image)


def test_synthesize_column_composite_boxes_cluster_into_n_columns_distinct_x_bands():
    crops = [_dark_square_crop(size=10) for _ in range(6)]
    rng = random.Random(3)

    composite = synthesize_column_composite(
        crops, canvas_size=(300, 300), n_columns=3, scale_range=(1.0, 1.0), rng=rng
    )

    assert len(composite.boxes) == 6
    distinct_x = sorted(set(b.x for b in composite.boxes))
    assert len(distinct_x) == 3


def test_synthesize_column_composite_handles_empty_crops_list():
    composite = synthesize_column_composite([], canvas_size=(50, 50), rng=random.Random(0))

    assert composite.boxes == []
    assert composite.image.shape == (50, 50)


def test_synthesize_column_composite_skips_crop_taller_than_canvas():
    crops = [_dark_square_crop(size=50)]
    rng = random.Random(4)

    composite = synthesize_column_composite(
        crops, canvas_size=(60, 30), n_columns=1, scale_range=(1.0, 1.0), rng=rng
    )

    assert composite.boxes == []
    assert composite.image.shape == (30, 60)


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


def _make_raw_crop_paths(tmp_path: Path) -> list[Path]:
    raw_dir = tmp_path / "raw"
    for cls in ["A1", "B2"]:
        cls_dir = raw_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(3):
            cv2.imwrite(str(cls_dir / f"{i}.png"), _dark_square_crop())
    return list_crop_paths(raw_dir)


def test_generate_mixed_dataset_writes_matching_images_and_labels(tmp_path: Path):
    crop_paths = _make_raw_crop_paths(tmp_path)
    output_dir = tmp_path / "synthetic"

    generate_mixed_dataset(crop_paths, output_dir, num_composites=5, seed=1)

    images = sorted((output_dir / "images").glob("*.png"))
    labels = sorted((output_dir / "labels").glob("*.txt"))
    assert len(images) == 5
    assert len(labels) == 5


def test_generate_mixed_dataset_exercises_both_column_and_scatter_paths(tmp_path: Path):
    crop_paths = _make_raw_crop_paths(tmp_path)
    output_dir = tmp_path / "synthetic"
    scatter_canvas_size = (640, 640)
    column_canvas_sizes = [(200, 640), (640, 200), (300, 640)]

    generate_mixed_dataset(
        crop_paths,
        output_dir,
        num_composites=30,
        dense_fraction=0.5,
        scatter_canvas_size=scatter_canvas_size,
        column_canvas_sizes=column_canvas_sizes,
        seed=1,
    )

    images = sorted((output_dir / "images").glob("*.png"))
    shapes = {cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).shape for p in images}

    # cv2 image shape is (height, width); canvas sizes above are (width, height)
    column_shapes = {(h, w) for (w, h) in column_canvas_sizes}
    scatter_shape = (scatter_canvas_size[1], scatter_canvas_size[0])

    assert shapes & column_shapes, f"no column-shaped composite found among {shapes}"
    assert scatter_shape in shapes, f"no scatter-shaped composite found among {shapes}"


def test_generate_mixed_dataset_dense_composites_fill_the_canvas(tmp_path: Path):
    # Regression guard: the dense branch must supply enough crops to stack a
    # column all the way down. With a fixed 3-10 crops it filled only the top
    # ~16% of a 640-tall canvas -- sparser than the scatter composites it is
    # meant to contrast with, and a systematic "never any sign in the bottom
    # 80%" artifact for the detector to exploit.
    raw_dir = tmp_path / "raw"
    cls_dir = raw_dir / "A1"
    cls_dir.mkdir(parents=True)
    for i in range(3):
        # Realistically sized crop: data/raw's median is 75x50 px, which is
        # what dense_crop_budget's estimate is calibrated against.
        cv2.imwrite(str(cls_dir / f"{i}.png"), _dark_square_crop(size=75))
    crop_paths = list_crop_paths(raw_dir)
    output_dir = tmp_path / "synthetic"
    canvas_w, canvas_h = 200, 640

    generate_mixed_dataset(
        crop_paths,
        output_dir,
        num_composites=3,
        dense_fraction=1.0,  # every composite takes the dense/column branch
        column_canvas_sizes=[(canvas_w, canvas_h)],
        seed=1,
    )

    for label_path in sorted((output_dir / "labels").glob("*.txt")):
        lines = [l for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        # y2 of each box, normalized to [0, 1] -- see box_to_yolo_line's
        # corner order (x1 y1 x2 y1 x2 y2 x1 y2).
        lowest_bottom = max(float(line.split()[6]) for line in lines)
        assert lowest_bottom > 0.8, (
            f"{label_path.name}: lowest glyph reaches only {lowest_bottom:.0%} "
            "down the canvas -- the column is not densely filled"
        )


def test_generate_mixed_dataset_raises_clear_error_on_empty_crop_paths(tmp_path: Path):
    output_dir = tmp_path / "synthetic"

    with pytest.raises(ValueError, match="crop_paths is empty"):
        generate_mixed_dataset([], output_dir, num_composites=1, seed=1)


def test_generate_mixed_dataset_writes_nothing_when_num_composites_is_zero(tmp_path: Path):
    crop_paths = _make_raw_crop_paths(tmp_path)
    output_dir = tmp_path / "synthetic"

    generate_mixed_dataset(crop_paths, output_dir, num_composites=0, seed=1)

    assert not (output_dir / "images").exists() or list((output_dir / "images").glob("*.png")) == []
