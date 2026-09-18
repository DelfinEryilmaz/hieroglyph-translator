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
