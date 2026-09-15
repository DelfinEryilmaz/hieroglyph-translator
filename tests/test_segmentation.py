import numpy as np
import cv2
from pathlib import Path

from hieroglyph.segmentation.classical import ClassicalSegmenter
from hieroglyph.segmentation.types import BoundingBox
from hieroglyph.segmentation.yolo import _xyxy_to_boxes, load_or_fallback


def _synthetic_image_with_shapes(shape_centers, shape_size=30, canvas_size=300):
    """A deterministic test image: black squares on a white background at
    known locations. No external image file needed, so this test is
    reproducible anywhere without depending on real photos."""
    image = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)  # white background
    half = shape_size // 2
    for cx, cy in shape_centers:
        cv2.rectangle(image, (cx - half, cy - half), (cx + half, cy + half), color=0, thickness=-1)
    return image


def test_detects_one_box_per_well_separated_shape():
    expected_centers = [(50, 50), (150, 50), (250, 250)]
    image = _synthetic_image_with_shapes(expected_centers)

    segmenter = ClassicalSegmenter()
    boxes = segmenter.detect(image)

    assert len(boxes) == len(expected_centers)

    # Every expected shape should have a detected box roughly centered on it
    # -- "roughly" because thresholding/blur can shift edges by a pixel or two.
    for cx, cy in expected_centers:
        assert any(
            abs(box.center[0] - cx) < 5 and abs(box.center[1] - cy) < 5 for box in boxes
        ), f"No detected box near expected shape at ({cx}, {cy})"


def test_blank_image_detects_nothing():
    blank = np.full((100, 100), 255, dtype=np.uint8)
    boxes = ClassicalSegmenter().detect(blank)
    assert boxes == []


def test_tiny_speck_is_filtered_as_noise():
    image = np.full((300, 300), 255, dtype=np.uint8)
    cv2.rectangle(image, (150, 150), (151, 151), color=0, thickness=-1)  # 1-2px speck

    boxes = ClassicalSegmenter().detect(image)
    assert boxes == [], "A 1-2px speck should be filtered out by min_area_fraction"


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
