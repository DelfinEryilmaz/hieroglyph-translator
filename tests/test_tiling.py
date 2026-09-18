from pathlib import Path

import numpy as np

from hieroglyph.segmentation.classical import ClassicalSegmenter
from hieroglyph.segmentation.tiling import (
    _iou,
    _offset_box,
    generate_tile_origins,
    load_tiled_or_fallback,
    merge_tiled_detections,
)
from hieroglyph.segmentation.types import BoundingBox


# ---------------------------------------------------------------------------
# generate_tile_origins
# ---------------------------------------------------------------------------


def test_generate_tile_origins_single_origin_when_smaller_than_tile_in_both_dims():
    origins = generate_tile_origins((357, 140), tile_size=(640, 640), overlap=0.2)

    assert origins == [(0, 0)]


def test_generate_tile_origins_single_origin_when_exactly_tile_size():
    origins = generate_tile_origins((640, 640), tile_size=(640, 640), overlap=0.2)

    assert origins == [(0, 0)]


def test_generate_tile_origins_tiles_long_axis_when_smaller_in_only_one_dimension():
    # A tall narrow column crop is smaller than the tile in x only. The
    # short axis degenerates to a single origin, but the LONG axis must
    # still be tiled -- otherwise detect()'s image[y:y+640, x:x+640] slice
    # would silently examine only the top 640 rows of a 1200-row image.
    origins = generate_tile_origins((300, 1200), tile_size=(640, 640), overlap=0.2)

    xs = sorted({x for x, y in origins})
    ys = sorted({y for x, y in origins})
    assert xs == [0]
    assert len(ys) > 1
    assert max(ys) == 1200 - 640  # last tile snaps to the bottom edge


def test_generate_tile_origins_tiles_long_axis_when_smaller_in_height_only():
    # Mirror of the above for a wide short strip (smaller in y only).
    origins = generate_tile_origins((1800, 300), tile_size=(640, 640), overlap=0.2)

    xs = sorted({x for x, y in origins})
    ys = sorted({y for x, y in origins})
    assert ys == [0]
    assert len(xs) > 1
    assert max(xs) == 1800 - 640


def test_generate_tile_origins_tiles_cover_every_pixel_for_representative_sizes():
    # Regression guard for the whole class of "silently examines only part
    # of the image" bugs: the union of the tile rectangles actually sliced
    # by detect() -- image[y:y+tile_h, x:x+tile_w], which numpy clips at the
    # image bounds -- must cover the image completely, for images small in
    # one dimension, both, or neither.
    tile_w, tile_h = 640, 640
    for img_w, img_h in [
        (400, 2000),  # small in x only (tall column crop)
        (639, 4000),  # small in x only, just under one tile
        (3000, 300),  # small in y only (wide strip)
        (357, 140),  # small in both
        (640, 640),  # exactly one tile
        (1600, 900),  # large in both
    ]:
        origins = generate_tile_origins((img_w, img_h), tile_size=(tile_w, tile_h), overlap=0.2)

        covered = np.zeros((img_h, img_w), dtype=bool)
        for x, y in origins:
            covered[y : y + tile_h, x : x + tile_w] = True

        uncovered = int((~covered).sum())
        assert uncovered == 0, f"{uncovered} px of the {img_w}x{img_h} image are never examined"


def test_generate_tile_origins_covers_large_image_with_overlap():
    image_size = (1600, 900)
    tile_size = (640, 640)
    overlap = 0.2

    origins = generate_tile_origins(image_size, tile_size=tile_size, overlap=overlap)

    assert len(origins) > 1

    img_w, img_h = image_size
    tile_w, tile_h = tile_size

    # Full coverage: every point in the image must fall inside at least one
    # tile -- checked at the image's far edges (the hardest points to cover).
    for edge_x in (0, img_w - 1):
        for edge_y in (0, img_h - 1):
            assert any(
                x <= edge_x < x + tile_w and y <= edge_y < y + tile_h for x, y in origins
            ), f"point ({edge_x}, {edge_y}) not covered by any tile"

    # No origin pushes a tile out of bounds.
    for x, y in origins:
        assert 0 <= x <= img_w - tile_w
        assert 0 <= y <= img_h - tile_h

    # Consecutive tiles along an axis overlap by at least the requested
    # fraction of tile_size (except possibly the final snapped-to-edge tile,
    # which can overlap by more than requested -- never less).
    xs = sorted({x for x, y in origins})
    for x1, x2 in zip(xs, xs[1:]):
        overlap_px = (x1 + tile_w) - x2
        assert overlap_px >= overlap * tile_w - 1  # -1 for rounding slack

    ys = sorted({y for x, y in origins})
    for y1, y2 in zip(ys, ys[1:]):
        overlap_px = (y1 + tile_h) - y2
        assert overlap_px >= overlap * tile_h - 1


def test_generate_tile_origins_last_tile_snaps_to_far_edge():
    origins = generate_tile_origins((1000, 640), tile_size=(640, 640), overlap=0.2)

    xs = [x for x, y in origins]
    assert max(xs) == 1000 - 640


# ---------------------------------------------------------------------------
# _offset_box
# ---------------------------------------------------------------------------


def test_offset_box_shifts_position_but_not_size():
    box = BoundingBox(x=10, y=20, width=30, height=40)

    shifted = _offset_box(box, dx=100, dy=200)

    assert shifted == BoundingBox(x=110, y=220, width=30, height=40)


def test_offset_box_zero_offset_is_identity():
    box = BoundingBox(x=5, y=6, width=7, height=8)

    assert _offset_box(box, dx=0, dy=0) == box


# ---------------------------------------------------------------------------
# _iou
# ---------------------------------------------------------------------------


def test_iou_no_overlap_is_zero():
    a = BoundingBox(x=0, y=0, width=10, height=10)
    b = BoundingBox(x=100, y=100, width=10, height=10)

    assert _iou(a, b) == 0.0


def test_iou_identical_boxes_is_one():
    a = BoundingBox(x=0, y=0, width=10, height=10)
    b = BoundingBox(x=0, y=0, width=10, height=10)

    assert _iou(a, b) == 1.0


def test_iou_partial_overlap_matches_hand_computed_value():
    # a: [0,10]x[0,10] area=100. b: [5,15]x[5,15] area=100.
    # intersection: [5,10]x[5,10] = 5*5 = 25. union = 100+100-25 = 175.
    a = BoundingBox(x=0, y=0, width=10, height=10)
    b = BoundingBox(x=5, y=5, width=10, height=10)

    assert _iou(a, b) == 25 / 175


def test_iou_one_box_fully_contains_another_is_much_less_than_one():
    # Distinguishes real IoU from synthesize.py's _boxes_overlap_fraction,
    # which uses the smaller box's area as the denominator and would report
    # 1.0 here. Real IoU uses the union, so full containment of a small box
    # inside a much larger one gives a small IoU.
    outer = BoundingBox(x=0, y=0, width=100, height=100)
    inner = BoundingBox(x=10, y=10, width=10, height=10)

    iou = _iou(outer, inner)

    assert iou == 100 / 10000
    assert iou < 0.2  # nowhere near 1.0, unlike _boxes_overlap_fraction


# ---------------------------------------------------------------------------
# merge_tiled_detections
# ---------------------------------------------------------------------------


def test_merge_tiled_detections_keeps_higher_confidence_of_overlapping_pair():
    low = BoundingBox(x=0, y=0, width=10, height=10)
    high = BoundingBox(x=1, y=1, width=10, height=10)  # heavily overlapping with `low`

    merged = merge_tiled_detections([(low, 0.3), (high, 0.9)], iou_threshold=0.5)

    assert merged == [high]


def test_merge_tiled_detections_keeps_both_non_overlapping_detections():
    a = BoundingBox(x=0, y=0, width=10, height=10)
    b = BoundingBox(x=500, y=500, width=10, height=10)

    merged = merge_tiled_detections([(a, 0.5), (b, 0.9)], iou_threshold=0.5)

    assert len(merged) == 2
    assert a in merged
    assert b in merged


def test_merge_tiled_detections_empty_input_returns_empty_list():
    assert merge_tiled_detections([], iou_threshold=0.5) == []


def test_merge_tiled_detections_below_threshold_overlap_keeps_both():
    # Two boxes that overlap a little, but below iou_threshold -- both kept.
    a = BoundingBox(x=0, y=0, width=10, height=10)
    b = BoundingBox(x=9, y=9, width=10, height=10)  # small corner overlap

    iou = _iou(a, b)
    assert iou < 0.5  # sanity check on the fixture itself

    merged = merge_tiled_detections([(a, 0.9), (b, 0.5)], iou_threshold=0.5)

    assert len(merged) == 2


def test_load_tiled_or_fallback_returns_classical_when_no_checkpoint(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.pt"

    segmenter = load_tiled_or_fallback(missing_path)

    assert isinstance(segmenter, ClassicalSegmenter)
