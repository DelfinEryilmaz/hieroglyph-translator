"""Tiled/sliding-window inference for large or high-resolution real photos.

Why this exists: YoloSegmenter (yolo.py) resizes the whole image down to the
model's `imgsz` (640) before running inference. On a large, high-resolution
photo of a densely-packed inscription, that downscale destroys the fine
detail small glyphs need to be detected. TiledYoloSegmenter instead splits
the image into overlapping tiles, runs the model on each tile at its own
native size (no destructive whole-image downscale), offsets each tile's
detections back into the original image's coordinate system, and merges
duplicate detections of signs that straddle a tile boundary (and so get
detected more than once, in more than one tile) via greedy NMS.

For an image smaller than one tile in BOTH dimensions -- e.g. an
already-cropped small photo -- this degenerates to a single tile covering
the whole image, i.e. behaves exactly like plain YoloSegmenter. An image
smaller than a tile in only ONE dimension (e.g. a tall narrow column crop)
still gets tiled along its long axis; only the short axis degenerates to a
single origin. See generate_tile_origins.

Coordinate/size convention: everywhere in this module, an (x, y) or (width,
height) pair is in (horizontal, vertical) order, matching BoundingBox's
x/width vs y/height convention and matching how `canvas_size` is used in
synthesize.py (`canvas_w, canvas_h = canvas_size`). This is the OPPOSITE
order from numpy's `image.shape`, which for a BGR image is
`(height, width, channels)` -- `TiledYoloSegmenter.detect` is responsible
for converting from one order to the other when it reads `image.shape`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from hieroglyph.segmentation.base import Segmenter
from hieroglyph.segmentation.classical import ClassicalSegmenter
from hieroglyph.segmentation.types import BoundingBox
from hieroglyph.segmentation.yolo import _xyxy_to_boxes


def _axis_origins(dim: int, tile_dim: int, overlap: float) -> list[int]:
    """Tile origins along one axis, covering [0, dim) with tiles of length
    tile_dim, given dim >= tile_dim. Consecutive origins are `tile_dim *
    (1 - overlap)` apart, except the last one, which is snapped back to
    `dim - tile_dim` so the final tile always reaches the far edge exactly
    (it may therefore overlap its neighbor by more than requested, never
    less -- that only ever improves coverage of a straddling sign).
    """
    if dim == tile_dim:
        return [0]

    stride = max(1, round(tile_dim * (1 - overlap)))
    last = dim - tile_dim
    origins = list(range(0, last, stride))
    if origins[-1] != last:
        origins.append(last)
    return origins


def generate_tile_origins(
    image_size: tuple[int, int],
    tile_size: tuple[int, int] = (640, 640),
    overlap: float = 0.2,
) -> list[tuple[int, int]]:
    """(x, y) top-left origins covering image_size with tile_size tiles and
    the given fractional overlap between neighbors -- overlap exists so a
    sign straddling a tile boundary is captured whole by at least one tile.

    image_size and tile_size are both (width, height), matching
    BoundingBox's x/width, y/height convention.

    The "image smaller than a tile" case is decided **per axis, not
    globally**: an axis no longer than its tile dimension gets a single
    origin 0 (the tile, clipped by the slice in `detect`, already spans that
    whole axis), while the other axis is still tiled normally if it is
    longer. Deciding this globally would be a silent data-loss bug: a tall
    narrow column crop (e.g. 400x2000) is smaller than a 640x640 tile in x
    only, and a single (0, 0) origin would leave `detect`'s
    `image[0:640, 0:640]` slice examining just the top 640 rows -- 32% of
    the image -- with no error and no warning.
    """
    img_w, img_h = image_size
    tile_w, tile_h = tile_size

    x_origins = [0] if img_w <= tile_w else _axis_origins(img_w, tile_w, overlap)
    y_origins = [0] if img_h <= tile_h else _axis_origins(img_h, tile_h, overlap)
    return [(x, y) for y in y_origins for x in x_origins]


def _offset_box(box: BoundingBox, dx: int, dy: int) -> BoundingBox:
    """Shift a box by (dx, dy) -- used to convert a tile-local detection
    into global image coordinates by offsetting by the tile's origin."""
    return BoundingBox(x=box.x + dx, y=box.y + dy, width=box.width, height=box.height)


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    """Standard intersection-over-union (union denominator) -- distinct from
    synthesize.py's _boxes_overlap_fraction (min-area denominator, used for
    a different purpose there: rejecting placement during data synthesis).
    """
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = a.area + b.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def merge_tiled_detections(
    detections: list[tuple[BoundingBox, float]], iou_threshold: float = 0.5
) -> list[BoundingBox]:
    """Greedy NMS over detections already offset to global image coordinates:
    sort by confidence descending, keep a box only if it doesn't overlap an
    already-kept box above iou_threshold. Confidence is used only internally
    here and never appears on the returned BoundingBox -- this is exactly
    the "internal only, never leaks onto BoundingBox" constraint from
    types.py.
    """
    ordered = sorted(detections, key=lambda pair: pair[1], reverse=True)

    kept: list[BoundingBox] = []
    for box, _confidence in ordered:
        if all(_iou(box, kept_box) <= iou_threshold for kept_box in kept):
            kept.append(box)
    return kept


class TiledYoloSegmenter(Segmenter):
    """Segmenter that runs YOLO over overlapping tiles of the image instead
    of the whole image at once, so large/high-res photos don't lose small,
    densely-packed signs to a whole-image downscale. See module docstring.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        tile_size: tuple[int, int] = (640, 640),
        overlap: float = 0.2,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> None:
        self.model = YOLO(str(checkpoint_path))
        self.tile_size = tile_size
        self.overlap = overlap
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        # image.shape is (height, width, ...); generate_tile_origins wants
        # (width, height) -- see module docstring.
        image_height, image_width = image.shape[0], image.shape[1]
        origins = generate_tile_origins((image_width, image_height), self.tile_size, self.overlap)
        tile_w, tile_h = self.tile_size

        all_detections: list[tuple[BoundingBox, float]] = []
        for x, y in origins:
            tile = image[y : y + tile_h, x : x + tile_w]
            results = self.model.predict(tile, conf=self.confidence_threshold, verbose=False)
            xyxy = results[0].boxes.xyxy.tolist()
            confidences = results[0].boxes.conf.tolist()
            tile_boxes = _xyxy_to_boxes(xyxy)
            for box, confidence in zip(tile_boxes, confidences):
                all_detections.append((_offset_box(box, x, y), confidence))

        return merge_tiled_detections(all_detections, self.iou_threshold)


def load_tiled_or_fallback(checkpoint_path: Path) -> Segmenter:
    """TiledYoloSegmenter if a fine-tuned checkpoint exists at checkpoint_path,
    otherwise ClassicalSegmenter -- mirrors yolo.py's load_or_fallback exactly,
    kept here (not there) to avoid a circular import, since this module
    already imports from yolo.py.
    """
    if checkpoint_path.exists():
        return TiledYoloSegmenter(checkpoint_path)
    return ClassicalSegmenter()
