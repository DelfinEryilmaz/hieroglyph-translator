"""A trained-detector Segmenter: YOLOv8 instance segmentation, used only
for its bounding boxes (see segmentation/types.py -- our interface never
needed the polygon masks YOLOv8-seg also produces).

Fine-tuned from a pretrained checkpoint sourced in
reports/2026-09-15-segmentation-detector-sourcing.md -- see that file for
where the starting weights came from and why we didn't train from scratch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from hieroglyph.segmentation.base import Segmenter
from hieroglyph.segmentation.classical import ClassicalSegmenter
from hieroglyph.segmentation.types import BoundingBox


def _xyxy_to_boxes(xyxy: list[tuple[float, float, float, float]]) -> list[BoundingBox]:
    """Convert YOLO's (x1, y1, x2, y2) corner format into our BoundingBox
    (x, y, width, height) convention. Kept separate from YoloSegmenter.detect
    so this conversion logic is testable without loading a real model.
    """
    boxes = []
    for x1, y1, x2, y2 in xyxy:
        boxes.append(
            BoundingBox(x=round(x1), y=round(y1), width=round(x2 - x1), height=round(y2 - y1))
        )
    return boxes


class YoloSegmenter(Segmenter):
    def __init__(self, checkpoint_path: Path, confidence_threshold: float = 0.25) -> None:
        self.model = YOLO(str(checkpoint_path))
        self.confidence_threshold = confidence_threshold

    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        results = self.model.predict(image, conf=self.confidence_threshold, verbose=False)
        xyxy = results[0].boxes.xyxy.tolist()
        return _xyxy_to_boxes(xyxy)


def load_or_fallback(checkpoint_path: Path) -> Segmenter:
    """YoloSegmenter if a fine-tuned checkpoint exists at checkpoint_path,
    otherwise ClassicalSegmenter -- lets callers (e.g. the Streamlit demo)
    work before notebooks/04_train_segmenter.ipynb has ever been run.
    """
    if checkpoint_path.exists():
        return YoloSegmenter(checkpoint_path)
    return ClassicalSegmenter()
