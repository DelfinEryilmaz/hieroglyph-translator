"""The segmentation interface: image -> list of bounding boxes.

This is the single most important structural decision in the whole project
(see the project plan and README). Every downstream stage (reading order,
classification, lookup) only ever talks to this interface, never to a
specific implementation. Right now the only implementation is classical
OpenCV contour detection (classical.py) — but because everything downstream
depends on this abstract interface instead, a future trained detector
(e.g. YOLO fine-tuned on synthesized composite images) could be dropped in
as a second implementation without changing anything in pipeline/inference.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from hieroglyph.segmentation.types import BoundingBox


class Segmenter(ABC):
    """Abstract base: anything that can find glyph regions in an image."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        """Return bounding boxes for each detected glyph region.

        `image` is a BGR (OpenCV's default channel order) or grayscale
        numpy array, as loaded by cv2.imread. Implementations should not
        mutate it.
        """
        raise NotImplementedError
