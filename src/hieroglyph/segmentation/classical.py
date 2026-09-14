"""Classical computer-vision glyph segmentation: no trained model involved.

The core idea, step by step:
1. Turn the photo into a black-and-white ("binary") mask where glyph pixels
   are white and background is black.
2. Merge nearby strokes of the same glyph into one solid blob (a hieroglyph
   is often several disconnected pen-strokes, not one connected shape).
3. Find the outline ("contour") of each blob and take its bounding box.
4. Throw out blobs that are too small (dust/noise) or too large (the whole
   image got picked up as one blob, usually meaning thresholding failed).

Known limitation (stated in the README too, not hidden): this works well on
clean, evenly-lit photos and struggles on heavily weathered/shadowed wall
photos, since step 1 (turning a messy photo into a clean black/white mask)
is exactly the part real-world noise breaks. This is why segmentation sits
behind the Segmenter interface (base.py) — a future trained detector can
replace this specific class without touching anything downstream.
"""

from __future__ import annotations

import cv2
import numpy as np

from hieroglyph.segmentation.base import Segmenter
from hieroglyph.segmentation.types import BoundingBox


class ClassicalSegmenter(Segmenter):
    def __init__(
        self,
        invert: bool = True,
        use_adaptive_threshold: bool = True,
        blur_kernel_size: int = 3,
        close_kernel_size: int = 7,
        min_area_fraction: float = 0.0005,
        max_area_fraction: float = 0.5,
    ) -> None:
        """
        invert: True assumes glyphs are DARKER than their background (typical
            for incised/shadowed carvings). Set False if your photos show
            light-colored glyphs on a dark background instead.
        use_adaptive_threshold: adaptive thresholding computes a different
            brightness cutoff per local region of the image, which copes
            better with uneven lighting across a photo than a single global
            cutoff. Set False to use Otsu's method (a single global cutoff)
            instead, which can work better on very clean, evenly-lit images.
        blur_kernel_size: mild blur before thresholding, to reduce speckle
            noise that would otherwise show up as tiny false-positive blobs.
        close_kernel_size: how aggressively nearby strokes get merged into
            one blob. Too small: one glyph splits into several boxes (seen
            empirically at the default=5 on a real glyph with a detached
            mark — fixed by raising to 7). Too large: separate neighboring
            glyphs merge into one box — a real risk on an actual inscription
            photo where glyphs sit close together, unlike our well-spaced
            test composite, so don't raise this further without checking
            against a densely-packed real photo.
        min_area_fraction / max_area_fraction: blobs outside this size range
            (as a fraction of the total image area) are discarded as noise
            or as a failed/over-merged threshold, respectively.
        """
        self.invert = invert
        self.use_adaptive_threshold = use_adaptive_threshold
        self.blur_kernel_size = blur_kernel_size
        self.close_kernel_size = close_kernel_size
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction

    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        gray = self._to_grayscale(image)
        binary = self._binarize(gray)
        binary = self._merge_nearby_strokes(binary)
        return self._extract_boxes(binary, image_area=gray.shape[0] * gray.shape[1])

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)

        # THRESH_BINARY_INV: pixels darker than the cutoff become white (255,
        # "foreground") and lighter pixels become black (0, "background") —
        # matches our `invert` assumption that glyphs are the darker marks.
        thresh_type = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY

        if self.use_adaptive_threshold:
            return cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresh_type,
                blockSize=25,  # local neighborhood size the cutoff is computed from
                C=5,  # constant subtracted from the local mean — higher = stricter
            )

        # Otsu's method picks one global cutoff automatically from the
        # image's overall brightness histogram.
        _, binary = cv2.threshold(blurred, 0, 255, thresh_type | cv2.THRESH_OTSU)
        return binary

    def _merge_nearby_strokes(self, binary: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.close_kernel_size, self.close_kernel_size)
        )
        # MORPH_CLOSE = dilate then erode: expands white regions enough to
        # bridge small gaps between strokes, then shrinks back down so blobs
        # don't end up permanently larger than the original strokes.
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def _extract_boxes(self, binary: np.ndarray, image_area: int) -> list[BoundingBox]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.min_area_fraction * image_area
        max_area = self.max_area_fraction * image_area

        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if min_area <= area <= max_area:
                boxes.append(BoundingBox(x=x, y=y, width=w, height=h))
        return boxes
