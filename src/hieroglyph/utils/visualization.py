"""Draw detected glyph boxes on an image for display.

Channel-order convention (matches the rest of the codebase, see
segmentation/base.py): input images here are BGR (OpenCV's default, as
loaded by cv2.imread) or grayscale -- NOT RGB. A caller working from PIL
(which loads RGB) needs to convert before calling anything in this
pipeline; get this wrong and colors/grayscale-weighting come out subtly
off, not crashed, which is exactly the kind of bug worth documenting
loudly rather than letting someone rediscover it by squinting at a photo.
"""

from __future__ import annotations

import cv2
import numpy as np

from hieroglyph.pipeline.inference import DetectedSign

BOX_COLOR_BGR = (60, 160, 60)
TEXT_COLOR_BGR = (255, 255, 255)


def draw_annotated_image(image: np.ndarray, signs: list[DetectedSign]) -> np.ndarray:
    """Return an RGB copy of `image` with numbered boxes for each sign, in reading order."""
    if image.ndim == 2:
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        annotated = image.copy()

    for i, sign in enumerate(signs, start=1):
        box = sign.box
        cv2.rectangle(annotated, (box.x, box.y), (box.x2, box.y2), BOX_COLOR_BGR, 2)

        label = f"{i}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (box.x, box.y - text_h - 6),
            (box.x + text_w + 6, box.y),
            BOX_COLOR_BGR,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (box.x + 3, box.y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            TEXT_COLOR_BGR,
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
