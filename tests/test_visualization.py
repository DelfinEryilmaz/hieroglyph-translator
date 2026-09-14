import numpy as np

from hieroglyph.pipeline.inference import DetectedSign
from hieroglyph.segmentation.types import BoundingBox
from hieroglyph.utils.visualization import draw_annotated_image


def _stub_sign(x, y, w=40, h=40, code="D21"):
    return DetectedSign(
        box=BoundingBox(x=x, y=y, width=w, height=h),
        gardiner_code=code,
        confidence=0.9,
        transliteration="r",
        gloss="Mouth",
    )


def test_output_is_rgb_with_same_dimensions_as_grayscale_input():
    image = np.full((100, 150), 255, dtype=np.uint8)
    annotated = draw_annotated_image(image, [_stub_sign(10, 10)])

    assert annotated.shape == (100, 150, 3)


def test_boxes_actually_change_pixels():
    image = np.full((100, 150), 255, dtype=np.uint8)
    annotated = draw_annotated_image(image, [_stub_sign(10, 10)])

    # A box was drawn, so the annotated image must differ from a plain
    # grayscale-to-RGB conversion of the blank input.
    assert not np.array_equal(annotated[:, :, 0], image)


def test_no_signs_leaves_image_unchanged_besides_color_conversion():
    image = np.full((50, 50), 200, dtype=np.uint8)
    annotated = draw_annotated_image(image, [])

    assert np.all(annotated == 200)
