from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from hieroglyph.lookup.gardiner_lookup import GardinerLookup
from hieroglyph.pipeline.inference import run_inference
from hieroglyph.segmentation.classical import ClassicalSegmenter

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "gardiner_signs_fixture.csv"


class _StubModel(nn.Module):
    """Always predicts the same class, regardless of input.

    This test isn't checking whether a model classifies correctly -- that's
    what test_classifier.py and the real evaluation notebook are for. It's
    checking whether the *pipeline wiring* (crop coordinates, batching,
    index-to-code mapping, reading order, lookup) is correct, which a real
    trained model would make much slower and harder to isolate.
    """

    def __init__(self, num_classes: int, always_predict: int):
        super().__init__()
        self.num_classes = num_classes
        self.always_predict = always_predict

    def forward(self, x):
        logits = torch.full((x.shape[0], self.num_classes), -10.0)
        logits[:, self.always_predict] = 10.0
        return logits


def _synthetic_two_glyph_image():
    image = np.full((150, 250), 255, dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (70, 70), color=0, thickness=-1)  # left square
    cv2.rectangle(image, (150, 30), (190, 70), color=0, thickness=-1)  # right square
    return image


def test_pipeline_wiring_end_to_end_with_stub_model():
    image = _synthetic_two_glyph_image()
    class_to_idx = {"D21": 0, "A55": 1}
    model = _StubModel(num_classes=2, always_predict=0)  # always predicts D21
    lookup = GardinerLookup(FIXTURE_PATH)

    result = run_inference(image, model, class_to_idx, lookup, segmenter=ClassicalSegmenter())

    assert len(result.signs) == 2
    assert all(sign.gardiner_code == "D21" for sign in result.signs)
    assert all(sign.gloss == "Mouth" for sign in result.signs)
    assert all(sign.transliteration == "r, jw" for sign in result.signs)

    # Reading order: left square should come before the right one.
    assert result.signs[0].box.x < result.signs[1].box.x
    assert result.concatenated_gloss == "Mouth Mouth"


def test_pipeline_returns_empty_result_when_nothing_detected():
    blank_image = np.full((100, 100), 255, dtype=np.uint8)
    class_to_idx = {"D21": 0}
    model = _StubModel(num_classes=1, always_predict=0)
    lookup = GardinerLookup(FIXTURE_PATH)

    result = run_inference(blank_image, model, class_to_idx, lookup, segmenter=ClassicalSegmenter())

    assert result.signs == []
    assert result.concatenated_gloss == ""


def test_pipeline_marks_unresolved_signs_in_concatenated_gloss():
    # D156 is one of the genuinely unresolved codes (empty gloss, by
    # design -- see reports/gardiner-lookup-table-sourcing.md). The
    # concatenated string should flag it rather than silently leave a gap.
    image = _synthetic_two_glyph_image()
    class_to_idx = {"D156": 0}
    model = _StubModel(num_classes=1, always_predict=0)
    lookup = GardinerLookup(FIXTURE_PATH)

    result = run_inference(image, model, class_to_idx, lookup, segmenter=ClassicalSegmenter())

    assert result.signs[0].gloss == ""
    assert "[D156]" in result.concatenated_gloss
