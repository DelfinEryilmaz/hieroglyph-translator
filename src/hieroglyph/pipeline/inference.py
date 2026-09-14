"""The full pipeline, stitched together: photo -> ordered per-sign glosses.

segment (find glyph regions) -> crop each region -> classify (batched, one
model forward pass) -> map predicted index back to a Gardiner code -> sort
into reading order -> look up transliteration/gloss -> assemble the result.

Every stage before this module already has its own interface/tests
(Segmenter, the classifier, sort_reading_order, GardinerLookup) -- this
module's only job is wiring them together correctly, which is exactly the
kind of thing that's easy to get subtly wrong (crop coordinates, tensor
batching, index<->code mapping) without ever raising an exception. That's
why the integration test below checks the *wiring*, not any single stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn

from hieroglyph.data.transforms import build_eval_transform
from hieroglyph.lookup.gardiner_lookup import GardinerLookup
from hieroglyph.pipeline.reading_order import sort_reading_order
from hieroglyph.segmentation.base import Segmenter
from hieroglyph.segmentation.classical import ClassicalSegmenter
from hieroglyph.segmentation.types import BoundingBox


@dataclass
class DetectedSign:
    box: BoundingBox
    gardiner_code: str
    confidence: float
    transliteration: str
    gloss: str


@dataclass
class PipelineResult:
    signs: list[DetectedSign]
    concatenated_gloss: str


def _crop_to_pil(image: np.ndarray, box: BoundingBox) -> Image.Image:
    crop = image[box.y : box.y2, box.x : box.x2]
    if crop.ndim == 2:
        return Image.fromarray(crop, mode="L")
    # OpenCV loads color images as BGR; PIL expects RGB, so convert the
    # channel order or colors would come out swapped (blue/red flipped).
    # Downstream the model treats this as grayscale anyway (see
    # transforms.py), but getting the conversion right here is still the
    # honest way to do it rather than relying on that to mask a bug.
    import cv2

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, mode="RGB")


def run_inference(
    image: np.ndarray,
    model: nn.Module,
    class_to_idx: dict[str, int],
    lookup: GardinerLookup,
    segmenter: Segmenter | None = None,
    device: str = "cpu",
) -> PipelineResult:
    segmenter = segmenter or ClassicalSegmenter()
    idx_to_class = {idx: code for code, idx in class_to_idx.items()}
    transform = build_eval_transform()

    boxes = segmenter.detect(image)
    if not boxes:
        return PipelineResult(signs=[], concatenated_gloss="")

    ordered_boxes = sort_reading_order(boxes)

    # One batched forward pass rather than one call per glyph -- much
    # faster on both CPU and GPU than looping model(crop) per sign.
    crops = torch.stack([transform(_crop_to_pil(image, box)) for box in ordered_boxes])

    model.eval()
    with torch.no_grad():
        logits = model(crops.to(device))
        probabilities = torch.softmax(logits, dim=1)
        confidences, predicted_indices = probabilities.max(dim=1)

    signs = []
    for box, idx, confidence in zip(ordered_boxes, predicted_indices.tolist(), confidences.tolist()):
        gardiner_code = idx_to_class[idx]
        entry = lookup.lookup(gardiner_code)
        signs.append(
            DetectedSign(
                box=box,
                gardiner_code=gardiner_code,
                confidence=confidence,
                transliteration=entry.transliteration,
                gloss=entry.gloss,
            )
        )

    concatenated_gloss = " ".join(sign.gloss or f"[{sign.gardiner_code}]" for sign in signs)
    return PipelineResult(signs=signs, concatenated_gloss=concatenated_gloss)
