"""Model definition: a ResNet18 backbone fine-tuned for hieroglyph classification.

We do NOT train a CNN from scratch. With only ~2,800 training images spread
across 171 classes (many classes with just a handful of examples — see
Phase 2's exploration notebook), there isn't nearly enough data to learn
good general visual features from zero.

Instead we start from a ResNet18 already trained on ImageNet (1.4 million
photos, 1000 everyday object classes: dogs, cars, chairs...) and swap out
only its final decision layer for ours. This is "transfer learning": the
early/middle convolutional layers already learned to detect edges, curves,
textures, and shapes — general-purpose visual building blocks that are
useful for recognizing hieroglyphs too, even though the model has never
seen one. Only the final layer starts blank and has to learn what those
171 sign classes actually look like.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a ResNet18 with its final classification layer replaced.

    ResNet18's original final layer (`model.fc`) is a single linear layer
    that turns the 512 numbers ResNet computes per image ("features") into
    1000 ImageNet class scores. We don't want ImageNet's 1000 classes — we
    want our 171 Gardiner sign classes — so we throw that final layer away
    and attach a fresh, untrained one sized for us. Every earlier layer
    keeps its ImageNet-learned weights as a head start; only this new final
    layer starts from random values and has everything to learn.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # in_features=512 is how many features ResNet18 extracts per image —
    # that doesn't change. out_features is what we're replacing: 1000
    # ImageNet classes -> our num_classes.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def save_checkpoint(model: nn.Module, class_to_idx: dict[str, int], path: Path) -> None:
    """Save model weights *and* the class_to_idx mapping together, in one file.

    Saving them separately would be a trap: if class_to_idx ever got
    regenerated later (e.g. the dataset changed slightly and dict ordering
    shifted), an old checkpoint's output index 47 could silently start
    meaning a different sign than it did when the model was trained.
    Bundling them together makes that mismatch impossible — whoever loads
    this file always gets the exact mapping that was true during training.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "class_to_idx": class_to_idx}, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> tuple[nn.Module, dict[str, int]]:
    """Load a checkpoint saved by save_checkpoint, rebuilding a matching model."""
    # weights_only=True: our checkpoint only ever contains tensors and a
    # plain str->int dict, so this is safe and avoids torch.load's arbitrary
    # code execution risk that comes with unpickling general Python objects.
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    class_to_idx = checkpoint["class_to_idx"]
    # pretrained=False: no point downloading ImageNet weights just to
    # immediately overwrite every one of them with the saved state_dict.
    model = build_model(num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # inference mode: disables dropout/batchnorm training behavior
    return model, class_to_idx
