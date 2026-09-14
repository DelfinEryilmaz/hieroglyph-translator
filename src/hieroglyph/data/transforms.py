"""Image transforms for feeding hieroglyph crops into a ResNet18 backbone.

ResNet18 was pretrained on ImageNet: 224x224 RGB images normalized with
ImageNet's mean/std. Our dataset is 50x75 grayscale, so every transform
pipeline here resizes and converts to 3-channel before normalizing the same
way the pretrained weights expect.
"""

from __future__ import annotations

from torchvision import transforms

# Standard ImageNet normalization stats — used because we're fine-tuning a
# backbone pretrained on ImageNet, not stats we computed ourselves. Using our
# own dataset's mean/std instead would mismatch what the pretrained
# convolution filters were calibrated for.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


def build_train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Transform for training: resize + normalize + mild augmentation.

    No horizontal flip (see notebooks/01_data_exploration.ipynb and the
    project plan) — mirroring a hieroglyph can turn it into a visually
    different, or even different-meaning, sign, so flipping would create
    misleading training examples rather than helpful augmentation.
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=5),  # small — real photos are rarely dead-level
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # simulates uneven lighting on stone
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Transform for val/test/inference: resize + normalize, nothing random.

    Deterministic on purpose — evaluation numbers need to be reproducible
    and comparable run to run, which random augmentation would break.
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
