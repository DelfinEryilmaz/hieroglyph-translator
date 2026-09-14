"""PyTorch Dataset for hieroglyph glyph images, plus the class<->index mapping.

We don't use torchvision.datasets.ImageFolder here because it assumes one
folder = one split with classes auto-discovered from subfolder names. Our
split instead comes from split_dataset() in split.py, which already decided
which images go to train/val/test (including routing rare classes entirely
to train) — this module just wraps that decision in something PyTorch's
DataLoader can iterate over.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from hieroglyph.data.split import SplitResult, samples_by_class_from_dir, split_dataset
from hieroglyph.data.transforms import build_eval_transform, build_train_transform


def build_class_to_idx(samples_by_class: dict[str, list[str]]) -> dict[str, int]:
    """Assign a stable index to every class, sorted alphabetically.

    Sorting makes this deterministic across runs and machines. That matters
    because this exact mapping has to be saved alongside the trained model
    checkpoint (see project plan, Phase 10) — if we built it in whatever
    order dict iteration happened to give us, a different run could assign
    different indices to the same sign, silently corrupting predictions made
    with an old checkpoint.
    """
    return {name: idx for idx, name in enumerate(sorted(samples_by_class))}


def save_class_to_idx(class_to_idx: dict[str, int], path: Path) -> None:
    path.write_text(json.dumps(class_to_idx, indent=2))


def load_class_to_idx(path: Path) -> dict[str, int]:
    return json.loads(path.read_text())


class HieroglyphDataset(Dataset):
    """Wraps a list of (image_path, class_name) pairs for PyTorch.

    `samples` is meant to be one of SplitResult.train / .val / .test.
    `class_to_idx` must be the *same* mapping across train/val/test/inference
    — build it once from the full dataset (build_class_to_idx) and pass it
    to every split's dataset, rather than rebuilding it per split, or the
    same index could mean a different sign in each one.
    """

    def __init__(
        self,
        samples: list[tuple[str, str]],
        class_to_idx: dict[str, int],
        transform=None,
    ) -> None:
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, class_name = self.samples[index]
        # .convert("L") forces single-channel grayscale even if some image
        # sneaks in as RGB/RGBA/P mode — transforms.py expands to 3 channels
        # afterward via transforms.Grayscale(num_output_channels=3).
        image = Image.open(path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[class_name]
        return image, label


def build_dataloaders(
    data_root: Path,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """One-call convenience: raw folder -> ready-to-use train/val/test DataLoaders.

    Ties together every piece above (split_dataset, class_to_idx, the two
    transform pipelines, HieroglyphDataset) so training code doesn't need to
    know about any of the intermediate steps. Returns the class_to_idx
    mapping too, since callers (training loop) need it to save alongside
    the checkpoint.
    """
    samples_by_class = samples_by_class_from_dir(data_root)
    class_to_idx = build_class_to_idx(samples_by_class)
    split: SplitResult = split_dataset(samples_by_class)

    train_ds = HieroglyphDataset(split.train, class_to_idx, transform=build_train_transform())
    val_ds = HieroglyphDataset(split.val, class_to_idx, transform=build_eval_transform())
    test_ds = HieroglyphDataset(split.test, class_to_idx, transform=build_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_to_idx
