"""Per-class train/val/test split with explicit handling for rare classes.

sklearn's stratified train_test_split needs at least 2 samples per class at
each split step, so it fails outright on classes with only 1 image. Here we
split each class independently instead: classes too small to meaningfully
hold anything back go entirely into train; other classes get at least 1
image guaranteed in val and 1 in test.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SplitResult:
    # Each list holds (image_path, class_name) pairs — the flat format
    # torchvision-style datasets expect, rather than nested per-class dicts.
    train: list[tuple[str, str]] = field(default_factory=list)
    val: list[tuple[str, str]] = field(default_factory=list)
    test: list[tuple[str, str]] = field(default_factory=list)


def samples_by_class_from_dir(root: Path) -> dict[str, list[str]]:
    """Build {class_name: [image_path, ...]} from data/raw's folder layout."""
    samples: dict[str, list[str]] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        samples[class_dir.name] = [str(f) for f in class_dir.iterdir() if f.is_file()]
    return samples


def split_dataset(
    samples_by_class: dict[str, list[str]],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    min_images_to_hold_out: int = 3,
    seed: int = 0,
) -> SplitResult:
    """Split each class's images into train/val/test independently.

    Classes with fewer than `min_images_to_hold_out` images go entirely into
    train — there isn't enough data to hold any back for val/test and still
    have something meaningful to train on. Other classes get at least 1 image
    in val and 1 in test, with the remainder in train.
    """
    # A single shared Random instance (not the global `random` module) makes
    # the split reproducible across runs without affecting other code that
    # happens to call random.* elsewhere.
    rng = random.Random(seed)
    result = SplitResult()

    # Split each class on its own, rather than shuffling+slicing the whole
    # dataset at once — that's what makes per-class rare-class handling
    # possible at all (a global split has no concept of "this class ran out
    # of images to allocate").
    for class_name, paths in samples_by_class.items():
        shuffled = paths[:]  # copy — don't mutate the caller's list
        rng.shuffle(shuffled)
        n = len(shuffled)

        if n < min_images_to_hold_out:
            # Too few images to hold any back and still train on something;
            # all go to train, none to val/test (see split.py module docstring).
            result.train.extend((p, class_name) for p in shuffled)
            continue

        # `max(1, ...)` guarantees at least one image in val and test once a
        # class clears the min_images_to_hold_out bar, even when val_frac/
        # test_frac would round down to 0 for a small class (e.g. n=3).
        n_val = max(1, round(n * val_frac))
        n_test = max(1, round(n * test_frac))
        n_train = n - n_val - n_test  # remainder — always >=1 given the bar above

        result.train.extend((p, class_name) for p in shuffled[:n_train])
        result.val.extend((p, class_name) for p in shuffled[n_train : n_train + n_val])
        result.test.extend((p, class_name) for p in shuffled[n_train + n_val :])

    return result
