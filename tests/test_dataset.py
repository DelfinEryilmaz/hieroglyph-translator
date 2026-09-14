from PIL import Image

from hieroglyph.data.dataset import HieroglyphDataset, build_class_to_idx
from hieroglyph.data.transforms import IMAGE_SIZE, build_eval_transform


def _make_fake_dataset(tmp_path):
    # Two tiny classes with hand-drawn (well, PIL-drawn) images — real image
    # bytes are needed since HieroglyphDataset actually opens them with PIL,
    # but the *content* doesn't matter for this test, only shape/label.
    samples_by_class = {}
    for class_name, color in [("A1", 0), ("D21", 255)]:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(2):
            img = Image.new("L", (50, 75), color=color)
            img.save(class_dir / f"{i}.png")
        samples_by_class[class_name] = [str(p) for p in class_dir.iterdir()]
    return samples_by_class


def test_dataset_returns_correctly_shaped_tensor_and_valid_label(tmp_path):
    samples_by_class = _make_fake_dataset(tmp_path)
    class_to_idx = build_class_to_idx(samples_by_class)

    # Flatten into the (path, class_name) format HieroglyphDataset expects —
    # the same shape split_dataset() would produce.
    samples = [(p, cls) for cls, paths in samples_by_class.items() for p in paths]
    dataset = HieroglyphDataset(samples, class_to_idx, transform=build_eval_transform())

    image, label = dataset[0]

    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert 0 <= label < len(class_to_idx)


def test_class_to_idx_is_stable_regardless_of_input_order():
    # Same classes, deliberately inserted in a different order — the
    # resulting mapping must be identical either way (alphabetical sort),
    # since this mapping gets saved with the model checkpoint and must not
    # depend on incidental dict ordering.
    forward = build_class_to_idx({"A1": [], "D21": [], "Z7": []})
    reversed_order = build_class_to_idx({"Z7": [], "D21": [], "A1": []})

    assert forward == reversed_order == {"A1": 0, "D21": 1, "Z7": 2}
