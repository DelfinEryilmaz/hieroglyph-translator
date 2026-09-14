import torch

from hieroglyph.models.classifier import build_model, load_checkpoint, save_checkpoint


def test_forward_pass_output_shape_matches_num_classes():
    # pretrained=False: this test only checks architecture/shape correctness,
    # so we skip downloading ImageNet weights — keeps the test fast and
    # runnable offline.
    model = build_model(num_classes=171, pretrained=False)
    model.eval()

    dummy_batch = torch.zeros(4, 3, 224, 224)  # 4 fake images, matching real input shape
    with torch.no_grad():
        output = model(dummy_batch)

    assert output.shape == (4, 171)


def test_checkpoint_round_trip_preserves_predictions(tmp_path):
    model = build_model(num_classes=5, pretrained=False)
    model.eval()
    class_to_idx = {"A1": 0, "D21": 1, "G17": 2, "N35": 3, "X1": 4}

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        original_output = model(dummy_input)

    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(model, class_to_idx, checkpoint_path)
    loaded_model, loaded_class_to_idx = load_checkpoint(checkpoint_path)

    with torch.no_grad():
        loaded_output = loaded_model(dummy_input)

    assert loaded_class_to_idx == class_to_idx
    # Same weights + same input must give the exact same output — this is
    # what proves save/load didn't silently corrupt or shuffle anything.
    assert torch.allclose(original_output, loaded_output)
