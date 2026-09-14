from hieroglyph.data.split import split_dataset


def test_rare_class_goes_entirely_to_train():
    samples = {"rare_sign": ["img1.png"]}
    result = split_dataset(samples, min_images_to_hold_out=3)

    assert result.train == [("img1.png", "rare_sign")]
    assert result.val == []
    assert result.test == []


def test_common_class_gets_val_and_test_representation():
    samples = {"common_sign": [f"img{i}.png" for i in range(20)]}
    result = split_dataset(samples, val_frac=0.15, test_frac=0.15, min_images_to_hold_out=3)

    assert len(result.val) >= 1
    assert len(result.test) >= 1
    assert len(result.train) + len(result.val) + len(result.test) == 20


# Mixes a class large enough to split (sign_a) with one too small (sign_b),
# and checks every original image shows up exactly once across the three
# output lists combined — no image dropped, none duplicated into two splits.
def test_split_is_a_partition_no_overlap_no_loss():
    samples = {"sign_a": [f"a{i}.png" for i in range(10)], "sign_b": [f"b{i}.png" for i in range(2)]}
    result = split_dataset(samples, min_images_to_hold_out=3)

    all_paths = [p for p, _ in result.train + result.val + result.test]
    expected = [f"a{i}.png" for i in range(10)] + [f"b{i}.png" for i in range(2)]

    assert sorted(all_paths) == sorted(expected)
    assert len(set(all_paths)) == len(all_paths)
