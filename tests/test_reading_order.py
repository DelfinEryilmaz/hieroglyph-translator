from hieroglyph.pipeline.reading_order import sort_reading_order
from hieroglyph.segmentation.types import BoundingBox


def _box(x, y, w=40, h=40):
    return BoundingBox(x=x, y=y, width=w, height=h)


def test_empty_and_single_box():
    assert sort_reading_order([]) == []

    single = _box(10, 10)
    assert sort_reading_order([single]) == [single]


def test_two_rows_three_columns_sorted_top_to_bottom_left_to_right():
    # Deliberately built out of reading order and out of insertion order,
    # to make sure sorting isn't accidentally relying on input order.
    row1_col3 = _box(220, 10)
    row2_col1 = _box(10, 110)
    row1_col1 = _box(10, 10)
    row2_col3 = _box(220, 110)
    row1_col2 = _box(115, 10)
    row2_col2 = _box(115, 110)

    boxes = [row1_col3, row2_col1, row1_col1, row2_col3, row1_col2, row2_col2]
    ordered = sort_reading_order(boxes)

    assert ordered == [row1_col1, row1_col2, row1_col3, row2_col1, row2_col2, row2_col3]


def test_slightly_misaligned_row_still_groups_together():
    # Real detected boxes rarely have perfectly identical y -- a few pixels
    # of jitter within the same row shouldn't split it into two rows.
    left = _box(10, 10, h=40)
    right = _box(60, 14, h=40)  # 4px lower, same row in practice

    ordered = sort_reading_order([right, left])
    assert ordered == [left, right]


def test_clearly_separate_rows_stay_separate():
    top = _box(10, 10, h=40)
    bottom = _box(10, 200, h=40)  # far below -- must be a different row

    ordered = sort_reading_order([bottom, top])
    assert ordered == [top, bottom]
