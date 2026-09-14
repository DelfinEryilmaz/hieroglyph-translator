"""Sort detected glyph boxes into a reading order.

Simplifying assumption: row-major, top-to-bottom then left-to-right within
each row -- like reading English text. This is a real limitation, not
solved here: actual Egyptian hieroglyphic reading order depends on which
way the glyphs (especially animal/human signs) face, and inscriptions can
run in columns, right-to-left, or left-to-right depending on that facing
direction. Getting that right would require the classifier to also learn
sign orientation, which is out of scope for this project (see README).
"""

from __future__ import annotations

from hieroglyph.segmentation.types import BoundingBox


def sort_reading_order(boxes: list[BoundingBox], row_tolerance: float = 0.6) -> list[BoundingBox]:
    """Group boxes into rows by vertical (y) proximity, then sort each row left-to-right.

    row_tolerance: two boxes are considered part of the same row if their
    vertical centers are within `row_tolerance * average_box_height` of each
    other. This needs to be a fraction of box height (not a fixed pixel
    count) because glyphs vary a lot in size -- a fixed pixel threshold that
    works for large glyphs would incorrectly merge rows of small ones.
    """
    if not boxes:
        return []

    # Sort by vertical position first so we can sweep top-to-bottom and
    # group into rows as we go.
    boxes_by_y = sorted(boxes, key=lambda b: b.center[1])

    rows: list[list[BoundingBox]] = []
    current_row: list[BoundingBox] = [boxes_by_y[0]]

    for box in boxes_by_y[1:]:
        row_avg_height = sum(b.height for b in current_row) / len(current_row)
        row_avg_y = sum(b.center[1] for b in current_row) / len(current_row)

        if abs(box.center[1] - row_avg_y) <= row_tolerance * row_avg_height:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    rows.append(current_row)

    # Within each row, sort left-to-right; rows themselves are already in
    # top-to-bottom order since we swept boxes_by_y in that order.
    ordered: list[BoundingBox] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b.center[0]))

    return ordered
