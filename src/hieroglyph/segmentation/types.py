"""Shared types for segmentation — kept separate from base.py so both
base.py and classical.py (and any future detector implementation) can
import BoundingBox without a circular import."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoundingBox:
    """A rectangle in image pixel coordinates: (x, y) is the top-left corner."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> int:
        return self.width * self.height
