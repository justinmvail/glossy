"""Utility functions for stroke editing.

This module provides utility functions for geometric operations and glyph
rendering. These utilities are used throughout the stroke editing package
and are also exported for use by external code.

The module exports the following functions:

Geometry utilities:
    angle_between: Compute angle between two direction vectors.
    point_in_region: Check if a point falls within a numpad region.

Rendering utilities:
    render_glyph_mask: Render a character as a binary mask.
    get_glyph_bbox: Get bounding box of non-zero pixels in a mask.

Example usage:
    Geometric calculations::

        from stroke_lib.utils import angle_between, point_in_region
        from stroke_lib.domain import Point

        # Angle between two directions
        d1 = Point(1, 0)  # Right
        d2 = Point(0, 1)  # Down
        angle = angle_between(d1, d2)  # pi/2 radians

        # Check point in numpad region
        is_in_region = point_in_region((50, 50), 5, (0, 0, 100, 100))

    Rendering glyphs::

        from stroke_lib.utils import render_glyph_mask, get_glyph_bbox

        mask = render_glyph_mask('/fonts/arial.ttf', 'A', canvas_size=224)
        if mask is not None:
            bbox = get_glyph_bbox(mask)
            print(f"Glyph bounds: {bbox.to_tuple()}")
"""

from .geometry import (
    angle_between,
    constrain_to_mask,
    generate_straight_line,
    point_in_region,
    resample_path,
    smooth_stroke,
)
from .rendering import get_glyph_bbox, render_glyph_mask

__all__ = [
    'angle_between', 'point_in_region',
    'smooth_stroke', 'resample_path', 'constrain_to_mask', 'generate_straight_line',
    'render_glyph_mask', 'get_glyph_bbox',
]
