"""Core path tracing utilities for stroke pipeline.

This module provides path tracing and waypoint resolution functionality
extracted from MinimalStrokePipeline to improve code organization.

Key Functions:
    - numpad_to_pixel: Convert numpad region to pixel coordinates
    - find_nearest_skeleton: Find nearest skeleton pixel to a position
    - infer_direction: Infer stroke direction from waypoint positions
    - Waypoint resolution helpers
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scipy.spatial import cKDTree

# Angle range for classifying segments as vertical (degrees from horizontal)
VERTICAL_ANGLE_MIN = 60
VERTICAL_ANGLE_MAX = 120

# Waist region calculation ratios (fraction of glyph height)
WAIST_TOLERANCE_RATIO = 0.15
WAIST_MARGIN_RATIO = 0.05


def numpad_to_pixel(region: int, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Convert numpad region (1-9) to pixel coordinates within bounding box.

    Uses a numpad-style coordinate system:
        7 | 8 | 9   (top)
        4 | 5 | 6   (middle)
        1 | 2 | 3   (bottom)

    Args:
        region: Numpad region number (1-9).
        bbox: Bounding box as (x_min, y_min, x_max, y_max).

    Returns:
        Tuple of (x, y) pixel coordinates at the center of the region.

    Raises:
        ValueError: If region is not 1-9.
    """
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    # Define region centers as fractions
    x_fractions = {1: 0.15, 2: 0.5, 3: 0.85, 4: 0.15, 5: 0.5, 6: 0.85, 7: 0.15, 8: 0.5, 9: 0.85}
    y_fractions = {1: 0.85, 2: 0.85, 3: 0.85, 4: 0.5, 5: 0.5, 6: 0.5, 7: 0.15, 8: 0.15, 9: 0.15}

    if region not in x_fractions:
        raise ValueError(f"Invalid numpad region: {region}")

    x = x_min + width * x_fractions[region]
    y = y_min + height * y_fractions[region]

    return (x, y)


def find_nearest_skeleton(
    pos: tuple[float, float],
    skel_tree: 'cKDTree',
    skel_list: list[tuple[int, int]]
) -> tuple[int, int]:
    """Find nearest skeleton pixel to a given position.

    Args:
        pos: Query position as (x, y).
        skel_tree: cKDTree built from skeleton pixels.
        skel_list: List of skeleton pixel coordinates.

    Returns:
        Nearest skeleton pixel as (x, y) integer coordinates.
    """
    _, idx = skel_tree.query(pos)
    return skel_list[idx]


def infer_direction(current_region: int, next_region: int) -> str | None:
    """Infer stroke direction from current and next waypoint regions.

    Determines the direction a stroke segment should take based on
    the relative positions of two waypoint regions on the numpad grid.

    Args:
        current_region: Current numpad region (1-9).
        next_region: Next numpad region (1-9), or None if this is terminal.

    Returns:
        Direction string: 'up', 'down', 'left', 'right', or None if
        direction cannot be determined (e.g., same region).

    Example:
        >>> infer_direction(8, 2)  # Top-center to bottom-center
        'down'
        >>> infer_direction(4, 6)  # Middle-left to middle-right
        'right'
    """
    if next_region is None:
        return None

    # Get row/column for each region (numpad layout)
    def region_pos(r):
        row = (r - 1) // 3  # 0=bottom, 1=middle, 2=top
        col = (r - 1) % 3   # 0=left, 1=center, 2=right
        return row, col

    curr_row, curr_col = region_pos(current_region)
    next_row, next_col = region_pos(next_region)

    row_diff = next_row - curr_row
    col_diff = next_col - curr_col

    # Prioritize vertical movement for larger vertical differences
    if abs(row_diff) >= abs(col_diff):
        if row_diff > 0:
            return 'up'
        elif row_diff < 0:
            return 'down'

    # Horizontal movement
    if col_diff > 0:
        return 'right'
    elif col_diff < 0:
        return 'left'

    return None


def is_vertical_stroke(stroke_template: list) -> bool:
    """Check if a stroke template represents a vertical stroke.

    A stroke is considered vertical if:
    1. It has exactly 2 waypoints
    2. The angle between them is within the vertical range (60-120 degrees)

    Args:
        stroke_template: List of waypoint specifications.

    Returns:
        True if the stroke is vertical, False otherwise.
    """
    if len(stroke_template) != 2:
        return False

    # Extract regions from waypoints
    def get_region(wp):
        if isinstance(wp, int):
            return wp
        elif isinstance(wp, tuple) and len(wp) >= 1:
            return wp[0] if isinstance(wp[0], int) else None
        return None

    r1 = get_region(stroke_template[0])
    r2 = get_region(stroke_template[1])

    if r1 is None or r2 is None:
        return False

    # Check if regions are vertically aligned (same column)
    col1 = (r1 - 1) % 3
    col2 = (r2 - 1) % 3

    return col1 == col2


def extract_region_from_waypoint(wp) -> int | None:
    """Extract the numpad region from a waypoint specification.

    Waypoints can be specified in multiple formats:
    - Integer: Direct region number (e.g., 8)
    - Tuple: (region, modifier) where modifier is 'v' for vertex, etc.
    - Other: Returns None

    Args:
        wp: Waypoint specification in any supported format.

    Returns:
        The numpad region (1-9) or None if not determinable.

    Examples:
        >>> extract_region_from_waypoint(8)
        8
        >>> extract_region_from_waypoint((7, 'v'))
        7
    """
    if isinstance(wp, int):
        return wp
    elif isinstance(wp, tuple) and len(wp) >= 1:
        if isinstance(wp[0], int):
            return wp[0]
    return None


def compute_waist_bounds(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Compute the vertical bounds of the waist region.

    The waist region is the middle third of the glyph, with some tolerance.

    Args:
        bbox: Bounding box as (x_min, y_min, x_max, y_max).

    Returns:
        Tuple of (waist_min_y, waist_max_y).
    """
    _, y_min, _, y_max = bbox
    height = y_max - y_min
    mid_y = (y_min + y_max) / 2

    tolerance = height * WAIST_TOLERANCE_RATIO
    return (mid_y - tolerance, mid_y + tolerance)


def filter_skeleton_by_region(
    skel_list: list[tuple[int, int]],
    region: int,
    bbox: tuple[int, int, int, int],
    margin_ratio: float = 0.15
) -> list[tuple[int, int]]:
    """Filter skeleton pixels to those within a numpad region.

    Args:
        skel_list: List of skeleton pixel coordinates.
        region: Numpad region (1-9).
        bbox: Bounding box as (x_min, y_min, x_max, y_max).
        margin_ratio: Fraction of dimension to use as margin. Default 0.15.

    Returns:
        List of skeleton pixels within the region.
    """
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    # Define region bounds
    col = (region - 1) % 3  # 0=left, 1=center, 2=right
    row = (region - 1) // 3  # 0=bottom, 1=middle, 2=top

    # X bounds
    if col == 0:
        rx_min, rx_max = x_min, x_min + width * (0.33 + margin_ratio)
    elif col == 1:
        rx_min, rx_max = x_min + width * (0.33 - margin_ratio), x_min + width * (0.67 + margin_ratio)
    else:
        rx_min, rx_max = x_min + width * (0.67 - margin_ratio), x_max

    # Y bounds (inverted because y increases downward)
    if row == 0:  # bottom
        ry_min, ry_max = y_min + height * (0.67 - margin_ratio), y_max
    elif row == 1:  # middle
        ry_min, ry_max = y_min + height * (0.33 - margin_ratio), y_min + height * (0.67 + margin_ratio)
    else:  # top
        ry_min, ry_max = y_min, y_min + height * (0.33 + margin_ratio)

    return [(x, y) for x, y in skel_list if rx_min <= x <= rx_max and ry_min <= y <= ry_max]


def find_extremum_pixel(
    pixels: list[tuple[int, int]],
    direction: str | None,
    default_pos: tuple[float, float]
) -> tuple[int, int]:
    """Find the extremum pixel in a direction from a list.

    Args:
        pixels: List of pixel coordinates.
        direction: 'up', 'down', 'left', 'right', or None.
        default_pos: Default position to use if pixels is empty.

    Returns:
        The extremum pixel, or nearest to default_pos if direction is None.
    """
    if not pixels:
        return (int(default_pos[0]), int(default_pos[1]))

    if direction == 'up':
        return min(pixels, key=lambda p: p[1])
    elif direction == 'down':
        return max(pixels, key=lambda p: p[1])
    elif direction == 'left':
        return min(pixels, key=lambda p: p[0])
    elif direction == 'right':
        return max(pixels, key=lambda p: p[0])
    else:
        # Find nearest to default position
        return min(pixels, key=lambda p: (p[0] - default_pos[0])**2 + (p[1] - default_pos[1])**2)
