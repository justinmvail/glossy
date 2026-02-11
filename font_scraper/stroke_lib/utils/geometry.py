"""Geometric utility functions.

This module provides utility functions for geometric calculations used in
stroke editing. These functions supplement the methods on the domain objects
(Point, BBox, etc.) with additional operations.

The module provides the following functions:
    angle_between: Compute angle between two direction vectors.
    point_in_region: Check if a point is in a numpad region (legacy).
    smooth_stroke: Apply Gaussian smoothing to stroke paths.
    resample_path: Resample a path to have evenly-spaced points.
    constrain_to_mask: Constrain stroke points to stay within a mask.
    generate_straight_line: Generate points along a line using Bresenham.

Example usage:
    Vector angle calculation::

        from stroke_lib.utils.geometry import angle_between
        from stroke_lib.domain import Point

        # Perpendicular vectors
        right = Point(1, 0)
        down = Point(0, 1)
        angle = angle_between(right, down)  # Returns pi/2

    Path smoothing::

        from stroke_lib.utils.geometry import smooth_stroke, resample_path

        points = [(0, 0), (10, 5), (20, 0), (30, 5)]
        smoothed = smooth_stroke(points, sigma=2.0)
        resampled = resample_path(smoothed, num_points=10)
"""

from __future__ import annotations

import math

import numpy as np

from ..domain.geometry import Point


def angle_between(d1: Point, d2: Point) -> float:
    """Angle between two direction vectors in radians.

    Computes the angle between two direction vectors using the dot product
    formula. Both vectors should be normalized for accurate results, though
    unnormalized vectors will work with decreased precision.

    Args:
        d1: First direction vector as a Point.
        d2: Second direction vector as a Point.

    Returns:
        Angle in radians, ranging from 0 (parallel) to pi (opposite).
        The result is clamped to handle numerical precision issues.

    Example:
        >>> from stroke_lib.domain import Point
        >>> right = Point(1, 0)
        >>> up = Point(0, -1)
        >>> angle_between(right, up)  # Returns approximately pi/2
        1.5707963267948966
    """
    dot = d1.dot(d2)
    return math.acos(max(-1.0, min(1.0, dot)))


def point_in_region(point: tuple[int, int], region: int, bbox: tuple[float, float, float, float]) -> bool:
    """Check if a point falls within a numpad region (1-9) of a bounding box.

    Legacy function for compatibility - use BBox.point_in_region for new code.

    The numpad layout divides the bounding box into a 3x3 grid:
        7 8 9  (top row)
        4 5 6  (middle row)
        1 2 3  (bottom row)

    Args:
        point: Pixel coordinates as (x, y) tuple.
        region: Numpad region number 1-9.
        bbox: Bounding box as (x_min, y_min, x_max, y_max) tuple.

    Returns:
        True if the point falls within the specified region of the
        bounding box. Returns False if the bounding box is too small
        (width or height less than 1 pixel).

    Example:
        >>> point_in_region((50, 50), 5, (0, 0, 100, 100))
        True
        >>> point_in_region((10, 10), 7, (0, 0, 100, 100))
        True
    """
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min

    if w < 1 or h < 1:
        return False

    col = (region - 1) % 3
    row = 2 - (region - 1) // 3

    x_start = x_min + col * w / 3
    x_end = x_min + (col + 1) * w / 3
    y_start = y_min + row * h / 3
    y_end = y_min + (row + 1) * h / 3

    return x_start <= x < x_end and y_start <= y < y_end


def smooth_stroke(points: list[tuple[float, float]], sigma: float = 2.0) -> list[tuple[float, float]]:
    """Apply Gaussian smoothing to a stroke path.

    Smooths the x and y coordinates of the path independently using
    a Gaussian filter. This reduces noise and jitter in stroke paths
    while preserving the overall shape.

    Args:
        points: List of (x, y) coordinate tuples defining the path.
        sigma: Standard deviation of the Gaussian kernel. Higher values
            produce more smoothing. Default is 2.0.

    Returns:
        List of smoothed (x, y) coordinate tuples. If the input has
        fewer than 3 points, returns the input unchanged.

    Example:
        >>> points = [(0, 0), (10, 5), (20, 0), (30, 5), (40, 0)]
        >>> smoothed = smooth_stroke(points, sigma=1.0)
    """
    if len(points) < 3:
        return points

    from scipy.ndimage import gaussian_filter1d

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    xs_smooth = gaussian_filter1d(xs, sigma=sigma, mode='nearest')
    ys_smooth = gaussian_filter1d(ys, sigma=sigma, mode='nearest')

    return list(zip(xs_smooth, ys_smooth))


def resample_path(path: list[tuple[float, float]], num_points: int) -> list[tuple[float, float]]:
    """Resample a path to have a specified number of evenly-spaced points.

    Creates a new path with points distributed at equal arc-length
    intervals along the original path. This is useful for normalizing
    stroke density or preparing paths for comparison.

    Args:
        path: List of (x, y) coordinate tuples defining the path.
        num_points: Desired number of points in the output path.

    Returns:
        List of (x, y) coordinate tuples with evenly-spaced points.
        Always includes the original start and end points. Returns
        the input unchanged if it has fewer than 2 points or if
        num_points is less than 2.

    Example:
        >>> path = [(0, 0), (100, 0), (100, 100)]
        >>> resampled = resample_path(path, num_points=5)
        >>> len(resampled)
        5
    """
    if len(path) < 2 or num_points < 2:
        return path

    # Calculate cumulative arc length
    distances = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        distances.append(distances[-1] + math.sqrt(dx*dx + dy*dy))

    total_length = distances[-1]
    if total_length < 0.001:
        return path

    # Resample at even intervals
    result = [path[0]]
    target_dist = 0.0
    step = total_length / (num_points - 1)

    for _i in range(1, num_points - 1):
        target_dist += step
        # Find segment containing target distance
        for j in range(1, len(distances)):
            if distances[j] >= target_dist:
                # Interpolate within segment
                t = (target_dist - distances[j-1]) / (distances[j] - distances[j-1])
                x = path[j-1][0] + t * (path[j][0] - path[j-1][0])
                y = path[j-1][1] + t * (path[j][1] - path[j-1][1])
                result.append((x, y))
                break

    result.append(path[-1])
    return result


def constrain_to_mask(points: list[tuple[float, float]], mask: np.ndarray) -> list[tuple[float, float]]:
    """Constrain stroke points to stay within a binary mask.

    For any point outside the mask, snaps it to the nearest inside pixel
    using scipy's distance transform with index mapping for efficiency.
    This is useful for ensuring strokes stay within glyph boundaries.

    Args:
        points: List of (x, y) coordinate tuples to constrain.
        mask: Binary numpy array where True/non-zero values indicate
            valid regions.

    Returns:
        List of (x, y) coordinate tuples with points constrained to
        the mask. Points already inside the mask are unchanged.
        Returns input unchanged if fewer than 2 points.

    Example:
        >>> import numpy as np
        >>> mask = np.zeros((100, 100), dtype=bool)
        >>> mask[20:80, 20:80] = True  # Square region
        >>> points = [(10, 50), (50, 50), (90, 50)]
        >>> constrained = constrain_to_mask(points, mask)
        >>> # First and last points will be snapped to mask edge
    """
    from scipy.ndimage import distance_transform_edt

    if len(points) < 2:
        return points

    h, w = mask.shape
    # Compute distance transform on inverted mask to find nearest inside pixels
    _, indices = distance_transform_edt(~mask, return_indices=True)

    result = []
    for x, y in points:
        ix = int(round(min(max(x, 0), w - 1)))
        iy = int(round(min(max(y, 0), h - 1)))
        if mask[iy, ix]:
            result.append((x, y))
        else:
            # Use distance transform indices to find nearest inside pixel
            ny = float(indices[0, iy, ix])
            nx = float(indices[1, iy, ix])
            result.append((nx, ny))

    return result


# ---------------------------------------------------------------------------
# Direction and distance utilities for path tracing
# ---------------------------------------------------------------------------

def point_distance_squared(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Compute squared Euclidean distance between two points.

    Using squared distance avoids the sqrt computation, which is useful
    when comparing distances (the ordering is preserved).

    Args:
        p1: First point as (x, y).
        p2: Second point as (x, y).

    Returns:
        Squared distance between the points.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy


def point_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Compute Euclidean distance between two points.

    Args:
        p1: First point as (x, y).
        p2: Second point as (x, y).

    Returns:
        Distance between the points.
    """
    return point_distance_squared(p1, p2) ** 0.5


def compute_direction_vector(p1: tuple[float, float], p2: tuple[float, float],
                              normalize: bool = True) -> tuple[float, float]:
    """Compute direction vector from p1 to p2.

    Args:
        p1: Start point as (x, y).
        p2: End point as (x, y).
        normalize: If True, return unit vector. If False, return raw delta.

    Returns:
        Direction vector (dx, dy). If normalize=True and length is near zero,
        returns (0, 0).
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if not normalize:
        return (dx, dy)
    length = (dx * dx + dy * dy) ** 0.5
    if length < 0.01:
        return (0.0, 0.0)
    return (dx / length, dy / length)


def compute_dot_product(d1: tuple[float, float], d2: tuple[float, float]) -> float:
    """Compute dot product of two direction vectors.

    Args:
        d1: First direction vector (dx, dy).
        d2: Second direction vector (dx, dy).

    Returns:
        Dot product value. For unit vectors, this is cos(angle).
    """
    return d1[0] * d2[0] + d1[1] * d2[1]


def pick_straightest_neighbor(current: tuple[int, int], path: list,
                               candidates: list, n_lookback: int = 4) -> tuple:
    """Pick the candidate that continues most straight from current direction.

    This is a shared utility for path tracing algorithms that need to choose
    the straightest continuation when multiple neighbors are available.

    Args:
        current: Current position (x, y).
        path: Path traced so far (list of (x, y) tuples).
        candidates: List of (neighbor, edge) tuples to choose from.
        n_lookback: Number of path points to look back for direction.

    Returns:
        Tuple of (next_pt, next_edge) for the straightest continuation.
    """
    if len(candidates) == 1:
        return candidates[0]

    n_look = min(n_lookback, len(path))
    dir_in = compute_direction_vector(path[-n_look], current, normalize=True)

    best_dot = -2.0
    next_pt, next_edge = candidates[0]
    for n, e in candidates:
        dir_out = compute_direction_vector(current, n, normalize=True)
        dot = compute_dot_product(dir_in, dir_out)
        if dot > best_dot:
            best_dot = dot
            next_pt, next_edge = n, e

    return next_pt, next_edge


def infer_direction_from_regions(current_region: int, next_region: int) -> str | None:
    """Infer the direction of travel based on numpad region transition.

    Determines the cardinal direction (up, down, left, right) when moving
    from one numpad region to another.

    Args:
        current_region: Starting numpad region (1-9).
        next_region: Destination numpad region (1-9).

    Returns:
        Direction string ('down', 'up', 'left', 'right') for clear
        vertical/horizontal moves, or None for diagonal moves.

    Note:
        Numpad layout reference:
            7 | 8 | 9   (row 0)
            4 | 5 | 6   (row 1)
            1 | 2 | 3   (row 2)

        Diagonal moves (e.g., 7->3, 1->9) return None because skeleton
        path tracing uses directional bias to choose branches at
        junctions. Cardinal directions map cleanly to segment angles
        (down=90 deg, up=-90 deg, etc.), but diagonals are ambiguous.
    """
    # Get row and column for each region (0-indexed from top-left)
    def region_to_row_col(r):
        row = 2 - (r - 1) // 3  # 7,8,9 -> row 0; 4,5,6 -> row 1; 1,2,3 -> row 2
        col = (r - 1) % 3       # 7,4,1 -> col 0; 8,5,2 -> col 1; 9,6,3 -> col 2
        return row, col

    r1, c1 = region_to_row_col(current_region)
    r2, c2 = region_to_row_col(next_region)

    dr = r2 - r1  # positive = down
    dc = c2 - c1  # positive = right

    # Only return direction for clear vertical/horizontal moves
    if abs(dr) > abs(dc) and dr != 0:
        return 'down' if dr > 0 else 'up'
    elif abs(dc) > abs(dr) and dc != 0:
        return 'right' if dc > 0 else 'left'

    return None


def generate_straight_line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Generate points along a straight line using Bresenham's algorithm.

    Produces a list of integer pixel coordinates along the line from
    start to end. The Bresenham algorithm ensures that all pixels are
    8-connected (no gaps in the line).

    Args:
        start: Starting point as (x, y) integer tuple.
        end: Ending point as (x, y) integer tuple.

    Returns:
        List of (x, y) integer tuples representing pixels along the line,
        including both start and end points.

    Example:
        >>> line = generate_straight_line((0, 0), (5, 3))
        >>> print(line)
        [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2), (5, 3)]
    """
    x0, y0 = start
    x1, y1 = end

    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points
