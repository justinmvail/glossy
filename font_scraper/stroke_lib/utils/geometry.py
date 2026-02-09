"""Geometric utility functions."""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List, Optional

from ..domain.geometry import Point, BBox


def angle_between(d1: Point, d2: Point) -> float:
    """Angle between two direction vectors in radians."""
    dot = d1.dot(d2)
    return math.acos(max(-1.0, min(1.0, dot)))


def point_in_region(point: Tuple[int, int], region: int, bbox: Tuple[float, float, float, float]) -> bool:
    """Check if a point falls within a numpad region (1-9) of a bounding box.

    Legacy function for compatibility - use BBox.point_in_region for new code.

    Numpad layout:
    7 8 9  (top)
    4 5 6  (middle)
    1 2 3  (bottom)
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


def smooth_stroke(points: List[Tuple[float, float]], sigma: float = 2.0) -> List[Tuple[float, float]]:
    """Apply Gaussian smoothing to a stroke path."""
    if len(points) < 3:
        return points

    from scipy.ndimage import gaussian_filter1d

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    xs_smooth = gaussian_filter1d(xs, sigma=sigma, mode='nearest')
    ys_smooth = gaussian_filter1d(ys, sigma=sigma, mode='nearest')

    return list(zip(xs_smooth, ys_smooth))


def resample_path(path: List[Tuple[float, float]], num_points: int) -> List[Tuple[float, float]]:
    """Resample a path to have a specified number of evenly-spaced points."""
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

    for i in range(1, num_points - 1):
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


def constrain_to_mask(points: List[Tuple[float, float]], mask: np.ndarray) -> List[Tuple[float, float]]:
    """Constrain stroke points to stay within a binary mask."""
    from scipy.ndimage import distance_transform_edt

    if len(points) < 2:
        return points

    h, w = mask.shape
    dist = distance_transform_edt(mask)

    result = []
    for x, y in points:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h and mask[iy, ix]:
            result.append((x, y))
        else:
            # Find nearest point inside mask
            search_radius = 10
            best_dist = float('inf')
            best_pt = (x, y)

            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx]:
                        d = dx*dx + dy*dy
                        if d < best_dist:
                            best_dist = d
                            best_pt = (float(nx), float(ny))

            result.append(best_pt)

    return result


def generate_straight_line(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Generate points along a straight line using Bresenham's algorithm."""
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
