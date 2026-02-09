"""Stroke utility functions.

This module contains miscellaneous utility functions for stroke processing
that don't fit neatly into other modules.
"""

import os
import re

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d

# Base directory for relative path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_font_path(font_path: str) -> str:
    """Resolve a font path to an absolute path.

    If path is relative, resolve it relative to BASE_DIR.
    """
    if os.path.isabs(font_path):
        return font_path
    return os.path.join(BASE_DIR, font_path)


def smooth_stroke(points: list[tuple], sigma: float = 2.0) -> list[tuple]:
    """Gaussian smooth a stroke's x and y coordinates independently."""
    if len(points) < 3:
        return list(points)

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)

    xs_smooth = gaussian_filter1d(xs, sigma=sigma, mode='nearest')
    ys_smooth = gaussian_filter1d(ys, sigma=sigma, mode='nearest')

    return [(float(x), float(y)) for x, y in zip(xs_smooth, ys_smooth)]


def constrain_to_mask(points: list[tuple], mask: np.ndarray) -> list[tuple]:
    """Constrain points to stay inside the glyph mask.

    Uses distance transform to snap outside points to nearest inside pixel.
    """
    if len(points) == 0:
        return []

    h, w = mask.shape
    _, snap_indices = distance_transform_edt(~mask, return_indices=True)

    result = []
    for x, y in points:
        ix = int(round(min(max(x, 0), w - 1)))
        iy = int(round(min(max(y, 0), h - 1)))

        if mask[iy, ix]:
            result.append((x, y))
        else:
            ny = float(snap_indices[0, iy, ix])
            nx = float(snap_indices[1, iy, ix])
            result.append((nx, ny))

    return result


def snap_inside(pos: tuple, mask: np.ndarray, snap_indices: np.ndarray) -> tuple:
    """Snap a position to the nearest mask pixel if outside."""
    h, w = mask.shape
    ix = int(round(min(max(pos[0], 0), w - 1)))
    iy = int(round(min(max(pos[1], 0), h - 1)))
    if mask[iy, ix]:
        return pos
    ny = float(snap_indices[0, iy, ix])
    nx = float(snap_indices[1, iy, ix])
    return (nx, ny)


def snap_deep_inside(pos: tuple, centroid: tuple, dist_in: np.ndarray,
                     mask: np.ndarray, snap_indices: np.ndarray) -> tuple:
    """Snap a position to be well inside the mask.

    Cast ray from pos toward centroid, find the point with maximum
    distance-from-edge (deepest inside the glyph).
    """
    h, w = mask.shape
    ix = int(round(min(max(pos[0], 0), w - 1)))
    iy = int(round(min(max(pos[1], 0), h - 1)))
    if mask[iy, ix] and dist_in[iy, ix] >= 5:
        return pos

    dx = centroid[0] - pos[0]
    dy = centroid[1] - pos[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1:
        return snap_inside(pos, mask, snap_indices)

    best_pos = snap_inside(pos, mask, snap_indices)
    best_depth = 0
    steps = int(length)
    for s in range(steps + 1):
        t = s / max(steps, 1)
        x = pos[0] + dx * t
        y = pos[1] + dy * t
        jx = int(round(min(max(x, 0), w - 1)))
        jy = int(round(min(max(y, 0), h - 1)))
        if mask[jy, jx] and dist_in[jy, jx] > best_depth:
            best_depth = dist_in[jy, jx]
            best_pos = (x, y)
        if best_depth > 5 and dist_in[jy, jx] < best_depth * 0.5:
            break
    return best_pos


def snap_to_glyph_edge(pos: tuple, centroid: tuple, mask: np.ndarray) -> tuple | None:
    """Snap a termination point to the nearest mask pixel."""
    h, w = mask.shape
    ix = int(round(min(max(pos[0], 0), w - 1)))
    iy = int(round(min(max(pos[1], 0), h - 1)))
    if mask[iy, ix]:
        return pos
    _dist_out, indices = distance_transform_edt(~mask, return_indices=True)
    ny = float(indices[0, iy, ix])
    nx = float(indices[1, iy, ix])
    nix, niy = int(round(nx)), int(round(ny))
    if 0 <= nix < w and 0 <= niy < h and mask[niy, nix]:
        return (nx, ny)
    return None


def parse_waypoint(wp) -> tuple[int, str]:
    """Parse a waypoint into (region_int, kind).

    Returns:
        (region, kind) where region is 1-9 and kind is 'terminal', 'vertex', or 'curve'.
    """
    if isinstance(wp, int):
        return (wp, 'terminal')
    m = re.match(r'^v\((\d)\)$', str(wp))
    if m:
        return (int(m.group(1)), 'vertex')
    m = re.match(r'^c\((\d)\)$', str(wp))
    if m:
        return (int(m.group(1)), 'curve')
    raise ValueError(f"Unknown waypoint format: {wp}")


def numpad_to_pixel(region: int, glyph_bbox: tuple) -> tuple[float, float]:
    """Map a numpad region (1-9) to pixel coordinates within the glyph bounding box."""
    from stroke_templates import NUMPAD_POS
    frac_x, frac_y = NUMPAD_POS[region]
    x_min, y_min, x_max, y_max = glyph_bbox
    return (x_min + frac_x * (x_max - x_min),
            y_min + frac_y * (y_max - y_min))


def linear_segment(p0: tuple, p1: tuple, step: float = 2.0) -> list[tuple]:
    """Generate evenly-spaced points along a line from p0 to p1."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dist = (dx * dx + dy * dy) ** 0.5
    n = max(2, int(round(dist / step)))
    return [(p0[0] + dx * i / (n - 1), p0[1] + dy * i / (n - 1)) for i in range(n)]


def catmull_rom_point(p0: tuple, p1: tuple, p2: tuple, p3: tuple,
                      t: float, alpha: float = 0.5) -> tuple:
    """Evaluate a single point on a Catmull-Rom spline segment."""
    def tj(ti, pi, pj):
        dx = pj[0] - pi[0]
        dy = pj[1] - pi[1]
        d = (dx * dx + dy * dy) ** 0.5
        return ti + max(d ** alpha, 1e-6)

    t0 = 0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)

    u = t1 + t * (t2 - t1)

    def lerp(a, b, ta, tb, u_):
        f = (u_ - ta) / max(tb - ta, 1e-10)
        return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))

    a1 = lerp(p0, p1, t0, t1, u)
    a2 = lerp(p1, p2, t1, t2, u)
    a3 = lerp(p2, p3, t2, t3, u)
    b1 = lerp(a1, a2, t0, t2, u)
    b2 = lerp(a2, a3, t1, t3, u)
    c = lerp(b1, b2, t1, t2, u)
    return c


def catmull_rom_segment(p_prev: tuple, p0: tuple, p1: tuple, p_next: tuple,
                        step: float = 2.0) -> list[tuple]:
    """Generate evenly-spaced points along a Catmull-Rom segment from p0 to p1."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dist = (dx * dx + dy * dy) ** 0.5
    n = max(2, int(round(dist / step)))
    return [catmull_rom_point(p_prev, p0, p1, p_next, i / (n - 1)) for i in range(n)]


def point_in_region(point: tuple[int, int], region: int,
                    bbox: tuple[int, int, int, int]) -> bool:
    """Check if a point falls within a numpad region.

    Region 1-9 maps to a 3x3 grid:
      7 8 9
      4 5 6
      1 2 3
    """
    x, y = point
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0

    # Region column (0=left, 1=center, 2=right)
    col = (region - 1) % 3
    # Region row (0=bottom, 1=middle, 2=top) - flip for screen coordinates
    row = 2 - (region - 1) // 3

    # Boundaries
    x_min = x0 + col * w / 3
    x_max = x0 + (col + 1) * w / 3
    y_min = y0 + row * h / 3
    y_max = y0 + (row + 1) * h / 3

    return x_min <= x <= x_max and y_min <= y <= y_max


def build_guide_path(waypoints_raw: list, glyph_bbox: tuple, mask: np.ndarray,
                     skel_features: dict | None = None) -> list[tuple]:
    """Build a guide path from parsed waypoints.

    waypoints_raw: list of raw waypoint values (int, 'v(n)', 'c(n)')
    skel_features: optional dict mapping region to list of skeleton feature positions.
    Returns list of (x, y) points sampled along the guide path.
    """
    parsed = [parse_waypoint(wp) for wp in waypoints_raw]
    n_wp = len(parsed)
    if n_wp < 2:
        return []

    # Compute glyph centroid from mask
    rows, cols = np.where(mask)
    if len(rows) == 0:
        centroid = ((glyph_bbox[0] + glyph_bbox[2]) / 2,
                    (glyph_bbox[1] + glyph_bbox[3]) / 2)
    else:
        centroid = (float(cols.mean()), float(rows.mean()))

    # Pre-compute distance fields for snapping
    h, w = mask.shape
    _dist_out, snap_indices = distance_transform_edt(~mask, return_indices=True)
    dist_in = distance_transform_edt(mask)

    def snap_to_skeleton_region(region):
        if skel_features is None:
            return None
        candidates = skel_features.get(region, [])
        if candidates:
            return candidates[0]
        target = numpad_to_pixel(region, glyph_bbox)
        all_skel = skel_features.get('all_skel', [])
        if not all_skel:
            return None
        best = min(all_skel, key=lambda p: (p[0]-target[0])**2 + (p[1]-target[1])**2)
        return best

    # Map waypoints to pixel positions
    positions = []
    for region, kind in parsed:
        skel_pos = snap_to_skeleton_region(region)
        if skel_pos is not None:
            pos = (float(skel_pos[0]), float(skel_pos[1]))
        elif kind == 'terminal':
            pos = snap_to_glyph_edge(numpad_to_pixel(region, glyph_bbox), centroid, mask)
            if pos is None:
                return []
        else:
            pos = snap_deep_inside(numpad_to_pixel(region, glyph_bbox),
                                   centroid, dist_in, mask, snap_indices)
            ix = int(round(min(max(pos[0], 0), w - 1)))
            iy = int(round(min(max(pos[1], 0), h - 1)))
            if not mask[iy, ix]:
                return []
        positions.append(pos)

    # Build path segments
    all_points = []
    for i in range(n_wp - 1):
        seg = linear_segment(positions[i], positions[i + 1], step=2.0)
        if all_points and seg:
            seg = seg[1:]
        all_points.extend(seg)

    # Constrain all guide points to be inside the mask
    constrained = []
    for x, y in all_points:
        ix = int(round(min(max(x, 0), w - 1)))
        iy = int(round(min(max(y, 0), h - 1)))
        if mask[iy, ix]:
            constrained.append((x, y))
        else:
            constrained.append(snap_inside((x, y), mask, snap_indices))

    return constrained


def find_skeleton_waypoints(mask: np.ndarray, glyph_bbox: tuple) -> dict | None:
    """Find skeleton endpoints and junctions as candidate waypoint positions.

    Returns dict mapping numpad region (1-9) to list of (x, y) skeleton
    feature positions in that region.
    """
    from collections import defaultdict

    from skimage.morphology import skeletonize

    skel = skeletonize(mask)
    ys, xs = np.where(skel)
    if len(xs) == 0:
        return None

    skel_set = set(zip(xs.tolist(), ys.tolist()))

    # Build adjacency
    adj = defaultdict(list)
    for (x, y) in skel_set:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                n = (x + dx, y + dy)
                if n in skel_set:
                    adj[(x, y)].append(n)

    skel_list = list(skel_set)
    dist_in = distance_transform_edt(mask)
    max_dist = float(dist_in.max()) if dist_in.max() > 0 else 1.0

    x_min, y_min, x_max, y_max = glyph_bbox
    bbox_diag = max(((x_max - x_min) ** 2 + (y_max - y_min) ** 2) ** 0.5, 1.0)

    region_features = {}
    for r in range(1, 10):
        rc = numpad_to_pixel(r, glyph_bbox)

        def score(p, rc=rc):
            dx = p[0] - rc[0]
            dy = p[1] - rc[1]
            proximity = (dx * dx + dy * dy) ** 0.5 / bbox_diag
            depth = dist_in[p[1], p[0]] / max_dist
            return proximity - 0.3 * depth

        best = min(skel_list, key=score)
        region_features[r] = [best]

    region_features['all_skel'] = skel_list
    return region_features


# Aliases for backwards compatibility
_smooth_stroke = smooth_stroke
_constrain_to_mask = constrain_to_mask
_snap_inside = snap_inside
_snap_deep_inside = snap_deep_inside
_snap_to_glyph_edge = snap_to_glyph_edge
_parse_waypoint = parse_waypoint
_numpad_to_pixel = numpad_to_pixel
_linear_segment = linear_segment
_catmull_rom_point = catmull_rom_point
_catmull_rom_segment = catmull_rom_segment
_build_guide_path = build_guide_path
_find_skeleton_waypoints = find_skeleton_waypoints
