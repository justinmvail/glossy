"""Stroke utility functions for glyph processing.

This module provides miscellaneous utility functions for stroke processing
operations including path smoothing, point snapping to glyph masks, waypoint
parsing, coordinate transformations, and spline interpolation. These utilities
support the stroke tracing and rendering pipeline.

Key functionality:
    - Gaussian smoothing of stroke paths
    - Snapping points to stay within glyph boundaries
    - Parsing numpad-style waypoint specifications
    - Catmull-Rom spline interpolation for smooth curves
    - Building guide paths from waypoint sequences
    - Finding skeleton features for waypoint targeting

Typical usage:
    from stroke_utils import smooth_stroke, build_guide_path

    # Smooth a noisy stroke path
    smoothed = smooth_stroke(raw_points, sigma=2.0)

    # Build a guide path through waypoints
    path = build_guide_path([7, 'v(5)', 3], bbox, mask)
"""

import os
import re

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d

# Base directory for relative path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_font_path(font_path: str) -> str:
    """Resolve a font path to an absolute path.

    If the provided path is relative, it is resolved relative to the
    module's base directory (BASE_DIR).

    Args:
        font_path: The font file path, either absolute or relative.

    Returns:
        The absolute path to the font file.
    """
    if os.path.isabs(font_path):
        return font_path
    return os.path.join(BASE_DIR, font_path)


def smooth_stroke(points: list[tuple], sigma: float = 2.0) -> list[tuple]:
    """Apply Gaussian smoothing to a stroke's coordinates.

    Smooths x and y coordinates independently using a 1D Gaussian filter
    to reduce noise while preserving the overall stroke shape.

    Args:
        points: List of (x, y) coordinate tuples representing the stroke path.
        sigma: Standard deviation for the Gaussian kernel. Higher values
            produce smoother results. Defaults to 2.0.

    Returns:
        A new list of (x, y) tuples with smoothed coordinates.

    Notes:
        - Returns the original points unchanged if fewer than 3 points
          are provided (insufficient for meaningful smoothing).
        - Uses 'nearest' mode at boundaries to avoid edge artifacts.
    """
    if len(points) < 3:
        return list(points)

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)

    xs_smooth = gaussian_filter1d(xs, sigma=sigma, mode='nearest')
    ys_smooth = gaussian_filter1d(ys, sigma=sigma, mode='nearest')

    return [(float(x), float(y)) for x, y in zip(xs_smooth, ys_smooth)]


def constrain_to_mask(points: list[tuple], mask: np.ndarray) -> list[tuple]:
    """Constrain points to stay inside the glyph mask.

    Points that fall outside the mask are snapped to the nearest pixel
    that is inside the mask, using a distance transform for efficient
    nearest-neighbor lookup.

    Args:
        points: List of (x, y) coordinate tuples to constrain.
        mask: Binary numpy array where True indicates glyph pixels.
            Shape should be (height, width).

    Returns:
        A new list of (x, y) tuples with all points inside the mask.
        Points already inside remain unchanged; outside points are
        moved to the nearest inside pixel.

    Notes:
        - Returns an empty list if no points are provided.
        - Coordinates are clamped to valid array indices before lookup.
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
    """Snap a position to the nearest mask pixel if outside.

    Uses precomputed snap indices from a distance transform for efficient
    nearest-neighbor lookup.

    Args:
        pos: The (x, y) position to potentially snap.
        mask: Binary numpy array where True indicates glyph pixels.
        snap_indices: Precomputed indices array from distance_transform_edt
            with return_indices=True. Shape is (2, height, width).

    Returns:
        The original position if inside the mask, or the coordinates of
        the nearest inside pixel if outside.
    """
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

    Casts a ray from the given position toward the glyph centroid and finds
    the point along the ray with maximum distance from the glyph edge. This
    ensures waypoints are placed deep within the glyph rather than near edges.

    Args:
        pos: The (x, y) starting position.
        centroid: The (x, y) centroid of the glyph to cast the ray toward.
        dist_in: Distance transform of the mask (distance from each pixel
            to the nearest edge). Larger values are deeper inside.
        mask: Binary numpy array where True indicates glyph pixels.
        snap_indices: Precomputed snap indices from distance_transform_edt.

    Returns:
        A position (x, y) that is well inside the glyph. If the original
        position is already at least 5 pixels from the edge, it is returned
        unchanged. Otherwise, the deepest point along the ray to the centroid
        is returned.

    Notes:
        - The search stops early once it finds a point deeper than 5 pixels
          and the depth starts decreasing, to avoid overshooting.
        - Falls back to snap_inside if the ray length is less than 1 pixel.
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
    """Snap a termination point to the nearest glyph edge pixel.

    Used for positioning stroke endpoints at the boundary of the glyph.

    Args:
        pos: The (x, y) position to snap.
        centroid: The (x, y) centroid of the glyph (currently unused but
            available for future direction-aware snapping).
        mask: Binary numpy array where True indicates glyph pixels.

    Returns:
        The snapped (x, y) position on the glyph edge, or None if snapping
        fails (e.g., the nearest pixel is outside the image bounds).

    Notes:
        - If the position is already inside the mask, it is returned unchanged.
        - Uses distance transform for efficient nearest-neighbor lookup.
    """
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
    """Parse a waypoint specification into region and kind.

    Waypoints can be specified as:
        - Integer (1-9): Terminal point in that numpad region
        - 'v(n)': Vertex point in region n
        - 'c(n)': Curve point in region n

    Args:
        wp: The waypoint specification. Can be an integer or a string
            matching 'v(digit)' or 'c(digit)'.

    Returns:
        A tuple of (region, kind) where:
            - region is an integer 1-9 indicating the numpad region
            - kind is one of 'terminal', 'vertex', or 'curve'

    Raises:
        ValueError: If the waypoint format is not recognized.
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
    """Map a numpad region (1-9) to pixel coordinates within a bounding box.

    Uses the numpad layout where regions are arranged as:
        7 8 9  (top)
        4 5 6  (middle)
        1 2 3  (bottom)

    Args:
        region: Integer 1-9 indicating the numpad region.
        glyph_bbox: Bounding box as (x_min, y_min, x_max, y_max).

    Returns:
        The (x, y) pixel coordinates corresponding to the center of
        the specified region within the bounding box.

    Notes:
        - Region positions are defined in stroke_templates.NUMPAD_POS
          as fractional coordinates (0-1 range).
    """
    from stroke_templates import NUMPAD_POS
    frac_x, frac_y = NUMPAD_POS[region]
    x_min, y_min, x_max, y_max = glyph_bbox
    return (x_min + frac_x * (x_max - x_min),
            y_min + frac_y * (y_max - y_min))


def linear_segment(p0: tuple, p1: tuple, step: float = 2.0) -> list[tuple]:
    """Generate evenly-spaced points along a line segment.

    Args:
        p0: Starting point as (x, y).
        p1: Ending point as (x, y).
        step: Approximate distance between consecutive points. Defaults to 2.0.

    Returns:
        A list of (x, y) tuples evenly distributed along the line from
        p0 to p1, inclusive of both endpoints.

    Notes:
        - Always returns at least 2 points (the endpoints).
        - The actual step size may be slightly adjusted to ensure even
          distribution along the entire segment.
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dist = (dx * dx + dy * dy) ** 0.5
    n = max(2, int(round(dist / step)))
    return [(p0[0] + dx * i / (n - 1), p0[1] + dy * i / (n - 1)) for i in range(n)]


def catmull_rom_point(p0: tuple, p1: tuple, p2: tuple, p3: tuple,
                      t: float, alpha: float = 0.5) -> tuple:
    """Evaluate a single point on a Catmull-Rom spline segment.

    Computes a point on the spline segment between p1 and p2, using p0 and p3
    as control points for tangent computation.

    Args:
        p0: First control point (before the segment).
        p1: Start of the spline segment.
        p2: End of the spline segment.
        p3: Last control point (after the segment).
        t: Parameter value in [0, 1] where 0 = p1 and 1 = p2.
        alpha: Tension parameter controlling spline type. Defaults to 0.5
            (centripetal Catmull-Rom). Use 0.0 for uniform, 1.0 for chordal.

    Returns:
        The (x, y) coordinates of the interpolated point.

    Notes:
        - Uses the centripetal parameterization by default, which avoids
          cusps and self-intersections that can occur with uniform splines.
        - The alpha parameter controls the "tightness" of the curve.
    """
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
    """Generate evenly-spaced points along a Catmull-Rom spline segment.

    Samples the spline between p0 and p1 at regular intervals.

    Args:
        p_prev: Control point before the segment (for tangent at p0).
        p0: Start of the spline segment.
        p1: End of the spline segment.
        p_next: Control point after the segment (for tangent at p1).
        step: Approximate distance between consecutive points. Defaults to 2.0.

    Returns:
        A list of (x, y) tuples sampled along the spline from p0 to p1.

    Notes:
        - The number of samples is based on the straight-line distance
          between p0 and p1, divided by the step size.
        - Always returns at least 2 points.
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dist = (dx * dx + dy * dy) ** 0.5
    n = max(2, int(round(dist / step)))
    return [catmull_rom_point(p_prev, p0, p1, p_next, i / (n - 1)) for i in range(n)]


def point_in_region(point: tuple[int, int], region: int,
                    bbox: tuple[int, int, int, int]) -> bool:
    """Check if a point falls within a numpad region.

    The bounding box is divided into a 3x3 grid corresponding to numpad
    regions arranged as:
        7 8 9  (top)
        4 5 6  (middle)
        1 2 3  (bottom)

    Args:
        point: The (x, y) coordinates to check.
        region: The numpad region (1-9) to test against.
        bbox: The bounding box as (x0, y0, x1, y1).

    Returns:
        True if the point lies within the specified region, False otherwise.

    Notes:
        - Region boundaries are inclusive on all sides.
        - Uses screen coordinates where y increases downward.
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
    """Build a guide path from a sequence of waypoints.

    Creates a path of (x, y) points that passes through the specified
    waypoints, constrained to stay within the glyph mask.

    Args:
        waypoints_raw: List of raw waypoint values. Each can be:
            - Integer (1-9): Terminal point in that numpad region
            - 'v(n)': Vertex point in region n
            - 'c(n)': Curve point in region n
        glyph_bbox: Bounding box as (x_min, y_min, x_max, y_max).
        mask: Binary numpy array where True indicates glyph pixels.
        skel_features: Optional dict mapping region numbers to lists of
            skeleton feature positions (from find_skeleton_waypoints).
            If provided, waypoints snap to skeleton features when available.

    Returns:
        A list of (x, y) points sampled along the guide path at regular
        intervals. Returns an empty list if:
            - Fewer than 2 waypoints are provided
            - The mask is empty
            - Any waypoint cannot be placed inside the mask

    Notes:
        - Terminal waypoints are placed at glyph edges.
        - Vertex and curve waypoints are placed deep inside the glyph.
        - Linear interpolation connects consecutive waypoints.
        - All points are constrained to lie within the mask.
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
    """Find skeleton features as candidate waypoint positions.

    Skeletonizes the glyph mask and identifies endpoints and junctions
    that can serve as natural waypoint targets for stroke paths.

    Args:
        mask: Binary numpy array where True indicates glyph pixels.
        glyph_bbox: Bounding box as (x_min, y_min, x_max, y_max).

    Returns:
        A dictionary with:
            - Keys 1-9: Each maps to a list containing the best skeleton
              position for that numpad region (based on proximity to region
              center with a bonus for being deep inside the glyph).
            - 'all_skel': List of all skeleton pixel positions.
        Returns None if the skeleton is empty.

    Notes:
        - Uses scikit-image skeletonize for morphological skeletonization.
        - The scoring function balances proximity to region center with
          depth inside the glyph (distance from edge).
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
