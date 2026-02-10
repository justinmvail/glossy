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

import re

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d

# Import shared utilities from stroke_lib (canonical implementations)
from stroke_lib.utils.geometry import (
    smooth_stroke,
    constrain_to_mask,
    point_in_region,
)

# Note: resolve_font_path is in stroke_flask.py (canonical location for Flask utilities)
# Note: smooth_stroke and constrain_to_mask are imported from
# stroke_lib.utils.geometry (canonical implementations) at the top of this file.


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


# Note: point_in_region is imported from stroke_lib.utils.geometry
# (canonical implementation) at the top of this file.


def _snap_to_skeleton_region(region: int, glyph_bbox: tuple,
                              skel_features: dict | None) -> tuple | None:
    """Find the best skeleton position for a region.

    Args:
        region: Numpad region (1-9).
        glyph_bbox: Bounding box as (x_min, y_min, x_max, y_max).
        skel_features: Optional dict mapping region numbers to skeleton positions.

    Returns:
        The (x, y) skeleton position, or None if no skeleton features available.
    """
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


def _resolve_waypoint_position(region: int, kind: str, glyph_bbox: tuple,
                                mask: np.ndarray, centroid: tuple,
                                dist_in: np.ndarray, snap_indices: np.ndarray,
                                skel_features: dict | None) -> tuple | None:
    """Resolve a waypoint to its pixel position.

    Args:
        region: Numpad region (1-9).
        kind: Waypoint kind ('terminal', 'vertex', or 'curve').
        glyph_bbox: Bounding box.
        mask: Binary glyph mask.
        centroid: Glyph centroid (x, y).
        dist_in: Distance transform of mask.
        snap_indices: Snap indices from distance transform.
        skel_features: Optional skeleton features dict.

    Returns:
        The (x, y) position, or None if waypoint cannot be placed.
    """
    h, w = mask.shape

    # Try skeleton position first
    skel_pos = _snap_to_skeleton_region(region, glyph_bbox, skel_features)
    if skel_pos is not None:
        return (float(skel_pos[0]), float(skel_pos[1]))

    # Handle based on waypoint kind
    if kind == 'terminal':
        return snap_to_glyph_edge(numpad_to_pixel(region, glyph_bbox), centroid, mask)

    # Vertex or curve - place deep inside
    pos = snap_deep_inside(numpad_to_pixel(region, glyph_bbox),
                           centroid, dist_in, mask, snap_indices)
    ix = int(round(min(max(pos[0], 0), w - 1)))
    iy = int(round(min(max(pos[1], 0), h - 1)))
    if not mask[iy, ix]:
        return None
    return pos


def _constrain_points_to_mask(points: list[tuple], mask: np.ndarray,
                               snap_indices: np.ndarray) -> list[tuple]:
    """Constrain a list of points to lie within the mask.

    Args:
        points: List of (x, y) coordinates.
        mask: Binary glyph mask.
        snap_indices: Snap indices from distance transform.

    Returns:
        List of constrained (x, y) coordinates.
    """
    h, w = mask.shape
    constrained = []
    for x, y in points:
        ix = int(round(min(max(x, 0), w - 1)))
        iy = int(round(min(max(y, 0), h - 1)))
        if mask[iy, ix]:
            constrained.append((x, y))
        else:
            constrained.append(snap_inside((x, y), mask, snap_indices))
    return constrained


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
    _dist_out, snap_indices = distance_transform_edt(~mask, return_indices=True)
    dist_in = distance_transform_edt(mask)

    # Resolve all waypoints to positions
    positions = []
    for region, kind in parsed:
        pos = _resolve_waypoint_position(
            region, kind, glyph_bbox, mask, centroid,
            dist_in, snap_indices, skel_features
        )
        if pos is None:
            return []
        positions.append(pos)

    # Build path segments between consecutive waypoints
    all_points = []
    for i in range(n_wp - 1):
        seg = linear_segment(positions[i], positions[i + 1], step=2.0)
        if all_points and seg:
            seg = seg[1:]
        all_points.extend(seg)

    # Constrain all guide points to be inside the mask
    return _constrain_points_to_mask(all_points, mask, snap_indices)


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
