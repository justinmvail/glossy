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

Data Structure Conventions:
    This module uses tuples for coordinate pairs:
    - Point: tuple[float, float] - A single (x, y) coordinate
    - BoundingBox: tuple[float, float, float, float] - (x_min, y_min, x_max, y_max)
    - Path: list[tuple[float, float]] - A sequence of connected points

Error Handling Conventions:
    - Returns None when a position cannot be computed (e.g., snap_to_glyph_edge)
    - Returns empty list [] when no valid path can be built (e.g., build_guide_path)
    - Raises ValueError for invalid waypoint formats (e.g., parse_waypoint)
    - Raises ImportError when required dependencies are missing

Typical usage:
    from stroke_utils import smooth_stroke, build_guide_path

    # Smooth a noisy stroke path
    smoothed = smooth_stroke(raw_points, sigma=2.0)

    # Build a guide path through waypoints
    path = build_guide_path([7, 'v(5)', 3], bbox, mask)
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

# Import shared utilities from stroke_lib (canonical implementations)
# These are re-exported for backwards compatibility
from stroke_lib.utils.geometry import (  # noqa: F401
    compute_direction_vector,
    compute_dot_product,
    constrain_to_mask,
    infer_direction_from_regions,
    pick_straightest_neighbor,
    point_distance,
    point_distance_squared,
    point_in_region,
    smooth_stroke,
)

# Note: resolve_font_path is in stroke_flask.py (canonical location for Flask utilities)
# Note: smooth_stroke and constrain_to_mask are imported from
# stroke_lib.utils.geometry (canonical implementations) at the top of this file.

# ---------------------------------------------------------------------------
# Snap Function Constants
# ---------------------------------------------------------------------------

# Minimum depth (distance from edge in pixels) for a point to be considered
# "deep inside" the glyph. Points closer to edges than this threshold will
# be moved deeper inside along a ray toward the centroid.
SNAP_DEEP_INSIDE_MIN_DEPTH = 5

# Default step size (in pixels) for generating evenly-spaced points along
# line segments. Smaller values produce denser point distributions.
LINEAR_SEGMENT_STEP = 2.0

# Multiplier for comparing current depth against best depth during ray
# casting. Used to stop early when depth decreases significantly.
DEPTH_MULTIPLIER = 0.5

# ---------------------------------------------------------------------------
# Skeleton Feature Constants
# ---------------------------------------------------------------------------

# Number of nearest neighbor candidates to consider per region when finding
# skeleton waypoints. Balances search speed vs accuracy.
KD_TREE_K_CANDIDATES = 50


# ---------------------------------------------------------------------------
# Geometry Utilities
# ---------------------------------------------------------------------------

def bbox_width_height(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Extract width and height from a bounding box.

    Args:
        bbox: Bounding box as (x_min, y_min, x_max, y_max).

    Returns:
        Tuple of (width, height).
    """
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def snap_inside(pos: tuple[float, float], mask: np.ndarray, snap_indices: np.ndarray) -> tuple[float, float]:
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


def snap_deep_inside(pos: tuple[float, float], centroid: tuple[float, float], dist_in: np.ndarray,
                     mask: np.ndarray, snap_indices: np.ndarray) -> tuple[float, float]:
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
        position is already at least SNAP_DEEP_INSIDE_MIN_DEPTH pixels from
        the edge, it is returned unchanged. Otherwise, the deepest point
        along the ray to the centroid is returned.

    Notes:
        - The search stops early once it finds a point deeper than
          SNAP_DEEP_INSIDE_MIN_DEPTH pixels and the depth starts decreasing,
          to avoid overshooting.
        - Falls back to snap_inside if the ray length is less than 1 pixel.
    """
    h, w = mask.shape
    ix = int(round(min(max(pos[0], 0), w - 1)))
    iy = int(round(min(max(pos[1], 0), h - 1)))
    if mask[iy, ix] and dist_in[iy, ix] >= SNAP_DEEP_INSIDE_MIN_DEPTH:
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
        if best_depth > SNAP_DEEP_INSIDE_MIN_DEPTH and dist_in[jy, jx] < best_depth * DEPTH_MULTIPLIER:
            break
    return best_pos


def snap_to_glyph_edge(pos: tuple[float, float], centroid: tuple[float, float], mask: np.ndarray) -> tuple[float, float] | None:
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


def parse_waypoint(wp: int | str) -> tuple[int, str]:
    """Parse a waypoint specification into region and kind.

    This is a compatibility wrapper around stroke_dataclasses._parse_waypoint_item().
    For new code, consider using ParsedWaypoint directly from stroke_dataclasses.

    Waypoints can be specified as:
        - Integer (1-9): Terminal point in that numpad region
        - 'v(n)': Vertex point in region n
        - 'c(n)': Curve point in region n
        - 'i(n)': Intersection point in region n

    Args:
        wp: The waypoint specification. Can be an integer or a string
            matching 'v(digit)', 'c(digit)', or 'i(digit)'.

    Returns:
        A tuple of (region, kind) where:
            - region is an integer 1-9 indicating the numpad region
            - kind is one of 'terminal', 'vertex', 'curve', or 'intersection'

    Raises:
        ValueError: If the waypoint format is not recognized.

    Note:
        The canonical waypoint parsing is in stroke_dataclasses.py which returns
        ParsedWaypoint objects. This function provides backwards compatibility
        by converting to (region, kind) tuples.
    """
    # Import the canonical parser from stroke_dataclasses
    from stroke_dataclasses import _parse_waypoint_item

    parsed = _parse_waypoint_item(wp)
    if parsed is None:
        raise ValueError(f"Unknown waypoint format: {wp}")

    # Convert ParsedWaypoint to legacy (region, kind) tuple format
    if parsed.is_intersection:
        return (parsed.region, 'intersection')
    elif parsed.is_vertex:
        return (parsed.region, 'vertex')
    elif parsed.is_curve:
        return (parsed.region, 'curve')
    else:
        return (parsed.region, 'terminal')


def numpad_to_pixel(region: int, glyph_bbox: tuple[float, float, float, float]) -> tuple[float, float]:
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
    try:
        from stroke_templates import NUMPAD_POS
    except ImportError as e:
        raise ImportError(f"Failed to import stroke_templates: {e}") from e
    frac_x, frac_y = NUMPAD_POS[region]
    x_min, y_min, x_max, y_max = glyph_bbox
    return (x_min + frac_x * (x_max - x_min),
            y_min + frac_y * (y_max - y_min))


def linear_segment(p0: tuple[float, float], p1: tuple[float, float], step: float = LINEAR_SEGMENT_STEP) -> list[tuple[float, float]]:
    """Generate evenly-spaced points along a line segment.

    Args:
        p0: Starting point as (x, y).
        p1: Ending point as (x, y).
        step: Approximate distance between consecutive points.
            Defaults to LINEAR_SEGMENT_STEP (2.0).

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


def _snap_to_skeleton_region(region: int, glyph_bbox: tuple[float, float, float, float],
                              skel_features: dict | None) -> tuple[float, float] | None:
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


def _resolve_waypoint_position(region: int, kind: str, glyph_bbox: tuple[float, float, float, float],
                                mask: np.ndarray, centroid: tuple[float, float],
                                dist_in: np.ndarray, snap_indices: np.ndarray,
                                skel_features: dict | None) -> tuple[float, float] | None:
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


def _constrain_points_to_mask(points: list[tuple[float, float]], mask: np.ndarray,
                               snap_indices: np.ndarray) -> list[tuple[float, float]]:
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


def compute_distance_cache(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute distance transform cache for a mask.

    Pre-computes distance transforms that can be reused across multiple
    calls to build_guide_path with the same mask.

    Args:
        mask: Binary numpy array where True indicates glyph pixels.

    Returns:
        Tuple of (snap_indices, dist_in) where:
        - snap_indices: Indices for snapping outside points to inside
        - dist_in: Distance from each pixel to nearest edge (for deep inside)
    """
    _dist_out, snap_indices = distance_transform_edt(~mask, return_indices=True)
    dist_in = distance_transform_edt(mask)
    return snap_indices, dist_in


def build_guide_path(waypoints_raw: list[int | str], glyph_bbox: tuple[float, float, float, float], mask: np.ndarray,
                     skel_features: dict | None = None,
                     dist_cache: tuple[np.ndarray, np.ndarray] | None = None) -> list[tuple[float, float]]:
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
        dist_cache: Optional pre-computed distance transforms as
            (snap_indices, dist_in) tuple. If provided, avoids recomputing
            distance transforms for multiple calls with the same mask.

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

    # Use cached distance fields or compute them
    if dist_cache is not None:
        snap_indices, dist_in = dist_cache
    else:
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


def find_skeleton_waypoints(mask: np.ndarray, glyph_bbox: tuple[float, float, float, float]) -> dict | None:
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
        - Uses KD-tree for efficient nearest-neighbor candidate search,
          reducing complexity from O(9*n) to O(9*k*log(n)) where k is
          the number of candidates considered per region.
    """
    from collections import defaultdict

    from scipy.spatial import cKDTree
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
    skel_array = np.array(skel_list)  # Shape: (n, 2) with (x, y) coords
    dist_in = distance_transform_edt(mask)
    max_dist = float(dist_in.max()) if dist_in.max() > 0 else 1.0

    x_min, y_min, x_max, y_max = glyph_bbox
    bbox_diag = max(((x_max - x_min) ** 2 + (y_max - y_min) ** 2) ** 0.5, 1.0)

    # Build KD-tree for efficient spatial queries
    # Number of candidates to consider per region (balances speed vs accuracy)
    k_candidates = min(KD_TREE_K_CANDIDATES, len(skel_list))
    tree = cKDTree(skel_array)

    region_features = {}
    for r in range(1, 10):
        rc = numpad_to_pixel(r, glyph_bbox)

        # Query k-nearest neighbors from KD-tree
        _, indices = tree.query(rc, k=k_candidates)
        if k_candidates == 1:
            indices = [indices]  # Ensure iterable for single result

        # Score only the candidates (not all skeleton pixels)
        best_score = float('inf')
        best_pt = skel_list[indices[0]]
        for idx in indices:
            p = skel_list[idx]
            dx = p[0] - rc[0]
            dy = p[1] - rc[1]
            proximity = (dx * dx + dy * dy) ** 0.5 / bbox_diag
            depth = dist_in[p[1], p[0]] / max_dist
            score = proximity - 0.3 * depth
            if score < best_score:
                best_score = score
                best_pt = p

        region_features[r] = [best_pt]

    region_features['all_skel'] = skel_list
    return region_features
