"""Shape primitives for parametric stroke fitting.

This module contains geometric shape generators used for fitting strokes
to font glyph point clouds. Each shape function takes fractional parameters
relative to a bounding box and returns an Nx2 numpy array of points.

Shape types:
- vline: Vertical line
- hline: Horizontal line
- diag: Diagonal line
- arc_right: Right-opening arc (semicircle or partial)
- arc_left: Left-opening arc
- loop: Full ellipse
- u_arc: U-shaped arc (bottom half of ellipse)
"""

from collections.abc import Callable

import numpy as np


def shape_vline(params: tuple, bbox: tuple, offset: tuple = (0, 0), n_pts: int = 60) -> np.ndarray:
    """Vertical line. params: (x_frac, y_start_frac, y_end_frac)."""
    xf, ysf, yef = params
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    x = x0 + xf * w + offset[0]
    ys = y0 + ysf * h + offset[1]
    ye = y0 + yef * h + offset[1]
    t = np.linspace(0, 1, n_pts)
    return np.column_stack([np.full(n_pts, x), ys + t * (ye - ys)])


def shape_hline(params: tuple, bbox: tuple, offset: tuple = (0, 0), n_pts: int = 60) -> np.ndarray:
    """Horizontal line. params: (y_frac, x_start_frac, x_end_frac)."""
    yf, xsf, xef = params
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    y = y0 + yf * h + offset[1]
    xs = x0 + xsf * w + offset[0]
    xe = x0 + xef * w + offset[0]
    t = np.linspace(0, 1, n_pts)
    return np.column_stack([xs + t * (xe - xs), np.full(n_pts, y)])


def shape_diag(params: tuple, bbox: tuple, offset: tuple = (0, 0), n_pts: int = 60) -> np.ndarray:
    """Diagonal line. params: (x0f, y0f, x1f, y1f)."""
    x0f, y0f, x1f, y1f = params
    bx0, by0, bx1, by1 = bbox
    w, h = bx1 - bx0, by1 - by0
    sx = bx0 + x0f * w + offset[0]
    sy = by0 + y0f * h + offset[1]
    ex = bx0 + x1f * w + offset[0]
    ey = by0 + y1f * h + offset[1]
    t = np.linspace(0, 1, n_pts)
    return np.column_stack([sx + t * (ex - sx), sy + t * (ey - sy)])


def shape_arc_right(params: tuple, bbox: tuple, offset: tuple = (0, 0), n_pts: int = 60) -> np.ndarray:
    """Right-opening arc. params: (cx_f, cy_f, rx_f, ry_f, ang_start, ang_end)."""
    cxf, cyf, rxf, ryf, a0, a1 = params
    bx0, by0, bx1, by1 = bbox
    w, h = bx1 - bx0, by1 - by0
    cx = bx0 + cxf * w + offset[0]
    cy = by0 + cyf * h + offset[1]
    rx = rxf * w
    ry = ryf * h
    angles = np.linspace(np.radians(a0), np.radians(a1), n_pts)
    return np.column_stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)])


def shape_arc_left(params: tuple, bbox: tuple, offset: tuple = (0, 0), n_pts: int = 60) -> np.ndarray:
    """Left-opening arc. params: (cx_f, cy_f, rx_f, ry_f, ang_start, ang_end)."""
    cxf, cyf, rxf, ryf, a0, a1 = params
    bx0, by0, bx1, by1 = bbox
    w, h = bx1 - bx0, by1 - by0
    cx = bx0 + cxf * w + offset[0]
    cy = by0 + cyf * h + offset[1]
    rx = rxf * w
    ry = ryf * h
    angles = np.linspace(np.radians(a0), np.radians(a1), n_pts)
    return np.column_stack([cx - rx * np.cos(angles), cy + ry * np.sin(angles)])


def shape_loop(params: tuple, bbox: tuple, offset: tuple = (0, 0), n_pts: int = 80) -> np.ndarray:
    """Full ellipse loop. params: (cx_f, cy_f, rx_f, ry_f)."""
    cxf, cyf, rxf, ryf = params
    bx0, by0, bx1, by1 = bbox
    w, h = bx1 - bx0, by1 - by0
    cx = bx0 + cxf * w + offset[0]
    cy = by0 + cyf * h + offset[1]
    rx = rxf * w
    ry = ryf * h
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    return np.column_stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)])


def shape_u_arc(params: tuple, bbox: tuple, offset: tuple = (0, 0), n_pts: int = 60) -> np.ndarray:
    """U-shaped arc (bottom half of ellipse). params: (cx_f, cy_f, rx_f, ry_f)."""
    cxf, cyf, rxf, ryf = params
    bx0, by0, bx1, by1 = bbox
    w, h = bx1 - bx0, by1 - by0
    cx = bx0 + cxf * w + offset[0]
    cy = by0 + cyf * h + offset[1]
    rx = rxf * w
    ry = ryf * h
    angles = np.linspace(0, np.pi, n_pts)
    return np.column_stack([cx - rx * np.cos(angles), cy + ry * np.sin(angles)])


# Shape function registry
SHAPE_FNS: dict[str, Callable] = {
    'vline': shape_vline,
    'hline': shape_hline,
    'diag': shape_diag,
    'arc_right': shape_arc_right,
    'arc_left': shape_arc_left,
    'loop': shape_loop,
    'u_arc': shape_u_arc,
}

# Bounds per shape type for differential_evolution optimization.
# All in bbox-fraction space except arc angles which are in degrees.
SHAPE_PARAM_BOUNDS: dict[str, list[tuple[float, float]]] = {
    'vline': [(0.0, 1.0), (0.0, 0.5), (0.5, 1.0)],
    'hline': [(0.0, 1.0), (0.0, 0.5), (0.5, 1.0)],
    'diag': [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    'arc_right': [(0.0, 0.8), (0.0, 1.0), (0.05, 0.8), (0.05, 0.8),
                  (-180, 0), (0, 180)],
    'arc_left': [(0.2, 1.0), (0.0, 1.0), (0.05, 0.8), (0.05, 0.8),
                 (-180, 0), (0, 180)],
    'loop': [(0.1, 0.9), (0.1, 0.9), (0.1, 0.6), (0.1, 0.6)],
    'u_arc': [(0.1, 0.9), (0.2, 1.0), (0.1, 0.6), (0.1, 0.6)],
}


def get_param_bounds(templates: list[dict]) -> tuple[list[tuple], list[tuple[int, int]]]:
    """Build flat bounds list + per-shape slice indices.

    Each template entry may include an optional 'bounds' key that overrides
    specific parameter bounds. Format: list of (lo, hi) or None per param.
    None entries keep the default from SHAPE_PARAM_BOUNDS.

    Args:
        templates: List of template dicts with 'shape' and optional 'bounds' keys

    Returns:
        Tuple of (bounds_list, slices) where slices maps each shape to its param range
    """
    bounds = []
    slices = []
    offset = 0
    for t in templates:
        sb = list(SHAPE_PARAM_BOUNDS[t['shape']])
        overrides = t.get('bounds')
        if overrides:
            for j, ov in enumerate(overrides):
                if ov is not None:
                    sb[j] = ov
        bounds.extend(sb)
        slices.append((offset, offset + len(sb)))
        offset += len(sb)
    return bounds, slices


def param_vector_to_shapes(param_vector: np.ndarray, shape_types: list[str],
                           slices: list[tuple[int, int]], bbox: tuple,
                           n_pts: int | None = None) -> list[np.ndarray]:
    """Convert flat parameter vector into list of Nx2 point arrays.

    When n_pts is None it is computed from the bbox diagonal so the shape
    path is dense enough for the matching radius to form a continuous band
    (~1.5 px between samples).

    Args:
        param_vector: Flat array of all shape parameters
        shape_types: List of shape type names
        slices: List of (start, end) indices into param_vector for each shape
        bbox: Bounding box (x0, y0, x1, y1)
        n_pts: Number of points per shape, or None for auto

    Returns:
        List of Nx2 numpy arrays, one per shape
    """
    if n_pts is None:
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        n_pts = max(60, int((bw * bw + bh * bh) ** 0.5 / 1.5))
    shapes = []
    for i, stype in enumerate(shape_types):
        start, end = slices[i]
        params = tuple(param_vector[start:end])
        shapes.append(SHAPE_FNS[stype](params, bbox, offset=(0, 0), n_pts=n_pts))
    return shapes


# ---------------------------------------------------------------------------
# Point cloud utilities
# ---------------------------------------------------------------------------

def make_point_cloud(mask: np.ndarray, spacing: int = 2) -> np.ndarray:
    """Create a grid of points inside the glyph mask.

    Args:
        mask: Binary mask where True = glyph pixels
        spacing: Grid spacing in pixels

    Returns:
        Nx2 array of (x, y) coordinates inside the mask
    """
    h, w = mask.shape
    ys, xs = np.mgrid[0:h:spacing, 0:w:spacing]
    xs = xs.ravel()
    ys = ys.ravel()
    inside = mask[ys, xs]
    return np.column_stack([xs[inside], ys[inside]]).astype(float)


def adaptive_radius(mask: np.ndarray, spacing: int = 2) -> float:
    """Compute matching radius based on stroke width.

    Uses the 95th percentile of the distance transform - close to the
    maximum stroke half-width - so the optimizer can cover points across
    the full width of even the thickest strokes. Floor at 1.5x grid
    spacing so the radius always reaches neighbouring grid points, even
    for very thin strokes.

    Args:
        mask: Binary mask where True = glyph pixels
        spacing: Grid spacing used for point cloud

    Returns:
        Radius value for point matching
    """
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(mask)
    vals = dist[mask]
    floor = spacing * 1.5
    if len(vals) == 0:
        return max(6.0, floor)
    return max(float(np.percentile(vals, 95)), floor)


def score_shape(shape_pts: np.ndarray, tree, radius: float, claimed: set | None = None) -> float:
    """Count cloud points within radius of shape path.

    Gives a bonus weight for unclaimed points.

    Args:
        shape_pts: Nx2 array of shape points
        tree: KDTree of point cloud
        radius: Matching radius
        claimed: Set of already-claimed point indices (optional)

    Returns:
        Score value (higher is better)
    """
    if len(shape_pts) == 0:
        return 0
    indices = tree.query_ball_point(shape_pts, radius)
    hit = set()
    for idx_list in indices:
        hit.update(idx_list)
    if claimed is None:
        return len(hit)
    unclaimed = hit - claimed
    return len(unclaimed) * 1.0 + len(hit & claimed) * 0.3


# Legacy aliases for internal compatibility (underscore prefix)
_make_point_cloud = make_point_cloud
_adaptive_radius = adaptive_radius
_score_shape = score_shape

_shape_vline = shape_vline
_shape_hline = shape_hline
_shape_diag = shape_diag
_shape_arc_right = shape_arc_right
_shape_arc_left = shape_arc_left
_shape_loop = shape_loop
_shape_u_arc = shape_u_arc
_get_param_bounds = get_param_bounds
_param_vector_to_shapes = param_vector_to_shapes
