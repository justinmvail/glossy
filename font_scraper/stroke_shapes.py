"""Shape primitives for parametric stroke fitting.

This module provides geometric shape generators used for fitting strokes
to font glyph point clouds. Each shape function takes fractional parameters
relative to a bounding box and returns an Nx2 numpy array of points.

The module supports the following shape types:
    - vline: Vertical line segment
    - hline: Horizontal line segment
    - diag: Diagonal line segment between two points
    - arc_right: Right-opening arc (semicircle or partial)
    - arc_left: Left-opening arc (mirrored semicircle)
    - loop: Full ellipse (closed curve)
    - u_arc: U-shaped arc (bottom half of ellipse)

Typical usage example:
    >>> from stroke_shapes import SHAPES, get_param_bounds
    >>> bbox = (0, 0, 100, 100)
    >>> params = (0.5, 0.0, 1.0)  # x_frac, y_start_frac, y_end_frac
    >>> points = SHAPES['vline'].generate(params, bbox)
    >>> print(points.shape)
    (60, 2)

Notes:
    All coordinate parameters are specified as fractions (0.0 to 1.0) of the
    bounding box dimensions. This allows templates to be scale-invariant and
    applicable to glyphs of any size.
"""

from abc import ABC, abstractmethod

import numpy as np

# Point cloud and shape sampling constants
MIN_SHAPE_POINTS = 60  # Minimum points to sample per shape
POINT_SPACING_TARGET = 1.5  # Target pixel spacing between shape samples
RADIUS_FLOOR_MULTIPLIER = 1.5  # Radius floor = spacing * this multiplier
MIN_RADIUS = 6.0  # Absolute minimum matching radius
DISTANCE_PERCENTILE = 95  # Percentile for stroke width estimation


# ---------------------------------------------------------------------------
# Shape Classes (Polymorphism Pattern)
# ---------------------------------------------------------------------------

class Shape(ABC):
    """Base class for parametric stroke shapes.

    Each shape subclass implements the generation of points for a specific
    geometric primitive (line, arc, loop, etc.). The class encapsulates both
    the generation logic and the parameter bounds for optimization.

    Subclasses must implement:
        - generate(): Create the shape points
        - get_bounds(): Return parameter bounds for optimization
        - param_count: Number of parameters for this shape

    Example:
        >>> shape = VLineShape()
        >>> points = shape.generate((0.5, 0.0, 1.0), (0, 0, 100, 100))
        >>> bounds = shape.get_bounds()
    """

    @abstractmethod
    def generate(self, params: tuple, bbox: tuple, offset: tuple = (0, 0),
                 n_pts: int = 60) -> np.ndarray:
        """Generate shape points.

        Args:
            params: Shape-specific parameters as fractions of bbox dimensions.
            bbox: Bounding box as (x0, y0, x1, y1).
            offset: Optional (dx, dy) offset to apply to all points.
            n_pts: Number of points to generate.

        Returns:
            np.ndarray: Nx2 array of (x, y) coordinates.
        """
        pass

    @abstractmethod
    def get_bounds(self) -> list[tuple[float, float]]:
        """Return parameter bounds for optimization.

        Returns:
            List of (min, max) tuples, one per parameter.
        """
        pass

    @property
    @abstractmethod
    def param_count(self) -> int:
        """Number of parameters this shape requires."""
        pass

    def get_default_params(self) -> tuple:
        """Return sensible default parameters for this shape.

        Default implementation returns midpoints of bounds.
        Subclasses may override for better defaults.
        """
        bounds = self.get_bounds()
        return tuple((lo + hi) / 2 for lo, hi in bounds)

    def validate_params(self, params: tuple) -> bool:
        """Check if parameters are within bounds.

        Args:
            params: Parameter tuple to validate.

        Returns:
            True if all parameters are within their bounds.
        """
        bounds = self.get_bounds()
        if len(params) != len(bounds):
            return False
        for val, (lo, hi) in zip(params, bounds):
            if val < lo or val > hi:
                return False
        return True


class VLineShape(Shape):
    """Vertical line segment shape.

    Parameters: (x_frac, y_start_frac, y_end_frac)
        - x_frac: Horizontal position as fraction of bbox width (0.0-1.0)
        - y_start_frac: Start y-position as fraction of bbox height
        - y_end_frac: End y-position as fraction of bbox height
    """

    def generate(self, params: tuple, bbox: tuple, offset: tuple = (0, 0),
                 n_pts: int = 60) -> np.ndarray:
        """Generate vertical line points. See Shape.generate."""
        xf, ysf, yef = params
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        x = x0 + xf * w + offset[0]
        ys = y0 + ysf * h + offset[1]
        ye = y0 + yef * h + offset[1]
        t = np.linspace(0, 1, n_pts)
        return np.column_stack([np.full(n_pts, x), ys + t * (ye - ys)])

    def get_bounds(self) -> list[tuple[float, float]]:
        """Return bounds: x_frac [0,1], y_start [0,0.5], y_end [0.5,1]."""
        return [(0.0, 1.0), (0.0, 0.5), (0.5, 1.0)]

    @property
    def param_count(self) -> int:
        """Return 3 (x_frac, y_start_frac, y_end_frac)."""
        return 3


class HLineShape(Shape):
    """Horizontal line segment shape.

    Parameters: (y_frac, x_start_frac, x_end_frac)
        - y_frac: Vertical position as fraction of bbox height (0.0-1.0)
        - x_start_frac: Start x-position as fraction of bbox width
        - x_end_frac: End x-position as fraction of bbox width
    """

    def generate(self, params: tuple, bbox: tuple, offset: tuple = (0, 0),
                 n_pts: int = 60) -> np.ndarray:
        """Generate horizontal line points. See Shape.generate."""
        yf, xsf, xef = params
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        y = y0 + yf * h + offset[1]
        xs = x0 + xsf * w + offset[0]
        xe = x0 + xef * w + offset[0]
        t = np.linspace(0, 1, n_pts)
        return np.column_stack([xs + t * (xe - xs), np.full(n_pts, y)])

    def get_bounds(self) -> list[tuple[float, float]]:
        """Return bounds: y_frac [0,1], x_start [0,0.5], x_end [0.5,1]."""
        return [(0.0, 1.0), (0.0, 0.5), (0.5, 1.0)]

    @property
    def param_count(self) -> int:
        """Return 3 (y_frac, x_start_frac, x_end_frac)."""
        return 3


class DiagShape(Shape):
    """Diagonal line segment shape.

    Parameters: (x0_frac, y0_frac, x1_frac, y1_frac)
        - x0_frac: Start x-position as fraction of bbox width (0.0-1.0)
        - y0_frac: Start y-position as fraction of bbox height
        - x1_frac: End x-position as fraction of bbox width
        - y1_frac: End y-position as fraction of bbox height
    """

    def generate(self, params: tuple, bbox: tuple, offset: tuple = (0, 0),
                 n_pts: int = 60) -> np.ndarray:
        """Generate diagonal line points. See Shape.generate."""
        x0f, y0f, x1f, y1f = params
        bx0, by0, bx1, by1 = bbox
        w, h = bx1 - bx0, by1 - by0
        sx = bx0 + x0f * w + offset[0]
        sy = by0 + y0f * h + offset[1]
        ex = bx0 + x1f * w + offset[0]
        ey = by0 + y1f * h + offset[1]
        t = np.linspace(0, 1, n_pts)
        return np.column_stack([sx + t * (ex - sx), sy + t * (ey - sy)])

    def get_bounds(self) -> list[tuple[float, float]]:
        """Return bounds: all four position fracs [0,1]."""
        return [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    @property
    def param_count(self) -> int:
        """Return 4 (x0_frac, y0_frac, x1_frac, y1_frac)."""
        return 4


class ArcRightShape(Shape):
    """Right-opening arc (partial ellipse) shape.

    Parameters: (cx_frac, cy_frac, rx_frac, ry_frac, ang_start, ang_end)
        - cx_frac: Center x-position as fraction of bbox width (0.0-1.0)
        - cy_frac: Center y-position as fraction of bbox height
        - rx_frac: Horizontal radius as fraction of bbox width
        - ry_frac: Vertical radius as fraction of bbox height
        - ang_start: Start angle in degrees (0 = right, 90 = down)
        - ang_end: End angle in degrees
    """

    def generate(self, params: tuple, bbox: tuple, offset: tuple = (0, 0),
                 n_pts: int = 60) -> np.ndarray:
        """Generate right-opening arc points. See Shape.generate."""
        cxf, cyf, rxf, ryf, a0, a1 = params
        bx0, by0, bx1, by1 = bbox
        w, h = bx1 - bx0, by1 - by0
        cx = bx0 + cxf * w + offset[0]
        cy = by0 + cyf * h + offset[1]
        rx = rxf * w
        ry = ryf * h
        angles = np.linspace(np.radians(a0), np.radians(a1), n_pts)
        return np.column_stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)])

    def get_bounds(self) -> list[tuple[float, float]]:
        """Return bounds for center, radii, and angles."""
        return [(0.0, 0.8), (0.0, 1.0), (0.05, 0.8), (0.05, 0.8),
                (-180, 0), (0, 180)]

    @property
    def param_count(self) -> int:
        """Return 6 (cx, cy, rx, ry, ang_start, ang_end)."""
        return 6


class ArcLeftShape(Shape):
    """Left-opening arc (partial ellipse) shape.

    Parameters: (cx_frac, cy_frac, rx_frac, ry_frac, ang_start, ang_end)
        - cx_frac: Center x-position as fraction of bbox width (0.0-1.0)
        - cy_frac: Center y-position as fraction of bbox height
        - rx_frac: Horizontal radius as fraction of bbox width
        - ry_frac: Vertical radius as fraction of bbox height
        - ang_start: Start angle in degrees (0 = left, 90 = down)
        - ang_end: End angle in degrees
    """

    def generate(self, params: tuple, bbox: tuple, offset: tuple = (0, 0),
                 n_pts: int = 60) -> np.ndarray:
        """Generate left-opening arc points. See Shape.generate."""
        cxf, cyf, rxf, ryf, a0, a1 = params
        bx0, by0, bx1, by1 = bbox
        w, h = bx1 - bx0, by1 - by0
        cx = bx0 + cxf * w + offset[0]
        cy = by0 + cyf * h + offset[1]
        rx = rxf * w
        ry = ryf * h
        angles = np.linspace(np.radians(a0), np.radians(a1), n_pts)
        return np.column_stack([cx - rx * np.cos(angles), cy + ry * np.sin(angles)])

    def get_bounds(self) -> list[tuple[float, float]]:
        """Return bounds for center, radii, and angles."""
        return [(0.2, 1.0), (0.0, 1.0), (0.05, 0.8), (0.05, 0.8),
                (-180, 0), (0, 180)]

    @property
    def param_count(self) -> int:
        """Return 6 (cx, cy, rx, ry, ang_start, ang_end)."""
        return 6


class LoopShape(Shape):
    """Full ellipse (closed loop) shape.

    Parameters: (cx_frac, cy_frac, rx_frac, ry_frac)
        - cx_frac: Center x-position as fraction of bbox width (0.0-1.0)
        - cy_frac: Center y-position as fraction of bbox height
        - rx_frac: Horizontal radius as fraction of bbox width
        - ry_frac: Vertical radius as fraction of bbox height
    """

    def generate(self, params: tuple, bbox: tuple, offset: tuple = (0, 0),
                 n_pts: int = 80) -> np.ndarray:
        """Generate full ellipse points. See Shape.generate."""
        cxf, cyf, rxf, ryf = params
        bx0, by0, bx1, by1 = bbox
        w, h = bx1 - bx0, by1 - by0
        cx = bx0 + cxf * w + offset[0]
        cy = by0 + cyf * h + offset[1]
        rx = rxf * w
        ry = ryf * h
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        return np.column_stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)])

    def get_bounds(self) -> list[tuple[float, float]]:
        """Return bounds for center and radii."""
        return [(0.1, 0.9), (0.1, 0.9), (0.1, 0.6), (0.1, 0.6)]

    @property
    def param_count(self) -> int:
        """Return 4 (cx_frac, cy_frac, rx_frac, ry_frac)."""
        return 4


class UArcShape(Shape):
    """U-shaped arc (bottom half of ellipse) shape.

    Parameters: (cx_frac, cy_frac, rx_frac, ry_frac)
        - cx_frac: Center x-position as fraction of bbox width (0.0-1.0)
        - cy_frac: Center y-position as fraction of bbox height
        - rx_frac: Horizontal radius as fraction of bbox width
        - ry_frac: Vertical radius as fraction of bbox height
    """

    def generate(self, params: tuple, bbox: tuple, offset: tuple = (0, 0),
                 n_pts: int = 60) -> np.ndarray:
        """Generate U-arc (bottom half ellipse) points. See Shape.generate."""
        cxf, cyf, rxf, ryf = params
        bx0, by0, bx1, by1 = bbox
        w, h = bx1 - bx0, by1 - by0
        cx = bx0 + cxf * w + offset[0]
        cy = by0 + cyf * h + offset[1]
        rx = rxf * w
        ry = ryf * h
        angles = np.linspace(0, np.pi, n_pts)
        return np.column_stack([cx - rx * np.cos(angles), cy + ry * np.sin(angles)])

    def get_bounds(self) -> list[tuple[float, float]]:
        """Return bounds for center and radii."""
        return [(0.1, 0.9), (0.2, 1.0), (0.1, 0.6), (0.1, 0.6)]

    @property
    def param_count(self) -> int:
        """Return 4 (cx_frac, cy_frac, rx_frac, ry_frac)."""
        return 4


# ---------------------------------------------------------------------------
# Shape Registry
# ---------------------------------------------------------------------------

# SHAPES: Maps shape type names to Shape instances.
# This is the primary registry for shape types and the preferred way to
# access shapes in new code.
SHAPES: dict[str, Shape] = {
    'vline': VLineShape(),
    'hline': HLineShape(),
    'diag': DiagShape(),
    'arc_right': ArcRightShape(),
    'arc_left': ArcLeftShape(),
    'loop': LoopShape(),
    'u_arc': UArcShape(),
}


def get_param_bounds(templates: list[dict]) -> tuple[list[tuple], list[tuple[int, int]]]:
    """Build flat bounds list and per-shape slice indices for optimization.

    Combines parameter bounds from multiple shape templates into a single
    flat list suitable for scipy.optimize.differential_evolution(). Also
    returns slice indices to extract each shape's parameters from the
    flattened vector.

    Args:
        templates: List of template dictionaries, each containing:
            - 'shape': str - Shape type name (e.g., 'vline', 'arc_right')
            - 'bounds': list[tuple|None], optional - Per-parameter bound
              overrides. None entries keep the default from the Shape class.

    Returns:
        tuple: A 2-tuple containing:
            - bounds_list: Flat list of (min, max) tuples for all parameters
              across all shapes, in order.
            - slices: List of (start, end) index tuples, one per shape,
              indicating which portion of the flattened parameter vector
              belongs to that shape.

    Example:
        >>> templates = [
        ...     {'shape': 'vline'},
        ...     {'shape': 'hline', 'bounds': [(0.4, 0.6), None, None]}
        ... ]
        >>> bounds, slices = get_param_bounds(templates)
        >>> len(bounds)  # 3 params for vline + 3 for hline
        6
        >>> slices
        [(0, 3), (3, 6)]
    """
    bounds = []
    slices = []
    offset = 0
    for t in templates:
        shape = SHAPES[t['shape']]
        sb = list(shape.get_bounds())
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
    """Convert flat parameter vector into list of shape point arrays.

    Takes a flattened parameter vector (as used by optimizers) and generates
    the corresponding shape geometries. Each shape type is dispatched to its
    Shape class with the appropriate parameter slice.

    Args:
        param_vector: Flat numpy array containing all shape parameters
            concatenated in order.
        shape_types: List of shape type names corresponding to each shape.
        slices: List of (start, end) index tuples from get_param_bounds(),
            indicating parameter boundaries for each shape.
        bbox: Bounding box as (x0, y0, x1, y1) for coordinate computation.
        n_pts: Number of points per shape. If None, computed automatically
            from bbox diagonal to ensure ~1.5 pixel spacing between samples.

    Returns:
        list[np.ndarray]: List of Nx2 point arrays, one per shape. Each array
            contains the (x, y) coordinates of the generated shape.

    Notes:
        When n_pts is None, the automatic calculation ensures the shape
        path is dense enough for the matching radius to form a continuous
        band during point cloud scoring.
    """
    if n_pts is None:
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        n_pts = max(MIN_SHAPE_POINTS, int((bw * bw + bh * bh) ** 0.5 / POINT_SPACING_TARGET))
    shapes = []
    for i, stype in enumerate(shape_types):
        start, end = slices[i]
        params = tuple(param_vector[start:end])
        shapes.append(SHAPES[stype].generate(params, bbox, offset=(0, 0), n_pts=n_pts))
    return shapes


# ---------------------------------------------------------------------------
# Point cloud utilities
# ---------------------------------------------------------------------------

def make_point_cloud(mask: np.ndarray, spacing: int = 2) -> np.ndarray:
    """Create a grid of points inside a glyph mask.

    Generates a regular grid of sample points at the specified spacing,
    then filters to only include points that fall within the glyph
    (where the mask is True).

    Args:
        mask: 2D boolean numpy array where True indicates glyph pixels.
            Shape should be (height, width).
        spacing: Grid spacing in pixels between sample points. Smaller
            values create denser point clouds. Defaults to 2.

    Returns:
        np.ndarray: An Nx2 array of (x, y) coordinates for points inside
            the mask. Returns empty array if no points fall inside.

    Notes:
        The point cloud is used for scoring shape fits - shapes that pass
        through more point cloud points are considered better fits for
        the glyph.
    """
    h, w = mask.shape
    ys, xs = np.mgrid[0:h:spacing, 0:w:spacing]
    xs = xs.ravel()
    ys = ys.ravel()
    inside = mask[ys, xs]
    return np.column_stack([xs[inside], ys[inside]]).astype(float)


def adaptive_radius(mask: np.ndarray, spacing: int = 2) -> float:
    """Compute matching radius based on stroke width.

    Analyzes the distance transform of the mask to estimate stroke width
    and returns an appropriate radius for point-to-shape matching during
    optimization.

    Args:
        mask: 2D boolean numpy array where True indicates glyph pixels.
        spacing: Grid spacing used for the point cloud. Used to set a
            minimum floor for the radius. Defaults to 2.

    Returns:
        float: The computed matching radius in pixels. This is the 95th
            percentile of the distance transform values, floored at 1.5x
            the grid spacing.

    Notes:
        Uses the 95th percentile of the distance transform - close to the
        maximum stroke half-width - so the optimizer can cover points
        across the full width of even the thickest strokes. The floor at
        1.5x grid spacing ensures the radius always reaches neighbouring
        grid points, even for very thin strokes.
    """
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(mask)
    vals = dist[mask]
    floor = spacing * RADIUS_FLOOR_MULTIPLIER
    if len(vals) == 0:
        return max(MIN_RADIUS, floor)
    return max(float(np.percentile(vals, DISTANCE_PERCENTILE)), floor)
