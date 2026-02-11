"""Stroke scoring functions for optimization.

This module provides functions for scoring strokes against target point clouds
and glyph masks during shape fitting optimization. It supports both gradient-based
optimization (via differentiable objectives) and greedy fitting approaches.

The scoring system evaluates strokes based on:
    - Coverage: How well strokes cover the target point cloud
    - Overlap penalty: Discourages redundant overlapping strokes
    - Snap penalty: Penalizes stroke points that fall outside the glyph mask
    - Edge penalty: Penalizes strokes that hug glyph edges instead of
      running through the interior

Design Patterns:
    The module implements the Composite Scoring Pattern, where penalty
    calculations are encapsulated in separate ScoringPenalty classes that
    can be combined via CompositeScorer. This allows:
    - Easy addition of new penalty types without modifying core logic
    - Configurable penalty weights per-use-case
    - Easy disabling of penalties for debugging

Key classes:
    - ScoringPenalty: Base class for penalty calculations
    - SnapPenalty, EdgePenalty, OverlapPenalty: Concrete penalty implementations
    - CompositeScorer: Combines coverage and penalties into final score

Key functions:
    - score_all_strokes: Main objective for multi-stroke optimization
    - score_raw_strokes: Score pre-built stroke arrays
    - quick_stroke_score: Fast coverage estimation for validation

Typical usage:
    from stroke_scoring import score_all_strokes, quick_stroke_score, CompositeScorer

    # Using the composite scorer pattern:
    scorer = CompositeScorer(penalties=[
        SnapPenalty(weight=0.5),
        EdgePenalty(weight=0.1),
        OverlapPenalty(weight=0.5),
    ])
    score = scorer.score(strokes, context)

    # Or using the legacy function API:
    result = minimize(
        lambda p: score_all_strokes(p, shape_types, slices, bbox,
                                    cloud_tree, n_cloud, radius,
                                    snap_yi, snap_xi, w, h, dist_map),
        initial_params
    )

    # Quick quality check
    coverage = quick_stroke_score(strokes, mask)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from stroke_shapes import (
    SHAPES,
)
from stroke_shapes import (
    param_vector_to_shapes as _param_vector_to_shapes,
)

# Scoring constants
DEFAULT_STROKE_HALF_WIDTH = 6.0  # Approximate stroke half-width for coverage


@dataclass
class ScoringContext:
    """Context object bundling parameters for stroke scoring.

    Groups related scoring parameters together for cleaner function signatures
    and easier testing. Use this instead of passing 10+ individual parameters.

    Attributes:
        cloud_tree: KD-tree of target point cloud for coverage queries.
        n_cloud: Number of points in the target cloud.
        radius: Coverage radius for point-to-stroke matching.
        snap_xi: 2D array mapping (y, x) to nearest mask x-coordinate.
        snap_yi: 2D array mapping (y, x) to nearest mask y-coordinate.
        w: Width of the mask/image.
        h: Height of the mask/image.
        dist_map: Optional distance transform for edge penalty.

    Example:
        >>> context = ScoringContext(
        ...     cloud_tree=cKDTree(points),
        ...     n_cloud=len(points),
        ...     radius=5.0,
        ...     snap_xi=snap_xi,
        ...     snap_yi=snap_yi,
        ...     w=224, h=224,
        ...     dist_map=distance_transform_edt(mask)
        ... )
        >>> score = score_all_strokes_ctx(param_vector, shape_types, slices, bbox, context)
    """
    cloud_tree: cKDTree
    n_cloud: int
    radius: float
    snap_xi: np.ndarray
    snap_yi: np.ndarray
    w: int
    h: int
    dist_map: np.ndarray | None = None


# Module-level constants for scoring
FREE_OVERLAP = 0.25       # Fraction of overlap allowed before penalty
SNAP_PENALTY_WEIGHT = 0.5  # Weight for off-mask penalty
EDGE_PENALTY_WEIGHT = 0.1  # Weight for near-edge penalty
OVERLAP_PENALTY_WEIGHT = 0.5  # Weight for overlap penalty
EDGE_THRESHOLD = 1.5      # Distance threshold for "near edge"
SNAP_THRESHOLD = 0.5      # Distance threshold for "off mask"


# ---------------------------------------------------------------------------
# Composite Scoring Pattern
# ---------------------------------------------------------------------------

class ScoringPenalty(ABC):
    """Base class for scoring penalties.

    Each penalty subclass computes a specific type of penalty (e.g., snap,
    edge, overlap) that contributes to the overall stroke score. Penalties
    are combined by CompositeScorer.

    Subclasses must implement the compute() method.

    Attributes:
        weight: Multiplier for this penalty in the composite score.

    Example:
        >>> class CustomPenalty(ScoringPenalty):
        ...     def compute(self, stroke_points, context, **kwargs):
        ...         # Custom penalty logic
        ...         return penalty_value
    """

    def __init__(self, weight: float = 1.0):
        """Initialize penalty with weight.

        Args:
            weight: Multiplier for this penalty. Higher weight means
                this penalty has more influence on the final score.
        """
        self.weight = weight

    @abstractmethod
    def compute(self, stroke_points: np.ndarray, context: ScoringContext,
                snapped_points: np.ndarray = None,
                per_shape_coverage: list[set] = None) -> float:
        """Compute penalty value (0 = no penalty).

        Args:
            stroke_points: Nx2 array of all stroke point coordinates.
            context: ScoringContext with cloud, snap arrays, and dimensions.
            snapped_points: Optional pre-computed snapped points.
            per_shape_coverage: Optional list of coverage sets per shape.

        Returns:
            Penalty value between 0 and 1, where 0 is no penalty.
        """
        pass


class SnapPenalty(ScoringPenalty):
    """Penalty for points outside the glyph mask.

    Computes the fraction of stroke points that fall outside the glyph
    mask (i.e., their snapped position differs from their original position).

    Attributes:
        threshold: Distance threshold for considering a point "off mask".
    """

    def __init__(self, weight: float = SNAP_PENALTY_WEIGHT,
                 threshold: float = SNAP_THRESHOLD):
        super().__init__(weight)
        self.threshold = threshold

    def compute(self, stroke_points: np.ndarray, context: ScoringContext,
                snapped_points: np.ndarray = None,
                per_shape_coverage: list[set] = None) -> float:
        """Compute snap penalty based on fraction of points off mask."""
        if snapped_points is None:
            snapped_points = _snap_points_to_mask(
                stroke_points, context.snap_xi, context.snap_yi,
                context.w, context.h
            )
        snap_dist = np.sqrt(
            (stroke_points[:, 0] - snapped_points[:, 0]) ** 2 +
            (stroke_points[:, 1] - snapped_points[:, 1]) ** 2
        )
        return float(np.mean(snap_dist > self.threshold))


class EdgePenalty(ScoringPenalty):
    """Penalty for points near the glyph edge.

    Penalizes strokes that hug the glyph boundary rather than running
    through the interior. Uses the distance transform to measure how
    far each point is from the edge.

    Attributes:
        threshold: Distance threshold for considering a point "near edge".
    """

    def __init__(self, weight: float = EDGE_PENALTY_WEIGHT,
                 threshold: float = EDGE_THRESHOLD):
        super().__init__(weight)
        self.threshold = threshold

    def compute(self, stroke_points: np.ndarray, context: ScoringContext,
                snapped_points: np.ndarray = None,
                per_shape_coverage: list[set] = None) -> float:
        """Compute edge penalty based on fraction of points near edge."""
        if context.dist_map is None:
            return 0.0

        if snapped_points is None:
            snapped_points = _snap_points_to_mask(
                stroke_points, context.snap_xi, context.snap_yi,
                context.w, context.h
            )

        sxi = np.clip(np.round(snapped_points[:, 0]).astype(int), 0, context.w - 1)
        syi = np.clip(np.round(snapped_points[:, 1]).astype(int), 0, context.h - 1)
        dt_vals = context.dist_map[syi, sxi]
        return float(np.mean(dt_vals < self.threshold))


class OverlapPenalty(ScoringPenalty):
    """Penalty for excessive overlap between shapes.

    Discourages redundant overlapping strokes by penalizing shapes that
    cover points already covered by other shapes, beyond a free overlap
    threshold.

    Attributes:
        free_overlap: Fraction of overlap allowed before penalty kicks in.
    """

    def __init__(self, weight: float = OVERLAP_PENALTY_WEIGHT,
                 free_overlap: float = FREE_OVERLAP):
        super().__init__(weight)
        self.free_overlap = free_overlap

    def compute(self, stroke_points: np.ndarray, context: ScoringContext,
                snapped_points: np.ndarray = None,
                per_shape_coverage: list[set] = None) -> float:
        """Compute overlap penalty based on inter-shape overlap."""
        if per_shape_coverage is None or len(per_shape_coverage) <= 1:
            return 0.0

        n_shapes = len(per_shape_coverage)
        overlap_excess = 0.0

        for i in range(n_shapes):
            if not per_shape_coverage[i]:
                continue
            others = set()
            for j in range(n_shapes):
                if j != i:
                    others |= per_shape_coverage[j]
            frac = len(per_shape_coverage[i] & others) / len(per_shape_coverage[i])
            if frac > self.free_overlap:
                overlap_excess += (frac - self.free_overlap)

        overlap_excess /= n_shapes
        return overlap_excess


class CompositeScorer:
    """Combines coverage and penalties into final score.

    The composite scorer orchestrates the scoring process:
    1. Computes snapped points (once, shared across penalties)
    2. Computes per-shape coverage sets
    3. Evaluates all penalties
    4. Combines coverage and penalties into final score

    Example:
        >>> scorer = CompositeScorer(penalties=[
        ...     SnapPenalty(weight=0.5),
        ...     EdgePenalty(weight=0.1),
        ...     OverlapPenalty(weight=0.5),
        ... ])
        >>> score = scorer.score(strokes, context)
    """

    def __init__(self, penalties: list[ScoringPenalty] = None):
        """Initialize with list of penalties.

        Args:
            penalties: List of ScoringPenalty instances. If None, uses
                default penalties (Snap, Edge, Overlap).
        """
        if penalties is None:
            penalties = [
                SnapPenalty(weight=SNAP_PENALTY_WEIGHT),
                EdgePenalty(weight=EDGE_PENALTY_WEIGHT),
                OverlapPenalty(weight=OVERLAP_PENALTY_WEIGHT),
            ]
        self.penalties = penalties

    def score(self, stroke_arrays: list[np.ndarray],
              context: ScoringContext) -> float:
        """Compute the final score for strokes.

        Args:
            stroke_arrays: List of Nx2 arrays, one per stroke.
            context: ScoringContext with scoring parameters.

        Returns:
            Negative float score for minimization. Lower (more negative)
            values indicate better coverage with fewer penalties.
        """
        if not stroke_arrays or all(len(s) == 0 for s in stroke_arrays):
            return 0.0

        processed = [s for s in stroke_arrays if len(s) >= 2]
        if not processed:
            return 0.0

        all_pts = np.concatenate(processed, axis=0)

        # Compute snapped points once (shared by penalties)
        snapped = _snap_points_to_mask(
            all_pts, context.snap_xi, context.snap_yi,
            context.w, context.h
        )

        # Compute per-shape coverage
        per_shape = _compute_per_shape_coverage(
            processed, snapped, context.cloud_tree, context.radius
        )
        covered_all = set().union(*per_shape) if per_shape else set()
        coverage = len(covered_all) / context.n_cloud

        # Compute all penalties
        total_penalty = 0.0
        for penalty in self.penalties:
            penalty_value = penalty.compute(
                all_pts, context,
                snapped_points=snapped,
                per_shape_coverage=per_shape
            )
            total_penalty += penalty.weight * penalty_value

        return -(coverage - total_penalty)

    def add_penalty(self, penalty: ScoringPenalty) -> None:
        """Add a penalty to the scorer.

        Args:
            penalty: ScoringPenalty instance to add.
        """
        self.penalties.append(penalty)

    def remove_penalty_by_type(self, penalty_type: type) -> bool:
        """Remove all penalties of a specific type.

        Args:
            penalty_type: Type of penalty to remove (e.g., SnapPenalty).

        Returns:
            True if any penalties were removed.
        """
        original_len = len(self.penalties)
        self.penalties = [p for p in self.penalties if not isinstance(p, penalty_type)]
        return len(self.penalties) < original_len


# Default scorer instance for convenience
_default_scorer = CompositeScorer()


def _snap_points_to_mask(all_pts: np.ndarray, snap_xi: np.ndarray,
                          snap_yi: np.ndarray, w: int, h: int) -> np.ndarray:
    """Snap stroke points to nearest mask pixels.

    Args:
        all_pts: Nx2 array of (x, y) stroke point coordinates.
        snap_xi: 2D array mapping (y, x) to nearest mask x-coordinate.
        snap_yi: 2D array mapping (y, x) to nearest mask y-coordinate.
        w: Width of the mask/image.
        h: Height of the mask/image.

    Returns:
        Nx2 array of snapped (x, y) coordinates.
    """
    xi = np.clip(np.round(all_pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(all_pts[:, 1]).astype(int), 0, h - 1)
    snapped_x = snap_xi[yi, xi].astype(float)
    snapped_y = snap_yi[yi, xi].astype(float)
    return np.column_stack([snapped_x, snapped_y])


def _compute_snap_penalty(all_pts: np.ndarray, snapped: np.ndarray) -> float:
    """Compute penalty for stroke points that fall outside the mask.

    .. deprecated::
        Use SnapPenalty class for new code.

    Args:
        all_pts: Original Nx2 array of stroke points.
        snapped: Snapped Nx2 array of stroke points.

    Returns:
        Weighted penalty based on fraction of points off mask.
    """
    snap_dist = np.sqrt((all_pts[:, 0] - snapped[:, 0]) ** 2 +
                        (all_pts[:, 1] - snapped[:, 1]) ** 2)
    off_mask = float(np.mean(snap_dist > SNAP_THRESHOLD))
    return SNAP_PENALTY_WEIGHT * off_mask


def _compute_edge_penalty(snapped: np.ndarray, dist_map: np.ndarray,
                          w: int, h: int) -> float:
    """Compute penalty for stroke points near the glyph edge.

    .. deprecated::
        Use EdgePenalty class for new code.

    Args:
        snapped: Nx2 array of snapped stroke point coordinates.
        dist_map: Distance transform of the mask.
        w: Width of the mask/image.
        h: Height of the mask/image.

    Returns:
        Weighted penalty based on fraction of points near edge.
    """
    if dist_map is None:
        return 0.0
    sxi = np.clip(np.round(snapped[:, 0]).astype(int), 0, w - 1)
    syi = np.clip(np.round(snapped[:, 1]).astype(int), 0, h - 1)
    dt_vals = dist_map[syi, sxi]
    near_edge = float(np.mean(dt_vals < EDGE_THRESHOLD))
    return EDGE_PENALTY_WEIGHT * near_edge


def _compute_per_shape_coverage(all_shapes: list[np.ndarray], snapped: np.ndarray,
                                 cloud_tree: cKDTree, radius: float) -> list[set]:
    """Compute coverage sets for each shape.

    Args:
        all_shapes: List of Nx2 arrays, one per shape.
        snapped: All snapped points concatenated.
        cloud_tree: KD-tree of target point cloud.
        radius: Coverage radius.

    Returns:
        List of sets, each containing indices of cloud points covered by that shape.
    """
    per_shape = []
    offset = 0
    for shape_pts in all_shapes:
        n = len(shape_pts)
        shape_snapped = snapped[offset:offset + n]
        offset += n
        hits = cloud_tree.query_ball_point(shape_snapped, radius)
        sc = set()
        for lst in hits:
            sc.update(lst)
        per_shape.append(sc)
    return per_shape


def _compute_overlap_penalty(per_shape: list[set]) -> float:
    """Compute penalty for excessive overlap between shapes.

    .. deprecated::
        Use OverlapPenalty class for new code.

    Args:
        per_shape: List of coverage sets, one per shape.

    Returns:
        Weighted penalty based on overlap exceeding FREE_OVERLAP threshold.
    """
    n_shapes = len(per_shape)
    if n_shapes <= 1:
        return 0.0

    overlap_excess = 0.0
    for i in range(n_shapes):
        if not per_shape[i]:
            continue
        others = set()
        for j in range(n_shapes):
            if j != i:
                others |= per_shape[j]
        frac = len(per_shape[i] & others) / len(per_shape[i])
        if frac > FREE_OVERLAP:
            overlap_excess += (frac - FREE_OVERLAP)
    overlap_excess /= n_shapes

    return OVERLAP_PENALTY_WEIGHT * overlap_excess


def score_all_strokes(param_vector: np.ndarray, shape_types: list[str],
                      slices: list[tuple[int, int]], bbox: tuple,
                      cloud_tree: cKDTree, n_cloud: int, radius: float,
                      snap_yi: np.ndarray, snap_xi: np.ndarray,
                      w: int, h: int, dist_map: np.ndarray = None) -> float:
    """Compute the optimization objective for a set of stroke parameters.

    This is the main objective function for stroke fitting optimization.
    It evaluates how well the parameterized strokes cover the target point
    cloud while penalizing strokes that fall outside the glyph or overlap
    excessively.

    The function returns a negative score (for minimization), computed as:
        score = -(coverage - overlap_penalty - snap_penalty - edge_penalty)

    Args:
        param_vector: Flat numpy array of shape parameters for all strokes.
        shape_types: List of shape type strings (e.g., 'line', 'arc', 'bezier')
            corresponding to each stroke.
        slices: List of (start, end) tuples indicating the parameter indices
            for each shape in param_vector.
        bbox: Bounding box as (x_min, y_min, x_max, y_max).
        cloud_tree: scipy cKDTree built from target point cloud coordinates.
        n_cloud: Total number of points in the target cloud.
        radius: Coverage radius - a cloud point is "covered" if within this
            distance of any stroke point.
        snap_yi: 2D array mapping (y, x) to nearest mask y-coordinate.
        snap_xi: 2D array mapping (y, x) to nearest mask x-coordinate.
        w: Width of the mask/image.
        h: Height of the mask/image.
        dist_map: Optional distance transform of the mask. If provided,
            enables edge penalty calculation.

    Returns:
        Negative float score for minimization. Lower (more negative) values
        indicate better stroke coverage with fewer penalties.

    Notes:
        - Stroke points are snapped to the nearest mask pixel before scoring,
          so the optimizer sees consistent benefits from mask-constrained paths.
        - Snap penalty: 0.5 * fraction of points outside mask
        - Edge penalty: 0.1 * fraction of points within 1.5px of edge
        - Overlap penalty: 0.5 * excess overlap beyond 25% free overlap
    """
    all_shapes = _param_vector_to_shapes(param_vector, shape_types, slices, bbox)
    all_pts = np.concatenate(all_shapes, axis=0)
    if len(all_pts) == 0:
        return 0.0

    # Snap points and compute penalties
    snapped = _snap_points_to_mask(all_pts, snap_xi, snap_yi, w, h)
    snap_penalty = _compute_snap_penalty(all_pts, snapped)
    edge_penalty = _compute_edge_penalty(snapped, dist_map, w, h)

    # Compute per-shape coverage
    per_shape = _compute_per_shape_coverage(all_shapes, snapped, cloud_tree, radius)
    covered_all = set().union(*per_shape) if per_shape else set()
    coverage = len(covered_all) / n_cloud

    # Compute overlap penalty
    overlap_penalty = _compute_overlap_penalty(per_shape)

    return -(coverage - overlap_penalty - snap_penalty - edge_penalty)


def score_all_strokes_ctx(param_vector: np.ndarray, shape_types: list[str],
                          slices: list[tuple[int, int]], bbox: tuple,
                          ctx: ScoringContext) -> float:
    """Score strokes using a ScoringContext for cleaner API.

    Wrapper around score_all_strokes that accepts a ScoringContext instead
    of individual parameters. Preferred for new code.

    Args:
        param_vector: Flat numpy array of shape parameters for all strokes.
        shape_types: List of shape type strings for each stroke.
        slices: List of (start, end) tuples for parameter indices.
        bbox: Bounding box as (x_min, y_min, x_max, y_max).
        ctx: ScoringContext with cloud, snap arrays, and dimensions.

    Returns:
        Negative float score for minimization.
    """
    return score_all_strokes(
        param_vector, shape_types, slices, bbox,
        ctx.cloud_tree, ctx.n_cloud, ctx.radius,
        ctx.snap_yi, ctx.snap_xi, ctx.w, ctx.h, ctx.dist_map
    )


def score_raw_strokes(stroke_arrays: list[np.ndarray], cloud_tree: cKDTree,
                      n_cloud: int, radius: float, snap_yi: np.ndarray,
                      snap_xi: np.ndarray, w: int, h: int,
                      dist_map: np.ndarray = None, mask: np.ndarray = None,
                      scorer: 'CompositeScorer' = None) -> float:
    """Score pre-built stroke point arrays against the target point cloud.

    Similar to score_all_strokes but accepts raw Nx2 coordinate arrays
    instead of shape parameters. Useful for scoring strokes that were
    generated through non-parametric methods (e.g., skeleton tracing).

    Args:
        stroke_arrays: List of Nx2 numpy arrays, one per stroke. Each array
            contains (x, y) coordinates for points along the stroke.
        cloud_tree: scipy cKDTree built from target point cloud coordinates.
        n_cloud: Total number of points in the target cloud.
        radius: Coverage radius for determining if cloud points are covered.
        snap_yi: 2D array mapping (y, x) to nearest mask y-coordinate.
        snap_xi: 2D array mapping (y, x) to nearest mask x-coordinate.
        w: Width of the mask/image.
        h: Height of the mask/image.
        dist_map: Optional distance transform for edge penalty calculation.
        mask: Binary mask array (currently unused, reserved for future use).
        scorer: Optional CompositeScorer for custom penalty configuration.
            If None, uses the default scorer with standard weights.

    Returns:
        Negative float score for minimization. Lower values indicate better
        stroke quality.

    Notes:
        - Strokes with fewer than 2 points are filtered out.
        - Returns 0.0 if no valid strokes are provided.
        - Uses the same penalty weights as score_all_strokes.

    Example:
        # Use custom scorer with adjusted weights
        scorer = CompositeScorer(penalties=[
            SnapPenalty(weight=0.3),
            EdgePenalty(weight=0.2),
        ])
        score = score_raw_strokes(strokes, cloud_tree, n_cloud, radius,
                                  snap_yi, snap_xi, w, h, scorer=scorer)
    """
    if scorer is not None:
        # Use the composite scorer pattern
        context = ScoringContext(
            cloud_tree=cloud_tree, n_cloud=n_cloud, radius=radius,
            snap_xi=snap_xi, snap_yi=snap_yi, w=w, h=h, dist_map=dist_map
        )
        return scorer.score(stroke_arrays, context)

    # Legacy implementation for backwards compatibility
    if not stroke_arrays or all(len(s) == 0 for s in stroke_arrays):
        return 0.0

    processed = [s for s in stroke_arrays if len(s) >= 2]
    if not processed:
        return 0.0

    all_pts = np.concatenate(processed, axis=0)

    # Snap points and compute penalties using shared helpers
    snapped = _snap_points_to_mask(all_pts, snap_xi, snap_yi, w, h)
    snap_penalty = _compute_snap_penalty(all_pts, snapped)
    edge_penalty = _compute_edge_penalty(snapped, dist_map, w, h)

    # Compute per-shape coverage
    per_shape = _compute_per_shape_coverage(processed, snapped, cloud_tree, radius)
    covered_all = set().union(*per_shape) if per_shape else set()
    coverage = len(covered_all) / n_cloud

    # Compute overlap penalty
    overlap_penalty = _compute_overlap_penalty(per_shape)

    return -(coverage - overlap_penalty - snap_penalty - edge_penalty)


def quick_stroke_score(strokes: list[list[list[float]]], mask: np.ndarray) -> float:
    """Compute a quick coverage score for stroke quality validation.

    Estimates what fraction of the glyph mask is covered by the given
    strokes, using distance transform dilation for efficient computation.

    Args:
        strokes: List of strokes, where each stroke is a list of [x, y]
            coordinate pairs.
        mask: Binary numpy array where True indicates glyph pixels.

    Returns:
        Float between 0 and 1 representing the fraction of glyph pixels
        that are within the coverage radius of any stroke point.

    Notes:
        - Uses a fixed coverage radius of 6 pixels (approximate stroke half-width).
        - Returns 0.0 if strokes is empty, mask is None, or glyph has no pixels.
        - Faster than full scoring but less accurate for optimization.
    """
    from scipy.ndimage import distance_transform_edt

    if not strokes or mask is None:
        return 0.0

    h, w = mask.shape

    # Get all stroke points
    all_pts = []
    for stroke in strokes:
        for pt in stroke:
            all_pts.append((int(round(pt[0])), int(round(pt[1]))))

    if not all_pts:
        return 0.0

    # Build a mask of stroke coverage using distance transform
    stroke_mask = np.zeros_like(mask, dtype=bool)
    for x, y in all_pts:
        if 0 <= x < w and 0 <= y < h:
            stroke_mask[y, x] = True

    # Dilate stroke mask using distance transform
    dist = distance_transform_edt(~stroke_mask)

    covered = dist <= DEFAULT_STROKE_HALF_WIDTH
    glyph_pixels = np.sum(mask)
    if glyph_pixels == 0:
        return 0.0

    covered_glyph = np.sum(mask & covered)
    return float(covered_glyph) / float(glyph_pixels)
