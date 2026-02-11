"""Affine transformation and optimization for strokes.

This module provides functions for optimizing stroke positions using affine
transformations (translation, rotation, scaling, and shear). The optimization
process adapts template strokes to better fit a target glyph mask through a
two-stage approach:

Algorithm Overview:
    Stage 1 - Global Affine Optimization:
        All strokes are transformed together using 6 parameters:
        - Translation (tx, ty): Shifts strokes in x and y directions
        - Scale (sx, sy): Stretches/compresses strokes along axes
        - Rotation (theta): Rotates strokes around the centroid
        - Shear: Applies horizontal shearing transformation

        The optimization uses configurable strategies (Strategy Pattern):
        - NelderMeadStrategy: Fast local search
        - DifferentialEvolutionStrategy: Global refinement
        - ChainedStrategy: Run multiple strategies in sequence

    Stage 2 - Per-Stroke Refinement:
        Each stroke is individually adjusted with translate+scale (4 params
        per stroke) to fine-tune positions after the global transform.

The module also provides an interface for DiffVG-based optimization using
differentiable rendering in Docker, which offers gradient-based optimization
as an alternative to the derivative-free methods used in affine optimization.

Typical usage:
    # Using the strategy pattern directly:
    strategy = ChainedStrategy([
        NelderMeadStrategy(OptimizationConfig(max_iterations=800)),
        DifferentialEvolutionStrategy(OptimizationConfig(max_iterations=20)),
    ])
    params, score = strategy.optimize(objective_fn, initial_params, bounds)

    # Or using the high-level API:
    result = optimize_affine(font_path, char, canvas_size,
                             template_to_strokes_fn, render_glyph_mask_fn,
                             resolve_font_path_fn, smooth_stroke_fn,
                             constrain_to_mask_fn)
    if result:
        strokes, score, mask, bbox = result
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from stroke_scoring import score_raw_strokes
from stroke_shapes import adaptive_radius, make_point_cloud

logger = logging.getLogger(__name__)

# --- Affine optimization hyperparameters ---
# Parameter bounds for global affine transformation
AFFINE_TRANSLATE_BOUNDS = (-20, 20)  # pixels
AFFINE_SCALE_BOUNDS = (0.7, 1.3)  # 70% to 130%
AFFINE_ROTATION_BOUNDS = (-15, 15)  # degrees
AFFINE_SHEAR_BOUNDS = (-0.3, 0.3)

# Nelder-Mead optimizer settings
NM_MAX_FUNC_EVALS = 800  # Maximum function evaluations
NM_X_TOLERANCE = 0.1  # Parameter convergence tolerance
NM_F_TOLERANCE = 0.002  # Function value convergence tolerance

# Differential Evolution optimizer settings
DE_MAX_ITERATIONS = 20  # Maximum generations
DE_POPULATION_SIZE = 10  # Population size multiplier
DE_TOLERANCE = 0.005  # Convergence tolerance


# ---------------------------------------------------------------------------
# Optimization Strategy Pattern
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    """Configuration for an optimization run.

    Encapsulates hyperparameters that control optimization behavior.
    Different strategies use different subsets of these parameters.

    Attributes:
        max_iterations: Maximum number of iterations/generations/function evals.
        tolerance: Convergence tolerance for stopping criterion.
        x_tolerance: Parameter space convergence tolerance (Nelder-Mead).
        f_tolerance: Function value convergence tolerance (Nelder-Mead).
        population_size: Population size multiplier (Differential Evolution).
        adaptive: Whether to use adaptive algorithm variants.
        polish: Whether to polish result with L-BFGS-B (DE only).

    Example:
        >>> config = OptimizationConfig(max_iterations=1000, tolerance=1e-4)
        >>> strategy = NelderMeadStrategy(config)
    """
    max_iterations: int = 100
    tolerance: float = 0.001
    x_tolerance: float = 0.1
    f_tolerance: float = 0.002
    population_size: int = 10
    adaptive: bool = True
    polish: bool = False


class OptimizationStrategy(ABC):
    """Base class for optimization strategies.

    Defines the interface for optimization algorithms. Strategies can be
    combined using ChainedStrategy to create multi-stage optimization
    pipelines.

    Subclasses must implement the optimize() method.

    Attributes:
        config: OptimizationConfig with hyperparameters.

    Example:
        >>> class MyStrategy(OptimizationStrategy):
        ...     def optimize(self, objective_fn, initial_params, bounds):
        ...         # Custom optimization logic
        ...         return best_params, best_score
    """

    def __init__(self, config: OptimizationConfig = None):
        """Initialize strategy with configuration.

        Args:
            config: Optimization configuration. If None, uses defaults.
        """
        self.config = config or OptimizationConfig()

    @abstractmethod
    def optimize(self, objective_fn, initial_params: np.ndarray,
                 bounds: list[tuple]) -> tuple[np.ndarray, float]:
        """Run optimization, return (best_params, best_score).

        Args:
            objective_fn: Function to minimize. Takes parameter array,
                returns scalar score (lower is better).
            initial_params: Starting point for optimization.
            bounds: List of (min, max) tuples for each parameter.

        Returns:
            Tuple of (best_params, best_score) where:
                - best_params: np.ndarray of optimized parameters
                - best_score: Final objective function value
        """
        pass


class NelderMeadStrategy(OptimizationStrategy):
    """Nelder-Mead simplex optimization strategy.

    A derivative-free local optimization method that works well for
    smooth objectives with moderate dimensionality. Fast but may get
    stuck in local minima.

    Uses scipy.optimize.minimize with method='Nelder-Mead'.

    Example:
        >>> strategy = NelderMeadStrategy(OptimizationConfig(max_iterations=500))
        >>> params, score = strategy.optimize(obj_fn, x0, bounds)
    """

    def optimize(self, objective_fn, initial_params: np.ndarray,
                 bounds: list[tuple]) -> tuple[np.ndarray, float]:
        """Run Nelder-Mead optimization.

        Note: bounds are not enforced by Nelder-Mead itself but can be
        used to clip the result.
        """
        from scipy.optimize import minimize

        result = minimize(
            objective_fn, initial_params, method='Nelder-Mead',
            options={
                'maxfev': self.config.max_iterations,
                'xatol': self.config.x_tolerance,
                'fatol': self.config.f_tolerance,
                'adaptive': self.config.adaptive,
            }
        )
        return result.x.copy(), result.fun


class DifferentialEvolutionStrategy(OptimizationStrategy):
    """Differential Evolution global optimization strategy.

    A population-based evolutionary algorithm that's effective at
    escaping local minima and exploring the parameter space globally.
    Slower than Nelder-Mead but more robust.

    Uses scipy.optimize.differential_evolution.

    Example:
        >>> strategy = DifferentialEvolutionStrategy(
        ...     OptimizationConfig(max_iterations=30, population_size=15)
        ... )
        >>> params, score = strategy.optimize(obj_fn, x0, bounds)
    """

    def optimize(self, objective_fn, initial_params: np.ndarray,
                 bounds: list[tuple]) -> tuple[np.ndarray, float]:
        """Run Differential Evolution optimization."""
        from scipy.optimize import differential_evolution

        try:
            result = differential_evolution(
                objective_fn, bounds=bounds,
                x0=initial_params,
                maxiter=self.config.max_iterations,
                popsize=self.config.population_size,
                tol=self.config.tolerance,
                polish=self.config.polish,
            )
            return result.x.copy(), result.fun
        except (ValueError, RuntimeError) as e:
            logger.debug("DE optimization failed: %s", e)
            # Return initial params with evaluated score on failure
            return initial_params.copy(), objective_fn(initial_params)


class ChainedStrategy(OptimizationStrategy):
    """Run multiple optimization strategies in sequence.

    Chains strategies together, using the output of each as the input
    to the next. Useful for combining fast local search (Nelder-Mead)
    with global refinement (Differential Evolution).

    Example:
        >>> chain = ChainedStrategy([
        ...     NelderMeadStrategy(OptimizationConfig(max_iterations=800)),
        ...     DifferentialEvolutionStrategy(OptimizationConfig(max_iterations=20)),
        ... ])
        >>> params, score = chain.optimize(obj_fn, x0, bounds)
    """

    def __init__(self, strategies: list[OptimizationStrategy]):
        """Initialize with list of strategies to run in sequence.

        Args:
            strategies: List of OptimizationStrategy instances.
                Each will be executed in order, with the output of
                one becoming the input to the next.
        """
        # Don't call super().__init__ as we don't need a config
        self.strategies = strategies

    def optimize(self, objective_fn, initial_params: np.ndarray,
                 bounds: list[tuple]) -> tuple[np.ndarray, float]:
        """Run all strategies in sequence."""
        params = initial_params.copy()
        score = float('inf')

        for strategy in self.strategies:
            new_params, new_score = strategy.optimize(
                objective_fn, params, bounds
            )
            # Only update if we improved
            if new_score < score:
                params = new_params
                score = new_score

        return params, score


# Default strategy for global affine optimization
def create_default_affine_strategy() -> ChainedStrategy:
    """Create the default optimization strategy for affine transforms.

    Returns:
        ChainedStrategy combining Nelder-Mead and Differential Evolution.
    """
    return ChainedStrategy([
        NelderMeadStrategy(OptimizationConfig(
            max_iterations=NM_MAX_FUNC_EVALS,
            x_tolerance=NM_X_TOLERANCE,
            f_tolerance=NM_F_TOLERANCE,
        )),
        DifferentialEvolutionStrategy(OptimizationConfig(
            max_iterations=DE_MAX_ITERATIONS,
            population_size=DE_POPULATION_SIZE,
            tolerance=DE_TOLERANCE,
        )),
    ])


def affine_transform_strokes(strokes: list[np.ndarray], params: tuple,
                             centroid: tuple[float, float]) -> list[np.ndarray]:
    """Apply a 6-parameter affine transformation to strokes around a centroid.

    Transforms each stroke by first centering on the centroid, applying rotation
    and shear, then scaling, and finally translating. The transformation order
    ensures rotation occurs around the centroid rather than the origin.

    The transformation matrix applied is:
        [sx  0 ] * [cos(theta)  -sin(theta) + shear] * (point - centroid) + centroid + [tx]
        [0   sy]   [sin(theta)   cos(theta)        ]                                   [ty]

    Args:
        strokes: List of Nx2 numpy arrays, where each array represents a stroke
            as a sequence of (x, y) coordinates.
        params: Tuple of 6 transformation parameters:
            - tx (float): Translation in x direction (pixels)
            - ty (float): Translation in y direction (pixels)
            - sx (float): Scale factor in x direction (typically 0.7-1.3)
            - sy (float): Scale factor in y direction (typically 0.7-1.3)
            - theta_deg (float): Rotation angle in degrees (typically -15 to +15)
            - shear (float): Horizontal shear factor (typically -0.3 to +0.3)
        centroid: (cx, cy) center point around which rotation and scaling occur,
            typically the centroid of the glyph mask.

    Returns:
        List of transformed Nx2 numpy arrays with the same structure as input.
        Each returned array is a new copy; input strokes are not modified.

    Note:
        The shear is applied to the y-component during the rotation step,
        creating a horizontal shearing effect that can help match italic or
        oblique letterforms.
    """
    tx, ty, sx, sy, theta_deg, shear = params
    theta = np.radians(theta_deg)
    cx, cy = centroid
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    result = []
    for stroke in strokes:
        pts = np.array(stroke, dtype=float)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        rx = dx * cos_t - dy * sin_t + shear * dy
        ry = dx * sin_t + dy * cos_t
        pts[:, 0] = cx + sx * rx + tx
        pts[:, 1] = cy + sy * ry + ty
        result.append(pts)
    return result


def prepare_affine_optimization(font_path: str, char: str, canvas_size: int,
                                strokes_raw: list, mask: np.ndarray,
                                smooth_stroke_fn, constrain_to_mask_fn) -> tuple | None:
    """Prepare data structures needed for affine stroke optimization.

    This function initializes the scoring infrastructure including the point cloud,
    KD-tree for fast nearest-neighbor queries, distance transform maps for snapping
    points to the mask boundary, and preprocesses the raw strokes.

    Args:
        font_path: Path to the font file (used for context, not directly accessed
            in this function).
        char: The character being optimized (used for context).
        canvas_size: Size of the square canvas in pixels.
        strokes_raw: Raw stroke data from template, where each stroke is a list
            of [x, y] coordinate pairs.
        mask: Binary numpy array of shape (H, W) where True indicates glyph pixels.
            Used to compute the point cloud and distance transforms.
        smooth_stroke_fn: Callable that smooths a stroke polyline. Signature:
            smooth_stroke_fn(points: list[tuple], sigma: float) -> list[tuple]
        constrain_to_mask_fn: Callable that constrains stroke points to lie within
            the mask. Signature:
            constrain_to_mask_fn(points: list[tuple], mask: ndarray) -> list[tuple]

    Returns:
        A tuple of (stroke_arrays, centroid, glyph_bbox, score_args) where:
            - stroke_arrays: List of preprocessed Nx2 numpy arrays
            - centroid: (x, y) center point of the glyph mask
            - glyph_bbox: (x_min, y_min, x_max, y_max) bounding box of the glyph
            - score_args: Tuple of arguments for score_raw_strokes function:
                (cloud_tree, n_cloud, radius, snap_yi, snap_xi, w, h, dist_map)

        Returns None if:
            - The mask contains no pixels (empty glyph)
            - The point cloud has fewer than 10 points
            - All strokes are invalid after preprocessing

    Note:
        The distance transform (dist_map) stores the distance from each pixel
        to the nearest mask boundary. The snap indices (snap_yi, snap_xi) provide
        the coordinates of the nearest mask pixel for each background pixel,
        enabling fast snapping of out-of-bounds points.
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None

    glyph_bbox = (float(cols.min()), float(rows.min()),
                  float(cols.max()), float(rows.max()))
    centroid = (float(cols.mean()), float(rows.mean()))
    cloud = make_point_cloud(mask, spacing=3)
    if len(cloud) < 10:
        return None
    cloud_tree = cKDTree(cloud)
    n_cloud = len(cloud)
    radius = adaptive_radius(mask, spacing=3)
    h, w = mask.shape
    dist_map = distance_transform_edt(mask)
    _, snap_indices = distance_transform_edt(~mask, return_indices=True)
    snap_yi, snap_xi = snap_indices[0], snap_indices[1]

    score_args = (cloud_tree, n_cloud, radius, snap_yi, snap_xi, w, h, dist_map)

    # Convert raw strokes to numpy arrays (pre-smooth once)
    stroke_arrays = []
    for s in strokes_raw:
        pl = [(float(p[0]), float(p[1])) for p in s]
        pl = smooth_stroke_fn(pl, sigma=2.0)
        pl = constrain_to_mask_fn(pl, mask)
        if len(pl) >= 2:
            stroke_arrays.append(np.array(pl))
    if not stroke_arrays:
        return None

    return stroke_arrays, centroid, glyph_bbox, score_args


def run_global_affine(stroke_arrays: list[np.ndarray], centroid: tuple[float, float],
                      score_args: tuple,
                      strategy: OptimizationStrategy = None) -> tuple[list[np.ndarray], np.ndarray, float]:
    """Execute Stage 1: global affine optimization on all strokes together.

    This stage finds optimal affine transformation parameters that apply uniformly
    to all strokes. By default, uses a two-phase approach:

    1. Nelder-Mead simplex method starting from identity transform (fast local search)
    2. Differential Evolution for global refinement (escapes local minima)

    The optimization strategy can be customized by passing a different strategy.

    Args:
        stroke_arrays: List of Nx2 numpy arrays, one per stroke. Each array
            contains (x, y) coordinates of stroke points.
        centroid: (x, y) center point for the affine transformation, typically
            the centroid of the target glyph mask.
        score_args: Tuple of scoring function arguments as returned by
            prepare_affine_optimization. Contains KD-tree, point cloud info,
            snap indices, and distance map.
        strategy: OptimizationStrategy to use. If None, uses the default
            ChainedStrategy with Nelder-Mead followed by Differential Evolution.

    Returns:
        A tuple of (best_strokes, best_params, best_score) where:
            - best_strokes: List of transformed Nx2 numpy arrays
            - best_params: numpy array of 6 optimal parameters
                [tx, ty, sx, sy, theta_deg, shear]
            - best_score: Final objective function value (lower is better;
                negate to get coverage score)

    Note:
        Parameter bounds used:
            - Translation: [-20, 20] pixels
            - Scale: [0.7, 1.3] (70% to 130%)
            - Rotation: [-15, 15] degrees
            - Shear: [-0.3, 0.3]

    Example:
        # Use custom strategy
        strategy = NelderMeadStrategy(OptimizationConfig(max_iterations=1000))
        strokes, params, score = run_global_affine(
            stroke_arrays, centroid, score_args, strategy=strategy
        )
    """
    affine_bounds = [
        AFFINE_TRANSLATE_BOUNDS, AFFINE_TRANSLATE_BOUNDS,  # translate x, y
        AFFINE_SCALE_BOUNDS, AFFINE_SCALE_BOUNDS,  # scale x, y
        AFFINE_ROTATION_BOUNDS,  # rotation degrees
        AFFINE_SHEAR_BOUNDS,  # shear
    ]

    def _affine_obj(params):
        transformed = affine_transform_strokes(stroke_arrays, params, centroid)
        return score_raw_strokes(transformed, *score_args)

    # Use provided strategy or create default
    if strategy is None:
        strategy = create_default_affine_strategy()

    # Initial parameters: identity transform
    x0 = np.array([0, 0, 1, 1, 0, 0], dtype=float)

    # Run optimization
    best_params, best_score = strategy.optimize(_affine_obj, x0, affine_bounds)

    # Apply best global affine
    best_strokes = affine_transform_strokes(stroke_arrays, best_params, centroid)
    return best_strokes, best_params, best_score


def run_per_stroke_refinement(best_strokes: list[np.ndarray], best_score: float,
                              score_args: tuple) -> tuple[list[np.ndarray], float]:
    """Execute Stage 2: per-stroke translate and scale refinement.

    After the global affine transformation, this stage fine-tunes each stroke
    individually. Each stroke gets 4 parameters: translation (dx, dy) and
    scale (sx, sy), applied around the stroke's own centroid.

    This allows strokes to adjust independently to better match local features
    of the glyph that the global transform couldn't capture.

    Args:
        best_strokes: List of Nx2 numpy arrays after Stage 1 global affine
            transformation. These serve as the baseline for refinement.
        best_score: Objective function value from Stage 1. Used to determine
            whether refinement improves the result.
        score_args: Tuple of scoring function arguments (same as Stage 1).

    Returns:
        A tuple of (final_strokes, final_score) where:
            - final_strokes: List of refined Nx2 numpy arrays. If refinement
                did not improve the score, returns the input best_strokes.
            - final_score: Final objective function value after refinement.

    Note:
        The transformation for each stroke i is:
            adjusted[i] = centroid_i + scale * (original - centroid_i) + translate

        where centroid_i is the mean of stroke i's points.

        Initial parameters are identity (no translation, unit scale) for all
        strokes. The Nelder-Mead optimizer is allowed up to 1500 function
        evaluations, with more tolerance for exploring the larger parameter
        space (4 * n_strokes dimensions).
    """
    from scipy.optimize import minimize

    n_strokes = len(best_strokes)

    def _per_stroke_obj(params):
        adjusted = []
        for si, base in enumerate(best_strokes):
            dx, dy, sx, sy = params[si * 4:(si + 1) * 4]
            pts = base.copy()
            c = pts.mean(axis=0)
            pts[:, 0] = c[0] + sx * (pts[:, 0] - c[0]) + dx
            pts[:, 1] = c[1] + sy * (pts[:, 1] - c[1]) + dy
            adjusted.append(pts)
        return score_raw_strokes(adjusted, *score_args)

    x0_per = np.array([0, 0, 1, 1] * n_strokes, dtype=float)
    nm2 = minimize(_per_stroke_obj, x0_per, method='Nelder-Mead',
                   options={'maxfev': 1500, 'xatol': 0.1, 'fatol': 0.002,
                            'adaptive': True})

    if nm2.fun < best_score:
        final_strokes = []
        for si, base in enumerate(best_strokes):
            dx, dy, sx, sy = nm2.x[si * 4:(si + 1) * 4]
            pts = base.copy()
            c = pts.mean(axis=0)
            pts[:, 0] = c[0] + sx * (pts[:, 0] - c[0]) + dx
            pts[:, 1] = c[1] + sy * (pts[:, 1] - c[1]) + dy
            final_strokes.append(pts)
        return final_strokes, nm2.fun
    else:
        return best_strokes, best_score


def optimize_affine(font_path: str, char: str, canvas_size: int,
                    template_to_strokes_fn, render_glyph_mask_fn,
                    resolve_font_path_fn, smooth_stroke_fn,
                    constrain_to_mask_fn) -> tuple | None:
    """Optimize template strokes via two-stage affine transformation.

    This is the main entry point for affine-based stroke optimization. It takes
    template strokes (e.g., from a reference font) and optimizes their positions
    to better match the target glyph using affine transformations.

    The optimization proceeds in two stages:
        Stage 1: Global affine transform (6 parameters) applied to all strokes
            together, finding overall position, scale, rotation, and shear.
        Stage 2: Per-stroke refinement (4 parameters each) for translate and
            scale adjustments to individual strokes.

    Args:
        font_path: Path to the target font file.
        char: The character to optimize strokes for.
        canvas_size: Size of the square canvas in pixels (e.g., 224).
        template_to_strokes_fn: Callable that extracts template strokes.
            Signature: template_to_strokes_fn(font_path, char, canvas_size)
                -> list of strokes or None
        render_glyph_mask_fn: Callable that renders the glyph as a binary mask.
            Signature: render_glyph_mask_fn(font_path, char, canvas_size)
                -> numpy array or None
        resolve_font_path_fn: Callable that resolves font path (e.g., handles
            aliases or relative paths).
            Signature: resolve_font_path_fn(font_path) -> resolved_path
        smooth_stroke_fn: Callable that applies Gaussian smoothing to strokes.
            Signature: smooth_stroke_fn(points, sigma) -> smoothed_points
        constrain_to_mask_fn: Callable that constrains stroke points to mask.
            Signature: constrain_to_mask_fn(points, mask) -> constrained_points

    Returns:
        A tuple of (strokes, score, mask, glyph_bbox) where:
            - strokes: List of optimized strokes, each as [[x, y], ...] with
                coordinates rounded to 1 decimal place
            - score: Final coverage score (higher is better; this is the
                negated objective function value)
            - mask: The binary glyph mask used for optimization
            - glyph_bbox: (x_min, y_min, x_max, y_max) bounding box

        Returns None if:
            - No template strokes are available for the character
            - The glyph mask cannot be rendered
            - Optimization setup fails (empty mask, insufficient points, etc.)

    Example:
        >>> result = optimize_affine(
        ...     '/fonts/target.ttf', 'A', 224,
        ...     template_to_strokes, render_glyph_mask,
        ...     resolve_font_path, smooth_stroke, constrain_to_mask
        ... )
        >>> if result:
        ...     strokes, score, mask, bbox = result
        ...     print(f"Optimized {len(strokes)} strokes with score {score:.2f}")
    """
    strokes_raw = template_to_strokes_fn(font_path, char, canvas_size)
    if not strokes_raw or len(strokes_raw) == 0:
        return None

    font_path = resolve_font_path_fn(font_path)
    mask = render_glyph_mask_fn(font_path, char, canvas_size)
    if mask is None:
        return None

    # Setup optimization data
    setup = prepare_affine_optimization(font_path, char, canvas_size, strokes_raw, mask,
                                        smooth_stroke_fn, constrain_to_mask_fn)
    if setup is None:
        return None
    stroke_arrays, centroid, glyph_bbox, score_args = setup

    # Stage 1: Global affine
    best_strokes, _best_params, best_score = run_global_affine(stroke_arrays, centroid, score_args)

    # Stage 2: Per-stroke refinement
    final_strokes, final_score = run_per_stroke_refinement(best_strokes, best_score, score_args)

    # Convert back to list format
    result_strokes = [[[round(float(p[0]), 1), round(float(p[1]), 1)]
                       for p in s] for s in final_strokes]

    return result_strokes, float(-final_score), mask, glyph_bbox
