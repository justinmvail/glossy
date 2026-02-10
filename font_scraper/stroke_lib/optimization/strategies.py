"""Optimization strategy implementations.

This module provides concrete optimization strategy implementations for
fitting stroke templates to glyph masks. Each strategy uses a different
approach to optimization, and they can be combined in a pipeline for
progressive refinement.

The module provides the following classes:
    OptimizationResult: Data class for strategy results.
    OptimizationStrategy: Protocol defining the strategy interface.
    AffineStrategy: Global optimization via affine transformation.
    GreedyStrategy: Per-shape greedy local optimization.
    JointRefinementStrategy: Global joint optimization using differential
        evolution.

Example usage:
    Using individual strategies::

        from stroke_lib.optimization.strategies import AffineStrategy

        strategy = AffineStrategy(max_iterations=100)
        result = strategy.optimize(mask, templates, bbox)

        print(f"Score: {result.score}")
        print(f"Converged: {result.converged}")

    Checking strategy results::

        if result.converged:
            for stroke in result.strokes:
                print(f"Stroke: {len(stroke)} points")
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Protocol
import numpy as np

from ..domain.geometry import Stroke, BBox


@dataclass
class OptimizationResult:
    """Result of an optimization strategy.

    Contains the optimized strokes, score, and metadata about the
    optimization process. This is the standard return type for all
    optimization strategies.

    Attributes:
        strokes: List of Stroke objects representing the optimized
            stroke paths.
        score: Quality score from 0.0 to 1.0, typically representing
            coverage of the glyph mask. Higher is better.
        params: Optional numpy array of parameter values that produced
            this result. Can be used as initial_params for subsequent
            optimization stages.
        converged: Whether the optimization algorithm converged to a
            solution (met stopping criteria normally).
        iterations: Number of iterations performed by the optimizer.

    Example:
        >>> result = strategy.optimize(mask, templates, bbox)
        >>> print(f"Score: {result.score:.3f}")
        >>> print(f"Found {len(result.strokes)} strokes")
    """
    strokes: List[Stroke]
    score: float
    params: Optional[np.ndarray] = None
    converged: bool = False
    iterations: int = 0


class OptimizationStrategy(Protocol):
    """Protocol for optimization strategies.

    Defines the interface that all optimization strategies must implement.
    Strategies take a glyph mask and template list and return optimized
    strokes that best match the glyph.

    All strategies should:
        - Accept optional initial parameters from previous stages
        - Respect the time_budget constraint
        - Return an OptimizationResult with strokes and score

    Example implementation::

        class CustomStrategy:
            def optimize(
                self,
                mask: np.ndarray,
                templates: List[str],
                bbox: BBox,
                initial_params: Optional[np.ndarray] = None,
                time_budget: float = 5.0,
            ) -> OptimizationResult:
                # Custom optimization logic here
                return OptimizationResult(strokes=[], score=0.0)
    """

    def optimize(
        self,
        mask: np.ndarray,
        templates: List[str],
        bbox: BBox,
        initial_params: Optional[np.ndarray] = None,
        time_budget: float = 5.0,
    ) -> OptimizationResult:
        """Run optimization.

        Args:
            mask: Binary glyph mask as numpy array of shape (H, W).
                Non-zero values indicate glyph pixels.
            templates: List of shape template names to fit. The strategy
                should generate strokes matching these templates.
            bbox: Bounding box of the glyph for coordinate normalization.
            initial_params: Optional initial parameter vector from a
                previous optimization stage. If None, the strategy should
                initialize parameters from scratch.
            time_budget: Maximum time in seconds for this optimization
                stage. Strategies should attempt to terminate gracefully
                when this limit is approached.

        Returns:
            OptimizationResult containing optimized strokes, score, and
            metadata about the optimization process.
        """
        ...


@dataclass
class AffineStrategy:
    """Optimization via affine transformation of template strokes.

    Applies global translation, rotation, and scaling to match templates
    to the glyph mask. This is typically the first stage in an optimization
    pipeline, providing a rough global alignment before local refinement.

    The strategy optimizes 6 parameters:
        - tx, ty: Translation in x and y
        - rotation: Rotation angle in radians
        - scale_x, scale_y: Scaling factors
        - shear: Shear transformation factor

    Uses Nelder-Mead simplex optimization from scipy.

    Attributes:
        max_iterations: Maximum number of optimization iterations.
            Default is 100.

    Example:
        >>> strategy = AffineStrategy(max_iterations=50)
        >>> result = strategy.optimize(mask, templates, bbox)
    """
    max_iterations: int = 100

    def optimize(
        self,
        mask: np.ndarray,
        templates: List[str],
        bbox: BBox,
        initial_params: Optional[np.ndarray] = None,
        time_budget: float = 5.0,
    ) -> OptimizationResult:
        """Optimize via affine transformation.

        Uses Nelder-Mead optimization to find the best affine transformation
        parameters that maximize coverage of the glyph mask.

        Args:
            mask: Binary glyph mask.
            templates: List of template names to transform.
            bbox: Glyph bounding box for centering.
            initial_params: Initial [tx, ty, rot, sx, sy, shear] parameters.
                If None, initializes to center of bbox with no rotation/shear.
            time_budget: Time limit in seconds.

        Returns:
            OptimizationResult with transformed strokes and coverage score.
        """
        from scipy.optimize import minimize

        if not templates:
            return OptimizationResult([], -1.0)

        # Initial parameters: [tx, ty, rotation, scale_x, scale_y, shear]
        if initial_params is None:
            cx, cy = bbox.center.x, bbox.center.y
            initial_params = np.array([cx, cy, 0.0, 1.0, 1.0, 0.0])

        def objective(params):
            tx, ty, rot, sx, sy, shear = params
            # Transform and score (simplified - actual implementation would
            # generate strokes and compute coverage)
            return -self._compute_coverage(mask, templates, params, bbox)

        result = minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': self.max_iterations}
        )

        # Generate final strokes from optimized params
        strokes = self._params_to_strokes(result.x, templates, bbox)

        return OptimizationResult(
            strokes=strokes,
            score=-result.fun,
            params=result.x,
            converged=result.success,
            iterations=result.nit,
        )

    def _compute_coverage(
        self,
        mask: np.ndarray,
        templates: List[str],
        params: np.ndarray,
        bbox: BBox
    ) -> float:
        """Compute mask coverage for given parameters.

        Calculates what fraction of the glyph mask is covered by the
        transformed template strokes.

        Args:
            mask: Binary glyph mask.
            templates: List of template names.
            params: Affine transformation parameters.
            bbox: Glyph bounding box.

        Returns:
            Coverage score from 0.0 to 1.0.
        """
        # Simplified coverage computation
        return 0.5  # Placeholder

    def _params_to_strokes(
        self,
        params: np.ndarray,
        templates: List[str],
        bbox: BBox
    ) -> List[Stroke]:
        """Convert parameters to stroke objects.

        Applies the affine transformation to template shapes and
        returns the resulting strokes.

        Args:
            params: Affine transformation parameters.
            templates: List of template names.
            bbox: Glyph bounding box.

        Returns:
            List of transformed Stroke objects.
        """
        # Placeholder - would transform template shapes
        return []


@dataclass
class GreedyStrategy:
    """Greedy per-shape optimization.

    Optimizes each shape independently to maximize local coverage.
    This strategy refines individual strokes one at a time, which
    is faster than joint optimization but may miss global optima.

    Typically used after AffineStrategy for local refinement.

    Attributes:
        max_iterations_per_shape: Maximum optimization iterations per
            individual shape. Default is 50.

    Example:
        >>> strategy = GreedyStrategy(max_iterations_per_shape=30)
        >>> result = strategy.optimize(mask, templates, bbox)
    """
    max_iterations_per_shape: int = 50

    def optimize(
        self,
        mask: np.ndarray,
        templates: List[str],
        bbox: BBox,
        initial_params: Optional[np.ndarray] = None,
        time_budget: float = 5.0,
    ) -> OptimizationResult:
        """Optimize each shape greedily.

        Iterates through templates and optimizes each one independently,
        respecting the time budget.

        Args:
            mask: Binary glyph mask.
            templates: List of template names to optimize.
            bbox: Glyph bounding box.
            initial_params: Initial parameters (may be used to initialize
                individual shape parameters).
            time_budget: Total time limit in seconds for all shapes.

        Returns:
            OptimizationResult with individually optimized strokes.
        """
        import time

        start_time = time.time()
        strokes = []
        total_score = 0.0

        for i, template in enumerate(templates):
            if time.time() - start_time > time_budget:
                break

            # Optimize this shape
            stroke, score = self._optimize_single_shape(
                mask, template, bbox, i, len(templates)
            )
            if stroke:
                strokes.append(stroke)
                total_score += score

        avg_score = total_score / len(templates) if templates else 0.0

        return OptimizationResult(
            strokes=strokes,
            score=avg_score,
            converged=True,
            iterations=len(templates),
        )

    def _optimize_single_shape(
        self,
        mask: np.ndarray,
        template: str,
        bbox: BBox,
        shape_index: int,
        total_shapes: int
    ) -> tuple[Optional[Stroke], float]:
        """Optimize a single shape.

        Performs local optimization for one template shape.

        Args:
            mask: Binary glyph mask.
            template: Template name to optimize.
            bbox: Glyph bounding box.
            shape_index: Index of this shape in the template list.
            total_shapes: Total number of shapes being optimized.

        Returns:
            Tuple of (optimized Stroke or None, score for this shape).
        """
        # Placeholder - would do actual optimization
        return None, 0.0


@dataclass
class JointRefinementStrategy:
    """Joint refinement of all shapes together.

    Uses differential evolution or similar global optimizer to
    refine all shape parameters simultaneously. This is typically
    the final stage in an optimization pipeline, providing a
    global polish after faster methods have found a good solution.

    Differential evolution is a population-based optimizer that
    explores the parameter space more thoroughly than gradient-based
    methods, but is more computationally expensive.

    Attributes:
        population_size: Number of candidate solutions in the population.
            Default is 15.
        max_iterations: Maximum number of evolutionary generations.
            Default is 100.

    Example:
        >>> strategy = JointRefinementStrategy(population_size=20, max_iterations=50)
        >>> result = strategy.optimize(mask, templates, bbox, initial_params=prev_params)
    """
    population_size: int = 15
    max_iterations: int = 100

    def optimize(
        self,
        mask: np.ndarray,
        templates: List[str],
        bbox: BBox,
        initial_params: Optional[np.ndarray] = None,
        time_budget: float = 5.0,
    ) -> OptimizationResult:
        """Joint optimization of all shapes.

        Uses scipy's differential_evolution to optimize all shape
        parameters simultaneously.

        Args:
            mask: Binary glyph mask.
            templates: List of template names.
            bbox: Glyph bounding box.
            initial_params: Initial parameter vector. Required for this
                strategy to have a starting point. Returns empty result
                if not provided.
            time_budget: Time limit in seconds. Optimization will attempt
                to terminate near this limit.

        Returns:
            OptimizationResult with jointly optimized strokes.
        """
        from scipy.optimize import differential_evolution
        import time

        if initial_params is None or len(templates) == 0:
            return OptimizationResult([], -1.0)

        # Set up bounds based on bbox
        n_params = len(initial_params)
        bounds = self._compute_bounds(n_params, bbox)

        start_time = time.time()

        def objective(params):
            return -self._compute_score(mask, templates, params, bbox)

        def callback(xk, convergence):
            return time.time() - start_time > time_budget

        result = differential_evolution(
            objective,
            bounds,
            x0=initial_params,
            maxiter=self.max_iterations,
            popsize=self.population_size,
            callback=callback,
            polish=False,
        )

        strokes = self._params_to_strokes(result.x, templates, bbox)

        return OptimizationResult(
            strokes=strokes,
            score=-result.fun,
            params=result.x,
            converged=result.success,
            iterations=result.nit,
        )

    def _compute_bounds(self, n_params: int, bbox: BBox) -> List[tuple]:
        """Compute parameter bounds.

        Generates bounds for each parameter based on the bounding box
        dimensions.

        Args:
            n_params: Number of parameters.
            bbox: Glyph bounding box for scale reference.

        Returns:
            List of (min, max) tuples for each parameter.
        """
        # Placeholder bounds
        return [(0, 224)] * n_params

    def _compute_score(
        self,
        mask: np.ndarray,
        templates: List[str],
        params: np.ndarray,
        bbox: BBox
    ) -> float:
        """Compute optimization score.

        Evaluates how well the current parameters fit the glyph mask.

        Args:
            mask: Binary glyph mask.
            templates: List of template names.
            params: Current parameter values.
            bbox: Glyph bounding box.

        Returns:
            Score from 0.0 to 1.0.
        """
        return 0.5  # Placeholder

    def _params_to_strokes(
        self,
        params: np.ndarray,
        templates: List[str],
        bbox: BBox
    ) -> List[Stroke]:
        """Convert parameters to strokes.

        Generates stroke objects from the optimized parameters.

        Args:
            params: Optimized parameter values.
            templates: List of template names.
            bbox: Glyph bounding box.

        Returns:
            List of Stroke objects.
        """
        return []  # Placeholder
