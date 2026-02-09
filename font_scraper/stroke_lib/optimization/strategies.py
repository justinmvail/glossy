"""Optimization strategy implementations."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Protocol
import numpy as np

from ..domain.geometry import Stroke, BBox


@dataclass
class OptimizationResult:
    """Result of an optimization strategy."""
    strokes: List[Stroke]
    score: float
    params: Optional[np.ndarray] = None
    converged: bool = False
    iterations: int = 0


class OptimizationStrategy(Protocol):
    """Protocol for optimization strategies."""

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
            mask: Binary glyph mask
            templates: List of shape template names
            bbox: Glyph bounding box
            initial_params: Optional initial parameter vector
            time_budget: Time limit in seconds

        Returns:
            OptimizationResult with best strokes found
        """
        ...


@dataclass
class AffineStrategy:
    """Optimization via affine transformation of template strokes.

    Applies global translation, rotation, and scaling to match templates
    to the glyph mask.
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
        """Optimize via affine transformation."""
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
        """Compute mask coverage for given parameters."""
        # Simplified coverage computation
        return 0.5  # Placeholder

    def _params_to_strokes(
        self,
        params: np.ndarray,
        templates: List[str],
        bbox: BBox
    ) -> List[Stroke]:
        """Convert parameters to stroke objects."""
        # Placeholder - would transform template shapes
        return []


@dataclass
class GreedyStrategy:
    """Greedy per-shape optimization.

    Optimizes each shape independently to maximize local coverage.
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
        """Optimize each shape greedily."""
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
        """Optimize a single shape."""
        # Placeholder - would do actual optimization
        return None, 0.0


@dataclass
class JointRefinementStrategy:
    """Joint refinement of all shapes together.

    Uses differential evolution or similar global optimizer to
    refine all shape parameters simultaneously.
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
        """Joint optimization of all shapes."""
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
        """Compute parameter bounds."""
        # Placeholder bounds
        return [(0, 224)] * n_params

    def _compute_score(
        self,
        mask: np.ndarray,
        templates: List[str],
        params: np.ndarray,
        bbox: BBox
    ) -> float:
        """Compute optimization score."""
        return 0.5  # Placeholder

    def _params_to_strokes(
        self,
        params: np.ndarray,
        templates: List[str],
        bbox: BBox
    ) -> List[Stroke]:
        """Convert parameters to strokes."""
        return []  # Placeholder
