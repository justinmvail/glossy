"""Stroke optimizer orchestrating multiple strategies.

This module provides the StrokeOptimizer class which coordinates multiple
optimization strategies in a pipeline. Each strategy refines the results
of the previous one, allowing for a progression from fast global methods
to more precise local refinement.

The optimizer manages time budgets, tracks the best results, and supports
early termination when a satisfactory score is achieved.

Example usage:
    Using the default optimizer::

        from stroke_lib.optimization.optimizer import create_default_optimizer

        optimizer = create_default_optimizer(
            progress_callback=lambda r, name: print(f"{name}: {r.score:.3f}")
        )
        result = optimizer.optimize(mask, templates, bbox, time_budget=10.0)

    Building a custom pipeline::

        from stroke_lib.optimization.optimizer import StrokeOptimizer
        from stroke_lib.optimization.strategies import AffineStrategy

        optimizer = StrokeOptimizer(score_threshold=0.90)
        optimizer.add_strategy(AffineStrategy(max_iterations=200))
        result = optimizer.optimize(mask, templates, bbox)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..domain.geometry import BBox
from .strategies import OptimizationResult, OptimizationStrategy


@dataclass
class StrokeOptimizer:
    """Orchestrates multiple optimization strategies.

    Runs strategies in sequence, using the best result from each
    as the starting point for the next. The optimizer manages time
    allocation across strategies and supports early termination
    when a satisfactory score threshold is reached.

    The optimization pipeline typically progresses from:
        1. Fast global methods (e.g., affine transformation)
        2. Per-shape refinement (e.g., greedy optimization)
        3. Global polish (e.g., joint refinement)

    Attributes:
        strategies: List of OptimizationStrategy instances to run in order.
        score_threshold: Stop early if score exceeds this value. Default is
            0.95, meaning optimization stops when 95% coverage is achieved.
        progress_callback: Optional function called after each strategy
            completes. Receives (result, strategy_name) arguments.

    Example:
        >>> optimizer = StrokeOptimizer(score_threshold=0.90)
        >>> optimizer.add_strategy(AffineStrategy())
        >>> optimizer.add_strategy(GreedyStrategy())
        >>> result = optimizer.optimize(mask, templates, bbox)
    """
    strategies: list[OptimizationStrategy] = field(default_factory=list)
    score_threshold: float = 0.95  # Stop early if score exceeds this
    progress_callback: Callable[[OptimizationResult, str], None] | None = None

    def optimize(
        self,
        mask: np.ndarray,
        templates: list[str],
        bbox: BBox,
        time_budget: float = 10.0,
    ) -> OptimizationResult:
        """Run all strategies and return best result.

        Executes each strategy in sequence, passing the best parameters
        from each stage to the next. Time is allocated proportionally
        among remaining strategies.

        Args:
            mask: Binary glyph mask as numpy array of shape (H, W).
                True/non-zero values indicate glyph pixels.
            templates: List of shape template names to fit to the mask.
            bbox: Bounding box of the glyph for coordinate reference.
            time_budget: Total time limit in seconds for all strategies.
                Default is 10.0 seconds. Time is distributed proportionally
                among strategies.

        Returns:
            OptimizationResult containing:
                - strokes: List of optimized Stroke objects
                - score: Best coverage score achieved (0.0 to 1.0)
                - params: Parameter vector for the best solution
                - converged: Whether optimization converged successfully
                - iterations: Total iterations across all strategies
        """
        start_time = time.time()
        best_result = OptimizationResult([], -float('inf'))
        current_params = None

        for i, strategy in enumerate(self.strategies):
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed

            if remaining <= 0:
                break

            # Allocate time proportionally to remaining strategies
            strategy_budget = remaining / (len(self.strategies) - i)

            result = strategy.optimize(
                mask=mask,
                templates=templates,
                bbox=bbox,
                initial_params=current_params,
                time_budget=strategy_budget,
            )

            if result.score > best_result.score:
                best_result = result
                current_params = result.params

            if self.progress_callback:
                strategy_name = type(strategy).__name__
                self.progress_callback(result, strategy_name)

            # Early termination if we hit threshold
            if best_result.score >= self.score_threshold:
                break

        return best_result

    def add_strategy(self, strategy: OptimizationStrategy) -> StrokeOptimizer:
        """Add a strategy to the pipeline (fluent interface).

        Appends a strategy to the end of the optimization pipeline.
        Returns self to allow method chaining.

        Args:
            strategy: OptimizationStrategy instance to add.

        Returns:
            Self for method chaining.

        Example:
            >>> optimizer = StrokeOptimizer()
            >>> optimizer.add_strategy(AffineStrategy()).add_strategy(GreedyStrategy())
        """
        self.strategies.append(strategy)
        return self


def create_default_optimizer(
    progress_callback: Callable | None = None
) -> StrokeOptimizer:
    """Create optimizer with default strategy pipeline.

    Creates a StrokeOptimizer with a recommended three-stage pipeline:
        1. AffineStrategy: Fast global alignment using affine transformation
        2. GreedyStrategy: Per-shape refinement for local optimization
        3. JointRefinementStrategy: Global polish using differential evolution

    This pipeline provides a good balance between speed and quality for
    most glyph optimization tasks.

    Args:
        progress_callback: Optional callback function called after each
            strategy completes. The callback receives (result, strategy_name)
            arguments, where result is an OptimizationResult and strategy_name
            is a string.

    Returns:
        Configured StrokeOptimizer ready for use.

    Example:
        >>> def on_progress(result, name):
        ...     print(f"{name} completed with score {result.score:.3f}")
        >>> optimizer = create_default_optimizer(progress_callback=on_progress)
        >>> result = optimizer.optimize(mask, templates, bbox)
    """
    from .strategies import AffineStrategy, GreedyStrategy, JointRefinementStrategy

    return StrokeOptimizer(
        strategies=[
            AffineStrategy(max_iterations=50),
            GreedyStrategy(max_iterations_per_shape=30),
            JointRefinementStrategy(population_size=10, max_iterations=50),
        ],
        progress_callback=progress_callback,
    )
