"""Stroke optimizer orchestrating multiple strategies."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import time
import numpy as np

from ..domain.geometry import Stroke, BBox
from .strategies import OptimizationStrategy, OptimizationResult


@dataclass
class StrokeOptimizer:
    """Orchestrates multiple optimization strategies.

    Runs strategies in sequence, using the best result from each
    as the starting point for the next.
    """
    strategies: List[OptimizationStrategy] = field(default_factory=list)
    score_threshold: float = 0.95  # Stop early if score exceeds this
    progress_callback: Optional[Callable[[OptimizationResult, str], None]] = None

    def optimize(
        self,
        mask: np.ndarray,
        templates: List[str],
        bbox: BBox,
        time_budget: float = 10.0,
    ) -> OptimizationResult:
        """Run all strategies and return best result.

        Args:
            mask: Binary glyph mask
            templates: List of shape template names
            bbox: Glyph bounding box
            time_budget: Total time limit in seconds

        Returns:
            Best OptimizationResult found
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

    def add_strategy(self, strategy: OptimizationStrategy) -> 'StrokeOptimizer':
        """Add a strategy to the pipeline (fluent interface)."""
        self.strategies.append(strategy)
        return self


def create_default_optimizer(
    progress_callback: Optional[Callable] = None
) -> StrokeOptimizer:
    """Create optimizer with default strategy pipeline.

    Default order:
    1. Affine (fast global alignment)
    2. Greedy (per-shape refinement)
    3. Joint refinement (global polish)
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
