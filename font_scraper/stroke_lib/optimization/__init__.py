"""Stroke optimization strategies.

This module provides optimization strategies for fitting stroke templates
to glyph masks. The optimization system uses a pipeline architecture where
multiple strategies can be chained together, each refining the results of
the previous stage.

The module exports the following classes:

Strategy classes:
    OptimizationStrategy: Protocol defining the interface for strategies.
    OptimizationResult: Data class containing optimization results.
    AffineStrategy: Global affine transformation optimization.
    GreedyStrategy: Per-shape greedy optimization.
    JointRefinementStrategy: Global joint refinement using differential
        evolution.

Orchestrator:
    StrokeOptimizer: Coordinates multiple strategies in a pipeline.

Example usage:
    Using the default optimizer::

        from stroke_lib.optimization import create_default_optimizer

        optimizer = create_default_optimizer()
        result = optimizer.optimize(
            mask=glyph_mask,
            templates=['vertical_stroke', 'horizontal_stroke'],
            bbox=glyph_bbox,
            time_budget=5.0
        )

        print(f"Score: {result.score}")
        for stroke in result.strokes:
            print(f"Stroke with {len(stroke)} points")

    Building a custom optimizer::

        from stroke_lib.optimization import (
            StrokeOptimizer, AffineStrategy, GreedyStrategy
        )

        optimizer = StrokeOptimizer()
        optimizer.add_strategy(AffineStrategy(max_iterations=100))
        optimizer.add_strategy(GreedyStrategy(max_iterations_per_shape=50))

        result = optimizer.optimize(mask, templates, bbox)
"""

from .optimizer import StrokeOptimizer, create_default_optimizer
from .strategies import (
    AffineStrategy,
    GreedyStrategy,
    JointRefinementStrategy,
    OptimizationResult,
    OptimizationStrategy,
)

__all__ = [
    'OptimizationStrategy',
    'OptimizationResult',
    'AffineStrategy',
    'GreedyStrategy',
    'JointRefinementStrategy',
    'StrokeOptimizer',
    'create_default_optimizer',
]
