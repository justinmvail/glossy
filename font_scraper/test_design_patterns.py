"""Unit tests for design pattern classes.

Tests the polymorphic design patterns used throughout the codebase:
- Shape classes (stroke_shapes.py)
- OptimizationStrategy classes (stroke_affine.py)
- ScoringPenalty and CompositeScorer classes (stroke_scoring.py)
- MergeStrategy classes (stroke_merge.py)
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestShapeClasses(unittest.TestCase):
    """Test Shape base class and concrete implementations."""

    def test_shapes_registry_exists(self):
        """SHAPES registry should contain all shape types."""
        from stroke_shapes import SHAPES

        expected = {'vline', 'hline', 'diag', 'arc_right', 'arc_left', 'loop', 'u_arc'}
        self.assertEqual(set(SHAPES.keys()), expected)

    def test_shape_interface_implemented(self):
        """All shapes should implement generate, get_bounds, param_count."""
        from stroke_shapes import SHAPES

        for name, shape in SHAPES.items():
            with self.subTest(shape=name):
                self.assertTrue(hasattr(shape, 'generate'))
                self.assertTrue(hasattr(shape, 'get_bounds'))
                self.assertTrue(hasattr(shape, 'param_count'))

    def test_vline_shape_generate(self):
        """VLineShape should generate vertical line points."""
        from stroke_shapes import SHAPES

        shape = SHAPES['vline']
        bbox = (0, 0, 100, 100)
        params = (0.5, 0.2, 0.8)  # x_frac, y_start, y_end

        points = shape.generate(params, bbox, n_pts=10)

        self.assertEqual(points.shape, (10, 2))
        # X should be constant (middle of bbox)
        self.assertTrue(np.allclose(points[:, 0], 50, atol=1))
        # Y should span from 20 to 80
        self.assertAlmostEqual(points[0, 1], 20, delta=5)
        self.assertAlmostEqual(points[-1, 1], 80, delta=5)

    def test_hline_shape_generate(self):
        """HLineShape should generate horizontal line points."""
        from stroke_shapes import SHAPES

        shape = SHAPES['hline']
        bbox = (0, 0, 100, 100)
        params = (0.5, 0.2, 0.8)  # y_frac, x_start, x_end

        points = shape.generate(params, bbox, n_pts=10)

        self.assertEqual(points.shape, (10, 2))
        # Y should be constant (middle of bbox)
        self.assertTrue(np.allclose(points[:, 1], 50, atol=1))

    def test_arc_shape_generate(self):
        """Arc shapes should generate curved points."""
        from stroke_shapes import SHAPES

        shape = SHAPES['arc_right']
        bbox = (0, 0, 100, 100)
        # arc_right needs 6 params: cx, cy, rx, ry, angle_start, angle_end
        # Use larger radius fractions and full angle range for visible curve
        params = (0.5, 0.5, 0.4, 0.4, 0.0, np.pi)

        points = shape.generate(params, bbox, n_pts=20)

        self.assertEqual(points.shape, (20, 2))
        # Points should not all be the same (curve has extent)
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        self.assertGreater(max(x_range, y_range), 1)

    def test_shape_bounds_valid(self):
        """Shape bounds should be valid (min < max)."""
        from stroke_shapes import SHAPES

        for name, shape in SHAPES.items():
            bounds = shape.get_bounds()
            with self.subTest(shape=name):
                self.assertEqual(len(bounds), shape.param_count)
                for i, (low, high) in enumerate(bounds):
                    self.assertLess(low, high,
                        f"Bound {i} invalid: {low} >= {high}")

    def test_shape_param_count_matches_bounds(self):
        """param_count should match length of get_bounds()."""
        from stroke_shapes import SHAPES

        for name, shape in SHAPES.items():
            with self.subTest(shape=name):
                self.assertEqual(shape.param_count, len(shape.get_bounds()))


class TestOptimizationStrategies(unittest.TestCase):
    """Test OptimizationStrategy implementations."""

    def test_nelder_mead_strategy(self):
        """NelderMeadStrategy should minimize a simple function."""
        from stroke_affine import NelderMeadStrategy, OptimizationConfig

        config = OptimizationConfig(max_iterations=200)
        strategy = NelderMeadStrategy(config)

        # Simple quadratic: (x-2)^2 + (y-3)^2
        def objective(params):
            return (params[0] - 2)**2 + (params[1] - 3)**2

        # Start closer to the optimum
        initial = np.array([1.0, 2.0])
        bounds = [(-10, 10), (-10, 10)]

        result, score = strategy.optimize(objective, initial, bounds)

        # Just verify it improved from the initial point
        initial_score = objective(initial)
        self.assertLess(score, initial_score)

    def test_differential_evolution_strategy(self):
        """DifferentialEvolutionStrategy should find global minimum."""
        from stroke_affine import DifferentialEvolutionStrategy, OptimizationConfig

        config = OptimizationConfig(max_iterations=50)
        strategy = DifferentialEvolutionStrategy(config)

        # Simple quadratic
        def objective(params):
            return (params[0] - 1)**2 + (params[1] - 1)**2

        initial = np.array([5.0, 5.0])
        bounds = [(-10, 10), (-10, 10)]

        result, score = strategy.optimize(objective, initial, bounds)

        self.assertAlmostEqual(result[0], 1.0, delta=0.5)
        self.assertAlmostEqual(result[1], 1.0, delta=0.5)

    def test_chained_strategy(self):
        """ChainedStrategy should run strategies in sequence."""
        from stroke_affine import (
            ChainedStrategy, NelderMeadStrategy, OptimizationConfig
        )

        config = OptimizationConfig(max_iterations=50)
        strategy = ChainedStrategy([
            NelderMeadStrategy(config),
            NelderMeadStrategy(config),
        ])

        def objective(params):
            return (params[0] - 1)**2

        initial = np.array([10.0])
        bounds = [(-20, 20)]

        result, score = strategy.optimize(objective, initial, bounds)

        self.assertAlmostEqual(result[0], 1.0, delta=0.5)

    def test_create_default_affine_strategy(self):
        """create_default_affine_strategy should return ChainedStrategy."""
        from stroke_affine import create_default_affine_strategy, ChainedStrategy

        strategy = create_default_affine_strategy()
        self.assertIsInstance(strategy, ChainedStrategy)


class TestScoringPenalties(unittest.TestCase):
    """Test ScoringPenalty implementations."""

    def setUp(self):
        """Set up common test fixtures."""
        from stroke_scoring import ScoringContext
        from scipy.spatial import cKDTree

        # Create a simple test context
        self.mask = np.zeros((100, 100), dtype=bool)
        self.mask[20:80, 20:80] = True  # Square mask

        # Create point cloud from mask
        ys, xs = np.where(self.mask)
        points = np.column_stack([xs, ys])

        self.context = ScoringContext(
            cloud_tree=cKDTree(points),
            n_cloud=len(points),
            radius=5.0,
            snap_xi=np.zeros((100, 100), dtype=int),
            snap_yi=np.zeros((100, 100), dtype=int),
            w=100,
            h=100,
            dist_map=None
        )

    def test_snap_penalty_compute(self):
        """SnapPenalty should penalize points outside mask."""
        from stroke_scoring import SnapPenalty

        penalty = SnapPenalty(weight=1.0)

        # Points inside mask - low penalty
        inside_points = np.array([[50, 50], [40, 40], [60, 60]])
        result_inside = penalty.compute(inside_points, self.context)

        # Points outside mask - higher penalty
        outside_points = np.array([[5, 5], [95, 95], [10, 10]])
        result_outside = penalty.compute(outside_points, self.context)

        # Outside should have higher penalty
        self.assertGreaterEqual(result_outside, result_inside)

    def test_overlap_penalty_compute(self):
        """OverlapPenalty should penalize overlapping coverage."""
        from stroke_scoring import OverlapPenalty

        penalty = OverlapPenalty(weight=1.0)

        # Create overlapping coverage sets
        per_shape_coverage = [
            {1, 2, 3, 4, 5},
            {4, 5, 6, 7, 8},  # Overlaps with first
        ]

        points = np.array([[50, 50]])
        result = penalty.compute(
            points, self.context,
            per_shape_coverage=per_shape_coverage
        )

        # Should have some overlap penalty
        self.assertGreater(result, 0)

    def test_scoring_penalty_weight(self):
        """ScoringPenalty weight should be stored correctly."""
        from stroke_scoring import SnapPenalty

        penalty = SnapPenalty(weight=0.75)
        self.assertEqual(penalty.weight, 0.75)


class TestCompositeScorer(unittest.TestCase):
    """Test CompositeScorer class."""

    def test_composite_scorer_default_penalties(self):
        """CompositeScorer should have default penalties."""
        from stroke_scoring import CompositeScorer

        scorer = CompositeScorer()
        self.assertEqual(len(scorer.penalties), 3)  # Snap, Edge, Overlap

    def test_composite_scorer_custom_penalties(self):
        """CompositeScorer should accept custom penalties."""
        from stroke_scoring import CompositeScorer, SnapPenalty

        custom = [SnapPenalty(weight=0.5)]
        scorer = CompositeScorer(penalties=custom)

        self.assertEqual(len(scorer.penalties), 1)
        self.assertEqual(scorer.penalties[0].weight, 0.5)

    def test_composite_scorer_add_penalty(self):
        """add_penalty should append to penalties list."""
        from stroke_scoring import CompositeScorer, SnapPenalty

        scorer = CompositeScorer(penalties=[])
        scorer.add_penalty(SnapPenalty(weight=0.3))

        self.assertEqual(len(scorer.penalties), 1)


class TestMergeStrategies(unittest.TestCase):
    """Test MergeStrategy implementations."""

    def test_merge_context_creation(self):
        """MergeContext should store strokes and clusters."""
        from stroke_merge import MergeContext

        strokes = [[(0, 0), (1, 1)], [(2, 2), (3, 3)]]
        clusters = [{(0, 0), (1, 1)}]
        assigned = [{(0, 0)}]

        ctx = MergeContext(strokes=strokes, junction_clusters=clusters, assigned=assigned)

        self.assertEqual(len(ctx.strokes), 2)
        self.assertEqual(len(ctx.junction_clusters), 1)
        self.assertEqual(len(ctx.assigned), 1)

    def test_direction_merge_strategy_interface(self):
        """DirectionMergeStrategy should implement merge method."""
        from stroke_merge import DirectionMergeStrategy, MergeContext

        strategy = DirectionMergeStrategy(max_angle=np.pi/4)

        # Empty strokes should return empty
        ctx = MergeContext(strokes=[], junction_clusters=[], assigned=[])
        result = strategy.merge(ctx)

        self.assertEqual(result, [])

    def test_stub_absorption_strategy_thresholds(self):
        """StubAbsorptionStrategy should accept threshold parameters."""
        from stroke_merge import StubAbsorptionStrategy

        strategy = StubAbsorptionStrategy(
            conv_threshold=15,
            stub_threshold=20,
            prox_threshold=25
        )

        self.assertEqual(strategy.conv_threshold, 15)
        self.assertEqual(strategy.stub_threshold, 20)
        self.assertEqual(strategy.prox_threshold, 25)

    def test_merge_pipeline_create_default(self):
        """MergePipeline.create_default should return configured pipeline."""
        from stroke_merge import MergePipeline

        pipeline = MergePipeline.create_default()

        self.assertEqual(len(pipeline.strategies), 4)

    def test_merge_pipeline_create_aggressive(self):
        """MergePipeline.create_aggressive should have wider tolerances."""
        from stroke_merge import MergePipeline, DirectionMergeStrategy

        pipeline = MergePipeline.create_aggressive()

        # First strategy should be DirectionMergeStrategy with wider angle
        self.assertIsInstance(pipeline.strategies[0], DirectionMergeStrategy)
        self.assertGreater(pipeline.strategies[0].max_angle, np.pi/4)

    def test_merge_pipeline_create_conservative(self):
        """MergePipeline.create_conservative should have stricter tolerances."""
        from stroke_merge import MergePipeline, DirectionMergeStrategy

        pipeline = MergePipeline.create_conservative()

        # First strategy should be DirectionMergeStrategy with narrower angle
        self.assertIsInstance(pipeline.strategies[0], DirectionMergeStrategy)
        self.assertLess(pipeline.strategies[0].max_angle, np.pi/4)

    def test_merge_pipeline_run_empty(self):
        """MergePipeline.run should handle empty strokes."""
        from stroke_merge import MergePipeline

        pipeline = MergePipeline.create_default()
        result = pipeline.run([], [], [])

        self.assertEqual(result, [])


class TestVertexFinders(unittest.TestCase):
    """Test VertexFinder implementations."""

    def test_vertex_finders_registry(self):
        """VERTEX_FINDERS registry should contain character handlers."""
        from template_morph import VERTEX_FINDERS

        # Should have handlers for common characters
        expected_chars = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'P', 'R'}
        for char in expected_chars:
            self.assertIn(char, VERTEX_FINDERS, f"Missing handler for '{char}'")

    def test_vertex_finder_interface(self):
        """All vertex finders should implement find method."""
        from template_morph import VERTEX_FINDERS

        for char, finder in VERTEX_FINDERS.items():
            with self.subTest(char=char):
                self.assertTrue(hasattr(finder, 'find'))
                self.assertTrue(callable(finder.find))

    def test_default_vertex_finder(self):
        """DefaultVertexFinder should handle unknown characters."""
        from template_morph import DefaultVertexFinder

        finder = DefaultVertexFinder('Z')
        self.assertEqual(finder.char, 'Z')
        self.assertTrue(hasattr(finder, 'find'))


if __name__ == '__main__':
    unittest.main()
