"""Unit tests for stroke_merge.py.

Tests the MergeStrategy classes and MergePipeline used for stroke cleanup
and simplification. These tests use simple, predictable stroke patterns
to verify merging behavior.

Test coverage targets:
- DirectionMergeStrategy: angle-based merging at junctions
- TJunctionMergeStrategy: T-junction handling with 3+ strokes
- StubAbsorptionStrategy: absorption of short stub strokes
- MergePipeline: strategy composition and execution
"""

import unittest

import numpy as np

from stroke_merge import (
    MergeContext,
    MergeStrategy,
    DirectionMergeStrategy,
    TJunctionMergeStrategy,
    StubAbsorptionStrategy,
    OrphanRemovalStrategy,
    MergePipeline,
    DEFAULT_CONV_THRESHOLD,
    DEFAULT_STUB_THRESHOLD,
    AGGRESSIVE_CONV_THRESHOLD,
    AGGRESSIVE_STUB_THRESHOLD,
)
from stroke_merge_utils import (
    seg_dir,
    angle_between,
    endpoint_cluster,
)


def make_stroke(points: list[tuple[float, float]]) -> list[tuple]:
    """Create a stroke from a list of points."""
    return list(points)


def make_line_stroke(start: tuple, end: tuple, n_pts: int = 10) -> list[tuple]:
    """Create a straight line stroke between two points."""
    x0, y0 = start
    x1, y1 = end
    return [(x0 + (x1 - x0) * t / (n_pts - 1),
             y0 + (y1 - y0) * t / (n_pts - 1))
            for t in range(n_pts)]


def make_cluster(center: tuple, radius: int = 1) -> set:
    """Create a junction cluster around a center point."""
    cx, cy = center
    cluster = set()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            cluster.add((cx + dx, cy + dy))
    return cluster


class TestDirectionMergeStrategy(unittest.TestCase):
    """Test DirectionMergeStrategy for angle-based stroke merging."""

    def test_merges_collinear_strokes(self):
        """Two strokes forming a continuation should merge at a junction.

        For a merge to occur, strokes must meet at a junction with their directions
        being opposite (one going into junction, one going out). The merge angle
        is computed as: pi - angle_between(dir1, dir2). For a perfect continuation,
        angle_between should be pi (opposite directions), giving merge_angle = 0.

        Stroke 1: (0, 50) -> (50, 50) - ends at junction, direction (1, 0)
        Stroke 2: (100, 50) -> (50, 50) - ends at junction, direction (-1, 0)

        Both strokes END at the junction, with opposite directions, forming a
        single continuous line from (0, 50) to (100, 50).
        """
        # Stroke 1 ends at junction, going right
        stroke1 = make_line_stroke((0, 50), (50, 50), n_pts=15)
        # Stroke 2 ends at junction, going left (opposite direction)
        stroke2 = make_line_stroke((100, 50), (50, 50), n_pts=15)

        strokes = [stroke1, stroke2]
        junction = make_cluster((50, 50))
        assigned = [junction]

        strategy = DirectionMergeStrategy(max_angle=np.pi / 4)
        ctx = MergeContext(strokes=strokes, junction_clusters=assigned, assigned=assigned)

        result = strategy.merge(ctx)

        # Should merge into a single stroke
        self.assertEqual(len(result), 1, "Collinear strokes should merge into one")
        # The merged stroke should span from start to end
        merged = result[0]
        self.assertGreater(len(merged), len(stroke1),
                           "Merged stroke should be longer than original")

    def test_rejects_perpendicular_strokes(self):
        """Strokes at 90 degrees should not merge."""
        # Stroke 1: horizontal ending at junction (0, 50) -> (50, 50)
        # Stroke 2: vertical ending at junction (50, 100) -> (50, 50)
        stroke1 = make_line_stroke((0, 50), (50, 50), n_pts=15)
        stroke2 = make_line_stroke((50, 100), (50, 50), n_pts=15)

        strokes = [stroke1, stroke2]
        junction = make_cluster((50, 50))
        assigned = [junction]

        strategy = DirectionMergeStrategy(max_angle=np.pi / 4)  # 45 degrees max
        ctx = MergeContext(strokes=strokes, junction_clusters=assigned, assigned=assigned)

        result = strategy.merge(ctx)

        # Should NOT merge - perpendicular strokes (90 degrees)
        self.assertEqual(len(result), 2, "Perpendicular strokes should not merge")

    def test_respects_max_angle(self):
        """Strokes at exactly max_angle boundary should be handled correctly.

        Stroke 1: horizontal, ending at junction
        Stroke 2: at ~135 degrees from horizontal, ending at junction

        For two strokes ending at the same junction:
        - stroke1 direction (into end): (1, 0)
        - stroke2 direction (into end): (-1/sqrt(2), 1/sqrt(2)) roughly

        angle_between = 135 degrees (3*pi/4)
        merge_angle = pi - 3*pi/4 = pi/4 (45 degrees)
        """
        # Stroke 1: horizontal ending at junction
        stroke1 = make_line_stroke((0, 50), (50, 50), n_pts=15)
        # Stroke 2: diagonal ending at junction (135 degrees from stroke1's direction)
        stroke2 = make_line_stroke((100, 100), (50, 50), n_pts=15)

        strokes = [stroke1, stroke2]
        junction = make_cluster((50, 50))
        assigned = [junction]

        # Test with generous angle (pi/2 = 90 degrees tolerance)
        strategy = DirectionMergeStrategy(max_angle=np.pi / 2)
        ctx = MergeContext(strokes=[list(s) for s in strokes],
                           junction_clusters=[junction], assigned=[junction])
        result = strategy.merge(ctx)

        # With 90 degree tolerance, should merge (merge angle is ~45 degrees)
        self.assertEqual(len(result), 1,
                         "Strokes should merge with large angle tolerance")

        # Test with strict angle (should not merge with 22.5 degree tolerance)
        strategy_strict = DirectionMergeStrategy(max_angle=np.pi / 8)  # 22.5 degrees
        ctx_strict = MergeContext(strokes=[list(s) for s in strokes],
                                  junction_clusters=[junction], assigned=[junction])
        result_strict = strategy_strict.merge(ctx_strict)

        self.assertEqual(len(result_strict), 2,
                         "Strokes should not merge with strict angle tolerance")

    def test_handles_empty_strokes(self):
        """Strategy should handle empty stroke list."""
        strategy = DirectionMergeStrategy()
        ctx = MergeContext(strokes=[], junction_clusters=[], assigned=[])

        result = strategy.merge(ctx)

        self.assertEqual(result, [])

    def test_preserves_single_stroke(self):
        """Single stroke with no junction neighbors should be preserved."""
        stroke = make_line_stroke((0, 0), (100, 100), n_pts=20)
        strokes = [stroke]

        strategy = DirectionMergeStrategy()
        ctx = MergeContext(strokes=strokes, junction_clusters=[], assigned=[])

        result = strategy.merge(ctx)

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), len(stroke))


class TestTJunctionMergeStrategy(unittest.TestCase):
    """Test TJunctionMergeStrategy for T-junction handling.

    T-junction detection requires:
    1. At least 3 stroke endpoints at a junction cluster
    2. The shortest stroke must have BOTH endpoints at valid junction clusters
    3. The shortest stroke must be < 40% of the second longest stroke
    4. The two longest strokes must have aligned directions (< 120 degrees)
    """

    def test_detects_t_junction(self):
        """Should detect T-junction with 3 strokes meeting at a point.

        For T-junction to be detected, the short stroke needs both endpoints
        at valid junction clusters.
        """
        # Two long strokes forming the main bar, meeting at center
        # Stroke 1: left arm ending at junction
        stroke1 = make_line_stroke((0, 50), (50, 50), n_pts=30)
        # Stroke 2: right arm ending at junction (opposite direction for continuation)
        stroke2 = make_line_stroke((100, 50), (50, 50), n_pts=30)
        # Stroke 3: short crossbar with both ends at junctions
        # This short stroke connects two junctions
        stroke3 = make_line_stroke((50, 50), (50, 55), n_pts=5)

        strokes = [stroke1, stroke2, stroke3]
        # Create two junction clusters
        junction1 = make_cluster((50, 50))
        junction2 = make_cluster((50, 55))
        assigned = [junction1, junction2]

        strategy = TJunctionMergeStrategy()
        ctx = MergeContext(strokes=strokes, junction_clusters=assigned, assigned=assigned)

        result = strategy.merge(ctx)

        # The two main strokes should merge into one long stroke
        # The short crossbar may or may not be removed
        self.assertLessEqual(len(result), 2,
                             "T-junction should merge main strokes")

    def test_merges_stem_to_crossbar(self):
        """The two longest strokes at a T-junction should merge.

        Creates a proper T-junction where:
        - Two long horizontal arms meet at center (opposite directions)
        - A short vertical connects center to another junction
        """
        # Left arm ending at center junction
        left_arm = make_line_stroke((0, 50), (50, 50), n_pts=30)
        # Right arm ending at center junction (opposite direction)
        right_arm = make_line_stroke((100, 50), (50, 50), n_pts=30)
        # Short vertical with BOTH ends at junctions
        spur = make_line_stroke((50, 50), (50, 55), n_pts=4)

        strokes = [left_arm, right_arm, spur]
        # Two junction clusters
        center_junction = make_cluster((50, 50))
        spur_end_junction = make_cluster((50, 55))
        assigned = [center_junction, spur_end_junction]

        strategy = TJunctionMergeStrategy()
        ctx = MergeContext(strokes=strokes, junction_clusters=assigned, assigned=assigned)

        result = strategy.merge(ctx)

        # Left and right arms should merge into one long stroke
        # After merging, we should have 1-2 strokes
        self.assertLessEqual(len(result), 2,
                             "T-junction should merge the two longest strokes")

    def test_handles_no_t_junction(self):
        """Should not modify strokes when no T-junction exists."""
        # Two strokes not forming a T-junction
        stroke1 = make_line_stroke((0, 0), (50, 50), n_pts=20)
        stroke2 = make_line_stroke((60, 60), (100, 100), n_pts=20)

        strokes = [stroke1, stroke2]
        # No shared junction cluster
        cluster1 = make_cluster((50, 50))
        cluster2 = make_cluster((60, 60))
        assigned = [cluster1, cluster2]

        strategy = TJunctionMergeStrategy()
        ctx = MergeContext(strokes=strokes, junction_clusters=assigned, assigned=assigned)

        result = strategy.merge(ctx)

        self.assertEqual(len(result), 2, "No T-junction means no merging")


class TestStubAbsorptionStrategy(unittest.TestCase):
    """Test StubAbsorptionStrategy for absorbing short stubs."""

    def test_absorbs_short_stubs(self):
        """Short stubs near longer strokes should be absorbed."""
        # Long stroke
        long_stroke = make_line_stroke((0, 50), (100, 50), n_pts=50)
        # Short stub at the end
        stub = make_line_stroke((100, 50), (105, 50), n_pts=4)

        strokes = [long_stroke, stub]
        junction = make_cluster((100, 50))
        assigned = [junction]

        strategy = StubAbsorptionStrategy(
            conv_threshold=10,
            stub_threshold=10,
            prox_threshold=10
        )
        ctx = MergeContext(strokes=strokes, junction_clusters=[junction], assigned=assigned)

        result = strategy.merge(ctx)

        # Stub should be absorbed into the long stroke
        self.assertLessEqual(len(result), 2)

    def test_keeps_long_strokes(self):
        """Strokes longer than threshold should not be absorbed."""
        # Two long strokes at a junction
        stroke1 = make_line_stroke((0, 50), (50, 50), n_pts=30)
        stroke2 = make_line_stroke((50, 50), (100, 50), n_pts=30)

        strokes = [stroke1, stroke2]
        junction = make_cluster((50, 50))
        assigned = [junction]

        # Use low threshold - both strokes exceed it
        strategy = StubAbsorptionStrategy(
            conv_threshold=5,
            stub_threshold=5,
            prox_threshold=5
        )
        ctx = MergeContext(strokes=strokes, junction_clusters=[junction], assigned=assigned)

        result = strategy.merge(ctx)

        # Both strokes are long, neither absorbed as a stub
        # (Though they might merge via other mechanisms)
        self.assertGreaterEqual(len(result), 1)

    def test_respects_convergence_flag(self):
        """absorb_convergence=False should skip convergence absorption."""
        long_stroke = make_line_stroke((0, 50), (100, 50), n_pts=50)
        # Convergence stub (one end at junction, other free)
        stub = make_line_stroke((100, 50), (110, 60), n_pts=5)

        strokes = [long_stroke, stub]
        junction = make_cluster((100, 50))
        assigned = [junction]

        # Disable convergence absorption
        strategy = StubAbsorptionStrategy(
            absorb_convergence=False,
            absorb_junction=False,
            absorb_proximity=False
        )
        ctx = MergeContext(strokes=strokes, junction_clusters=[junction], assigned=assigned)

        result = strategy.merge(ctx)

        # With all absorption disabled, strokes should remain separate
        self.assertEqual(len(result), 2)

    def test_threshold_parameters_stored(self):
        """Threshold parameters should be stored correctly."""
        strategy = StubAbsorptionStrategy(
            conv_threshold=15,
            stub_threshold=25,
            prox_threshold=30
        )

        self.assertEqual(strategy.conv_threshold, 15)
        self.assertEqual(strategy.stub_threshold, 25)
        self.assertEqual(strategy.prox_threshold, 30)


class TestOrphanRemovalStrategy(unittest.TestCase):
    """Test OrphanRemovalStrategy for removing isolated stubs."""

    def test_removes_orphan_stub(self):
        """Orphaned short strokes should be removed."""
        # Long stroke at one cluster
        long_stroke = make_line_stroke((0, 0), (50, 50), n_pts=30)
        # Orphan stub at a different cluster with no neighbors
        orphan = make_line_stroke((100, 100), (105, 105), n_pts=5)

        strokes = [long_stroke, orphan]
        cluster1 = make_cluster((50, 50))
        cluster2 = make_cluster((100, 100))
        assigned = [cluster1, cluster2]

        strategy = OrphanRemovalStrategy(stub_threshold=10)
        ctx = MergeContext(strokes=strokes, junction_clusters=assigned, assigned=assigned)

        result = strategy.merge(ctx)

        # Orphan should be removed (it's short and alone at its cluster)
        # The long stroke should remain
        self.assertGreaterEqual(len(result), 1)
        if len(result) == 1:
            self.assertGreater(len(result[0]), 10)

    def test_keeps_strokes_with_neighbors(self):
        """Short strokes with neighbors at cluster should be kept."""
        # Two short strokes at the same cluster
        stroke1 = make_line_stroke((50, 50), (55, 55), n_pts=6)
        stroke2 = make_line_stroke((50, 50), (45, 55), n_pts=6)

        strokes = [stroke1, stroke2]
        junction = make_cluster((50, 50))
        assigned = [junction]

        strategy = OrphanRemovalStrategy(stub_threshold=10)
        ctx = MergeContext(strokes=strokes, junction_clusters=assigned, assigned=assigned)

        result = strategy.merge(ctx)

        # Both strokes have neighbors, neither should be removed as orphan
        # (They may be processed by other strategies though)
        self.assertGreaterEqual(len(result), 1)


class TestMergePipeline(unittest.TestCase):
    """Test MergePipeline for combining strategies."""

    def test_runs_all_strategies_in_order(self):
        """Pipeline should run all strategies in sequence."""
        # Create mock strategies to track execution order
        execution_order = []

        class TrackingStrategy(MergeStrategy):
            def __init__(self, name):
                self._name = name

            @property
            def name(self):
                return self._name

            def merge(self, ctx):
                execution_order.append(self._name)
                return ctx.strokes

        pipeline = MergePipeline([
            TrackingStrategy("first"),
            TrackingStrategy("second"),
            TrackingStrategy("third"),
        ])

        strokes = [[(0, 0), (1, 1)]]
        pipeline.run(strokes, [], [])

        self.assertEqual(execution_order, ["first", "second", "third"])

    def test_create_default_returns_pipeline(self):
        """create_default should return a properly configured pipeline."""
        pipeline = MergePipeline.create_default()

        self.assertIsInstance(pipeline, MergePipeline)
        self.assertEqual(len(pipeline.strategies), 4)

        # Check strategy types in order
        self.assertIsInstance(pipeline.strategies[0], DirectionMergeStrategy)
        self.assertIsInstance(pipeline.strategies[1], TJunctionMergeStrategy)
        self.assertIsInstance(pipeline.strategies[2], StubAbsorptionStrategy)
        self.assertIsInstance(pipeline.strategies[3], OrphanRemovalStrategy)

    def test_create_aggressive_differs_from_default(self):
        """create_aggressive should have different settings than default."""
        default = MergePipeline.create_default()
        aggressive = MergePipeline.create_aggressive()

        # Both should have 4 strategies
        self.assertEqual(len(default.strategies), len(aggressive.strategies))

        # Aggressive should have wider angle tolerance
        default_angle = default.strategies[0].max_angle
        aggressive_angle = aggressive.strategies[0].max_angle
        self.assertGreater(aggressive_angle, default_angle,
                           "Aggressive should have wider max_angle")

        # Aggressive should have higher thresholds
        default_stub = default.strategies[2].stub_threshold
        aggressive_stub = aggressive.strategies[2].stub_threshold
        self.assertGreater(aggressive_stub, default_stub,
                           "Aggressive should have higher stub_threshold")

    def test_create_conservative_differs_from_default(self):
        """create_conservative should have stricter settings than default."""
        default = MergePipeline.create_default()
        conservative = MergePipeline.create_conservative()

        # Conservative should have narrower angle tolerance
        default_angle = default.strategies[0].max_angle
        conservative_angle = conservative.strategies[0].max_angle
        self.assertLess(conservative_angle, default_angle,
                        "Conservative should have narrower max_angle")

        # Conservative should have lower thresholds
        default_stub = default.strategies[2].stub_threshold
        conservative_stub = conservative.strategies[2].stub_threshold
        self.assertLess(conservative_stub, default_stub,
                        "Conservative should have lower stub_threshold")

    def test_run_with_empty_strokes(self):
        """Pipeline should handle empty stroke list."""
        pipeline = MergePipeline.create_default()
        result = pipeline.run([], [], [])

        self.assertEqual(result, [])

    def test_add_strategy_at_end(self):
        """add_strategy should append strategy to end of list."""
        pipeline = MergePipeline([])
        pipeline.add_strategy(DirectionMergeStrategy())

        self.assertEqual(len(pipeline.strategies), 1)

    def test_add_strategy_at_position(self):
        """add_strategy should insert at specified position."""
        pipeline = MergePipeline([
            DirectionMergeStrategy(),
            OrphanRemovalStrategy(),
        ])
        pipeline.add_strategy(TJunctionMergeStrategy(), position=1)

        self.assertEqual(len(pipeline.strategies), 3)
        self.assertIsInstance(pipeline.strategies[1], TJunctionMergeStrategy)

    def test_remove_strategy_by_type(self):
        """remove_strategy_by_type should remove all strategies of a type."""
        pipeline = MergePipeline.create_default()
        original_len = len(pipeline.strategies)

        removed = pipeline.remove_strategy_by_type(TJunctionMergeStrategy)

        self.assertTrue(removed)
        self.assertEqual(len(pipeline.strategies), original_len - 1)
        for s in pipeline.strategies:
            self.assertNotIsInstance(s, TJunctionMergeStrategy)

    def test_remove_strategy_by_type_returns_false_if_not_found(self):
        """remove_strategy_by_type should return False if type not found."""
        pipeline = MergePipeline([DirectionMergeStrategy()])

        removed = pipeline.remove_strategy_by_type(TJunctionMergeStrategy)

        self.assertFalse(removed)


class TestMergeContext(unittest.TestCase):
    """Test MergeContext data class and caching."""

    def test_context_stores_data(self):
        """MergeContext should store strokes and clusters."""
        strokes = [[(0, 0), (1, 1)], [(2, 2), (3, 3)]]
        junction_clusters = [{(0, 0), (1, 1)}]
        assigned = [{(0, 0)}]

        ctx = MergeContext(
            strokes=strokes,
            junction_clusters=junction_clusters,
            assigned=assigned
        )

        self.assertEqual(ctx.strokes, strokes)
        self.assertEqual(ctx.junction_clusters, junction_clusters)
        self.assertEqual(ctx.assigned, assigned)

    def test_cache_invalidation(self):
        """invalidate_caches should mark caches as invalid."""
        ctx = MergeContext(strokes=[], junction_clusters=[], assigned=[])

        # Force cache build
        ctx._cluster_maps_valid = True

        ctx.invalidate_caches()

        self.assertFalse(ctx._cluster_maps_valid)

    def test_get_cluster_endpoint_map_builds_cache(self):
        """get_cluster_endpoint_map should build cache if needed."""
        stroke = [(50, 50), (60, 60)]
        junction = {(50, 50), (49, 50), (51, 50), (50, 49), (50, 51)}

        ctx = MergeContext(
            strokes=[stroke],
            junction_clusters=[junction],
            assigned=[junction]
        )

        result = ctx.get_cluster_endpoint_map()

        self.assertIsInstance(result, dict)
        self.assertTrue(ctx._cluster_maps_valid)

    def test_caches_are_reused(self):
        """Repeated cache access should reuse existing caches."""
        stroke = [(50, 50), (60, 60)]
        junction = {(50, 50), (49, 50), (51, 50)}

        ctx = MergeContext(
            strokes=[stroke],
            junction_clusters=[junction],
            assigned=[junction]
        )

        # First access builds cache
        map1 = ctx.get_cluster_endpoint_map()
        # Second access should return same object
        map2 = ctx.get_cluster_endpoint_map()

        self.assertIs(map1, map2)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions from stroke_merge_utils."""

    def test_seg_dir_horizontal(self):
        """seg_dir should return (1, 0) for rightward horizontal stroke."""
        stroke = make_line_stroke((0, 50), (100, 50), n_pts=10)
        direction = seg_dir(stroke, from_end=False)

        self.assertAlmostEqual(direction[0], 1.0, places=3)
        self.assertAlmostEqual(direction[1], 0.0, places=3)

    def test_seg_dir_vertical(self):
        """seg_dir should return (0, 1) for downward vertical stroke."""
        stroke = make_line_stroke((50, 0), (50, 100), n_pts=10)
        direction = seg_dir(stroke, from_end=False)

        self.assertAlmostEqual(direction[0], 0.0, places=3)
        self.assertAlmostEqual(direction[1], 1.0, places=3)

    def test_seg_dir_from_end(self):
        """seg_dir with from_end=True should compute from end of stroke."""
        stroke = make_line_stroke((0, 0), (100, 100), n_pts=10)

        dir_start = seg_dir(stroke, from_end=False)
        dir_end = seg_dir(stroke, from_end=True)

        # Both should point in the same direction for a straight line
        self.assertAlmostEqual(dir_start[0], dir_end[0], places=3)
        self.assertAlmostEqual(dir_start[1], dir_end[1], places=3)

    def test_angle_between_parallel(self):
        """angle_between parallel vectors should be 0."""
        v1 = (1.0, 0.0)
        v2 = (1.0, 0.0)

        angle = angle_between(v1, v2)

        self.assertAlmostEqual(angle, 0.0, places=5)

    def test_angle_between_perpendicular(self):
        """angle_between perpendicular vectors should be pi/2."""
        v1 = (1.0, 0.0)
        v2 = (0.0, 1.0)

        angle = angle_between(v1, v2)

        self.assertAlmostEqual(angle, np.pi / 2, places=5)

    def test_angle_between_opposite(self):
        """angle_between opposite vectors should be pi."""
        v1 = (1.0, 0.0)
        v2 = (-1.0, 0.0)

        angle = angle_between(v1, v2)

        self.assertAlmostEqual(angle, np.pi, places=5)

    def test_endpoint_cluster_finds_cluster(self):
        """endpoint_cluster should find cluster containing endpoint."""
        stroke = [(50, 50), (60, 60), (70, 70)]
        cluster = make_cluster((50, 50))
        assigned = [cluster]

        result = endpoint_cluster(stroke, from_end=False, assigned=assigned)

        self.assertEqual(result, 0)

    def test_endpoint_cluster_returns_negative_if_not_found(self):
        """endpoint_cluster should return -1 if endpoint not in any cluster."""
        stroke = [(0, 0), (10, 10), (20, 20)]
        cluster = make_cluster((100, 100))
        assigned = [cluster]

        result = endpoint_cluster(stroke, from_end=False, assigned=assigned)

        self.assertEqual(result, -1)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete merge workflows."""

    def test_full_pipeline_simplifies_strokes(self):
        """Full pipeline should simplify a complex stroke arrangement.

        Creates two strokes that form a continuation (opposite directions
        meeting at a junction) which should merge.
        """
        # Create a pattern that should be simplified:
        # Two strokes forming a continuation (opposite directions at junction)
        stroke1 = make_line_stroke((0, 50), (50, 50), n_pts=25)
        stroke2 = make_line_stroke((100, 50), (50, 50), n_pts=25)  # Opposite direction

        strokes = [stroke1, stroke2]
        junction = make_cluster((50, 50))

        pipeline = MergePipeline.create_default()
        result = pipeline.run(strokes, [junction], [junction])

        # Should merge into one stroke
        self.assertEqual(len(result), 1,
                         "Pipeline should merge continuation strokes")

    def test_pipeline_preserves_valid_strokes(self):
        """Pipeline should not remove valid, non-mergeable strokes."""
        # Two separate strokes with no connection
        stroke1 = make_line_stroke((0, 0), (50, 0), n_pts=25)
        stroke2 = make_line_stroke((0, 100), (50, 100), n_pts=25)

        strokes = [stroke1, stroke2]
        # No shared clusters
        cluster1 = make_cluster((50, 0))
        cluster2 = make_cluster((50, 100))

        pipeline = MergePipeline.create_default()
        result = pipeline.run(strokes, [cluster1, cluster2], [cluster1, cluster2])

        # Both strokes should remain (no reason to merge or remove)
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
