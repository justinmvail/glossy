#!/usr/bin/env python3
"""Unit tests for stroke_scoring.py.

Tests the composite scoring pattern implementation including:
- SnapPenalty: Penalizes points outside the glyph mask
- EdgePenalty: Penalizes points near the glyph edge
- OverlapPenalty: Penalizes excessive overlap between shapes
- CompositeScorer: Combines penalties with coverage calculation
- quick_stroke_score: Fast coverage estimation function

Example:
    Run all scoring tests::

        $ python3 -m pytest tests/unit/test_scoring.py -v

    Or with unittest::

        $ python3 -m unittest tests.unit.test_scoring -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial import cKDTree

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stroke_scoring import (
    CompositeScorer,
    EdgePenalty,
    OverlapPenalty,
    ScoringContext,
    ScoringPenalty,
    SnapPenalty,
    quick_stroke_score,
    score_all_strokes,
    score_all_strokes_ctx,
    score_raw_strokes,
    _compute_edge_penalty,
    _compute_overlap_penalty,
    _compute_per_shape_coverage,
    _compute_snap_penalty,
    _snap_points_to_mask,
    DEFAULT_STROKE_HALF_WIDTH,
    EDGE_PENALTY_WEIGHT,
    EDGE_THRESHOLD,
    FREE_OVERLAP,
    OVERLAP_PENALTY_WEIGHT,
    SNAP_PENALTY_WEIGHT,
    SNAP_THRESHOLD,
)


def create_mock_context(
    w: int = 100,
    h: int = 100,
    n_cloud: int = 100,
    radius: float = 5.0,
    cloud_points: np.ndarray = None,
    snap_xi: np.ndarray = None,
    snap_yi: np.ndarray = None,
    dist_map: np.ndarray = None,
) -> ScoringContext:
    """Create a mock ScoringContext for testing.

    Args:
        w: Width of the mask/image.
        h: Height of the mask/image.
        n_cloud: Number of points in the target cloud.
        radius: Coverage radius for point matching.
        cloud_points: Optional custom cloud points. If None, uses random points.
        snap_xi: Optional snap_xi array. If None, creates identity mapping.
        snap_yi: Optional snap_yi array. If None, creates identity mapping.
        dist_map: Optional distance transform map.

    Returns:
        ScoringContext configured for testing.
    """
    if cloud_points is None:
        cloud_points = np.random.rand(n_cloud, 2) * [w, h]

    cloud_tree = cKDTree(cloud_points)

    # Create identity snap maps (points map to themselves = inside mask)
    if snap_xi is None:
        snap_xi = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(float)
    if snap_yi is None:
        snap_yi = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(float)

    return ScoringContext(
        cloud_tree=cloud_tree,
        n_cloud=n_cloud,
        radius=radius,
        snap_xi=snap_xi,
        snap_yi=snap_yi,
        w=w,
        h=h,
        dist_map=dist_map,
    )


class TestSnapPenalty(unittest.TestCase):
    """Tests for SnapPenalty class."""

    def test_no_penalty_when_inside_mask(self):
        """Points inside the mask should incur no snap penalty."""
        # Create identity snap maps (all points map to themselves)
        w, h = 100, 100
        snap_xi = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(float)
        snap_yi = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(float)

        context = create_mock_context(w=w, h=h, snap_xi=snap_xi, snap_yi=snap_yi)

        # Points that are already on integer coordinates (inside mask)
        stroke_points = np.array([
            [10.0, 10.0],
            [20.0, 20.0],
            [30.0, 30.0],
            [50.0, 50.0],
        ])

        penalty = SnapPenalty(weight=1.0)
        result = penalty.compute(stroke_points, context)

        # All points are inside mask, so penalty should be 0
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_full_penalty_when_outside_mask(self):
        """Points outside the mask should incur full snap penalty."""
        w, h = 100, 100

        # Create snap maps that map all points to a fixed location (far away)
        # This simulates all points being outside the mask
        snap_xi = np.full((h, w), 0.0)  # All x snap to 0
        snap_yi = np.full((h, w), 0.0)  # All y snap to 0

        context = create_mock_context(w=w, h=h, snap_xi=snap_xi, snap_yi=snap_yi)

        # Points far from (0, 0) - all will be "off mask"
        stroke_points = np.array([
            [50.0, 50.0],
            [60.0, 60.0],
            [70.0, 70.0],
            [80.0, 80.0],
        ])

        penalty = SnapPenalty(weight=1.0, threshold=0.5)
        result = penalty.compute(stroke_points, context)

        # All points are outside mask, so penalty should be 1.0
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_partial_penalty_at_boundary(self):
        """Mixture of inside/outside points should give partial penalty."""
        w, h = 100, 100

        # Create snap maps where left half maps to itself, right half maps far
        snap_xi = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(float)
        snap_yi = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(float)

        # Right half snaps to x=0 (off by 50+ pixels)
        snap_xi[:, 50:] = 0.0

        context = create_mock_context(w=w, h=h, snap_xi=snap_xi, snap_yi=snap_yi)

        # 2 points in left half (inside), 2 points in right half (outside)
        stroke_points = np.array([
            [10.0, 10.0],  # inside
            [20.0, 20.0],  # inside
            [60.0, 10.0],  # outside (x > 50)
            [70.0, 20.0],  # outside (x > 50)
        ])

        penalty = SnapPenalty(weight=1.0, threshold=0.5)
        result = penalty.compute(stroke_points, context)

        # 2 out of 4 points are outside, so penalty should be 0.5
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_weight_applied_correctly(self):
        """Penalty weight should not affect compute(), only weighted in scorer."""
        w, h = 100, 100
        snap_xi = np.full((h, w), 0.0)
        snap_yi = np.full((h, w), 0.0)

        context = create_mock_context(w=w, h=h, snap_xi=snap_xi, snap_yi=snap_yi)

        stroke_points = np.array([[50.0, 50.0], [60.0, 60.0]])

        penalty_full = SnapPenalty(weight=1.0)
        penalty_half = SnapPenalty(weight=0.5)

        result_full = penalty_full.compute(stroke_points, context)
        result_half = penalty_half.compute(stroke_points, context)

        # compute() returns raw penalty, weight is applied by scorer
        self.assertEqual(result_full, result_half)

    def test_custom_threshold(self):
        """Custom threshold should change off-mask detection."""
        w, h = 100, 100

        # Snap maps that move points by 1 pixel
        snap_xi = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(float)
        snap_yi = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(float)
        snap_xi = np.clip(snap_xi + 1, 0, w - 1)  # Shift by 1

        context = create_mock_context(w=w, h=h, snap_xi=snap_xi, snap_yi=snap_yi)

        stroke_points = np.array([[10.0, 10.0], [20.0, 20.0]])

        # With threshold 0.5, distance of 1 is "off mask"
        penalty_strict = SnapPenalty(weight=1.0, threshold=0.5)
        result_strict = penalty_strict.compute(stroke_points, context)
        self.assertAlmostEqual(result_strict, 1.0, places=5)

        # With threshold 2.0, distance of 1 is "on mask"
        penalty_loose = SnapPenalty(weight=1.0, threshold=2.0)
        result_loose = penalty_loose.compute(stroke_points, context)
        self.assertAlmostEqual(result_loose, 0.0, places=5)

    def test_pre_computed_snapped_points(self):
        """Should use pre-computed snapped points if provided."""
        context = create_mock_context()

        stroke_points = np.array([[50.0, 50.0], [60.0, 60.0]])
        # Pre-computed snapped points at same location (no penalty)
        snapped_points = stroke_points.copy()

        penalty = SnapPenalty(weight=1.0)
        result = penalty.compute(stroke_points, context, snapped_points=snapped_points)

        self.assertAlmostEqual(result, 0.0, places=5)


class TestEdgePenalty(unittest.TestCase):
    """Tests for EdgePenalty class."""

    def test_no_penalty_when_no_dist_map(self):
        """Should return 0.0 when dist_map is None."""
        context = create_mock_context(dist_map=None)

        stroke_points = np.array([[10.0, 10.0], [20.0, 20.0]])

        penalty = EdgePenalty(weight=1.0)
        result = penalty.compute(stroke_points, context)

        self.assertAlmostEqual(result, 0.0, places=5)

    def test_no_penalty_when_far_from_edge(self):
        """Points far from edge should incur no penalty."""
        w, h = 100, 100

        # Distance map where center is far from edge
        dist_map = np.full((h, w), 20.0)  # All points 20 pixels from edge

        context = create_mock_context(w=w, h=h, dist_map=dist_map)

        stroke_points = np.array([[50.0, 50.0], [60.0, 60.0]])

        penalty = EdgePenalty(weight=1.0, threshold=1.5)
        result = penalty.compute(stroke_points, context)

        self.assertAlmostEqual(result, 0.0, places=5)

    def test_full_penalty_when_near_edge(self):
        """Points near edge should incur full penalty."""
        w, h = 100, 100

        # Distance map where all points are near edge
        dist_map = np.full((h, w), 0.5)  # All points 0.5 pixels from edge

        context = create_mock_context(w=w, h=h, dist_map=dist_map)

        stroke_points = np.array([[50.0, 50.0], [60.0, 60.0]])

        penalty = EdgePenalty(weight=1.0, threshold=1.5)
        result = penalty.compute(stroke_points, context)

        self.assertAlmostEqual(result, 1.0, places=5)

    def test_partial_penalty_mixed_distances(self):
        """Mix of near/far edge points should give partial penalty."""
        w, h = 100, 100

        # Left half near edge, right half far from edge
        dist_map = np.full((h, w), 10.0)
        dist_map[:, :50] = 0.5  # Left half near edge

        context = create_mock_context(w=w, h=h, dist_map=dist_map)

        # 2 points in left half (near edge), 2 in right (far)
        stroke_points = np.array([
            [10.0, 10.0],  # near edge
            [20.0, 20.0],  # near edge
            [60.0, 10.0],  # far from edge
            [70.0, 20.0],  # far from edge
        ])

        penalty = EdgePenalty(weight=1.0, threshold=1.5)
        result = penalty.compute(stroke_points, context)

        self.assertAlmostEqual(result, 0.5, places=5)

    def test_custom_threshold(self):
        """Custom threshold should change near-edge detection."""
        w, h = 100, 100
        dist_map = np.full((h, w), 2.0)  # All points 2 pixels from edge

        context = create_mock_context(w=w, h=h, dist_map=dist_map)

        stroke_points = np.array([[50.0, 50.0], [60.0, 60.0]])

        # With threshold 1.5, distance of 2 is not "near edge"
        penalty_strict = EdgePenalty(weight=1.0, threshold=1.5)
        result_strict = penalty_strict.compute(stroke_points, context)
        self.assertAlmostEqual(result_strict, 0.0, places=5)

        # With threshold 3.0, distance of 2 is "near edge"
        penalty_loose = EdgePenalty(weight=1.0, threshold=3.0)
        result_loose = penalty_loose.compute(stroke_points, context)
        self.assertAlmostEqual(result_loose, 1.0, places=5)


class TestOverlapPenalty(unittest.TestCase):
    """Tests for OverlapPenalty class."""

    def test_no_penalty_with_single_shape(self):
        """Single shape should have no overlap penalty."""
        context = create_mock_context()
        stroke_points = np.array([[10.0, 10.0], [20.0, 20.0]])

        per_shape_coverage = [{0, 1, 2, 3, 4}]  # Single shape covers 5 points

        penalty = OverlapPenalty(weight=1.0)
        result = penalty.compute(
            stroke_points, context, per_shape_coverage=per_shape_coverage
        )

        self.assertAlmostEqual(result, 0.0, places=5)

    def test_no_penalty_with_no_overlap(self):
        """Disjoint shapes should have no overlap penalty."""
        context = create_mock_context()
        stroke_points = np.array([[10.0, 10.0], [20.0, 20.0]])

        # Two shapes covering different points
        per_shape_coverage = [
            {0, 1, 2, 3, 4},     # Shape 1
            {5, 6, 7, 8, 9},     # Shape 2 (no overlap)
        ]

        penalty = OverlapPenalty(weight=1.0)
        result = penalty.compute(
            stroke_points, context, per_shape_coverage=per_shape_coverage
        )

        self.assertAlmostEqual(result, 0.0, places=5)

    def test_no_penalty_within_free_overlap(self):
        """Overlap within FREE_OVERLAP threshold should not be penalized."""
        context = create_mock_context()
        stroke_points = np.array([[10.0, 10.0], [20.0, 20.0]])

        # Each shape covers 10 points, 2 overlap (20% < 25% FREE_OVERLAP)
        per_shape_coverage = [
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},           # Shape 1
            {8, 9, 10, 11, 12, 13, 14, 15, 16, 17},   # Shape 2 (2 overlap)
        ]

        penalty = OverlapPenalty(weight=1.0, free_overlap=0.25)
        result = penalty.compute(
            stroke_points, context, per_shape_coverage=per_shape_coverage
        )

        self.assertAlmostEqual(result, 0.0, places=5)

    def test_penalty_when_overlap_exceeds_threshold(self):
        """Overlap exceeding FREE_OVERLAP threshold should be penalized."""
        context = create_mock_context()
        stroke_points = np.array([[10.0, 10.0], [20.0, 20.0]])

        # Each shape covers 10 points, 5 overlap (50% > 25% FREE_OVERLAP)
        per_shape_coverage = [
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},           # Shape 1
            {5, 6, 7, 8, 9, 10, 11, 12, 13, 14},      # Shape 2 (5 overlap = 50%)
        ]

        penalty = OverlapPenalty(weight=1.0, free_overlap=0.25)
        result = penalty.compute(
            stroke_points, context, per_shape_coverage=per_shape_coverage
        )

        # Excess = (0.5 - 0.25) = 0.25 per shape, averaged over 2 shapes = 0.25
        self.assertGreater(result, 0.0)
        self.assertAlmostEqual(result, 0.25, places=5)

    def test_full_overlap_penalty(self):
        """Complete overlap should give maximum penalty."""
        context = create_mock_context()
        stroke_points = np.array([[10.0, 10.0], [20.0, 20.0]])

        # Both shapes cover exactly the same points (100% overlap)
        per_shape_coverage = [
            {0, 1, 2, 3, 4},
            {0, 1, 2, 3, 4},  # Identical (100% overlap)
        ]

        penalty = OverlapPenalty(weight=1.0, free_overlap=0.25)
        result = penalty.compute(
            stroke_points, context, per_shape_coverage=per_shape_coverage
        )

        # Excess = (1.0 - 0.25) = 0.75 per shape, averaged = 0.75
        self.assertAlmostEqual(result, 0.75, places=5)

    def test_empty_shape_handled(self):
        """Empty shapes should not cause errors."""
        context = create_mock_context()
        stroke_points = np.array([[10.0, 10.0], [20.0, 20.0]])

        per_shape_coverage = [
            set(),               # Empty shape
            {0, 1, 2, 3, 4},     # Normal shape
        ]

        penalty = OverlapPenalty(weight=1.0)
        result = penalty.compute(
            stroke_points, context, per_shape_coverage=per_shape_coverage
        )

        # Should not raise, should handle gracefully
        self.assertGreaterEqual(result, 0.0)

    def test_none_per_shape_coverage(self):
        """Should return 0.0 when per_shape_coverage is None."""
        context = create_mock_context()
        stroke_points = np.array([[10.0, 10.0], [20.0, 20.0]])

        penalty = OverlapPenalty(weight=1.0)
        result = penalty.compute(stroke_points, context, per_shape_coverage=None)

        self.assertAlmostEqual(result, 0.0, places=5)


class TestCompositeScorer(unittest.TestCase):
    """Tests for CompositeScorer class."""

    def test_combines_penalties_with_weights(self):
        """Scorer should combine penalties according to their weights."""
        w, h = 100, 100
        n_cloud = 100

        # Create cloud points in the center
        cloud_points = np.array([[50.0, 50.0], [51.0, 51.0], [52.0, 52.0]])
        n_cloud = len(cloud_points)

        # Identity snap maps (inside mask)
        snap_xi = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(float)
        snap_yi = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(float)

        # Distance map far from edge
        dist_map = np.full((h, w), 20.0)

        context = create_mock_context(
            w=w, h=h, n_cloud=n_cloud, cloud_points=cloud_points,
            snap_xi=snap_xi, snap_yi=snap_yi, dist_map=dist_map, radius=5.0
        )

        # Stroke covering the cloud points
        stroke_arrays = [np.array([[50.0, 50.0], [51.0, 51.0], [52.0, 52.0]])]

        # Scorer with only snap penalty
        scorer = CompositeScorer(penalties=[SnapPenalty(weight=0.5)])
        score = scorer.score(stroke_arrays, context)

        # Score should be negative (for minimization) and coverage-based
        self.assertLess(score, 0.0)

    def test_coverage_calculation(self):
        """Scorer should calculate coverage correctly."""
        w, h = 100, 100

        # Create a simple cloud of 4 points
        cloud_points = np.array([
            [10.0, 10.0],
            [20.0, 20.0],
            [30.0, 30.0],
            [40.0, 40.0],
        ])
        n_cloud = len(cloud_points)

        context = create_mock_context(
            w=w, h=h, n_cloud=n_cloud, cloud_points=cloud_points, radius=5.0
        )

        # Stroke covering 2 of 4 points (50% coverage)
        stroke_arrays = [np.array([[10.0, 10.0], [20.0, 20.0]])]

        # No penalties
        scorer = CompositeScorer(penalties=[])
        score = scorer.score(stroke_arrays, context)

        # Coverage = 0.5, no penalties, score = -(0.5 - 0) = -0.5
        self.assertAlmostEqual(score, -0.5, places=2)

    def test_score_returns_negative(self):
        """Score should be negative for minimization."""
        w, h = 100, 100
        cloud_points = np.array([[50.0, 50.0]])
        n_cloud = len(cloud_points)

        context = create_mock_context(
            w=w, h=h, n_cloud=n_cloud, cloud_points=cloud_points, radius=10.0
        )

        # Stroke that covers the cloud point
        stroke_arrays = [np.array([[50.0, 50.0], [55.0, 55.0]])]

        scorer = CompositeScorer(penalties=[])
        score = scorer.score(stroke_arrays, context)

        # With coverage > 0 and no penalties, score should be negative
        self.assertLess(score, 0.0)

    def test_empty_strokes_return_zero(self):
        """Empty strokes should return score of 0.0."""
        context = create_mock_context()

        scorer = CompositeScorer()

        # Empty list
        score1 = scorer.score([], context)
        self.assertAlmostEqual(score1, 0.0, places=5)

        # List with empty arrays
        score2 = scorer.score([np.array([]).reshape(0, 2)], context)
        self.assertAlmostEqual(score2, 0.0, places=5)

    def test_strokes_with_one_point_filtered(self):
        """Strokes with fewer than 2 points should be filtered."""
        context = create_mock_context()

        scorer = CompositeScorer()

        # Single-point strokes should be filtered
        stroke_arrays = [
            np.array([[10.0, 10.0]]),  # Only 1 point - filtered
        ]
        score = scorer.score(stroke_arrays, context)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_default_penalties(self):
        """Default scorer should have Snap, Edge, and Overlap penalties."""
        scorer = CompositeScorer()

        self.assertEqual(len(scorer.penalties), 3)

        penalty_types = [type(p) for p in scorer.penalties]
        self.assertIn(SnapPenalty, penalty_types)
        self.assertIn(EdgePenalty, penalty_types)
        self.assertIn(OverlapPenalty, penalty_types)

    def test_add_penalty(self):
        """add_penalty should add a penalty to the scorer."""
        scorer = CompositeScorer(penalties=[])
        self.assertEqual(len(scorer.penalties), 0)

        scorer.add_penalty(SnapPenalty(weight=0.5))
        self.assertEqual(len(scorer.penalties), 1)
        self.assertIsInstance(scorer.penalties[0], SnapPenalty)

    def test_remove_penalty_by_type(self):
        """remove_penalty_by_type should remove all penalties of that type."""
        scorer = CompositeScorer(penalties=[
            SnapPenalty(weight=0.5),
            EdgePenalty(weight=0.1),
            SnapPenalty(weight=0.3),  # Second snap penalty
        ])
        self.assertEqual(len(scorer.penalties), 3)

        removed = scorer.remove_penalty_by_type(SnapPenalty)

        self.assertTrue(removed)
        self.assertEqual(len(scorer.penalties), 1)
        self.assertIsInstance(scorer.penalties[0], EdgePenalty)

    def test_remove_penalty_by_type_not_found(self):
        """remove_penalty_by_type should return False if type not found."""
        scorer = CompositeScorer(penalties=[SnapPenalty(weight=0.5)])

        removed = scorer.remove_penalty_by_type(EdgePenalty)

        self.assertFalse(removed)
        self.assertEqual(len(scorer.penalties), 1)


class TestQuickStrokeScore(unittest.TestCase):
    """Tests for quick_stroke_score function."""

    def test_empty_strokes_returns_zero(self):
        """Empty strokes should return 0.0."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True  # 20x20 glyph

        result = quick_stroke_score([], mask)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_none_mask_returns_zero(self):
        """None mask should return 0.0."""
        strokes = [[[50, 50], [60, 60]]]

        result = quick_stroke_score(strokes, None)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_empty_mask_returns_zero(self):
        """Mask with no glyph pixels should return 0.0."""
        mask = np.zeros((100, 100), dtype=bool)
        strokes = [[[50, 50], [60, 60]]]

        result = quick_stroke_score(strokes, mask)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_stroke_covering_glyph(self):
        """Stroke covering glyph should return high coverage."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 45:55] = True  # 10x10 glyph in center

        # Stroke running through the center
        strokes = [[[45, 50], [46, 50], [47, 50], [48, 50], [49, 50],
                    [50, 50], [51, 50], [52, 50], [53, 50], [54, 50]]]

        result = quick_stroke_score(strokes, mask)

        # Should have significant coverage
        self.assertGreater(result, 0.3)

    def test_stroke_outside_glyph(self):
        """Stroke outside glyph should return low coverage."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 45:55] = True  # Glyph in center

        # Stroke in corner (far from glyph)
        strokes = [[[5, 5], [10, 10], [15, 15]]]

        result = quick_stroke_score(strokes, mask)

        # Should have low coverage
        self.assertLess(result, 0.2)

    def test_full_coverage(self):
        """Dense strokes covering entire glyph should return ~1.0."""
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 10:40] = True  # 30x30 glyph

        # Dense grid of strokes covering the glyph
        strokes = []
        for y in range(10, 40, 5):
            stroke = [[x, y] for x in range(10, 40)]
            strokes.append(stroke)
        for x in range(10, 40, 5):
            stroke = [[x, y] for y in range(10, 40)]
            strokes.append(stroke)

        result = quick_stroke_score(strokes, mask)

        # Should have near-full coverage
        self.assertGreater(result, 0.8)

    def test_returns_float_in_valid_range(self):
        """Result should always be a float between 0 and 1."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True

        strokes = [[[50, 50], [55, 55], [60, 60]]]

        result = quick_stroke_score(strokes, mask)

        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_stroke_with_out_of_bounds_points(self):
        """Points outside mask bounds should be handled gracefully."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True

        # Some points outside bounds
        strokes = [[[-10, 50], [50, 50], [110, 50], [50, -10], [50, 110]]]

        # Should not raise
        result = quick_stroke_score(strokes, mask)
        self.assertIsInstance(result, float)


class TestHelperFunctions(unittest.TestCase):
    """Tests for internal helper functions."""

    def test_snap_points_to_mask_basic(self):
        """_snap_points_to_mask should snap points correctly."""
        w, h = 100, 100
        snap_xi = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(float)
        snap_yi = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(float)

        all_pts = np.array([[10.0, 10.0], [20.5, 20.5], [30.0, 30.0]])

        snapped = _snap_points_to_mask(all_pts, snap_xi, snap_yi, w, h)

        self.assertEqual(snapped.shape, (3, 2))
        # With identity snap maps, points should map to themselves (rounded)
        np.testing.assert_array_almost_equal(snapped[0], [10.0, 10.0])

    def test_snap_points_to_mask_empty(self):
        """_snap_points_to_mask should handle empty arrays."""
        w, h = 100, 100
        snap_xi = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(float)
        snap_yi = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(float)

        all_pts = np.empty((0, 2), dtype=float)

        snapped = _snap_points_to_mask(all_pts, snap_xi, snap_yi, w, h)

        self.assertEqual(snapped.shape, (0, 2))

    def test_snap_points_to_mask_clipping(self):
        """_snap_points_to_mask should clip out-of-bounds points."""
        w, h = 100, 100
        snap_xi = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(float)
        snap_yi = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(float)

        # Points outside bounds
        all_pts = np.array([[-10.0, 50.0], [50.0, -10.0], [150.0, 50.0], [50.0, 150.0]])

        # Should not raise
        snapped = _snap_points_to_mask(all_pts, snap_xi, snap_yi, w, h)

        self.assertEqual(snapped.shape, (4, 2))

    def test_snap_points_to_mask_invalid_dimensions(self):
        """_snap_points_to_mask should raise on invalid dimensions."""
        snap_xi = np.zeros((10, 10))
        snap_yi = np.zeros((10, 10))
        all_pts = np.array([[5.0, 5.0]])

        with self.assertRaises(ValueError):
            _snap_points_to_mask(all_pts, snap_xi, snap_yi, 0, 10)

        with self.assertRaises(ValueError):
            _snap_points_to_mask(all_pts, snap_xi, snap_yi, 10, 0)

    def test_compute_per_shape_coverage(self):
        """_compute_per_shape_coverage should compute coverage sets correctly."""
        # Create cloud and tree
        cloud_points = np.array([
            [10.0, 10.0],
            [20.0, 20.0],
            [30.0, 30.0],
            [40.0, 40.0],
        ])
        cloud_tree = cKDTree(cloud_points)
        radius = 5.0

        # Two shapes, each covering 2 points
        all_shapes = [
            np.array([[10.0, 10.0], [20.0, 20.0]]),  # Covers points 0, 1
            np.array([[30.0, 30.0], [40.0, 40.0]]),  # Covers points 2, 3
        ]
        snapped = np.concatenate(all_shapes)

        per_shape = _compute_per_shape_coverage(all_shapes, snapped, cloud_tree, radius)

        self.assertEqual(len(per_shape), 2)
        self.assertIn(0, per_shape[0])
        self.assertIn(1, per_shape[0])
        self.assertIn(2, per_shape[1])
        self.assertIn(3, per_shape[1])


class TestLegacyFunctions(unittest.TestCase):
    """Tests for legacy deprecated functions."""

    def test_compute_snap_penalty_deprecated(self):
        """_compute_snap_penalty should compute weighted penalty."""
        all_pts = np.array([[50.0, 50.0], [60.0, 60.0]])
        # Snapped points far away
        snapped = np.array([[0.0, 0.0], [0.0, 0.0]])

        result = _compute_snap_penalty(all_pts, snapped)

        # All points off mask, so penalty = SNAP_PENALTY_WEIGHT * 1.0
        self.assertAlmostEqual(result, SNAP_PENALTY_WEIGHT, places=5)

    def test_compute_edge_penalty_deprecated(self):
        """_compute_edge_penalty should compute weighted penalty."""
        w, h = 100, 100
        snapped = np.array([[50.0, 50.0], [60.0, 60.0]])

        # All near edge
        dist_map = np.full((h, w), 0.5)

        result = _compute_edge_penalty(snapped, dist_map, w, h)

        # All points near edge, so penalty = EDGE_PENALTY_WEIGHT * 1.0
        self.assertAlmostEqual(result, EDGE_PENALTY_WEIGHT, places=5)

    def test_compute_edge_penalty_no_dist_map(self):
        """_compute_edge_penalty should return 0 when dist_map is None."""
        snapped = np.array([[50.0, 50.0]])

        result = _compute_edge_penalty(snapped, None, 100, 100)

        self.assertAlmostEqual(result, 0.0, places=5)

    def test_compute_overlap_penalty_deprecated(self):
        """_compute_overlap_penalty should compute weighted penalty."""
        # Full overlap
        per_shape = [
            {0, 1, 2, 3, 4},
            {0, 1, 2, 3, 4},  # 100% overlap
        ]

        result = _compute_overlap_penalty(per_shape)

        # Expected: OVERLAP_PENALTY_WEIGHT * (1.0 - FREE_OVERLAP)
        expected = OVERLAP_PENALTY_WEIGHT * (1.0 - FREE_OVERLAP)
        self.assertAlmostEqual(result, expected, places=5)

    def test_compute_overlap_penalty_single_shape(self):
        """_compute_overlap_penalty should return 0 for single shape."""
        per_shape = [{0, 1, 2, 3, 4}]

        result = _compute_overlap_penalty(per_shape)

        self.assertAlmostEqual(result, 0.0, places=5)


class TestScoreAllStrokes(unittest.TestCase):
    """Tests for score_all_strokes and score_all_strokes_ctx functions."""

    def test_score_all_strokes_basic(self):
        """score_all_strokes should compute valid scores from parameters."""
        w, h = 100, 100
        cloud_points = np.array([[50.0, 50.0], [55.0, 55.0], [60.0, 60.0]])
        n_cloud = len(cloud_points)
        cloud_tree = cKDTree(cloud_points)

        snap_xi = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(float)
        snap_yi = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(float)

        # Mock param_vector_to_shapes to return simple strokes
        with patch('stroke_scoring._param_vector_to_shapes') as mock_shapes:
            mock_shapes.return_value = [
                np.array([[50.0, 50.0], [55.0, 55.0], [60.0, 60.0]])
            ]

            param_vector = np.array([0.5, 0.5, 0.5, 0.5])  # Dummy params
            shape_types = ['line']
            slices = [(0, 4)]
            bbox = (0, 0, 100, 100)

            score = score_all_strokes(
                param_vector, shape_types, slices, bbox,
                cloud_tree, n_cloud, 5.0,
                snap_yi, snap_xi, w, h
            )

            # Score should be negative (for minimization)
            self.assertLess(score, 0.0)

    def test_score_all_strokes_ctx_uses_context(self):
        """score_all_strokes_ctx should use ScoringContext correctly."""
        w, h = 100, 100
        cloud_points = np.array([[50.0, 50.0], [55.0, 55.0]])
        n_cloud = len(cloud_points)
        cloud_tree = cKDTree(cloud_points)

        snap_xi = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(float)
        snap_yi = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(float)

        ctx = ScoringContext(
            cloud_tree=cloud_tree,
            n_cloud=n_cloud,
            radius=5.0,
            snap_xi=snap_xi,
            snap_yi=snap_yi,
            w=w,
            h=h,
            dist_map=None,
        )

        with patch('stroke_scoring._param_vector_to_shapes') as mock_shapes:
            mock_shapes.return_value = [
                np.array([[50.0, 50.0], [55.0, 55.0]])
            ]

            param_vector = np.array([0.5, 0.5, 0.5, 0.5])
            shape_types = ['line']
            slices = [(0, 4)]
            bbox = (0, 0, 100, 100)

            score = score_all_strokes_ctx(
                param_vector, shape_types, slices, bbox, ctx
            )

            self.assertLess(score, 0.0)

    def test_score_all_strokes_with_dist_map(self):
        """score_all_strokes should use dist_map for edge penalty."""
        w, h = 100, 100
        cloud_points = np.array([[50.0, 50.0]])
        n_cloud = len(cloud_points)
        cloud_tree = cKDTree(cloud_points)

        snap_xi = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(float)
        snap_yi = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(float)

        # Points near edge
        dist_map = np.full((h, w), 0.5)

        with patch('stroke_scoring._param_vector_to_shapes') as mock_shapes:
            mock_shapes.return_value = [
                np.array([[50.0, 50.0], [51.0, 51.0]])
            ]

            param_vector = np.array([0.5, 0.5, 0.5, 0.5])
            shape_types = ['line']
            slices = [(0, 4)]
            bbox = (0, 0, 100, 100)

            score_with = score_all_strokes(
                param_vector, shape_types, slices, bbox,
                cloud_tree, n_cloud, 5.0,
                snap_yi, snap_xi, w, h, dist_map=dist_map
            )

            # Score should be less negative (worse) due to edge penalty
            self.assertLess(score_with, 0.0)


class TestScoreRawStrokes(unittest.TestCase):
    """Tests for score_raw_strokes function."""

    def test_basic_scoring(self):
        """score_raw_strokes should compute valid scores."""
        w, h = 100, 100
        cloud_points = np.array([[50.0, 50.0], [51.0, 51.0], [52.0, 52.0]])
        n_cloud = len(cloud_points)
        cloud_tree = cKDTree(cloud_points)

        snap_xi = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(float)
        snap_yi = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(float)

        stroke_arrays = [np.array([[50.0, 50.0], [51.0, 51.0], [52.0, 52.0]])]

        score = score_raw_strokes(
            stroke_arrays, cloud_tree, n_cloud, 5.0,
            snap_yi, snap_xi, w, h
        )

        # Score should be negative (for minimization)
        self.assertLess(score, 0.0)

    def test_custom_scorer(self):
        """score_raw_strokes should use custom scorer if provided."""
        w, h = 100, 100
        cloud_points = np.array([[50.0, 50.0]])
        n_cloud = len(cloud_points)
        cloud_tree = cKDTree(cloud_points)

        snap_xi = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(float)
        snap_yi = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(float)

        stroke_arrays = [np.array([[50.0, 50.0], [55.0, 55.0]])]

        # Custom scorer with no penalties
        custom_scorer = CompositeScorer(penalties=[])

        score = score_raw_strokes(
            stroke_arrays, cloud_tree, n_cloud, 5.0,
            snap_yi, snap_xi, w, h, scorer=custom_scorer
        )

        # Without penalties, score should just be -coverage
        self.assertLess(score, 0.0)

    def test_empty_strokes(self):
        """score_raw_strokes should return 0 for empty strokes."""
        cloud_points = np.array([[50.0, 50.0]])
        cloud_tree = cKDTree(cloud_points)

        snap_xi = np.arange(100).reshape(1, -1).repeat(100, axis=0).astype(float)
        snap_yi = np.arange(100).reshape(-1, 1).repeat(100, axis=1).astype(float)

        score = score_raw_strokes([], cloud_tree, 1, 5.0, snap_yi, snap_xi, 100, 100)

        self.assertAlmostEqual(score, 0.0, places=5)


class TestScoringPenaltyBaseClass(unittest.TestCase):
    """Tests for the ScoringPenalty abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """ScoringPenalty should not be directly instantiable."""
        with self.assertRaises(TypeError):
            ScoringPenalty()

    def test_subclass_must_implement_compute(self):
        """Subclass without compute() should raise TypeError."""
        class IncompletePenalty(ScoringPenalty):
            pass

        with self.assertRaises(TypeError):
            IncompletePenalty()

    def test_subclass_with_compute_works(self):
        """Subclass with compute() should instantiate."""
        class CompletePenalty(ScoringPenalty):
            def compute(self, stroke_points, context, **kwargs):
                return 0.5

        penalty = CompletePenalty(weight=0.3)
        self.assertEqual(penalty.weight, 0.3)


class TestScoringContext(unittest.TestCase):
    """Tests for ScoringContext dataclass."""

    def test_creation(self):
        """ScoringContext should store all attributes."""
        cloud_points = np.array([[50.0, 50.0]])
        cloud_tree = cKDTree(cloud_points)
        snap_xi = np.zeros((100, 100))
        snap_yi = np.zeros((100, 100))
        dist_map = np.ones((100, 100))

        ctx = ScoringContext(
            cloud_tree=cloud_tree,
            n_cloud=1,
            radius=5.0,
            snap_xi=snap_xi,
            snap_yi=snap_yi,
            w=100,
            h=100,
            dist_map=dist_map,
        )

        self.assertEqual(ctx.n_cloud, 1)
        self.assertEqual(ctx.radius, 5.0)
        self.assertEqual(ctx.w, 100)
        self.assertEqual(ctx.h, 100)
        self.assertIsNotNone(ctx.dist_map)

    def test_dist_map_optional(self):
        """dist_map should default to None."""
        cloud_points = np.array([[50.0, 50.0]])
        cloud_tree = cKDTree(cloud_points)
        snap_xi = np.zeros((100, 100))
        snap_yi = np.zeros((100, 100))

        ctx = ScoringContext(
            cloud_tree=cloud_tree,
            n_cloud=1,
            radius=5.0,
            snap_xi=snap_xi,
            snap_yi=snap_yi,
            w=100,
            h=100,
        )

        self.assertIsNone(ctx.dist_map)


class TestModuleConstants(unittest.TestCase):
    """Tests for module-level constants."""

    def test_default_stroke_half_width(self):
        """DEFAULT_STROKE_HALF_WIDTH should be reasonable."""
        self.assertGreater(DEFAULT_STROKE_HALF_WIDTH, 0)
        self.assertLess(DEFAULT_STROKE_HALF_WIDTH, 50)

    def test_penalty_weights_in_range(self):
        """Penalty weights should be between 0 and 1."""
        self.assertGreaterEqual(SNAP_PENALTY_WEIGHT, 0)
        self.assertLessEqual(SNAP_PENALTY_WEIGHT, 1)

        self.assertGreaterEqual(EDGE_PENALTY_WEIGHT, 0)
        self.assertLessEqual(EDGE_PENALTY_WEIGHT, 1)

        self.assertGreaterEqual(OVERLAP_PENALTY_WEIGHT, 0)
        self.assertLessEqual(OVERLAP_PENALTY_WEIGHT, 1)

    def test_thresholds_positive(self):
        """Thresholds should be positive."""
        self.assertGreater(EDGE_THRESHOLD, 0)
        self.assertGreater(SNAP_THRESHOLD, 0)

    def test_free_overlap_in_range(self):
        """FREE_OVERLAP should be between 0 and 1."""
        self.assertGreaterEqual(FREE_OVERLAP, 0)
        self.assertLessEqual(FREE_OVERLAP, 1)


if __name__ == '__main__':
    unittest.main()
