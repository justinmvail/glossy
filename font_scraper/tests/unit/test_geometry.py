"""Unit tests for geometry utility functions.

Tests the pure geometry functions in stroke_lib.utils.geometry:
    - point_distance: Euclidean distance between two points
    - point_distance_squared: Squared distance (faster for comparisons)
    - pick_straightest_neighbor: Pick straightest continuation from candidates
    - infer_direction_from_regions: Infer stroke direction from mask regions
    - angle_between: Angle between two direction vectors
    - compute_direction_vector: Compute (optionally normalized) direction vector
    - compute_dot_product: Dot product of two vectors
"""

import math
import unittest

from stroke_lib.domain.geometry import Point
from stroke_lib.utils.geometry import (
    angle_between,
    compute_direction_vector,
    compute_dot_product,
    infer_direction_from_regions,
    pick_straightest_neighbor,
    point_distance,
    point_distance_squared,
)


class TestPointDistance(unittest.TestCase):
    """Tests for point_distance function."""

    def test_same_point_zero_distance(self):
        """Distance from a point to itself is zero."""
        p = (10.0, 20.0)
        self.assertEqual(point_distance(p, p), 0.0)

    def test_horizontal_distance(self):
        """Distance along horizontal axis."""
        p1 = (0.0, 0.0)
        p2 = (10.0, 0.0)
        self.assertEqual(point_distance(p1, p2), 10.0)

    def test_vertical_distance(self):
        """Distance along vertical axis."""
        p1 = (0.0, 0.0)
        p2 = (0.0, 10.0)
        self.assertEqual(point_distance(p1, p2), 10.0)

    def test_diagonal_distance(self):
        """Distance along diagonal (3-4-5 triangle)."""
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        self.assertEqual(point_distance(p1, p2), 5.0)

    def test_negative_coordinates(self):
        """Distance with negative coordinates."""
        p1 = (-3.0, -4.0)
        p2 = (0.0, 0.0)
        self.assertEqual(point_distance(p1, p2), 5.0)

    def test_symmetry(self):
        """Distance is symmetric: d(p1, p2) == d(p2, p1)."""
        p1 = (1.0, 2.0)
        p2 = (4.0, 6.0)
        self.assertEqual(point_distance(p1, p2), point_distance(p2, p1))

    def test_unit_diagonal(self):
        """Distance of unit diagonal is sqrt(2)."""
        p1 = (0.0, 0.0)
        p2 = (1.0, 1.0)
        self.assertAlmostEqual(point_distance(p1, p2), math.sqrt(2), places=10)


class TestPointDistanceSquared(unittest.TestCase):
    """Tests for point_distance_squared function."""

    def test_same_point_zero_distance(self):
        """Squared distance from a point to itself is zero."""
        p = (10.0, 20.0)
        self.assertEqual(point_distance_squared(p, p), 0.0)

    def test_horizontal_distance_squared(self):
        """Squared distance along horizontal axis."""
        p1 = (0.0, 0.0)
        p2 = (10.0, 0.0)
        self.assertEqual(point_distance_squared(p1, p2), 100.0)

    def test_vertical_distance_squared(self):
        """Squared distance along vertical axis."""
        p1 = (0.0, 0.0)
        p2 = (0.0, 10.0)
        self.assertEqual(point_distance_squared(p1, p2), 100.0)

    def test_diagonal_distance_squared(self):
        """Squared distance along diagonal (3-4-5 triangle)."""
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        self.assertEqual(point_distance_squared(p1, p2), 25.0)

    def test_consistency_with_point_distance(self):
        """point_distance_squared equals point_distance squared."""
        p1 = (1.5, 2.7)
        p2 = (4.3, 8.1)
        d_squared = point_distance_squared(p1, p2)
        d = point_distance(p1, p2)
        self.assertAlmostEqual(d_squared, d * d, places=10)

    def test_preserves_ordering(self):
        """Squared distance preserves ordering for comparisons."""
        origin = (0.0, 0.0)
        near = (1.0, 1.0)
        far = (3.0, 4.0)
        # near is closer to origin than far
        self.assertLess(
            point_distance_squared(origin, near),
            point_distance_squared(origin, far)
        )


class TestPickStraightestNeighbor(unittest.TestCase):
    """Tests for pick_straightest_neighbor function."""

    def test_single_candidate(self):
        """With single candidate, returns that candidate."""
        current = (10, 10)
        path = [(0, 10), (5, 10), (10, 10)]
        candidates = [((15, 10), 'edge1')]
        result = pick_straightest_neighbor(current, path, candidates)
        self.assertEqual(result, ((15, 10), 'edge1'))

    def test_straight_continuation_preferred(self):
        """Straight continuation is preferred over turns."""
        current = (10, 10)
        path = [(0, 10), (5, 10), (10, 10)]  # Moving right
        candidates = [
            ((15, 10), 'straight'),  # Continue right
            ((10, 15), 'down'),      # Turn down
            ((10, 5), 'up'),         # Turn up
        ]
        result = pick_straightest_neighbor(current, path, candidates)
        self.assertEqual(result[1], 'straight')

    def test_backward_not_preferred(self):
        """Backward direction is least preferred."""
        current = (10, 10)
        path = [(0, 10), (5, 10), (10, 10)]  # Moving right
        candidates = [
            ((5, 10), 'backward'),   # Go back left
            ((10, 15), 'down'),      # Turn down
        ]
        result = pick_straightest_neighbor(current, path, candidates)
        self.assertEqual(result[1], 'down')

    def test_diagonal_continuation(self):
        """Diagonal path continues diagonally."""
        current = (10, 10)
        path = [(0, 0), (5, 5), (10, 10)]  # Moving down-right diagonal
        candidates = [
            ((15, 15), 'diagonal'),  # Continue diagonal
            ((15, 10), 'horizontal'),  # Go horizontal
            ((10, 15), 'vertical'),    # Go vertical
        ]
        result = pick_straightest_neighbor(current, path, candidates)
        self.assertEqual(result[1], 'diagonal')

    def test_short_path_uses_available_points(self):
        """With short path, uses all available points for direction."""
        current = (5, 5)
        path = [(0, 0), (5, 5)]  # Only 2 points
        candidates = [
            ((10, 10), 'continue'),
            ((10, 5), 'right'),
        ]
        result = pick_straightest_neighbor(current, path, candidates)
        self.assertEqual(result[1], 'continue')


class TestInferDirectionFromRegions(unittest.TestCase):
    """Tests for infer_direction_from_regions function.

    Numpad layout:
        7 | 8 | 9   (row 0, top)
        4 | 5 | 6   (row 1, middle)
        1 | 2 | 3   (row 2, bottom)
    """

    def test_down_vertical_movement(self):
        """Moving from top to bottom regions returns 'down'."""
        self.assertEqual(infer_direction_from_regions(8, 2), 'down')
        self.assertEqual(infer_direction_from_regions(7, 1), 'down')
        self.assertEqual(infer_direction_from_regions(9, 3), 'down')
        self.assertEqual(infer_direction_from_regions(8, 5), 'down')
        self.assertEqual(infer_direction_from_regions(5, 2), 'down')

    def test_up_vertical_movement(self):
        """Moving from bottom to top regions returns 'up'."""
        self.assertEqual(infer_direction_from_regions(2, 8), 'up')
        self.assertEqual(infer_direction_from_regions(1, 7), 'up')
        self.assertEqual(infer_direction_from_regions(3, 9), 'up')
        self.assertEqual(infer_direction_from_regions(2, 5), 'up')
        self.assertEqual(infer_direction_from_regions(5, 8), 'up')

    def test_right_horizontal_movement(self):
        """Moving from left to right regions returns 'right'."""
        self.assertEqual(infer_direction_from_regions(4, 6), 'right')
        self.assertEqual(infer_direction_from_regions(7, 9), 'right')
        self.assertEqual(infer_direction_from_regions(1, 3), 'right')
        self.assertEqual(infer_direction_from_regions(4, 5), 'right')
        self.assertEqual(infer_direction_from_regions(5, 6), 'right')

    def test_left_horizontal_movement(self):
        """Moving from right to left regions returns 'left'."""
        self.assertEqual(infer_direction_from_regions(6, 4), 'left')
        self.assertEqual(infer_direction_from_regions(9, 7), 'left')
        self.assertEqual(infer_direction_from_regions(3, 1), 'left')
        self.assertEqual(infer_direction_from_regions(6, 5), 'left')
        self.assertEqual(infer_direction_from_regions(5, 4), 'left')

    def test_diagonal_returns_none(self):
        """Diagonal movement returns None (ambiguous direction)."""
        # Pure diagonals
        self.assertIsNone(infer_direction_from_regions(7, 3))  # top-left to bottom-right
        self.assertIsNone(infer_direction_from_regions(9, 1))  # top-right to bottom-left
        self.assertIsNone(infer_direction_from_regions(1, 9))  # bottom-left to top-right
        self.assertIsNone(infer_direction_from_regions(3, 7))  # bottom-right to top-left

    def test_same_region_returns_none(self):
        """Same region returns None (no movement)."""
        self.assertIsNone(infer_direction_from_regions(5, 5))
        self.assertIsNone(infer_direction_from_regions(1, 1))
        self.assertIsNone(infer_direction_from_regions(9, 9))


class TestAngleBetween(unittest.TestCase):
    """Tests for angle_between function."""

    def test_parallel_vectors_zero_angle(self):
        """Parallel vectors have zero angle."""
        v1 = Point(1, 0)
        v2 = Point(1, 0)
        self.assertAlmostEqual(angle_between(v1, v2), 0.0, places=10)

    def test_opposite_vectors_pi_angle(self):
        """Opposite vectors have pi angle."""
        v1 = Point(1, 0)
        v2 = Point(-1, 0)
        self.assertAlmostEqual(angle_between(v1, v2), math.pi, places=10)

    def test_perpendicular_vectors_half_pi(self):
        """Perpendicular vectors have pi/2 angle."""
        v1 = Point(1, 0)
        v2 = Point(0, 1)
        self.assertAlmostEqual(angle_between(v1, v2), math.pi / 2, places=10)

    def test_45_degree_angle(self):
        """45 degree angle between vectors."""
        v1 = Point(1, 0)
        v2 = Point(1, 1).normalized()
        self.assertAlmostEqual(angle_between(v1, v2), math.pi / 4, places=10)

    def test_symmetry(self):
        """angle_between(v1, v2) == angle_between(v2, v1)."""
        v1 = Point(1, 2).normalized()
        v2 = Point(3, 1).normalized()
        self.assertAlmostEqual(
            angle_between(v1, v2),
            angle_between(v2, v1),
            places=10
        )

    def test_normalized_vectors(self):
        """Works correctly with normalized vectors."""
        v1 = Point(3, 4).normalized()  # (0.6, 0.8)
        v2 = Point(0, 1)  # Already unit vector
        angle = angle_between(v1, v2)
        # arccos(0.8) since dot product is 0*0.6 + 1*0.8 = 0.8
        expected = math.acos(0.8)
        self.assertAlmostEqual(angle, expected, places=10)


class TestComputeDirectionVector(unittest.TestCase):
    """Tests for compute_direction_vector function."""

    def test_horizontal_direction_normalized(self):
        """Horizontal direction normalized to unit vector."""
        p1 = (0.0, 0.0)
        p2 = (10.0, 0.0)
        dx, dy = compute_direction_vector(p1, p2, normalize=True)
        self.assertAlmostEqual(dx, 1.0, places=10)
        self.assertAlmostEqual(dy, 0.0, places=10)

    def test_vertical_direction_normalized(self):
        """Vertical direction normalized to unit vector."""
        p1 = (0.0, 0.0)
        p2 = (0.0, 10.0)
        dx, dy = compute_direction_vector(p1, p2, normalize=True)
        self.assertAlmostEqual(dx, 0.0, places=10)
        self.assertAlmostEqual(dy, 1.0, places=10)

    def test_diagonal_direction_normalized(self):
        """Diagonal direction normalized to unit vector."""
        p1 = (0.0, 0.0)
        p2 = (10.0, 10.0)
        dx, dy = compute_direction_vector(p1, p2, normalize=True)
        expected = 1.0 / math.sqrt(2)
        self.assertAlmostEqual(dx, expected, places=10)
        self.assertAlmostEqual(dy, expected, places=10)

    def test_unnormalized_direction(self):
        """Unnormalized direction returns raw delta."""
        p1 = (5.0, 10.0)
        p2 = (15.0, 30.0)
        dx, dy = compute_direction_vector(p1, p2, normalize=False)
        self.assertEqual(dx, 10.0)
        self.assertEqual(dy, 20.0)

    def test_same_point_returns_zero(self):
        """Same point returns (0, 0) when normalized."""
        p = (5.0, 5.0)
        dx, dy = compute_direction_vector(p, p, normalize=True)
        self.assertEqual(dx, 0.0)
        self.assertEqual(dy, 0.0)

    def test_negative_direction(self):
        """Direction with negative components."""
        p1 = (10.0, 10.0)
        p2 = (0.0, 0.0)
        dx, dy = compute_direction_vector(p1, p2, normalize=True)
        expected = -1.0 / math.sqrt(2)
        self.assertAlmostEqual(dx, expected, places=10)
        self.assertAlmostEqual(dy, expected, places=10)


class TestComputeDotProduct(unittest.TestCase):
    """Tests for compute_dot_product function."""

    def test_parallel_same_direction(self):
        """Parallel unit vectors same direction have dot product 1."""
        v1 = (1.0, 0.0)
        v2 = (1.0, 0.0)
        self.assertAlmostEqual(compute_dot_product(v1, v2), 1.0, places=10)

    def test_parallel_opposite_direction(self):
        """Parallel unit vectors opposite direction have dot product -1."""
        v1 = (1.0, 0.0)
        v2 = (-1.0, 0.0)
        self.assertAlmostEqual(compute_dot_product(v1, v2), -1.0, places=10)

    def test_perpendicular_vectors(self):
        """Perpendicular unit vectors have dot product 0."""
        v1 = (1.0, 0.0)
        v2 = (0.0, 1.0)
        self.assertAlmostEqual(compute_dot_product(v1, v2), 0.0, places=10)

    def test_45_degree_vectors(self):
        """45 degree unit vectors have dot product cos(45) = sqrt(2)/2."""
        v1 = (1.0, 0.0)
        sqrt2_over_2 = 1.0 / math.sqrt(2)
        v2 = (sqrt2_over_2, sqrt2_over_2)
        expected = sqrt2_over_2  # cos(45 degrees)
        self.assertAlmostEqual(compute_dot_product(v1, v2), expected, places=10)

    def test_non_unit_vectors(self):
        """Dot product works with non-unit vectors."""
        v1 = (3.0, 4.0)
        v2 = (1.0, 2.0)
        # 3*1 + 4*2 = 11
        self.assertEqual(compute_dot_product(v1, v2), 11.0)

    def test_symmetry(self):
        """Dot product is symmetric: v1.v2 == v2.v1."""
        v1 = (1.5, 2.5)
        v2 = (3.5, 4.5)
        self.assertEqual(
            compute_dot_product(v1, v2),
            compute_dot_product(v2, v1)
        )


if __name__ == '__main__':
    unittest.main()
