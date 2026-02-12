"""Unit tests for stroke_shapes.py.

Tests the Shape class hierarchy, SHAPES registry, and utility functions.
Target: 95% coverage of stroke_shapes.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stroke_shapes import (
    SHAPES,
    Shape,
    VLineShape,
    HLineShape,
    DiagShape,
    ArcRightShape,
    ArcLeftShape,
    LoopShape,
    UArcShape,
    get_param_bounds,
    param_vector_to_shapes,
    make_point_cloud,
    adaptive_radius,
    MIN_SHAPE_POINTS,
    POINT_SPACING_TARGET,
    RADIUS_FLOOR_MULTIPLIER,
    MIN_RADIUS,
    DISTANCE_PERCENTILE,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_bbox():
    """Standard 100x100 bounding box at origin."""
    return (0, 0, 100, 100)


@pytest.fixture
def offset_bbox():
    """Bounding box offset from origin."""
    return (50, 50, 150, 150)


@pytest.fixture
def rectangular_bbox():
    """Non-square bounding box."""
    return (0, 0, 200, 100)


# ---------------------------------------------------------------------------
# TestVLineShape
# ---------------------------------------------------------------------------

class TestVLineShape:
    """Tests for the VLineShape class."""

    def test_generates_vertical_points(self, standard_bbox):
        """Test that VLineShape generates a vertical line."""
        shape = VLineShape()
        params = (0.5, 0.0, 1.0)  # x=50, y from 0 to 100
        points = shape.generate(params, standard_bbox, n_pts=60)

        # All x-coordinates should be the same (vertical line)
        assert np.allclose(points[:, 0], 50.0)
        # Y-coordinates should span from 0 to 100
        assert np.isclose(points[0, 1], 0.0)
        assert np.isclose(points[-1, 1], 100.0)

    def test_respects_bbox(self, offset_bbox):
        """Test that VLineShape respects bounding box offset."""
        shape = VLineShape()
        params = (0.5, 0.0, 1.0)  # Midpoint of bbox
        points = shape.generate(params, offset_bbox, n_pts=60)

        # X should be at midpoint of offset bbox (50 + 0.5*100 = 100)
        assert np.allclose(points[:, 0], 100.0)
        # Y should span from 50 to 150
        assert np.isclose(points[0, 1], 50.0)
        assert np.isclose(points[-1, 1], 150.0)

    def test_param_bounds_valid(self):
        """Test that param bounds are valid tuples with min < max."""
        shape = VLineShape()
        bounds = shape.get_bounds()

        assert len(bounds) == 3
        for lo, hi in bounds:
            assert lo < hi
            assert isinstance(lo, (int, float))
            assert isinstance(hi, (int, float))

    def test_param_count_matches_bounds(self):
        """Test that param_count matches length of bounds."""
        shape = VLineShape()
        assert shape.param_count == len(shape.get_bounds())

    def test_offset_applied(self, standard_bbox):
        """Test that offset is correctly applied to points."""
        shape = VLineShape()
        params = (0.5, 0.0, 1.0)
        offset = (10, 20)
        points = shape.generate(params, standard_bbox, offset=offset, n_pts=60)

        # X should be 50 + 10 = 60
        assert np.allclose(points[:, 0], 60.0)
        # Y should start at 0 + 20 = 20
        assert np.isclose(points[0, 1], 20.0)

    def test_n_pts_parameter(self, standard_bbox):
        """Test that n_pts controls number of points generated."""
        shape = VLineShape()
        params = (0.5, 0.0, 1.0)

        for n_pts in [10, 60, 100]:
            points = shape.generate(params, standard_bbox, n_pts=n_pts)
            assert len(points) == n_pts

    def test_partial_line(self, standard_bbox):
        """Test generating partial vertical lines."""
        shape = VLineShape()
        params = (0.5, 0.25, 0.75)  # y from 25 to 75
        points = shape.generate(params, standard_bbox, n_pts=60)

        assert np.isclose(points[0, 1], 25.0)
        assert np.isclose(points[-1, 1], 75.0)


# ---------------------------------------------------------------------------
# TestHLineShape
# ---------------------------------------------------------------------------

class TestHLineShape:
    """Tests for the HLineShape class."""

    def test_generates_horizontal_points(self, standard_bbox):
        """Test that HLineShape generates a horizontal line."""
        shape = HLineShape()
        params = (0.5, 0.0, 1.0)  # y=50, x from 0 to 100
        points = shape.generate(params, standard_bbox, n_pts=60)

        # All y-coordinates should be the same (horizontal line)
        assert np.allclose(points[:, 1], 50.0)
        # X-coordinates should span from 0 to 100
        assert np.isclose(points[0, 0], 0.0)
        assert np.isclose(points[-1, 0], 100.0)

    def test_respects_bbox(self, offset_bbox):
        """Test that HLineShape respects bounding box offset."""
        shape = HLineShape()
        params = (0.5, 0.0, 1.0)  # Midpoint of bbox
        points = shape.generate(params, offset_bbox, n_pts=60)

        # Y should be at midpoint of offset bbox (50 + 0.5*100 = 100)
        assert np.allclose(points[:, 1], 100.0)
        # X should span from 50 to 150
        assert np.isclose(points[0, 0], 50.0)
        assert np.isclose(points[-1, 0], 150.0)

    def test_param_bounds_valid(self):
        """Test that param bounds are valid tuples with min < max."""
        shape = HLineShape()
        bounds = shape.get_bounds()

        assert len(bounds) == 3
        for lo, hi in bounds:
            assert lo < hi
            assert isinstance(lo, (int, float))
            assert isinstance(hi, (int, float))

    def test_param_count_matches_bounds(self):
        """Test that param_count matches length of bounds."""
        shape = HLineShape()
        assert shape.param_count == len(shape.get_bounds())

    def test_offset_applied(self, standard_bbox):
        """Test that offset is correctly applied to points."""
        shape = HLineShape()
        params = (0.5, 0.0, 1.0)
        offset = (10, 20)
        points = shape.generate(params, standard_bbox, offset=offset, n_pts=60)

        # Y should be 50 + 20 = 70
        assert np.allclose(points[:, 1], 70.0)
        # X should start at 0 + 10 = 10
        assert np.isclose(points[0, 0], 10.0)

    def test_n_pts_parameter(self, standard_bbox):
        """Test that n_pts controls number of points generated."""
        shape = HLineShape()
        params = (0.5, 0.0, 1.0)

        for n_pts in [10, 60, 100]:
            points = shape.generate(params, standard_bbox, n_pts=n_pts)
            assert len(points) == n_pts

    def test_partial_line(self, standard_bbox):
        """Test generating partial horizontal lines."""
        shape = HLineShape()
        params = (0.5, 0.25, 0.75)  # x from 25 to 75
        points = shape.generate(params, standard_bbox, n_pts=60)

        assert np.isclose(points[0, 0], 25.0)
        assert np.isclose(points[-1, 0], 75.0)


# ---------------------------------------------------------------------------
# TestDiagShape
# ---------------------------------------------------------------------------

class TestDiagShape:
    """Tests for the DiagShape class."""

    def test_generates_diagonal_points(self, standard_bbox):
        """Test that DiagShape generates a diagonal line."""
        shape = DiagShape()
        params = (0.0, 0.0, 1.0, 1.0)  # From (0,0) to (100,100)
        points = shape.generate(params, standard_bbox, n_pts=60)

        # Start point
        assert np.isclose(points[0, 0], 0.0)
        assert np.isclose(points[0, 1], 0.0)
        # End point
        assert np.isclose(points[-1, 0], 100.0)
        assert np.isclose(points[-1, 1], 100.0)
        # Points should be on diagonal (x == y)
        assert np.allclose(points[:, 0], points[:, 1])

    def test_respects_bbox(self, offset_bbox):
        """Test that DiagShape respects bounding box offset."""
        shape = DiagShape()
        params = (0.0, 0.0, 1.0, 1.0)
        points = shape.generate(params, offset_bbox, n_pts=60)

        # Start at (50, 50), end at (150, 150)
        assert np.isclose(points[0, 0], 50.0)
        assert np.isclose(points[0, 1], 50.0)
        assert np.isclose(points[-1, 0], 150.0)
        assert np.isclose(points[-1, 1], 150.0)

    def test_param_bounds_valid(self):
        """Test that param bounds are valid tuples with min < max."""
        shape = DiagShape()
        bounds = shape.get_bounds()

        assert len(bounds) == 4
        for lo, hi in bounds:
            assert lo < hi

    def test_param_count_matches_bounds(self):
        """Test that param_count matches length of bounds."""
        shape = DiagShape()
        assert shape.param_count == len(shape.get_bounds())

    def test_anti_diagonal(self, standard_bbox):
        """Test generating anti-diagonal line."""
        shape = DiagShape()
        params = (1.0, 0.0, 0.0, 1.0)  # From (100,0) to (0,100)
        points = shape.generate(params, standard_bbox, n_pts=60)

        assert np.isclose(points[0, 0], 100.0)
        assert np.isclose(points[0, 1], 0.0)
        assert np.isclose(points[-1, 0], 0.0)
        assert np.isclose(points[-1, 1], 100.0)


# ---------------------------------------------------------------------------
# TestArcShapes
# ---------------------------------------------------------------------------

class TestArcShapes:
    """Tests for arc shapes: ArcRightShape and ArcLeftShape."""

    def test_arc_right_generates_semicircle(self, standard_bbox):
        """Test that ArcRightShape generates right-opening arc."""
        shape = ArcRightShape()
        params = (0.5, 0.5, 0.3, 0.3, -90, 90)  # Centered semicircle
        points = shape.generate(params, standard_bbox, n_pts=60)

        # Should have correct number of points
        assert len(points) == 60
        # Center should be at (50, 50)
        cx, cy = 50.0, 50.0
        rx, ry = 30.0, 30.0
        # First point at angle -90 (top)
        assert np.isclose(points[0, 0], cx + rx * np.cos(np.radians(-90)), atol=1e-10)
        assert np.isclose(points[0, 1], cy + ry * np.sin(np.radians(-90)), atol=1e-10)
        # Last point at angle 90 (bottom)
        assert np.isclose(points[-1, 0], cx + rx * np.cos(np.radians(90)), atol=1e-10)
        assert np.isclose(points[-1, 1], cy + ry * np.sin(np.radians(90)), atol=1e-10)

    def test_arc_left_generates_semicircle(self, standard_bbox):
        """Test that ArcLeftShape generates left-opening arc."""
        shape = ArcLeftShape()
        params = (0.5, 0.5, 0.3, 0.3, -90, 90)  # Centered semicircle
        points = shape.generate(params, standard_bbox, n_pts=60)

        # Should have correct number of points
        assert len(points) == 60
        # Center should be at (50, 50)
        cx, cy = 50.0, 50.0
        rx, ry = 30.0, 30.0
        # First point at angle -90 (top) - note the minus sign for x
        assert np.isclose(points[0, 0], cx - rx * np.cos(np.radians(-90)), atol=1e-10)
        assert np.isclose(points[0, 1], cy + ry * np.sin(np.radians(-90)), atol=1e-10)

    def test_arc_right_param_bounds_valid(self):
        """Test ArcRightShape param bounds."""
        shape = ArcRightShape()
        bounds = shape.get_bounds()

        assert len(bounds) == 6
        for lo, hi in bounds:
            assert lo < hi

    def test_arc_left_param_bounds_valid(self):
        """Test ArcLeftShape param bounds."""
        shape = ArcLeftShape()
        bounds = shape.get_bounds()

        assert len(bounds) == 6
        for lo, hi in bounds:
            assert lo < hi

    def test_arc_shapes_param_count(self):
        """Test that arc shapes have correct param count."""
        assert ArcRightShape().param_count == 6
        assert ArcLeftShape().param_count == 6

    def test_arc_with_offset(self, standard_bbox):
        """Test that arc shapes apply offset correctly."""
        shape = ArcRightShape()
        params = (0.5, 0.5, 0.3, 0.3, -90, 90)
        offset = (10, 20)
        points = shape.generate(params, standard_bbox, offset=offset, n_pts=60)

        # Center should be shifted by offset
        cx_shifted = 50.0 + 10
        cy_shifted = 50.0 + 20
        # Check midpoint (angle 0, which is rightmost point)
        mid_idx = len(points) // 2
        # At angle 0, x = cx + rx, y = cy
        expected_x = cx_shifted + 30.0  # rx = 0.3 * 100
        expected_y = cy_shifted
        assert np.isclose(points[mid_idx, 0], expected_x, atol=1.0)
        assert np.isclose(points[mid_idx, 1], expected_y, atol=1.0)


# ---------------------------------------------------------------------------
# TestLoopShape
# ---------------------------------------------------------------------------

class TestLoopShape:
    """Tests for the LoopShape (full ellipse) class."""

    def test_generates_full_ellipse(self, standard_bbox):
        """Test that LoopShape generates a full ellipse."""
        shape = LoopShape()
        params = (0.5, 0.5, 0.3, 0.3)  # Centered circle
        points = shape.generate(params, standard_bbox, n_pts=80)

        # Should generate correct number of points (default 80 for loop)
        assert len(points) == 80

        # All points should be on the ellipse
        cx, cy = 50.0, 50.0
        rx, ry = 30.0, 30.0
        for pt in points:
            # Verify point is on ellipse: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 = 1
            normalized = ((pt[0] - cx) / rx) ** 2 + ((pt[1] - cy) / ry) ** 2
            assert np.isclose(normalized, 1.0, atol=1e-10)

    def test_param_bounds_valid(self):
        """Test LoopShape param bounds."""
        shape = LoopShape()
        bounds = shape.get_bounds()

        assert len(bounds) == 4
        for lo, hi in bounds:
            assert lo < hi

    def test_param_count_matches_bounds(self):
        """Test that param_count matches length of bounds."""
        shape = LoopShape()
        assert shape.param_count == len(shape.get_bounds())

    def test_elliptical_loop(self, rectangular_bbox):
        """Test loop with different radii fractions."""
        shape = LoopShape()
        params = (0.5, 0.5, 0.4, 0.3)  # Different rx and ry fractions
        points = shape.generate(params, rectangular_bbox, n_pts=80)

        # Verify center and radii
        cx = 0 + 0.5 * 200  # 100
        cy = 0 + 0.5 * 100  # 50
        rx = 0.4 * 200  # 80
        ry = 0.3 * 100  # 30

        # Check all points on ellipse
        for pt in points:
            normalized = ((pt[0] - cx) / rx) ** 2 + ((pt[1] - cy) / ry) ** 2
            assert np.isclose(normalized, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# TestUArcShape
# ---------------------------------------------------------------------------

class TestUArcShape:
    """Tests for the UArcShape (U-shaped arc) class."""

    def test_generates_u_arc(self, standard_bbox):
        """Test that UArcShape generates bottom half of ellipse."""
        shape = UArcShape()
        params = (0.5, 0.5, 0.3, 0.3)
        points = shape.generate(params, standard_bbox, n_pts=60)

        # Should have correct number of points
        assert len(points) == 60

        # All y-values should be >= center y (bottom half)
        cy = 50.0
        assert np.all(points[:, 1] >= cy - 1e-10)

    def test_param_bounds_valid(self):
        """Test UArcShape param bounds."""
        shape = UArcShape()
        bounds = shape.get_bounds()

        assert len(bounds) == 4
        for lo, hi in bounds:
            assert lo < hi

    def test_param_count_matches_bounds(self):
        """Test that param_count matches length of bounds."""
        shape = UArcShape()
        assert shape.param_count == len(shape.get_bounds())


# ---------------------------------------------------------------------------
# Parametrized tests for ALL shapes in SHAPES registry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape_name", SHAPES.keys())
def test_shape_generates_valid_points(shape_name):
    """Test that each shape in SHAPES registry generates valid points."""
    shape = SHAPES[shape_name]
    bbox = (0, 0, 100, 100)

    # Use default parameters
    params = shape.get_default_params()
    points = shape.generate(params, bbox, n_pts=60)

    # Assert points were generated
    assert len(points) > 0
    assert points.shape[1] == 2  # Each point has x and y

    # Assert all points are approximately within bbox (with some margin for arcs)
    margin = 50  # Allow margin for arc shapes that may extend outside
    assert np.all(points[:, 0] >= bbox[0] - margin)
    assert np.all(points[:, 0] <= bbox[2] + margin)
    assert np.all(points[:, 1] >= bbox[1] - margin)
    assert np.all(points[:, 1] <= bbox[3] + margin)


@pytest.mark.parametrize("shape_name", SHAPES.keys())
def test_param_count_matches_get_bounds(shape_name):
    """Test that param_count property matches get_bounds() length for each shape."""
    shape = SHAPES[shape_name]
    bounds = shape.get_bounds()

    assert shape.param_count == len(bounds)


@pytest.mark.parametrize("shape_name", SHAPES.keys())
def test_shape_can_be_called_without_errors(shape_name):
    """Test that all shapes can be called without errors."""
    shape = SHAPES[shape_name]
    bbox = (0, 0, 100, 100)

    # Use default parameters
    params = shape.get_default_params()

    # Should not raise any exceptions
    try:
        points = shape.generate(params, bbox)
        assert points is not None
        assert isinstance(points, np.ndarray)
    except Exception as e:
        pytest.fail(f"Shape {shape_name} raised exception: {e}")


@pytest.mark.parametrize("shape_name", SHAPES.keys())
def test_shape_default_params_within_bounds(shape_name):
    """Test that default params are within bounds for each shape."""
    shape = SHAPES[shape_name]
    default_params = shape.get_default_params()
    bounds = shape.get_bounds()

    assert len(default_params) == len(bounds)
    for param, (lo, hi) in zip(default_params, bounds):
        assert lo <= param <= hi, f"Default param {param} not in bounds [{lo}, {hi}]"


@pytest.mark.parametrize("shape_name", SHAPES.keys())
def test_shape_validate_params(shape_name):
    """Test validate_params method for each shape."""
    shape = SHAPES[shape_name]
    bounds = shape.get_bounds()

    # Valid params (midpoints)
    valid_params = tuple((lo + hi) / 2 for lo, hi in bounds)
    assert shape.validate_params(valid_params)

    # Invalid params (too few)
    if len(bounds) > 0:
        assert not shape.validate_params(valid_params[:-1])

    # Invalid params (out of bounds low)
    invalid_low = tuple(lo - 1 for lo, hi in bounds)
    assert not shape.validate_params(invalid_low)

    # Invalid params (out of bounds high)
    invalid_high = tuple(hi + 1 for lo, hi in bounds)
    assert not shape.validate_params(invalid_high)


# ---------------------------------------------------------------------------
# Test SHAPES registry
# ---------------------------------------------------------------------------

class TestSHAPESRegistry:
    """Tests for the SHAPES registry."""

    def test_registry_contains_all_expected_shapes(self):
        """Test that all expected shapes are in the registry."""
        expected_shapes = ['vline', 'hline', 'diag', 'arc_right', 'arc_left', 'loop', 'u_arc']
        for shape_name in expected_shapes:
            assert shape_name in SHAPES, f"Expected shape {shape_name} not in SHAPES"

    def test_registry_values_are_shape_instances(self):
        """Test that all registry values are Shape subclass instances."""
        for name, shape in SHAPES.items():
            assert isinstance(shape, Shape), f"{name} is not a Shape instance"

    def test_registry_shape_types(self):
        """Test that registry contains correct shape class types."""
        assert isinstance(SHAPES['vline'], VLineShape)
        assert isinstance(SHAPES['hline'], HLineShape)
        assert isinstance(SHAPES['diag'], DiagShape)
        assert isinstance(SHAPES['arc_right'], ArcRightShape)
        assert isinstance(SHAPES['arc_left'], ArcLeftShape)
        assert isinstance(SHAPES['loop'], LoopShape)
        assert isinstance(SHAPES['u_arc'], UArcShape)


# ---------------------------------------------------------------------------
# Test get_param_bounds function
# ---------------------------------------------------------------------------

class TestGetParamBounds:
    """Tests for the get_param_bounds utility function."""

    def test_single_template(self):
        """Test bounds extraction for single template."""
        templates = [{'shape': 'vline'}]
        bounds, slices = get_param_bounds(templates)

        assert len(bounds) == 3  # VLineShape has 3 params
        assert slices == [(0, 3)]

    def test_multiple_templates(self):
        """Test bounds extraction for multiple templates."""
        templates = [
            {'shape': 'vline'},
            {'shape': 'hline'},
        ]
        bounds, slices = get_param_bounds(templates)

        assert len(bounds) == 6  # 3 + 3
        assert slices == [(0, 3), (3, 6)]

    def test_bounds_override(self):
        """Test that bounds can be overridden per template."""
        templates = [
            {'shape': 'vline', 'bounds': [(0.4, 0.6), None, None]},
        ]
        bounds, slices = get_param_bounds(templates)

        # First bound should be overridden
        assert bounds[0] == (0.4, 0.6)
        # Other bounds should be defaults from VLineShape
        default_bounds = VLineShape().get_bounds()
        assert bounds[1] == default_bounds[1]
        assert bounds[2] == default_bounds[2]

    def test_mixed_shapes(self):
        """Test bounds extraction for mixed shape types."""
        templates = [
            {'shape': 'vline'},
            {'shape': 'diag'},
            {'shape': 'loop'},
        ]
        bounds, slices = get_param_bounds(templates)

        # 3 + 4 + 4 = 11 total params
        assert len(bounds) == 11
        assert slices == [(0, 3), (3, 7), (7, 11)]


# ---------------------------------------------------------------------------
# Test param_vector_to_shapes function
# ---------------------------------------------------------------------------

class TestParamVectorToShapes:
    """Tests for the param_vector_to_shapes utility function."""

    def test_basic_conversion(self):
        """Test basic parameter vector to shapes conversion."""
        shape_types = ['vline', 'hline']
        slices = [(0, 3), (3, 6)]
        bbox = (0, 0, 100, 100)

        # Create parameter vector with midpoint values
        param_vector = np.array([0.5, 0.25, 0.75, 0.5, 0.25, 0.75])

        shapes = param_vector_to_shapes(param_vector, shape_types, slices, bbox, n_pts=60)

        assert len(shapes) == 2
        assert len(shapes[0]) == 60
        assert len(shapes[1]) == 60

    def test_auto_n_pts_calculation(self):
        """Test automatic n_pts calculation based on bbox."""
        shape_types = ['vline']
        slices = [(0, 3)]
        bbox = (0, 0, 100, 100)
        param_vector = np.array([0.5, 0.0, 1.0])

        # With n_pts=None, should auto-calculate
        shapes = param_vector_to_shapes(param_vector, shape_types, slices, bbox, n_pts=None)

        # Should use formula: max(MIN_SHAPE_POINTS, int(diagonal / POINT_SPACING_TARGET))
        diagonal = (100**2 + 100**2) ** 0.5
        expected_n_pts = max(MIN_SHAPE_POINTS, int(diagonal / POINT_SPACING_TARGET))
        assert len(shapes[0]) == expected_n_pts

    def test_all_shape_types(self):
        """Test conversion with all shape types."""
        shape_types = list(SHAPES.keys())
        bbox = (0, 0, 100, 100)

        # Build param vector with default params for each shape
        param_list = []
        slices = []
        offset = 0
        for stype in shape_types:
            shape = SHAPES[stype]
            params = shape.get_default_params()
            param_list.extend(params)
            slices.append((offset, offset + len(params)))
            offset += len(params)

        param_vector = np.array(param_list)
        shapes = param_vector_to_shapes(param_vector, shape_types, slices, bbox, n_pts=60)

        assert len(shapes) == len(shape_types)
        for shape_points in shapes:
            assert len(shape_points) == 60
            assert shape_points.shape[1] == 2


# ---------------------------------------------------------------------------
# Test make_point_cloud function
# ---------------------------------------------------------------------------

class TestMakePointCloud:
    """Tests for the make_point_cloud utility function."""

    def test_basic_point_cloud(self):
        """Test basic point cloud generation."""
        # Create a simple square mask
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True

        points = make_point_cloud(mask, spacing=2)

        # Should have points
        assert len(points) > 0
        # All points should be within the True region
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            assert mask[y, x], f"Point ({x}, {y}) is outside mask"

    def test_empty_mask(self):
        """Test point cloud with empty mask."""
        mask = np.zeros((100, 100), dtype=bool)
        points = make_point_cloud(mask, spacing=2)

        # Should return empty array
        assert len(points) == 0

    def test_full_mask(self):
        """Test point cloud with fully True mask."""
        mask = np.ones((100, 100), dtype=bool)
        points = make_point_cloud(mask, spacing=2)

        # Should have approximately (100/2)^2 = 2500 points
        assert len(points) > 0
        # Grid of 50x50 points
        assert len(points) == 50 * 50

    def test_spacing_affects_density(self):
        """Test that spacing parameter affects point density."""
        mask = np.ones((100, 100), dtype=bool)

        points_sparse = make_point_cloud(mask, spacing=4)
        points_dense = make_point_cloud(mask, spacing=2)

        # Denser spacing should produce more points
        assert len(points_dense) > len(points_sparse)

    def test_point_dtype(self):
        """Test that returned points have float dtype."""
        mask = np.ones((50, 50), dtype=bool)
        points = make_point_cloud(mask, spacing=2)

        assert points.dtype == float


# ---------------------------------------------------------------------------
# Test adaptive_radius function
# ---------------------------------------------------------------------------

class TestAdaptiveRadius:
    """Tests for the adaptive_radius utility function."""

    def test_basic_radius(self):
        """Test basic radius computation."""
        # Create a solid rectangle mask
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 30:70] = True

        radius = adaptive_radius(mask, spacing=2)

        # Should return a positive value
        assert radius > 0

    def test_empty_mask_returns_floor(self):
        """Test that empty mask returns minimum radius."""
        mask = np.zeros((100, 100), dtype=bool)
        radius = adaptive_radius(mask, spacing=2)

        # Should return max of MIN_RADIUS and spacing * RADIUS_FLOOR_MULTIPLIER
        expected_floor = 2 * RADIUS_FLOOR_MULTIPLIER
        assert radius == max(MIN_RADIUS, expected_floor)

    def test_thin_stroke_radius(self):
        """Test radius for thin stroke."""
        # Create a thin horizontal line
        mask = np.zeros((100, 100), dtype=bool)
        mask[48:52, 10:90] = True  # 4 pixel wide line

        radius = adaptive_radius(mask, spacing=2)

        # Radius should be at least the floor
        floor = 2 * RADIUS_FLOOR_MULTIPLIER
        assert radius >= floor

    def test_thick_stroke_radius(self):
        """Test radius for thick stroke."""
        # Create a thick circle
        mask = np.zeros((100, 100), dtype=bool)
        y, x = np.ogrid[:100, :100]
        mask[(x - 50) ** 2 + (y - 50) ** 2 < 30 ** 2] = True

        radius = adaptive_radius(mask, spacing=2)

        # Should be larger than for thin strokes
        assert radius > 2 * RADIUS_FLOOR_MULTIPLIER


# ---------------------------------------------------------------------------
# Test module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    """Tests for module-level constants."""

    def test_constants_are_reasonable(self):
        """Test that module constants have reasonable values."""
        assert MIN_SHAPE_POINTS > 0
        assert POINT_SPACING_TARGET > 0
        assert RADIUS_FLOOR_MULTIPLIER > 0
        assert MIN_RADIUS > 0
        assert 0 < DISTANCE_PERCENTILE <= 100

    def test_min_shape_points_value(self):
        """Test MIN_SHAPE_POINTS has expected value."""
        assert MIN_SHAPE_POINTS == 60

    def test_point_spacing_target_value(self):
        """Test POINT_SPACING_TARGET has expected value."""
        assert POINT_SPACING_TARGET == 1.5


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_size_bbox(self):
        """Test handling of zero-size bounding box."""
        shape = VLineShape()
        bbox = (50, 50, 50, 50)  # Zero width and height
        params = (0.5, 0.0, 1.0)

        # Should not crash
        points = shape.generate(params, bbox, n_pts=60)
        assert len(points) == 60

    def test_negative_bbox_dimensions(self):
        """Test handling of inverted bounding box."""
        shape = VLineShape()
        bbox = (100, 100, 0, 0)  # Inverted
        params = (0.5, 0.0, 1.0)

        # Should still generate points (behavior depends on implementation)
        points = shape.generate(params, bbox, n_pts=60)
        assert len(points) == 60

    def test_very_large_bbox(self):
        """Test handling of very large bounding box."""
        shape = VLineShape()
        bbox = (0, 0, 10000, 10000)
        params = (0.5, 0.0, 1.0)

        points = shape.generate(params, bbox, n_pts=60)
        assert len(points) == 60
        # X should be at midpoint
        assert np.allclose(points[:, 0], 5000.0)

    def test_fractional_params_at_boundaries(self):
        """Test parameters at exact boundary values."""
        shape = VLineShape()
        bbox = (0, 0, 100, 100)

        # Test with boundary values
        params = (0.0, 0.0, 0.5)  # At lower bounds
        points = shape.generate(params, bbox, n_pts=60)
        assert np.allclose(points[:, 0], 0.0)

        params = (1.0, 0.5, 1.0)  # At upper bounds
        points = shape.generate(params, bbox, n_pts=60)
        assert np.allclose(points[:, 0], 100.0)


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_flow(self):
        """Test full flow from templates to shape points."""
        templates = [
            {'shape': 'vline'},
            {'shape': 'hline'},
            {'shape': 'diag'},
        ]
        bbox = (0, 0, 100, 100)

        # Get bounds
        bounds, slices = get_param_bounds(templates)

        # Create param vector with midpoints
        param_vector = np.array([(lo + hi) / 2 for lo, hi in bounds])

        # Convert to shapes
        shape_types = [t['shape'] for t in templates]
        shapes = param_vector_to_shapes(param_vector, shape_types, slices, bbox, n_pts=60)

        # Verify output
        assert len(shapes) == 3
        for shape_points in shapes:
            assert len(shape_points) == 60
            assert shape_points.shape == (60, 2)

    def test_shape_coverage_scoring_compatible(self):
        """Test that generated shapes are compatible with point cloud scoring."""
        # Create a mask
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 40:60] = True  # Vertical bar

        # Generate point cloud
        points = make_point_cloud(mask, spacing=2)
        assert len(points) > 0

        # Generate a vertical line shape
        shape = VLineShape()
        params = (0.5, 0.2, 0.8)
        shape_points = shape.generate(params, (0, 0, 100, 100), n_pts=60)

        # Both should be 2D arrays suitable for distance calculations
        assert points.ndim == 2
        assert shape_points.ndim == 2
        assert points.shape[1] == 2
        assert shape_points.shape[1] == 2
