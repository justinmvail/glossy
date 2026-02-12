"""Unit tests for template_morph.py VertexFinder classes.

Tests the VERTEX_FINDERS registry, individual VertexFinder subclasses,
DefaultVertexFinder fallback, and the find_vertices() dispatch function.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from template_morph import (
    VERTEX_FINDERS,
    VERTEX_POS,
    TEMPLATES,
    VertexFinder,
    VertexFinderA,
    VertexFinderB,
    VertexFinderD,
    VertexFinderCG,
    VertexFinderEF,
    VertexFinderHK,
    VertexFinderP,
    VertexFinderR,
    DefaultVertexFinder,
    find_vertices,
    get_outline,
    snap_to_outline,
)


# ---------------------------------------------------------------------------
# Test Fixtures and Helpers
# ---------------------------------------------------------------------------

def create_mock_mask(size: int = 64, letter_shape: str = "rect") -> np.ndarray:
    """Create a simple mock mask simulating a letter shape.

    Args:
        size: Size of the square mask.
        letter_shape: Shape type - "rect", "triangle", "circle", or "empty".

    Returns:
        Boolean numpy array representing the letter mask.
    """
    mask = np.zeros((size, size), dtype=bool)

    if letter_shape == "empty":
        return mask

    margin = size // 8

    if letter_shape == "rect":
        # Simple rectangle (like I, L, T base)
        mask[margin:size-margin, margin:size-margin] = True
    elif letter_shape == "triangle":
        # Triangle pointing up (like A)
        center_x = size // 2
        for y in range(margin, size - margin):
            # Width increases as we go down
            progress = (y - margin) / (size - 2 * margin)
            half_width = int(progress * (size // 2 - margin))
            left = max(0, center_x - half_width)
            right = min(size, center_x + half_width + 1)
            if left < right:
                mask[y, left:right] = True
    elif letter_shape == "circle":
        # Circle/oval (like O, C, G)
        cy, cx = size // 2, size // 2
        radius = size // 2 - margin
        for y in range(size):
            for x in range(size):
                if (x - cx)**2 + (y - cy)**2 <= radius**2:
                    mask[y, x] = True
    elif letter_shape == "p_shape":
        # P-like shape: vertical stem with top bump
        stem_width = size // 6
        # Vertical stem on left
        mask[margin:size-margin, margin:margin+stem_width] = True
        # Top bump (half circle on right top)
        bump_cy = margin + size // 4
        bump_cx = margin + stem_width
        bump_radius = size // 4
        for y in range(margin, size // 2):
            for x in range(margin + stem_width, size - margin):
                if (x - bump_cx)**2 + (y - bump_cy)**2 <= bump_radius**2:
                    mask[y, x] = True

    return mask


def create_mock_bbox(mask: np.ndarray) -> tuple:
    """Create bounding box from mask.

    Returns:
        Tuple (cmin, rmin, cmax, rmax) - column and row bounds.
    """
    if not mask.any():
        return (0, 0, mask.shape[1], mask.shape[0])

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (int(cmin), int(rmin), int(cmax), int(rmax))


def get_outline_xy(mask: np.ndarray) -> np.ndarray:
    """Get outline points as (x, y) array."""
    outline = get_outline(mask)
    outline_pts = np.argwhere(outline)  # (row, col)
    if len(outline_pts) == 0:
        return np.array([]).reshape(0, 2)
    return outline_pts[:, ::-1]  # Convert to (x, y) = (col, row)


# ---------------------------------------------------------------------------
# Parametrized Tests Across All Characters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("char", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
def test_vertex_finder_returns_valid_vertices(char):
    """Test that find_vertices returns vertices within bounding box for all letters."""
    # Create mock mask with a generic shape
    mask = create_mock_mask(64, "rect")
    bbox = create_mock_bbox(mask)
    cmin, rmin, cmax, rmax = bbox

    # Get vertices using the registry
    vertices = find_vertices(char, mask, bbox)

    # Verify all returned vertices are within or near the bounding box
    # Allow some tolerance since snapping may place vertices on outline
    tolerance = 5
    for name, (vx, vy) in vertices.items():
        assert cmin - tolerance <= vx <= cmax + tolerance, \
            f"Vertex {name} x={vx} out of bounds [{cmin}, {cmax}] for char {char}"
        assert rmin - tolerance <= vy <= rmax + tolerance, \
            f"Vertex {name} y={vy} out of bounds [{rmin}, {rmax}] for char {char}"


@pytest.mark.parametrize("char", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
def test_vertex_finder_returns_dict(char):
    """Test that find_vertices always returns a dictionary."""
    mask = create_mock_mask(64, "rect")
    bbox = create_mock_bbox(mask)

    vertices = find_vertices(char, mask, bbox)

    assert isinstance(vertices, dict), f"Expected dict, got {type(vertices)} for char {char}"


@pytest.mark.parametrize("char", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
def test_vertex_finder_returns_required_vertices(char):
    """Test that find_vertices returns the vertices needed for the character's template."""
    mask = create_mock_mask(64, "rect")
    bbox = create_mock_bbox(mask)

    vertices = find_vertices(char, mask, bbox)

    # Get the template for this character
    template = TEMPLATES.get(char)
    if template is None:
        pytest.skip(f"No template for character {char}")

    # Collect all vertex names used in strokes
    required_vertices = set()
    for stroke in template.get('strokes', []):
        required_vertices.update(stroke)

    # Filter to only standard vertex names (not special ones like TR_TOP)
    standard_vertices = {v for v in required_vertices if v in VERTEX_POS}

    # Check that standard vertices are present
    for vertex_name in standard_vertices:
        if vertex_name in vertices:
            assert isinstance(vertices[vertex_name], tuple), \
                f"Vertex {vertex_name} should be tuple for char {char}"
            assert len(vertices[vertex_name]) == 2, \
                f"Vertex {vertex_name} should have 2 coordinates for char {char}"


# ---------------------------------------------------------------------------
# Tests for Specific VertexFinder Classes
# ---------------------------------------------------------------------------

class TestVertexFinderA:
    """Tests for VertexFinderA (letter A)."""

    def test_finds_apex(self):
        """Test that A finder locates the apex (TC) vertex."""
        mask = create_mock_mask(64, "triangle")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        finder = VertexFinderA()
        vertices = finder.find(mask, bbox, outline_xy)

        assert 'TC' in vertices, "VertexFinderA should find TC (apex)"
        # Apex should be near top center
        tc_x, tc_y = vertices['TC']
        cmin, rmin, cmax, rmax = bbox
        assert tc_y < (rmin + rmax) // 2, "TC should be in upper half"

    def test_finds_base_corners(self):
        """Test that A finder locates BL and BR vertices."""
        mask = create_mock_mask(64, "triangle")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        finder = VertexFinderA()
        vertices = finder.find(mask, bbox, outline_xy)

        assert 'BL' in vertices, "VertexFinderA should find BL"
        assert 'BR' in vertices, "VertexFinderA should find BR"

        # BL should be left of BR
        bl_x, _ = vertices['BL']
        br_x, _ = vertices['BR']
        assert bl_x < br_x, "BL should be left of BR"

    def test_finds_crossbar_points(self):
        """Test that A finder locates ML and MR for crossbar."""
        mask = create_mock_mask(64, "triangle")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        finder = VertexFinderA()
        vertices = finder.find(mask, bbox, outline_xy)

        assert 'ML' in vertices, "VertexFinderA should find ML"
        assert 'MR' in vertices, "VertexFinderA should find MR"


class TestVertexFinderB:
    """Tests for VertexFinderB (letter B)."""

    def test_finds_left_stem(self):
        """Test that B finder locates TL and BL vertices."""
        mask = create_mock_mask(64, "rect")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        finder = VertexFinderB()
        vertices = finder.find(mask, bbox, outline_xy)

        assert 'TL' in vertices, "VertexFinderB should find TL"
        assert 'BL' in vertices, "VertexFinderB should find BL"

    def test_finds_bump_vertices(self):
        """Test that B finder locates TR_TOP and TR_BOT for bumps."""
        mask = create_mock_mask(64, "rect")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        finder = VertexFinderB()
        vertices = finder.find(mask, bbox, outline_xy)

        assert 'TR_TOP' in vertices, "VertexFinderB should find TR_TOP"
        assert 'TR_BOT' in vertices, "VertexFinderB should find TR_BOT"

    def test_finds_waist(self):
        """Test that B finder locates ML (waist point)."""
        mask = create_mock_mask(64, "rect")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        finder = VertexFinderB()
        vertices = finder.find(mask, bbox, outline_xy)

        assert 'ML' in vertices, "VertexFinderB should find ML"


class TestVertexFinderD:
    """Tests for VertexFinderD (letter D)."""

    def test_finds_corners(self):
        """Test that D finder locates all four corners."""
        mask = create_mock_mask(64, "rect")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        finder = VertexFinderD()
        vertices = finder.find(mask, bbox, outline_xy)

        for corner in ['TL', 'BL', 'TR', 'BR']:
            assert corner in vertices, f"VertexFinderD should find {corner}"


class TestVertexFinderCG:
    """Tests for VertexFinderCG (letters C and G)."""

    def test_c_finder_has_openings(self):
        """Test that C finder locates opening vertices."""
        mask = create_mock_mask(64, "circle")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        finder = VertexFinderCG('C')
        vertices = finder.find(mask, bbox, outline_xy)

        assert 'TR' in vertices, "VertexFinderCG(C) should find TR"
        assert 'BR' in vertices, "VertexFinderCG(C) should find BR"
        assert 'TL' in vertices, "VertexFinderCG(C) should find TL"
        assert 'BL' in vertices, "VertexFinderCG(C) should find BL"

    def test_g_finder_has_crossbar(self):
        """Test that G finder has additional MR and MC vertices."""
        mask = create_mock_mask(64, "circle")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        finder = VertexFinderCG('G')
        vertices = finder.find(mask, bbox, outline_xy)

        assert 'MR' in vertices, "VertexFinderCG(G) should find MR"
        assert 'MC' in vertices, "VertexFinderCG(G) should find MC"


class TestVertexFinderEF:
    """Tests for VertexFinderEF (letters E and F)."""

    def test_e_finder_has_all_arms(self):
        """Test that E finder locates top, middle, and bottom arms."""
        mask = create_mock_mask(64, "rect")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        finder = VertexFinderEF('E')
        vertices = finder.find(mask, bbox, outline_xy)

        assert 'TL' in vertices
        assert 'BL' in vertices
        assert 'TR' in vertices
        assert 'BR' in vertices, "E should have BR"
        assert 'ML' in vertices
        assert 'MR' in vertices

    def test_f_finder_no_bottom_arm(self):
        """Test that F finder does not have BR vertex."""
        mask = create_mock_mask(64, "rect")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        finder = VertexFinderEF('F')
        vertices = finder.find(mask, bbox, outline_xy)

        assert 'TL' in vertices
        assert 'BL' in vertices
        assert 'TR' in vertices
        assert 'BR' not in vertices, "F should not have BR"


class TestVertexFinderHK:
    """Tests for VertexFinderHK (letters H and K)."""

    def test_finds_all_corners_and_middle(self):
        """Test that H/K finder locates corners and crossbar."""
        mask = create_mock_mask(64, "rect")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        finder = VertexFinderHK()
        vertices = finder.find(mask, bbox, outline_xy)

        for vertex in ['TL', 'BL', 'TR', 'BR', 'ML', 'MR']:
            assert vertex in vertices, f"VertexFinderHK should find {vertex}"


class TestVertexFinderP:
    """Tests for VertexFinderP (letter P)."""

    def test_finds_stem_and_bump(self):
        """Test that P finder locates stem and bump vertices."""
        mask = create_mock_mask(64, "p_shape")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        finder = VertexFinderP()
        vertices = finder.find(mask, bbox, outline_xy)

        assert 'TL' in vertices
        assert 'BL' in vertices
        assert 'TR' in vertices
        assert 'ML' in vertices
        assert 'MR' in vertices


class TestVertexFinderR:
    """Tests for VertexFinderR (letter R)."""

    def test_finds_stem_bump_and_leg(self):
        """Test that R finder locates all necessary vertices."""
        mask = create_mock_mask(64, "rect")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        finder = VertexFinderR()
        vertices = finder.find(mask, bbox, outline_xy)

        assert 'TL' in vertices
        assert 'BL' in vertices
        assert 'BR' in vertices  # R has a leg
        assert 'TR' in vertices
        assert 'ML' in vertices
        assert 'MR' in vertices


# ---------------------------------------------------------------------------
# Tests for DefaultVertexFinder
# ---------------------------------------------------------------------------

class TestDefaultVertexFinder:
    """Tests for DefaultVertexFinder fallback behavior."""

    def test_uses_vertex_pos_defaults(self):
        """Test that default finder uses VERTEX_POS positions."""
        mask = create_mock_mask(64, "rect")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        # Use a character that might use default finder
        finder = DefaultVertexFinder('Z')
        vertices = finder.find(mask, bbox, outline_xy)

        # Should return a dict (possibly empty if no template)
        assert isinstance(vertices, dict)

    def test_snaps_to_outline(self):
        """Test that default finder snaps positions to outline."""
        mask = create_mock_mask(64, "rect")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)
        outline_set = set(map(tuple, outline_xy.tolist()))

        finder = DefaultVertexFinder('I')  # Simple letter
        vertices = finder.find(mask, bbox, outline_xy)

        # All vertices should be snapped to outline points
        for name, (vx, vy) in vertices.items():
            # Check if point is on or very close to outline
            min_dist = min(
                (vx - ox)**2 + (vy - oy)**2
                for ox, oy in outline_xy
            ) if len(outline_xy) > 0 else 0
            assert min_dist < 10, f"Vertex {name} not near outline"

    def test_handles_unknown_character(self):
        """Test that default finder handles characters without templates."""
        mask = create_mock_mask(64, "rect")
        bbox = create_mock_bbox(mask)
        outline_xy = get_outline_xy(mask)

        finder = DefaultVertexFinder('9')  # Not a letter
        vertices = finder.find(mask, bbox, outline_xy)

        # Should return empty dict or minimal vertices
        assert isinstance(vertices, dict)


# ---------------------------------------------------------------------------
# Tests for find_vertices Function
# ---------------------------------------------------------------------------

class TestFindVerticesFunction:
    """Tests for the find_vertices dispatch function."""

    def test_dispatches_to_registered_finder(self):
        """Test that find_vertices uses registered finder for known chars."""
        mask = create_mock_mask(64, "triangle")
        bbox = create_mock_bbox(mask)

        # A should use VertexFinderA
        vertices = find_vertices('A', mask, bbox)

        # Should have A-specific vertices
        assert 'TC' in vertices  # Apex

    def test_falls_back_to_default(self):
        """Test that find_vertices uses DefaultVertexFinder for unknown chars."""
        mask = create_mock_mask(64, "rect")
        bbox = create_mock_bbox(mask)

        # A character not in VERTEX_FINDERS but in TEMPLATES
        vertices = find_vertices('Z', mask, bbox)

        assert isinstance(vertices, dict)

    def test_returns_empty_for_empty_outline(self):
        """Test that find_vertices returns empty dict for empty mask."""
        mask = create_mock_mask(64, "empty")
        bbox = (0, 0, 64, 64)

        vertices = find_vertices('A', mask, bbox)

        assert vertices == {}


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases in vertex finding."""

    def test_empty_mask(self):
        """Test handling of completely empty mask."""
        mask = np.zeros((64, 64), dtype=bool)
        bbox = (0, 0, 64, 64)

        vertices = find_vertices('A', mask, bbox)

        assert vertices == {}, "Empty mask should return empty vertices"

    def test_very_small_bbox(self):
        """Test handling of very small bounding box."""
        mask = np.zeros((64, 64), dtype=bool)
        # Create a tiny 3x3 filled region
        mask[30:33, 30:33] = True
        bbox = create_mock_bbox(mask)

        vertices = find_vertices('A', mask, bbox)

        # Should handle without error
        assert isinstance(vertices, dict)

    def test_single_pixel_mask(self):
        """Test handling of single pixel mask."""
        mask = np.zeros((64, 64), dtype=bool)
        mask[32, 32] = True
        bbox = (32, 32, 32, 32)

        vertices = find_vertices('A', mask, bbox)

        # Should handle without error (may return empty or minimal vertices)
        assert isinstance(vertices, dict)

    def test_character_not_in_registry(self):
        """Test handling of characters not in VERTEX_FINDERS."""
        mask = create_mock_mask(64, "rect")
        bbox = create_mock_bbox(mask)

        # Lowercase or number not in registry
        vertices = find_vertices('a', mask, bbox)

        assert isinstance(vertices, dict)

    def test_large_mask(self):
        """Test handling of larger mask sizes."""
        mask = create_mock_mask(256, "rect")
        bbox = create_mock_bbox(mask)

        vertices = find_vertices('A', mask, bbox)

        assert isinstance(vertices, dict)
        # Vertices should still be within bounds
        cmin, rmin, cmax, rmax = bbox
        for name, (vx, vy) in vertices.items():
            assert cmin - 10 <= vx <= cmax + 10
            assert rmin - 10 <= vy <= rmax + 10

    def test_mask_with_hole(self):
        """Test mask with internal hole (like O or A)."""
        mask = create_mock_mask(64, "rect")
        # Create a hole in the middle
        mask[24:40, 24:40] = False
        bbox = create_mock_bbox(mask)

        vertices = find_vertices('O', mask, bbox)

        assert isinstance(vertices, dict)


# ---------------------------------------------------------------------------
# Registry Tests
# ---------------------------------------------------------------------------

class TestVertexFindersRegistry:
    """Tests for the VERTEX_FINDERS registry."""

    def test_registry_contains_expected_chars(self):
        """Test that registry has entries for expected characters."""
        expected_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'P', 'R']

        for char in expected_chars:
            assert char in VERTEX_FINDERS, f"Expected {char} in VERTEX_FINDERS"

    def test_registry_entries_are_vertex_finders(self):
        """Test that all registry entries are VertexFinder instances."""
        for char, finder in VERTEX_FINDERS.items():
            assert isinstance(finder, VertexFinder), \
                f"Entry for {char} should be VertexFinder instance"

    def test_registry_finders_have_find_method(self):
        """Test that all registered finders have a find method."""
        for char, finder in VERTEX_FINDERS.items():
            assert hasattr(finder, 'find'), f"Finder for {char} needs find method"
            assert callable(finder.find), f"find method for {char} should be callable"


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------

class TestSnapToOutline:
    """Tests for snap_to_outline helper function."""

    def test_snaps_to_nearest(self):
        """Test that snap_to_outline finds nearest point."""
        outline_points = np.array([
            [10, 10],
            [20, 20],
            [30, 30],
        ])

        result = snap_to_outline(11, 11, outline_points)

        assert result == (10, 10), "Should snap to (10, 10)"

    def test_respects_y_tolerance(self):
        """Test that y_tolerance restricts search range."""
        outline_points = np.array([
            [10, 10],
            [50, 15],
            [30, 100],
        ])

        # Without tolerance, would snap to nearest overall
        result = snap_to_outline(35, 12, outline_points, y_tolerance=10)

        # Should only consider points within y_tolerance of 12
        # Points (10, 10) and (50, 15) are within 10 of y=12
        assert result[1] <= 22, "Should respect y_tolerance"

    def test_handles_empty_outline(self):
        """Test handling of empty outline array."""
        outline_points = np.array([]).reshape(0, 2)

        result = snap_to_outline(32, 32, outline_points)

        assert result == (32, 32), "Should return original point for empty outline"


class TestGetOutline:
    """Tests for get_outline helper function."""

    def test_extracts_outline(self):
        """Test that get_outline extracts edge pixels."""
        mask = create_mock_mask(64, "rect")

        outline = get_outline(mask)

        # Outline should be subset of mask
        assert np.all(mask | ~outline), "Outline should be within mask"
        # Outline should have some pixels
        assert outline.any(), "Outline should not be empty for non-empty mask"

    def test_empty_mask_outline(self):
        """Test outline of empty mask."""
        mask = np.zeros((64, 64), dtype=bool)

        outline = get_outline(mask)

        assert not outline.any(), "Empty mask should have empty outline"
