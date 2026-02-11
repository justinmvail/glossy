#!/usr/bin/env python3
"""Template-based stroke generation for font characters.

This module provides a template morphing approach to generate stroke paths
for font characters. Each letter has predefined vertices and stroke paths
that are morphed to fit the actual rendered font outline.

The morphing process:
    1. Render the character and extract its outline
    2. Find letter-specific vertices (corners, endpoints) on the outline
    3. Interpolate stroke paths between vertices
    4. Iteratively refine strokes to stay inside the font shape

This approach produces more consistent and semantically meaningful stroke
paths compared to pure skeletonization, as it uses knowledge of how letters
are typically written.

Design Patterns:
    The module uses a Registry Pattern for character-specific vertex finders.
    Each character with special vertex detection logic has a VertexFinder
    subclass registered in VERTEX_FINDERS. This allows:
    - Adding new character handlers without modifying find_vertices()
    - Isolating and testing character-specific logic independently
    - Clear extension points for new characters

Supported Characters:
    Currently supports A-Z uppercase letters. Each letter has a template
    defining its stroke paths and vertex positions.

Example:
    Generate strokes for a single character::

        from template_morph import morph_to_font, visualize_letter

        mask, vertices, strokes = morph_to_font('A', '/path/to/font.ttf')
        for i, stroke in enumerate(strokes):
            print(f"Stroke {i}: {len(stroke)} points")

    Create visualization of all letters::

        from template_morph import visualize_alphabet
        visualize_alphabet('/path/to/font.ttf', output_path='/tmp/strokes.png')

    Register a custom vertex finder::

        from template_morph import VertexFinder, VERTEX_FINDERS

        class MyCharVertexFinder(VertexFinder):
            def find(self, font_mask, bbox, outline_xy):
                # Custom vertex detection logic
                return {'TL': (x1, y1), 'BR': (x2, y2)}

        VERTEX_FINDERS['X'] = MyCharVertexFinder

Attributes:
    TEMPLATES: Dict mapping characters to their stroke templates.
    VERTEX_POS: Dict mapping vertex names to relative positions.
    VERTEX_FINDERS: Dict mapping characters to VertexFinder classes.
"""

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

# Letter templates: vertices connected by strokes
TEMPLATES = {
    'A': {'strokes': [['BL', 'TC'], ['TC', 'BR'], ['ML', 'MR']], 'no_morph': [2]},
    'B': {'strokes': [['TL', 'BL'], ['TL', 'TR_TOP', 'ML'], ['ML', 'TR_BOT', 'BL']], 'no_morph': [1, 2]},
    'C': {'strokes': [['TR', 'TL', 'BL', 'BR']]},
    'D': {'strokes': [['TL', 'BL'], ['TL', 'TR', 'BR', 'BL']]},
    'E': {'strokes': [['TL', 'BL'], ['TL', 'TR'], ['ML', 'MR'], ['BL', 'BR']]},
    'F': {'strokes': [['TL', 'BL'], ['TL', 'TR'], ['ML', 'MR']]},
    'G': {'strokes': [['TR', 'TL', 'BL', 'BR', 'MR'], ['MR', 'MC']]},
    'H': {'strokes': [['TL', 'BL'], ['TR', 'BR'], ['ML', 'MR']]},
    'I': {'strokes': [['TC', 'BC']]},
    'J': {'strokes': [['TR', 'BR', 'BC', 'BL']]},
    'K': {'strokes': [['TL', 'BL'], ['TR', 'ML'], ['ML', 'BR']]},
    'L': {'strokes': [['TL', 'BL'], ['BL', 'BR']]},
    'M': {'strokes': [['BL', 'TL', 'MC', 'TR', 'BR']]},
    'N': {'strokes': [['BL', 'TL', 'BR', 'TR']]},
    'O': {'strokes': [['TC', 'TL', 'BL', 'BC', 'BR', 'TR', 'TC']]},
    'P': {'strokes': [['TL', 'BL'], ['TL', 'TR', 'MR', 'ML']]},
    'Q': {'strokes': [['TC', 'TL', 'BL', 'BC', 'BR', 'TR', 'TC'], ['MC', 'BR']]},
    'R': {'strokes': [['TL', 'BL'], ['TL', 'TR', 'MR', 'ML'], ['ML', 'BR']]},
    'S': {'strokes': [['TR', 'TL', 'ML', 'MR', 'BR', 'BL']]},
    'T': {'strokes': [['TL', 'TR'], ['TC', 'BC']]},
    'U': {'strokes': [['TL', 'BL', 'BC', 'BR', 'TR']]},
    'V': {'strokes': [['TL', 'BC', 'TR']]},
    'W': {'strokes': [['TL', 'BL', 'MC', 'BR', 'TR']]},
    'X': {'strokes': [['TL', 'BR'], ['TR', 'BL']]},
    'Y': {'strokes': [['TL', 'MC'], ['TR', 'MC'], ['MC', 'BC']]},
    'Z': {'strokes': [['TL', 'TR', 'BL', 'BR']]},
}
"""dict: Letter templates defining stroke paths.

Each entry maps a character to a dict with:
    - 'strokes': List of stroke paths, where each path is a list of vertex names
    - 'no_morph': Optional list of stroke indices that should not be morphed
      (useful for crossbars that intentionally cross the letter shape)

Vertex names use a grid convention:
    - T/M/B = Top/Middle/Bottom row
    - L/C/R = Left/Center/Right column
    - Special names like TR_TOP, TR_BOT for letters with multiple bumps
"""

# Default vertex positions (relative to bounding box, 0-1)
VERTEX_POS = {
    'TL': (0.0, 0.0), 'TC': (0.5, 0.0), 'TR': (1.0, 0.0),
    'ML': (0.0, 0.5), 'MC': (0.5, 0.5), 'MR': (1.0, 0.5),
    'BL': (0.0, 1.0), 'BC': (0.5, 1.0), 'BR': (1.0, 1.0),
}
"""dict: Default vertex positions as (x, y) fractions of bounding box.

Standard 9-point grid positions used as starting points for vertex
detection. Actual vertex positions are refined by snapping to the
font outline.

Grid layout::

    TL --- TC --- TR
    |      |      |
    ML --- MC --- MR
    |      |      |
    BL --- BC --- BR
"""


# ---------------------------------------------------------------------------
# Vertex Finder Registry Pattern
# ---------------------------------------------------------------------------

class VertexFinder(ABC):
    """Base class for character-specific vertex detection.

    Each subclass implements vertex detection logic for one or more
    characters. The registry pattern allows character-specific handlers
    to be added without modifying the main find_vertices() function.

    Subclasses must implement the find() method.

    Example:
        >>> class VertexFinderA(VertexFinder):
        ...     def find(self, font_mask, bbox, outline_xy):
        ...         # Find apex, base corners, and crossbar vertices
        ...         return {'TC': (50, 10), 'BL': (10, 90), ...}
    """

    @abstractmethod
    def find(self, font_mask: np.ndarray, bbox: tuple,
             outline_xy: np.ndarray) -> dict[str, tuple[int, int]]:
        """Find vertices for this character.

        Args:
            font_mask: Boolean numpy array of font pixels.
            bbox: Tuple (cmin, rmin, cmax, rmax) of bounding box.
            outline_xy: Numpy array of (x, y) outline points.

        Returns:
            Dict mapping vertex names (e.g., 'TL', 'TC', 'BR') to
            (x, y) pixel coordinates.
        """
        pass


class VertexFinderA(VertexFinder):
    """Vertex finder for letter A."""

    def find(self, font_mask: np.ndarray, bbox: tuple,
             outline_xy: np.ndarray) -> dict[str, tuple[int, int]]:
        cmin, rmin, cmax, rmax = bbox
        h = rmax - rmin
        vertices = {}

        # TC = topmost point (apex)
        top_idx = np.argmin(outline_xy[:, 1])
        vertices['TC'] = (int(outline_xy[top_idx, 0]), int(outline_xy[top_idx, 1]))
        vertices['BL'] = _default_vertex_pos('BL', bbox, outline_xy)
        vertices['BR'] = _default_vertex_pos('BR', bbox, outline_xy)

        # Find crossbar height: middle of the gap region between legs
        gap_rows = []
        for test_y in range(rmin + h // 4, rmax - h // 8):
            row = font_mask[test_y, :]
            if not row.any():
                continue
            cols_filled = np.where(row)[0]
            diffs = np.diff(cols_filled)
            if np.max(diffs) > 3:
                gap_rows.append(test_y)
        best_y = gap_rows[len(gap_rows) // 2] if gap_rows else rmin + int(h * 0.6)

        # Find ML/MR by scanning rows around crossbar height
        scan_range = range(max(rmin, best_y - h // 8), min(rmax, best_y + h // 8 + 1))
        far_left_x, far_left_y = cmax, best_y
        far_right_x, far_right_y = cmin, best_y
        for scan_y in scan_range:
            rp = np.where(font_mask[scan_y, :])[0]
            if len(rp) == 0:
                continue
            if rp[0] < far_left_x:
                far_left_x, far_left_y = int(rp[0]), scan_y
            if rp[-1] > far_right_x:
                far_right_x, far_right_y = int(rp[-1]), scan_y

        vertices['ML'] = snap_to_outline(far_left_x, far_left_y, outline_xy, y_tolerance=max(5, h // 8))
        vertices['MR'] = snap_to_outline(far_right_x, far_right_y, outline_xy, y_tolerance=max(5, h // 8))
        return vertices


class VertexFinderB(VertexFinder):
    """Vertex finder for letter B."""

    def find(self, font_mask: np.ndarray, bbox: tuple,
             outline_xy: np.ndarray) -> dict[str, tuple[int, int]]:
        cmin, rmin, cmax, rmax = bbox
        vertices = {}

        vertices['TL'] = _default_vertex_pos('TL', bbox, outline_xy)
        vertices['BL'] = _default_vertex_pos('BL', bbox, outline_xy)

        waist_y, _ = _find_waist_height(font_mask, rmin, rmax)
        vertices['ML'] = _leftmost_in_range(outline_xy, waist_y - 3, waist_y + 3) or \
            _default_vertex_pos('ML', bbox, outline_xy)

        top_bump = _rightmost_in_range(outline_xy, rmin, waist_y)
        vertices['TR_TOP'] = top_bump if top_bump else _default_vertex_pos('TR', bbox, outline_xy)

        bot_bump = _rightmost_in_range(outline_xy, waist_y, rmax)
        vertices['TR_BOT'] = bot_bump if bot_bump else _default_vertex_pos('BR', bbox, outline_xy)
        return vertices


class VertexFinderD(VertexFinder):
    """Vertex finder for letter D."""

    def find(self, font_mask: np.ndarray, bbox: tuple,
             outline_xy: np.ndarray) -> dict[str, tuple[int, int]]:
        cmin, rmin, cmax, rmax = bbox
        h = rmax - rmin
        vertices = {}

        vertices['TL'] = _default_vertex_pos('TL', bbox, outline_xy)
        vertices['BL'] = _default_vertex_pos('BL', bbox, outline_xy)
        tr = _rightmost_in_range(outline_xy, rmin, rmin + h // 3)
        vertices['TR'] = tr if tr else _default_vertex_pos('TR', bbox, outline_xy)
        br = _rightmost_in_range(outline_xy, rmax - h // 3, rmax)
        vertices['BR'] = br if br else _default_vertex_pos('BR', bbox, outline_xy)
        return vertices


class VertexFinderCG(VertexFinder):
    """Vertex finder for letters C and G."""

    def __init__(self, char: str):
        self.char = char

    def find(self, font_mask: np.ndarray, bbox: tuple,
             outline_xy: np.ndarray) -> dict[str, tuple[int, int]]:
        cmin, rmin, cmax, rmax = bbox
        h = rmax - rmin
        vertices = {}

        tr = _rightmost_in_range(outline_xy, rmin, rmin + int(h * 0.3))
        vertices['TR'] = tr if tr else _default_vertex_pos('TR', bbox, outline_xy)
        br = _rightmost_in_range(outline_xy, rmax - int(h * 0.3), rmax)
        vertices['BR'] = br if br else _default_vertex_pos('BR', bbox, outline_xy)
        tl = _leftmost_in_range(outline_xy, rmin + int(h * 0.2), rmin + int(h * 0.5))
        vertices['TL'] = tl if tl else _default_vertex_pos('TL', bbox, outline_xy)
        bl = _leftmost_in_range(outline_xy, rmin + int(h * 0.5), rmax - int(h * 0.2))
        vertices['BL'] = bl if bl else _default_vertex_pos('BL', bbox, outline_xy)
        if self.char == 'G':
            vertices['MR'] = _default_vertex_pos('MR', bbox, outline_xy)
            vertices['MC'] = _default_vertex_pos('MC', bbox, outline_xy)
        return vertices


class VertexFinderEF(VertexFinder):
    """Vertex finder for letters E and F."""

    def __init__(self, char: str):
        self.char = char

    def find(self, font_mask: np.ndarray, bbox: tuple,
             outline_xy: np.ndarray) -> dict[str, tuple[int, int]]:
        cmin, rmin, cmax, rmax = bbox
        h = rmax - rmin
        mid_y = rmin + h // 2
        vertices = {}

        vertices['TL'] = _default_vertex_pos('TL', bbox, outline_xy)
        vertices['BL'] = _default_vertex_pos('BL', bbox, outline_xy)
        vertices['TR'] = _default_vertex_pos('TR', bbox, outline_xy)
        vertices['ML'] = snap_to_outline(cmin, mid_y, outline_xy)
        mr = _rightmost_in_range(outline_xy, mid_y - h // 8, mid_y + h // 8)
        vertices['MR'] = mr if mr else _default_vertex_pos('MR', bbox, outline_xy)
        if self.char == 'E':
            vertices['BR'] = _default_vertex_pos('BR', bbox, outline_xy)
        return vertices


class VertexFinderHK(VertexFinder):
    """Vertex finder for letters H and K."""

    def find(self, font_mask: np.ndarray, bbox: tuple,
             outline_xy: np.ndarray) -> dict[str, tuple[int, int]]:
        cmin, rmin, cmax, rmax = bbox
        h = rmax - rmin
        mid_y = rmin + h // 2
        vertices = {}

        vertices['TL'] = _default_vertex_pos('TL', bbox, outline_xy)
        vertices['BL'] = _default_vertex_pos('BL', bbox, outline_xy)
        vertices['TR'] = _default_vertex_pos('TR', bbox, outline_xy)
        vertices['BR'] = _default_vertex_pos('BR', bbox, outline_xy)
        vertices['ML'] = snap_to_outline(cmin, mid_y, outline_xy)
        vertices['MR'] = snap_to_outline(cmax, mid_y, outline_xy)
        return vertices


class VertexFinderP(VertexFinder):
    """Vertex finder for letter P."""

    def find(self, font_mask: np.ndarray, bbox: tuple,
             outline_xy: np.ndarray) -> dict[str, tuple[int, int]]:
        cmin, rmin, cmax, rmax = bbox
        h = rmax - rmin
        vertices = {}

        vertices['TL'] = _default_vertex_pos('TL', bbox, outline_xy)
        vertices['BL'] = _default_vertex_pos('BL', bbox, outline_xy)
        tr = _rightmost_in_range(outline_xy, rmin, rmin + h // 3)
        vertices['TR'] = tr if tr else _default_vertex_pos('TR', bbox, outline_xy)
        best_y, min_right = _find_waist_height(font_mask, rmin, rmax)
        vertices['ML'] = snap_to_outline(min_right, best_y, outline_xy)
        mr = _rightmost_in_range(outline_xy, rmin + h // 4, rmin + h // 2)
        vertices['MR'] = mr if mr else _default_vertex_pos('MR', bbox, outline_xy)
        return vertices


class VertexFinderR(VertexFinder):
    """Vertex finder for letter R."""

    def find(self, font_mask: np.ndarray, bbox: tuple,
             outline_xy: np.ndarray) -> dict[str, tuple[int, int]]:
        cmin, rmin, cmax, rmax = bbox
        h = rmax - rmin
        vertices = {}

        vertices['TL'] = _default_vertex_pos('TL', bbox, outline_xy)
        vertices['BL'] = _default_vertex_pos('BL', bbox, outline_xy)
        vertices['BR'] = _default_vertex_pos('BR', bbox, outline_xy)
        tr = _rightmost_in_range(outline_xy, rmin, rmin + h // 3)
        vertices['TR'] = tr if tr else _default_vertex_pos('TR', bbox, outline_xy)
        best_y, min_right = _find_waist_height(font_mask, rmin, rmax)
        vertices['ML'] = snap_to_outline(min_right, best_y, outline_xy)
        mr = _rightmost_in_range(outline_xy, rmin + h // 4, rmin + h // 2)
        vertices['MR'] = mr if mr else _default_vertex_pos('MR', bbox, outline_xy)
        return vertices


class DefaultVertexFinder(VertexFinder):
    """Fallback vertex finder for characters without specific handlers."""

    def __init__(self, char: str):
        self.char = char

    def find(self, font_mask: np.ndarray, bbox: tuple,
             outline_xy: np.ndarray) -> dict[str, tuple[int, int]]:
        """Default: compute standard positions and snap to outline."""
        vertices = {}
        for name in set(sum(TEMPLATES.get(self.char, {}).get('strokes', []), [])):
            if name in VERTEX_POS:
                vertices[name] = _default_vertex_pos(name, bbox, outline_xy)
        return vertices


# Registry mapping characters to their VertexFinder classes/instances
# Note: Some finders need character context (C/G, E/F), so we use factory functions
VERTEX_FINDERS: dict[str, VertexFinder | type[VertexFinder]] = {
    'A': VertexFinderA(),
    'B': VertexFinderB(),
    'D': VertexFinderD(),
    'C': VertexFinderCG('C'),
    'G': VertexFinderCG('G'),
    'E': VertexFinderEF('E'),
    'F': VertexFinderEF('F'),
    'H': VertexFinderHK(),
    'K': VertexFinderHK(),
    'P': VertexFinderP(),
    'R': VertexFinderR(),
}


def render_font_mask(font_path: str, char: str, font_size: int = 200,
                     canvas_size: int = 512) -> tuple[np.ndarray | None, tuple | None]:
    """Render a character and return binary mask plus bounding box.

    Creates a binary image of the character and computes its tight
    bounding box.

    Args:
        font_path: Path to the font file.
        char: Single character to render.
        font_size: Font size in points for rendering.
        canvas_size: Size of the square canvas in pixels.

    Returns:
        Tuple of (mask, bbox):
            - mask: Boolean numpy array where True = ink pixels
            - bbox: Tuple (cmin, rmin, cmax, rmax) of tight bounding box
        Returns (None, None) if character cannot be rendered.
    """
    from stroke_rendering import GlyphRenderer, RenderConfig

    try:
        config = RenderConfig(
            canvas_size=canvas_size,
            font_size=font_size,
            fill_threshold=1.0,  # No auto-scaling for template morphing
        )
        renderer = GlyphRenderer(font_path, font_size=font_size, config=config)
        return renderer.render_mask_with_bbox(char, canvas_size=canvas_size)
    except OSError:
        return None, None


def _find_waist_height(font_mask: np.ndarray, rmin: int, rmax: int) -> tuple[int, int]:
    """Find the waist height (narrowest rightward extent) in the middle region.

    Used for characters like B, P, R where the waist is where bumps meet.

    Args:
        font_mask: Boolean numpy array of font pixels.
        rmin: Top of bounding box.
        rmax: Bottom of bounding box.

    Returns:
        Tuple of (waist_y, min_right_extent).
    """
    h = rmax - rmin
    mid_y = rmin + h // 2
    waist_y = mid_y
    min_right = font_mask.shape[1] + 1

    for test_y in range(rmin + h // 3, rmin + 2 * h // 3):
        row = font_mask[test_y, :]
        if not row.any():
            continue
        right_extent = np.where(row)[0][-1]
        if right_extent < min_right:
            min_right = right_extent
            waist_y = test_y

    return waist_y, min_right


def get_outline(font_mask: np.ndarray) -> np.ndarray:
    """Get font outline pixels (edge between filled and empty).

    Extracts the 1-pixel-wide outline of the font by subtracting
    an eroded version from the original.

    Args:
        font_mask: Boolean numpy array of font pixels.

    Returns:
        Boolean numpy array where True = outline pixels.
    """
    eroded = ndimage.binary_erosion(font_mask)
    outline = font_mask & ~eroded
    return outline


def snap_to_outline(x: int, y: int, outline_points: np.ndarray,
                    y_tolerance: int | None = None) -> tuple[int, int]:
    """Snap a point to nearest outline pixel.

    Finds the closest outline point to the given coordinates,
    optionally restricting the search to a Y range.

    Args:
        x: X coordinate to snap.
        y: Y coordinate to snap.
        outline_points: Numpy array of shape (N, 2) with (x, y) points.
        y_tolerance: If set, only consider outline points within this
            Y distance from the target Y.

    Returns:
        Tuple (x, y) of the nearest outline point.
    """
    if len(outline_points) == 0:
        return x, y
    if y_tolerance is not None:
        mask = np.abs(outline_points[:, 1] - y) <= y_tolerance
        candidates = outline_points[mask]
        if len(candidates) == 0:
            candidates = outline_points  # fallback to all
    else:
        candidates = outline_points
    dists = (candidates[:, 0] - x)**2 + (candidates[:, 1] - y)**2
    idx = np.argmin(dists)
    return int(candidates[idx, 0]), int(candidates[idx, 1])


def _rightmost_in_range(outline_xy: np.ndarray, y_start: int, y_end: int):
    """Find rightmost outline point in a y range.

    Args:
        outline_xy: Numpy array of (x, y) outline points.
        y_start: Minimum y coordinate.
        y_end: Maximum y coordinate.

    Returns:
        Tuple (x, y) of rightmost point, or None if no points in range.
    """
    mask_range = (outline_xy[:, 1] >= y_start) & (outline_xy[:, 1] <= y_end)
    pts = outline_xy[mask_range]
    if len(pts) == 0:
        return None
    idx = np.argmax(pts[:, 0])
    return int(pts[idx, 0]), int(pts[idx, 1])


def _leftmost_in_range(outline_xy: np.ndarray, y_start: int, y_end: int):
    """Find leftmost outline point in a y range.

    Args:
        outline_xy: Numpy array of (x, y) outline points.
        y_start: Minimum y coordinate.
        y_end: Maximum y coordinate.

    Returns:
        Tuple (x, y) of leftmost point, or None if no points in range.
    """
    mask_range = (outline_xy[:, 1] >= y_start) & (outline_xy[:, 1] <= y_end)
    pts = outline_xy[mask_range]
    if len(pts) == 0:
        return None
    idx = np.argmin(pts[:, 0])
    return int(pts[idx, 0]), int(pts[idx, 1])


def _default_vertex_pos(name: str, bbox: tuple, outline_xy: np.ndarray):
    """Get default position from VERTEX_POS and snap to outline.

    Args:
        name: Vertex name (e.g., 'TL', 'BR').
        bbox: Tuple (cmin, rmin, cmax, rmax).
        outline_xy: Numpy array of (x, y) outline points.

    Returns:
        Tuple (x, y) snapped to nearest outline point.
    """
    cmin, rmin, cmax, rmax = bbox
    w = cmax - cmin
    h = rmax - rmin
    rx, ry = VERTEX_POS[name]
    x = cmin + int(rx * w)
    y = rmin + int(ry * h)
    return snap_to_outline(x, y, outline_xy)


def find_vertices(char: str, font_mask: np.ndarray, bbox: tuple) -> dict:
    """Find letter-specific vertex positions on the font outline.

    Uses the VERTEX_FINDERS registry to dispatch to character-specific
    handlers. If no handler is registered, falls back to DefaultVertexFinder.

    Args:
        char: The character being processed.
        font_mask: Boolean numpy array of font pixels.
        bbox: Tuple (cmin, rmin, cmax, rmax) of bounding box.

    Returns:
        Dict mapping vertex names (e.g., 'TL', 'TC', 'BR') to (x, y)
        pixel coordinates on the outline.

    Example:
        >>> vertices = find_vertices('A', mask, bbox)
        >>> print(vertices['TC'])  # Apex of the A
        (112, 45)

    Note:
        To add support for a new character, create a VertexFinder subclass
        and register it in VERTEX_FINDERS:

            class VertexFinderX(VertexFinder):
                def find(self, font_mask, bbox, outline_xy):
                    return {'TL': ..., 'BR': ...}

            VERTEX_FINDERS['X'] = VertexFinderX()
    """
    outline = get_outline(font_mask)
    outline_pts = np.argwhere(outline)  # (row, col)
    if len(outline_pts) == 0:
        return {}
    # Convert to (x, y) = (col, row)
    outline_xy = outline_pts[:, ::-1]

    # Look up finder in registry, fall back to default
    finder = VERTEX_FINDERS.get(char)
    if finder is None:
        finder = DefaultVertexFinder(char)

    return finder.find(font_mask, bbox, outline_xy)


def interpolate_stroke(vertices_list: list, n_points: int = 50) -> list:
    """Create smooth curve through vertex points.

    Generates evenly-spaced points along straight line segments
    connecting the vertices.

    Args:
        vertices_list: List of (x, y) vertex coordinates.
        n_points: Target total number of points to generate.

    Returns:
        Numpy array of shape (N, 2) with interpolated points.
    """
    if len(vertices_list) < 2:
        return np.array(vertices_list)

    # Create points along each segment
    all_points = []
    total_dist = 0
    for i in range(len(vertices_list) - 1):
        dx = vertices_list[i+1][0] - vertices_list[i][0]
        dy = vertices_list[i+1][1] - vertices_list[i][1]
        total_dist += np.sqrt(dx*dx + dy*dy)

    if total_dist == 0:
        return np.array(vertices_list)

    for i in range(len(vertices_list) - 1):
        p1 = vertices_list[i]
        p2 = vertices_list[i+1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        seg_dist = np.sqrt(dx*dx + dy*dy)
        seg_pts = max(2, int(n_points * seg_dist / total_dist))

        for j in range(seg_pts):
            t = j / seg_pts
            x = p1[0] + t * dx
            y = p1[1] + t * dy
            all_points.append((x, y))

    all_points.append(vertices_list[-1])
    return np.array(all_points, dtype=float)


def refine_stroke(points: list, font_mask: np.ndarray,
                  max_iterations: int = 100) -> list:
    """Iteratively morph stroke points to be inside the font.

    Adjusts points that fall outside the font shape by snapping them
    to the nearest font pixel, with smoothing to maintain stroke
    continuity.

    Args:
        points: Numpy array of shape (N, 2) with stroke points.
        font_mask: Boolean numpy array of font pixels.
        max_iterations: Maximum refinement iterations.

    Returns:
        Numpy array of shape (N, 2) with refined points, all
        guaranteed to be inside the font mask.
    """
    h, w = font_mask.shape
    font_ys, font_xs = np.where(font_mask)
    if len(font_xs) == 0:
        return points

    pts = points.copy()

    for _iteration in range(max_iterations):
        # Check which points are outside
        outside = []
        for i, (px, py) in enumerate(pts):
            ix, iy = int(round(px)), int(round(py))
            if ix < 0 or ix >= w or iy < 0 or iy >= h or not font_mask[iy, ix]:
                outside.append(i)

        if len(outside) == 0:
            break

        # Snap outside points to nearest font pixel
        for i in outside:
            px, py = pts[i]
            dists = (font_xs - px)**2 + (font_ys - py)**2
            nearest = np.argmin(dists)
            pts[i] = [font_xs[nearest], font_ys[nearest]]

        # Smooth (keep endpoints fixed)
        if len(pts) > 2:
            smoothed = pts.copy()
            for i in range(1, len(pts) - 1):
                smoothed[i] = 0.2 * pts[i-1] + 0.6 * pts[i] + 0.2 * pts[i+1]
            pts = smoothed

    # Final hard snap: force ALL points inside
    for i, (px, py) in enumerate(pts):
        ix, iy = int(round(px)), int(round(py))
        if ix < 0 or ix >= w or iy < 0 or iy >= h or not font_mask[iy, ix]:
            dists = (font_xs - px)**2 + (font_ys - py)**2
            nearest = np.argmin(dists)
            pts[i] = [font_xs[nearest], font_ys[nearest]]

    return pts


def morph_to_font(char: str, font_path: str, font_size: int = 200,
                  canvas_size: int = 512) -> tuple:
    """Full pipeline: render font, find vertices, generate and morph strokes.

    Main entry point for stroke generation. Renders the character,
    locates vertices on the outline, interpolates stroke paths, and
    morphs them to stay inside the font shape.

    Args:
        char: Single uppercase character to process.
        font_path: Path to the font file.
        font_size: Font size in points for rendering.
        canvas_size: Size of the square canvas in pixels.

    Returns:
        Tuple of (mask, vertices, strokes):
            - mask: Boolean numpy array of font pixels
            - vertices: Dict mapping vertex names to (x, y) coordinates
            - strokes: List of numpy arrays, each shape (N, 2) with stroke points
        Returns (None, None, None) if character is not supported or
        cannot be rendered.
    """
    if char not in TEMPLATES:
        return None, None, None

    mask, bbox = render_font_mask(font_path, char, font_size, canvas_size)
    if mask is None:
        return None, None, None

    vertices = find_vertices(char, mask, bbox)
    template = TEMPLATES[char]

    no_morph_indices = set(template.get('no_morph', []))

    strokes = []
    for si, stroke_def in enumerate(template['strokes']):
        # Get vertex positions for this stroke
        vert_positions = []
        for v_name in stroke_def:
            if v_name in vertices:
                vert_positions.append(vertices[v_name])
            elif v_name in VERTEX_POS:
                # Fallback to default
                cmin, rmin, cmax, rmax = bbox
                rx, ry = VERTEX_POS[v_name]
                w = cmax - cmin
                h = rmax - rmin
                vert_positions.append((cmin + int(rx * w), rmin + int(ry * h)))

        if len(vert_positions) < 2:
            continue

        # Interpolate straight lines between vertices
        raw_points = interpolate_stroke(vert_positions, n_points=60)

        if si in no_morph_indices:
            # Keep as straight line (e.g. A's crossbar crosses the diagonal legs)
            strokes.append(raw_points)
        else:
            # Morph to fit inside font
            refined = refine_stroke(raw_points, mask)
            strokes.append(refined)

    return mask, vertices, strokes


def visualize_letter(char: str, font_path: str, font_size: int = 200,
                     canvas_size: int = 512) -> np.ndarray | None:
    """Create visualization of template morph for a single letter.

    Renders the font in dark gray with colored stroke paths overlaid
    and white dots marking the detected vertices.

    Args:
        char: Single uppercase character to visualize.
        font_path: Path to the font file.
        font_size: Font size in points for rendering.
        canvas_size: Size of the square output image in pixels.

    Returns:
        RGB numpy array of shape (canvas_size, canvas_size, 3), or
        None if the character cannot be processed.
    """
    mask, vertices, strokes = morph_to_font(char, font_path, font_size, canvas_size)
    if mask is None:
        return None

    # Create RGB image
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    # Draw font in dark gray
    img[mask] = [60, 60, 60]

    # Draw strokes in distinct colors
    colors = [
        (255, 80, 80),    # red
        (80, 255, 80),    # green
        (80, 80, 255),    # blue
        (255, 255, 80),   # yellow
        (255, 80, 255),   # magenta
    ]

    for si, stroke in enumerate(strokes):
        color = colors[si % len(colors)]
        for i in range(len(stroke) - 1):
            x1, y1 = int(round(stroke[i][0])), int(round(stroke[i][1]))
            x2, y2 = int(round(stroke[i+1][0])), int(round(stroke[i+1][1]))
            # Draw line using PIL for anti-aliasing
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=3)
            img = np.array(pil_img)

    # Draw vertices as white dots
    if vertices:
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        for name, (vx, vy) in vertices.items():
            draw.ellipse([(vx-4, vy-4), (vx+4, vy+4)], fill=(255, 255, 255))
            draw.text((vx+6, vy-6), name, fill=(255, 255, 255))
        img = np.array(pil_img)

    return img


def visualize_alphabet(font_path: str, font_size: int = 200,
                       output_path: str = '/tmp/template_morph_AZ.png') -> str:
    """Create grid visualization of A-Z stroke templates.

    Generates a 6x5 grid showing all 26 uppercase letters with their
    detected vertices and morphed stroke paths.

    Args:
        font_path: Path to the font file.
        font_size: Font size in points for rendering.
        output_path: Path to save the output image.

    Returns:
        The output_path where the image was saved.
    """
    cols = 6
    rows = 5  # 26 letters, 6 per row = 5 rows (last has 2)
    cell = 300
    padding = 10

    total_w = cols * cell
    total_h = rows * cell
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        row = i // cols
        col = i % cols
        x_off = col * cell
        y_off = row * cell

        img = visualize_letter(char, font_path, font_size, canvas_size=cell - padding * 2)
        if img is not None:
            canvas[y_off+padding:y_off+padding+img.shape[0],
                   x_off+padding:x_off+padding+img.shape[1]] = img

        # Add letter label
        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)
        draw.text((x_off + 5, y_off + 5), char, fill=(200, 200, 200))
        canvas = np.array(pil_canvas)

    Image.fromarray(canvas).save(output_path)
    print(f"Saved to {output_path}")
    return output_path


if __name__ == '__main__':
    import sys
    font_path = sys.argv[1] if len(sys.argv) > 1 else "fonts/dafont/003 Engineer_'s Hand.ttf"
    font_size = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    output = sys.argv[3] if len(sys.argv) > 3 else '/tmp/template_morph_AZ.png'
    visualize_alphabet(font_path, font_size, output)
