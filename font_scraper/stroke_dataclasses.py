"""Dataclasses and constants for stroke processing.

This module provides the core data structures and configuration classes used
throughout the stroke processing pipeline. It defines dataclasses for:

- Segment tracing configuration (direction hints, rendering styles)
- Waypoint parsing and representation (regions, curve types, intersections)
- Skeleton analysis results for stroke generation
- Resolved waypoints with computed pixel positions
- Variant evaluation results for template matching

The module also contains constants that control path tracing behavior, such as
direction bias distances and intersection detection thresholds.

Typical usage example:

    from stroke_dataclasses import parse_stroke_template, SegmentConfig

    # Parse a stroke template into waypoints and configs
    template = [9, 8, 'straight', 7, 'v(4)', 1]
    waypoints, configs = parse_stroke_template(template)

    # Access individual waypoint properties
    for wp in waypoints:
        print(f"Region {wp.region}, is_vertex={wp.is_vertex}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from scipy.spatial import cKDTree


# --- Segment tracing configuration ---
@dataclass
class SegmentConfig:
    """Configuration for how to trace a segment between waypoints.

    This dataclass specifies rendering hints for a single segment of a stroke,
    controlling the initial tracing direction and whether to use skeleton-following
    or direct line drawing.

    Attributes:
        direction: Initial direction bias for segment tracing. Valid values are
            'down', 'up', 'left', 'right', or None for no bias. The direction
            hint only affects the first DIRECTION_BIAS_PIXELS pixels of tracing.
        straight: If True, draw a direct line between waypoints instead of
            following the skeleton path. Defaults to False.
    """
    direction: str | None = None  # 'down', 'up', 'left', 'right' - initial direction bias
    straight: bool = False           # True = draw direct line, False = follow skeleton


@dataclass
class ParsedWaypoint:
    """A parsed waypoint with its position and type info.

    Represents a waypoint extracted from a stroke template, containing the
    numpad region identifier and flags indicating special waypoint types
    (curves, vertices, or intersections).

    The numpad region uses a 3x3 grid layout matching a numeric keypad:
        7 | 8 | 9
        ---------
        4 | 5 | 6
        ---------
        1 | 2 | 3

    Attributes:
        region: Numpad region identifier (1-9) indicating the waypoint's
            approximate position within the glyph bounding box.
        is_curve: If True, this waypoint represents a smooth curve point,
            parsed from 'c(n)' notation in the template.
        is_vertex: If True, this waypoint represents a sharp vertex,
            parsed from 'v(n)' notation in the template.
        is_intersection: If True, this waypoint represents a self-crossing
            point, parsed from 'i(n)' notation in the template.
    """
    region: int                      # Numpad region 1-9
    is_curve: bool = False           # c(n) - smooth curve
    is_vertex: bool = False          # v(n) - sharp vertex
    is_intersection: bool = False    # i(n) - self-crossing point


# Constants for path tracing
DIRECTION_BIAS_PIXELS = 15    # Only apply direction bias for first N pixels
NEAR_ENDPOINT_DIST = 5        # Distance to consider "near" start/end for avoid_pixels
ARRIVAL_BRANCH_SIZE = 3       # Pixels to track as arrival branch at intersections

# Valid hint strings
DIRECTION_HINTS = {'down', 'up', 'left', 'right'}
STYLE_HINTS = {'straight'}
ALL_HINTS = DIRECTION_HINTS | STYLE_HINTS


# --- Pipeline data classes for minimal_strokes_from_skeleton ---

@dataclass
class SkeletonAnalysis:
    """Analyzed skeleton data for stroke generation.

    Contains the complete results of skeleton analysis for a glyph, including
    the binary mask, classified segments, spatial indices, and bounding box
    information needed for stroke generation.

    This dataclass serves as the primary data container passed through the
    stroke generation pipeline after initial skeleton extraction and analysis.

    Attributes:
        mask: Binary numpy array representing the glyph mask, where non-zero
            values indicate glyph pixels.
        info: Dictionary containing skeleton metadata and statistics from
            the skeletonization process.
        segments: List of dictionaries, each describing a classified segment
            of the skeleton with its properties and endpoints.
        vertical_segments: Subset of segments that have been classified as
            primarily vertical strokes.
        bbox: Bounding box of the glyph as (min_x, min_y, max_x, max_y).
        skel_list: List of (row, col) tuples representing all skeleton pixel
            coordinates.
        skel_tree: A scipy.spatial.cKDTree built from skel_list for efficient
            nearest-neighbor queries during path tracing.
        glyph_rows: Numpy array of row indices for all glyph (non-background)
            pixels.
        glyph_cols: Numpy array of column indices for all glyph (non-background)
            pixels, corresponding to glyph_rows.
    """
    mask: np.ndarray
    info: dict[str, Any]  # skeleton info dict
    segments: list[dict[str, Any]]  # classified segments
    vertical_segments: list[dict[str, Any]]
    bbox: tuple[float, float, float, float]
    skel_list: list[tuple[int, int]]
    skel_tree: cKDTree
    glyph_rows: np.ndarray  # row indices of glyph pixels
    glyph_cols: np.ndarray  # column indices of glyph pixels


@dataclass
class ResolvedWaypoint:
    """A waypoint with its resolved pixel position.

    Represents a waypoint after its abstract region identifier has been
    resolved to concrete pixel coordinates. This is the result of mapping
    a ParsedWaypoint to actual skeleton positions.

    Attributes:
        position: The (x, y) pixel coordinates of the resolved waypoint
            position on the skeleton.
        region: The original numpad region (1-9) from the parsed waypoint.
        is_curve: If True, this waypoint represents a smooth curve point.
        is_vertex: If True, this waypoint represents a sharp vertex.
        is_intersection: If True, this waypoint represents a self-crossing
            point in the stroke path.
        apex_extension: Optional tuple containing apex extension information
            for waypoints at stroke extremities. Format is ('top' or 'bottom',
            (x, y)) where the string indicates the extension direction and
            the tuple contains the extension endpoint coordinates.
    """
    position: tuple[float, float]
    region: int
    is_curve: bool = False
    is_vertex: bool = False
    is_intersection: bool = False
    apex_extension: tuple[str, tuple[float, float]] | None = None  # ('top'/'bottom', (x, y))


@dataclass
class VariantResult:
    """Result of evaluating one template variant.

    Contains the output of applying a single template variant to a glyph,
    including the generated strokes, a quality score, and the variant
    identifier for debugging and selection purposes.

    Attributes:
        strokes: List of strokes, where each stroke is a list of points,
            and each point is a list of float coordinates [x, y]. Set to
            None if the variant failed to produce valid strokes.
        score: Numeric quality score for this variant result. Lower scores
            typically indicate better matches. Used to select the best
            variant when multiple templates are evaluated.
        variant_name: String identifier for the template variant, used for
            logging and debugging which variant produced this result.
    """
    strokes: list[list[list[float]]] | None
    score: float
    variant_name: str


# Regex patterns for waypoint parsing
_VERTEX_PATTERN = re.compile(r'^v\((\d)\)$')
_CURVE_PATTERN = re.compile(r'^c\((\d)\)$')
_INTERSECTION_PATTERN = re.compile(r'^i\((\d)\)$')


def _apply_hint_to_config(hint: str, config: SegmentConfig) -> None:
    """Apply a hint string to a segment configuration.

    Args:
        hint: Hint string ('down', 'up', 'left', 'right', or 'straight').
        config: SegmentConfig to modify in place.
    """
    if hint in DIRECTION_HINTS:
        config.direction = hint
    elif hint == 'straight':
        config.straight = True


def _parse_waypoint_item(item) -> ParsedWaypoint | None:
    """Parse a single template item into a ParsedWaypoint.

    Args:
        item: Integer region or string notation (v(n), c(n), i(n)).

    Returns:
        ParsedWaypoint if the item is a valid waypoint, None otherwise.
    """
    if isinstance(item, int):
        return ParsedWaypoint(region=item)

    item_str = str(item)

    # Try vertex pattern
    m = _VERTEX_PATTERN.match(item_str)
    if m:
        return ParsedWaypoint(region=int(m.group(1)), is_vertex=True)

    # Try curve pattern
    m = _CURVE_PATTERN.match(item_str)
    if m:
        return ParsedWaypoint(region=int(m.group(1)), is_curve=True)

    # Try intersection pattern
    m = _INTERSECTION_PATTERN.match(item_str)
    if m:
        return ParsedWaypoint(region=int(m.group(1)), is_intersection=True)

    return None


def parse_stroke_template(stroke_template: list) -> tuple[list[ParsedWaypoint], list[SegmentConfig]]:
    """Parse a stroke template into waypoints and segment configurations.

    Converts a raw stroke template containing mixed waypoint identifiers and
    hint strings into structured ParsedWaypoint objects and SegmentConfig
    objects. The function handles multiple waypoint notations:

    - Integer values (1-9): Direct numpad region references
    - 'v(n)': Vertex waypoint at region n (sharp corner)
    - 'c(n)': Curve waypoint at region n (smooth transition)
    - 'i(n)': Intersection waypoint at region n (self-crossing)

    Hint strings ('down', 'up', 'left', 'right', 'straight') are accumulated
    and applied to the segment ending at the next waypoint.

    Args:
        stroke_template: Raw template list containing waypoint identifiers
            (integers or special notation strings) and optional hint strings.
            Example: [9, 8, 7, 'straight', 4, 'i(5)', 6]

    Returns:
        A tuple containing two elements:
            - waypoints: List of ParsedWaypoint objects, one per actual
                waypoint in the template (hint strings are excluded).
            - segment_configs: List of SegmentConfig objects, one per segment
                between consecutive waypoints. Length is len(waypoints) - 1.

    Example:
        >>> template = [9, 'down', 8, 'straight', 7, 'v(4)', 1]
        >>> waypoints, configs = parse_stroke_template(template)
        >>> len(waypoints)
        5
        >>> len(configs)
        4
        >>> configs[0].direction
        'down'
        >>> configs[1].straight
        True
        >>> waypoints[3].is_vertex
        True
    """
    waypoints = []
    segment_configs = []
    pending_config = SegmentConfig()

    for item in stroke_template:
        # Handle tuples (legacy format)
        if isinstance(item, tuple):
            item = item[0]

        # Check if it's a hint
        if isinstance(item, str) and item in ALL_HINTS:
            _apply_hint_to_config(item, pending_config)
            continue

        # Try to parse as waypoint
        wp = _parse_waypoint_item(item)
        if wp is None:
            continue  # Unknown format, skip

        waypoints.append(wp)

        # Store the pending config for the segment ENDING at this waypoint
        # (First waypoint has no preceding segment)
        if len(waypoints) > 1:
            segment_configs.append(pending_config)
            pending_config = SegmentConfig()

    return waypoints, segment_configs
