"""Dataclasses and constants for stroke processing.

This module contains shared dataclasses used across the stroke processing modules.
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Set, Dict


# --- Segment tracing configuration ---
@dataclass
class SegmentConfig:
    """Configuration for how to trace a segment between waypoints."""
    direction: Optional[str] = None  # 'down', 'up', 'left', 'right' - initial direction bias
    straight: bool = False           # True = draw direct line, False = follow skeleton


@dataclass
class ParsedWaypoint:
    """A parsed waypoint with its position and type info."""
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
    """Analyzed skeleton data for stroke generation."""
    mask: np.ndarray
    info: Dict  # skeleton info dict
    segments: List[Dict]  # classified segments
    vertical_segments: List[Dict]
    bbox: Tuple[float, float, float, float]
    skel_list: List[Tuple[int, int]]
    skel_tree: object  # cKDTree
    glyph_rows: np.ndarray  # row indices of glyph pixels
    glyph_cols: np.ndarray  # column indices of glyph pixels


@dataclass
class ResolvedWaypoint:
    """A waypoint with its resolved pixel position."""
    position: Tuple[float, float]
    region: int
    is_curve: bool = False
    is_vertex: bool = False
    is_intersection: bool = False
    apex_extension: Optional[Tuple[str, Tuple[float, float]]] = None  # ('top'/'bottom', (x, y))


@dataclass
class VariantResult:
    """Result of evaluating one template variant."""
    strokes: Optional[List[List[List[float]]]]
    score: float
    variant_name: str


def parse_stroke_template(stroke_template: List) -> Tuple[List[ParsedWaypoint], List[SegmentConfig]]:
    """Parse a stroke template into waypoints and segment configurations.

    Args:
        stroke_template: Raw template like [9, 8, 7, 'straight', 4, 'i(5)', 6]

    Returns:
        Tuple of:
        - List of ParsedWaypoint (one per actual waypoint, hints excluded)
        - List of SegmentConfig (one per segment, len = len(waypoints) - 1)
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
            if item in DIRECTION_HINTS:
                pending_config.direction = item
            elif item == 'straight':
                pending_config.straight = True
            continue

        # Parse waypoint type
        wp = ParsedWaypoint(region=0)

        if isinstance(item, int):
            wp.region = item
        else:
            # Try v(n), c(n), i(n) patterns
            m = re.match(r'^v\((\d)\)$', str(item))
            if m:
                wp.region = int(m.group(1))
                wp.is_vertex = True
            else:
                m = re.match(r'^c\((\d)\)$', str(item))
                if m:
                    wp.region = int(m.group(1))
                    wp.is_curve = True
                else:
                    m = re.match(r'^i\((\d)\)$', str(item))
                    if m:
                        wp.region = int(m.group(1))
                        wp.is_intersection = True
                    else:
                        continue  # Unknown format, skip

        waypoints.append(wp)

        # Store the pending config for the segment ENDING at this waypoint
        # (First waypoint has no preceding segment)
        if len(waypoints) > 1:
            segment_configs.append(pending_config)
            pending_config = SegmentConfig()

    return waypoints, segment_configs
