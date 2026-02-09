"""Numpad-based stroke templates."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import re

from ..domain.geometry import Point, BBox


# Numpad position mapping (region 1-9 to fractional x,y coordinates)
# Layout:
# 7 8 9
# 4 5 6
# 1 2 3
NUMPAD_POS: Dict[int, Tuple[float, float]] = {
    1: (0.167, 0.833),  # bottom-left
    2: (0.500, 0.833),  # bottom-center
    3: (0.833, 0.833),  # bottom-right
    4: (0.167, 0.500),  # middle-left
    5: (0.500, 0.500),  # center
    6: (0.833, 0.500),  # middle-right
    7: (0.167, 0.167),  # top-left
    8: (0.500, 0.167),  # top-center
    9: (0.833, 0.167),  # top-right
}

# Direction hints for path tracing
DIRECTION_HINTS = {'up', 'down', 'left', 'right'}
STYLE_HINTS = {'straight'}
ALL_HINTS = DIRECTION_HINTS | STYLE_HINTS


@dataclass
class WaypointInfo:
    """Parsed waypoint information."""
    region: int
    is_vertex: bool = False
    is_curve: bool = False
    is_intersection: bool = False

    @property
    def is_terminal(self) -> bool:
        """True if this is a plain terminal (not vertex/curve/intersection)."""
        return not (self.is_vertex or self.is_curve or self.is_intersection)


@dataclass
class NumpadTemplate:
    """A stroke template using numpad notation.

    Waypoints can be:
    - Integer (1-9): Terminal position in numpad region
    - 'v(n)': Vertex at region n (sharp corner)
    - 'c(n)': Curve apex at region n
    - 'i(n)': Intersection point at region n
    - Direction hints: 'up', 'down', 'left', 'right'
    - Style hints: 'straight'
    """
    name: str
    strokes: List[List[Union[int, str, Tuple]]]

    def parse_waypoint(self, wp: Union[int, str, Tuple]) -> Optional[WaypointInfo]:
        """Parse a waypoint into structured info."""
        # Handle tuple (legacy format)
        wp_val = wp[0] if isinstance(wp, tuple) else wp

        # Skip hints
        if isinstance(wp_val, str) and wp_val in ALL_HINTS:
            return None

        # Plain integer
        if isinstance(wp_val, int):
            return WaypointInfo(region=wp_val)

        # Vertex v(n)
        m = re.match(r'^v\((\d)\)$', str(wp_val))
        if m:
            return WaypointInfo(region=int(m.group(1)), is_vertex=True)

        # Curve c(n)
        m = re.match(r'^c\((\d)\)$', str(wp_val))
        if m:
            return WaypointInfo(region=int(m.group(1)), is_curve=True)

        # Intersection i(n)
        m = re.match(r'^i\((\d)\)$', str(wp_val))
        if m:
            return WaypointInfo(region=int(m.group(1)), is_intersection=True)

        return None

    def get_waypoints(self, stroke_index: int) -> List[WaypointInfo]:
        """Get parsed waypoints for a stroke."""
        if stroke_index >= len(self.strokes):
            return []

        waypoints = []
        for wp in self.strokes[stroke_index]:
            info = self.parse_waypoint(wp)
            if info:
                waypoints.append(info)
        return waypoints

    def get_direction_hint(self, stroke_index: int, waypoint_index: int) -> Optional[str]:
        """Get direction hint following a waypoint."""
        if stroke_index >= len(self.strokes):
            return None

        stroke = self.strokes[stroke_index]
        # Count actual waypoints to find position
        wp_count = 0
        for i, wp in enumerate(stroke):
            wp_val = wp[0] if isinstance(wp, tuple) else wp
            if isinstance(wp_val, str) and wp_val in ALL_HINTS:
                continue

            if wp_count == waypoint_index:
                # Look ahead for hints
                for j in range(i + 1, len(stroke)):
                    next_wp = stroke[j]
                    next_val = next_wp[0] if isinstance(next_wp, tuple) else next_wp
                    if isinstance(next_val, str) and next_val in DIRECTION_HINTS:
                        return next_val
                    elif isinstance(next_val, str) and next_val not in STYLE_HINTS:
                        break
                    elif not isinstance(next_val, str):
                        break
                return None

            wp_count += 1

        return None

    def is_straight_segment(self, stroke_index: int, waypoint_index: int) -> bool:
        """Check if segment after waypoint has 'straight' hint."""
        if stroke_index >= len(self.strokes):
            return False

        stroke = self.strokes[stroke_index]
        wp_count = 0
        for i, wp in enumerate(stroke):
            wp_val = wp[0] if isinstance(wp, tuple) else wp
            if isinstance(wp_val, str) and wp_val in ALL_HINTS:
                continue

            if wp_count == waypoint_index:
                for j in range(i + 1, len(stroke)):
                    next_wp = stroke[j]
                    next_val = next_wp[0] if isinstance(next_wp, tuple) else next_wp
                    if next_val == 'straight':
                        return True
                    elif not isinstance(next_val, str) or next_val not in STYLE_HINTS:
                        break
                return False

            wp_count += 1

        return False

    def to_pixel_positions(self, bbox: BBox) -> List[List[Point]]:
        """Convert template to pixel positions within bounding box."""
        result = []
        for stroke in self.strokes:
            points = []
            for wp in stroke:
                info = self.parse_waypoint(wp)
                if info:
                    pos = bbox.numpad_region_center(info.region)
                    points.append(pos)
            if points:
                result.append(points)
        return result


def extract_region(wp: Union[int, str, Tuple]) -> Optional[int]:
    """Extract region number from a waypoint (for compatibility)."""
    if isinstance(wp, tuple):
        return wp[0] if isinstance(wp[0], int) else None
    return wp if isinstance(wp, int) else None


def is_vertical_stroke(stroke_template: List) -> bool:
    """Check if stroke template represents a vertical line."""
    if len(stroke_template) != 2:
        return False
    r1 = extract_region(stroke_template[0])
    r2 = extract_region(stroke_template[1])
    if r1 is None or r2 is None:
        return False
    # Same column: (7,4,1), (8,5,2), (9,6,3)
    return (r1 - 1) % 3 == (r2 - 1) % 3
