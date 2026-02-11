"""Geometric value objects for stroke editing.

This module provides immutable geometric primitives and data structures
used throughout the stroke editing package. All classes are designed
to be value objects with clear semantics and convenient operations.

The module provides the following classes:
    Point: Immutable 2D point supporting vector arithmetic.
    BBox: Immutable bounding box with numpad region operations.
    Stroke: Mutable sequence of points representing a stroke path.
    Segment: Immutable skeleton segment with geometric metadata.

These classes form the foundation of the geometric operations in the
stroke editing system and are used extensively by the analysis and
optimization modules.

Example usage:
    Point operations::

        from stroke_lib.domain.geometry import Point

        p1 = Point(10, 20)
        p2 = Point(30, 40)

        # Vector arithmetic
        diff = p2 - p1  # Point(20, 20)
        scaled = p1 * 2  # Point(20, 40)

        # Distance and direction
        dist = p1.distance_to(p2)
        direction = (p2 - p1).normalized()

    Bounding box operations::

        from stroke_lib.domain.geometry import BBox, Point

        bbox = BBox(0, 0, 300, 300)

        # Properties
        print(bbox.center)  # Point(150, 150)
        print(bbox.width)   # 300

        # Numpad regions
        region_center = bbox.numpad_region_center(5)  # Center region
        is_in_region = bbox.point_in_region(Point(50, 50), 7)  # Top-left
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Point:
    """Immutable 2D point.

    Represents a point in 2D space with x and y coordinates. Supports
    vector arithmetic operations and common geometric calculations.
    Being frozen (immutable), Point objects can be used as dictionary
    keys and in sets.

    Attributes:
        x: X coordinate of the point.
        y: Y coordinate of the point.

    Example:
        >>> p1 = Point(10, 20)
        >>> p2 = Point(30, 40)
        >>> p1.distance_to(p2)
        28.284271247461902
        >>> (p2 - p1).normalized()
        Point(x=0.7071067811865475, y=0.7071067811865475)
    """
    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        """Euclidean distance to another point.

        Args:
            other: The point to calculate distance to.

        Returns:
            Euclidean distance between this point and the other point.
        """
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def __add__(self, other: Point) -> Point:
        """Add two points (vector addition).

        Args:
            other: Point to add.

        Returns:
            New Point representing the vector sum.
        """
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        """Subtract two points (vector subtraction).

        Args:
            other: Point to subtract.

        Returns:
            New Point representing the vector difference.
        """
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Point:
        """Multiply point by scalar.

        Args:
            scalar: Value to multiply coordinates by.

        Returns:
            New Point with scaled coordinates.
        """
        return Point(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> Point:
        """Divide point by scalar.

        Args:
            scalar: Value to divide coordinates by.

        Returns:
            New Point with divided coordinates.
        """
        return Point(self.x / scalar, self.y / scalar)

    def dot(self, other: Point) -> float:
        """Dot product treating points as vectors.

        Args:
            other: Point to compute dot product with.

        Returns:
            Scalar dot product value.
        """
        return self.x * other.x + self.y * other.y

    def length(self) -> float:
        """Length when treated as a vector from origin.

        Returns:
            Euclidean length of the vector from origin to this point.
        """
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalized(self) -> Point:
        """Unit vector in same direction.

        Returns:
            New Point representing a unit vector in the same direction.
            Returns Point(0, 0) if the length is nearly zero.
        """
        length = self.length()
        if length < 0.0001:
            return Point(0.0, 0.0)
        return self / length

    def to_tuple(self) -> tuple[float, float]:
        """Convert to tuple for compatibility.

        Returns:
            Tuple of (x, y) coordinates.
        """
        return (self.x, self.y)

    def to_int_tuple(self) -> tuple[int, int]:
        """Convert to integer tuple for pixel operations.

        Returns:
            Tuple of (x, y) coordinates rounded to integers.
        """
        return (int(round(self.x)), int(round(self.y)))

    def to_list(self) -> list[float]:
        """Convert to list for JSON serialization.

        Returns:
            List of [x, y] coordinates as floats.
        """
        return [float(self.x), float(self.y)]

    @classmethod
    def from_tuple(cls, t: tuple[float, float]) -> Point:
        """Create from tuple.

        Args:
            t: Tuple of (x, y) coordinates.

        Returns:
            New Point with the given coordinates.
        """
        return cls(t[0], t[1])

    @classmethod
    def from_list(cls, lst: list[float]) -> Point:
        """Create from list.

        Args:
            lst: List of [x, y] coordinates.

        Returns:
            New Point with the given coordinates.
        """
        return cls(lst[0], lst[1])


@dataclass(frozen=True)
class BBox:
    """Immutable bounding box.

    Represents an axis-aligned bounding box defined by minimum and
    maximum x and y coordinates. Provides properties for dimensions
    and corners, as well as methods for containment testing and
    numpad region operations.

    The numpad region system divides the bounding box into a 3x3 grid
    numbered like a phone numpad:
        7 8 9  (top row)
        4 5 6  (middle row)
        1 2 3  (bottom row)

    Attributes:
        x_min: Minimum x coordinate (left edge).
        y_min: Minimum y coordinate (top edge).
        x_max: Maximum x coordinate (right edge).
        y_max: Maximum y coordinate (bottom edge).

    Example:
        >>> bbox = BBox(0, 0, 300, 300)
        >>> bbox.width
        300
        >>> bbox.center
        Point(x=150.0, y=150.0)
        >>> bbox.numpad_region_center(5)  # Center region
        Point(x=150.0, y=150.0)
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        """Width of the bounding box.

        Returns:
            Horizontal span from x_min to x_max.
        """
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Height of the bounding box.

        Returns:
            Vertical span from y_min to y_max.
        """
        return self.y_max - self.y_min

    @property
    def center(self) -> Point:
        """Center point of the bounding box.

        Returns:
            Point at the geometric center.
        """
        return Point(
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2
        )

    @property
    def top_left(self) -> Point:
        """Top-left corner of the bounding box.

        Returns:
            Point at (x_min, y_min).
        """
        return Point(self.x_min, self.y_min)

    @property
    def bottom_right(self) -> Point:
        """Bottom-right corner of the bounding box.

        Returns:
            Point at (x_max, y_max).
        """
        return Point(self.x_max, self.y_max)

    def contains(self, point: Point) -> bool:
        """Check if point is inside bounding box.

        Args:
            point: Point to test for containment.

        Returns:
            True if the point is inside or on the boundary of the box.
        """
        return (self.x_min <= point.x <= self.x_max and
                self.y_min <= point.y <= self.y_max)

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Convert to tuple for compatibility.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max).
        """
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @classmethod
    def from_tuple(cls, t: tuple[float, float, float, float]) -> BBox:
        """Create from tuple.

        Args:
            t: Tuple of (x_min, y_min, x_max, y_max).

        Returns:
            New BBox with the given coordinates.
        """
        return cls(t[0], t[1], t[2], t[3])

    @classmethod
    def from_points(cls, points: list[Point]) -> BBox:
        """Create bounding box containing all points.

        Args:
            points: List of points to compute bounds for.

        Returns:
            Smallest BBox containing all given points.
            Returns BBox(0, 0, 0, 0) if the list is empty.
        """
        if not points:
            return cls(0, 0, 0, 0)
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        return cls(min(xs), min(ys), max(xs), max(ys))

    def numpad_region_center(self, region: int) -> Point:
        """Get center of numpad region (1-9) within this bbox.

        The numpad layout divides the bounding box into a 3x3 grid:
            7 8 9  (top row)
            4 5 6  (middle row)
            1 2 3  (bottom row)

        Args:
            region: Region number 1-9 following numpad layout.

        Returns:
            Point at the center of the specified region.
        """
        # Map region to fractional position
        col = (region - 1) % 3  # 0, 1, 2
        row = 2 - (region - 1) // 3  # 2, 1, 0 (inverted for y-down)

        frac_x = (col + 0.5) / 3
        frac_y = (row + 0.5) / 3

        return Point(
            self.x_min + frac_x * self.width,
            self.y_min + frac_y * self.height
        )

    def point_in_region(self, point: Point, region: int) -> bool:
        """Check if point falls within a numpad region (1-9).

        Args:
            point: Point to test.
            region: Region number 1-9 following numpad layout.

        Returns:
            True if the point is within the specified region.
        """
        col = (region - 1) % 3
        row = 2 - (region - 1) // 3

        x_start = self.x_min + col * self.width / 3
        x_end = self.x_min + (col + 1) * self.width / 3
        y_start = self.y_min + row * self.height / 3
        y_end = self.y_min + (row + 1) * self.height / 3

        return x_start <= point.x < x_end and y_start <= point.y < y_end


@dataclass
class Stroke:
    """A stroke as a sequence of points.

    Represents a continuous stroke path as an ordered list of Point
    objects. Provides methods for accessing endpoints, computing
    geometric properties, and serialization.

    Attributes:
        points: List of Point objects defining the stroke path.

    Example:
        >>> stroke = Stroke([Point(0, 0), Point(50, 50), Point(100, 100)])
        >>> len(stroke)
        3
        >>> stroke.start
        Point(x=0, y=0)
        >>> stroke.length()
        141.4213562373095
    """
    points: list[Point] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of points in the stroke.

        Returns:
            Number of points in the stroke.
        """
        return len(self.points)

    def __iter__(self) -> Iterator[Point]:
        """Iterate over points in the stroke.

        Returns:
            Iterator over Point objects.
        """
        return iter(self.points)

    def __getitem__(self, idx) -> Point:
        """Get point at index.

        Args:
            idx: Index or slice to access.

        Returns:
            Point at the specified index.
        """
        return self.points[idx]

    @property
    def start(self) -> Point:
        """First point of stroke.

        Returns:
            First Point in the stroke, or Point(0, 0) if empty.
        """
        return self.points[0] if self.points else Point(0, 0)

    @property
    def end(self) -> Point:
        """Last point of stroke.

        Returns:
            Last Point in the stroke, or Point(0, 0) if empty.
        """
        return self.points[-1] if self.points else Point(0, 0)

    @property
    def bbox(self) -> BBox:
        """Bounding box of stroke.

        Returns:
            BBox containing all points in the stroke.
        """
        return BBox.from_points(self.points)

    def length(self) -> float:
        """Total arc length of stroke.

        Computes the sum of distances between consecutive points.

        Returns:
            Total path length of the stroke.
        """
        total = 0.0
        for i in range(1, len(self.points)):
            total += self.points[i].distance_to(self.points[i - 1])
        return total

    def direction_at_end(self, from_end: bool = True, n_points: int = 8) -> Point:
        """Get direction vector at one end of stroke.

        Computes a direction vector based on the last n_points at the
        specified end of the stroke.

        Args:
            from_end: If True, compute direction at the end of the stroke.
                If False, compute direction at the start.
            n_points: Number of points to use for direction calculation.
                Default is 8.

        Returns:
            Normalized direction vector as a Point, or Point(0, 0) if
            the stroke has fewer than 2 points.
        """
        if len(self.points) < 2:
            return Point(0, 0)

        if from_end:
            pts = self.points[-min(n_points, len(self.points)):]
        else:
            pts = self.points[:min(n_points, len(self.points))][::-1]

        dx = pts[-1].x - pts[0].x
        dy = pts[-1].y - pts[0].y
        return Point(dx, dy).normalized()

    def reversed(self) -> Stroke:
        """Return stroke with reversed point order.

        Returns:
            New Stroke with points in reverse order.
        """
        return Stroke(list(reversed(self.points)))

    def to_list(self) -> list[list[float]]:
        """Convert to nested list for JSON serialization.

        Returns:
            List of [x, y] coordinate lists.
        """
        return [p.to_list() for p in self.points]

    @classmethod
    def from_list(cls, lst: list[list[float]]) -> Stroke:
        """Create from nested list.

        Args:
            lst: List of [x, y] coordinate lists.

        Returns:
            New Stroke with points from the list.
        """
        return cls([Point.from_list(p) for p in lst])

    @classmethod
    def from_tuples(cls, tuples: list[tuple[float, float]]) -> Stroke:
        """Create from list of tuples.

        Args:
            tuples: List of (x, y) coordinate tuples.

        Returns:
            New Stroke with points from the tuples.
        """
        return cls([Point.from_tuple(t) for t in tuples])


@dataclass(frozen=True)
class Segment:
    """A skeleton segment with metadata.

    Represents a segment of the skeleton graph with computed geometric
    properties. Segments connect endpoints or junction pixels and store
    information about their orientation and connectivity.

    Attributes:
        start: Starting point of the segment.
        end: Ending point of the segment.
        angle: Direction angle in degrees. 0 = horizontal right,
            90 = vertical down, -90 = vertical up.
        length: Euclidean distance from start to end.
        start_junction: Index of junction cluster at start, or -1 if
            the start is an endpoint (not a junction).
        end_junction: Index of junction cluster at end, or -1 if
            the end is an endpoint.

    Example:
        >>> seg = Segment(Point(0, 0), Point(0, 100), 90.0, 100.0)
        >>> seg.is_vertical
        True
        >>> seg.direction
        Point(x=0.0, y=1.0)
    """
    start: Point
    end: Point
    angle: float  # degrees, 0 = horizontal, 90 = vertical
    length: float
    start_junction: int = -1  # junction cluster index, -1 if endpoint
    end_junction: int = -1

    @property
    def is_vertical(self) -> bool:
        """Check if segment is approximately vertical (60-120 degrees).

        Returns:
            True if the absolute angle is between 60 and 120 degrees.
        """
        return 60 <= abs(self.angle) <= 120

    @property
    def is_horizontal(self) -> bool:
        """Check if segment is approximately horizontal.

        Returns:
            True if the absolute angle is less than 30 degrees or
            greater than 150 degrees.
        """
        return abs(self.angle) <= 30 or abs(self.angle) >= 150

    @property
    def direction(self) -> Point:
        """Unit direction vector from start to end.

        Returns:
            Normalized Point representing the direction from start to end.
        """
        return (self.end - self.start).normalized()

    @classmethod
    def from_dict(cls, d: dict) -> Segment:
        """Create from dictionary (for compatibility).

        Args:
            d: Dictionary with 'start', 'end', 'angle', 'length', and
                optionally 'start_junction' and 'end_junction' keys.

        Returns:
            New Segment with properties from the dictionary.
        """
        return cls(
            start=Point.from_tuple(d['start']) if isinstance(d['start'], tuple) else Point(d['start'][0], d['start'][1]),
            end=Point.from_tuple(d['end']) if isinstance(d['end'], tuple) else Point(d['end'][0], d['end'][1]),
            angle=d['angle'],
            length=d['length'],
            start_junction=d.get('start_junction', -1),
            end_junction=d.get('end_junction', -1),
        )
