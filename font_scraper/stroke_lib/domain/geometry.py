"""Geometric value objects for stroke editing."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Iterator
import math


@dataclass(frozen=True)
class Point:
    """Immutable 2D point."""
    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        """Euclidean distance to another point."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Point:
        return Point(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> Point:
        return Point(self.x / scalar, self.y / scalar)

    def dot(self, other: Point) -> float:
        """Dot product treating points as vectors."""
        return self.x * other.x + self.y * other.y

    def length(self) -> float:
        """Length when treated as a vector from origin."""
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalized(self) -> Point:
        """Unit vector in same direction."""
        length = self.length()
        if length < 0.0001:
            return Point(0.0, 0.0)
        return self / length

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple for compatibility."""
        return (self.x, self.y)

    def to_int_tuple(self) -> Tuple[int, int]:
        """Convert to integer tuple for pixel operations."""
        return (int(round(self.x)), int(round(self.y)))

    def to_list(self) -> List[float]:
        """Convert to list for JSON serialization."""
        return [float(self.x), float(self.y)]

    @classmethod
    def from_tuple(cls, t: Tuple[float, float]) -> Point:
        """Create from tuple."""
        return cls(t[0], t[1])

    @classmethod
    def from_list(cls, lst: List[float]) -> Point:
        """Create from list."""
        return cls(lst[0], lst[1])


@dataclass(frozen=True)
class BBox:
    """Immutable bounding box."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> Point:
        return Point(
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2
        )

    @property
    def top_left(self) -> Point:
        return Point(self.x_min, self.y_min)

    @property
    def bottom_right(self) -> Point:
        return Point(self.x_max, self.y_max)

    def contains(self, point: Point) -> bool:
        """Check if point is inside bounding box."""
        return (self.x_min <= point.x <= self.x_max and
                self.y_min <= point.y <= self.y_max)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Convert to tuple for compatibility."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float, float]) -> BBox:
        """Create from tuple."""
        return cls(t[0], t[1], t[2], t[3])

    @classmethod
    def from_points(cls, points: List[Point]) -> BBox:
        """Create bounding box containing all points."""
        if not points:
            return cls(0, 0, 0, 0)
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        return cls(min(xs), min(ys), max(xs), max(ys))

    def numpad_region_center(self, region: int) -> Point:
        """Get center of numpad region (1-9) within this bbox.

        Numpad layout:
        7 8 9  (top)
        4 5 6  (middle)
        1 2 3  (bottom)
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
        """Check if point falls within a numpad region (1-9)."""
        col = (region - 1) % 3
        row = 2 - (region - 1) // 3

        x_start = self.x_min + col * self.width / 3
        x_end = self.x_min + (col + 1) * self.width / 3
        y_start = self.y_min + row * self.height / 3
        y_end = self.y_min + (row + 1) * self.height / 3

        return x_start <= point.x < x_end and y_start <= point.y < y_end


@dataclass
class Stroke:
    """A stroke as a sequence of points."""
    points: List[Point] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[Point]:
        return iter(self.points)

    def __getitem__(self, idx) -> Point:
        return self.points[idx]

    @property
    def start(self) -> Point:
        """First point of stroke."""
        return self.points[0] if self.points else Point(0, 0)

    @property
    def end(self) -> Point:
        """Last point of stroke."""
        return self.points[-1] if self.points else Point(0, 0)

    @property
    def bbox(self) -> BBox:
        """Bounding box of stroke."""
        return BBox.from_points(self.points)

    def length(self) -> float:
        """Total arc length of stroke."""
        total = 0.0
        for i in range(1, len(self.points)):
            total += self.points[i].distance_to(self.points[i - 1])
        return total

    def direction_at_end(self, from_end: bool = True, n_points: int = 8) -> Point:
        """Get direction vector at one end of stroke."""
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
        """Return stroke with reversed point order."""
        return Stroke(list(reversed(self.points)))

    def to_list(self) -> List[List[float]]:
        """Convert to nested list for JSON serialization."""
        return [p.to_list() for p in self.points]

    @classmethod
    def from_list(cls, lst: List[List[float]]) -> Stroke:
        """Create from nested list."""
        return cls([Point.from_list(p) for p in lst])

    @classmethod
    def from_tuples(cls, tuples: List[Tuple[float, float]]) -> Stroke:
        """Create from list of tuples."""
        return cls([Point.from_tuple(t) for t in tuples])


@dataclass(frozen=True)
class Segment:
    """A skeleton segment with metadata."""
    start: Point
    end: Point
    angle: float  # degrees, 0 = horizontal, 90 = vertical
    length: float
    start_junction: int = -1  # junction cluster index, -1 if endpoint
    end_junction: int = -1

    @property
    def is_vertical(self) -> bool:
        """Check if segment is approximately vertical (60-120 degrees)."""
        return 60 <= abs(self.angle) <= 120

    @property
    def is_horizontal(self) -> bool:
        """Check if segment is approximately horizontal."""
        return abs(self.angle) <= 30 or abs(self.angle) >= 150

    @property
    def direction(self) -> Point:
        """Unit direction vector from start to end."""
        return (self.end - self.start).normalized()

    @classmethod
    def from_dict(cls, d: dict) -> Segment:
        """Create from dictionary (for compatibility)."""
        return cls(
            start=Point.from_tuple(d['start']) if isinstance(d['start'], tuple) else Point(d['start'][0], d['start'][1]),
            end=Point.from_tuple(d['end']) if isinstance(d['end'], tuple) else Point(d['end'][0], d['end'][1]),
            angle=d['angle'],
            length=d['length'],
            start_junction=d.get('start_junction', -1),
            end_junction=d.get('end_junction', -1),
        )
