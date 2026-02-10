"""Skeleton-related domain objects.

This module provides data structures for representing skeleton analysis
results, including markers, junction clusters, and complete skeleton
information. These objects encapsulate the results of morphological
skeleton extraction and analysis.

The module provides the following classes:
    MarkerType: Enumeration of marker types (vertex, intersection, termination).
    Marker: A detected marker with position and type.
    JunctionCluster: A cluster of connected junction pixels.
    SkeletonInfo: Complete results of skeleton analysis.

These classes are used by the SkeletonAnalyzer to return analysis results
and by downstream code to work with skeleton structure.

Example usage:
    Working with markers::

        from stroke_lib.domain.skeleton import Marker, MarkerType
        from stroke_lib.domain.geometry import Point

        marker = Marker(Point(100, 50), MarkerType.VERTEX)
        print(marker.to_dict())
        # {'x': 100.0, 'y': 50.0, 'type': 'vertex'}

    Working with skeleton info::

        from stroke_lib.analysis import SkeletonAnalyzer

        analyzer = SkeletonAnalyzer()
        info = analyzer.analyze(mask)

        # Access skeleton properties
        print(f"Skeleton has {len(info.skel_set)} pixels")
        print(f"Endpoints: {len(info.endpoints)}")

        # Check if a pixel is a stop point
        if info.is_stop_point((100, 50)):
            print("This pixel is an endpoint or junction")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional
from enum import Enum

from .geometry import Point


class MarkerType(Enum):
    """Types of skeleton markers.

    Defines the three types of structural markers that can be detected
    in a glyph skeleton:
        VERTEX: A junction point where multiple strokes meet at sharp
            angles, typically at corners of letters.
        INTERSECTION: A junction point where strokes cross each other,
            such as in the letter 'X'.
        TERMINATION: An endpoint where a stroke ends, such as at the
            tips of serifs or the ends of strokes.

    Example:
        >>> marker_type = MarkerType.VERTEX
        >>> marker_type.value
        'vertex'
    """
    VERTEX = 'vertex'
    INTERSECTION = 'intersection'
    TERMINATION = 'termination'


@dataclass
class Marker:
    """A skeleton marker (vertex, intersection, or termination).

    Represents a detected structural point in the skeleton with its
    position and type. Markers are used to identify important
    structural features of glyphs for stroke editing and analysis.

    Attributes:
        position: Point indicating the x, y coordinates of the marker.
        marker_type: MarkerType enum indicating the type of marker.

    Example:
        >>> marker = Marker(Point(112.5, 45.0), MarkerType.VERTEX)
        >>> marker.to_dict()
        {'x': 112.5, 'y': 45.0, 'type': 'vertex'}
    """
    position: Point
    marker_type: MarkerType

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with 'x', 'y' (rounded to 1 decimal place),
            and 'type' (string value of MarkerType) keys.
        """
        return {
            'x': round(self.position.x, 1),
            'y': round(self.position.y, 1),
            'type': self.marker_type.value
        }

    @classmethod
    def from_dict(cls, d: dict) -> Marker:
        """Create from dictionary.

        Args:
            d: Dictionary with 'x', 'y', and 'type' keys.

        Returns:
            New Marker with properties from the dictionary.
        """
        return cls(
            position=Point(d['x'], d['y']),
            marker_type=MarkerType(d['type'])
        )


@dataclass
class JunctionCluster:
    """A cluster of junction pixels in the skeleton.

    Represents a group of connected junction pixels that together form
    a single logical junction point. In thick strokes, junctions may
    span multiple pixels, so clustering groups them together.

    Attributes:
        pixels: Set of (x, y) tuples representing the pixel coordinates
            of junction pixels in this cluster.
        index: Unique index of this cluster within the skeleton.

    Example:
        >>> cluster = JunctionCluster(pixels={(10, 10), (10, 11), (11, 10)}, index=0)
        >>> cluster.centroid
        Point(x=10.333333333333334, y=10.333333333333334)
    """
    pixels: Set[Tuple[int, int]]
    index: int

    @property
    def centroid(self) -> Point:
        """Center of the cluster.

        Computes the arithmetic mean of all pixel coordinates.

        Returns:
            Point at the centroid of the cluster pixels.
            Returns Point(0, 0) if the cluster is empty.
        """
        if not self.pixels:
            return Point(0, 0)
        cx = sum(p[0] for p in self.pixels) / len(self.pixels)
        cy = sum(p[1] for p in self.pixels) / len(self.pixels)
        return Point(cx, cy)

    def contains(self, point: Tuple[int, int]) -> bool:
        """Check if a pixel is in this cluster.

        Args:
            point: Pixel coordinates as (x, y) tuple.

        Returns:
            True if the pixel is part of this cluster.
        """
        return point in self.pixels


@dataclass
class SkeletonInfo:
    """Complete skeleton analysis result.

    Contains all information extracted from skeleton analysis, including
    the skeleton pixels, adjacency graph, endpoints, junction pixels,
    and clustered junctions. This is the primary result object returned
    by SkeletonAnalyzer.analyze().

    Attributes:
        skel_set: Set of (x, y) tuples for all skeleton pixels.
        adj: Adjacency dictionary mapping each pixel to its set of
            8-connected neighbor pixels in the skeleton.
        endpoints: Set of (x, y) tuples for pixels with degree 1
            (single neighbor).
        junction_pixels: Set of (x, y) tuples for pixels with degree 3+
            (three or more neighbors).
        junction_clusters: List of pixel sets, where each set contains
            the pixels belonging to a junction cluster.
        assigned: Dictionary mapping junction pixels to their cluster
            index. Pixels not in a junction are not in this dict.

    Example:
        >>> info = analyzer.analyze(mask)
        >>> len(info.skel_set)  # Total skeleton pixels
        234
        >>> len(info.endpoints)  # Number of endpoints
        4
        >>> info.is_stop_point((100, 50))
        True
    """
    skel_set: Set[Tuple[int, int]]
    adj: Dict[Tuple[int, int], Set[Tuple[int, int]]]
    endpoints: Set[Tuple[int, int]]
    junction_pixels: Set[Tuple[int, int]]
    junction_clusters: List[Set[Tuple[int, int]]]
    assigned: Dict[Tuple[int, int], int]  # pixel -> cluster index

    @classmethod
    def from_dict(cls, d: dict) -> SkeletonInfo:
        """Create from dictionary (for compatibility with existing code).

        Args:
            d: Dictionary with keys matching the attribute names:
                'skel_set', 'adj', 'endpoints', 'junction_pixels',
                'junction_clusters', 'assigned'.

        Returns:
            New SkeletonInfo with data from the dictionary.
        """
        return cls(
            skel_set=d['skel_set'],
            adj=d['adj'],
            endpoints=d['endpoints'],
            junction_pixels=d['junction_pixels'],
            junction_clusters=d['junction_clusters'],
            assigned=d['assigned'],
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility.

        Returns:
            Dictionary with all skeleton data. Note that sets will need
            to be converted for JSON serialization.
        """
        return {
            'skel_set': self.skel_set,
            'adj': self.adj,
            'endpoints': self.endpoints,
            'junction_pixels': self.junction_pixels,
            'junction_clusters': self.junction_clusters,
            'assigned': self.assigned,
        }

    def get_cluster_for_pixel(self, pixel: Tuple[int, int]) -> Optional[int]:
        """Get the junction cluster index for a pixel, or None if not in a cluster.

        Args:
            pixel: Pixel coordinates as (x, y) tuple.

        Returns:
            Cluster index if the pixel is in a junction cluster,
            None otherwise.
        """
        return self.assigned.get(pixel)

    def is_stop_point(self, pixel: Tuple[int, int]) -> bool:
        """Check if pixel is an endpoint or junction pixel.

        Stop points are pixels where stroke tracing should stop,
        either because they are endpoints (degree 1) or junctions
        (degree 3+).

        Args:
            pixel: Pixel coordinates as (x, y) tuple.

        Returns:
            True if the pixel is an endpoint or junction pixel.
        """
        return pixel in self.endpoints or pixel in self.junction_pixels
