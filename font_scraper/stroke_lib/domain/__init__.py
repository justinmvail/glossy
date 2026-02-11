"""Domain objects for stroke editing.

This module provides the core value objects and data structures used
throughout the stroke editing package. These objects represent the
fundamental concepts in the domain: geometric primitives, strokes,
skeleton analysis results, and markers.

The module exports the following classes:

Geometry classes:
    Point: Immutable 2D point with vector operations.
    BBox: Immutable bounding box with numpad region support.
    Stroke: Sequence of points representing a stroke path.
    Segment: Skeleton segment with geometric metadata.

Skeleton classes:
    SkeletonInfo: Complete skeleton analysis result.
    JunctionCluster: Cluster of junction pixels.
    Marker: Detected marker (vertex, intersection, or termination).

Example usage:
    Working with geometry::

        from stroke_lib.domain import Point, BBox, Stroke

        # Create points
        p1 = Point(0, 0)
        p2 = Point(100, 100)
        distance = p1.distance_to(p2)

        # Create bounding box
        bbox = BBox(0, 0, 200, 200)
        center = bbox.center

        # Create stroke
        stroke = Stroke([p1, Point(50, 50), p2])
        print(f"Stroke length: {stroke.length()}")

    Working with skeleton data::

        from stroke_lib.domain import SkeletonInfo, Marker

        # Markers from analysis
        markers = analyzer.detect_markers(mask)
        for m in markers:
            print(f"{m.marker_type.value} at {m.position}")
"""

from .geometry import BBox, Point, Segment, Stroke
from .skeleton import JunctionCluster, Marker, SkeletonInfo

__all__ = [
    'Point', 'BBox', 'Stroke', 'Segment',
    'SkeletonInfo', 'JunctionCluster', 'Marker',
]
