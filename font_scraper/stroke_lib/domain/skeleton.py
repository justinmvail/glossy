"""Skeleton-related domain objects."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional
from enum import Enum

from .geometry import Point


class MarkerType(Enum):
    """Types of skeleton markers."""
    VERTEX = 'vertex'
    INTERSECTION = 'intersection'
    TERMINATION = 'termination'


@dataclass
class Marker:
    """A skeleton marker (vertex, intersection, or termination)."""
    position: Point
    marker_type: MarkerType

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'x': round(self.position.x, 1),
            'y': round(self.position.y, 1),
            'type': self.marker_type.value
        }

    @classmethod
    def from_dict(cls, d: dict) -> Marker:
        """Create from dictionary."""
        return cls(
            position=Point(d['x'], d['y']),
            marker_type=MarkerType(d['type'])
        )


@dataclass
class JunctionCluster:
    """A cluster of junction pixels in the skeleton."""
    pixels: Set[Tuple[int, int]]
    index: int

    @property
    def centroid(self) -> Point:
        """Center of the cluster."""
        if not self.pixels:
            return Point(0, 0)
        cx = sum(p[0] for p in self.pixels) / len(self.pixels)
        cy = sum(p[1] for p in self.pixels) / len(self.pixels)
        return Point(cx, cy)

    def contains(self, point: Tuple[int, int]) -> bool:
        """Check if a pixel is in this cluster."""
        return point in self.pixels


@dataclass
class SkeletonInfo:
    """Complete skeleton analysis result."""
    skel_set: Set[Tuple[int, int]]
    adj: Dict[Tuple[int, int], Set[Tuple[int, int]]]
    endpoints: Set[Tuple[int, int]]
    junction_pixels: Set[Tuple[int, int]]
    junction_clusters: List[Set[Tuple[int, int]]]
    assigned: Dict[Tuple[int, int], int]  # pixel -> cluster index

    @classmethod
    def from_dict(cls, d: dict) -> SkeletonInfo:
        """Create from dictionary (for compatibility with existing code)."""
        return cls(
            skel_set=d['skel_set'],
            adj=d['adj'],
            endpoints=d['endpoints'],
            junction_pixels=d['junction_pixels'],
            junction_clusters=d['junction_clusters'],
            assigned=d['assigned'],
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility."""
        return {
            'skel_set': self.skel_set,
            'adj': self.adj,
            'endpoints': self.endpoints,
            'junction_pixels': self.junction_pixels,
            'junction_clusters': self.junction_clusters,
            'assigned': self.assigned,
        }

    def get_cluster_for_pixel(self, pixel: Tuple[int, int]) -> Optional[int]:
        """Get the junction cluster index for a pixel, or None if not in a cluster."""
        return self.assigned.get(pixel)

    def is_stop_point(self, pixel: Tuple[int, int]) -> bool:
        """Check if pixel is an endpoint or junction pixel."""
        return pixel in self.endpoints or pixel in self.junction_pixels
