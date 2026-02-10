"""Stroke Editor Package.

A refactored, modular architecture for stroke editing functionality. This package
provides tools for analyzing font glyphs, extracting skeleton information, detecting
markers, and performing stroke optimization.

The package is organized into the following modules:
    domain: Core value objects including Point, BBox, Stroke, Segment, and
        skeleton-related data structures.
    analysis: Skeleton analysis and segment classification tools for extracting
        structural information from glyph masks.
    optimization: Stroke optimization strategies using affine transformations,
        greedy per-shape optimization, and joint refinement.
    templates: Numpad-based stroke templates and a repository for managing them.
    utils: Geometry utilities and glyph rendering functions.
    api: Service layer providing high-level interfaces for API integration.

Example usage:
    Basic marker detection::

        from stroke_lib.api import StrokeService

        service = StrokeService()
        markers = service.detect_markers('/path/to/font.ttf', 'A')
        for marker in markers:
            print(f"Marker at ({marker['x']}, {marker['y']}): {marker['type']}")

    Low-level skeleton analysis::

        from stroke_lib.analysis import SkeletonAnalyzer

        analyzer = SkeletonAnalyzer()
        info = analyzer.analyze(mask)
        strokes = analyzer.to_strokes(mask)

    Working with domain objects::

        from stroke_lib.domain import Point, BBox, Stroke

        bbox = BBox(0, 0, 100, 100)
        center = bbox.center
        stroke = Stroke([Point(0, 0), Point(50, 50), Point(100, 100)])
        print(f"Stroke length: {stroke.length()}")

Attributes:
    __version__ (str): Package version string.
    __all__ (list): List of public symbols exported by this package.
"""

from .domain import Point, BBox, Stroke, Segment, SkeletonInfo, Marker
from .analysis import SkeletonAnalyzer, SegmentClassifier
from .api import StrokeService, FontService

__all__ = [
    # Domain objects
    'Point', 'BBox', 'Stroke', 'Segment', 'SkeletonInfo', 'Marker',
    # Analysis
    'SkeletonAnalyzer', 'SegmentClassifier',
    # Services
    'StrokeService', 'FontService',
]

__version__ = '2.0.0'
