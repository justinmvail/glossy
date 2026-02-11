"""Stroke Editor Package.

A refactored, modular architecture for stroke editing functionality. This package
provides tools for analyzing font glyphs, extracting skeleton information, detecting
markers, and performing stroke optimization.

Architecture Overview:
    This package (stroke_lib) provides a clean, object-oriented API layered on top
    of the main codebase's procedural modules (stroke_*.py files in the parent
    directory). The relationship is:

    - stroke_lib uses stroke_rendering, stroke_flask for font/glyph utilities
    - stroke_lib.analysis wraps stroke_skeleton functionality with OOP interfaces
    - stroke_lib.domain provides typed data structures (Point, Stroke, etc.)
    - stroke_lib.api offers high-level services for external consumers

    The main codebase modules remain the source of truth for algorithms, while
    stroke_lib provides ergonomic wrappers and type safety for API consumers.

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

from .analysis import SegmentClassifier, SkeletonAnalyzer
from .api import FontService, StrokeService
from .domain import BBox, Marker, Point, Segment, SkeletonInfo, Stroke

__all__ = [
    # Domain objects
    'Point', 'BBox', 'Stroke', 'Segment', 'SkeletonInfo', 'Marker',
    # Analysis
    'SkeletonAnalyzer', 'SegmentClassifier',
    # Services
    'StrokeService', 'FontService',
]

__version__ = '2.0.0'
