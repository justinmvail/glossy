"""Stroke Editor Package

A refactored, modular architecture for stroke editing functionality.

Modules:
    domain: Value objects (Point, BBox, Stroke, etc.)
    analysis: Skeleton analysis and classification
    optimization: Stroke optimization strategies
    templates: Numpad templates and repository
    utils: Geometry and rendering utilities
    api: Service layer for API integration

Example usage:
    from stroke_lib.domain import Point, BBox, Stroke
    from stroke_lib.analysis import SkeletonAnalyzer
    from stroke_lib.api import StrokeService

    # Analyze a glyph
    service = StrokeService()
    markers = service.detect_markers('/path/to/font.ttf', 'A')

    # Or use lower-level API
    analyzer = SkeletonAnalyzer()
    info = analyzer.analyze(mask)
    strokes = analyzer.to_strokes(mask)
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
