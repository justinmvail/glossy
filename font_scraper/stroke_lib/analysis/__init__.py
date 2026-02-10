"""Skeleton and glyph analysis module.

This module provides tools for analyzing glyph skeletons and classifying
skeleton segments. It is the core analysis layer for extracting structural
information from binary glyph masks.

The module exports two main classes:
    SkeletonAnalyzer: Facade for skeleton extraction, marker detection, and
        stroke tracing operations.
    SegmentClassifier: Classifies skeleton segments by direction (vertical,
        horizontal) and finds connected segment chains.

Example usage:
    Analyze a glyph skeleton::

        from stroke_lib.analysis import SkeletonAnalyzer
        import numpy as np

        # Assuming mask is a binary numpy array
        analyzer = SkeletonAnalyzer(merge_distance=12)
        info = analyzer.analyze(mask)
        if info:
            print(f"Endpoints: {len(info.endpoints)}")
            print(f"Junction clusters: {len(info.junction_clusters)}")

    Classify segments::

        from stroke_lib.analysis import SegmentClassifier

        classifier = SegmentClassifier()
        segments = classifier.classify(info)
        vertical = classifier.find_vertical_segments(segments)
"""

from .skeleton import SkeletonAnalyzer
from .segments import SegmentClassifier

__all__ = ['SkeletonAnalyzer', 'SegmentClassifier']
