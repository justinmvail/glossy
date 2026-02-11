"""API layer for stroke editing.

This module provides the service layer for stroke editing operations,
offering high-level interfaces suitable for API integration. The services
encapsulate the complexity of skeleton analysis, stroke extraction, and
font database operations.

The module exports two service classes:
    StrokeService: Provides stroke-related operations including marker
        detection, stroke extraction, and glyph analysis.
    FontService: Handles font database operations for storing and
        retrieving stroke data.

Example usage:
    Detect markers for a glyph::

        from stroke_lib.api import StrokeService

        service = StrokeService()
        markers = service.detect_markers('/path/to/font.ttf', 'A')
        for marker in markers:
            print(f"{marker['type']} at ({marker['x']}, {marker['y']})")

    Get comprehensive glyph information::

        info = service.get_glyph_info('/path/to/font.ttf', 'B')
        if info:
            print(f"Bounding box: {info['bbox']}")
            print(f"Segments: {info['segments']}")

    Work with font database::

        from stroke_lib.api import FontService

        font_service = FontService('/path/to/fonts.db')
        font_path = font_service.get_font_path(font_id=123)
        strokes = font_service.get_character_strokes(123, 'A')
"""

from .services import FontService, StrokeService

__all__ = ['StrokeService', 'FontService']
