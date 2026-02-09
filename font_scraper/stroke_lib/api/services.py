"""Service layer for stroke editing operations."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

from ..domain.geometry import Stroke, BBox, Point
from ..domain.skeleton import Marker
from ..analysis.skeleton import SkeletonAnalyzer
from ..analysis.segments import SegmentClassifier
from ..utils.rendering import render_glyph_mask, get_glyph_bbox


@dataclass
class StrokeService:
    """Service for stroke-related operations.

    Provides a clean interface for the API layer, hiding implementation details.
    """

    def __init__(self):
        self.skeleton_analyzer = SkeletonAnalyzer()
        self.segment_classifier = SegmentClassifier()

    def detect_markers(self, font_path: str, char: str, canvas_size: int = 224) -> List[Dict]:
        """Detect skeleton markers for a glyph.

        Args:
            font_path: Path to font file
            char: Character to analyze
            canvas_size: Canvas size for rendering

        Returns:
            List of marker dictionaries with x, y, type
        """
        mask = render_glyph_mask(font_path, char, canvas_size)
        if mask is None:
            return []

        markers = self.skeleton_analyzer.detect_markers(mask)
        return [m.to_dict() for m in markers]

    def extract_strokes_from_skeleton(
        self,
        font_path: str,
        char: str,
        canvas_size: int = 224,
        min_stroke_len: int = 5
    ) -> Optional[List[List[List[float]]]]:
        """Extract strokes from glyph skeleton.

        Args:
            font_path: Path to font file
            char: Character to process
            canvas_size: Canvas size for rendering
            min_stroke_len: Minimum points per stroke

        Returns:
            Nested list of strokes [[[x,y], ...], ...] or None
        """
        mask = render_glyph_mask(font_path, char, canvas_size)
        if mask is None:
            return None

        strokes = self.skeleton_analyzer.to_strokes(mask, min_stroke_len)
        if not strokes:
            return None

        return [s.to_list() for s in strokes]

    def get_glyph_info(
        self,
        font_path: str,
        char: str,
        canvas_size: int = 224
    ) -> Optional[Dict[str, Any]]:
        """Get comprehensive glyph information.

        Args:
            font_path: Path to font file
            char: Character to analyze
            canvas_size: Canvas size for rendering

        Returns:
            Dictionary with bbox, skeleton info, segments, etc.
        """
        mask = render_glyph_mask(font_path, char, canvas_size)
        if mask is None:
            return None

        bbox = get_glyph_bbox(mask)
        if bbox is None:
            return None

        info = self.skeleton_analyzer.analyze(mask)
        if info is None:
            return None

        segments = self.segment_classifier.classify(info)

        return {
            'bbox': bbox.to_tuple(),
            'skeleton_pixels': len(info.skel_set),
            'endpoints': len(info.endpoints),
            'junctions': len(info.junction_clusters),
            'segments': len(segments),
            'vertical_segments': len([s for s in segments if s.is_vertical]),
        }

    def analyze_shape_metrics(
        self,
        font_path: str,
        text: str,
        size: int = 100
    ) -> Dict[str, Any]:
        """Analyze shape metrics for text rendering.

        Args:
            font_path: Path to font file
            text: Text to analyze
            size: Font size

        Returns:
            Dictionary with shape_count, max_width_pct, etc.
        """
        from ..utils.rendering import render_text_for_analysis
        from scipy import ndimage

        arr = render_text_for_analysis(font_path, text, size)
        if arr is None:
            return {'error': 'Failed to render'}

        # Label connected components
        labeled, num_shapes = ndimage.label(arr)

        # Find widest shape
        width = arr.shape[1]
        max_width = 0
        for i in range(1, num_shapes + 1):
            cols = np.where(np.any(labeled == i, axis=0))[0]
            if len(cols) > 0:
                shape_width = cols[-1] - cols[0]
                max_width = max(max_width, shape_width)

        return {
            'shape_count': num_shapes,
            'max_width_pct': max_width / width if width > 0 else 0,
            'total_pixels': int(arr.sum()),
        }


@dataclass
class FontService:
    """Service for font-related operations."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_font_path(self, font_id: int) -> Optional[str]:
        """Get file path for a font by ID."""
        import sqlite3

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT file_path FROM fonts WHERE id = ?",
                (font_id,)
            )
            row = cursor.fetchone()
            conn.close()
            return row['file_path'] if row else None
        except Exception:
            return None

    def get_character_strokes(
        self,
        font_id: int,
        char: str
    ) -> Optional[List[List[List[float]]]]:
        """Get saved strokes for a character."""
        import sqlite3
        import json

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT strokes FROM characters WHERE font_id = ? AND char = ?",
                (font_id, char)
            )
            row = cursor.fetchone()
            conn.close()

            if row and row['strokes']:
                return json.loads(row['strokes'])
            return None
        except Exception:
            return None

    def save_character_strokes(
        self,
        font_id: int,
        char: str,
        strokes: List[List[List[float]]]
    ) -> bool:
        """Save strokes for a character."""
        import sqlite3
        import json

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                INSERT OR REPLACE INTO characters (font_id, char, strokes)
                VALUES (?, ?, ?)
                """,
                (font_id, char, json.dumps(strokes))
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
