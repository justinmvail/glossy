"""Service layer for stroke editing operations.

This module provides high-level service classes that encapsulate the
complexity of skeleton analysis, stroke extraction, and font database
operations. These services are designed for API integration and provide
clean, dictionary-based interfaces suitable for JSON serialization.

The module contains two main service classes:
    StrokeService: Handles stroke-related operations including marker
        detection, stroke extraction from skeletons, and shape analysis.
    FontService: Manages font database operations for storing and
        retrieving character stroke data.

Example usage:
    StrokeService operations::

        from stroke_lib.api.services import StrokeService

        service = StrokeService()

        # Detect markers
        markers = service.detect_markers('/fonts/arial.ttf', 'A')

        # Extract strokes
        strokes = service.extract_strokes_from_skeleton('/fonts/arial.ttf', 'A')

        # Get comprehensive info
        info = service.get_glyph_info('/fonts/arial.ttf', 'A')

    FontService operations::

        from stroke_lib.api.services import FontService

        font_service = FontService('/data/fonts.db')
        path = font_service.get_font_path(123)
        strokes = font_service.get_character_strokes(123, 'A')
        font_service.save_character_strokes(123, 'A', strokes)
"""

from __future__ import annotations
import logging
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

from ..domain.geometry import Stroke, BBox, Point
from ..domain.skeleton import Marker
from ..analysis.skeleton import SkeletonAnalyzer
from ..analysis.segments import SegmentClassifier
from ..utils.rendering import render_glyph_mask, get_glyph_bbox

# Logger for service errors
_logger = logging.getLogger(__name__)


@dataclass
class StrokeService:
    """Service for stroke-related operations.

    Provides a clean interface for the API layer, hiding implementation
    details of skeleton analysis and stroke extraction. All methods return
    serializable data structures (dictionaries, lists) suitable for JSON
    responses.

    The service coordinates multiple analysis components:
        - SkeletonAnalyzer for skeleton extraction and marker detection
        - SegmentClassifier for segment analysis
        - Rendering utilities for glyph mask generation

    Attributes:
        skeleton_analyzer: SkeletonAnalyzer instance for skeleton operations.
        segment_classifier: SegmentClassifier instance for segment analysis.

    Example:
        >>> service = StrokeService()
        >>> markers = service.detect_markers('/fonts/arial.ttf', 'A')
        >>> for m in markers:
        ...     print(f"{m['type']}: ({m['x']}, {m['y']})")
    """

    def __init__(self):
        """Initialize the stroke service.

        Creates instances of SkeletonAnalyzer and SegmentClassifier for
        performing stroke-related operations.
        """
        self.skeleton_analyzer = SkeletonAnalyzer()
        self.segment_classifier = SegmentClassifier()

    def detect_markers(self, font_path: str, char: str, canvas_size: int = 224) -> List[Dict]:
        """Detect skeleton markers for a glyph.

        Renders the specified character from the font and analyzes its
        skeleton to detect vertex, intersection, and termination markers.

        Args:
            font_path: Path to the font file (TTF, OTF, etc.).
            char: Single character to analyze.
            canvas_size: Size of the square canvas for rendering the glyph.
                Larger sizes provide more detail but take longer to process.
                Default is 224 pixels.

        Returns:
            List of marker dictionaries, each containing:
                - 'x' (float): X coordinate of the marker
                - 'y' (float): Y coordinate of the marker
                - 'type' (str): Marker type ('vertex', 'intersection', or
                    'termination')
            Returns empty list if rendering or analysis fails.

        Example:
            >>> service = StrokeService()
            >>> markers = service.detect_markers('/fonts/arial.ttf', 'A')
            >>> print(markers[0])
            {'x': 112.0, 'y': 45.0, 'type': 'vertex'}
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

        Renders the character, extracts its skeleton, and traces stroke
        paths through the skeleton graph. Returns strokes as nested lists
        suitable for JSON serialization.

        Args:
            font_path: Path to the font file.
            char: Single character to process.
            canvas_size: Size of the square canvas for rendering.
                Default is 224 pixels.
            min_stroke_len: Minimum number of points required for a valid
                stroke. Shorter strokes are filtered out. Default is 5.

        Returns:
            Nested list structure [[[x, y], ...], ...] where each inner
            list is a stroke containing [x, y] coordinate pairs.
            Returns None if rendering or analysis fails.

        Example:
            >>> service = StrokeService()
            >>> strokes = service.extract_strokes_from_skeleton('/fonts/arial.ttf', 'l')
            >>> print(len(strokes))  # Number of strokes
            1
            >>> print(len(strokes[0]))  # Points in first stroke
            45
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

        Performs full skeleton analysis and returns a dictionary with
        bounding box, skeleton statistics, and segment information.

        Args:
            font_path: Path to the font file.
            char: Single character to analyze.
            canvas_size: Size of the square canvas for rendering.
                Default is 224 pixels.

        Returns:
            Dictionary containing:
                - 'bbox' (tuple): Bounding box as (x_min, y_min, x_max, y_max)
                - 'skeleton_pixels' (int): Number of pixels in the skeleton
                - 'endpoints' (int): Number of skeleton endpoints
                - 'junctions' (int): Number of junction clusters
                - 'segments' (int): Total number of segments
                - 'vertical_segments' (int): Number of vertical segments
            Returns None if rendering or analysis fails.

        Example:
            >>> service = StrokeService()
            >>> info = service.get_glyph_info('/fonts/arial.ttf', 'H')
            >>> print(info['junctions'])
            2
            >>> print(info['vertical_segments'])
            2
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

        Renders the text and analyzes connected components to compute
        shape statistics useful for layout and rendering decisions.

        Args:
            font_path: Path to the font file.
            text: Text string to analyze (can be multiple characters).
            size: Font size for rendering. Default is 100.

        Returns:
            Dictionary containing:
                - 'shape_count' (int): Number of connected components
                - 'max_width_pct' (float): Width of widest shape as fraction
                    of total width (0.0 to 1.0)
                - 'total_pixels' (int): Total number of foreground pixels
            If rendering fails, returns {'error': 'Failed to render'}.

        Example:
            >>> service = StrokeService()
            >>> metrics = service.analyze_shape_metrics('/fonts/arial.ttf', 'Hello')
            >>> print(metrics['shape_count'])
            5
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
    """Service for font-related database operations.

    Provides methods for storing and retrieving font and character stroke
    data from a SQLite database. This service handles all database
    connections and query execution internally.

    Attributes:
        db_path: Path to the SQLite database file containing font data.
        connection_factory: Optional callable that returns a database connection.
            If not provided, uses sqlite3.connect(db_path).

    Example:
        >>> service = FontService('/data/fonts.db')
        >>> path = service.get_font_path(123)
        >>> strokes = service.get_character_strokes(123, 'A')
        >>> service.save_character_strokes(123, 'A', strokes)

        # For testing with a mock database:
        >>> mock_conn = create_mock_connection()
        >>> service = FontService('/data/fonts.db', connection_factory=lambda: mock_conn)
    """

    def __init__(self, db_path: str, connection_factory=None):
        """Initialize the font service.

        Args:
            db_path: Path to the SQLite database file. The database should
                contain 'fonts' and 'characters' tables with appropriate
                schema.
            connection_factory: Optional callable returning a database connection.
                Useful for testing with mock databases. If None, uses
                sqlite3.connect(db_path).
        """
        self.db_path = db_path
        self._connection_factory = connection_factory

    def _get_connection(self):
        """Get a database connection using the configured factory.

        Returns:
            Database connection object.
        """
        if self._connection_factory:
            return self._connection_factory()
        return sqlite3.connect(self.db_path)

    def get_font_path(self, font_id: int) -> Optional[str]:
        """Get file path for a font by ID.

        Queries the database for the file path of a font with the given ID.

        Args:
            font_id: Unique identifier for the font in the database.

        Returns:
            File path string for the font file, or None if the font is not
            found or a database error occurs.

        Example:
            >>> service = FontService('/data/fonts.db')
            >>> path = service.get_font_path(123)
            >>> print(path)
            '/fonts/Arial-Regular.ttf'
        """
        conn = None
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT file_path FROM fonts WHERE id = ?",
                (font_id,)
            )
            row = cursor.fetchone()
            return row['file_path'] if row else None
        except sqlite3.Error as e:
            _logger.warning("Database error getting font path for id=%s: %s", font_id, e)
            return None
        except Exception as e:
            _logger.error("Unexpected error getting font path: %s", e, exc_info=True)
            return None
        finally:
            if conn:
                conn.close()

    def get_character_strokes(
        self,
        font_id: int,
        char: str
    ) -> Optional[List[List[List[float]]]]:
        """Get saved strokes for a character.

        Retrieves previously saved stroke data for a specific character
        in a specific font from the database.

        Args:
            font_id: Unique identifier for the font.
            char: Single character to retrieve strokes for.

        Returns:
            Nested list structure [[[x, y], ...], ...] representing the
            saved strokes, or None if no strokes are saved or a database
            error occurs.

        Example:
            >>> service = FontService('/data/fonts.db')
            >>> strokes = service.get_character_strokes(123, 'A')
            >>> if strokes:
            ...     print(f"Found {len(strokes)} strokes")
        """
        import json

        conn = None
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT strokes FROM characters WHERE font_id = ? AND char = ?",
                (font_id, char)
            )
            row = cursor.fetchone()

            if row and row['strokes']:
                return json.loads(row['strokes'])
            return None
        except sqlite3.Error as e:
            _logger.warning("Database error getting strokes for font=%s char=%s: %s",
                           font_id, char, e)
            return None
        except json.JSONDecodeError as e:
            _logger.warning("Invalid JSON in strokes for font=%s char=%s: %s",
                           font_id, char, e)
            return None
        except Exception as e:
            _logger.error("Unexpected error getting strokes: %s", e, exc_info=True)
            return None
        finally:
            if conn:
                conn.close()

    def save_character_strokes(
        self,
        font_id: int,
        char: str,
        strokes: List[List[List[float]]]
    ) -> bool:
        """Save strokes for a character.

        Stores stroke data for a character in the database, replacing any
        existing data for the same font/character combination.

        Args:
            font_id: Unique identifier for the font.
            char: Single character to save strokes for.
            strokes: Nested list structure [[[x, y], ...], ...] representing
                the strokes to save.

        Returns:
            True if the save was successful, False if a database error
            occurred.

        Example:
            >>> service = FontService('/data/fonts.db')
            >>> strokes = [[[0.0, 0.0], [10.0, 100.0]]]
            >>> success = service.save_character_strokes(123, 'l', strokes)
            >>> print(success)
            True
        """
        import json

        conn = None
        try:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT OR REPLACE INTO characters (font_id, char, strokes)
                VALUES (?, ?, ?)
                """,
                (font_id, char, json.dumps(strokes))
            )
            conn.commit()
            return True
        except sqlite3.Error as e:
            _logger.warning("Database error saving strokes for font=%s char=%s: %s",
                           font_id, char, e)
            return False
        except (TypeError, ValueError) as e:
            _logger.warning("Invalid stroke data for font=%s char=%s: %s",
                           font_id, char, e)
            return False
        except Exception as e:
            _logger.error("Unexpected error saving strokes: %s", e, exc_info=True)
            return False
        finally:
            if conn:
                conn.close()
