"""MinimalStrokePipeline: converts font glyphs to stroke representations via skeleton tracing.

This module provides a pipeline for generating stroke representations from font glyphs.
It works by analyzing the skeleton (medial axis) of a rendered glyph and tracing paths
through it based on configurable templates. The pipeline supports multiple methods for
stroke extraction, including template-based waypoint tracing and pure skeleton analysis.

Typical usage:

    from stroke_pipeline import MinimalStrokePipeline

    # Initialize the pipeline with required callback functions
    pipeline = MinimalStrokePipeline(
        font_path="fonts/MyFont.ttf",
        char="A",
        canvas_size=224,
        resolve_font_path_fn=resolve_font_path,
        render_glyph_mask_fn=render_glyph_mask,
        analyze_skeleton_fn=analyze_skeleton,
        find_skeleton_segments_fn=find_skeleton_segments,
        point_in_region_fn=point_in_region,
        trace_segment_fn=trace_segment,
        trace_to_region_fn=trace_to_region,
        generate_straight_line_fn=generate_straight_line,
        resample_path_fn=resample_path,
        skeleton_to_strokes_fn=skeleton_to_strokes,
        apply_stroke_template_fn=apply_stroke_template,
        adjust_stroke_paths_fn=adjust_stroke_paths,
        quick_stroke_score_fn=quick_stroke_score,
    )

    # Run the pipeline with a template
    template = [[(8, 'v'), (2, 'v')], [7, 9]]  # Example for letter 'A'
    strokes = pipeline.run(template, trace_paths=True)

    # Or evaluate all variants to find the best stroke representation
    best_result = pipeline.evaluate_all_variants()
    print(f"Best variant: {best_result.variant_name}, Score: {best_result.score}")

Key Classes:
    MinimalStrokePipeline: Main pipeline class that orchestrates the stroke extraction
        process through multiple stages: analysis, waypoint resolution, path tracing,
        and variant evaluation.

The pipeline uses a numpad-based coordinate system (1-9) to define template waypoints,
where each number corresponds to a region of the glyph's bounding box:

    7 | 8 | 9   (top-left, top-center, top-right)
    4 | 5 | 6   (middle-left, center, middle-right)
    1 | 2 | 3   (bottom-left, bottom-center, bottom-right)

Dependencies:
    - numpy: For array operations and mask processing
    - scipy.spatial.cKDTree: For efficient nearest-neighbor skeleton pixel lookups
    - stroke_dataclasses: Data structures for waypoints, segments, and analysis results
    - stroke_templates: Predefined templates for common characters
"""

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from stroke_dataclasses import (
    ARRIVAL_BRANCH_SIZE,
    ParsedWaypoint,
    ResolvedWaypoint,
    SegmentConfig,
    SkeletonAnalysis,
    VariantResult,
    parse_stroke_template,
)
from stroke_pipeline_core import (
    VERTICAL_ANGLE_MIN,
    VERTICAL_ANGLE_MAX,
    WAIST_TOLERANCE_RATIO,
    WAIST_MARGIN_RATIO,
    extract_region_from_waypoint,
    infer_direction as core_infer_direction,
    is_vertical_stroke as core_is_vertical_stroke,
)
from stroke_templates import NUMPAD_POS, NUMPAD_TEMPLATE_VARIANTS
from stroke_utils import infer_direction_from_regions, point_distance, point_in_region

# "Truly vertical" angle thresholds (narrower range than VERTICAL_ANGLE_MIN/MAX)
# Used for preferring strictly vertical segments when available
TRULY_VERTICAL_ANGLE_MIN = 75
TRULY_VERTICAL_ANGLE_MAX = 105

# Direction-to-angle range mapping for segment classification
# Angle 0 = right, 90 = down, 180/-180 = left, -90 = up
ANGLE_RANGES = {
    'down': (45, 135),      # 45 to 135 degrees (pointing downward)
    'up': (-135, -45),      # -135 to -45 degrees (pointing upward)
    'right': (-45, 45),     # -45 to 45 degrees (pointing right)
    'left': (135, 180, -180, -135),  # pointing left (wraps around)
}

# Distance thresholds for junction/segment proximity checks (in pixels)
DISTANCE_THRESHOLD_LARGE = 10   # Minimum segment length for direction filtering
DISTANCE_THRESHOLD_SMALL = 3    # Close proximity to segment start
DISTANCE_THRESHOLD_MEDIUM = 5   # Medium proximity for fallback checks
from stroke_variant_evaluator import VariantEvaluator, try_skeleton_method


@dataclass
class PipelineConfig:
    """Configuration for a stroke pipeline.

    Encapsulates all tunable parameters for pipeline creation, allowing
    easy creation of different pipeline configurations for various use cases.

    Attributes:
        canvas_size: Size of the square canvas for glyph rendering. Larger
            values provide more detail but slower processing. Default: 224.
        trace_paths: If True, trace actual skeleton paths between waypoints.
            If False, return only resolved waypoint positions. Default: True.
        use_skeleton_fallback: If True, fall back to skeleton-based stroke
            extraction when template-based methods fail. Default: True.
        min_stroke_len: Minimum stroke length for skeleton extraction.
            Shorter strokes are filtered out. Default: 5.
        resample_points: Target number of points when resampling stroke paths.
            Higher values give smoother paths. Default: 30.
        score_stroke_penalty: Penalty weight applied per extra/missing stroke
            compared to expected count. Default: 0.3.
        apply_stroke_template_fn: Optional callback to apply character-specific
            templates to strokes. Signature: (strokes, char) -> strokes.
            Default: None (uses no-op lambda).
        adjust_stroke_paths_fn: Optional callback to adjust stroke paths after
            extraction. Signature: (strokes, char, mask) -> strokes.
            Default: None (uses no-op lambda).
    """
    from typing import Callable
    canvas_size: int = 224
    trace_paths: bool = True
    use_skeleton_fallback: bool = True
    min_stroke_len: int = 5
    resample_points: int = 30
    score_stroke_penalty: float = 0.3
    apply_stroke_template_fn: Callable | None = None
    adjust_stroke_paths_fn: Callable | None = None


class PipelineFactory:
    """Factory for creating configured stroke pipelines.

    Provides named configurations for common use cases, making it easy to
    create pipelines optimized for different scenarios (speed vs. quality).

    Example:
        >>> # Standard pipeline
        >>> pipeline = PipelineFactory.create_default('fonts/Roboto.ttf', 'A')

        >>> # Fast pipeline for batch processing
        >>> pipeline = PipelineFactory.create_fast('fonts/Roboto.ttf', 'A')

        >>> # High quality for final output
        >>> pipeline = PipelineFactory.create_high_quality('fonts/Roboto.ttf', 'A')

        >>> # Custom configuration
        >>> config = PipelineConfig(canvas_size=448, resample_points=50)
        >>> pipeline = PipelineFactory.create_with_config('fonts/Roboto.ttf', 'A', config)
    """

    @staticmethod
    def create_default(font_path: str, char: str) -> 'MinimalStrokePipeline':
        """Create a pipeline with default settings.

        Standard configuration suitable for most use cases. Balances
        quality and performance.

        Args:
            font_path: Path to the font file.
            char: Character to process.

        Returns:
            Configured MinimalStrokePipeline instance.
        """
        return MinimalStrokePipeline.create_default(font_path, char)

    @staticmethod
    def create_fast(font_path: str, char: str) -> 'MinimalStrokePipeline':
        """Create a fast pipeline with reduced quality settings.

        Optimized for batch processing where speed is more important than
        perfect quality. Uses smaller canvas and fewer resample points.

        Args:
            font_path: Path to the font file.
            char: Character to process.

        Returns:
            Configured MinimalStrokePipeline instance with fast settings.
        """
        config = PipelineConfig(
            canvas_size=128,
            resample_points=20,
            min_stroke_len=3,
        )
        return PipelineFactory.create_with_config(font_path, char, config)

    @staticmethod
    def create_high_quality(font_path: str, char: str) -> 'MinimalStrokePipeline':
        """Create a high-quality pipeline for best results.

        Optimized for final output where quality matters most. Uses larger
        canvas and more resample points for smoother strokes.

        Args:
            font_path: Path to the font file.
            char: Character to process.

        Returns:
            Configured MinimalStrokePipeline instance with high-quality settings.
        """
        config = PipelineConfig(
            canvas_size=448,
            resample_points=50,
            min_stroke_len=8,
        )
        return PipelineFactory.create_with_config(font_path, char, config)

    @staticmethod
    def create_with_config(font_path: str, char: str,
                           config: PipelineConfig) -> 'MinimalStrokePipeline':
        """Create a pipeline with custom configuration.

        Args:
            font_path: Path to the font file.
            char: Character to process.
            config: Custom pipeline configuration.

        Returns:
            Configured MinimalStrokePipeline instance.
        """
        # Lazy imports to avoid circular dependencies
        from stroke_core import analyze_skeleton_legacy as analyze_skeleton
        from stroke_core import skel_strokes
        from stroke_flask import resolve_font_path
        from stroke_rendering import render_glyph_mask
        from stroke_scoring import quick_stroke_score
        from stroke_skeleton import (
            find_skeleton_segments,
            generate_straight_line,
            resample_path,
            trace_segment,
            trace_to_region,
        )
        from stroke_utils import point_in_region

        # Create wrapper functions that use config values
        def configured_resample_path(path, num_points=None):
            return resample_path(path, num_points or config.resample_points)

        def configured_skeleton_to_strokes(mask, min_stroke_len=None):
            return skel_strokes(mask, min_stroke_len or config.min_stroke_len)

        # Use config callbacks or defaults
        apply_template_fn = config.apply_stroke_template_fn or (lambda st, c: st)
        adjust_paths_fn = config.adjust_stroke_paths_fn or (lambda st, c, m: st)

        return MinimalStrokePipeline(
            font_path=font_path,
            char=char,
            canvas_size=config.canvas_size,
            resolve_font_path_fn=resolve_font_path,
            render_glyph_mask_fn=render_glyph_mask,
            analyze_skeleton_fn=analyze_skeleton,
            find_skeleton_segments_fn=find_skeleton_segments,
            point_in_region_fn=point_in_region,
            trace_segment_fn=trace_segment,
            trace_to_region_fn=trace_to_region,
            generate_straight_line_fn=generate_straight_line,
            resample_path_fn=configured_resample_path,
            skeleton_to_strokes_fn=configured_skeleton_to_strokes,
            apply_stroke_template_fn=apply_template_fn,
            adjust_stroke_paths_fn=adjust_paths_fn,
            quick_stroke_score_fn=quick_stroke_score,
        )


class MinimalStrokePipeline:
    """Pipeline for generating minimal strokes from skeleton analysis.

    This class implements a multi-stage pipeline for extracting stroke paths from
    font glyphs. It uses skeleton (medial axis) analysis combined with template-based
    waypoint resolution to produce ordered stroke sequences suitable for rendering
    or further processing.

    The pipeline operates in four main stages:
        1. analyze(): Render the glyph mask and analyze its skeleton structure
        2. resolve_waypoints(): Convert template waypoints to pixel positions
        3. trace_paths(): Connect waypoints by tracing along the skeleton
        4. evaluate_variant(): Score the resulting strokes for quality

    Attributes:
        font_path (str): Resolved path to the font file.
        char (str): The character being processed.
        canvas_size (int): Size of the square canvas for rendering (default: 224).
        analysis (SkeletonAnalysis | None): Cached skeleton analysis result,
            lazily loaded on first access.

    Example:
        >>> pipeline = MinimalStrokePipeline(
        ...     font_path="fonts/Roboto.ttf",
        ...     char="H",
        ...     canvas_size=224,
        ...     # ... callback functions ...
        ... )
        >>> # Get the best stroke representation
        >>> result = pipeline.evaluate_all_variants()
        >>> if result.strokes:
        ...     for i, stroke in enumerate(result.strokes):
        ...         print(f"Stroke {i}: {len(stroke)} points")

    Note:
        This class requires external callback functions for rendering, skeleton
        analysis, and path tracing. These are injected via constructor parameters
        to allow for flexible implementations and testing.
    """

    def __init__(self, font_path: str, char: str, canvas_size: int = 224,
                 resolve_font_path_fn=None, render_glyph_mask_fn=None,
                 analyze_skeleton_fn=None, find_skeleton_segments_fn=None,
                 point_in_region_fn=None, trace_segment_fn=None,
                 trace_to_region_fn=None, generate_straight_line_fn=None,
                 resample_path_fn=None, skeleton_to_strokes_fn=None,
                 apply_stroke_template_fn=None, adjust_stroke_paths_fn=None,
                 quick_stroke_score_fn=None):
        """Initialize the pipeline with function dependencies.

        Args:
            font_path: Path to the font file. Will be resolved using
                resolve_font_path_fn if provided.
            char: The character to process.
            canvas_size: Size of the square canvas for glyph rendering.
                Defaults to 224 pixels.
            resolve_font_path_fn: Callback to resolve relative font paths to
                absolute paths. If None, font_path is used as-is.
            render_glyph_mask_fn: Callback to render a glyph as a binary mask.
                Signature: (font_path, char, canvas_size) -> np.ndarray | None
            analyze_skeleton_fn: Callback to analyze a binary mask's skeleton.
                Returns a dict with 'skel_set', 'adj', 'endpoints', 'junction_pixels'.
            find_skeleton_segments_fn: Callback to find linear segments in the skeleton.
                Returns a list of segment dicts with 'start', 'end', 'angle', 'length'.
            point_in_region_fn: Callback to test if a point is in a numpad region.
                Signature: (point, region, bbox) -> bool
            trace_segment_fn: Callback to trace a path between two skeleton points.
                Returns a list of (x, y) tuples.
            trace_to_region_fn: Callback to trace from a point to a numpad region.
                Returns a list of (x, y) tuples.
            generate_straight_line_fn: Callback to generate a straight line path.
                Signature: (start, end) -> list[tuple]
            resample_path_fn: Callback to resample a path to a target number of points.
                Signature: (path, num_points) -> list[tuple]
            skeleton_to_strokes_fn: Callback for pure skeleton-based stroke extraction.
                Signature: (mask, min_stroke_len) -> list[list[tuple]]
            apply_stroke_template_fn: Callback to apply character-specific templates.
                Signature: (strokes, char) -> list
            adjust_stroke_paths_fn: Callback to adjust stroke paths (merging, etc.).
                Signature: (strokes, char, mask) -> list
            quick_stroke_score_fn: Callback to score stroke quality.
                Signature: (strokes, mask) -> float

        Note:
            All callback functions are optional but required for the corresponding
            pipeline functionality. The pipeline will fail gracefully if a required
            callback is missing for a specific operation.
        """
        self.font_path = resolve_font_path_fn(font_path) if resolve_font_path_fn else font_path
        self.char = char
        self.canvas_size = canvas_size
        self._analysis: SkeletonAnalysis | None = None

        # Caches for performance optimization
        self._kdtree_cache: dict[tuple, tuple] = {}  # Cache for KDTree query results
        self._junction_segment_map: dict | None = None  # Cache for junction-to-segment map

        # Store function references
        self._render_glyph_mask = render_glyph_mask_fn
        self._analyze_skeleton = analyze_skeleton_fn
        self._find_skeleton_segments = find_skeleton_segments_fn
        self._point_in_region = point_in_region_fn
        self._trace_segment = trace_segment_fn
        self._trace_to_region = trace_to_region_fn
        self._generate_straight_line = generate_straight_line_fn
        self._resample_path = resample_path_fn
        self._skeleton_to_strokes = skeleton_to_strokes_fn
        self._apply_stroke_template = apply_stroke_template_fn
        self._adjust_stroke_paths = adjust_stroke_paths_fn
        self._quick_stroke_score = quick_stroke_score_fn

    @classmethod
    def create_default(cls, font_path: str, char: str,
                       canvas_size: int = 224) -> 'MinimalStrokePipeline':
        """Factory method to create a pipeline with default dependencies.

        Creates a MinimalStrokePipeline pre-configured with the standard
        implementation functions. Use this for production code; use the
        constructor directly for testing with mock functions.

        Args:
            font_path: Path to the font file.
            char: Character to process.
            canvas_size: Canvas size for rendering. Defaults to 224.

        Returns:
            Configured MinimalStrokePipeline instance.

        Example:
            >>> pipeline = MinimalStrokePipeline.create_default('fonts/Roboto.ttf', 'A')
            >>> result = pipeline.evaluate_all_variants()
        """
        # Lazy imports to avoid circular dependencies
        from stroke_core import analyze_skeleton_legacy as analyze_skeleton
        from stroke_core import skel_strokes
        from stroke_flask import resolve_font_path
        from stroke_rendering import render_glyph_mask
        from stroke_scoring import quick_stroke_score
        from stroke_skeleton import (
            find_skeleton_segments,
            generate_straight_line,
            resample_path,
            trace_segment,
            trace_to_region,
        )
        from stroke_utils import point_in_region

        return cls(
            font_path=font_path,
            char=char,
            canvas_size=canvas_size,
            resolve_font_path_fn=resolve_font_path,
            render_glyph_mask_fn=render_glyph_mask,
            analyze_skeleton_fn=analyze_skeleton,
            find_skeleton_segments_fn=find_skeleton_segments,
            point_in_region_fn=point_in_region,
            trace_segment_fn=trace_segment,
            trace_to_region_fn=trace_to_region,
            generate_straight_line_fn=generate_straight_line,
            resample_path_fn=resample_path,
            skeleton_to_strokes_fn=skel_strokes,
            apply_stroke_template_fn=lambda st, c: st,  # Default no-op
            adjust_stroke_paths_fn=lambda st, c, m: st,  # Default no-op
            quick_stroke_score_fn=quick_stroke_score,
        )

    @property
    def analysis(self) -> SkeletonAnalysis | None:
        """Lazy-load skeleton analysis.

        Returns:
            SkeletonAnalysis object containing the skeleton data, or None if
            the glyph could not be rendered or has no pixels.

        Note:
            The analysis is computed once on first access and cached for
            subsequent calls.
        """
        if self._analysis is None:
            self._analysis = self._do_analyze()
        return self._analysis

    def _do_analyze(self) -> SkeletonAnalysis | None:
        """Stage 1: Render mask and analyze skeleton.

        Renders the glyph to a binary mask, extracts the skeleton, and computes
        various structural features including segments, junctions, and endpoints.

        Returns:
            SkeletonAnalysis containing:
                - mask: Binary mask of the rendered glyph
                - info: Dictionary with skeleton data (skel_set, adj, endpoints, etc.)
                - segments: All detected linear segments
                - vertical_segments: Segments with angle between 60-120 degrees
                - bbox: Bounding box as (x_min, y_min, x_max, y_max)
                - skel_list: List of skeleton pixel coordinates
                - skel_tree: KD-tree for efficient nearest-neighbor queries
                - glyph_rows, glyph_cols: Arrays of glyph pixel coordinates
            Returns None if rendering fails or the glyph has no pixels.
        """
        mask = self._render_glyph_mask(self.font_path, self.char, self.canvas_size)
        if mask is None:
            return None

        rows, cols = np.where(mask)
        if len(rows) == 0:
            return None

        bbox = (float(cols.min()), float(rows.min()),
                float(cols.max()), float(rows.max()))

        info = self._analyze_skeleton(mask)
        if info is None:
            return None

        segments = self._find_skeleton_segments(info)
        vertical_segments = [s for s in segments if VERTICAL_ANGLE_MIN <= abs(s['angle']) <= VERTICAL_ANGLE_MAX]

        skel_list = list(info['skel_set'])
        if not skel_list:
            return None

        skel_tree = cKDTree(skel_list)

        # Clear caches when analysis changes (new data)
        self._kdtree_cache.clear()
        self._junction_segment_map = None

        return SkeletonAnalysis(
            mask=mask, info=info, segments=segments,
            vertical_segments=vertical_segments, bbox=bbox,
            skel_list=skel_list, skel_tree=skel_tree,
            glyph_rows=rows, glyph_cols=cols,
        )

    def numpad_to_pixel(self, region: int) -> tuple[float, float]:
        """Map numpad region (1-9) to pixel coordinates.

        Converts a numpad-style region identifier to the corresponding pixel
        position within the glyph's bounding box.

        Args:
            region: Numpad region number (1-9), where:
                7=top-left, 8=top-center, 9=top-right,
                4=mid-left, 5=center, 6=mid-right,
                1=bottom-left, 2=bottom-center, 3=bottom-right

        Returns:
            Tuple of (x, y) pixel coordinates corresponding to the region center.

        Raises:
            AttributeError: If analysis has not been computed or is None.
            KeyError: If region is not in NUMPAD_POS.
        """
        bbox = self.analysis.bbox
        if bbox is None or len(bbox) < 4:
            raise ValueError("Invalid bbox: analysis.bbox must have 4 elements")
        frac_x, frac_y = NUMPAD_POS[region]
        x = bbox[0] + frac_x * (bbox[2] - bbox[0])
        y = bbox[1] + frac_y * (bbox[3] - bbox[1])
        return (x, y)

    def find_nearest_skeleton(self, pos: tuple[float, float]) -> tuple[int, int]:
        """Find the nearest skeleton pixel to a position.

        Uses a KD-tree for efficient O(log n) nearest-neighbor lookup.
        Results are cached to avoid redundant queries for the same position.

        Args:
            pos: Target position as (x, y) coordinates.

        Returns:
            The (x, y) coordinates of the nearest skeleton pixel.

        Raises:
            AttributeError: If analysis has not been computed or is None.
        """
        # Round position to create a hashable cache key
        cache_key = (round(pos[0], 2), round(pos[1], 2))
        if cache_key in self._kdtree_cache:
            return self._kdtree_cache[cache_key]

        _, idx = self.analysis.skel_tree.query(pos)
        result = self.analysis.skel_list[idx]
        self._kdtree_cache[cache_key] = result
        return result

    def _get_junction_segment_map(self, segments: list) -> dict:
        """Get or build the cached junction-to-segment index map.

        Args:
            segments: List of segment dictionaries with 'start_junction' and 'end_junction'.

        Returns:
            Dict mapping junction_id -> list of segment indices.
        """
        # Build cache key based on segment count and first/last segment identity
        # (segments list is derived from analysis.vertical_segments, which is stable)
        cache_key = (len(segments), id(segments[0]) if segments else None)
        if self._junction_segment_map is not None:
            cached_key, cached_map = self._junction_segment_map
            if cached_key == cache_key:
                return cached_map

        # Build the junction-to-segment map
        junc_segs = defaultdict(list)
        for i, seg in enumerate(segments):
            for j in [seg['start_junction'], seg['end_junction']]:
                if j >= 0:
                    junc_segs[j].append(i)

        result = dict(junc_segs)
        self._junction_segment_map = (cache_key, result)
        return result

    def find_best_vertical_segment(self, template_start: tuple[float, float],
                                   template_end: tuple[float, float]) -> tuple | None:
        """Find vertical skeleton segment(s) closest to template positions.

        Searches for vertical segments in the skeleton that best match the
        expected start and end positions from the template. Considers chains
        of connected segments to handle vertical strokes that span multiple
        skeleton segments.

        Args:
            template_start: Expected start position (x, y) from template.
            template_end: Expected end position (x, y) from template.

        Returns:
            A tuple of ((start_x, start_y), (end_x, end_y)) for the best matching
            vertical segment chain, or None if no vertical segments exist.

        Note:
            "Vertical" is defined as segments with angle between 60-120 degrees
            (where 90 degrees is perfectly vertical). The algorithm prefers
            "truly vertical" segments (75-105 degrees) when available.
        """
        vertical_segments = self.analysis.vertical_segments
        if not vertical_segments:
            return None
        truly_vertical = [s for s in vertical_segments if TRULY_VERTICAL_ANGLE_MIN <= abs(s['angle']) <= TRULY_VERTICAL_ANGLE_MAX] or vertical_segments
        # Get cached junction-to-segment map (avoids rebuilding on each call)
        junc_segs = self._get_junction_segment_map(truly_vertical)
        visited, chains = set(), []
        for i in range(len(truly_vertical)):
            if i in visited:
                continue
            chain, queue = [], deque([i])
            while queue:
                idx = queue.popleft()
                if idx in visited:
                    continue
                visited.add(idx)
                chain.append(idx)
                for j in [truly_vertical[idx]['start_junction'], truly_vertical[idx]['end_junction']]:
                    if j >= 0:
                        queue.extend(o for o in junc_segs[j] if o not in visited)
            if chain:
                chains.append(chain)
        # Score chains by distance to template positions (using shared utility)
        best, best_score = None, float('inf')
        # Early termination threshold - if we find a very close match, stop searching
        EARLY_EXIT_THRESHOLD = 5.0  # pixels
        for chain in chains:
            pts = sorted([p for i in chain for p in [truly_vertical[i]['start'], truly_vertical[i]['end']]], key=lambda p: p[1])
            if not pts:
                continue
            top, bot = pts[0], pts[-1]
            s1, s2 = point_distance(top, template_start) + point_distance(bot, template_end), point_distance(bot, template_start) + point_distance(top, template_end)
            score = min(s1, s2)
            if score < best_score:
                best_score, best = score, (top, bot) if s1 <= s2 else (bot, top)
                # Early exit if we found a very close match
                if best_score < EARLY_EXIT_THRESHOLD:
                    break
        return best

    def resolve_waypoint(self, wp: ParsedWaypoint, next_direction: str | None,
                         mid_x: float, mid_y: float, top_bound: float,
                         bot_bound: float, waist_margin: float) -> ResolvedWaypoint:
        """Stage 2: Resolve a single waypoint to pixel coordinates.

        Converts a parsed template waypoint into a concrete pixel position on the
        skeleton. The resolution strategy depends on the waypoint type:
        - Intersection waypoints: Find the nearest junction pixel
        - Vertex waypoints: Find extremum points with optional apex extension
        - Curve waypoints: Find the curve apex in the appropriate region
        - Terminal waypoints: Find endpoints or direction-appropriate positions

        Args:
            wp: Parsed waypoint containing region and type flags.
            next_direction: Direction of travel to the next waypoint
                ('up', 'down', 'left', 'right', or None).
            mid_x: X-coordinate of the bounding box center.
            mid_y: Y-coordinate of the bounding box center.
            top_bound: Y-coordinate of the top third boundary.
            bot_bound: Y-coordinate of the bottom third boundary.
            waist_margin: Margin around mid_y for waist region calculations.

        Returns:
            ResolvedWaypoint with the computed pixel position and metadata.

        Raises:
            ValueError: If wp.region is not a valid numpad region (1-9).
        """
        # Validate region is in valid range
        if not isinstance(wp.region, int) or wp.region < 1 or wp.region > 9:
            raise ValueError(f"Invalid waypoint region: {wp.region}. Must be 1-9.")

        analysis = self.analysis
        bbox = analysis.bbox
        if bbox is None or len(bbox) < 4:
            raise ValueError("Invalid bbox: analysis.bbox must have 4 elements")
        skel_list = analysis.skel_list
        info = analysis.info

        template_pos = self.numpad_to_pixel(wp.region)

        if wp.is_intersection:
            junction_pixels_in_region = [p for p in info.get('junction_pixels', [])
                                         if self._point_in_region(p, wp.region, bbox)]
            if junction_pixels_in_region:
                pos = min(junction_pixels_in_region, key=lambda p:
                         (p[0] - template_pos[0])**2 + (p[1] - template_pos[1])**2)
            else:
                pos = self.find_nearest_skeleton(template_pos)
            return ResolvedWaypoint(
                position=(float(pos[0]), float(pos[1])),
                region=wp.region, is_intersection=True
            )

        if wp.is_vertex:
            return self._resolve_vertex(wp.region, template_pos, top_bound, bot_bound, mid_x, mid_y)

        if wp.is_curve:
            is_above = template_pos[1] < mid_y
            region_pixels = [p for p in skel_list if (p[1] < mid_y - waist_margin if is_above else p[1] > mid_y + waist_margin)]
            if region_pixels:
                if template_pos[0] > mid_x:
                    apex = max(region_pixels, key=lambda p: p[0])
                elif template_pos[0] < mid_x:
                    apex = min(region_pixels, key=lambda p: p[0])
                else:
                    apex = self.find_nearest_skeleton(template_pos)
            else:
                apex = self.find_nearest_skeleton(template_pos)
            return ResolvedWaypoint(position=(float(apex[0]), float(apex[1])), region=wp.region, is_curve=True)

        return self._resolve_terminal(wp.region, template_pos, next_direction,
                                      mid_x, top_bound, bot_bound)

    def _find_extremum_by_direction(self, pixels: list, direction: str | None,
                                     template_pos: tuple[float, float]) -> tuple | None:
        """Find extremum point from pixel list based on direction.

        Args:
            pixels: List of (x, y) pixel positions.
            direction: Direction hint ('up', 'down', 'left', 'right', or None).
            template_pos: Template position for nearest-point fallback.

        Returns:
            The extremum pixel or None if no pixels.
        """
        if not pixels:
            return None

        if direction == 'down':
            return min(pixels, key=lambda p: p[1])
        elif direction == 'up':
            return max(pixels, key=lambda p: p[1])
        elif direction == 'left':
            return max(pixels, key=lambda p: p[0])
        elif direction == 'right':
            return min(pixels, key=lambda p: p[0])
        else:
            return min(pixels, key=lambda p:
                      (p[0] - template_pos[0])**2 + (p[1] - template_pos[1])**2)

    def _get_region_pixels_with_fallback(self, region: int, template_pos: tuple[float, float],
                                          top_bound: float, bot_bound: float) -> list:
        """Get skeleton pixels in region, with Y-band fallback.

        Args:
            region: Numpad region (1-9).
            template_pos: Template position as (x, y) pixel coordinates.
            top_bound: Y-coordinate of the top third boundary.
            bot_bound: Y-coordinate of the bottom third boundary.

        Returns:
            List of skeleton pixels in the region or Y-band fallback.
        """
        skel_list = self.analysis.skel_list
        bbox = self.analysis.bbox

        region_pixels = [p for p in skel_list if self._point_in_region(p, region, bbox)]
        if region_pixels:
            return region_pixels

        # Fallback to Y-band based on template position
        if template_pos[1] < top_bound:
            return [p for p in skel_list if p[1] < top_bound]
        if template_pos[1] > bot_bound:
            return [p for p in skel_list if p[1] > bot_bound]
        return [p for p in skel_list if top_bound <= p[1] <= bot_bound]

    def _get_endpoints_in_region(self, region: int) -> list:
        """Get all endpoints within a region.

        Args:
            region: Numpad region (1-9).

        Returns:
            List of endpoint coordinates in the region.
        """
        bbox = self.analysis.bbox
        endpoints = self.analysis.info['endpoints']
        return [ep for ep in endpoints if self._point_in_region(ep, region, bbox)]

    def _find_nearest_endpoint_in_region(self, region: int,
                                          template_pos: tuple[float, float]) -> tuple | None:
        """Find nearest endpoint within a region.

        Args:
            region: Numpad region (1-9).
            template_pos: Template position for distance calculation.

        Returns:
            Nearest endpoint coordinates or None if no endpoints in region.
        """
        endpoints_in_region = self._get_endpoints_in_region(region)
        if not endpoints_in_region:
            return None
        return min(endpoints_in_region, key=lambda p:
                   (p[0] - template_pos[0])**2 + (p[1] - template_pos[1])**2)

    def _find_region_fallback_extremum(self, region: int, region_pixels: list,
                                        template_pos: tuple[float, float],
                                        mid_x: float) -> tuple:
        """Find fallback extremum based on region position.

        Args:
            region: Numpad region (1-9).
            region_pixels: List of skeleton pixels in region.
            template_pos: Template position for corner regions.
            mid_x: X-coordinate of bounding box center.

        Returns:
            Extremum pixel coordinates.
        """
        is_corner = region in [7, 9, 1, 3]
        if is_corner:
            return min(region_pixels, key=lambda p:
                       abs(p[0] - template_pos[0]) + abs(p[1] - template_pos[1]))
        if region == 8:
            return min(region_pixels, key=lambda p: p[1])
        if region == 2:
            return max(region_pixels, key=lambda p: p[1])
        if template_pos[0] < mid_x:
            return min(region_pixels, key=lambda p: p[0])
        return max(region_pixels, key=lambda p: p[0])

    def _resolve_terminal(self, region: int, template_pos: tuple[float, float],
                          next_direction: str | None, mid_x: float,
                          top_bound: float, bot_bound: float) -> ResolvedWaypoint:
        """Resolve a terminal waypoint position.

        Handles waypoints that are not marked as intersection, vertex, or curve.
        These are typically stroke start/end points that should align with
        skeleton endpoints or extremum positions.

        Args:
            region: Numpad region (1-9) for this waypoint.
            template_pos: Template position as (x, y) pixel coordinates.
            next_direction: Direction to the next waypoint ('up', 'down',
                'left', 'right', or None).
            mid_x: X-coordinate of the bounding box center.
            top_bound: Y-coordinate of the top third boundary.
            bot_bound: Y-coordinate of the bottom third boundary.

        Returns:
            ResolvedWaypoint with the computed position.
        """
        region_pixels = self._get_region_pixels_with_fallback(
            region, template_pos, top_bound, bot_bound)

        if not region_pixels:
            pos = self.find_nearest_skeleton(template_pos)
            return ResolvedWaypoint(position=(float(pos[0]), float(pos[1])), region=region)

        is_corner = region in [7, 9, 1, 3]
        extremum = None

        # For corners, prefer endpoints by direction
        if is_corner:
            endpoints_in_region = self._get_endpoints_in_region(region)
            extremum = self._find_extremum_by_direction(endpoints_in_region, next_direction, template_pos)

        # For non-corners, try junction-based direction
        if extremum is None and next_direction and not is_corner:
            extremum = self._find_junction_for_direction(
                region_pixels, next_direction, template_pos, region, self.analysis.bbox)

        # Fallback: direction-based extremum from region pixels
        if extremum is None:
            extremum = self._find_extremum_by_direction(region_pixels, next_direction, template_pos)

        # Try nearest endpoint if still None
        if extremum is None:
            extremum = self._find_nearest_endpoint_in_region(region, template_pos)

        # Final fallback based on region position
        if extremum is None:
            extremum = self._find_region_fallback_extremum(
                region, region_pixels, template_pos, mid_x)

        return ResolvedWaypoint(position=(float(extremum[0]), float(extremum[1])), region=region)

    def _angle_in_direction_range(self, angle: float, direction: str) -> bool:
        """Check if an angle falls within the range for a direction.

        Args:
            angle: Segment angle in degrees.
            direction: Direction ('down', 'up', 'left', 'right').

        Returns:
            True if the angle is in the direction's range.
        """
        if direction == 'left':
            return angle > 135 or angle < -135
        low, high = ANGLE_RANGES[direction]
        return low <= angle <= high

    def _find_directional_segments(self, direction: str) -> list:
        """Find segments oriented in a specific direction with sufficient length.

        Args:
            direction: Desired direction ('down', 'up', 'left', 'right').

        Returns:
            List of segment dicts that match the direction criteria.
        """
        segments = self.analysis.segments
        return [seg for seg in segments
                if seg['length'] > DISTANCE_THRESHOLD_LARGE
                and self._angle_in_direction_range(seg['angle'], direction)]

    def _collect_junction_candidates_from_segments(self, good_segments: list,
                                                    junction_pixels: set,
                                                    region_pixels: list,
                                                    region: int, bbox: tuple) -> set:
        """Collect candidate junction pixels from segment start points.

        Args:
            good_segments: List of segments in the desired direction.
            junction_pixels: Set of known junction pixel coordinates.
            region_pixels: List of skeleton pixels in the region.
            region: Numpad region (1-9).
            bbox: Bounding box as (x_min, y_min, x_max, y_max).

        Returns:
            Set of candidate junction pixel coordinates.
        """
        candidates = set()
        for seg in good_segments:
            start = seg['start']
            if self._point_in_region(start, region, bbox):
                candidates.add(start)
            # Check junction pixels near segment start
            for jp in junction_pixels:
                if (abs(jp[0] - start[0]) <= DISTANCE_THRESHOLD_SMALL
                        and abs(jp[1] - start[1]) <= DISTANCE_THRESHOLD_SMALL
                        and (self._point_in_region(jp, region, bbox) or jp in region_pixels)):
                    candidates.add(jp)
        return candidates

    def _find_junction_pixels_near_segments(self, good_segments: list,
                                             junction_pixels: set,
                                             region_pixels: list,
                                             region: int, bbox: tuple) -> set:
        """Find junction pixels in region close to good segment starts.

        Args:
            good_segments: List of segments in the desired direction.
            junction_pixels: Set of known junction pixel coordinates.
            region_pixels: List of skeleton pixels in the region.
            region: Numpad region (1-9).
            bbox: Bounding box as (x_min, y_min, x_max, y_max).

        Returns:
            Set of candidate junction pixel coordinates.
        """
        candidates = set()
        for jp in junction_pixels:
            if jp not in region_pixels and not self._point_in_region(jp, region, bbox):
                continue
            for seg in good_segments:
                start = seg['start']
                if (abs(jp[0] - start[0]) <= DISTANCE_THRESHOLD_MEDIUM
                        and abs(jp[1] - start[1]) <= DISTANCE_THRESHOLD_MEDIUM):
                    candidates.add(jp)
                    break
        return candidates

    def _find_junction_for_direction(self, region_pixels: list, direction: str,
                                     template_pos: tuple[float, float],
                                     region: int, bbox: tuple) -> tuple | None:
        """Find a junction pixel that connects to a segment going in the specified direction.

        This method helps avoid picking stub endpoints when the stroke needs to
        continue in a specific direction. It finds junction pixels that are
        connected to segments oriented in the desired direction.

        Args:
            region_pixels: List of skeleton pixels in the target region.
            direction: Desired direction ('down', 'up', 'left', 'right').
            template_pos: Template position for distance-based tiebreaking.
            region: Numpad region (1-9).
            bbox: Bounding box as (x_min, y_min, x_max, y_max).

        Returns:
            Junction pixel coordinates (x, y) if found, None otherwise.

        Note:
            Direction-to-angle mapping:
                - 'down': 45 to 135 degrees
                - 'up': -135 to -45 degrees
                - 'right': -45 to 45 degrees
                - 'left': >135 or <-135 degrees (wraps around)
        """
        good_segments = self._find_directional_segments(direction)
        if not good_segments:
            return None

        junction_pixels = set(self.analysis.info.get('junction_pixels', []))

        # First pass: segment start points and nearby junctions
        candidates = self._collect_junction_candidates_from_segments(
            good_segments, junction_pixels, region_pixels, region, bbox)

        # Second pass: junction pixels in region near segment starts
        if not candidates:
            candidates = self._find_junction_pixels_near_segments(
                good_segments, junction_pixels, region_pixels, region, bbox)

        if not candidates:
            return None

        # Pick the junction closest to template position
        return min(candidates, key=lambda p:
                   (p[0] - template_pos[0])**2 + (p[1] - template_pos[1])**2)

    def _infer_direction(self, current_region: int, next_region: int) -> str | None:
        """Infer the direction of travel based on numpad region transition.

        Delegates to shared utility infer_direction_from_regions().
        See that function for full documentation.

        Args:
            current_region: Starting numpad region (1-9).
            next_region: Destination numpad region (1-9).

        Returns:
            Direction string ('down', 'up', 'left', 'right') for clear
            vertical/horizontal moves, or None for diagonal moves.
        """
        return infer_direction_from_regions(current_region, next_region)

    def _resolve_vertex(self, region: int, template_pos: tuple[float, float],
                        top_bound: float, bot_bound: float,
                        mid_x: float, mid_y: float) -> ResolvedWaypoint:
        """Resolve a vertex waypoint with optional apex extension.

        Handles vertex waypoints (marked with 'v' in templates), which represent
        extremum points like the top of an 'A' or bottom of a 'V'. May include
        an apex extension beyond the skeleton for sharp points.

        Args:
            region: Numpad region (1-9) for this waypoint.
            template_pos: Template position as (x, y) pixel coordinates.
            top_bound: Y-coordinate of the top third boundary.
            bot_bound: Y-coordinate of the bottom third boundary.
            mid_x: X-coordinate of the bounding box center.
            mid_y: Y-coordinate of the bounding box center.

        Returns:
            ResolvedWaypoint with is_vertex=True and optional apex_extension.
            The apex_extension is a tuple of (direction, (x, y)) where direction
            is 'top' or 'bottom'.

        Note:
            Apex extension is computed by finding glyph pixels that extend beyond
            the skeleton extremum, useful for rendering sharp points that the
            skeleton (medial axis) doesn't fully capture.
        """
        analysis = self.analysis
        bbox, skel_list, mask = analysis.bbox, analysis.skel_list, analysis.mask
        if bbox is None or len(bbox) < 4:
            # Fallback to nearest skeleton point if bbox is invalid
            nearest = self.find_nearest_skeleton((mid_x, mid_y))
            return ResolvedWaypoint(position=(float(nearest[0]), float(nearest[1])), region=region, is_vertex=True)
        rows, cols = analysis.glyph_rows, analysis.glyph_cols
        apex_extension = None

        # Top or bottom vertex: find extremum and check for apex
        is_top = template_pos[1] < top_bound
        is_bottom = template_pos[1] > bot_bound
        if (is_top or is_bottom) and skel_list:
            extremum = min(skel_list, key=lambda p: p[1]) if is_top else max(skel_list, key=lambda p: p[1])
            skel_x, skel_y = extremum[0], extremum[1]
            col_start, col_end = max(0, int(skel_x) - 5), min(mask.shape[1], int(skel_x) + 6)
            glyph_cols_filtered = cols[(cols >= col_start) & (cols < col_end)]
            glyph_rows_filtered = rows[(cols >= col_start) & (cols < col_end)]
            if len(glyph_rows_filtered) > 0:
                idx = glyph_rows_filtered.argmin() if is_top else glyph_rows_filtered.argmax()
                glyph_y, glyph_x = glyph_rows_filtered[idx], glyph_cols_filtered[idx]
                if (is_top and glyph_y < skel_y) or (is_bottom and glyph_y > skel_y):
                    apex_extension = ('top' if is_top else 'bottom', (float(glyph_x), float(glyph_y)))
            return ResolvedWaypoint(position=(float(skel_x), float(skel_y)),
                                    region=region, is_vertex=True, apex_extension=apex_extension)

        # Waist vertex: find leftmost/rightmost near middle
        waist_tolerance = (bbox[3] - bbox[1]) * WAIST_TOLERANCE_RATIO
        waist_pixels = [p for p in skel_list if abs(p[1] - mid_y) < waist_tolerance]
        if waist_pixels:
            vertex_pt = min(waist_pixels, key=lambda p: p[0]) if template_pos[0] < mid_x else max(waist_pixels, key=lambda p: p[0])
            return ResolvedWaypoint(position=(float(vertex_pt[0]), float(vertex_pt[1])), region=region, is_vertex=True)

        nearest = self.find_nearest_skeleton(template_pos)
        return ResolvedWaypoint(position=(float(nearest[0]), float(nearest[1])), region=region, is_vertex=True)

    def process_stroke_template(self, stroke_template: list,
                                global_traced: set[tuple[int, int]],
                                trace_paths: bool = True) -> tuple[list[list[float]], set[tuple[int, int]]] | None:
        """Process a single stroke template into stroke points.

        Converts a stroke template (list of waypoints) into a traced path of
        pixel coordinates. Handles special cases like vertical strokes and
        applies path tracing between resolved waypoints.

        Args:
            stroke_template: List of waypoint definitions. Each waypoint can be:
                - An integer (numpad region)
                - A tuple like (region, 'v') for vertex
                - A tuple like (region, 'i') for intersection
            global_traced: Set of already-traced pixels to avoid (not modified).
            trace_paths: If True, trace actual skeleton paths between waypoints.
                If False, return only the resolved waypoint positions.

        Returns:
            Tuple of (stroke_path, new_traced) where:
                - stroke_path: List of [x, y] coordinate pairs
                - new_traced: Updated set including newly traced pixels
            Returns None if the template could not be processed.

        Note:
            Vertical strokes (two waypoints in the same numpad column) are
            handled specially using segment detection for better results.
        """
        analysis = self.analysis
        if analysis is None:
            return None

        bbox = analysis.bbox
        if bbox is None or len(bbox) < 4:
            return None

        mid_x = (bbox[0] + bbox[2]) / 2
        mid_y = (bbox[1] + bbox[3]) / 2
        h = bbox[3] - bbox[1]
        third_h = h / 3
        top_bound = bbox[1] + third_h
        bot_bound = bbox[1] + 2 * third_h
        waist_margin = h * WAIST_MARGIN_RATIO

        if self._is_vertical_stroke(stroke_template):
            return self._process_vertical_stroke(stroke_template), global_traced

        waypoints, segment_configs = parse_stroke_template(stroke_template)
        if len(waypoints) < 2:
            return None

        resolved = []
        for i, wp in enumerate(waypoints):
            next_dir = segment_configs[i].direction if i < len(segment_configs) else None

            # Infer direction from next waypoint if not explicitly specified
            if next_dir is None and i + 1 < len(waypoints):
                next_dir = self._infer_direction(wp.region, waypoints[i + 1].region)

            resolved_wp = self.resolve_waypoint(wp, next_dir, mid_x, mid_y,
                                                top_bound, bot_bound, waist_margin)
            resolved.append(resolved_wp)

        if not trace_paths:
            return [[rw.position[0], rw.position[1]] for rw in resolved], global_traced

        return self._trace_resolved_waypoints(resolved, segment_configs, global_traced)

    def _is_vertical_stroke(self, stroke_template: list) -> bool:
        """Check if template represents a vertical line.

        Determines whether a stroke template defines a vertical stroke by
        checking if it has exactly two waypoints in the same numpad column.

        Args:
            stroke_template: List of waypoint definitions.

        Returns:
            True if the template represents a vertical stroke (two waypoints
            in the same column), False otherwise.
        """
        if len(stroke_template) != 2:
            return False

        r1 = extract_region_from_waypoint(stroke_template[0])
        r2 = extract_region_from_waypoint(stroke_template[1])
        if r1 is None or r2 is None:
            return False

        col1 = (r1 - 1) % 3
        col2 = (r2 - 1) % 3
        return col1 == col2

    def _process_vertical_stroke(self, stroke_template: list) -> list[list[float]]:
        """Process a vertical stroke using segment detection and path tracing.

        Specialized handling for vertical strokes that uses segment detection
        to find the best vertical skeleton segment, then traces a path along it.

        Args:
            stroke_template: Two-element list of waypoint definitions forming
                a vertical stroke.

        Returns:
            List of [x, y] coordinate pairs forming the stroke path. Returns
            a simple two-point line if tracing fails.
        """
        r1 = extract_region_from_waypoint(stroke_template[0])
        r2 = extract_region_from_waypoint(stroke_template[1])
        p1, p2 = self.numpad_to_pixel(r1), self.numpad_to_pixel(r2)
        seg = self.find_best_vertical_segment(p1, p2)
        start_pt = (int(round(seg[0][0])), int(round(seg[0][1]))) if seg else self.find_nearest_skeleton(p1)
        end_pt = (int(round(seg[1][0])), int(round(seg[1][1]))) if seg else self.find_nearest_skeleton(p2)
        # Trace the full path along skeleton
        info = self.analysis.info
        traced = self._trace_segment(start_pt, end_pt, SegmentConfig(), info['adj'], info['skel_set'], avoid_pixels=None)
        if traced and len(traced) > 2:
            resampled = self._resample_path(traced, num_points=min(30, len(traced)))
            return [[float(p[0]), float(p[1])] for p in resampled]
        return [[float(start_pt[0]), float(start_pt[1])], [float(end_pt[0]), float(end_pt[1])]]

    def _trace_single_segment(self, start_pt: tuple, end_pt: tuple,
                               config: SegmentConfig, resolved_i: ResolvedWaypoint,
                               resolved_next: ResolvedWaypoint, already_traced: set,
                               arrival_branch: set) -> tuple[list | None, set]:
        """Trace a single segment between two waypoints.

        Args:
            start_pt: Starting point.
            end_pt: Ending point.
            config: Segment configuration.
            resolved_i: Current waypoint.
            resolved_next: Target waypoint.
            already_traced: Set of already-traced pixels.
            arrival_branch: Set of arrival branch pixels.

        Returns:
            Tuple of (traced_path, updated_arrival_branch).
        """
        info = self.analysis.info

        if config.straight:
            return self._generate_straight_line(start_pt, end_pt), set()

        if resolved_next.is_intersection:
            traced = self._trace_segment(start_pt, end_pt, config, info['adj'],
                                         info['skel_set'], avoid_pixels=None)
            if traced and len(traced) >= 2:
                return traced, set(traced[-ARRIVAL_BRANCH_SIZE:])
            return traced, set()

        if resolved_i.is_intersection and arrival_branch:
            traced = self._trace_segment(start_pt, end_pt, config, info['adj'],
                                         info['skel_set'], avoid_pixels=already_traced,
                                         fallback_avoid=arrival_branch)
            return traced, set()

        if resolved_next.is_curve and resolved_next.region is not None:
            traced = self._trace_to_region(start_pt, resolved_next.region,
                                           self.analysis.bbox, info['adj'],
                                           info['skel_set'], avoid_pixels=already_traced)
            if traced is None:
                traced = self._trace_to_region(start_pt, resolved_next.region,
                                               self.analysis.bbox, info['adj'],
                                               info['skel_set'], avoid_pixels=None)
            return traced, arrival_branch

        traced = self._trace_segment(start_pt, end_pt, config, info['adj'],
                                     info['skel_set'], avoid_pixels=already_traced)
        return traced, arrival_branch

    def _apply_apex_extensions(self, full_path: list, resolved: list[ResolvedWaypoint]) -> None:
        """Insert apex extension points into the path.

        Args:
            full_path: Path to modify in place.
            resolved: List of resolved waypoints with potential apex extensions.
        """
        for rw in resolved:
            if rw.apex_extension:
                direction, apex_pt = rw.apex_extension
                sx, sy = int(round(rw.position[0])), int(round(rw.position[1]))
                for j, fp in enumerate(full_path):
                    if abs(fp[0] - sx) <= 2 and abs(fp[1] - sy) <= 2:
                        apex = (int(round(apex_pt[0])), int(round(apex_pt[1])))
                        full_path.insert(j if direction == 'top' else j + 1, apex)
                        break

    def _trace_resolved_waypoints(self, resolved: list[ResolvedWaypoint],
                                  segment_configs: list[SegmentConfig],
                                  global_traced: set[tuple[int, int]]) -> tuple[list[list[float]], set[tuple[int, int]]]:
        """Trace paths between resolved waypoints.

        Connects the sequence of resolved waypoints by tracing paths along
        the skeleton. Handles various waypoint types (intersection, curve)
        with appropriate tracing strategies and avoidance of previously
        traced pixels.

        Args:
            resolved: List of resolved waypoints with pixel positions.
            segment_configs: Configuration for each segment between waypoints.
            global_traced: Set of already-traced pixels to avoid (not modified).

        Returns:
            Tuple of (stroke_path, new_traced) where:
                - stroke_path: List of [x, y] coordinate pairs
                - new_traced: Updated set including newly traced pixels

        Note:
            The tracing logic handles several special cases:
            - Straight segments (config.straight=True): Direct line generation
            - Intersection targets: No pixel avoidance, records arrival branch
            - Leaving intersections: Avoids the arrival branch
            - Curve targets: Traces to region rather than specific point
        """
        full_path = []
        already_traced = set(global_traced)
        arrival_branch = set()

        current_pt = (int(round(resolved[0].position[0])),
                      int(round(resolved[0].position[1])))

        for i in range(len(resolved) - 1):
            start_pt = current_pt
            end_pt = (int(round(resolved[i+1].position[0])),
                      int(round(resolved[i+1].position[1])))

            config = segment_configs[i] if i < len(segment_configs) else SegmentConfig()

            traced, arrival_branch = self._trace_single_segment(
                start_pt, end_pt, config, resolved[i], resolved[i+1],
                already_traced, arrival_branch
            )

            if traced:
                if i == 0:
                    full_path.extend(traced)
                    already_traced.update(traced)
                else:
                    full_path.extend(traced[1:])
                    already_traced.update(traced[1:])
                current_pt = traced[-1]
            else:
                if i == 0:
                    full_path.append(start_pt)
                    already_traced.add(start_pt)
                break

        self._apply_apex_extensions(full_path, resolved)

        if len(full_path) > 3:
            full_path = self._resample_path(full_path, num_points=min(30, len(full_path)))

        # Return both the path and the new traced set (immutable approach)
        new_traced = global_traced | already_traced
        return [[float(p[0]), float(p[1])] for p in full_path], new_traced

    def run(self, template: list[list], trace_paths: bool = True) -> list[list[list[float]]] | None:
        """Run the full pipeline for a given template.

        Processes a complete stroke template (multiple strokes) and returns
        the traced paths for all strokes.

        Args:
            template: List of stroke templates, where each stroke template is
                a list of waypoint definitions.
            trace_paths: If True, trace actual skeleton paths between waypoints.
                If False, return only the resolved waypoint positions.

        Returns:
            List of strokes, where each stroke is a list of [x, y] coordinate
            pairs. Returns None if analysis fails or no valid strokes are produced.

        Example:
            >>> # Template for letter 'A': two diagonal strokes + crossbar
            >>> template = [
            ...     [(8, 'v'), 1],      # Left diagonal
            ...     [(8, 'v'), 3],      # Right diagonal
            ...     [4, 6],             # Crossbar
            ... ]
            >>> strokes = pipeline.run(template)
        """
        if self.analysis is None:
            return None

        strokes = []
        global_traced = set()

        for stroke_template in template:
            result = self.process_stroke_template(stroke_template, global_traced, trace_paths)
            if result is None:
                continue
            stroke_points, global_traced = result
            if stroke_points and len(stroke_points) >= 2:
                strokes.append(stroke_points)

        return strokes if strokes else None

    def try_skeleton_method(self) -> VariantResult:
        """Try pure skeleton method and return result with score.

        Delegates to stroke_variant_evaluator.try_skeleton_method().
        See that function for full documentation.

        Returns:
            VariantResult with strokes, score, and variant_name='skeleton'.
        """
        return try_skeleton_method(
            analysis=self.analysis,
            char=self.char,
            skeleton_to_strokes_fn=self._skeleton_to_strokes,
            apply_stroke_template_fn=self._apply_stroke_template,
            adjust_stroke_paths_fn=self._adjust_stroke_paths,
            quick_stroke_score_fn=self._quick_stroke_score,
        )

    def evaluate_all_variants(self) -> VariantResult:
        """Evaluate all template variants and skeleton, return best result.

        Delegates to VariantEvaluator for the evaluation logic.
        See VariantEvaluator.evaluate_all_variants() for full documentation.

        Returns:
            VariantResult with best strokes, score, and variant_name.
        """
        evaluator = VariantEvaluator(
            char=self.char,
            run_fn=self.run,
            try_skeleton_fn=self.try_skeleton_method,
            quick_score_fn=self._quick_stroke_score,
            get_mask_fn=lambda: self.analysis.mask if self.analysis else None,
        )
        return evaluator.evaluate_all_variants()
