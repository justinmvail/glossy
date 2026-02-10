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

from collections import defaultdict

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
from stroke_templates import NUMPAD_POS, NUMPAD_TEMPLATE_VARIANTS


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
        vertical_segments = [s for s in segments if 60 <= abs(s['angle']) <= 120]

        skel_list = list(info['skel_set'])
        if not skel_list:
            return None

        skel_tree = cKDTree(skel_list)

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
        frac_x, frac_y = NUMPAD_POS[region]
        x = bbox[0] + frac_x * (bbox[2] - bbox[0])
        y = bbox[1] + frac_y * (bbox[3] - bbox[1])
        return (x, y)

    def find_nearest_skeleton(self, pos: tuple[float, float]) -> tuple[int, int]:
        """Find the nearest skeleton pixel to a position.

        Uses a KD-tree for efficient O(log n) nearest-neighbor lookup.

        Args:
            pos: Target position as (x, y) coordinates.

        Returns:
            The (x, y) coordinates of the nearest skeleton pixel.

        Raises:
            AttributeError: If analysis has not been computed or is None.
        """
        _, idx = self.analysis.skel_tree.query(pos)
        return self.analysis.skel_list[idx]

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
        truly_vertical = [s for s in vertical_segments if 75 <= abs(s['angle']) <= 105] or vertical_segments
        # Build junction-to-segment map and find connected chains
        junc_segs = defaultdict(list)
        for i, seg in enumerate(truly_vertical):
            for j in [seg['start_junction'], seg['end_junction']]:
                if j >= 0:
                    junc_segs[j].append(i)
        visited, chains = set(), []
        for i in range(len(truly_vertical)):
            if i in visited:
                continue
            chain, queue = [], [i]
            while queue:
                idx = queue.pop(0)
                if idx in visited:
                    continue
                visited.add(idx)
                chain.append(idx)
                for j in [truly_vertical[idx]['start_junction'], truly_vertical[idx]['end_junction']]:
                    if j >= 0:
                        queue.extend(o for o in junc_segs[j] if o not in visited)
            if chain:
                chains.append(chain)
        # Score chains by distance to template positions
        def dist(p1, p2):
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        best, best_score = None, float('inf')
        for chain in chains:
            pts = sorted([p for i in chain for p in [truly_vertical[i]['start'], truly_vertical[i]['end']]], key=lambda p: p[1])
            if not pts:
                continue
            top, bot = pts[0], pts[-1]
            s1, s2 = dist(top, template_start) + dist(bot, template_end), dist(bot, template_start) + dist(top, template_end)
            score = min(s1, s2)
            if score < best_score:
                best_score, best = score, (top, bot) if s1 <= s2 else (bot, top)
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
        """
        analysis = self.analysis
        bbox = analysis.bbox
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

        Note:
            For corner regions (7, 9, 1, 3), the algorithm prefers skeleton
            endpoints and uses the next_direction to select appropriate
            extremum points for stroke ordering.
        """
        analysis = self.analysis
        bbox = analysis.bbox
        skel_list = analysis.skel_list
        info = analysis.info

        region_pixels = [p for p in skel_list if self._point_in_region(p, region, bbox)]

        if not region_pixels:
            if template_pos[1] < top_bound:
                region_pixels = [p for p in skel_list if p[1] < top_bound]
            elif template_pos[1] > bot_bound:
                region_pixels = [p for p in skel_list if p[1] > bot_bound]
            else:
                region_pixels = [p for p in skel_list if top_bound <= p[1] <= bot_bound]

        if not region_pixels:
            pos = self.find_nearest_skeleton(template_pos)
            return ResolvedWaypoint(position=(float(pos[0]), float(pos[1])), region=region)

        extremum = None
        is_corner = region in [7, 9, 1, 3]

        # For corner regions, prefer endpoints based on direction
        # (start from the extremum in the opposite direction to trace toward)
        if is_corner:
            endpoints_in_region = [ep for ep in info['endpoints'] if self._point_in_region(ep, region, bbox)]
            if endpoints_in_region:
                if next_direction == 'down':
                    # Going down: start from topmost endpoint
                    extremum = min(endpoints_in_region, key=lambda p: p[1])
                elif next_direction == 'up':
                    # Going up: start from bottommost endpoint
                    extremum = max(endpoints_in_region, key=lambda p: p[1])
                elif next_direction == 'left':
                    # Going left: start from rightmost endpoint
                    extremum = max(endpoints_in_region, key=lambda p: p[0])
                elif next_direction == 'right':
                    # Going right: start from leftmost endpoint
                    extremum = min(endpoints_in_region, key=lambda p: p[0])
                else:
                    # No direction: find endpoint closest to template position
                    extremum = min(endpoints_in_region, key=lambda p:
                                  (p[0] - template_pos[0])**2 + (p[1] - template_pos[1])**2)

        # For non-corner regions with direction, prefer junctions connecting to
        # segments going in that direction
        if extremum is None and next_direction in ('down', 'up', 'left', 'right') and not is_corner:
            extremum = self._find_junction_for_direction(
                region_pixels, next_direction, template_pos, region, bbox
            )

        # Fallback to direction-based extremum
        if extremum is None:
            if next_direction == 'down':
                extremum = min(region_pixels, key=lambda p: p[1])
            elif next_direction == 'up':
                extremum = max(region_pixels, key=lambda p: p[1])
            elif next_direction == 'left':
                extremum = max(region_pixels, key=lambda p: p[0])
            elif next_direction == 'right':
                extremum = min(region_pixels, key=lambda p: p[0])
            elif info['endpoints']:
                endpoints_in_region = [ep for ep in info['endpoints'] if self._point_in_region(ep, region, bbox)]
                if endpoints_in_region:
                    extremum = min(endpoints_in_region, key=lambda p:
                                  (p[0] - template_pos[0])**2 + (p[1] - template_pos[1])**2)

        if extremum is None:
            is_corner = region in [7, 9, 1, 3]
            if is_corner:
                extremum = min(region_pixels, key=lambda p:
                              abs(p[0] - template_pos[0]) + abs(p[1] - template_pos[1]))
            elif region == 8:
                extremum = min(region_pixels, key=lambda p: p[1])
            elif region == 2:
                extremum = max(region_pixels, key=lambda p: p[1])
            elif template_pos[0] < mid_x:
                extremum = min(region_pixels, key=lambda p: p[0])
            else:
                extremum = max(region_pixels, key=lambda p: p[0])

        return ResolvedWaypoint(position=(float(extremum[0]), float(extremum[1])), region=region)

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
        analysis = self.analysis
        segments = analysis.segments
        junction_pixels = set(analysis.info.get('junction_pixels', []))

        # Define angle ranges for each direction (segments store angle in degrees)
        # Angle 0 = right, 90 = down, 180/-180 = left, -90 = up
        angle_ranges = {
            'down': (45, 135),      # 45° to 135° (pointing downward)
            'up': (-135, -45),      # -135° to -45° (pointing upward)
            'right': (-45, 45),     # -45° to 45° (pointing right)
            'left': (135, 180, -180, -135),  # pointing left (wraps around)
        }

        def angle_matches(angle, direction):
            if direction == 'left':
                return angle > 135 or angle < -135
            low, high = angle_ranges[direction]
            return low <= angle <= high

        # Find segments that go in the desired direction and have length > 10
        # (to avoid tiny stubs)
        good_segments = []
        for seg in segments:
            if seg['length'] > 10 and angle_matches(seg['angle'], direction):
                good_segments.append(seg)

        if not good_segments:
            return None

        # Find junction pixels that are start points of these good segments
        candidate_junctions = set()
        for seg in good_segments:
            start = seg['start']
            # Check if start is in region or close to region
            if self._point_in_region(start, region, bbox):
                candidate_junctions.add(start)
            # Also check junction pixels near segment start
            for jp in junction_pixels:
                if abs(jp[0] - start[0]) <= 3 and abs(jp[1] - start[1]) <= 3:
                    if self._point_in_region(jp, region, bbox) or jp in region_pixels:
                        candidate_junctions.add(jp)

        if not candidate_junctions:
            # Try junction pixels in region that are close to good segment starts
            for jp in junction_pixels:
                if jp in region_pixels or self._point_in_region(jp, region, bbox):
                    for seg in good_segments:
                        start = seg['start']
                        if abs(jp[0] - start[0]) <= 5 and abs(jp[1] - start[1]) <= 5:
                            candidate_junctions.add(jp)
                            break

        if not candidate_junctions:
            return None

        # Pick the junction closest to template position
        return min(candidate_junctions, key=lambda p:
                   (p[0] - template_pos[0])**2 + (p[1] - template_pos[1])**2)

    def _infer_direction(self, current_region: int, next_region: int) -> str | None:
        """Infer the direction of travel based on numpad region transition.

        Determines the cardinal direction (up, down, left, right) when moving
        from one numpad region to another.

        Args:
            current_region: Starting numpad region (1-9).
            next_region: Destination numpad region (1-9).

        Returns:
            Direction string ('down', 'up', 'left', 'right') for clear
            vertical/horizontal moves, or None for diagonal moves.

        Note:
            Numpad layout reference:
                7 | 8 | 9   (row 0)
                4 | 5 | 6   (row 1)
                1 | 2 | 3   (row 2)
        """
        # Get row and column for each region (0-indexed from top-left)
        def region_to_rc(r):
            row = 2 - (r - 1) // 3  # 7,8,9 -> row 0; 4,5,6 -> row 1; 1,2,3 -> row 2
            col = (r - 1) % 3       # 7,4,1 -> col 0; 8,5,2 -> col 1; 9,6,3 -> col 2
            return row, col

        r1, c1 = region_to_rc(current_region)
        r2, c2 = region_to_rc(next_region)

        dr = r2 - r1  # positive = down
        dc = c2 - c1  # positive = right

        # Only return direction for clear vertical/horizontal moves
        if abs(dr) > abs(dc) and dr != 0:
            return 'down' if dr > 0 else 'up'
        elif abs(dc) > abs(dr) and dc != 0:
            return 'right' if dc > 0 else 'left'

        return None

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
        waist_tolerance = (bbox[3] - bbox[1]) * 0.15
        waist_pixels = [p for p in skel_list if abs(p[1] - mid_y) < waist_tolerance]
        if waist_pixels:
            vertex_pt = min(waist_pixels, key=lambda p: p[0]) if template_pos[0] < mid_x else max(waist_pixels, key=lambda p: p[0])
            return ResolvedWaypoint(position=(float(vertex_pt[0]), float(vertex_pt[1])), region=region, is_vertex=True)

        nearest = self.find_nearest_skeleton(template_pos)
        return ResolvedWaypoint(position=(float(nearest[0]), float(nearest[1])), region=region, is_vertex=True)

    def process_stroke_template(self, stroke_template: list,
                                global_traced: set[tuple[int, int]],
                                trace_paths: bool = True) -> list[list[float]] | None:
        """Process a single stroke template into stroke points.

        Converts a stroke template (list of waypoints) into a traced path of
        pixel coordinates. Handles special cases like vertical strokes and
        applies path tracing between resolved waypoints.

        Args:
            stroke_template: List of waypoint definitions. Each waypoint can be:
                - An integer (numpad region)
                - A tuple like (region, 'v') for vertex
                - A tuple like (region, 'i') for intersection
            global_traced: Set of already-traced pixels to avoid (modified in place).
            trace_paths: If True, trace actual skeleton paths between waypoints.
                If False, return only the resolved waypoint positions.

        Returns:
            List of [x, y] coordinate pairs forming the stroke path, or None
            if the template could not be processed.

        Note:
            Vertical strokes (two waypoints in the same numpad column) are
            handled specially using segment detection for better results.
        """
        analysis = self.analysis
        if analysis is None:
            return None

        bbox = analysis.bbox
        mid_x = (bbox[0] + bbox[2]) / 2
        mid_y = (bbox[1] + bbox[3]) / 2
        h = bbox[3] - bbox[1]
        third_h = h / 3
        top_bound = bbox[1] + third_h
        bot_bound = bbox[1] + 2 * third_h
        waist_margin = h * 0.05

        if self._is_vertical_stroke(stroke_template):
            return self._process_vertical_stroke(stroke_template)

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
            return [[rw.position[0], rw.position[1]] for rw in resolved]

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

        def extract_region(wp):
            if isinstance(wp, tuple):
                return wp[0] if isinstance(wp[0], int) else None
            return wp if isinstance(wp, int) else None

        r1, r2 = extract_region(stroke_template[0]), extract_region(stroke_template[1])
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
        def extract_region(wp):
            if isinstance(wp, tuple):
                return wp[0] if isinstance(wp[0], int) else None
            return wp if isinstance(wp, int) else None

        r1, r2 = extract_region(stroke_template[0]), extract_region(stroke_template[1])
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

    def _trace_resolved_waypoints(self, resolved: list[ResolvedWaypoint],
                                  segment_configs: list[SegmentConfig],
                                  global_traced: set[tuple[int, int]]) -> list[list[float]]:
        """Trace paths between resolved waypoints.

        Connects the sequence of resolved waypoints by tracing paths along
        the skeleton. Handles various waypoint types (intersection, curve)
        with appropriate tracing strategies and avoidance of previously
        traced pixels.

        Args:
            resolved: List of resolved waypoints with pixel positions.
            segment_configs: Configuration for each segment between waypoints.
            global_traced: Set of already-traced pixels to avoid (modified in place).

        Returns:
            List of [x, y] coordinate pairs forming the complete stroke path.

        Note:
            The tracing logic handles several special cases:
            - Straight segments (config.straight=True): Direct line generation
            - Intersection targets: No pixel avoidance, records arrival branch
            - Leaving intersections: Avoids the arrival branch
            - Curve targets: Traces to region rather than specific point
        """
        analysis = self.analysis
        info = analysis.info

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
            current_is_intersection = resolved[i].is_intersection
            target_is_curve = resolved[i+1].is_curve
            target_is_intersection = resolved[i+1].is_intersection
            target_region = resolved[i+1].region

            if config.straight:
                traced = self._generate_straight_line(start_pt, end_pt)
            elif target_is_intersection:
                traced = self._trace_segment(start_pt, end_pt, config, info['adj'], info['skel_set'],
                                       avoid_pixels=None)
                if traced and len(traced) >= 2:
                    arrival_branch = set(traced[-ARRIVAL_BRANCH_SIZE:])
            elif current_is_intersection and arrival_branch:
                traced = self._trace_segment(start_pt, end_pt, config, info['adj'], info['skel_set'],
                                       avoid_pixels=already_traced, fallback_avoid=arrival_branch)
                arrival_branch = set()
            elif target_is_curve and target_region is not None:
                traced = self._trace_to_region(start_pt, target_region, analysis.bbox,
                                          info['adj'], info['skel_set'], avoid_pixels=already_traced)
                if traced is None:
                    traced = self._trace_to_region(start_pt, target_region, analysis.bbox,
                                              info['adj'], info['skel_set'], avoid_pixels=None)
            else:
                traced = self._trace_segment(start_pt, end_pt, config, info['adj'], info['skel_set'],
                                       avoid_pixels=already_traced)

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

        # Apply apex extensions
        for rw in resolved:
            if rw.apex_extension:
                direction, apex_pt = rw.apex_extension
                sx, sy = int(round(rw.position[0])), int(round(rw.position[1]))
                for j, fp in enumerate(full_path):
                    if abs(fp[0] - sx) <= 2 and abs(fp[1] - sy) <= 2:
                        apex = (int(round(apex_pt[0])), int(round(apex_pt[1])))
                        full_path.insert(j if direction == 'top' else j + 1, apex)
                        break
        global_traced.update(already_traced)
        if len(full_path) > 3:
            full_path = self._resample_path(full_path, num_points=min(30, len(full_path)))
        return [[float(p[0]), float(p[1])] for p in full_path]

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
            stroke_points = self.process_stroke_template(stroke_template, global_traced, trace_paths)
            if stroke_points and len(stroke_points) >= 2:
                strokes.append(stroke_points)

        return strokes if strokes else None

    def try_skeleton_method(self) -> VariantResult:
        """Try pure skeleton method and return result with score.

        Attempts to extract strokes directly from the skeleton structure without
        using templates. This method is useful for characters without predefined
        templates or as a fallback when template-based methods fail.

        Returns:
            VariantResult containing:
                - strokes: List of stroke paths, or None if extraction failed
                - score: Quality score (higher is better), -1 on failure
                - variant_name: Always 'skeleton' for this method

        Note:
            The method compares raw skeleton strokes against merged/adjusted
            strokes and keeps the version with the better score. This prevents
            over-merging from degrading stroke quality.
        """
        if self.analysis is None:
            return VariantResult(strokes=None, score=-1, variant_name='skeleton')

        mask = self.analysis.mask
        skel_strokes = self._skeleton_to_strokes(mask, min_stroke_len=5)

        if not skel_strokes:
            return VariantResult(strokes=None, score=-1, variant_name='skeleton')

        # Score raw strokes before any merging
        raw_score = self._quick_stroke_score(skel_strokes, mask)
        raw_strokes = skel_strokes

        # Apply merge/template adjustments
        skel_strokes = self._apply_stroke_template(skel_strokes, self.char)
        skel_strokes = self._adjust_stroke_paths(skel_strokes, self.char, mask)

        if not skel_strokes:
            return VariantResult(strokes=None, score=-1, variant_name='skeleton')

        merged_score = self._quick_stroke_score(skel_strokes, mask)

        # Keep raw strokes if merge degraded the score
        if merged_score < raw_score - 0.01:
            return VariantResult(strokes=raw_strokes, score=raw_score, variant_name='skeleton')

        return VariantResult(strokes=skel_strokes, score=merged_score, variant_name='skeleton')

    def evaluate_all_variants(self) -> VariantResult:
        """Evaluate all template variants and skeleton, return best result.

        Tries all predefined template variants for the character (if any exist)
        as well as the pure skeleton method. Returns the variant that produces
        the highest quality score.

        Returns:
            VariantResult containing:
                - strokes: Best stroke paths found, or None if all methods failed
                - score: Quality score of the best result
                - variant_name: Name of the winning variant ('skeleton' or template name)

        Note:
            Skeleton results are penalized if their stroke count differs from
            the expected count (based on template definitions). This helps
            prefer template-based results when they produce the correct
            number of strokes.

            When scores are tied, skeleton results are preferred since they
            are derived from the actual glyph structure rather than templates.
        """
        variants = NUMPAD_TEMPLATE_VARIANTS.get(self.char)

        if not variants:
            return self.try_skeleton_method()

        if self.analysis is None:
            return VariantResult(strokes=None, score=-1, variant_name=None)

        mask = self.analysis.mask
        best = VariantResult(strokes=None, score=-1, variant_name=None)

        for var_name, variant_template in variants.items():
            strokes = self.run(variant_template, trace_paths=True)
            if strokes:
                score = self._quick_stroke_score(strokes, mask)
                if score > best.score:
                    best = VariantResult(strokes=strokes, score=score, variant_name=var_name)

        skel_result = self.try_skeleton_method()
        if skel_result.strokes:
            expected_counts = [len(t) for t in variants.values()]
            if expected_counts:
                expected_count = min(expected_counts)
                if len(skel_result.strokes) != expected_count:
                    skel_result.score -= 0.3 * abs(len(skel_result.strokes) - expected_count)

            # Prefer skeleton when scores are tied (skeleton is derived from actual glyph)
            if skel_result.score >= best.score:
                best = skel_result

        return best
