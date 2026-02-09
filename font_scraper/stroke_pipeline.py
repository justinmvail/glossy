"""MinimalStrokePipeline: converts font glyphs to stroke representations via skeleton tracing."""

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

    Stages:
    1. analyze() - Render mask and analyze skeleton
    2. resolve_waypoints() - Find pixel positions for template waypoints
    3. trace_paths() - Connect waypoints along skeleton
    4. evaluate_variant() - Score a complete variant result
    """

    def __init__(self, font_path: str, char: str, canvas_size: int = 224,
                 resolve_font_path_fn=None, render_glyph_mask_fn=None,
                 analyze_skeleton_fn=None, find_skeleton_segments_fn=None,
                 point_in_region_fn=None, trace_segment_fn=None,
                 trace_to_region_fn=None, generate_straight_line_fn=None,
                 resample_path_fn=None, skeleton_to_strokes_fn=None,
                 apply_stroke_template_fn=None, adjust_stroke_paths_fn=None,
                 quick_stroke_score_fn=None):
        """Initialize the pipeline with function dependencies."""
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
        """Lazy-load skeleton analysis."""
        if self._analysis is None:
            self._analysis = self._do_analyze()
        return self._analysis

    def _do_analyze(self) -> SkeletonAnalysis | None:
        """Stage 1: Render mask and analyze skeleton."""
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
        """Map numpad region (1-9) to pixel coordinates."""
        bbox = self.analysis.bbox
        frac_x, frac_y = NUMPAD_POS[region]
        x = bbox[0] + frac_x * (bbox[2] - bbox[0])
        y = bbox[1] + frac_y * (bbox[3] - bbox[1])
        return (x, y)

    def find_nearest_skeleton(self, pos: tuple[float, float]) -> tuple[int, int]:
        """Find the nearest skeleton pixel to a position."""
        _, idx = self.analysis.skel_tree.query(pos)
        return self.analysis.skel_list[idx]

    def find_best_vertical_segment(self, template_start: tuple[float, float],
                                   template_end: tuple[float, float]) -> tuple | None:
        """Find vertical skeleton segment(s) closest to template positions."""
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
        """Stage 2: Resolve a single waypoint to pixel coordinates."""
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
        """Resolve a terminal waypoint position."""
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

    def _resolve_vertex(self, region: int, template_pos: tuple[float, float],
                        top_bound: float, bot_bound: float,
                        mid_x: float, mid_y: float) -> ResolvedWaypoint:
        """Resolve a vertex waypoint with optional apex extension."""
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
        """Process a single stroke template into stroke points."""
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
            resolved_wp = self.resolve_waypoint(wp, next_dir, mid_x, mid_y,
                                                top_bound, bot_bound, waist_margin)
            resolved.append(resolved_wp)

        if not trace_paths:
            return [[rw.position[0], rw.position[1]] for rw in resolved]

        return self._trace_resolved_waypoints(resolved, segment_configs, global_traced)

    def _is_vertical_stroke(self, stroke_template: list) -> bool:
        """Check if template represents a vertical line."""
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
        """Process a vertical stroke using segment detection and path tracing."""
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
        """Trace paths between resolved waypoints."""
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
        """Run the full pipeline for a given template."""
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
        """Try pure skeleton method and return result with score."""
        if self.analysis is None:
            return VariantResult(strokes=None, score=-1, variant_name='skeleton')

        mask = self.analysis.mask
        skel_strokes = self._skeleton_to_strokes(mask, min_stroke_len=5)

        if not skel_strokes:
            return VariantResult(strokes=None, score=-1, variant_name='skeleton')

        skel_strokes = self._apply_stroke_template(skel_strokes, self.char)
        skel_strokes = self._adjust_stroke_paths(skel_strokes, self.char, mask)

        if not skel_strokes:
            return VariantResult(strokes=None, score=-1, variant_name='skeleton')

        score = self._quick_stroke_score(skel_strokes, mask)
        return VariantResult(strokes=skel_strokes, score=score, variant_name='skeleton')

    def evaluate_all_variants(self) -> VariantResult:
        """Evaluate all template variants and skeleton, return best result."""
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

            if skel_result.score > best.score:
                best = skel_result

        return best
