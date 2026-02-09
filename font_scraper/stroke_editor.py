#!/usr/bin/env python3
"""Stroke Editor - Web app for viewing and editing InkSight stroke data."""

import sqlite3
import json
import io
import base64
import os
import re
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional, List, Tuple, Set, Dict
from urllib.parse import quote as urlquote
from flask import Flask, render_template, request, jsonify, send_file, Response
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen
from inksight_vectorizer import InkSightVectorizer

# Import from refactored stroke_lib package
from stroke_lib.domain.geometry import (
    Point as SLPoint,
    BBox as SLBBox,
    Stroke as SLStroke,
    Segment as SLSegment,
)
from stroke_lib.domain.skeleton import SkeletonInfo as SLSkeletonInfo, Marker as SLMarker
from stroke_lib.analysis.skeleton import SkeletonAnalyzer as SLSkeletonAnalyzer
from stroke_lib.analysis.segments import SegmentClassifier as SLSegmentClassifier
from stroke_lib.templates.numpad import (
    NumpadTemplate as SLNumpadTemplate,
    NUMPAD_POS as SL_NUMPAD_POS,
    extract_region as sl_extract_region,
    is_vertical_stroke as sl_is_vertical_stroke,
)
from stroke_lib.utils.geometry import (
    point_in_region as sl_point_in_region,
    smooth_stroke as sl_smooth_stroke,
    resample_path as sl_resample_path,
    constrain_to_mask as sl_constrain_to_mask,
    generate_straight_line as sl_generate_straight_line,
)
from stroke_lib.utils.rendering import (
    render_glyph_mask as sl_render_glyph_mask,
    get_glyph_bbox as sl_get_glyph_bbox,
)
try:
    from docker.diffvg_docker import DiffVGDocker
    _diffvg_docker = DiffVGDocker()
except ImportError:
    _diffvg_docker = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
DB_PATH = os.path.join(BASE_DIR, 'fonts.db')


# --- Segment tracing configuration ---
@dataclass
class SegmentConfig:
    """Configuration for how to trace a segment between waypoints."""
    direction: Optional[str] = None  # 'down', 'up', 'left', 'right' - initial direction bias
    straight: bool = False           # True = draw direct line, False = follow skeleton


@dataclass
class ParsedWaypoint:
    """A parsed waypoint with its position and type info."""
    region: int                      # Numpad region 1-9
    is_curve: bool = False           # c(n) - smooth curve
    is_vertex: bool = False          # v(n) - sharp vertex
    is_intersection: bool = False    # i(n) - self-crossing point


# Constants for path tracing
DIRECTION_BIAS_PIXELS = 15    # Only apply direction bias for first N pixels
NEAR_ENDPOINT_DIST = 5        # Distance to consider "near" start/end for avoid_pixels
ARRIVAL_BRANCH_SIZE = 3       # Pixels to track as arrival branch at intersections

# Valid hint strings
DIRECTION_HINTS = {'down', 'up', 'left', 'right'}
STYLE_HINTS = {'straight'}
ALL_HINTS = DIRECTION_HINTS | STYLE_HINTS


# --- Pipeline data classes for minimal_strokes_from_skeleton ---

@dataclass
class SkeletonAnalysis:
    """Analyzed skeleton data for stroke generation."""
    mask: np.ndarray
    info: Dict  # skeleton info dict
    segments: List[Dict]  # classified segments
    vertical_segments: List[Dict]
    bbox: Tuple[float, float, float, float]
    skel_list: List[Tuple[int, int]]
    skel_tree: 'cKDTree'
    glyph_rows: np.ndarray  # row indices of glyph pixels
    glyph_cols: np.ndarray  # column indices of glyph pixels


@dataclass
class ResolvedWaypoint:
    """A waypoint with its resolved pixel position."""
    position: Tuple[float, float]
    region: int
    is_curve: bool = False
    is_vertex: bool = False
    is_intersection: bool = False
    apex_extension: Optional[Tuple[str, Tuple[float, float]]] = None  # ('top'/'bottom', (x, y))


@dataclass
class VariantResult:
    """Result of evaluating one template variant."""
    strokes: Optional[List[List[List[float]]]]
    score: float
    variant_name: str


class MinimalStrokePipeline:
    """Pipeline for generating minimal strokes from skeleton analysis.

    Stages:
    1. analyze() - Render mask and analyze skeleton
    2. resolve_waypoints() - Find pixel positions for template waypoints
    3. trace_paths() - Connect waypoints along skeleton
    4. evaluate_variant() - Score a complete variant result
    """

    def __init__(self, font_path: str, char: str, canvas_size: int = 224):
        self.font_path = resolve_font_path(font_path)
        self.char = char
        self.canvas_size = canvas_size
        self._analysis: Optional[SkeletonAnalysis] = None

    @property
    def analysis(self) -> Optional[SkeletonAnalysis]:
        """Lazy-load skeleton analysis."""
        if self._analysis is None:
            self._analysis = self._analyze()
        return self._analysis

    def _analyze(self) -> Optional[SkeletonAnalysis]:
        """Stage 1: Render mask and analyze skeleton."""
        mask = render_glyph_mask(self.font_path, self.char, self.canvas_size)
        if mask is None:
            return None

        rows, cols = np.where(mask)
        if len(rows) == 0:
            return None

        bbox = (float(cols.min()), float(rows.min()),
                float(cols.max()), float(rows.max()))

        info = _analyze_skeleton(mask)
        if info is None:
            return None

        segments = _find_skeleton_segments(info)
        vertical_segments = [s for s in segments if 60 <= abs(s['angle']) <= 120]

        skel_list = list(info['skel_set'])
        if not skel_list:
            return None

        skel_tree = cKDTree(skel_list)

        return SkeletonAnalysis(
            mask=mask,
            info=info,
            segments=segments,
            vertical_segments=vertical_segments,
            bbox=bbox,
            skel_list=skel_list,
            skel_tree=skel_tree,
            glyph_rows=rows,
            glyph_cols=cols,
        )

    def numpad_to_pixel(self, region: int) -> Tuple[float, float]:
        """Map numpad region (1-9) to pixel coordinates."""
        bbox = self.analysis.bbox
        frac_x, frac_y = NUMPAD_POS[region]
        x = bbox[0] + frac_x * (bbox[2] - bbox[0])
        y = bbox[1] + frac_y * (bbox[3] - bbox[1])
        return (x, y)

    def find_nearest_skeleton(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Find the nearest skeleton pixel to a position."""
        _, idx = self.analysis.skel_tree.query(pos)
        return self.analysis.skel_list[idx]

    def find_best_vertical_segment(self, template_start: Tuple[float, float],
                                   template_end: Tuple[float, float]) -> Optional[Tuple]:
        """Find vertical skeleton segment(s) closest to template positions."""
        vertical_segments = self.analysis.vertical_segments
        if not vertical_segments:
            return None

        # Filter to truly vertical segments
        truly_vertical = [s for s in vertical_segments if 75 <= abs(s['angle']) <= 105]
        if not truly_vertical:
            truly_vertical = vertical_segments

        # Build graph of connected vertical segments
        junction_to_segs = defaultdict(list)
        for i, seg in enumerate(truly_vertical):
            if seg['start_junction'] >= 0:
                junction_to_segs[seg['start_junction']].append(i)
            if seg['end_junction'] >= 0:
                junction_to_segs[seg['end_junction']].append(i)

        # Find chains by grouping connected segments
        visited = set()
        chains = []

        for i in range(len(truly_vertical)):
            if i in visited:
                continue
            chain = []
            queue = [i]
            while queue:
                seg_idx = queue.pop(0)
                if seg_idx in visited:
                    continue
                visited.add(seg_idx)
                chain.append(seg_idx)
                seg = truly_vertical[seg_idx]
                for junc in [seg['start_junction'], seg['end_junction']]:
                    if junc >= 0:
                        for other_idx in junction_to_segs[junc]:
                            if other_idx not in visited:
                                queue.append(other_idx)
            if chain:
                chains.append(chain)

        # Find best chain matching template
        best = None
        best_score = float('inf')

        for chain in chains:
            points = []
            for seg_idx in chain:
                seg = truly_vertical[seg_idx]
                points.append(seg['start'])
                points.append(seg['end'])

            if not points:
                continue

            points.sort(key=lambda p: p[1])
            top, bottom = points[0], points[-1]

            d1 = ((top[0] - template_start[0])**2 + (top[1] - template_start[1])**2)**0.5
            d2 = ((bottom[0] - template_end[0])**2 + (bottom[1] - template_end[1])**2)**0.5
            d1r = ((bottom[0] - template_start[0])**2 + (bottom[1] - template_start[1])**2)**0.5
            d2r = ((top[0] - template_end[0])**2 + (top[1] - template_end[1])**2)**0.5

            score = min(d1 + d2, d1r + d2r)
            if score < best_score:
                best_score = score
                best = (top, bottom) if d1 + d2 <= d1r + d2r else (bottom, top)

        return best

    def resolve_waypoint(self, wp: ParsedWaypoint, next_direction: Optional[str],
                         mid_x: float, mid_y: float, top_bound: float,
                         bot_bound: float, waist_margin: float) -> ResolvedWaypoint:
        """Stage 2: Resolve a single waypoint to pixel coordinates."""
        analysis = self.analysis
        bbox = analysis.bbox
        skel_list = analysis.skel_list
        info = analysis.info

        template_pos = self.numpad_to_pixel(wp.region)
        apex_extension = None

        if wp.is_intersection:
            # Find junction point in the region
            junction_pixels_in_region = [p for p in info.get('junction_pixels', [])
                                         if point_in_region(p, wp.region, bbox)]
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
            # Find extremum skeleton pixel for apex
            result = self._resolve_vertex(wp.region, template_pos, top_bound, bot_bound, mid_x, mid_y)
            return result

        if wp.is_curve:
            # Find curve apex
            if template_pos[1] < mid_y:
                region_pixels = [p for p in skel_list if p[1] < mid_y - waist_margin]
            else:
                region_pixels = [p for p in skel_list if p[1] > mid_y + waist_margin]

            if region_pixels:
                if template_pos[0] > mid_x:
                    apex = max(region_pixels, key=lambda p: p[0])
                elif template_pos[0] < mid_x:
                    apex = min(region_pixels, key=lambda p: p[0])
                else:
                    apex = self.find_nearest_skeleton(template_pos)
            else:
                apex = self.find_nearest_skeleton(template_pos)

            return ResolvedWaypoint(
                position=(float(apex[0]), float(apex[1])),
                region=wp.region, is_curve=True
            )

        # Terminal waypoint (plain int)
        return self._resolve_terminal(wp.region, template_pos, next_direction,
                                      mid_x, top_bound, bot_bound)

    def _resolve_terminal(self, region: int, template_pos: Tuple[float, float],
                          next_direction: Optional[str], mid_x: float,
                          top_bound: float, bot_bound: float) -> ResolvedWaypoint:
        """Resolve a terminal waypoint position."""
        analysis = self.analysis
        bbox = analysis.bbox
        skel_list = analysis.skel_list
        info = analysis.info

        region_pixels = [p for p in skel_list if point_in_region(p, region, bbox)]

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

        # Direction hint determines placement
        if next_direction == 'down':
            extremum = min(region_pixels, key=lambda p: p[1])
        elif next_direction == 'up':
            extremum = max(region_pixels, key=lambda p: p[1])
        elif next_direction == 'left':
            extremum = max(region_pixels, key=lambda p: p[0])
        elif next_direction == 'right':
            extremum = min(region_pixels, key=lambda p: p[0])
        elif info['endpoints']:
            endpoints_in_region = [ep for ep in info['endpoints'] if point_in_region(ep, region, bbox)]
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

    def _resolve_vertex(self, region: int, template_pos: Tuple[float, float],
                        top_bound: float, bot_bound: float,
                        mid_x: float, mid_y: float) -> ResolvedWaypoint:
        """Resolve a vertex waypoint with optional apex extension."""
        analysis = self.analysis
        bbox = analysis.bbox
        skel_list = analysis.skel_list
        mask = analysis.mask
        rows, cols = analysis.glyph_rows, analysis.glyph_cols

        apex_extension = None

        if template_pos[1] < top_bound:  # Top vertex
            if skel_list:
                topmost = min(skel_list, key=lambda p: p[1])
                skel_x, skel_y = topmost[0], topmost[1]

                col_start = max(0, int(skel_x) - 5)
                col_end = min(mask.shape[1], int(skel_x) + 6)
                glyph_cols_filtered = cols[(cols >= col_start) & (cols < col_end)]
                glyph_rows_filtered = rows[(cols >= col_start) & (cols < col_end)]

                if len(glyph_rows_filtered) > 0:
                    glyph_top_idx = glyph_rows_filtered.argmin()
                    glyph_top_y = glyph_rows_filtered[glyph_top_idx]
                    glyph_top_x = glyph_cols_filtered[glyph_top_idx]
                    if glyph_top_y < skel_y:
                        apex_extension = ('top', (float(glyph_top_x), float(glyph_top_y)))

                return ResolvedWaypoint(
                    position=(float(skel_x), float(skel_y)),
                    region=region, is_vertex=True, apex_extension=apex_extension
                )

        elif template_pos[1] > bot_bound:  # Bottom vertex
            if skel_list:
                bottommost = max(skel_list, key=lambda p: p[1])
                skel_x, skel_y = bottommost[0], bottommost[1]

                col_start = max(0, int(skel_x) - 5)
                col_end = min(mask.shape[1], int(skel_x) + 6)
                glyph_cols_filtered = cols[(cols >= col_start) & (cols < col_end)]
                glyph_rows_filtered = rows[(cols >= col_start) & (cols < col_end)]

                if len(glyph_rows_filtered) > 0:
                    glyph_bot_idx = glyph_rows_filtered.argmax()
                    glyph_bot_y = glyph_rows_filtered[glyph_bot_idx]
                    glyph_bot_x = glyph_cols_filtered[glyph_bot_idx]
                    if glyph_bot_y > skel_y:
                        apex_extension = ('bottom', (float(glyph_bot_x), float(glyph_bot_y)))

                return ResolvedWaypoint(
                    position=(float(skel_x), float(skel_y)),
                    region=region, is_vertex=True, apex_extension=apex_extension
                )

        else:  # Middle row - waist level
            waist_tolerance = (bbox[3] - bbox[1]) * 0.15
            waist_pixels = [p for p in skel_list if abs(p[1] - mid_y) < waist_tolerance]

            if waist_pixels:
                if template_pos[0] < mid_x:
                    vertex_pt = min(waist_pixels, key=lambda p: p[0])
                else:
                    vertex_pt = max(waist_pixels, key=lambda p: p[0])
                return ResolvedWaypoint(
                    position=(float(vertex_pt[0]), float(vertex_pt[1])),
                    region=region, is_vertex=True
                )

        # Fallback
        nearest = self.find_nearest_skeleton(template_pos)
        return ResolvedWaypoint(
            position=(float(nearest[0]), float(nearest[1])),
            region=region, is_vertex=True
        )

    def process_stroke_template(self, stroke_template: List,
                                global_traced: Set[Tuple[int, int]],
                                trace_paths: bool = True) -> Optional[List[List[float]]]:
        """Process a single stroke template into stroke points.

        Returns list of [x, y] points or None if processing failed.
        """
        analysis = self.analysis
        if analysis is None:
            return None

        bbox = analysis.bbox
        info = analysis.info
        skel_list = analysis.skel_list

        # Pre-calculate bbox-derived values
        mid_x = (bbox[0] + bbox[2]) / 2
        mid_y = (bbox[1] + bbox[3]) / 2
        h = bbox[3] - bbox[1]
        third_h = h / 3
        top_bound = bbox[1] + third_h
        bot_bound = bbox[1] + 2 * third_h
        waist_margin = h * 0.05

        # Check for vertical stroke special case
        if self._is_vertical_stroke(stroke_template):
            return self._process_vertical_stroke(stroke_template)

        # Parse template and resolve waypoints
        waypoints, segment_configs = parse_stroke_template(stroke_template)
        if len(waypoints) < 2:
            return None

        # Resolve each waypoint to pixel coordinates
        resolved = []
        for i, wp in enumerate(waypoints):
            next_dir = segment_configs[i].direction if i < len(segment_configs) else None
            resolved_wp = self.resolve_waypoint(wp, next_dir, mid_x, mid_y,
                                                top_bound, bot_bound, waist_margin)
            resolved.append(resolved_wp)

        if not trace_paths:
            return [[rw.position[0], rw.position[1]] for rw in resolved]

        # Trace paths between waypoints
        return self._trace_resolved_waypoints(resolved, segment_configs, global_traced)

    def _is_vertical_stroke(self, stroke_template: List) -> bool:
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

    def _process_vertical_stroke(self, stroke_template: List) -> List[List[float]]:
        """Process a vertical stroke using segment detection."""
        def extract_region(wp):
            if isinstance(wp, tuple):
                return wp[0] if isinstance(wp[0], int) else None
            return wp if isinstance(wp, int) else None

        r1, r2 = extract_region(stroke_template[0]), extract_region(stroke_template[1])
        p1, p2 = self.numpad_to_pixel(r1), self.numpad_to_pixel(r2)

        seg = self.find_best_vertical_segment(p1, p2)
        if seg:
            return [[float(seg[0][0]), float(seg[0][1])],
                    [float(seg[1][0]), float(seg[1][1])]]

        # Fallback
        n1 = self.find_nearest_skeleton(p1)
        n2 = self.find_nearest_skeleton(p2)
        return [[float(n1[0]), float(n1[1])], [float(n2[0]), float(n2[1])]]

    def _trace_resolved_waypoints(self, resolved: List[ResolvedWaypoint],
                                  segment_configs: List[SegmentConfig],
                                  global_traced: Set[Tuple[int, int]]) -> List[List[float]]:
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
                traced = _generate_straight_line(start_pt, end_pt)
            elif target_is_intersection:
                traced = trace_segment(start_pt, end_pt, config, info['adj'], info['skel_set'],
                                       avoid_pixels=None)
                if traced and len(traced) >= 2:
                    arrival_branch = set(traced[-ARRIVAL_BRANCH_SIZE:])
            elif current_is_intersection and arrival_branch:
                traced = trace_segment(start_pt, end_pt, config, info['adj'], info['skel_set'],
                                       avoid_pixels=already_traced, fallback_avoid=arrival_branch)
                arrival_branch = set()
            elif target_is_curve and target_region is not None:
                traced = _trace_to_region(start_pt, target_region, analysis.bbox,
                                          info['adj'], info['skel_set'], avoid_pixels=already_traced)
                if traced is None:
                    traced = _trace_to_region(start_pt, target_region, analysis.bbox,
                                              info['adj'], info['skel_set'], avoid_pixels=None)
            else:
                traced = trace_segment(start_pt, end_pt, config, info['adj'], info['skel_set'],
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
        for i, rw in enumerate(resolved):
            if rw.apex_extension:
                direction, apex_pt = rw.apex_extension
                skel_x, skel_y = int(round(rw.position[0])), int(round(rw.position[1]))

                for j, fp in enumerate(full_path):
                    if abs(fp[0] - skel_x) <= 2 and abs(fp[1] - skel_y) <= 2:
                        apex_tuple = (int(round(apex_pt[0])), int(round(apex_pt[1])))
                        if direction == 'top':
                            full_path.insert(j, apex_tuple)
                        else:
                            full_path.insert(j + 1, apex_tuple)
                        break

        # Update global traced set
        global_traced.update(already_traced)

        # Resample
        if len(full_path) > 3:
            resampled = _resample_path(full_path, num_points=min(30, len(full_path)))
            return [[float(p[0]), float(p[1])] for p in resampled]

        return [[float(p[0]), float(p[1])] for p in full_path]

    def run(self, template: List[List], trace_paths: bool = True) -> Optional[List[List[List[float]]]]:
        """Run the full pipeline for a given template.

        Args:
            template: List of stroke templates, each a list of waypoints
            trace_paths: Whether to trace paths between waypoints

        Returns:
            List of strokes, each a list of [x, y] points, or None on failure
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
        """Try pure skeleton method and return result with score."""
        if self.analysis is None:
            return VariantResult(strokes=None, score=-1, variant_name='skeleton')

        mask = self.analysis.mask
        skel_strokes = skeleton_to_strokes(mask, min_stroke_len=5)

        if not skel_strokes:
            return VariantResult(strokes=None, score=-1, variant_name='skeleton')

        skel_strokes = apply_stroke_template(skel_strokes, self.char)
        skel_strokes = adjust_stroke_paths(skel_strokes, self.char, mask)

        if not skel_strokes:
            return VariantResult(strokes=None, score=-1, variant_name='skeleton')

        score = _quick_stroke_score(skel_strokes, mask)
        return VariantResult(strokes=skel_strokes, score=score, variant_name='skeleton')

    def evaluate_all_variants(self) -> VariantResult:
        """Evaluate all template variants and skeleton, return best result."""
        variants = NUMPAD_TEMPLATE_VARIANTS.get(self.char)

        if not variants:
            # No templates - try skeleton only
            return self.try_skeleton_method()

        if self.analysis is None:
            return VariantResult(strokes=None, score=-1, variant_name=None)

        mask = self.analysis.mask
        best = VariantResult(strokes=None, score=-1, variant_name=None)

        # Try each template variant
        for var_name, variant_template in variants.items():
            strokes = self.run(variant_template, trace_paths=True)
            if strokes:
                score = _quick_stroke_score(strokes, mask)
                if score > best.score:
                    best = VariantResult(strokes=strokes, score=score, variant_name=var_name)

        # Try skeleton method
        skel_result = self.try_skeleton_method()
        if skel_result.strokes:
            # Penalize wrong stroke count
            expected_counts = [len(t) for t in variants.values()]
            if expected_counts:
                expected_count = min(expected_counts)
                if len(skel_result.strokes) != expected_count:
                    skel_result.score -= 0.3 * abs(len(skel_result.strokes) - expected_count)

            if skel_result.score > best.score:
                best = skel_result

        return best


def point_in_region(point: Tuple[int, int], region: int, bbox: Tuple[int, int, int, int]) -> bool:
    """Check if a point falls within a numpad region.

    Delegates to stroke_lib.utils.geometry.point_in_region.
    """
    return sl_point_in_region(point, region, bbox)


def parse_stroke_template(stroke_template: List) -> Tuple[List[ParsedWaypoint], List[SegmentConfig]]:
    """Parse a stroke template into waypoints and segment configurations.

    Args:
        stroke_template: Raw template like [9, 8, 7, 'straight', 4, 'i(5)', 6]

    Returns:
        Tuple of:
        - List of ParsedWaypoint (one per actual waypoint, hints excluded)
        - List of SegmentConfig (one per segment, len = len(waypoints) - 1)
    """
    waypoints = []
    segment_configs = []
    pending_config = SegmentConfig()

    for item in stroke_template:
        # Handle tuples (legacy format)
        if isinstance(item, tuple):
            item = item[0]

        # Check if it's a hint
        if isinstance(item, str) and item in ALL_HINTS:
            if item in DIRECTION_HINTS:
                pending_config.direction = item
            elif item == 'straight':
                pending_config.straight = True
            continue

        # Parse waypoint type
        wp = ParsedWaypoint(region=0)

        if isinstance(item, int):
            wp.region = item
        else:
            # Try v(n), c(n), i(n) patterns
            m = re.match(r'^v\((\d)\)$', str(item))
            if m:
                wp.region = int(m.group(1))
                wp.is_vertex = True
            else:
                m = re.match(r'^c\((\d)\)$', str(item))
                if m:
                    wp.region = int(m.group(1))
                    wp.is_curve = True
                else:
                    m = re.match(r'^i\((\d)\)$', str(item))
                    if m:
                        wp.region = int(m.group(1))
                        wp.is_intersection = True
                    else:
                        continue  # Unknown format, skip

        waypoints.append(wp)

        # Store the pending config for the segment ENDING at this waypoint
        # (First waypoint has no preceding segment)
        if len(waypoints) > 1:
            segment_configs.append(pending_config)
            pending_config = SegmentConfig()

    return waypoints, segment_configs


@app.template_filter('urlencode')
def urlencode_filter(s):
    return urlquote(str(s), safe='')


# --- Letter stroke templates ---
# Each letter maps to a list of strokes. Each stroke is a tuple of regions
# it passes through: (start, end) or (start, via, end).
# Regions use a 3x3 grid: TL TC TR / ML MC MR / BL BC BR
# Shared endpoint regions between strokes indicate junctions.
# A 'via' region distinguishes strokes that share start+end but take
# different paths (e.g. B's vertical vs 3-shape both go TL→BL).
LETTER_TEMPLATES = {
    # --- Uppercase ---
    'A': [('TC', 'BL'), ('TC', 'BR'), ('ML', 'MR')],
    'B': [('TL', 'BL'), ('TL', 'MR', 'BL')],
    'C': [('TR', 'BR')],
    'D': [('TL', 'BL'), ('TL', 'MR', 'BL')],
    'E': [('TL', 'BL'), ('TL', 'TR'), ('ML', 'MR'), ('BL', 'BR')],
    'F': [('TL', 'BL'), ('TL', 'TR'), ('ML', 'MR')],
    'G': [('TR', 'MR'), ('MR', 'MC')],
    'H': [('TL', 'BL'), ('TR', 'BR'), ('ML', 'MR')],
    'I': [('TC', 'BC')],
    'J': [('TR', 'BC')],
    'K': [('TL', 'BL'), ('TR', 'ML'), ('ML', 'BR')],
    'L': [('TL', 'BL'), ('BL', 'BR')],
    'M': [('BL', 'TL', 'BC'), ('BC', 'TR', 'BR')],
    'N': [('TL', 'BL'), ('TL', 'BR'), ('TR', 'BR')],
    'O': [('TC', 'ML', 'BC'), ('TC', 'MR', 'BC')],
    'P': [('TL', 'BL'), ('TL', 'MR', 'ML')],
    'Q': [('TC', 'ML', 'BC'), ('TC', 'MR', 'BC'), ('MC', 'BR')],
    'R': [('TL', 'BL'), ('TL', 'MR', 'ML'), ('ML', 'BR')],
    'S': [('TR', 'BL')],
    'T': [('TL', 'TR'), ('TC', 'BC')],
    'U': [('TL', 'BC', 'TR')],
    'V': [('TL', 'BC'), ('BC', 'TR')],
    'W': [('TL', 'BL'), ('BL', 'TC'), ('TC', 'BR'), ('BR', 'TR')],
    'X': [('TL', 'BR'), ('TR', 'BL')],
    'Y': [('TL', 'MC'), ('TR', 'MC'), ('MC', 'BC')],
    'Z': [('TL', 'TR'), ('TR', 'BL'), ('BL', 'BR')],

    # --- Lowercase ---
    'a': [('TR', 'BR'), ('MR', 'ML', 'BC', 'MR')],
    'b': [('TL', 'BL'), ('BL', 'MR', 'BL')],
    'c': [('TR', 'BR')],
    'd': [('TR', 'BR'), ('BR', 'ML', 'TR')],
    'e': [('MR', 'ML', 'BC', 'MR')],
    'f': [('TR', 'BC'), ('ML', 'MR')],
    'g': [('TR', 'MR', 'ML', 'BC', 'TR'), ('TR', 'BR')],
    'h': [('TL', 'BL'), ('ML', 'MR', 'BR')],
    'i': [('TC', 'BC')],
    'j': [('TC', 'BC', 'BL')],
    'k': [('TL', 'BL'), ('TR', 'ML'), ('ML', 'BR')],
    'l': [('TC', 'BC')],
    'm': [('TL', 'BL'), ('TL', 'MC', 'BC'), ('MC', 'TR', 'BR')],
    'n': [('TL', 'BL'), ('TL', 'TR', 'BR')],
    'o': [('TC', 'ML', 'BC'), ('TC', 'MR', 'BC')],
    'p': [('TL', 'BL'), ('TL', 'MR', 'TL')],
    'q': [('TR', 'ML', 'TR'), ('TR', 'BR')],
    'r': [('TL', 'BL'), ('TL', 'TR')],
    's': [('TR', 'BL')],
    't': [('TC', 'BC'), ('ML', 'MR')],
    'u': [('TL', 'BC', 'BR'), ('TR', 'BR')],
    'v': [('TL', 'BC'), ('BC', 'TR')],
    'w': [('TL', 'BL'), ('BL', 'MC'), ('MC', 'BR'), ('BR', 'TR')],
    'x': [('TL', 'BR'), ('TR', 'BL')],
    'y': [('TL', 'MC'), ('TR', 'MC'), ('MC', 'BL')],
    'z': [('TL', 'TR'), ('TR', 'BL'), ('BL', 'BR')],

    # --- Digits ---
    '0': [('TC', 'ML', 'BC'), ('TC', 'MR', 'BC')],
    '1': [('TC', 'BC')],
    '2': [('TL', 'TR', 'MR', 'BL', 'BR')],
    '3': [('TL', 'MR', 'ML'), ('ML', 'MR', 'BL')],
    '4': [('TL', 'ML'), ('ML', 'MR'), ('TR', 'BR')],
    '5': [('TR', 'TL', 'ML'), ('ML', 'MR', 'BC', 'BL')],
    '6': [('TR', 'TL', 'BL', 'BC', 'MR', 'ML')],
    '7': [('TL', 'TR', 'BL')],
    '8': [('MC', 'TR', 'TL', 'MC'), ('MC', 'BL', 'BR', 'MC')],
    '9': [('MR', 'TL', 'TC', 'MR'), ('MR', 'BL')],
}

# --- Numpad-grid stroke templates ---
# Numpad positions:
#   7  8  9
#   4  5  6
#   1  2  3
# Waypoint types:
#   int        → termination (stroke starts/ends at glyph edge near region)
#   'v(n)'     → sharp vertex (abrupt direction change)
#   'c(n)'     → smooth curve vertex (smooth direction change)
#   'i(n)'     → intersection (go straight through, for self-crossing strokes)
#   'down'/'up'/'left'/'right' → direction hint for next segment path tracing
#   'straight' → prefer paths with no sharp direction changes
# Format: position alone (7) or tuple (deprecated, hints removed)
#
# NUMPAD_TEMPLATE_VARIANTS: Maps character -> dict of variant_name -> template
# Each character can have multiple valid stroke patterns. The system will try
# each variant and keep the best-scoring one.

NUMPAD_TEMPLATE_VARIANTS = {
    # --- Uppercase ---
    'A': {
        'pointed': [[1, 'v(8)', 3], [4, 6]],
        'flat_top': [[1, 7, 9, 3], [4, 6]],
    },
    'B': {
        'default': [[7, 1], [7, 'c(9)', 'v(6)', 'c(3)', 1]],
    },
    'C': {
        'default': [[9, 'c(7)', 'c(1)', 3]],
    },
    'D': {
        'default': [[7, 1], [7, 'c(9)', 'c(3)', 1]],
    },
    'E': {
        'default': [[9, 7, 1, 3], [4, 6]],
    },
    'F': {
        'default': [[9, 7, 1], [4, 6]],
    },
    'G': {
        'default': [[9, 'c(7)', 'c(1)', 3, 6], [6, 5]],
    },
    'H': {
        'default': [[7, 1], [9, 3], [4, 6]],
    },
    'I': {
        'default': [[8, 2]],
    },
    'J': {
        'default': [[8, 'c(1)']],
    },
    'K': {
        'default': [[7, 1], [9, 'v(4)', 3]],
    },
    'L': {
        'default': [[7, 1, 3]],
    },
    'M': {
        'default': [[1, 7, 'v(2)', 9, 3]],
    },
    'N': {
        'default': [[1, 7, 3, 9]],
    },
    'O': {
        'closed': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8]],
    },
    'P': {
        'default': [[1, 7, 'c(9)', 'c(6)', 4]],
    },
    'Q': {
        'default': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8], [5, 3]],
    },
    'R': {
        'default': [[1, 7, 'c(9)', 'c(6)', 4], [4, 3]],
    },
    'S': {
        'default': [[9, 'c(7)', 'c(4)', 'c(6)', 'c(3)', 1]],
    },
    'T': {
        'default': [[7, 9], [8, 2]],
    },
    'U': {
        'default': [[7, 'c(1)', 'c(3)', 9]],
    },
    'V': {
        'default': [[7, 'v(2)', 9]],
    },
    'W': {
        'default': [[7, 'v(1)', 'v(5)', 'v(3)', 9]],
    },
    'X': {
        'default': [[7, 3], [9, 1]],
    },
    'Y': {
        'default': [[7, 'v(5)', 9], [5, 2]],
    },
    'Z': {
        'default': [[7, 9, 1, 3]],
    },

    # --- Lowercase ---
    'a': {
        'default': [[9, 'c(7)', 'c(1)', 3], [9, 3]],
    },
    'b': {
        'default': [[7, 1, 'c(3)', 'c(9)', 4]],
    },
    'c': {
        'default': [[9, 'c(7)', 'c(1)', 3]],
    },
    'd': {
        'default': [[9, 'c(7)', 'c(1)', 3], [9, 3]],
    },
    'e': {
        'default': [[6, 4, 'c(1)', 'c(3)', 'c(6)']],
    },
    'f': {
        'default': [[9, 'c(8)', 2], [4, 6]],
    },
    'g': {
        'default': [[9, 'c(7)', 'c(1)', 'c(3)', 9], [9, 'c(3)']],
    },
    'h': {
        'default': [[7, 1], [4, 'c(9)', 3]],
    },
    'i': {
        'default': [[8, 2]],
    },
    'j': {
        'default': [[8, 'c(1)']],
    },
    'k': {
        'default': [[7, 1], [9, 'v(4)', 3]],
    },
    'l': {
        'default': [[8, 2]],
    },
    'm': {
        'default': [[1, 4, 'c(8)', 5], [5, 'c(9)', 3]],
    },
    'n': {
        'default': [[1, 4, 'c(9)', 3]],
    },
    'o': {
        'closed': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8]],
    },
    'p': {
        'default': [[4, 'c(7)', 'c(9)', 'c(6)', 1]],
    },
    'q': {
        'default': [[9, 'c(7)', 'c(1)', 'c(3)', 9], [9, 3]],
    },
    'r': {
        'default': [[1, 4, 'c(9)']],
    },
    's': {
        'default': [[9, 'c(7)', 'c(4)', 'c(6)', 'c(3)', 1]],
    },
    't': {
        'default': [[8, 2], [7, 9]],
    },
    'u': {
        'default': [[7, 'c(1)', 'c(3)', 9], [9, 3]],
    },
    'v': {
        'default': [[7, 'v(2)', 9]],
    },
    'w': {
        'default': [[7, 'v(1)', 'v(5)', 'v(3)', 9]],
    },
    'x': {
        'default': [[7, 3], [9, 1]],
    },
    'y': {
        'default': [[7, 'v(5)'], [9, 'v(5)', 'c(1)']],
    },
    'z': {
        'default': [[7, 9, 1, 3]],
    },

    # --- Digits ---
    '0': {
        'closed': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8]],
        'open': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)']],  # Incomplete loop from top
        'ccw_open': [[7, 'c(4)', 'c(1)', 'c(2)', 'c(3)', 'c(6)', 'c(9)', 'c(8)', 'c(7)', 4]],  # Full CCW loop: 7-4-1-2-3-6-9-8-7-4
    },
    '1': {
        'with_serif': [[7, 8, 2]],
        'simple': [[8, 2]],
        'with_base': [[4, 'v(8)', 2], [1, 2, 3]],  # Diagonal (4→8→2) + horizontal base
    },
    '2': {
        'default': [[7, 'v(4)', 1, 3]],  # Endpoints: 7 -> junction at 4 -> 1 or 3
    },
    '3': {
        'default': [[7, 'c(9)', 'v(4)', 'c(3)', 1]],
    },
    '4': {
        'open': [[7, 'v(4)', 6], [9, 3]],
        'closed': [[7, 'v(4)', 6, 9], [9, 3]],
    },
    '5': {
        'default': [[9, 7, 4, 'c(6)', 'c(3)', 1]],
        'two_stroke': [[9, 8, 7], [7, 4, 5, 6, 3, 2, 1]],
    },
    '6': {
        'default': [[9, 'c(7)', 'c(1)', 'c(3)', 'c(6)', 4]],
        'end_at_1': [[9, 'c(7)', 'c(1)', 'c(3)', 'c(6)', 1]],
    },
    '7': {
        'default': [[7, 9, 1]],
    },
    '8': {
        'default': [[8, 'c(7)', 'c(4)', 'c(1)', 'c(2)', 'c(3)', 'c(6)', 'c(9)', 8]],
        'from_9': [[9, 8, 7, 4, 'straight', 'i(5)', 'straight', 6, 3, 2, 1, 'i(5)', 9]],
    },
    '9': {
        'default': [[6, 'c(9)', 'c(8)', 'c(7)', 'c(4)', 6], [6, 'c(3)', 1]],
    },
}

# Legacy compatibility: NUMPAD_TEMPLATES returns first variant for each char
NUMPAD_TEMPLATES = {
    char: list(variants.values())[0]
    for char, variants in NUMPAD_TEMPLATE_VARIANTS.items()
}

# Numpad region positions as (col_fraction, row_fraction) within glyph bbox.
# (0,0) = top-left, (1,1) = bottom-right
NUMPAD_POS = {
    7: (0.0, 0.0),
    8: (0.5, 0.0),
    9: (1.0, 0.0),
    4: (0.0, 0.5),
    5: (0.5, 0.5),
    6: (1.0, 0.5),
    1: (0.0, 1.0),
    2: (0.5, 1.0),
    3: (1.0, 1.0),
}


def _parse_waypoint(wp):
    """Parse a waypoint into (region_int, kind).

    Returns:
        (region, kind) where region is 1-9 and kind is 'terminal', 'vertex', or 'curve'.
    """
    if isinstance(wp, int):
        return (wp, 'terminal')
    m = re.match(r'^v\((\d)\)$', str(wp))
    if m:
        return (int(m.group(1)), 'vertex')
    m = re.match(r'^c\((\d)\)$', str(wp))
    if m:
        return (int(m.group(1)), 'curve')
    raise ValueError(f"Unknown waypoint format: {wp}")


def _numpad_to_pixel(region, glyph_bbox):
    """Map a numpad region (1-9) to pixel coordinates within the glyph bounding box.

    glyph_bbox: (x_min, y_min, x_max, y_max) in pixel space.
    """
    frac_x, frac_y = NUMPAD_POS[region]
    x_min, y_min, x_max, y_max = glyph_bbox
    return (x_min + frac_x * (x_max - x_min),
            y_min + frac_y * (y_max - y_min))


def _snap_to_glyph_edge(pos, centroid, mask):
    """Snap a termination point to the nearest mask pixel.

    Fallback for when skeleton-based snapping isn't available.
    """
    h, w = mask.shape
    ix = int(round(min(max(pos[0], 0), w - 1)))
    iy = int(round(min(max(pos[1], 0), h - 1)))
    if mask[iy, ix]:
        return pos
    dist_out, indices = distance_transform_edt(~mask, return_indices=True)
    ny = float(indices[0, iy, ix])
    nx = float(indices[1, iy, ix])
    nix, niy = int(round(nx)), int(round(ny))
    if 0 <= nix < w and 0 <= niy < h and mask[niy, nix]:
        return (nx, ny)
    return None


def _find_skeleton_waypoints(mask, glyph_bbox):
    """Find skeleton endpoints and junctions as candidate waypoint positions.

    Returns dict mapping numpad region (1-9) to list of (x, y) skeleton
    feature positions in that region, sorted by distance from region center.
    """
    skel = skeletonize(mask)
    ys, xs = np.where(skel)
    if len(xs) == 0:
        return None

    skel_set = set(zip(xs.tolist(), ys.tolist()))

    # Build adjacency
    adj = defaultdict(list)
    for (x, y) in skel_set:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                n = (x + dx, y + dy)
                if n in skel_set:
                    adj[(x, y)].append(n)

    # For each numpad region, find the best skeleton pixel.
    # Balance proximity to the region center with being well-centered
    # in the stroke (high distance transform = thick part of stroke).
    skel_list = list(skel_set)
    dist_in = distance_transform_edt(mask)
    max_dist = float(dist_in.max()) if dist_in.max() > 0 else 1.0

    # Diagonal of bbox for normalizing distances
    x_min, y_min, x_max, y_max = glyph_bbox
    bbox_diag = max(((x_max - x_min) ** 2 + (y_max - y_min) ** 2) ** 0.5, 1.0)

    region_features = {}
    for r in range(1, 10):
        rc = _numpad_to_pixel(r, glyph_bbox)

        def score(p, rc=rc):
            dx = p[0] - rc[0]
            dy = p[1] - rc[1]
            proximity = (dx * dx + dy * dy) ** 0.5 / bbox_diag  # 0-1
            depth = dist_in[p[1], p[0]] / max_dist  # 0-1, higher = more centered
            # Lower score = better. Penalize distance, reward depth.
            return proximity - 0.3 * depth

        best = min(skel_list, key=score)
        region_features[r] = [best]

    region_features['all_skel'] = skel_list
    return region_features


def _linear_segment(p0, p1, step=2.0):
    """Generate evenly-spaced points along a line from p0 to p1."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dist = (dx * dx + dy * dy) ** 0.5
    n = max(2, int(round(dist / step)))
    return [(p0[0] + dx * i / (n - 1), p0[1] + dy * i / (n - 1)) for i in range(n)]


def _catmull_rom_point(p0, p1, p2, p3, t, alpha=0.5):
    """Evaluate a single point on a Catmull-Rom spline segment."""
    def tj(ti, pi, pj):
        dx = pj[0] - pi[0]
        dy = pj[1] - pi[1]
        d = (dx * dx + dy * dy) ** 0.5
        return ti + max(d ** alpha, 1e-6)

    t0 = 0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)

    u = t1 + t * (t2 - t1)

    def lerp(a, b, ta, tb, u_):
        f = (u_ - ta) / max(tb - ta, 1e-10)
        return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))

    a1 = lerp(p0, p1, t0, t1, u)
    a2 = lerp(p1, p2, t1, t2, u)
    a3 = lerp(p2, p3, t2, t3, u)
    b1 = lerp(a1, a2, t0, t2, u)
    b2 = lerp(a2, a3, t1, t3, u)
    c = lerp(b1, b2, t1, t2, u)
    return c


def _catmull_rom_segment(p_prev, p0, p1, p_next, step=2.0):
    """Generate evenly-spaced points along a Catmull-Rom segment from p0 to p1."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dist = (dx * dx + dy * dy) ** 0.5
    n = max(2, int(round(dist / step)))
    return [_catmull_rom_point(p_prev, p0, p1, p_next, i / (n - 1)) for i in range(n)]


def _snap_inside(pos, mask, snap_indices):
    """Snap a position to the nearest mask pixel if outside.

    Args:
        pos: (x, y) position to snap
        mask: Binary mask array
        snap_indices: Snap indices from distance_transform_edt(~mask)

    Returns:
        Snapped (x, y) position
    """
    h, w = mask.shape
    ix = int(round(min(max(pos[0], 0), w - 1)))
    iy = int(round(min(max(pos[1], 0), h - 1)))
    if mask[iy, ix]:
        return pos
    ny = float(snap_indices[0, iy, ix])
    nx = float(snap_indices[1, iy, ix])
    return (nx, ny)


def _snap_deep_inside(pos, centroid, dist_in, mask, snap_indices):
    """Snap a position to be well inside the mask.

    Cast ray from pos toward centroid, find the point with maximum
    distance-from-edge (deepest inside the glyph).

    Args:
        pos: (x, y) position to snap
        centroid: (x, y) glyph centroid
        dist_in: Distance transform from mask boundary inward
        mask: Binary mask array
        snap_indices: Snap indices from distance_transform_edt(~mask)

    Returns:
        Snapped (x, y) position deep inside mask
    """
    h, w = mask.shape
    ix = int(round(min(max(pos[0], 0), w - 1)))
    iy = int(round(min(max(pos[1], 0), h - 1)))
    if mask[iy, ix] and dist_in[iy, ix] >= 5:
        return pos

    # Walk from pos toward centroid, find deepest point
    dx = centroid[0] - pos[0]
    dy = centroid[1] - pos[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1:
        return _snap_inside(pos, mask, snap_indices)

    best_pos = _snap_inside(pos, mask, snap_indices)
    best_depth = 0
    steps = int(length)
    for s in range(steps + 1):
        t = s / max(steps, 1)
        x = pos[0] + dx * t
        y = pos[1] + dy * t
        jx = int(round(min(max(x, 0), w - 1)))
        jy = int(round(min(max(y, 0), h - 1)))
        if mask[jy, jx] and dist_in[jy, jx] > best_depth:
            best_depth = dist_in[jy, jx]
            best_pos = (x, y)
        # Stop early once we've passed through the deepest part
        # and depth starts decreasing significantly
        if best_depth > 5 and dist_in[jy, jx] < best_depth * 0.5:
            break
    return best_pos


def _snap_to_skeleton_region(region, skel_features, glyph_bbox):
    """Find the closest skeleton feature to the numpad region center.

    Args:
        region: Numpad region number 1-9
        skel_features: Dict from _find_skeleton_waypoints
        glyph_bbox: Glyph bounding box

    Returns:
        Skeleton point (x, y) or None
    """
    if skel_features is None:
        return None
    candidates = skel_features.get(region, [])
    if candidates:
        return candidates[0]  # Already sorted by distance from region center

    # No feature in this region — find the nearest skeleton point
    # from all skeleton points
    target = _numpad_to_pixel(region, glyph_bbox)
    all_skel = skel_features.get('all_skel', [])
    if not all_skel:
        return None
    best = min(all_skel, key=lambda p: (p[0]-target[0])**2 + (p[1]-target[1])**2)
    return best


def _build_guide_path(waypoints_raw, glyph_bbox, mask, skel_features=None):
    """Build a guide path from parsed waypoints.

    waypoints_raw: list of raw waypoint values (int, 'v(n)', 'c(n)')
    skel_features: optional dict from _find_skeleton_waypoints, maps region
                   to list of skeleton feature positions.
    Returns list of (x, y) points sampled along the guide path.
    """
    parsed = [_parse_waypoint(wp) for wp in waypoints_raw]
    n_wp = len(parsed)
    if n_wp < 2:
        return []

    # Compute glyph centroid from mask
    rows, cols = np.where(mask)
    if len(rows) == 0:
        centroid = ((glyph_bbox[0] + glyph_bbox[2]) / 2,
                    (glyph_bbox[1] + glyph_bbox[3]) / 2)
    else:
        centroid = (float(cols.mean()), float(rows.mean()))

    # Pre-compute distance fields for snapping
    h, w = mask.shape
    dist_out, snap_indices = distance_transform_edt(~mask, return_indices=True)
    dist_in = distance_transform_edt(mask)

    # Map waypoints to pixel positions — prefer skeleton features
    positions = []
    for region, kind in parsed:
        # Try skeleton feature first
        skel_pos = _snap_to_skeleton_region(region, skel_features, glyph_bbox)
        if skel_pos is not None:
            pos = (float(skel_pos[0]), float(skel_pos[1]))
        elif kind == 'terminal':
            pos = _snap_to_glyph_edge(
                _numpad_to_pixel(region, glyph_bbox), centroid, mask)
            if pos is None:
                return []
        else:
            pos = _snap_deep_inside(_numpad_to_pixel(region, glyph_bbox),
                                    centroid, dist_in, mask, snap_indices)
            ix = int(round(min(max(pos[0], 0), w - 1)))
            iy = int(round(min(max(pos[1], 0), h - 1)))
            if not mask[iy, ix]:
                return []
        positions.append(pos)

    # Build path segments — straight lines between all waypoints
    all_points = []
    for i in range(n_wp - 1):
        seg = _linear_segment(positions[i], positions[i + 1], step=2.0)
        if all_points and seg:
            seg = seg[1:]
        all_points.extend(seg)

    # Constrain all guide points to be inside the mask
    constrained = []
    for x, y in all_points:
        ix = int(round(min(max(x, 0), w - 1)))
        iy = int(round(min(max(y, 0), h - 1)))
        if mask[iy, ix]:
            constrained.append((x, y))
        else:
            constrained.append(_snap_inside((x, y), mask, snap_indices))

    return constrained


def _get_pixel_contours(font_path, char, canvas_size=224):
    """Extract glyph contours as pixel-space polylines.

    Returns list of polylines, each a list of (x, y) tuples.
    """
    font_path = resolve_font_path(font_path)
    contours, tt = _extract_contours(font_path, char)
    if not contours:
        return []

    transform = _font_to_pixel_transform(tt, font_path, char, canvas_size)

    pixel_contours = []
    for c in contours:
        pc = [transform(p[0], p[1]) for p in c]
        # Close the contour if not already closed
        if len(pc) >= 2:
            d = ((pc[0][0] - pc[-1][0]) ** 2 + (pc[0][1] - pc[-1][1]) ** 2) ** 0.5
            if d > 0.5:
                pc.append(pc[0])
        pixel_contours.append(pc)
    return pixel_contours


def _contour_segments(pixel_contours):
    """Build flat list of line segments from pixel contours.

    Returns list of ((x0,y0), (x1,y1)) tuples.
    """
    segments = []
    for contour in pixel_contours:
        for i in range(len(contour) - 1):
            segments.append((contour[i], contour[i + 1]))
    return segments


def _ray_segment_intersection(origin, direction, seg_a, seg_b):
    """Find intersection parameter t of ray with line segment.

    Ray: origin + t * direction (t >= 0)
    Segment: seg_a to seg_b

    Returns t (distance along ray) or None if no intersection.
    """
    ox, oy = origin
    dx, dy = direction
    ax, ay = seg_a
    bx, by = seg_b

    # Ray: P = O + t * D
    # Segment: P = A + s * (B - A), 0 <= s <= 1
    sx, sy = bx - ax, by - ay

    denom = dx * sy - dy * sx
    if abs(denom) < 1e-10:
        return None  # Parallel

    t = ((ax - ox) * sy - (ay - oy) * sx) / denom
    s = ((ax - ox) * dy - (ay - oy) * dx) / denom

    if t > 0.5 and 0 <= s <= 1:
        return t
    return None


def _find_cross_section_midpoint(point, tangent, segments, mask):
    """Find the stroke center at a guide path point via cross-section ray casting.

    Cast perpendicular rays in both directions, find nearest contour intersection
    on each side, return midpoint.
    """
    # Perpendicular to tangent
    perp = (-tangent[1], tangent[0])
    h, w = mask.shape

    # Cast ray in positive perpendicular direction
    best_t_pos = None
    for seg_a, seg_b in segments:
        t = _ray_segment_intersection(point, perp, seg_a, seg_b)
        if t is not None and (best_t_pos is None or t < best_t_pos):
            best_t_pos = t

    # Cast ray in negative perpendicular direction
    neg_perp = (perp[0] * -1, perp[1] * -1)
    best_t_neg = None
    for seg_a, seg_b in segments:
        t = _ray_segment_intersection(point, neg_perp, seg_a, seg_b)
        if t is not None and (best_t_neg is None or t < best_t_neg):
            best_t_neg = t

    if best_t_pos is not None and best_t_neg is not None:
        # Intersection points
        pos_pt = (point[0] + perp[0] * best_t_pos,
                  point[1] + perp[1] * best_t_pos)
        neg_pt = (point[0] + neg_perp[0] * best_t_neg,
                  point[1] + neg_perp[1] * best_t_neg)
        # Midpoint
        mx = (pos_pt[0] + neg_pt[0]) / 2
        my = (pos_pt[1] + neg_pt[1]) / 2
        # Hard constraint: midpoint must be inside mask
        mix = int(round(min(max(mx, 0), w - 1)))
        miy = int(round(min(max(my, 0), h - 1)))
        if mask[miy, mix]:
            return (mx, my)
        # Midpoint outside mask — fall through to guide position

    # Fallback: use guide position only if inside mask
    ix = int(round(min(max(point[0], 0), w - 1)))
    iy = int(round(min(max(point[1], 0), h - 1)))
    if mask[iy, ix]:
        return point
    # No valid inside position — return None to signal skip
    return None


def _smooth_stroke(points, sigma=2.0):
    """Gaussian smooth a stroke's x and y coordinates independently.

    Delegates to stroke_lib.utils.geometry.smooth_stroke.
    """
    return sl_smooth_stroke(points, sigma)


def _constrain_to_mask(points, mask):
    """Constrain points to stay inside the glyph mask.

    Delegates to stroke_lib.utils.geometry.constrain_to_mask.
    """
    return sl_constrain_to_mask(points, mask)


# ---------------------------------------------------------------------------
# Shape primitives for point-cloud stroke fitting
# ---------------------------------------------------------------------------

def _shape_vline(params, bbox, offset=(0, 0), n_pts=60):
    """Vertical line.  params: (x_frac, y_start_frac, y_end_frac)."""
    xf, ysf, yef = params
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    x = x0 + xf * w + offset[0]
    ys = y0 + ysf * h + offset[1]
    ye = y0 + yef * h + offset[1]
    t = np.linspace(0, 1, n_pts)
    return np.column_stack([np.full(n_pts, x), ys + t * (ye - ys)])


def _shape_hline(params, bbox, offset=(0, 0), n_pts=60):
    """Horizontal line.  params: (y_frac, x_start_frac, x_end_frac)."""
    yf, xsf, xef = params
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    y = y0 + yf * h + offset[1]
    xs = x0 + xsf * w + offset[0]
    xe = x0 + xef * w + offset[0]
    t = np.linspace(0, 1, n_pts)
    return np.column_stack([xs + t * (xe - xs), np.full(n_pts, y)])


def _shape_diag(params, bbox, offset=(0, 0), n_pts=60):
    """Diagonal line.  params: (x0f, y0f, x1f, y1f)."""
    x0f, y0f, x1f, y1f = params
    bx0, by0, bx1, by1 = bbox
    w, h = bx1 - bx0, by1 - by0
    sx = bx0 + x0f * w + offset[0]
    sy = by0 + y0f * h + offset[1]
    ex = bx0 + x1f * w + offset[0]
    ey = by0 + y1f * h + offset[1]
    t = np.linspace(0, 1, n_pts)
    return np.column_stack([sx + t * (ex - sx), sy + t * (ey - sy)])


def _shape_arc_right(params, bbox, offset=(0, 0), n_pts=60):
    """Right-opening arc.  params: (cx_f, cy_f, rx_f, ry_f, ang_start, ang_end)."""
    cxf, cyf, rxf, ryf, a0, a1 = params
    bx0, by0, bx1, by1 = bbox
    w, h = bx1 - bx0, by1 - by0
    cx = bx0 + cxf * w + offset[0]
    cy = by0 + cyf * h + offset[1]
    rx = rxf * w
    ry = ryf * h
    angles = np.linspace(np.radians(a0), np.radians(a1), n_pts)
    return np.column_stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)])


def _shape_arc_left(params, bbox, offset=(0, 0), n_pts=60):
    """Left-opening arc.  params: (cx_f, cy_f, rx_f, ry_f, ang_start, ang_end)."""
    cxf, cyf, rxf, ryf, a0, a1 = params
    bx0, by0, bx1, by1 = bbox
    w, h = bx1 - bx0, by1 - by0
    cx = bx0 + cxf * w + offset[0]
    cy = by0 + cyf * h + offset[1]
    rx = rxf * w
    ry = ryf * h
    angles = np.linspace(np.radians(a0), np.radians(a1), n_pts)
    return np.column_stack([cx - rx * np.cos(angles), cy + ry * np.sin(angles)])


def _shape_loop(params, bbox, offset=(0, 0), n_pts=80):
    """Full ellipse loop.  params: (cx_f, cy_f, rx_f, ry_f)."""
    cxf, cyf, rxf, ryf = params
    bx0, by0, bx1, by1 = bbox
    w, h = bx1 - bx0, by1 - by0
    cx = bx0 + cxf * w + offset[0]
    cy = by0 + cyf * h + offset[1]
    rx = rxf * w
    ry = ryf * h
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    return np.column_stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)])


def _shape_u_arc(params, bbox, offset=(0, 0), n_pts=60):
    """U-shaped arc (bottom half of ellipse).  params: (cx_f, cy_f, rx_f, ry_f)."""
    cxf, cyf, rxf, ryf = params
    bx0, by0, bx1, by1 = bbox
    w, h = bx1 - bx0, by1 - by0
    cx = bx0 + cxf * w + offset[0]
    cy = by0 + cyf * h + offset[1]
    rx = rxf * w
    ry = ryf * h
    angles = np.linspace(0, np.pi, n_pts)
    return np.column_stack([cx - rx * np.cos(angles), cy + ry * np.sin(angles)])


SHAPE_FNS = {
    'vline': _shape_vline,
    'hline': _shape_hline,
    'diag': _shape_diag,
    'arc_right': _shape_arc_right,
    'arc_left': _shape_arc_left,
    'loop': _shape_loop,
    'u_arc': _shape_u_arc,
}

# Bounds per shape type for differential_evolution optimisation.
# All in bbox-fraction space except arc angles which are in degrees.
SHAPE_PARAM_BOUNDS = {
    'vline': [(0.0, 1.0), (0.0, 0.5), (0.5, 1.0)],
    'hline': [(0.0, 1.0), (0.0, 0.5), (0.5, 1.0)],
    'diag': [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    'arc_right': [(0.0, 0.8), (0.0, 1.0), (0.05, 0.8), (0.05, 0.8),
                  (-180, 0), (0, 180)],
    'arc_left': [(0.2, 1.0), (0.0, 1.0), (0.05, 0.8), (0.05, 0.8),
                 (-180, 0), (0, 180)],
    'loop': [(0.1, 0.9), (0.1, 0.9), (0.1, 0.6), (0.1, 0.6)],
    'u_arc': [(0.1, 0.9), (0.2, 1.0), (0.1, 0.6), (0.1, 0.6)],
}


def _get_param_bounds(templates):
    """Build flat bounds list + per-shape slice indices.

    Each template entry may include an optional 'bounds' key that overrides
    specific parameter bounds.  Format: list of (lo, hi) or None per param.
    None entries keep the default from SHAPE_PARAM_BOUNDS.
    """
    bounds = []
    slices = []
    offset = 0
    for t in templates:
        sb = list(SHAPE_PARAM_BOUNDS[t['shape']])
        overrides = t.get('bounds')
        if overrides:
            for j, ov in enumerate(overrides):
                if ov is not None:
                    sb[j] = ov
        bounds.extend(sb)
        slices.append((offset, offset + len(sb)))
        offset += len(sb)
    return bounds, slices


def _param_vector_to_shapes(param_vector, shape_types, slices, bbox, n_pts=None):
    """Convert flat parameter vector into list of Nx2 point arrays.

    When n_pts is None it is computed from the bbox diagonal so the shape
    path is dense enough for the matching radius to form a continuous band
    (~1.5 px between samples).
    """
    if n_pts is None:
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        n_pts = max(60, int((bw * bw + bh * bh) ** 0.5 / 1.5))
    shapes = []
    for i, stype in enumerate(shape_types):
        start, end = slices[i]
        params = tuple(param_vector[start:end])
        shapes.append(SHAPE_FNS[stype](params, bbox, offset=(0, 0), n_pts=n_pts))
    return shapes


def _score_all_strokes(param_vector, shape_types, slices, bbox, cloud_tree,
                       n_cloud, radius, snap_yi, snap_xi, w, h, dist_map=None):
    """Objective for optimisation (minimisation → returns -score).

    Snaps stroke points to nearest mask pixel before scoring so the
    optimiser sees the same benefit as the post-processing pipeline.

    Score = coverage − snap_penalty − edge_penalty
      coverage:       fraction of cloud points within radius of any stroke
      snap_penalty:   fraction of stroke points in white space (hard penalty)
      edge_penalty:   penalises strokes hugging the glyph edge instead of
                      running through the interior
    """
    all_shapes = _param_vector_to_shapes(param_vector, shape_types, slices, bbox)
    all_pts = np.concatenate(all_shapes, axis=0)
    if len(all_pts) == 0:
        return 0.0

    # Snap all stroke points to nearest mask pixel
    xi = np.clip(np.round(all_pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(all_pts[:, 1]).astype(int), 0, h - 1)
    snapped_x = snap_xi[yi, xi].astype(float)
    snapped_y = snap_yi[yi, xi].astype(float)
    snapped = np.column_stack([snapped_x, snapped_y])

    # White-space penalty: fraction of stroke points that lie outside the mask.
    # Any point with snap_dist > 0.5 is off the mask (in white space).
    # This is a hard penalty — shapes must stay on the glyph.
    snap_dist = np.sqrt((all_pts[:, 0] - snapped_x) ** 2 +
                        (all_pts[:, 1] - snapped_y) ** 2)
    off_mask = float(np.mean(snap_dist > 0.5))  # fraction of points in white space
    snap_penalty = 0.5 * off_mask  # strong but not crushing penalty

    # Edge penalty: penalise stroke points near the glyph boundary.
    # dist_map gives distance from each pixel to nearest background pixel.
    # Points with small dist_map values are close to the edge.
    edge_penalty = 0.0
    if dist_map is not None:
        sxi = np.clip(np.round(snapped_x).astype(int), 0, w - 1)
        syi = np.clip(np.round(snapped_y).astype(int), 0, h - 1)
        dt_vals = dist_map[syi, sxi]
        # Fraction of snapped points within 1.5px of the edge
        near_edge = float(np.mean(dt_vals < 1.5))
        edge_penalty = 0.1 * near_edge

    # Per-shape coverage sets
    per_shape = []
    offset = 0
    for i in range(len(shape_types)):
        n = len(all_shapes[i])
        shape_snapped = snapped[offset:offset + n]
        offset += n
        hits = cloud_tree.query_ball_point(shape_snapped, radius)
        sc = set()
        for lst in hits:
            sc.update(lst)
        per_shape.append(sc)

    covered_all = set().union(*per_shape) if per_shape else set()
    coverage = len(covered_all) / n_cloud

    # Overlap penalty: for each shape, measure what fraction of its covered
    # points are also covered by other shapes.  Small overlaps at vertices
    # and crossings are normal (~20-30%), so only penalise the excess above
    # a generous free allowance.
    FREE_OVERLAP = 0.25  # 25% overlap is free (junctions, crossings)
    overlap_excess = 0.0
    n_shapes = len(per_shape)
    if n_shapes > 1:
        for i in range(n_shapes):
            if not per_shape[i]:
                continue
            others = set()
            for j in range(n_shapes):
                if j != i:
                    others |= per_shape[j]
            frac = len(per_shape[i] & others) / len(per_shape[i])
            if frac > FREE_OVERLAP:
                overlap_excess += (frac - FREE_OVERLAP)
        overlap_excess /= n_shapes  # average per shape

    overlap_penalty = 0.5 * overlap_excess

    return -(coverage - overlap_penalty - snap_penalty - edge_penalty)


# ---------------------------------------------------------------------------
# Raw-stroke scoring & affine optimisation (template-first approach)
# ---------------------------------------------------------------------------

def _score_raw_strokes(stroke_arrays, cloud_tree, n_cloud, radius,
                       snap_yi, snap_xi, w, h, dist_map=None, mask=None):
    """Score pre-built stroke point arrays against the target point cloud.

    Like _score_all_strokes but accepts raw Nx2 arrays (one per stroke)
    instead of a shape-parameter vector.  Strokes are assumed already
    smoothed; only snap-to-mask is applied (via the snap arrays).
    """
    if not stroke_arrays or all(len(s) == 0 for s in stroke_arrays):
        return 0.0

    processed = [s for s in stroke_arrays if len(s) >= 2]
    if not processed:
        return 0.0

    all_pts = np.concatenate(processed, axis=0)
    xi = np.clip(np.round(all_pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(all_pts[:, 1]).astype(int), 0, h - 1)
    snapped_x = snap_xi[yi, xi].astype(float)
    snapped_y = snap_yi[yi, xi].astype(float)
    snapped = np.column_stack([snapped_x, snapped_y])

    snap_dist = np.sqrt((all_pts[:, 0] - snapped_x) ** 2 +
                        (all_pts[:, 1] - snapped_y) ** 2)
    off_mask = float(np.mean(snap_dist > 0.5))
    snap_penalty = 0.5 * off_mask

    edge_penalty = 0.0
    if dist_map is not None:
        sxi = np.clip(np.round(snapped_x).astype(int), 0, w - 1)
        syi = np.clip(np.round(snapped_y).astype(int), 0, h - 1)
        near_edge = float(np.mean(dist_map[syi, sxi] < 1.5))
        edge_penalty = 0.1 * near_edge

    per_shape = []
    offset = 0
    for arr in processed:
        n = len(arr)
        shape_snapped = snapped[offset:offset + n]
        offset += n
        hits = cloud_tree.query_ball_point(shape_snapped, radius)
        sc = set()
        for lst in hits:
            sc.update(lst)
        per_shape.append(sc)

    covered_all = set().union(*per_shape) if per_shape else set()
    coverage = len(covered_all) / n_cloud

    FREE_OVERLAP = 0.25
    overlap_excess = 0.0
    n_shapes = len(per_shape)
    if n_shapes > 1:
        for i in range(n_shapes):
            if not per_shape[i]:
                continue
            others = set()
            for j in range(n_shapes):
                if j != i:
                    others |= per_shape[j]
            frac = len(per_shape[i] & others) / len(per_shape[i])
            if frac > FREE_OVERLAP:
                overlap_excess += (frac - FREE_OVERLAP)
        overlap_excess /= n_shapes

    overlap_penalty = 0.5 * overlap_excess
    return -(coverage - overlap_penalty - snap_penalty - edge_penalty)


def _affine_transform_strokes(strokes, params, centroid):
    """Apply affine transform to strokes around a centroid.

    params: (tx, ty, sx, sy, theta_deg, shear)
    Returns list of Nx2 numpy arrays.
    """
    tx, ty, sx, sy, theta_deg, shear = params
    theta = np.radians(theta_deg)
    cx, cy = centroid
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    result = []
    for stroke in strokes:
        pts = np.array(stroke, dtype=float)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        rx = dx * cos_t - dy * sin_t + shear * dy
        ry = dx * sin_t + dy * cos_t
        pts[:, 0] = cx + sx * rx + tx
        pts[:, 1] = cy + sy * ry + ty
        result.append(pts)
    return result


def _prepare_affine_optimization(font_path, char, canvas_size, strokes_raw, mask):
    """Setup optimization data structures for affine stroke optimization.

    Args:
        font_path: Path to font file
        char: Character being optimized
        canvas_size: Canvas size for rendering
        strokes_raw: Raw strokes from template
        mask: Glyph mask array

    Returns:
        Tuple of (stroke_arrays, centroid, glyph_bbox, score_args) or None if setup fails
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None

    glyph_bbox = (float(cols.min()), float(rows.min()),
                  float(cols.max()), float(rows.max()))
    centroid = (float(cols.mean()), float(rows.mean()))
    cloud = _make_point_cloud(mask, spacing=3)
    if len(cloud) < 10:
        return None
    cloud_tree = cKDTree(cloud)
    n_cloud = len(cloud)
    radius = _adaptive_radius(mask, spacing=3)
    h, w = mask.shape
    dist_map = distance_transform_edt(mask)
    _, snap_indices = distance_transform_edt(~mask, return_indices=True)
    snap_yi, snap_xi = snap_indices[0], snap_indices[1]

    score_args = (cloud_tree, n_cloud, radius, snap_yi, snap_xi, w, h, dist_map)

    # Convert raw strokes to numpy arrays (pre-smooth once)
    stroke_arrays = []
    for s in strokes_raw:
        pl = [(float(p[0]), float(p[1])) for p in s]
        pl = _smooth_stroke(pl, sigma=2.0)
        pl = _constrain_to_mask(pl, mask)
        if len(pl) >= 2:
            stroke_arrays.append(np.array(pl))
    if not stroke_arrays:
        return None

    return stroke_arrays, centroid, glyph_bbox, score_args


def _run_global_affine(stroke_arrays, centroid, score_args):
    """Run Stage 1 global affine optimization on all strokes together.

    Args:
        stroke_arrays: List of Nx2 numpy arrays, one per stroke
        centroid: (x, y) center point for affine transform
        score_args: Tuple of scoring function arguments

    Returns:
        Tuple of (best_strokes, best_params, best_score)
    """
    from scipy.optimize import minimize, differential_evolution

    affine_bounds = [(-20, 20), (-20, 20),  # translate
                     (0.7, 1.3), (0.7, 1.3),  # scale
                     (-15, 15),  # rotation degrees
                     (-0.3, 0.3)]  # shear

    def _affine_obj(params):
        transformed = _affine_transform_strokes(stroke_arrays, params, centroid)
        return _score_raw_strokes(transformed, *score_args)

    # Quick NM from identity
    x0 = np.array([0, 0, 1, 1, 0, 0], dtype=float)
    nm = minimize(_affine_obj, x0, method='Nelder-Mead',
                  options={'maxfev': 800, 'xatol': 0.1, 'fatol': 0.002,
                           'adaptive': True})
    best_params = nm.x.copy()
    best_score = nm.fun

    # DE refinement (quick)
    try:
        de = differential_evolution(_affine_obj, bounds=affine_bounds,
                                    x0=best_params, maxiter=20, popsize=10,
                                    tol=0.005, polish=False)
        if de.fun < best_score:
            best_params = de.x.copy()
            best_score = de.fun
    except Exception:
        pass

    # Apply best global affine
    best_strokes = _affine_transform_strokes(stroke_arrays, best_params, centroid)
    return best_strokes, best_params, best_score


def _run_per_stroke_refinement(best_strokes, best_score, score_args):
    """Run Stage 2 per-stroke translate+scale refinement.

    Args:
        best_strokes: List of Nx2 numpy arrays after global affine
        best_score: Score from global affine stage
        score_args: Tuple of scoring function arguments

    Returns:
        Tuple of (final_strokes, final_score)
    """
    from scipy.optimize import minimize

    n_strokes = len(best_strokes)

    def _per_stroke_obj(params):
        adjusted = []
        for si, base in enumerate(best_strokes):
            dx, dy, sx, sy = params[si * 4:(si + 1) * 4]
            pts = base.copy()
            c = pts.mean(axis=0)
            pts[:, 0] = c[0] + sx * (pts[:, 0] - c[0]) + dx
            pts[:, 1] = c[1] + sy * (pts[:, 1] - c[1]) + dy
            adjusted.append(pts)
        return _score_raw_strokes(adjusted, *score_args)

    x0_per = np.array([0, 0, 1, 1] * n_strokes, dtype=float)
    nm2 = minimize(_per_stroke_obj, x0_per, method='Nelder-Mead',
                   options={'maxfev': 1500, 'xatol': 0.1, 'fatol': 0.002,
                            'adaptive': True})

    if nm2.fun < best_score:
        final_strokes = []
        for si, base in enumerate(best_strokes):
            dx, dy, sx, sy = nm2.x[si * 4:(si + 1) * 4]
            pts = base.copy()
            c = pts.mean(axis=0)
            pts[:, 0] = c[0] + sx * (pts[:, 0] - c[0]) + dx
            pts[:, 1] = c[1] + sy * (pts[:, 1] - c[1]) + dy
            final_strokes.append(pts)
        return final_strokes, nm2.fun
    else:
        return best_strokes, best_score


def _optimize_affine(font_path, char, canvas_size=224):
    """Optimise template strokes via affine transforms.

    Stage 1: Global affine (6 params) on all strokes together.
    Stage 2: Per-stroke translate+scale refinement.

    Returns (strokes, score, mask, glyph_bbox) or None if no template.
    """
    strokes_raw = template_to_strokes(font_path, char, canvas_size)
    if not strokes_raw or len(strokes_raw) == 0:
        return None

    font_path = resolve_font_path(font_path)
    mask = render_glyph_mask(font_path, char, canvas_size)
    if mask is None:
        return None

    # Setup optimization data
    setup = _prepare_affine_optimization(font_path, char, canvas_size, strokes_raw, mask)
    if setup is None:
        return None
    stroke_arrays, centroid, glyph_bbox, score_args = setup

    # Stage 1: Global affine
    best_strokes, best_params, best_score = _run_global_affine(stroke_arrays, centroid, score_args)

    # Stage 2: Per-stroke refinement
    final_strokes, final_score = _run_per_stroke_refinement(best_strokes, best_score, score_args)

    # Convert back to list format
    result_strokes = [[[round(float(p[0]), 1), round(float(p[1]), 1)]
                       for p in s] for s in final_strokes]

    return result_strokes, float(-final_score), mask, glyph_bbox


def _optimize_diffvg(font_path, char, canvas_size=224):
    """Optimise strokes using DiffVG differentiable rendering in Docker.

    Uses gradient-based optimization through a differentiable rasterizer
    to fit polyline strokes to the glyph mask. Requires Docker with the
    diffvg-optimizer image built.

    Returns same format as _optimize_affine: (strokes, score, mask, bbox)
    or None on failure.
    """
    if _diffvg_docker is None:
        return None

    # Get initial strokes from template
    tpl = template_to_strokes(font_path, char, canvas_size)
    if tpl is None:
        return None

    initial_strokes = tpl if not isinstance(tpl, tuple) else tpl[0]
    if not initial_strokes:
        return None

    result = _diffvg_docker.optimize(
        font_path=font_path,
        char=char,
        initial_strokes=initial_strokes,
        canvas_size=canvas_size,
        num_iterations=500,
        stroke_width=8.0,
        timeout=300,
    )

    if 'error' in result:
        print(f'DiffVG error for {char}: {result["error"]}')
        return None

    diffvg_strokes = result.get('strokes', [])
    diffvg_score = result.get('score', 0.0)

    if not diffvg_strokes or diffvg_score <= 0:
        return None

    # Render glyph mask and bbox (same as _optimize_affine does)
    mask = render_glyph_mask(font_path, char, canvas_size)
    if mask is None:
        return None
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    glyph_bbox = (float(xs.min()), float(ys.min()),
                  float(xs.max()), float(ys.max()))

    return diffvg_strokes, diffvg_score, mask, glyph_bbox


# ---------------------------------------------------------------------------
# Shape templates for all 62 characters
# ---------------------------------------------------------------------------

SHAPE_TEMPLATES = {
    # --- Uppercase ---
    # 'group' key assigns shapes to strokes.  Shapes with the same group
    # are concatenated into a single stroke (joined at their nearest
    # endpoints).  Omitted group defaults to a unique stroke per shape.
    'A': [
        {'shape': 'diag', 'params': (0.5, 0.0, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 0.0, 1.0, 1.0), 'group': 0},
        {'shape': 'hline', 'params': (0.55, 0.2, 0.8)},
    ],
    'B': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'arc_right', 'params': (0.15, 0.25, 0.45, 0.24, -90, 90),
         'bounds': [None, (0.10, 0.35), (0.15, 0.65), (0.10, 0.28), (-100, -80), (80, 100)]},
        {'shape': 'arc_right', 'params': (0.15, 0.75, 0.45, 0.24, -90, 90),
         'bounds': [None, (0.65, 0.90), (0.15, 0.65), (0.10, 0.28), (-100, -80), (80, 100)]},
    ],
    'C': [
        {'shape': 'arc_left', 'params': (0.85, 0.5, 0.5, 0.5, -90, 90)},
    ],
    'D': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.5, 0.5, 0.5, -90, 90), 'group': 0},
    ],
    'E': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'hline', 'params': (0.0, 0.15, 1.0)},
        {'shape': 'hline', 'params': (0.5, 0.15, 0.85)},
        {'shape': 'hline', 'params': (1.0, 0.15, 1.0)},
    ],
    'F': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'hline', 'params': (0.0, 0.15, 1.0)},
        {'shape': 'hline', 'params': (0.5, 0.15, 0.85)},
    ],
    'G': [
        {'shape': 'arc_left', 'params': (0.85, 0.5, 0.5, 0.5, -90, 90), 'group': 0},
        {'shape': 'hline', 'params': (0.5, 0.5, 1.0), 'group': 0},
    ],
    'H': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'vline', 'params': (0.85, 0.0, 1.0)},
        {'shape': 'hline', 'params': (0.5, 0.15, 0.85)},
    ],
    'I': [
        {'shape': 'vline', 'params': (0.5, 0.0, 1.0)},
    ],
    'J': [
        {'shape': 'vline', 'params': (0.7, 0.0, 0.7), 'group': 0},
        {'shape': 'u_arc', 'params': (0.5, 0.7, 0.3, 0.3), 'group': 0},
    ],
    'K': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'diag', 'params': (1.0, 0.0, 0.15, 0.5), 'group': 1},
        {'shape': 'diag', 'params': (0.15, 0.5, 1.0, 1.0), 'group': 1},
    ],
    'L': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'hline', 'params': (1.0, 0.15, 1.0), 'group': 0},
    ],
    'M': [
        {'shape': 'vline', 'params': (0.05, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.05, 0.0, 0.5, 0.5), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 0.5, 0.95, 0.0), 'group': 0},
        {'shape': 'vline', 'params': (0.95, 0.0, 1.0), 'group': 0},
    ],
    'N': [
        {'shape': 'vline', 'params': (0.1, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.1, 0.0, 0.9, 1.0), 'group': 0},
        {'shape': 'vline', 'params': (0.9, 0.0, 1.0), 'group': 0},
    ],
    'O': [
        {'shape': 'loop', 'params': (0.5, 0.5, 0.45, 0.48)},
    ],
    'P': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.25, 0.45, 0.25, -90, 90), 'group': 0},
    ],
    'Q': [
        {'shape': 'loop', 'params': (0.5, 0.45, 0.45, 0.45)},
        {'shape': 'diag', 'params': (0.5, 0.7, 0.95, 1.0)},
    ],
    'R': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.25, 0.45, 0.25, -90, 90), 'group': 0},
        {'shape': 'diag', 'params': (0.4, 0.5, 1.0, 1.0)},
    ],
    'S': [
        {'shape': 'arc_left', 'params': (0.6, 0.28, 0.4, 0.22, -90, 90),
         'bounds': [None, (0.12, 0.32), (0.15, 0.6), (0.05, 0.18), (-100, -80), (80, 100)], 'group': 0},
        {'shape': 'arc_right', 'params': (0.4, 0.72, 0.4, 0.22, -90, 90),
         'bounds': [None, (0.68, 0.88), (0.15, 0.6), (0.05, 0.18), (-100, -80), (80, 100)], 'group': 0},
    ],
    'T': [
        {'shape': 'hline', 'params': (0.0, 0.0, 1.0)},
        {'shape': 'vline', 'params': (0.5, 0.0, 1.0)},
    ],
    'U': [
        {'shape': 'vline', 'params': (0.15, 0.0, 0.65), 'group': 0},
        {'shape': 'u_arc', 'params': (0.5, 0.65, 0.35, 0.35), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.0, 0.65), 'group': 0},
    ],
    'V': [
        {'shape': 'diag', 'params': (0.0, 0.0, 0.5, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 1.0, 1.0, 0.0), 'group': 0},
    ],
    'W': [
        {'shape': 'diag', 'params': (0.0, 0.0, 0.25, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.25, 1.0, 0.5, 0.4), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 0.4, 0.75, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.75, 1.0, 1.0, 0.0), 'group': 0},
    ],
    'X': [
        {'shape': 'diag', 'params': (0.0, 0.0, 1.0, 1.0)},
        {'shape': 'diag', 'params': (1.0, 0.0, 0.0, 1.0)},
    ],
    'Y': [
        {'shape': 'diag', 'params': (0.0, 0.0, 0.5, 0.5), 'group': 0},
        {'shape': 'diag', 'params': (1.0, 0.0, 0.5, 0.5), 'group': 0},
        {'shape': 'vline', 'params': (0.5, 0.5, 1.0), 'group': 0},
    ],
    'Z': [
        {'shape': 'hline', 'params': (0.0, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (1.0, 0.0, 0.0, 1.0), 'group': 0},
        {'shape': 'hline', 'params': (1.0, 0.0, 1.0), 'group': 0},
    ],

    # --- Lowercase ---
    'a': [
        {'shape': 'arc_left', 'params': (0.7, 0.5, 0.4, 0.45, -90, 90), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.1, 1.0), 'group': 0},
    ],
    'b': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.6, 0.4, 0.4, -90, 90), 'group': 0},
    ],
    'c': [
        {'shape': 'arc_left', 'params': (0.8, 0.5, 0.45, 0.48, -90, 90)},
    ],
    'd': [
        {'shape': 'arc_left', 'params': (0.7, 0.6, 0.4, 0.4, -90, 90), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.0, 1.0), 'group': 0},
    ],
    'e': [
        {'shape': 'hline', 'params': (0.45, 0.15, 0.85), 'group': 0},
        {'shape': 'arc_left', 'params': (0.7, 0.55, 0.4, 0.4, 0, 90), 'group': 0},
    ],
    'f': [
        {'shape': 'vline', 'params': (0.4, 0.15, 1.0), 'group': 0},
        {'shape': 'arc_left', 'params': (0.7, 0.15, 0.3, 0.15, -90, 0), 'group': 0},
        {'shape': 'hline', 'params': (0.35, 0.15, 0.7)},
    ],
    'g': [
        {'shape': 'arc_left', 'params': (0.7, 0.35, 0.4, 0.35, -90, 90), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.1, 0.85), 'group': 0},
        {'shape': 'u_arc', 'params': (0.5, 0.85, 0.35, 0.15), 'group': 0},
    ],
    'h': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.4, 0.35, 0.2, -90, 0), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.4, 1.0), 'group': 0},
    ],
    'i': [
        {'shape': 'vline', 'params': (0.5, 0.25, 1.0)},
    ],
    'j': [
        {'shape': 'vline', 'params': (0.5, 0.25, 0.8), 'group': 0},
        {'shape': 'u_arc', 'params': (0.3, 0.8, 0.2, 0.2), 'group': 0},
    ],
    'k': [
        {'shape': 'vline', 'params': (0.15, 0.0, 1.0)},
        {'shape': 'diag', 'params': (0.85, 0.25, 0.15, 0.55), 'group': 1},
        {'shape': 'diag', 'params': (0.15, 0.55, 0.85, 1.0), 'group': 1},
    ],
    'l': [
        {'shape': 'vline', 'params': (0.5, 0.0, 1.0)},
    ],
    'm': [
        {'shape': 'vline', 'params': (0.08, 0.2, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.08, 0.35, 0.22, 0.18, -90, 0), 'group': 0},
        {'shape': 'vline', 'params': (0.5, 0.2, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.5, 0.35, 0.22, 0.18, -90, 0), 'group': 0},
        {'shape': 'vline', 'params': (0.92, 0.2, 1.0), 'group': 0},
    ],
    'n': [
        {'shape': 'vline', 'params': (0.15, 0.2, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.4, 0.35, 0.22, -90, 0), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.2, 1.0), 'group': 0},
    ],
    'o': [
        {'shape': 'loop', 'params': (0.5, 0.55, 0.4, 0.42)},
    ],
    'p': [
        {'shape': 'vline', 'params': (0.15, 0.2, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.15, 0.35, 0.4, 0.3, -90, 90), 'group': 0},
    ],
    'q': [
        {'shape': 'arc_left', 'params': (0.7, 0.35, 0.4, 0.3, -90, 90), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.2, 1.0), 'group': 0},
    ],
    'r': [
        {'shape': 'vline', 'params': (0.2, 0.2, 1.0), 'group': 0},
        {'shape': 'arc_right', 'params': (0.2, 0.35, 0.3, 0.18, -90, 0), 'group': 0},
    ],
    's': [
        {'shape': 'arc_left', 'params': (0.6, 0.32, 0.35, 0.22, -90, 90),
         'bounds': [None, (0.05, 0.45), None, None, None, None], 'group': 0},
        {'shape': 'arc_right', 'params': (0.4, 0.68, 0.35, 0.22, -90, 90),
         'bounds': [None, (0.55, 0.95), None, None, None, None], 'group': 0},
    ],
    't': [
        {'shape': 'vline', 'params': (0.4, 0.0, 1.0)},
        {'shape': 'hline', 'params': (0.3, 0.1, 0.75)},
    ],
    'u': [
        {'shape': 'vline', 'params': (0.15, 0.2, 0.65), 'group': 0},
        {'shape': 'u_arc', 'params': (0.5, 0.65, 0.35, 0.35), 'group': 0},
        {'shape': 'vline', 'params': (0.85, 0.2, 1.0), 'group': 0},
    ],
    'v': [
        {'shape': 'diag', 'params': (0.0, 0.2, 0.5, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 1.0, 1.0, 0.2), 'group': 0},
    ],
    'w': [
        {'shape': 'diag', 'params': (0.0, 0.2, 0.25, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.25, 1.0, 0.5, 0.5), 'group': 0},
        {'shape': 'diag', 'params': (0.5, 0.5, 0.75, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (0.75, 1.0, 1.0, 0.2), 'group': 0},
    ],
    'x': [
        {'shape': 'diag', 'params': (0.0, 0.2, 1.0, 1.0)},
        {'shape': 'diag', 'params': (1.0, 0.2, 0.0, 1.0)},
    ],
    'y': [
        {'shape': 'diag', 'params': (0.0, 0.2, 0.5, 0.6), 'group': 0},
        {'shape': 'diag', 'params': (1.0, 0.2, 0.15, 1.0), 'group': 0},
    ],
    'z': [
        {'shape': 'hline', 'params': (0.2, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (1.0, 0.2, 0.0, 1.0), 'group': 0},
        {'shape': 'hline', 'params': (1.0, 0.0, 1.0), 'group': 0},
    ],

    # --- Digits ---
    '0': [
        {'shape': 'loop', 'params': (0.5, 0.5, 0.42, 0.48)},
    ],
    '1': [
        {'shape': 'vline', 'params': (0.5, 0.0, 1.0)},
    ],
    '2': [
        {'shape': 'arc_left', 'params': (0.6, 0.25, 0.4, 0.25, -90, 45), 'group': 0},
        {'shape': 'diag', 'params': (0.7, 0.4, 0.0, 1.0), 'group': 0},
        {'shape': 'hline', 'params': (1.0, 0.0, 1.0), 'group': 0},
    ],
    '3': [
        {'shape': 'arc_right', 'params': (0.35, 0.27, 0.4, 0.27, -90, 90),
         'bounds': [None, (0.05, 0.45), None, None, None, None], 'group': 0},
        {'shape': 'arc_right', 'params': (0.35, 0.73, 0.4, 0.27, -90, 90),
         'bounds': [None, (0.55, 0.95), None, None, None, None], 'group': 0},
    ],
    '4': [
        {'shape': 'diag', 'params': (0.0, 0.0, 0.0, 0.6), 'group': 0},
        {'shape': 'hline', 'params': (0.6, 0.0, 0.85), 'group': 0},
        {'shape': 'vline', 'params': (0.7, 0.0, 1.0)},
    ],
    '5': [
        {'shape': 'hline', 'params': (0.0, 0.0, 0.9), 'group': 0},
        {'shape': 'vline', 'params': (0.1, 0.0, 0.45), 'group': 0},
        {'shape': 'arc_right', 'params': (0.2, 0.7, 0.45, 0.3, -90, 90), 'group': 0},
    ],
    '6': [
        {'shape': 'arc_left', 'params': (0.75, 0.3, 0.4, 0.35, -90, 60), 'group': 0},
        {'shape': 'loop', 'params': (0.5, 0.65, 0.38, 0.32), 'group': 0},
    ],
    '7': [
        {'shape': 'hline', 'params': (0.0, 0.0, 1.0), 'group': 0},
        {'shape': 'diag', 'params': (1.0, 0.0, 0.25, 1.0), 'group': 0},
    ],
    '8': [
        {'shape': 'loop', 'params': (0.5, 0.27, 0.32, 0.25),
         'bounds': [None, (0.1, 0.45), None, None], 'group': 0},
        {'shape': 'loop', 'params': (0.5, 0.73, 0.38, 0.27),
         'bounds': [None, (0.55, 0.9), None, None], 'group': 0},
    ],
    '9': [
        {'shape': 'loop', 'params': (0.5, 0.35, 0.38, 0.32), 'group': 0},
        {'shape': 'arc_right', 'params': (0.25, 0.7, 0.4, 0.35, -60, 90), 'group': 0},
    ],
}


# ---------------------------------------------------------------------------
# Point cloud, scoring, and optimizer
# ---------------------------------------------------------------------------

def _make_point_cloud(mask, spacing=2):
    """Create a grid of points inside the glyph mask."""
    h, w = mask.shape
    ys, xs = np.mgrid[0:h:spacing, 0:w:spacing]
    xs = xs.ravel()
    ys = ys.ravel()
    inside = mask[ys, xs]
    return np.column_stack([xs[inside], ys[inside]]).astype(float)


def _adaptive_radius(mask, spacing=2):
    """Compute matching radius based on stroke width.

    Uses the 95th percentile of the distance transform — close to the
    maximum stroke half-width — so the optimiser can cover points across
    the full width of even the thickest strokes.  Floor at 1.5x grid
    spacing so the radius always reaches neighbouring grid points, even
    for very thin strokes.
    """
    dist = distance_transform_edt(mask)
    vals = dist[mask]
    floor = spacing * 1.5
    if len(vals) == 0:
        return max(6.0, floor)
    return max(float(np.percentile(vals, 95)), floor)


def _score_shape(shape_pts, tree, radius, claimed=None):
    """Count cloud points within radius of shape path.

    Gives a bonus weight for unclaimed points.
    """
    if len(shape_pts) == 0:
        return 0
    indices = tree.query_ball_point(shape_pts, radius)
    hit = set()
    for idx_list in indices:
        hit.update(idx_list)
    if claimed is None:
        return len(hit)
    unclaimed = hit - claimed
    return len(unclaimed) * 1.0 + len(hit & claimed) * 0.3


def _optimize_shape(shape_name, params, cloud, tree, mask, bbox, radius, claimed):
    """Grid-search to optimise shape offset (dx, dy) and scale.

    Three-phase: scale sweep, coarse translate, fine translate.
    Returns (best_points, covered_indices).
    """
    fn = SHAPE_FNS[shape_name]

    # Enough points to densely trace the shape, capped to keep queries fast.
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    diag = (bw * bw + bh * bh) ** 0.5
    n_pts = min(max(60, int(diag * 0.5)), 100)

    def _gen_scaled(dx, dy, scale):
        """Generate shape points, then scale around the shape centroid."""
        pts = fn(params, bbox, offset=(dx, dy), n_pts=n_pts)
        if scale != 1.0 and len(pts) > 0:
            cx = pts[:, 0].mean()
            cy = pts[:, 1].mean()
            pts = np.column_stack([
                cx + (pts[:, 0] - cx) * scale,
                cy + (pts[:, 1] - cy) * scale,
            ])
        return pts

    def evaluate(dx, dy, scale):
        pts = _gen_scaled(dx, dy, scale)
        return _score_shape(pts, tree, radius, claimed), pts

    # Phase 1: joint coarse search over scale + translate (±36px, step 6)
    scales = [0.8, 1.0, 1.25, 1.5]
    best_scale = 1.0
    best_score = -1
    best_dx, best_dy = 0, 0
    best_pts = None
    for s in scales:
        for dx in range(-36, 37, 6):
            for dy in range(-36, 37, 6):
                sc, pts = evaluate(dx, dy, s)
                if sc > best_score:
                    best_score = sc
                    best_dx, best_dy = dx, dy
                    best_scale = s
                    best_pts = pts

    # Phase 2: fine translate (±5px around best coarse, at best scale)
    cdx, cdy = best_dx, best_dy
    for dx in range(cdx - 5, cdx + 6):
        for dy in range(cdy - 5, cdy + 6):
            sc, pts = evaluate(dx, dy, best_scale)
            if sc > best_score:
                best_score = sc
                best_dx, best_dy = dx, dy
                best_pts = pts

    # Determine covered indices
    if best_pts is not None and len(best_pts) > 0:
        indices = tree.query_ball_point(best_pts, radius)
        covered = set()
        for idx_list in indices:
            covered.update(idx_list)
    else:
        covered = set()
        best_pts = fn(params, bbox, n_pts=n_pts)

    return best_pts, covered


# ---------------------------------------------------------------------------
# Main entry point: shape_fit_to_strokes
# ---------------------------------------------------------------------------

def shape_fit_to_strokes(font_path, char, canvas_size=224, return_markers=False):
    """Generate strokes by fitting shape templates to a glyph point cloud.

    Returns list of strokes [[[x,y], ...], ...] or None.
    If return_markers=True, returns (strokes, markers).
    """
    templates = SHAPE_TEMPLATES.get(char)
    if not templates:
        return None

    font_path = resolve_font_path(font_path)
    mask = render_glyph_mask(font_path, char, canvas_size)
    if mask is None:
        return None

    # Bounding box
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    glyph_bbox = (float(cols.min()), float(rows.min()),
                  float(cols.max()), float(rows.max()))

    # Point cloud + KDTree
    cloud = _make_point_cloud(mask, spacing=4)
    if len(cloud) < 10:
        return None
    tree = cKDTree(cloud)

    radius = _adaptive_radius(mask)
    claimed = set()

    strokes = []
    all_markers = []

    for si, tmpl in enumerate(templates):
        shape_name = tmpl['shape']
        params = tmpl['params']

        # Optimize
        pts, covered = _optimize_shape(
            shape_name, params, cloud, tree, mask, glyph_bbox, radius, claimed
        )
        claimed |= covered

        # Post-process
        point_list = [(float(p[0]), float(p[1])) for p in pts]
        point_list = _smooth_stroke(point_list, sigma=2.0)
        point_list = _constrain_to_mask(point_list, mask)

        if len(point_list) < 2:
            continue

        final_stroke = [[round(x, 1), round(y, 1)] for x, y in point_list]
        strokes.append(final_stroke)

        # Markers
        if return_markers:
            # Start marker
            all_markers.append({
                'x': final_stroke[0][0], 'y': final_stroke[0][1],
                'type': 'start', 'label': 'S', 'stroke_id': si
            })
            # End marker
            all_markers.append({
                'x': final_stroke[-1][0], 'y': final_stroke[-1][1],
                'type': 'stop', 'label': 'E', 'stroke_id': si
            })
            # Curve markers for arcs/loops at midpoint
            if shape_name in ('arc_right', 'arc_left', 'loop', 'u_arc'):
                mid_idx = len(final_stroke) // 2
                all_markers.append({
                    'x': final_stroke[mid_idx][0],
                    'y': final_stroke[mid_idx][1],
                    'type': 'curve', 'label': 'C', 'stroke_id': si
                })
            # Vertex markers for diag (sharp join)
            if shape_name == 'diag':
                q1 = len(final_stroke) // 4
                q3 = 3 * len(final_stroke) // 4
                for qi in (q1, q3):
                    all_markers.append({
                        'x': final_stroke[qi][0],
                        'y': final_stroke[qi][1],
                        'type': 'vertex', 'label': 'V', 'stroke_id': si
                    })

    if not strokes:
        return None
    if return_markers:
        return strokes, all_markers
    return strokes


def _load_cached_params(font_path, char):
    """Load cached optimised shape params from DB. Returns (params_list, score) or None."""
    try:
        db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        row = db.execute("""
            SELECT c.shape_params_cache
            FROM characters c
            JOIN fonts f ON c.font_id = f.id
            WHERE f.file_path = ? AND c.char = ?
              AND c.shape_params_cache IS NOT NULL
        """, (font_path, char)).fetchone()
        db.close()
        if row and row['shape_params_cache']:
            data = json.loads(row['shape_params_cache'])
            return np.array(data['params'], dtype=float), data['score']
    except Exception:
        pass
    return None


def _save_cached_params(font_path, char, params, score):
    """Save optimised shape params to DB cache."""
    try:
        cache_json = json.dumps({'params': params.tolist(), 'score': float(score)})
        db = sqlite3.connect(DB_PATH)
        # Find font_id from path
        font_row = db.execute("SELECT id FROM fonts WHERE file_path = ?",
                              (font_path,)).fetchone()
        if not font_row:
            db.close()
            return
        font_id = font_row[0]
        # Upsert character row
        row = db.execute("SELECT id FROM characters WHERE font_id = ? AND char = ?",
                         (font_id, char)).fetchone()
        if row:
            db.execute("UPDATE characters SET shape_params_cache = ? WHERE id = ?",
                       (cache_json, row[0]))
        else:
            db.execute(
                "INSERT INTO characters (font_id, char, shape_params_cache) VALUES (?, ?, ?)",
                (font_id, char, cache_json))
        db.commit()
        db.close()
    except Exception:
        pass


def _score_single_shape(params, shape_type, bbox, uncovered_pts, uncovered_tree,
                        n_uncovered, radius, snap_yi, snap_xi, w, h, n_pts=None):
    """Score a single shape against uncovered points only (for greedy fitting).
    Returns negative coverage of uncovered points minus snap penalty."""
    if n_pts is None:
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        n_pts = max(60, int((bw * bw + bh * bh) ** 0.5 / 1.5))
    pts = SHAPE_FNS[shape_type](tuple(params), bbox, offset=(0, 0), n_pts=n_pts)
    if len(pts) == 0:
        return 0.0
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    snapped_x = snap_xi[yi, xi].astype(float)
    snapped_y = snap_yi[yi, xi].astype(float)
    snapped = np.column_stack([snapped_x, snapped_y])
    snap_dist = np.sqrt((pts[:, 0] - snapped_x) ** 2 +
                        (pts[:, 1] - snapped_y) ** 2)
    off_mask = float(np.mean(snap_dist > 0.5))
    snap_penalty = 0.5 * off_mask
    hits = uncovered_tree.query_ball_point(snapped, radius)
    covered = set()
    for lst in hits:
        covered.update(lst)
    coverage = len(covered) / max(n_uncovered, 1)
    return -(coverage - snap_penalty)


def _setup_auto_fit(font_path, char, canvas_size, templates):
    """Setup data structures for auto_fit_strokes optimization.

    Args:
        font_path: Resolved font path
        char: Character to fit
        canvas_size: Canvas size
        templates: Shape templates from SHAPE_TEMPLATES

    Returns:
        Dict with all setup data, or None on failure
    """
    mask = render_glyph_mask(font_path, char, canvas_size)
    if mask is None:
        return None

    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    glyph_bbox = (float(cols.min()), float(rows.min()),
                  float(cols.max()), float(rows.max()))

    cloud = _make_point_cloud(mask, spacing=3)
    if len(cloud) < 10:
        return None
    cloud_tree = cKDTree(cloud)
    n_cloud = len(cloud)
    radius = _adaptive_radius(mask, spacing=3)
    h, w = mask.shape

    dist_map = distance_transform_edt(mask)
    _, snap_indices = distance_transform_edt(~mask, return_indices=True)
    snap_yi = snap_indices[0]
    snap_xi = snap_indices[1]

    shape_types = [t['shape'] for t in templates]
    bounds, slices = _get_param_bounds(templates)

    x0 = []
    for t in templates:
        x0.extend(t['params'])
    x0 = np.array(x0, dtype=float)

    bounds_lo = np.array([b[0] for b in bounds])
    bounds_hi = np.array([b[1] for b in bounds])

    return {
        'mask': mask, 'glyph_bbox': glyph_bbox, 'cloud': cloud,
        'cloud_tree': cloud_tree, 'n_cloud': n_cloud, 'radius': radius,
        'h': h, 'w': w, 'dist_map': dist_map, 'snap_yi': snap_yi, 'snap_xi': snap_xi,
        'shape_types': shape_types, 'bounds': bounds, 'slices': slices,
        'x0': x0, 'bounds_lo': bounds_lo, 'bounds_hi': bounds_hi,
    }


def _try_gradient_optimization(font_path, char, canvas_size):
    """Try DiffVG and affine optimization approaches.

    Args:
        font_path: Resolved font path
        char: Character to fit
        canvas_size: Canvas size

    Returns:
        Tuple of (strokes, score) for best result, or (None, 0.0)
    """
    best_result = (None, 0.0)

    # Try DiffVG first (gradient-based, typically better)
    diffvg_result = _optimize_diffvg(font_path, char, canvas_size)
    if diffvg_result is not None:
        dv_strokes, dv_score, _, _ = diffvg_result
        if dv_strokes and dv_score > 0:
            best_result = (dv_strokes, dv_score)

    # Also try affine optimization
    affine_result = _optimize_affine(font_path, char, canvas_size)
    if affine_result is not None:
        affine_strokes, affine_score, _, _ = affine_result
        aff_stroke_list = [[[round(float(x), 1), round(float(y), 1)] for x, y in s]
                           for s in affine_strokes if len(s) >= 2]
        if aff_stroke_list and affine_score > best_result[1]:
            best_result = (aff_stroke_list, affine_score)

    return best_result


def _greedy_shape_optimization(templates, setup, x0, elapsed_fn, time_budget):
    """Phase 1: Greedy per-shape optimization.

    Args:
        templates: Shape templates
        setup: Setup dict from _setup_auto_fit
        x0: Initial parameter vector
        elapsed_fn: Function returning elapsed time
        time_budget: Total time budget

    Returns:
        Optimized parameter vector
    """
    from scipy.optimize import differential_evolution, minimize

    greedy_x = x0.copy()
    uncovered_mask = np.ones(setup['n_cloud'], dtype=bool)

    for si in range(len(templates)):
        if elapsed_fn() >= time_budget * 0.4:
            break
        start, end = setup['slices'][si]
        stype = setup['shape_types'][si]
        s_bounds = setup['bounds'][start:end]
        s_x0 = greedy_x[start:end].copy()

        uncov_idx = np.where(uncovered_mask)[0]
        if len(uncov_idx) < 5:
            break
        uncov_pts = setup['cloud'][uncov_idx]
        uncov_tree = cKDTree(uncov_pts)

        s_args = (stype, setup['glyph_bbox'], uncov_pts, uncov_tree,
                  len(uncov_pts), setup['radius'], setup['snap_yi'],
                  setup['snap_xi'], setup['w'], setup['h'])

        s_lo = setup['bounds_lo'][start:end]
        s_hi = setup['bounds_hi'][start:end]

        def _score_single_clamped(p, *a, _lo=s_lo, _hi=s_hi):
            return _score_single_shape(np.clip(p, _lo, _hi), *a)

        nm_r = minimize(
            _score_single_clamped, s_x0, args=s_args, method='Nelder-Mead',
            options={'maxfev': 800, 'xatol': 0.2, 'fatol': 0.002, 'adaptive': True},
        )
        best_s = np.clip(nm_r.x, s_lo, s_hi).copy()
        best_sf = nm_r.fun

        if elapsed_fn() < time_budget * 0.35:
            try:
                s_clipped = np.clip(best_s, [b[0] for b in s_bounds],
                                    [b[1] for b in s_bounds])
                de_r = differential_evolution(
                    _score_single_shape, bounds=s_bounds, args=s_args,
                    x0=s_clipped, maxiter=30, popsize=12, tol=0.005,
                    seed=None, polish=False, disp=False,
                )
                if de_r.fun < best_sf:
                    best_s = de_r.x.copy()
            except Exception:
                pass

        greedy_x[start:end] = best_s

        # Mark covered points
        n_pts_shape = max(60, int(((setup['glyph_bbox'][2]-setup['glyph_bbox'][0])**2 +
                                    (setup['glyph_bbox'][3]-setup['glyph_bbox'][1])**2)**0.5 / 1.5))
        pts = SHAPE_FNS[stype](tuple(best_s), setup['glyph_bbox'], offset=(0, 0),
                               n_pts=n_pts_shape)
        if len(pts) > 0:
            xi = np.clip(np.round(pts[:, 0]).astype(int), 0, setup['w'] - 1)
            yi = np.clip(np.round(pts[:, 1]).astype(int), 0, setup['h'] - 1)
            snapped_x = setup['snap_xi'][yi, xi].astype(float)
            snapped_y = setup['snap_yi'][yi, xi].astype(float)
            snapped = np.column_stack([snapped_x, snapped_y])
            hits = setup['cloud_tree'].query_ball_point(snapped, setup['radius'])
            newly_covered = set()
            for lst in hits:
                newly_covered.update(lst)
            for idx in newly_covered:
                uncovered_mask[idx] = False

    return greedy_x


def _joint_optimization_cycle(best_x, best_fun, setup, elapsed_fn, time_budget,
                               stale_threshold=0.001, stale_cycles=2):
    """Phase 2: Joint NM→DE→NM refinement cycles until stagnation.

    Args:
        best_x: Starting parameter vector
        best_fun: Starting score
        setup: Setup dict from _setup_auto_fit
        elapsed_fn: Function returning elapsed time
        time_budget: Total time budget
        stale_threshold: Improvement threshold for stagnation detection
        stale_cycles: Number of stale cycles before stopping

    Returns:
        Tuple of (best_x, best_fun, cached_score)
    """
    from scipy.optimize import differential_evolution, minimize

    joint_args = (setup['shape_types'], setup['slices'], setup['glyph_bbox'],
                  setup['cloud_tree'], setup['n_cloud'], setup['radius'],
                  setup['snap_yi'], setup['snap_xi'], setup['w'], setup['h'],
                  setup['dist_map'])
    bounds = setup['bounds']
    bounds_lo = setup['bounds_lo']
    bounds_hi = setup['bounds_hi']

    def _clamp(x):
        return np.clip(x, bounds_lo, bounds_hi)

    def _score_all_clamped(params, *args):
        return _score_all_strokes(_clamp(params), *args)

    def _update_best(x, fun):
        nonlocal best_x, best_fun
        if fun < best_fun:
            best_x = x.copy()
            best_fun = fun

    def _perfect():
        return best_fun <= -0.99

    class _EarlyStop(Exception):
        pass

    stale_count = 0
    cached_score = None

    while not _perfect() and elapsed_fn() < time_budget and stale_count < stale_cycles:
        score_at_cycle_start = best_fun

        # NM refinement
        if not _perfect() and elapsed_fn() < time_budget:
            remaining_fev = max(500, int(min(30.0, time_budget - elapsed_fn()) / 0.0003))
            nm_result = minimize(
                _score_all_clamped, best_x, args=joint_args, method='Nelder-Mead',
                options={'maxfev': remaining_fev, 'xatol': 0.2, 'fatol': 0.0005,
                         'adaptive': True},
            )
            _update_best(_clamp(nm_result.x), nm_result.fun)

        # DE global search
        if not _perfect() and elapsed_fn() < time_budget:
            nm_x = best_x.copy()
            for i, (lo, hi) in enumerate(bounds):
                nm_x[i] = np.clip(nm_x[i], lo, hi)

            def _de_callback(xk, convergence=0):
                _update_best(xk, _score_all_strokes(xk, *joint_args))
                if _perfect() or elapsed_fn() >= time_budget:
                    raise _EarlyStop()

            try:
                de_result = differential_evolution(
                    _score_all_strokes, bounds=bounds, args=joint_args,
                    x0=nm_x, maxiter=200, popsize=20, tol=0.002,
                    seed=None, mutation=(0.5, 1.0), recombination=0.7,
                    polish=False, disp=False, callback=_de_callback,
                )
                _update_best(de_result.x, de_result.fun)
            except _EarlyStop:
                pass

        # NM polish
        if not _perfect() and elapsed_fn() < time_budget:
            remaining_fev = max(200, int(min(15.0, time_budget - elapsed_fn()) / 0.0003))
            nm2 = minimize(
                _score_all_clamped, best_x, args=joint_args, method='Nelder-Mead',
                options={'maxfev': remaining_fev, 'xatol': 0.1, 'fatol': 0.0005,
                         'adaptive': True},
            )
            _update_best(_clamp(nm2.x), nm2.fun)

        improvement = score_at_cycle_start - best_fun
        if improvement < stale_threshold:
            stale_count += 1
        else:
            stale_count = 0

        current_score = float(-best_fun)
        if cached_score is None or current_score > cached_score:
            cached_score = current_score

    return best_x, best_fun, cached_score


def _shapes_to_strokes(best_x, templates, setup, mask):
    """Convert optimized parameters to stroke point lists.

    Args:
        best_x: Optimized parameter vector
        templates: Shape templates
        setup: Setup dict
        mask: Glyph mask

    Returns:
        List of (shape_strokes, group_id) tuples for each group
    """
    from collections import OrderedDict

    best_shapes = _param_vector_to_shapes(best_x, setup['shape_types'],
                                          setup['slices'], setup['glyph_bbox'])

    groups = OrderedDict()
    _auto_gid = 1000
    for si, (tmpl, stype, pts) in enumerate(zip(templates, setup['shape_types'], best_shapes)):
        gid = tmpl.get('group')
        if gid is None:
            gid = _auto_gid
            _auto_gid += 1
        groups.setdefault(gid, []).append((si, stype, pts))

    result = []
    for gid, members in groups.items():
        shape_strokes = []
        for si, stype, pts in members:
            point_list = [(float(p[0]), float(p[1])) for p in pts]
            point_list = _smooth_stroke(point_list, sigma=2.0)
            point_list = _constrain_to_mask(point_list, mask)
            if len(point_list) >= 2:
                shape_strokes.append((si, stype, point_list))
        if shape_strokes:
            result.append((shape_strokes, gid))

    return result


def _generate_stroke_markers(strokes, return_markers):
    """Generate markers for stroke endpoints and curves.

    Args:
        strokes: List of stroke dicts with 'stroke', 'shapes', 'chain' keys
        return_markers: Whether to generate markers

    Returns:
        List of marker dicts if return_markers, else empty list
    """
    if not return_markers:
        return []

    all_markers = []
    for stroke_idx, stroke_info in enumerate(strokes):
        final_stroke = stroke_info['stroke']
        all_markers.append({
            'x': final_stroke[0][0], 'y': final_stroke[0][1],
            'type': 'start', 'label': 'S', 'stroke_id': stroke_idx
        })
        all_markers.append({
            'x': final_stroke[-1][0], 'y': final_stroke[-1][1],
            'type': 'stop', 'label': 'E', 'stroke_id': stroke_idx
        })

        # Add curve markers for arcs/loops
        shapes_to_check = stroke_info.get('chain', stroke_info['shapes'])
        for si, stype, point_list in shapes_to_check:
            if stype in ('arc_right', 'arc_left', 'loop', 'u_arc'):
                mid_pt = point_list[len(point_list) // 2]
                all_markers.append({
                    'x': round(mid_pt[0], 1), 'y': round(mid_pt[1], 1),
                    'type': 'curve', 'label': 'C', 'stroke_id': stroke_idx
                })

    return all_markers


def _build_shape_chain(start_shape, rest):
    """Build a chain of shapes by joining nearest endpoints.

    Args:
        start_shape: Initial shape tuple (si, stype, points)
        rest: List of remaining shapes to chain

    Returns:
        Tuple of (chain, total_gap) where chain is list of shapes
        and total_gap is sum of connection distances.
    """
    chain = [start_shape]
    remaining = list(rest)
    total_gap = 0.0

    while remaining:
        last_end = chain[-1][2][-1]
        best_dist = float('inf')
        best_idx = 0
        best_flip = False

        for ri, (rsi, rstype, rpts) in enumerate(remaining):
            d_start = (last_end[0] - rpts[0][0])**2 + (last_end[1] - rpts[0][1])**2
            d_end = (last_end[0] - rpts[-1][0])**2 + (last_end[1] - rpts[-1][1])**2
            if d_start < best_dist:
                best_dist, best_idx, best_flip = d_start, ri, False
            if d_end < best_dist:
                best_dist, best_idx, best_flip = d_end, ri, True

        total_gap += best_dist ** 0.5
        chosen = remaining.pop(best_idx)
        if best_flip:
            chosen = (chosen[0], chosen[1], list(reversed(chosen[2])))
        chain.append(chosen)

    return chain, total_gap


def _join_shapes_to_stroke(shape_strokes, return_markers, stroke_idx):
    """Join multiple shapes into a single stroke with optional markers.

    Args:
        shape_strokes: List of (si, stype, point_list) tuples
        return_markers: Whether to generate vertex markers
        stroke_idx: Current stroke index for marker assignment

    Returns:
        Dict with 'stroke' (point list), 'chain' (shape chain), 'markers' (list)
    """
    markers = []

    if len(shape_strokes) == 1:
        si, stype, point_list = shape_strokes[0]
        final_stroke = [[round(x, 1), round(y, 1)] for x, y in point_list]
        return {'stroke': final_stroke, 'chain': shape_strokes, 'markers': markers}

    # Try both directions and pick the one with smaller gaps
    s0 = shape_strokes[0]
    s0_flip = (s0[0], s0[1], list(reversed(s0[2])))
    chain_fwd, gap_fwd = _build_shape_chain(s0, shape_strokes[1:])
    chain_rev, gap_rev = _build_shape_chain(s0_flip, shape_strokes[1:])
    chain = chain_fwd if gap_fwd <= gap_rev else chain_rev

    # Combine points, adding vertex markers at joints
    combined = []
    for ci, (csi, cstype, cpts) in enumerate(chain):
        if ci > 0:
            jpt = cpts[0]
            if return_markers:
                markers.append({
                    'x': round(jpt[0], 1), 'y': round(jpt[1], 1),
                    'type': 'vertex', 'label': 'V', 'stroke_id': stroke_idx
                })
            # Skip overlapping points
            skip = 0
            if combined:
                last = combined[-1]
                for pi, p in enumerate(cpts):
                    if ((p[0] - last[0])**2 + (p[1] - last[1])**2) < 4.0:
                        skip = pi + 1
                    else:
                        break
            combined.extend(cpts[skip:])
        else:
            combined.extend(cpts)

    final_stroke = [[round(x, 1), round(y, 1)] for x, y in combined]
    return {'stroke': final_stroke, 'chain': chain, 'markers': markers}


def _simple_stroke_markers(strokes):
    """Generate simple start/stop markers for strokes (no curve markers).

    Args:
        strokes: List of stroke point lists

    Returns:
        List of marker dicts
    """
    markers = []
    for si, st in enumerate(strokes):
        markers.append({'x': st[0][0], 'y': st[0][1],
                        'type': 'start', 'label': 'S', 'stroke_id': si})
        markers.append({'x': st[-1][0], 'y': st[-1][1],
                        'type': 'stop', 'label': 'E', 'stroke_id': si})
    return markers


def _assemble_strokes_from_shapes(shape_groups, return_markers):
    """Convert shape groups to final strokes with optional markers.

    Args:
        shape_groups: List of (shape_strokes, group_id) from _shapes_to_strokes
        return_markers: Whether to generate markers

    Returns:
        Tuple of (strokes, markers) where strokes is list of point lists
    """
    strokes = []
    all_markers = []
    stroke_idx = 0

    for shape_strokes, gid in shape_groups:
        result = _join_shapes_to_stroke(shape_strokes, return_markers, stroke_idx)

        if len(result['stroke']) < 2:
            continue

        strokes.append(result['stroke'])
        all_markers.extend(result['markers'])

        if return_markers:
            final_stroke = result['stroke']
            all_markers.append({
                'x': final_stroke[0][0], 'y': final_stroke[0][1],
                'type': 'start', 'label': 'S', 'stroke_id': stroke_idx
            })
            all_markers.append({
                'x': final_stroke[-1][0], 'y': final_stroke[-1][1],
                'type': 'stop', 'label': 'E', 'stroke_id': stroke_idx
            })

            # Add curve markers for arcs/loops
            for si, stype, point_list in result['chain']:
                if stype in ('arc_right', 'arc_left', 'loop', 'u_arc'):
                    mid_pt = point_list[len(point_list) // 2]
                    all_markers.append({
                        'x': round(mid_pt[0], 1), 'y': round(mid_pt[1], 1),
                        'type': 'curve', 'label': 'C', 'stroke_id': stroke_idx
                    })

        stroke_idx += 1

    return strokes, all_markers


def auto_fit_strokes(font_path, char, canvas_size=224, return_markers=False):
    """Generate strokes by optimising shape parameters with a greedy per-shape
    approach followed by joint refinement.

    Phase 0: DiffVG/Affine template matching
    Phase 1: Greedy — optimise each shape against uncovered points
    Phase 2: Joint NM→DE→NM refinement cycles

    Caches winning params in DB so subsequent runs start from the best
    known position and can improve further.

    Returns list of strokes [[[x,y], ...], ...] or None.
    If return_markers=True returns (strokes, markers).
    """
    import time as _time

    templates = SHAPE_TEMPLATES.get(char)
    if not templates:
        return None

    font_path = resolve_font_path(font_path)

    # Setup optimization data structures
    setup = _setup_auto_fit(font_path, char, canvas_size, templates)
    if setup is None:
        return None

    _t_start = _time.monotonic()
    _TIME_BUDGET = 3600.0

    def _elapsed():
        return _time.monotonic() - _t_start

    # Load cached params
    rel_path = font_path
    if rel_path.startswith(BASE_DIR):
        rel_path = os.path.relpath(font_path, BASE_DIR)
    cached = _load_cached_params(rel_path, char)
    x0 = setup['x0']
    cached_score = None
    if cached is not None and len(cached[0]) == len(x0):
        x0 = cached[0]
        cached_score = cached[1]

    for i, (lo, hi) in enumerate(setup['bounds']):
        x0[i] = np.clip(x0[i], lo, hi)

    # ---- Phase 0: Try gradient-based optimization ----
    affine_strokes_result = _try_gradient_optimization(font_path, char, canvas_size)
    if affine_strokes_result[0] is not None and affine_strokes_result[1] >= 0.85:
        strokes = affine_strokes_result[0]
        if return_markers:
            return strokes, _simple_stroke_markers(strokes)
        return strokes

    # ---- Phase 1: Greedy per-shape optimization ----
    greedy_x = _greedy_shape_optimization(templates, setup, x0, _elapsed, _TIME_BUDGET)

    # ---- Phase 2: Joint refinement ----
    joint_args = (setup['shape_types'], setup['slices'], setup['glyph_bbox'],
                  setup['cloud_tree'], setup['n_cloud'], setup['radius'],
                  setup['snap_yi'], setup['snap_xi'], setup['w'], setup['h'],
                  setup['dist_map'])

    best_x = greedy_x.copy()
    best_fun = _score_all_strokes(greedy_x, *joint_args)

    x0_fun = _score_all_strokes(x0, *joint_args)
    if x0_fun < best_fun:
        best_x = x0.copy()
        best_fun = x0_fun

    best_x, best_fun, _ = _joint_optimization_cycle(
        best_x, best_fun, setup, _elapsed, _TIME_BUDGET)

    # Compare with gradient optimization result
    final_score = float(-best_fun)
    if affine_strokes_result[0] is not None and affine_strokes_result[1] > final_score:
        strokes = affine_strokes_result[0]
        if return_markers:
            return strokes, _simple_stroke_markers(strokes)
        return strokes

    # Cache winning params
    if cached_score is None or final_score > cached_score:
        _save_cached_params(rel_path, char, best_x, final_score)

    # Convert to strokes using helper
    shape_groups = _shapes_to_strokes(best_x, templates, setup, setup['mask'])
    strokes, all_markers = _assemble_strokes_from_shapes(shape_groups, return_markers)

    if not strokes:
        return None
    if return_markers:
        return strokes, all_markers
    return strokes


def _find_skeleton_segments(info):
    """Find and classify skeleton path segments between junctions.

    Returns list of segments, each with:
    - 'path': list of (x,y) points
    - 'start_junction': index or -1 for endpoint
    - 'end_junction': index or -1 for endpoint
    - 'angle': direction in degrees (0=right, 90=down)
    - 'length': number of pixels
    """
    from collections import deque

    adj = info['adj']
    junction_pixels = info['junction_pixels']
    junction_clusters = info['junction_clusters']
    endpoints = info['endpoints']

    segments = []

    # Map pixels to their junction cluster index
    pixel_to_junction = {}
    for i, cluster in enumerate(junction_clusters):
        for p in cluster:
            pixel_to_junction[p] = i

    # Find all segments between junctions/endpoints
    visited_pairs = set()

    # Start points: junction cluster border pixels and endpoints
    start_points = []
    for i, cluster in enumerate(junction_clusters):
        for p in cluster:
            for nb in adj[p]:
                if nb not in junction_pixels:
                    start_points.append((p, nb, i))  # (start, next, junction_idx)
    for ep in endpoints:
        if ep not in junction_pixels:
            start_points.append((ep, None, -1))

    for start, first_step, start_junc in start_points:
        if first_step is None:
            # Endpoint - find its neighbor
            neighbors = adj[start]
            if not neighbors:
                continue
            first_step = neighbors[0]

        # Trace path until we hit another junction or endpoint
        path = [start, first_step]
        current = first_step
        prev = start

        while current not in junction_pixels and current not in endpoints:
            neighbors = [n for n in adj[current] if n != prev]
            if not neighbors:
                break
            if len(neighbors) > 1:
                break  # Unexpected branching
            prev = current
            current = neighbors[0]
            path.append(current)

        # Determine end junction/endpoint
        if current in junction_pixels:
            end_junc = pixel_to_junction.get(current, -1)
        elif current in endpoints:
            end_junc = -1  # Mark as endpoint with special value
        else:
            end_junc = -1

        # Avoid duplicate segments
        pair = (min(start_junc, end_junc), max(start_junc, end_junc),
                min(start, current), max(start, current))
        if pair in visited_pairs:
            continue
        visited_pairs.add(pair)

        if len(path) >= 2:
            dx = path[-1][0] - path[0][0]
            dy = path[-1][1] - path[0][1]
            angle = np.degrees(np.arctan2(dy, dx))

            segments.append({
                'path': path,
                'start': path[0],
                'end': path[-1],
                'start_junction': start_junc,
                'end_junction': end_junc,
                'angle': angle,
                'length': len(path),
            })

    return segments


def _generate_straight_line(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Generate a straight line of pixels from start to end.

    Delegates to stroke_lib.utils.geometry.generate_straight_line.
    """
    return sl_generate_straight_line(start, end)


def _snap_to_skeleton(point: Tuple, skel_set: Set) -> Tuple:
    """Find the nearest skeleton pixel to a point.

    Args:
        point: (x, y) point to snap
        skel_set: set of skeleton pixels

    Returns:
        Nearest skeleton pixel, or original point if skel_set is empty.
    """
    point = tuple(point) if not isinstance(point, tuple) else point
    if point in skel_set or not skel_set:
        return point

    min_dist = float('inf')
    nearest = point
    for p in skel_set:
        d = (p[0] - point[0])**2 + (p[1] - point[1])**2
        if d < min_dist:
            min_dist = d
            nearest = p
    return nearest


def _trace_skeleton_path(start, end, adj, skel_set, max_steps=500, avoid_pixels=None,
                         direction: str = None):
    """Trace a path along skeleton pixels from start to end using BFS.

    Args:
        start: (x, y) starting point (should be on or near skeleton)
        end: (x, y) ending point (should be on or near skeleton)
        adj: adjacency dict from _analyze_skeleton
        skel_set: set of skeleton pixels
        max_steps: maximum path length to prevent infinite loops
        avoid_pixels: set of pixels to avoid (already traced in this stroke)
        direction: 'down', 'up', 'left', 'right' - bias for initial direction

    Returns:
        List of (x, y) points along the path, or None if no path found.
    """
    if avoid_pixels is None:
        avoid_pixels = set()

    # Snap start/end to skeleton
    start = _snap_to_skeleton(start, skel_set)
    end = _snap_to_skeleton(end, skel_set)

    if start == end:
        return [start]

    # BFS to find path
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()

        if len(path) > max_steps:
            continue

        # Check if we reached the end (within 3 pixels)
        dist_to_end = ((current[0] - end[0])**2 + (current[1] - end[1])**2)**0.5
        if dist_to_end < 3 or current == end:
            return path + [end] if current != end else path

        # Get neighbors, optionally sorted by direction preference
        neighbors = adj.get(current, [])
        if direction and len(path) < DIRECTION_BIAS_PIXELS:
            if direction == 'down':
                neighbors = sorted(neighbors, key=lambda p: -p[1])
            elif direction == 'up':
                neighbors = sorted(neighbors, key=lambda p: p[1])
            elif direction == 'right':
                neighbors = sorted(neighbors, key=lambda p: -p[0])
            elif direction == 'left':
                neighbors = sorted(neighbors, key=lambda p: p[0])

        for neighbor in neighbors:
            if neighbor not in visited:
                # Skip pixels we've already traced (except near start/end)
                if neighbor in avoid_pixels:
                    dist_to_start = ((neighbor[0] - start[0])**2 + (neighbor[1] - start[1])**2)**0.5
                    dist_to_end_n = ((neighbor[0] - end[0])**2 + (neighbor[1] - end[1])**2)**0.5
                    if dist_to_start > NEAR_ENDPOINT_DIST and dist_to_end_n > NEAR_ENDPOINT_DIST:
                        continue
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    # No path found - return None (don't fallback to allowing double-back)
    return None


def trace_segment(start: Tuple[int, int], end: Tuple[int, int],
                  config: SegmentConfig, adj: Dict, skel_set: Set,
                  avoid_pixels: Set = None,
                  fallback_avoid: Set = None) -> List[Tuple[int, int]]:
    """Unified segment tracing - handles both straight lines and skeleton paths.

    Args:
        start: (x, y) starting point
        end: (x, y) ending point
        config: SegmentConfig with direction and straight settings
        adj: adjacency dict from _analyze_skeleton
        skel_set: set of skeleton pixels
        avoid_pixels: set of pixels to avoid (primary)
        fallback_avoid: set of pixels to try avoiding if primary fails (before giving up)

    Returns:
        List of (x, y) points along the segment, or None if no path found.
    """
    if config.straight:
        # Draw a direct line, ignoring skeleton
        return _generate_straight_line(start, end)

    # Trace along skeleton with primary avoidance
    traced = _trace_skeleton_path(start, end, adj, skel_set,
                                   avoid_pixels=avoid_pixels,
                                   direction=config.direction)

    # Fallback 1: try with fallback_avoid set (if provided)
    if traced is None and fallback_avoid is not None:
        traced = _trace_skeleton_path(start, end, adj, skel_set,
                                       avoid_pixels=fallback_avoid,
                                       direction=config.direction)

    # Fallback 2: retry without any avoidance
    if traced is None:
        traced = _trace_skeleton_path(start, end, adj, skel_set,
                                       avoid_pixels=None,
                                       direction=config.direction)

    return traced


def _trace_to_region(start, target_region, bbox, adj, skel_set, max_steps=500, avoid_pixels=None):
    """Trace along skeleton from start until entering target region.

    Instead of pathfinding to a specific point, this follows the skeleton
    and stops as soon as any point falls within the target region.

    Args:
        start: (x, y) starting point
        target_region: numpad region number (1-9) to reach
        bbox: (x_min, y_min, x_max, y_max) glyph bounding box
        adj: adjacency dict from _analyze_skeleton
        skel_set: set of skeleton pixels
        max_steps: maximum path length
        avoid_pixels: set of pixels to avoid

    Returns:
        List of (x, y) points along the path, or None if no path found.
    """
    if avoid_pixels is None:
        avoid_pixels = set()

    # Snap start to skeleton
    start = _snap_to_skeleton(start, skel_set)

    # Check if we're already in the target region
    if point_in_region(start, target_region, bbox):
        return [start]

    # Calculate target region center for direction guidance
    x_min, y_min, x_max, y_max = bbox
    frac_x, frac_y = NUMPAD_POS[target_region]
    target_center = (x_min + frac_x * (x_max - x_min),
                     y_min + frac_y * (y_max - y_min))

    # Use greedy traversal - prefer neighbors closer to target region
    path = [start]
    visited = {start}
    current = start

    for _ in range(max_steps):
        neighbors = adj.get(current, [])
        if not neighbors:
            break

        # Filter out visited and avoided pixels (strictly - no exceptions)
        valid_neighbors = []
        for n in neighbors:
            if n in visited:
                continue
            if n in avoid_pixels:
                continue  # Strictly avoid all already-traced pixels
            valid_neighbors.append(n)

        if not valid_neighbors:
            break

        # Choose next neighbor - prefer the one closest to target region center
        best = None
        best_dist = float('inf')
        for n in valid_neighbors:
            # Distance to target region center
            dist = ((n[0] - target_center[0])**2 + (n[1] - target_center[1])**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best = n

        if best is None:
            break

        visited.add(best)
        path.append(best)
        current = best

        # Check if we've entered the target region
        if point_in_region(current, target_region, bbox):
            return path

    # Didn't reach target region - return path so far (partial success)
    # This allows the stroke to continue as far as possible
    return path if len(path) > 1 else None


def _resample_path(path, num_points=20):
    """Resample a path to have approximately num_points evenly spaced points.

    Delegates to stroke_lib.utils.geometry.resample_path.
    """
    return sl_resample_path(path, num_points)


def _quick_stroke_score(strokes, mask, stroke_width=8):
    """Quick scoring function for comparing stroke variants.

    Returns a score based on coverage (how much of the glyph is covered)
    minus overshoot penalty (strokes outside the glyph).
    Higher is better.
    """
    h, w = mask.shape
    stroke_img = Image.new('L', (w, h), 255)
    draw = ImageDraw.Draw(stroke_img)

    for stroke in strokes:
        if len(stroke) >= 2:
            points = [(p[0], p[1]) for p in stroke]
            draw.line(points, fill=0, width=stroke_width)

    stroke_mask = np.array(stroke_img) < 128
    glyph_mask = mask > 0

    glyph_pixels = glyph_mask.sum()
    if glyph_pixels == 0:
        return 0.0

    covered = (glyph_mask & stroke_mask).sum()
    coverage = covered / glyph_pixels

    stroke_pixels = stroke_mask.sum()
    if stroke_pixels == 0:
        return 0.0

    outside = (stroke_mask & ~glyph_mask).sum()
    overshoot = outside / stroke_pixels

    # Score: coverage minus overshoot penalty
    return coverage - (overshoot * 0.5)


# ============================================================================
# Helper functions for minimal_strokes_from_skeleton
# ============================================================================

def _numpad_to_pixel(region, bbox):
    """Map numpad region (1-9) to pixel coordinates (center of region)."""
    frac_x, frac_y = NUMPAD_POS[region]
    x = bbox[0] + frac_x * (bbox[2] - bbox[0])
    y = bbox[1] + frac_y * (bbox[3] - bbox[1])
    return (x, y)


def _extract_waypoint_region(wp):
    """Extract the region number from a waypoint (handles tuples and plain ints)."""
    if isinstance(wp, tuple):
        return wp[0] if isinstance(wp[0], int) else None
    return wp if isinstance(wp, int) else None


def _is_vertical_stroke(stroke_template):
    """Check if a stroke template represents a vertical line."""
    if len(stroke_template) != 2:
        return False
    r1 = _extract_waypoint_region(stroke_template[0])
    r2 = _extract_waypoint_region(stroke_template[1])
    if r1 is None or r2 is None:
        return False
    # Same column in numpad: (7,4,1), (8,5,2), (9,6,3)
    col1 = (r1 - 1) % 3
    col2 = (r2 - 1) % 3
    return col1 == col2


def _find_best_vertical_segment(vertical_segments, template_start, template_end):
    """Find the vertical skeleton segment(s) closest to template positions.

    Chains connected vertical segments to get the full stroke.
    """
    if not vertical_segments:
        return None

    # Filter to only truly vertical segments (not diagonals)
    truly_vertical = [s for s in vertical_segments if 75 <= abs(s['angle']) <= 105]
    if not truly_vertical:
        truly_vertical = vertical_segments  # Fallback

    # Build a graph of connected vertical segments
    junction_to_segs = defaultdict(list)
    for i, seg in enumerate(truly_vertical):
        if seg['start_junction'] >= 0:
            junction_to_segs[seg['start_junction']].append(i)
        if seg['end_junction'] >= 0:
            junction_to_segs[seg['end_junction']].append(i)

    # Find chains by grouping segments that share any junction
    visited = set()
    chains = []

    for i in range(len(truly_vertical)):
        if i in visited:
            continue

        # BFS to find all connected segments
        chain = []
        queue = [i]
        while queue:
            seg_idx = queue.pop(0)
            if seg_idx in visited:
                continue
            visited.add(seg_idx)
            chain.append(seg_idx)

            seg = truly_vertical[seg_idx]
            # Find segments sharing start or end junction
            for junc in [seg['start_junction'], seg['end_junction']]:
                if junc >= 0:
                    for other_idx in junction_to_segs[junc]:
                        if other_idx not in visited:
                            queue.append(other_idx)

        if chain:
            chains.append(chain)

    # For each chain, get the full path endpoints
    best = None
    best_score = float('inf')

    for chain in chains:
        # Get all endpoints of segments in chain
        points = []
        for seg_idx in chain:
            seg = truly_vertical[seg_idx]
            points.append(seg['start'])
            points.append(seg['end'])

        if not points:
            continue

        # Find topmost and bottommost points
        points.sort(key=lambda p: p[1])  # Sort by y
        top = points[0]
        bottom = points[-1]

        # Score by match to template
        d1 = ((top[0] - template_start[0])**2 +
              (top[1] - template_start[1])**2)**0.5
        d2 = ((bottom[0] - template_end[0])**2 +
              (bottom[1] - template_end[1])**2)**0.5
        d1r = ((bottom[0] - template_start[0])**2 +
               (bottom[1] - template_start[1])**2)**0.5
        d2r = ((top[0] - template_end[0])**2 +
               (top[1] - template_end[1])**2)**0.5

        score = min(d1 + d2, d1r + d2r)
        if score < best_score:
            best_score = score
            if d1 + d2 <= d1r + d2r:
                best = (top, bottom)
            else:
                best = (bottom, top)

    return best


def _select_best_variant(char, font_path, canvas_size):
    """Try all variants for a character and return the best one.

    Returns (best_strokes, best_variant_name) or (None, None) if no variants.
    """
    variants = NUMPAD_TEMPLATE_VARIANTS.get(char)
    if not variants:
        return None, None

    if len(variants) == 1:
        return list(variants.keys())[0], list(variants.values())[0]

    # Try all variants and pick the best
    font_path_resolved = resolve_font_path(font_path)
    mask = render_glyph_mask(font_path_resolved, char, canvas_size)
    if mask is None:
        return None, None

    best_strokes = None
    best_variant = None
    best_score = -1

    for var_name, variant_template in variants.items():
        strokes = minimal_strokes_from_skeleton(
            font_path, char, canvas_size, trace_paths=True,
            template=variant_template, return_variant=False
        )
        if strokes:
            score = _quick_stroke_score(strokes, mask)
            if score > best_score:
                best_score = score
                best_strokes = strokes
                best_variant = var_name

    return best_variant, best_strokes


def _find_terminal_waypoint(wp_val, region, template_pos, skel_list, info, bbox,
                            top_bound, bot_bound, mid_x, next_direction):
    """Find the skeleton position for a terminal waypoint (plain int)."""
    # Filter skeleton pixels to those actually IN the target region
    region_pixels = [p for p in skel_list if point_in_region(p, region, bbox)]

    # If no pixels in exact region, fall back to vertical third
    if not region_pixels:
        if template_pos[1] < top_bound:  # Top (positions 7,8,9)
            region_pixels = [p for p in skel_list if p[1] < top_bound]
        elif template_pos[1] > bot_bound:  # Bottom (positions 1,2,3)
            region_pixels = [p for p in skel_list if p[1] > bot_bound]
        else:  # Middle (positions 4,5,6)
            region_pixels = [p for p in skel_list if top_bound <= p[1] <= bot_bound]

    if not region_pixels:
        return None

    extremum = None

    # If there's a direction hint for the next segment, place waypoint
    # at the extreme position to give room to move in that direction
    if next_direction == 'down':
        extremum = min(region_pixels, key=lambda p: p[1])
    elif next_direction == 'up':
        extremum = max(region_pixels, key=lambda p: p[1])
    elif next_direction == 'left':
        extremum = max(region_pixels, key=lambda p: p[0])
    elif next_direction == 'right':
        extremum = min(region_pixels, key=lambda p: p[0])
    # Check if there's an endpoint in this region - prefer it for terminals
    elif info['endpoints']:
        endpoints_in_region = [ep for ep in info['endpoints'] if point_in_region(ep, region, bbox)]
        if endpoints_in_region:
            extremum = min(endpoints_in_region, key=lambda p:
                          (p[0] - template_pos[0])**2 + (p[1] - template_pos[1])**2)

    if extremum is None:
        # No hint - use default logic based on position
        is_corner_pos = region in [7, 9, 1, 3]

        if is_corner_pos:
            def corner_dist(p):
                dx = abs(p[0] - template_pos[0])
                dy = abs(p[1] - template_pos[1])
                return dx + dy
            extremum = min(region_pixels, key=corner_dist)
        elif region == 8:
            extremum = min(region_pixels, key=lambda p: p[1])
        elif region == 2:
            extremum = max(region_pixels, key=lambda p: p[1])
        elif template_pos[0] < mid_x:
            extremum = min(region_pixels, key=lambda p: p[0])
        else:
            extremum = max(region_pixels, key=lambda p: p[0])

    return extremum


def _find_vertex_waypoint(region, template_pos, skel_list, mask, bbox,
                          top_bound, bot_bound, mid_x, mid_y, rows, cols):
    """Find the skeleton position for a vertex waypoint v(n).

    Returns (skeleton_point, apex_extension_or_None).
    apex_extension is ('top'/'bottom', [apex_x, apex_y]) if glyph extends beyond skeleton.
    """
    if template_pos[1] < top_bound:  # Top row (7,8,9)
        if skel_list:
            topmost = min(skel_list, key=lambda p: p[1])
            skel_x, skel_y = topmost[0], topmost[1]
            # Search for glyph pixels in a narrow band above skeleton
            col_start = max(0, int(skel_x) - 5)
            col_end = min(mask.shape[1], int(skel_x) + 6)
            glyph_cols = cols[(cols >= col_start) & (cols < col_end)]
            glyph_rows = rows[(cols >= col_start) & (cols < col_end)]
            if len(glyph_rows) > 0:
                glyph_top_idx = glyph_rows.argmin()
                glyph_top_y = glyph_rows[glyph_top_idx]
                glyph_top_x = glyph_cols[glyph_top_idx]
                if glyph_top_y < skel_y:
                    return (skel_x, skel_y), ('top', [float(glyph_top_x), float(glyph_top_y)])
            return (skel_x, skel_y), None
        return (template_pos[0], template_pos[1]), None

    elif template_pos[1] > bot_bound:  # Bottom row (1,2,3)
        if skel_list:
            bottommost = max(skel_list, key=lambda p: p[1])
            skel_x, skel_y = bottommost[0], bottommost[1]
            col_start = max(0, int(skel_x) - 5)
            col_end = min(mask.shape[1], int(skel_x) + 6)
            glyph_cols = cols[(cols >= col_start) & (cols < col_end)]
            glyph_rows = rows[(cols >= col_start) & (cols < col_end)]
            if len(glyph_rows) > 0:
                glyph_bot_idx = glyph_rows.argmax()
                glyph_bot_y = glyph_rows[glyph_bot_idx]
                glyph_bot_x = glyph_cols[glyph_bot_idx]
                if glyph_bot_y > skel_y:
                    return (skel_x, skel_y), ('bottom', [float(glyph_bot_x), float(glyph_bot_y)])
            return (skel_x, skel_y), None
        return (template_pos[0], template_pos[1]), None

    else:  # Middle row (4,5,6) - waist level
        waist_tolerance = (bbox[3] - bbox[1]) * 0.15
        waist_pixels = [p for p in skel_list
                       if abs(p[1] - mid_y) < waist_tolerance]
        if waist_pixels:
            if template_pos[0] < mid_x:  # Left side (4)
                vertex_pt = min(waist_pixels, key=lambda p: p[0])
            else:  # Right side (6)
                vertex_pt = max(waist_pixels, key=lambda p: p[0])
            return vertex_pt, None
        # Fallback - find nearest
        if skel_list:
            skel_tree = cKDTree(skel_list)
            _, idx = skel_tree.query(template_pos)
            return skel_list[idx], None
        return (template_pos[0], template_pos[1]), None


def _find_curve_waypoint(region, template_pos, skel_list, skel_tree, mid_x, mid_y, waist_margin):
    """Find the skeleton position for a curve waypoint c(n)."""
    # Filter skeleton pixels well above or below waist (mid_y)
    if template_pos[1] < mid_y:  # Above waist (positions 7,8,9)
        region_pixels = [p for p in skel_list if p[1] < mid_y - waist_margin]
    else:  # Below waist (positions 1,2,3)
        region_pixels = [p for p in skel_list if p[1] > mid_y + waist_margin]

    if region_pixels:
        # Find apex based on template position direction
        if template_pos[0] > mid_x:  # Right side (positions 3,6,9)
            apex = max(region_pixels, key=lambda p: p[0])
        elif template_pos[0] < mid_x:  # Left side (positions 1,4,7)
            apex = min(region_pixels, key=lambda p: p[0])
        else:  # Center (positions 2,5,8)
            _, idx = skel_tree.query(template_pos)
            apex = skel_list[idx]
        return apex

    # Fallback to nearest skeleton pixel
    _, idx = skel_tree.query(template_pos)
    return skel_list[idx]


def _trace_stroke_path(stroke_points, waypoint_info, segment_directions, segment_straight,
                       info, global_traced):
    """Trace skeleton paths between consecutive waypoints.

    Returns the full traced path as a list of (x, y) tuples.
    """
    full_path = []
    already_traced = set(global_traced)
    arrival_branch = set()

    current_pt = (int(round(stroke_points[0][0])), int(round(stroke_points[0][1])))

    for i in range(len(stroke_points) - 1):
        start_pt = current_pt
        end_pt = (int(round(stroke_points[i+1][0])), int(round(stroke_points[i+1][1])))

        seg_direction = segment_directions.get(i)
        seg_straight = segment_straight.get(i, False)
        current_is_intersection = waypoint_info[i][1] if i < len(waypoint_info) else False
        target_is_curve, target_is_intersection, target_region = waypoint_info[i + 1] if i + 1 < len(waypoint_info) else (False, False, None)

        config = SegmentConfig(direction=seg_direction, straight=seg_straight)

        # Determine avoidance strategy based on context
        if config.straight:
            traced = _generate_straight_line(start_pt, end_pt)
        elif target_is_intersection:
            traced = trace_segment(start_pt, end_pt, config, info['adj'], info['skel_set'],
                                   avoid_pixels=None)
            if traced and len(traced) >= 2:
                arrival_branch = set(traced[-ARRIVAL_BRANCH_SIZE:])
        elif current_is_intersection and arrival_branch:
            traced = trace_segment(start_pt, end_pt, config, info['adj'], info['skel_set'],
                                   avoid_pixels=already_traced,
                                   fallback_avoid=arrival_branch)
            arrival_branch = set()
        elif target_is_curve and target_region is not None:
            # Get bbox from info if available, otherwise skip region-based tracing
            traced = _trace_to_region(start_pt, target_region, info.get('bbox'), info['adj'], info['skel_set'],
                                      avoid_pixels=already_traced)
            if traced is None:
                traced = _trace_to_region(start_pt, target_region, info.get('bbox'), info['adj'], info['skel_set'],
                                          avoid_pixels=None)
        else:
            traced = trace_segment(start_pt, end_pt, config, info['adj'], info['skel_set'],
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

    return full_path, already_traced


def _apply_apex_extensions(full_path, stroke_points, apex_extensions):
    """Apply apex extensions to the traced path.

    Inserts glyph apex points at the appropriate positions.
    """
    for point_idx, (direction, apex_pt) in apex_extensions.items():
        if point_idx < len(stroke_points):
            skel_pt = stroke_points[point_idx]
            skel_x, skel_y = int(round(skel_pt[0])), int(round(skel_pt[1]))

            for j, fp in enumerate(full_path):
                if abs(fp[0] - skel_x) <= 2 and abs(fp[1] - skel_y) <= 2:
                    apex_tuple = (int(round(apex_pt[0])), int(round(apex_pt[1])))
                    if direction == 'top':
                        full_path.insert(j, apex_tuple)
                    else:  # 'bottom'
                        full_path.insert(j + 1, apex_tuple)
                    break


def minimal_strokes_from_skeleton(font_path, char, canvas_size=224, trace_paths=True,
                                   template=None, return_variant=False):
    """Generate minimal strokes by combining template topology with skeleton keypoints.

    Uses MinimalStrokePipeline to:
    1. Analyze skeleton to find key points
    2. Resolve template waypoints to skeleton positions
    3. Trace paths between waypoints
    4. Evaluate and select best variant

    Args:
        font_path: Path to font file
        char: Character to generate strokes for
        canvas_size: Size of canvas (default 224)
        trace_paths: Whether to trace paths between waypoints
        template: Optional specific template to use (skips variant selection)
        return_variant: If True, returns (strokes, variant_name) tuple

    Returns:
        List of strokes as [[[x,y], ...], ...] or None on failure.
        If return_variant=True, returns (strokes, variant_name) or (None, None).
    """
    pipeline = MinimalStrokePipeline(font_path, char, canvas_size)

    # If no template provided, evaluate all variants and pick best
    if template is None:
        result = pipeline.evaluate_all_variants()
        if return_variant:
            return result.strokes, result.variant_name
        return result.strokes

    # Use provided template directly
    strokes = pipeline.run(template, trace_paths=trace_paths)
    if return_variant:
        return strokes, None
    return strokes



def template_to_strokes(font_path, char, canvas_size=224, return_markers=False):
    """Generate strokes using numpad-grid template and contour midpoint algorithm.

    Returns list of strokes as [[[x,y], ...], ...] or None if template not available.
    If return_markers=True, returns (strokes, markers) where markers is a list of
    waypoint dicts with type 'start'/'stop'/'vertex'/'curve'.
    """
    template = NUMPAD_TEMPLATES.get(char)
    if not template:
        return None

    font_path = resolve_font_path(font_path)
    mask = render_glyph_mask(font_path, char, canvas_size)
    if mask is None:
        return None

    # Compute glyph bounding box from mask
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    glyph_bbox = (float(cols.min()), float(rows.min()),
                  float(cols.max()), float(rows.max()))
    centroid = (float(cols.mean()), float(rows.mean()))

    # Find skeleton features for waypoint placement
    skel_features = _find_skeleton_waypoints(mask, glyph_bbox)

    # Get pixel contours and build segment list
    pixel_contours = _get_pixel_contours(font_path, char, canvas_size)
    if not pixel_contours:
        return None
    segments = _contour_segments(pixel_contours)
    if not segments:
        return None

    all_markers = []

    strokes = []
    for si, stroke_wps in enumerate(template):
        # Build guide path
        guide = _build_guide_path(stroke_wps, glyph_bbox, mask, skel_features)
        if len(guide) < 2:
            continue

        # Smooth the guide path and constrain to mask
        smoothed = _smooth_stroke(guide, sigma=2.0)
        constrained = _constrain_to_mask(smoothed, mask)

        if len(constrained) >= 2:
            final_stroke = [[round(x, 1), round(y, 1)] for x, y in constrained]
            strokes.append(final_stroke)

            # Collect waypoint markers after stroke is built — terminals
            # use the final stroke endpoints (already centered by cross-sections)
            if return_markers:
                parsed = [_parse_waypoint(wp) for wp in stroke_wps]
                n_wp_local = len(parsed)
                for wi, (region, kind) in enumerate(parsed):
                    if kind == 'terminal':
                        mtype = 'start' if wi == 0 else 'stop'
                        if wi == 0:
                            pos = (final_stroke[0][0], final_stroke[0][1])
                        else:
                            pos = (final_stroke[-1][0], final_stroke[-1][1])
                    elif kind == 'vertex':
                        mtype = 'vertex'
                        frac = wi / max(n_wp_local - 1, 1)
                        idx = int(round(frac * (len(final_stroke) - 1)))
                        pos = (final_stroke[idx][0], final_stroke[idx][1])
                    else:
                        mtype = 'curve'
                        frac = wi / max(n_wp_local - 1, 1)
                        idx = int(round(frac * (len(final_stroke) - 1)))
                        pos = (final_stroke[idx][0], final_stroke[idx][1])
                    if mtype == 'start':
                        mlabel = 'S'
                    elif mtype == 'stop':
                        mlabel = 'E'
                    else:
                        mlabel = str(stroke_wps[wi])
                    all_markers.append({
                        'x': round(pos[0], 1), 'y': round(pos[1], 1),
                        'type': mtype, 'label': mlabel,
                        'stroke_id': si
                    })

    if not strokes:
        return (None, []) if return_markers else None
    if return_markers:
        return strokes, all_markers
    return strokes


def _point_to_region(x, y, bbox):
    """Map a point to a 3x3 grid region (TL, TC, TR, ML, MC, MR, BL, BC, BR)."""
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min
    if w < 1 or h < 1:
        return 'MC'
    rx = (x - x_min) / w
    ry = (y - y_min) / h
    col = 'L' if rx < 0.33 else ('C' if rx < 0.67 else 'R')
    row = 'T' if ry < 0.33 else ('M' if ry < 0.67 else 'B')
    return row + col


def _stroke_signature(stroke, bbox):
    """Get (start_region, mid_region, end_region) for a stroke."""
    s = _point_to_region(stroke[0][0], stroke[0][1], bbox)
    e = _point_to_region(stroke[-1][0], stroke[-1][1], bbox)
    mid = stroke[len(stroke) // 2]
    m = _point_to_region(mid[0], mid[1], bbox)
    return (s, m, e)


def _template_match_score(sig, template_stroke):
    """Score how well a stroke signature matches a template stroke.
    Returns 0-5 (higher = better). Checks both stroke orientations."""
    s, m, e = sig
    ts = template_stroke[0]
    te = template_stroke[-1]
    tv = template_stroke[1] if len(template_stroke) > 2 else None

    best = 0
    for cs, ce in [(s, e), (e, s)]:
        score = 0
        if cs == ts:
            score += 2
        if ce == te:
            score += 2
        if tv and m == tv:
            score += 1
        best = max(best, score)
    return best


def _strokes_bbox(strokes):
    """Bounding box of all stroke points as (x_min, y_min, x_max, y_max)."""
    all_x = [p[0] for s in strokes for p in s]
    all_y = [p[1] for s in strokes for p in s]
    return (min(all_x), min(all_y), max(all_x), max(all_y))


def apply_stroke_template(strokes, char):
    """Use letter template to merge skeleton strokes to the expected count.

    If the skeleton produced more strokes than the template expects,
    iteratively merge the pair whose result best matches a template stroke.
    """
    template = LETTER_TEMPLATES.get(char)
    if not template or not strokes:
        return strokes
    target_count = len(template)
    if len(strokes) <= target_count:
        return strokes

    bbox = _strokes_bbox(strokes)

    while len(strokes) > target_count:
        best_score = -float('inf')
        best_pair = None
        best_merged = None

        for i in range(len(strokes)):
            for j in range(i + 1, len(strokes)):
                si, sj = strokes[i], strokes[j]

                # Try all 4 endpoint-to-endpoint orientations
                combos = [
                    (si[-1], sj[0], si + sj),
                    (si[-1], sj[-1], si + list(reversed(sj))),
                    (si[0], sj[0], list(reversed(si)) + sj),
                    (si[0], sj[-1], list(reversed(si)) + list(reversed(sj))),
                ]

                for pt_a, pt_b, merged in combos:
                    gap = ((pt_a[0] - pt_b[0]) ** 2 + (pt_a[1] - pt_b[1]) ** 2) ** 0.5
                    sig = _stroke_signature(merged, bbox)
                    match = max(_template_match_score(sig, ts) for ts in template)
                    # Penalize large gaps (normalize by bbox diagonal)
                    diag = ((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) ** 0.5
                    gap_penalty = gap / max(diag, 1.0) * 4.0
                    score = match - gap_penalty

                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)
                        best_merged = merged

        if best_pair is None:
            break

        i, j = best_pair
        strokes = [s for k, s in enumerate(strokes) if k != i and k != j]
        strokes.append(best_merged)

    return strokes


def _region_center(region, bbox):
    """Return the (x, y) center of a named region within a bounding box."""
    x_min, y_min, x_max, y_max = bbox
    w, h = x_max - x_min, y_max - y_min
    col_map = {'L': x_min + w / 6, 'C': x_min + w / 2, 'R': x_min + 5 * w / 6}
    row_map = {'T': y_min + h / 6, 'M': y_min + h / 2, 'B': y_min + 5 * h / 6}
    return (col_map[region[1]], row_map[region[0]])


def _match_strokes_to_template(strokes, template, bbox):
    """Assign each stroke to the best-matching template entry.

    Returns list of template indices, one per stroke. Uses greedy
    best-first matching. Unmatched strokes get index -1.
    """
    n_strokes = len(strokes)
    n_tmpl = len(template)

    # Score matrix: strokes x template entries
    scores = []
    for si, stroke in enumerate(strokes):
        sig = _stroke_signature(stroke, bbox)
        row = [_template_match_score(sig, ts) for ts in template]
        scores.append(row)

    assigned = [-1] * n_strokes
    used_tmpl = set()

    # Greedy: pick highest score pairs first
    pairs = []
    for si in range(n_strokes):
        for ti in range(n_tmpl):
            pairs.append((scores[si][ti], si, ti))
    pairs.sort(reverse=True)

    for score, si, ti in pairs:
        if assigned[si] >= 0 or ti in used_tmpl:
            continue
        assigned[si] = ti
        used_tmpl.add(ti)

    return assigned


def adjust_stroke_paths(strokes, char, mask):
    """Nudge stroke paths toward template via regions they should pass through.

    For each stroke matched to a template entry with exactly one via region
    (3-element tuple), if the stroke's midpoint doesn't reach that region,
    smoothly pull the middle portion toward it, constraining points to stay
    inside the glyph mask.

    Templates with more than 3 elements (multi-via) are skipped since
    the single-via nudge logic doesn't apply cleanly.
    """
    template = LETTER_TEMPLATES.get(char)
    if not template or not strokes:
        return strokes

    bbox = _strokes_bbox(strokes)
    h, w = mask.shape
    assigned = _match_strokes_to_template(strokes, template, bbox)

    result = []
    for si, stroke in enumerate(strokes):
        ti = assigned[si]
        if ti < 0 or len(template[ti]) != 3:
            # Only adjust strokes with exactly one via region
            result.append(stroke)
            continue

        ts = template[ti]
        via_region = ts[1]

        # Check if midpoint already in the via region
        mid_pt = stroke[len(stroke) // 2]
        mid_region = _point_to_region(mid_pt[0], mid_pt[1], bbox)
        if mid_region == via_region:
            result.append(stroke)
            continue

        # Compute target position for the via region
        target_x, target_y = _region_center(via_region, bbox)

        # Measure how far the midpoint is from the target region center.
        # Scale blend so short distances get mild nudging.
        diag = ((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) ** 0.5
        mid_dist = ((mid_pt[0] - target_x) ** 2 + (mid_pt[1] - target_y) ** 2) ** 0.5
        # Blend: 0.3 at half-diagonal distance, capped at 0.5
        max_blend = min(0.5, 0.3 * mid_dist / max(diag * 0.5, 1.0))

        # Nudge middle portion of the stroke toward the target
        n = len(stroke)
        new_stroke = [list(p) for p in stroke]
        start_i = n // 4
        end_i = 3 * n // 4
        span = max(end_i - start_i - 1, 1)

        for i in range(start_i, end_i):
            t = (i - start_i) / span
            # Sine envelope: strongest at midpoint, zero at edges
            blend = max_blend * np.sin(t * np.pi)
            nx = new_stroke[i][0] + blend * (target_x - new_stroke[i][0])
            ny = new_stroke[i][1] + blend * (target_y - new_stroke[i][1])

            # Constrain to stay inside the glyph mask
            ix = int(round(min(max(nx, 0), w - 1)))
            iy = int(round(min(max(ny, 0), h - 1)))
            if mask[iy, ix]:
                new_stroke[i] = [nx, ny]
            else:
                # Binary search: find furthest valid point along the nudge vector
                orig_x, orig_y = stroke[i][0], stroke[i][1]
                lo, hi = 0.0, blend
                best_x, best_y = orig_x, orig_y
                for _ in range(8):
                    mid_b = (lo + hi) / 2
                    mx = orig_x + mid_b * (target_x - orig_x)
                    my = orig_y + mid_b * (target_y - orig_y)
                    mix = int(round(min(max(mx, 0), w - 1)))
                    miy = int(round(min(max(my, 0), h - 1)))
                    if mask[miy, mix]:
                        lo = mid_b
                        best_x, best_y = mx, my
                    else:
                        hi = mid_b
                new_stroke[i] = [best_x, best_y]

        result.append(new_stroke)

    return result


# Characters to show in grid (same set InkSight processes)
CHARS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_test_tables():
    """Ensure test tracking tables exist."""
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS test_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            font_id INTEGER NOT NULL,
            run_date TEXT NOT NULL,
            chars_tested INTEGER NOT NULL,
            chars_ok INTEGER NOT NULL,
            avg_score REAL,
            avg_coverage REAL,
            avg_overshoot REAL,
            avg_stroke_count REAL,
            avg_topology REAL,
            results_json TEXT,
            FOREIGN KEY (font_id) REFERENCES fonts(id)
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_test_runs_font ON test_runs(font_id)")
    db.commit()
    db.close()


def resolve_font_path(font_path):
    """Resolve a possibly-relative font path against BASE_DIR."""
    if os.path.isabs(font_path):
        return font_path
    return os.path.join(BASE_DIR, font_path)


def render_char_image(font_path, char, font_size=200, canvas_size=224):
    """Render a character centered on a square canvas, return as PNG bytes."""
    font_path = resolve_font_path(font_path)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return None

    img = Image.new('L', (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    bbox = font.getbbox(char)
    if not bbox:
        return None

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Scale down if character is too large
    if w > canvas_size * 0.9 or h > canvas_size * 0.9:
        scale = min(canvas_size * 0.9 / w, canvas_size * 0.9 / h)
        font_size = int(font_size * scale)
        font = ImageFont.truetype(font_path, font_size)
        bbox = font.getbbox(char)
        if not bbox:
            return None
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

    x = (canvas_size - w) // 2 - bbox[0]
    y = (canvas_size - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()


@app.route('/')
def font_list():
    """List fonts that have stroke data."""
    db = get_db()
    show_rejected = request.args.get('rejected') == '1'
    if show_rejected:
        fonts = db.execute("""
            SELECT f.id, f.name, f.source, f.file_path,
                   COALESCE(cs.char_count, 0) as char_count, 1 as rejected
            FROM fonts f
            JOIN font_removals fr ON fr.font_id = f.id AND fr.reason_id = 8
            LEFT JOIN (
                SELECT font_id, COUNT(*) as char_count
                FROM characters WHERE strokes_raw IS NOT NULL
                GROUP BY font_id
            ) cs ON cs.font_id = f.id
            ORDER BY f.name
        """).fetchall()
    else:
        fonts = db.execute("""
            SELECT f.id, f.name, f.source, f.file_path,
                   COALESCE(cs.char_count, 0) as char_count, 0 as rejected
            FROM fonts f
            LEFT JOIN font_removals rej ON rej.font_id = f.id AND rej.reason_id = 8
            LEFT JOIN font_removals dup ON dup.font_id = f.id AND dup.reason_id = 2
            LEFT JOIN (
                SELECT font_id, COUNT(*) as char_count
                FROM characters WHERE strokes_raw IS NOT NULL
                GROUP BY font_id
            ) cs ON cs.font_id = f.id
            WHERE rej.id IS NULL
              AND dup.id IS NULL
            ORDER BY f.name
        """).fetchall()
    db.close()
    return render_template('font_list.html', fonts=fonts, show_rejected=show_rejected)


@app.route('/font/<int:font_id>')
def char_grid(font_id):
    """Show character grid for a font."""
    db = get_db()
    font = db.execute("SELECT * FROM fonts WHERE id = ?", (font_id,)).fetchone()
    if not font:
        return "Font not found", 404

    chars = db.execute("""
        SELECT char, strokes_raw, point_count
        FROM characters
        WHERE font_id = ? AND strokes_raw IS NOT NULL
        ORDER BY char
    """, (font_id,)).fetchall()

    # If no characters with strokes, show all default chars for editing
    if not chars:
        chars = [{'char': c, 'strokes_raw': None, 'point_count': 0} for c in CHARS]

    db.close()
    return render_template('char_grid.html', font=font, chars=chars)


@app.route('/edit/<int:font_id>')
def edit_char(font_id):
    """Main editor page. Char passed as ?c= query param."""
    char = request.args.get('c')
    if not char:
        return "Missing character parameter ?c=", 400

    db = get_db()
    font = db.execute("SELECT * FROM fonts WHERE id = ?", (font_id,)).fetchone()
    if not font:
        return "Font not found", 404

    # Always use full character set for prev/next navigation
    char_list = CHARS
    db.close()

    return render_template('editor.html', font=font, char=char, char_list=char_list)


@app.route('/api/char/<int:font_id>')
def api_get_char(font_id):
    """Return stroke data and rendered font image as JSON."""
    char = request.args.get('c')
    if not char:
        return jsonify(error="Missing ?c= parameter"), 400

    db = get_db()
    font = db.execute("SELECT * FROM fonts WHERE id = ?", (font_id,)).fetchone()
    if not font:
        return jsonify(error="Font not found"), 404

    row = db.execute("""
        SELECT strokes_raw, markers FROM characters
        WHERE font_id = ? AND char = ?
    """, (font_id, char)).fetchone()
    db.close()

    strokes = json.loads(row['strokes_raw']) if row and row['strokes_raw'] else []
    markers = json.loads(row['markers']) if row and row['markers'] else []

    # Render font character image
    img_bytes = render_char_image(font['file_path'], char)
    img_b64 = None
    if img_bytes:
        img_b64 = base64.b64encode(img_bytes).decode('ascii')

    return jsonify(strokes=strokes, markers=markers, image=img_b64)


@app.route('/api/char/<int:font_id>', methods=['POST'])
def api_save_char(font_id):
    """Save edited strokes back to DB."""
    char = request.args.get('c')
    if not char:
        return jsonify(error="Missing ?c= parameter"), 400

    data = request.get_json()
    if not data or 'strokes' not in data:
        return jsonify(error="Missing strokes data"), 400

    strokes = data['strokes']
    markers = data.get('markers', [])
    total_points = sum(len(s) for s in strokes)
    markers_json = json.dumps(markers) if markers else None

    db = get_db()
    # Check if character row exists
    existing = db.execute(
        "SELECT id FROM characters WHERE font_id = ? AND char = ?",
        (font_id, char)
    ).fetchone()

    if existing:
        db.execute("""
            UPDATE characters
            SET strokes_raw = ?, point_count = ?, markers = ?
            WHERE font_id = ? AND char = ?
        """, (json.dumps(strokes), total_points, markers_json, font_id, char))
    else:
        db.execute("""
            INSERT INTO characters (font_id, char, strokes_raw, point_count, markers)
            VALUES (?, ?, ?, ?, ?)
        """, (font_id, char, json.dumps(strokes), total_points, markers_json))
    db.commit()
    db.close()

    return jsonify(ok=True)


@app.route('/api/render/<int:font_id>')
def api_render(font_id):
    """Serve rendered font character as PNG."""
    char = request.args.get('c')
    if not char:
        return "Missing ?c= parameter", 400

    db = get_db()
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?", (font_id,)).fetchone()
    db.close()
    if not font:
        return "Font not found", 404

    img_bytes = render_char_image(font['file_path'], char)
    if not img_bytes:
        return "Could not render", 500

    return send_file(io.BytesIO(img_bytes), mimetype='image/png')


@app.route('/api/thin-preview/<int:font_id>')
def api_thin_preview(font_id):
    """Render thinned glyph mask as PNG for previewing thinning effect.

    Query params:
        c: character (required)
        thin: number of thinning iterations (default 5)
    """
    from skimage.morphology import thin

    char = request.args.get('c')
    if not char:
        return "Missing ?c= parameter", 400

    thin_iterations = int(request.args.get('thin', 5))

    db = get_db()
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?", (font_id,)).fetchone()
    db.close()
    if not font:
        return "Font not found", 404

    mask = render_glyph_mask(font['file_path'], char)
    if mask is None:
        return "Could not render glyph", 500

    # Apply topology-preserving thinning
    if thin_iterations > 0:
        thinned = thin(mask, max_num_iter=thin_iterations)
    else:
        thinned = mask

    # Create image: white background, gray for original, black for thinned
    img = np.full((224, 224, 3), 255, dtype=np.uint8)
    img[mask] = [200, 200, 200]  # Original glyph in light gray
    img[thinned] = [0, 0, 0]     # Thinned result in black

    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


STROKE_COLORS = [
    (255, 80, 80), (80, 180, 255), (80, 220, 80), (255, 180, 40),
    (200, 100, 255), (255, 120, 200), (100, 220, 220), (180, 180, 80),
]


def check_case_mismatch(font_path, threshold=0.80):
    """Check if lowercase letters match their uppercase counterparts.

    Returns a list of lowercase letters that appear identical to uppercase.
    Normalizes glyphs to same size before comparing to catch small-caps fonts.

    Args:
        font_path: Path to font file
        threshold: Similarity threshold (0-1). Higher = more similar required to flag.

    Returns:
        List of lowercase letters that match uppercase (e.g., ['r', 'g'])
    """
    from PIL import Image, ImageDraw, ImageFont

    # Letters to check
    letters_to_check = 'abcdefghijklmnopqrstuvwxyz'
    mismatched = []

    try:
        font_size = 100
        pil_font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return []

    # Normalized comparison size
    norm_size = 64

    for lower in letters_to_check:
        upper = lower.upper()

        try:
            # Get bounding boxes
            l_bbox = pil_font.getbbox(lower)
            u_bbox = pil_font.getbbox(upper)

            if not l_bbox or not u_bbox:
                continue

            l_w = l_bbox[2] - l_bbox[0]
            l_h = l_bbox[3] - l_bbox[1]
            u_w = u_bbox[2] - u_bbox[0]
            u_h = u_bbox[3] - u_bbox[1]

            if l_w < 5 or l_h < 5 or u_w < 5 or u_h < 5:
                continue

            # Render lowercase at its natural size with padding
            l_img = Image.new('L', (l_w + 10, l_h + 10), 255)
            l_draw = ImageDraw.Draw(l_img)
            l_draw.text((5 - l_bbox[0], 5 - l_bbox[1]), lower, fill=0, font=pil_font)

            # Render uppercase at its natural size with padding
            u_img = Image.new('L', (u_w + 10, u_h + 10), 255)
            u_draw = ImageDraw.Draw(u_img)
            u_draw.text((5 - u_bbox[0], 5 - u_bbox[1]), upper, fill=0, font=pil_font)

            # Scale both to same normalized size for comparison
            l_scaled = l_img.resize((norm_size, norm_size), Image.Resampling.BILINEAR)
            u_scaled = u_img.resize((norm_size, norm_size), Image.Resampling.BILINEAR)

            l_arr = np.array(l_scaled) < 128
            u_arr = np.array(u_scaled) < 128

            # Compare using intersection over union (IoU)
            intersection = np.sum(l_arr & u_arr)
            union = np.sum(l_arr | u_arr)

            if union > 0:
                iou = intersection / union
                if iou >= threshold:
                    mismatched.append(lower)

        except Exception:
            continue

    return mismatched


def _render_text_for_analysis(pil_font, text):
    """Render text using PIL font and return binary array plus dimensions.

    Args:
        pil_font: PIL ImageFont object
        text: Text string to render

    Returns:
        Tuple of (binary_array, width, height) or (None, 0, 0) on failure
    """
    bbox = pil_font.getbbox(text)
    if not bbox:
        return None, 0, 0

    width = bbox[2] - bbox[0] + 20
    height = bbox[3] - bbox[1] + 20

    img = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(img)
    draw.text((10 - bbox[0], 10 - bbox[1]), text, fill=0, font=pil_font)

    arr = np.array(img) < 128
    return arr, width, height


def _analyze_shape_metrics(arr, width):
    """Analyze connected components and compute shape metrics.

    Args:
        arr: Binary numpy array of rendered text
        width: Image width for percentage calculation

    Returns:
        Tuple of (num_shapes, max_width_pct, labeled_array)
    """
    from scipy import ndimage

    labeled, num_shapes = ndimage.label(arr)

    max_width_pct = 0
    if num_shapes > 0:
        for i in range(1, num_shapes + 1):
            component = (labeled == i)
            cols = np.any(component, axis=0)
            comp_width = np.sum(cols)
            width_pct = comp_width / width
            if width_pct > max_width_pct:
                max_width_pct = width_pct

    return num_shapes, max_width_pct, labeled


def _check_char_holes(pil_font, char):
    """Check if a character has holes (cursive detection).

    Args:
        pil_font: PIL ImageFont object
        char: Character to check

    Returns:
        True if character has holes, False otherwise
    """
    from scipy import ndimage

    try:
        bbox = pil_font.getbbox(char)
        if not bbox:
            return False

        width = bbox[2] - bbox[0] + 10
        height = bbox[3] - bbox[1] + 10

        img = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(img)
        draw.text((5 - bbox[0], 5 - bbox[1]), char, fill=0, font=pil_font)

        arr = np.array(img) < 128
        filled = ndimage.binary_fill_holes(arr)
        holes = filled & ~arr
        return bool(np.any(holes))
    except Exception:
        return False


def _check_char_shape_count(pil_font, char, expected):
    """Check if a character has expected number of connected components.

    Args:
        pil_font: PIL ImageFont object
        char: Character to check
        expected: Expected shape count

    Returns:
        Tuple of (is_ok, actual_count)
    """
    from scipy import ndimage

    try:
        bbox = pil_font.getbbox(char)
        if not bbox:
            return False, 0

        width = bbox[2] - bbox[0] + 10
        height = bbox[3] - bbox[1] + 10

        img = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(img)
        draw.text((5 - bbox[0], 5 - bbox[1]), char, fill=0, font=pil_font)

        arr = np.array(img) < 128
        _, num_shapes = ndimage.label(arr)
        return num_shapes == expected, num_shapes
    except Exception:
        return False, 0


@app.route('/api/check-connected/<int:font_id>')
def api_check_connected(font_id):
    """Check if font renders text with appropriate shape count.

    Returns shape count for "Hello World" - should be 10-15 for normal fonts.
    < 10 means connected/cursive, > 15 means overly decorative.
    """
    db = get_db()
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?", (font_id,)).fetchone()
    db.close()
    if not font:
        return jsonify(error="Font not found"), 404

    font_path = resolve_font_path(font['file_path'])

    if not os.path.exists(font_path):
        return jsonify(error="Font file missing"), 404

    try:
        font_size = 60
        try:
            pil_font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            return jsonify(error=f"Can't load font: {e}"), 500

        # Render and analyze "Hello World"
        arr, width, height = _render_text_for_analysis(pil_font, "Hello World")
        if arr is None:
            return jsonify(error="Could not render"), 500

        num_shapes, max_width_pct, _ = _analyze_shape_metrics(arr, width)

        # Check for holes in letter "l"
        l_has_hole = _check_char_holes(pil_font, 'l')

        # Check for "!" glyph - should have exactly 2 components
        exclaim_ok, exclaim_shapes = _check_char_shape_count(pil_font, '!', 2)

        # Check for lowercase/uppercase case mismatches
        case_mismatches = check_case_mismatch(font_path)

        # Bad if: wrong shape count OR largest component too wide OR "l" has hole OR "!" missing/invalid
        bad_shapes = num_shapes < 10 or num_shapes > 15
        bad_width = max_width_pct > 0.225
        bad_hole = l_has_hole
        bad_exclaim = not exclaim_ok
        bad = bad_shapes or bad_width or bad_hole or bad_exclaim

        return jsonify(
            shapes=int(num_shapes),
            min_required=10,
            max_allowed=15,
            max_width_pct=round(float(max_width_pct) * 100, 1),
            l_has_hole=l_has_hole,
            exclaim_ok=exclaim_ok,
            exclaim_shapes=int(exclaim_shapes),
            case_mismatches=case_mismatches,
            case_mismatch_count=len(case_mismatches),
            bad=bool(bad),
            reason='exclaim' if bad_exclaim and not bad_hole and not bad_width and not bad_shapes else ('hole' if bad_hole and not bad_shapes and not bad_width else ('width' if bad_width and not bad_shapes else ('shapes' if bad_shapes else None)))
        )

    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/api/reject-connected', methods=['POST'])
def api_reject_connected():
    """Check all fonts and reject those with bad shape counts (< 10 or > 15 shapes in 'Hello World')."""
    db = get_db()
    # Get non-rejected fonts (no entry in font_removals with reason_id 8)
    fonts = db.execute("""
        SELECT f.id, f.file_path
        FROM fonts f
        LEFT JOIN font_removals fr ON fr.font_id = f.id AND fr.reason_id = 8
        WHERE fr.id IS NULL
    """).fetchall()

    rejected = []
    checked = 0

    for font in fonts:
        font_path = resolve_font_path(font['file_path'])

        try:
            font_size = 60
            pil_font = ImageFont.truetype(font_path, font_size)

            # Render and analyze "Hello World"
            arr, width, height = _render_text_for_analysis(pil_font, "Hello World")
            if arr is None:
                continue

            num_shapes, max_width_pct, _ = _analyze_shape_metrics(arr, width)

            # Check for holes in letter "l"
            l_has_hole = _check_char_holes(pil_font, 'l')

            # Check for "!" glyph - should have exactly 2 components
            exclaim_ok, exclaim_shapes = _check_char_shape_count(pil_font, '!', 2)

            # Check for case mismatches (lowercase = uppercase)
            case_mismatches = check_case_mismatch(font_path)

            checked += 1

            bad_shapes = num_shapes < 10 or num_shapes > 15
            bad_width = max_width_pct > 0.225
            bad_hole = l_has_hole
            bad_exclaim = not exclaim_ok
            bad_case = len(case_mismatches) > 0

            if bad_shapes or bad_width or bad_hole or bad_exclaim or bad_case:
                # Insert rejection into font_removals table
                if num_shapes < 10:
                    reason = f'Connected letters: {num_shapes} shapes < 10 required'
                elif num_shapes > 15:
                    reason = f'Too decorative: {num_shapes} shapes > 15 max'
                elif bad_width:
                    reason = f'Cursive: largest component spans {max_width_pct*100:.1f}% width (>22.5%)'
                elif bad_hole:
                    reason = f'Cursive: letter "l" has hole/loop'
                elif bad_exclaim:
                    reason = f'Missing/invalid "!" glyph ({exclaim_shapes} shapes, need 2)'
                elif bad_case:
                    reason = f'Case mismatch: lowercase matches uppercase for: {", ".join(case_mismatches)}'
                db.execute(
                    "INSERT INTO font_removals (font_id, reason_id, details) VALUES (?, 8, ?)",
                    (font['id'], reason)
                )
                rejected.append({'id': font['id'], 'shapes': num_shapes, 'width_pct': round(max_width_pct*100, 1), 'l_hole': l_has_hole, 'exclaim_ok': exclaim_ok, 'case_mismatches': case_mismatches})

        except Exception:
            continue

    db.commit()
    db.close()

    return jsonify(ok=True, checked=checked, rejected=len(rejected), fonts=rejected)


@app.route('/api/font-sample/<int:font_id>')
def api_font_sample(font_id):
    """Render sample text in a font as PNG for font list preview."""
    text = request.args.get('text', 'Hello World!')
    height = int(request.args.get('h', 40))

    db = get_db()
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?", (font_id,)).fetchone()
    db.close()
    if not font:
        return "Font not found", 404

    font_path = resolve_font_path(font['file_path'])

    try:
        from PIL import Image, ImageDraw, ImageFont

        # Find font size that fits the height
        font_size = int(height * 0.85)
        try:
            pil_font = ImageFont.truetype(font_path, font_size)
        except Exception:
            return "Could not load font", 500

        # Get text dimensions
        bbox = pil_font.getbbox(text)
        if not bbox:
            return "Could not render text", 500

        text_width = bbox[2] - bbox[0] + 10
        text_height = height

        # Create image with transparent background
        img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw text in white
        x = 5 - bbox[0]
        y = (height - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=pil_font)

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return f"Error: {e}", 500


@app.route('/api/preview/<int:font_id>')
def api_preview(font_id):
    """Render a character with strokes overlaid as a small PNG thumbnail."""
    char = request.args.get('c')
    if not char:
        return "Missing ?c= parameter", 400

    db = get_db()
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?", (font_id,)).fetchone()
    row = db.execute(
        "SELECT strokes_raw FROM characters WHERE font_id = ? AND char = ?",
        (font_id, char)
    ).fetchone()
    db.close()
    if not font:
        return "Font not found", 404

    # Render glyph as grayscale background
    img_bytes = render_char_image(font['file_path'], char)
    if not img_bytes:
        return "Could not render", 500

    gray = Image.open(io.BytesIO(img_bytes)).convert('L')
    arr = np.array(gray)
    # White background, glyph pixels as semi-transparent gray
    rgba = np.full((*arr.shape, 4), 255, dtype=np.uint8)
    glyph_mask = arr < 200
    rgba[glyph_mask, 0] = arr[glyph_mask]
    rgba[glyph_mask, 1] = arr[glyph_mask]
    rgba[glyph_mask, 2] = arr[glyph_mask]
    rgba[glyph_mask, 3] = 60
    bg = Image.fromarray(rgba, 'RGBA')

    draw = ImageDraw.Draw(bg)

    # Draw 9-region grid (numpad/nonants) if requested
    show_grid = request.args.get('grid', '').lower() in ('1', 'true', 'yes')
    if show_grid:
        # Find glyph bounding box
        rows_idx, cols_idx = np.where(glyph_mask)
        if len(rows_idx) > 0:
            x_min, x_max = cols_idx.min(), cols_idx.max()
            y_min, y_max = rows_idx.min(), rows_idx.max()
            w = x_max - x_min
            h = y_max - y_min

            # Draw vertical lines at 1/3 and 2/3
            grid_color = (100, 100, 255, 180)
            x1 = x_min + w / 3
            x2 = x_min + 2 * w / 3
            draw.line([(x1, y_min), (x1, y_max)], fill=grid_color, width=1)
            draw.line([(x2, y_min), (x2, y_max)], fill=grid_color, width=1)

            # Draw horizontal lines at 1/3 and 2/3
            y1 = y_min + h / 3
            y2 = y_min + 2 * h / 3
            draw.line([(x_min, y1), (x_max, y1)], fill=grid_color, width=1)
            draw.line([(x_min, y2), (x_max, y2)], fill=grid_color, width=1)

            # Draw bounding box
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=grid_color, width=1)

            # Label regions 1-9 (numpad layout: 7 8 9 / 4 5 6 / 1 2 3)
            label_color = (80, 80, 200, 220)
            positions = {
                7: (x_min + w/6, y_min + h/6),
                8: (x_min + w/2, y_min + h/6),
                9: (x_min + 5*w/6, y_min + h/6),
                4: (x_min + w/6, y_min + h/2),
                5: (x_min + w/2, y_min + h/2),
                6: (x_min + 5*w/6, y_min + h/2),
                1: (x_min + w/6, y_min + 5*h/6),
                2: (x_min + w/2, y_min + 5*h/6),
                3: (x_min + 5*w/6, y_min + 5*h/6),
            }
            for num, (px, py) in positions.items():
                draw.text((px - 4, py - 6), str(num), fill=label_color)

    # Draw strokes
    if row and row['strokes_raw']:
        strokes = json.loads(row['strokes_raw'])
        for si, stroke in enumerate(strokes):
            color = STROKE_COLORS[si % len(STROKE_COLORS)]
            if len(stroke) >= 2:
                pts = [(p[0], p[1]) for p in stroke]
                draw.line(pts, fill=color + (255,), width=2)

    buf = io.BytesIO()
    bg.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/api/process/<int:font_id>', methods=['POST'])
def api_process(font_id):
    """Run stroke post-processing (extend_to_connect) on provided strokes."""
    char = request.args.get('c')
    if not char:
        return jsonify(error="Missing ?c= parameter"), 400

    data = request.get_json()
    if not data or 'strokes' not in data:
        return jsonify(error="Missing strokes data"), 400

    max_ext = data.get('max_extension', 8.0)
    smooth = data.get('smooth', False)
    smooth_sigma = data.get('smooth_sigma', 1.5)

    # Extract locked flags (3rd element == 1) before converting to xy arrays
    locked_flags = []
    for s in data['strokes']:
        flags = []
        for p in s:
            flags.append(len(p) >= 3 and p[2] == 1)
        locked_flags.append(flags)

    # Convert to numpy arrays (xy only)
    np_strokes = [np.array([[p[0], p[1]] for p in s], dtype=float) for s in data['strokes']]

    # Optionally smooth (scale sigma down for short strokes to avoid over-smoothing)
    # Locked (vertex) points are preserved: smooth each segment between them
    # independently so the Gaussian filter doesn't pull neighbors away.
    if smooth:
        smoothed = []
        for si, s in enumerate(np_strokes):
            if len(s) < 3:
                smoothed.append(s)
                continue
            flags = locked_flags[si]
            # Find indices of locked points
            locked_idxs = [i for i, f in enumerate(flags) if f]
            if not locked_idxs:
                # No locked points: smooth the whole stroke
                effective_sigma = smooth_sigma * min(1.0, (len(s) - 2) / 30.0)
                smoothed.append(InkSightVectorizer.smooth_gaussian(s, sigma=effective_sigma))
            else:
                # Smooth segments between locked points independently
                result = s.copy()
                # Build segment boundaries: [0, lock1, lock2, ..., len-1]
                bounds = [0] + locked_idxs + [len(s) - 1]
                # Deduplicate and sort
                bounds = sorted(set(bounds))
                for bi in range(len(bounds) - 1):
                    start, end = bounds[bi], bounds[bi + 1]
                    seg = s[start:end + 1]
                    if len(seg) >= 3:
                        effective_sigma = smooth_sigma * min(1.0, (len(seg) - 2) / 30.0)
                        sm = InkSightVectorizer.smooth_gaussian(seg, sigma=effective_sigma)
                        # Keep locked endpoints unchanged
                        if flags[start]:
                            sm[0] = s[start]
                        if flags[end]:
                            sm[-1] = s[end]
                        result[start:end + 1] = sm
                smoothed.append(result)
        np_strokes = smoothed

    # Extend to connect (optional)
    if data.get('connect', True):
        vectorizer = InkSightVectorizer()
        np_strokes = vectorizer.extend_to_connect(np_strokes, max_extension=max_ext)

    # Convert back to lists, restoring locked flags
    result = []
    for si, s in enumerate(np_strokes):
        stroke_out = []
        pts = s.tolist()
        flags = locked_flags[si] if si < len(locked_flags) else []
        for pi, pt in enumerate(pts):
            if pi < len(flags) and flags[pi]:
                stroke_out.append([pt[0], pt[1], 1])
            else:
                stroke_out.append([pt[0], pt[1]])
        result.append(stroke_out)
    return jsonify(strokes=result)


def render_glyph_mask(font_path, char, canvas_size=224):
    """Render a character as a binary mask (True = inside glyph)."""
    font_path = resolve_font_path(font_path)
    font_size = 200
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return None

    img = Image.new('L', (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    bbox = font.getbbox(char)
    if not bbox:
        return None

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    if w > canvas_size * 0.9 or h > canvas_size * 0.9:
        scale = min(canvas_size * 0.9 / w, canvas_size * 0.9 / h)
        font_size = int(font_size * scale)
        font = ImageFont.truetype(font_path, font_size)
        bbox = font.getbbox(char)
        if not bbox:
            return None
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

    x = (canvas_size - w) // 2 - bbox[0]
    y = (canvas_size - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)

    # Binary mask: True where glyph is (dark pixels)
    return np.array(img) < 128


def _flatten_bezier_quad(p0, p1, p2, steps=15):
    """Flatten a quadratic Bezier curve to a point sequence."""
    pts = []
    for i in range(1, steps + 1):
        t = i / steps
        x = (1-t)**2*p0[0] + 2*(1-t)*t*p1[0] + t**2*p2[0]
        y = (1-t)**2*p0[1] + 2*(1-t)*t*p1[1] + t**2*p2[1]
        pts.append((x, y))
    return pts


def _flatten_bezier_cubic(p0, p1, p2, p3, steps=20):
    """Flatten a cubic Bezier curve to a point sequence."""
    pts = []
    for i in range(1, steps + 1):
        t = i / steps
        x = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
        y = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
        pts.append((x, y))
    return pts


def _extract_contours(font_path, char):
    """Extract glyph contours from a font using fontTools RecordingPen."""
    tt = TTFont(font_path)
    cmap = tt.getBestCmap()
    if not cmap:
        return None, tt
    glyph_name = cmap.get(ord(char))
    if not glyph_name:
        return None, tt
    glyphset = tt.getGlyphSet()
    pen = RecordingPen()
    glyphset[glyph_name].draw(pen)

    contours = []
    current = []
    current_pos = (0, 0)
    for op, args in pen.value:
        if op == 'moveTo':
            if current:
                contours.append(current)
            current = [args[0]]
            current_pos = args[0]
        elif op == 'lineTo':
            current.append(args[0])
            current_pos = args[0]
        elif op == 'curveTo':
            pts = _flatten_bezier_cubic(current_pos, args[0], args[1], args[2])
            current.extend(pts)
            current_pos = args[2]
        elif op == 'qCurveTo':
            if len(args) == 2:
                pts = _flatten_bezier_quad(current_pos, args[0], args[1])
                current.extend(pts)
                current_pos = args[1]
            else:
                for i in range(len(args) - 1):
                    if i < len(args) - 2:
                        mid = ((args[i][0]+args[i+1][0])/2, (args[i][1]+args[i+1][1])/2)
                        pts = _flatten_bezier_quad(current_pos, args[i], mid)
                        current.extend(pts)
                        current_pos = mid
                    else:
                        pts = _flatten_bezier_quad(current_pos, args[i], args[i+1])
                        current.extend(pts)
                        current_pos = args[i+1]
        elif op in ('closePath', 'endPath'):
            if current:
                contours.append(current)
                current = []
    if current:
        contours.append(current)
    return contours, tt


def _font_to_pixel_transform(tt, font_path, char, canvas_size=224):
    """Build a transform function from font units to pixel coordinates.

    Replicates the centering logic in render_glyph_mask / render_char_image.
    """
    pil_font = ImageFont.truetype(font_path, 200)
    bbox = pil_font.getbbox(char)
    font_size = 200
    if bbox:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w > canvas_size * 0.9 or h > canvas_size * 0.9:
            scale = min(canvas_size * 0.9 / w, canvas_size * 0.9 / h)
            font_size = int(font_size * scale)
            pil_font = ImageFont.truetype(font_path, font_size)
            bbox = pil_font.getbbox(char)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        offset_x = (canvas_size - w) // 2 - bbox[0]
        offset_y = (canvas_size - h) // 2 - bbox[1]
    else:
        offset_x, offset_y = 0, 0
    upem = tt['head'].unitsPerEm
    px_per_unit = font_size / upem
    ascender = tt['hhea'].ascent

    def transform(fx, fy):
        return (fx * px_per_unit + offset_x, (ascender - fy) * px_per_unit + offset_y)

    return transform


def contour_to_strokes(font_path, char, canvas_size=224):
    """Extract strokes by splitting the outer font contour at top/bottom extremal points.

    Returns two strokes as [[[x,y], ...], [[x,y], ...]] tracing the left and right
    sides of the letter outline.
    """
    font_path = resolve_font_path(font_path)
    contours, tt = _extract_contours(font_path, char)
    if not contours:
        return None

    transform = _font_to_pixel_transform(tt, font_path, char, canvas_size)

    # Transform all contours to pixel space
    pixel_contours = []
    for c in contours:
        pixel_contours.append([transform(p[0], p[1]) for p in c])

    if not pixel_contours:
        return None

    # Find largest contour (outer outline)
    largest_idx = max(range(len(pixel_contours)), key=lambda i: len(pixel_contours[i]))
    outer = pixel_contours[largest_idx]

    if len(outer) < 4:
        return None

    # Split at topmost (min y) and bottommost (max y) points
    pts = np.array(outer)
    top_idx = int(np.argmin(pts[:, 1]))
    bot_idx = int(np.argmax(pts[:, 1]))

    if top_idx == bot_idx:
        return [[[float(p[0]), float(p[1])] for p in outer]]

    # Split into two halves
    if top_idx < bot_idx:
        half1 = outer[top_idx:bot_idx + 1]
        half2 = outer[bot_idx:] + outer[:top_idx + 1]
    else:
        half1 = outer[top_idx:] + outer[:bot_idx + 1]
        half2 = outer[bot_idx:top_idx + 1]

    return [
        [[float(p[0]), float(p[1])] for p in half1],
        [[float(p[0]), float(p[1])] for p in half2],
    ]


def contour_detect_markers(font_path, char, canvas_size=224):
    """Detect termination markers from contour split points (top/bottom of letter)."""
    font_path = resolve_font_path(font_path)
    contours, tt = _extract_contours(font_path, char)
    if not contours:
        return []

    transform = _font_to_pixel_transform(tt, font_path, char, canvas_size)

    pixel_contours = []
    for c in contours:
        pixel_contours.append([transform(p[0], p[1]) for p in c])

    if not pixel_contours:
        return []

    largest_idx = max(range(len(pixel_contours)), key=lambda i: len(pixel_contours[i]))
    outer = pixel_contours[largest_idx]

    if len(outer) < 4:
        return []

    pts = np.array(outer)
    top_idx = int(np.argmin(pts[:, 1]))
    bot_idx = int(np.argmax(pts[:, 1]))

    markers = []
    top_pt = outer[top_idx]
    bot_pt = outer[bot_idx]
    markers.append({'x': round(float(top_pt[0]), 1), 'y': round(float(top_pt[1]), 1), 'type': 'termination'})
    markers.append({'x': round(float(bot_pt[0]), 1), 'y': round(float(bot_pt[1]), 1), 'type': 'termination'})
    return markers


@app.route('/api/snap/<int:font_id>', methods=['POST'])
def api_snap(font_id):
    """Snap stroke points to nearest position inside the font glyph outline."""
    char = request.args.get('c')
    if not char:
        return jsonify(error="Missing ?c= parameter"), 400

    data = request.get_json()
    if not data or 'strokes' not in data:
        return jsonify(error="Missing strokes data"), 400

    db = get_db()
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?", (font_id,)).fetchone()
    db.close()
    if not font:
        return jsonify(error="Font not found"), 404

    mask = render_glyph_mask(font['file_path'], char)
    if mask is None:
        return jsonify(error="Could not render glyph"), 500

    # Distance transform on the OUTSIDE (inverted mask).
    # For pixels outside the glyph, gives distance to nearest glyph pixel
    # and indices of that nearest pixel.
    outside = ~mask
    dist_out, indices = distance_transform_edt(outside, return_indices=True)

    # Distance transform on the INSIDE: how deep each pixel is from the edge.
    dist_in = distance_transform_edt(mask)

    # Margin: snapped points should be at least this deep inside the glyph
    # so that center-borders can ray-cast properly from them.
    MARGIN = 2.0

    h, w = mask.shape
    result = []
    for stroke in data['strokes']:
        snapped = []
        for p in stroke:
            x, y = p[0], p[1]
            locked = len(p) >= 3 and p[2] == 1
            # Clamp to canvas bounds
            ix = int(round(min(max(x, 0), w - 1)))
            iy = int(round(min(max(y, 0), h - 1)))

            if mask[iy, ix] and dist_in[iy, ix] >= MARGIN:
                # Already well inside glyph
                snapped.append(p[:])
            else:
                # Find nearest glyph boundary pixel first
                if mask[iy, ix]:
                    # Inside but too close to edge - use current position
                    bx, by = float(ix), float(iy)
                else:
                    # Outside - snap to nearest glyph pixel
                    by = float(indices[0, iy, ix])
                    bx = float(indices[1, iy, ix])

                # Nudge inward: walk from boundary pixel toward interior
                # using the gradient of the interior distance field
                bix, biy = int(round(bx)), int(round(by))
                bix = min(max(bix, 0), w - 1)
                biy = min(max(biy, 0), h - 1)

                if dist_in[biy, bix] >= MARGIN:
                    # Boundary pixel is already deep enough (shouldn't happen often)
                    sp = [bx, by, 1] if locked else [bx, by]
                    snapped.append(sp)
                else:
                    # Search in a small neighborhood for the nearest pixel
                    # that's at least MARGIN deep inside the glyph
                    best_d = float('inf')
                    best_x, best_y = bx, by
                    search_r = int(MARGIN + 3)
                    for sy in range(max(0, biy - search_r), min(h, biy + search_r + 1)):
                        for sx in range(max(0, bix - search_r), min(w, bix + search_r + 1)):
                            if dist_in[sy, sx] >= MARGIN:
                                dd = (sx - bx) ** 2 + (sy - by) ** 2
                                if dd < best_d:
                                    best_d = dd
                                    best_x, best_y = float(sx), float(sy)
                    sp = [best_x, best_y, 1] if locked else [best_x, best_y]
                    snapped.append(sp)
        result.append(snapped)

    return jsonify(strokes=result)


@app.route('/api/center/<int:font_id>', methods=['POST'])
def api_center(font_id):
    """Center stroke points on the font glyph."""
    char = request.args.get('c')
    if not char:
        return jsonify(error="Missing ?c= parameter"), 400

    data = request.get_json()
    if not data or 'strokes' not in data:
        return jsonify(error="Missing strokes data"), 400

    db = get_db()
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?", (font_id,)).fetchone()
    db.close()
    if not font:
        return jsonify(error="Font not found"), 404

    mask = render_glyph_mask(font['file_path'], char)
    if mask is None:
        return jsonify(error="Could not render glyph"), 500

    # Find glyph bounding box center
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return jsonify(error="Empty glyph"), 500
    glyph_cx = (cols.min() + cols.max()) / 2.0
    glyph_cy = (rows.min() + rows.max()) / 2.0

    # Find stroke points bounding box center
    all_x, all_y = [], []
    for stroke in data['strokes']:
        for p in stroke:
            all_x.append(p[0])
            all_y.append(p[1])

    if not all_x:
        return jsonify(strokes=data['strokes'])

    stroke_cx = (min(all_x) + max(all_x)) / 2.0
    stroke_cy = (min(all_y) + max(all_y)) / 2.0

    # Translate all points
    dx = glyph_cx - stroke_cx
    dy = glyph_cy - stroke_cy

    result = []
    for stroke in data['strokes']:
        result.append([[p[0] + dx, p[1] + dy] for p in stroke])

    return jsonify(strokes=result)


@app.route('/api/reject/<int:font_id>', methods=['POST'])
def api_reject_font(font_id):
    """Mark a font as rejected (manual removal reason)."""
    db = get_db()
    font = db.execute("SELECT id FROM fonts WHERE id = ?", (font_id,)).fetchone()
    if not font:
        db.close()
        return jsonify(error="Font not found"), 404

    # Check if already rejected
    existing = db.execute(
        "SELECT id FROM font_removals WHERE font_id = ? AND reason_id = 8",
        (font_id,)
    ).fetchone()

    if existing:
        db.close()
        return jsonify(ok=True, status='already_rejected')

    db.execute(
        "INSERT INTO font_removals (font_id, reason_id, details) VALUES (?, 8, 'Rejected in stroke editor')",
        (font_id,)
    )
    db.commit()
    db.close()
    return jsonify(ok=True, status='rejected')


@app.route('/api/unreject/<int:font_id>', methods=['POST'])
def api_unreject_font(font_id):
    """Remove manual rejection from a font."""
    db = get_db()
    db.execute(
        "DELETE FROM font_removals WHERE font_id = ? AND reason_id = 8",
        (font_id,)
    )
    db.commit()
    db.close()
    return jsonify(ok=True, status='unrejected')


@app.route('/api/unreject-all', methods=['POST'])
def api_unreject_all():
    """Remove all manual rejections (reason_id = 8)."""
    db = get_db()
    result = db.execute("DELETE FROM font_removals WHERE reason_id = 8")
    restored = result.rowcount
    db.commit()
    db.close()
    return jsonify(ok=True, restored=restored)


@app.route('/api/test-run/<int:font_id>', methods=['POST'])
def api_run_tests(font_id):
    """Run stroke quality tests for all characters in a font."""
    from test_minimal_strokes import test_letter

    ensure_test_tables()

    db = get_db()
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?", (font_id,)).fetchone()
    if not font:
        db.close()
        return jsonify(error="Font not found"), 404

    font_path = resolve_font_path(font['file_path'])

    # Test all characters
    chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
    results = []

    for char in chars:
        result = test_letter(font_path, char)
        # Also save the stroke data for historical comparison
        try:
            strokes = minimal_strokes_from_skeleton(font_path, char, 224)
            if strokes:
                result['strokes'] = strokes
        except Exception:
            pass
        results.append(result)

    # Calculate averages
    ok_results = [r for r in results if r['status'] == 'ok']
    n = len(ok_results) if ok_results else 1

    avg_score = sum(r['score'] for r in ok_results) / n if ok_results else 0
    avg_coverage = sum(r['coverage'] for r in ok_results) / n if ok_results else 0
    avg_overshoot = sum(r['overshoot'] for r in ok_results) / n if ok_results else 0
    avg_stroke_count = sum(r['stroke_count_score'] for r in ok_results) / n if ok_results else 0
    avg_topology = sum(r['topology_score'] for r in ok_results) / n if ok_results else 0

    # Store in database
    from datetime import datetime
    run_date = datetime.now().isoformat()

    db.execute("""
        INSERT INTO test_runs (font_id, run_date, chars_tested, chars_ok,
            avg_score, avg_coverage, avg_overshoot, avg_stroke_count, avg_topology,
            results_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (font_id, run_date, len(results), len(ok_results),
          avg_score, avg_coverage, avg_overshoot, avg_stroke_count, avg_topology,
          json.dumps(results)))
    db.commit()
    db.close()

    # Find worst character (lowest score)
    worst_char = None
    worst_score = 1.0
    for r in ok_results:
        if r['score'] < worst_score:
            worst_score = r['score']
            worst_char = r['char']

    # Include detailed results if requested
    include_details = request.args.get('details', 'false').lower() == 'true'

    response = {
        'ok': True,
        'chars_tested': len(results),
        'chars_ok': len(ok_results),
        'avg_score': round(avg_score, 3),
        'avg_coverage': round(avg_coverage, 3),
        'avg_overshoot': round(avg_overshoot, 3),
        'avg_stroke_count': round(avg_stroke_count, 3),
        'avg_topology': round(avg_topology, 3),
        'worst_char': worst_char,
        'worst_score': round(worst_score, 3) if worst_char else None
    }

    if include_details:
        response['results'] = results

    return jsonify(response)


@app.route('/api/test-history/<int:font_id>')
def api_test_history(font_id):
    """Get test run history for a font."""
    ensure_test_tables()

    limit = request.args.get('limit', 10, type=int)

    db = get_db()
    runs = db.execute("""
        SELECT id, run_date, chars_tested, chars_ok,
               avg_score, avg_coverage, avg_overshoot, avg_stroke_count, avg_topology
        FROM test_runs
        WHERE font_id = ?
        ORDER BY run_date DESC
        LIMIT ?
    """, (font_id, limit)).fetchall()
    db.close()

    return jsonify(runs=[dict(r) for r in runs])


@app.route('/api/test-run-detail/<int:run_id>')
def api_test_run_detail(run_id):
    """Get detailed results for a specific test run."""
    ensure_test_tables()

    db = get_db()
    run = db.execute("""
        SELECT * FROM test_runs WHERE id = ?
    """, (run_id,)).fetchone()
    db.close()

    if not run:
        return jsonify(error="Run not found"), 404

    result = dict(run)
    if result.get('results_json'):
        result['results'] = json.loads(result['results_json'])
        del result['results_json']

    return jsonify(result)


@app.route('/api/preview-from-run/<int:run_id>')
def api_preview_from_run(run_id):
    """Render a stroke preview from stored test run data.

    Query params:
        c: character to render
        font_id: font ID (to get the glyph mask)
    """
    char = request.args.get('c', 'A')
    font_id = request.args.get('font_id', type=int)

    ensure_test_tables()
    db = get_db()
    run = db.execute("SELECT * FROM test_runs WHERE id = ?", (run_id,)).fetchone()

    if not run:
        db.close()
        return "Run not found", 404

    # Parse results to find the character's strokes
    results = json.loads(run['results_json']) if run['results_json'] else []
    char_result = None
    for r in results:
        if r.get('char') == char:
            char_result = r
            break

    if not char_result or 'strokes' not in char_result:
        db.close()
        # Return a placeholder image indicating no data
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (224, 224), (26, 26, 46))
        draw = ImageDraw.Draw(img)
        draw.text((60, 100), "No stroke data", fill=(100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    strokes = char_result['strokes']

    # Get font path to render glyph background
    if not font_id:
        font_id = run['font_id']
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?", (font_id,)).fetchone()
    db.close()

    if not font:
        return "Font not found", 404

    font_path = resolve_font_path(font['file_path'])

    # Render the preview with stored strokes
    from PIL import Image, ImageDraw

    # Get glyph mask for background
    mask = render_glyph_mask(font_path, char, 224)

    # Create image
    img = Image.new('RGB', (224, 224), (26, 26, 46))
    draw = ImageDraw.Draw(img)

    # Draw glyph silhouette
    if mask is not None:
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x] > 0:
                    img.putpixel((x, y), (60, 60, 80))

    # Draw strokes
    colors = [(126, 184, 247), (247, 184, 126), (184, 247, 126), (247, 126, 184)]
    for i, stroke in enumerate(strokes):
        if len(stroke) >= 2:
            color = colors[i % len(colors)]
            points = [(int(p[0]), int(p[1])) for p in stroke]
            draw.line(points, fill=color, width=3)
            # Draw endpoints
            draw.ellipse([points[0][0]-4, points[0][1]-4,
                         points[0][0]+4, points[0][1]+4], fill=(100, 255, 100))
            draw.ellipse([points[-1][0]-4, points[-1][1]-4,
                         points[-1][0]+4, points[-1][1]+4], fill=(255, 100, 100))

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/api/compare-runs')
def api_compare_runs():
    """Compare two test runs and return differences.

    Query params:
        run1: older run ID
        run2: newer run ID
        OR
        font_id: compare last two runs for this font
    """
    ensure_test_tables()
    db = get_db()

    run1_id = request.args.get('run1', type=int)
    run2_id = request.args.get('run2', type=int)
    font_id = request.args.get('font_id', type=int)

    if font_id and not (run1_id and run2_id):
        # Get last two runs for this font
        runs = db.execute("""
            SELECT id FROM test_runs WHERE font_id = ?
            ORDER BY run_date DESC LIMIT 2
        """, (font_id,)).fetchall()
        if len(runs) < 2:
            db.close()
            return jsonify(error="Need at least 2 runs to compare"), 400
        run2_id = runs[0]['id']  # newer
        run1_id = runs[1]['id']  # older

    if not run1_id or not run2_id:
        db.close()
        return jsonify(error="Must specify run1 & run2, or font_id"), 400

    run1 = db.execute("SELECT * FROM test_runs WHERE id = ?", (run1_id,)).fetchone()
    run2 = db.execute("SELECT * FROM test_runs WHERE id = ?", (run2_id,)).fetchone()
    db.close()

    if not run1 or not run2:
        return jsonify(error="Run not found"), 404

    results1 = json.loads(run1['results_json']) if run1['results_json'] else []
    results2 = json.loads(run2['results_json']) if run2['results_json'] else []

    # Build lookup by char
    map1 = {r['char']: r for r in results1}
    map2 = {r['char']: r for r in results2}

    comparisons = []
    all_chars = set(map1.keys()) | set(map2.keys())

    for char in sorted(all_chars):
        r1 = map1.get(char, {})
        r2 = map2.get(char, {})

        old_score = r1.get('score', 0) if r1.get('status') == 'ok' else None
        new_score = r2.get('score', 0) if r2.get('status') == 'ok' else None

        if old_score is not None and new_score is not None:
            delta = new_score - old_score
        else:
            delta = None

        comparisons.append({
            'char': char,
            'old_score': old_score,
            'new_score': new_score,
            'delta': round(delta, 3) if delta is not None else None,
            'old_coverage': r1.get('coverage'),
            'new_coverage': r2.get('coverage'),
            'old_topology': r1.get('topology_score'),
            'new_topology': r2.get('topology_score'),
        })

    # Sort by delta (biggest regressions first)
    comparisons.sort(key=lambda x: x['delta'] if x['delta'] is not None else 0)

    return jsonify(
        ok=True,
        run1_id=run1_id,
        run2_id=run2_id,
        run1_date=run1['run_date'],
        run2_date=run2['run_date'],
        font_id=run1['font_id'],
        old_avg=run1['avg_score'],
        new_avg=run2['avg_score'],
        comparisons=comparisons
    )


@app.route('/compare/<int:font_id>')
def compare_page(font_id):
    """Show comparison page for a font's last two test runs."""
    db = get_db()
    font = db.execute("SELECT * FROM fonts WHERE id = ?", (font_id,)).fetchone()
    db.close()
    if not font:
        return "Font not found", 404
    return render_template('compare.html', font=font)


def _ray_to_border(mask, x, y, dx, dy, max_steps=300):
    """Walk from (x,y) in direction (dx,dy) until leaving the glyph mask.
    Returns distance to border, or None if never left within max_steps."""
    h, w = mask.shape
    cx, cy = x, y
    for step in range(1, max_steps):
        nx = x + dx * step
        ny = y + dy * step
        ix, iy = int(round(nx)), int(round(ny))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return step  # hit canvas edge
        if not mask[iy, ix]:
            return step  # hit border (left glyph)
    return None


@app.route('/api/center-borders/<int:font_id>', methods=['POST'])
def api_center_borders(font_id):
    """Center each stroke point between the two closest parallel glyph borders."""
    char = request.args.get('c')
    if not char:
        return jsonify(error="Missing ?c= parameter"), 400

    data = request.get_json()
    if not data or 'strokes' not in data:
        return jsonify(error="Missing strokes data"), 400

    db = get_db()
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?", (font_id,)).fetchone()
    db.close()
    if not font:
        return jsonify(error="Font not found"), 404

    mask = render_glyph_mask(font['file_path'], char)
    if mask is None:
        return jsonify(error="Could not render glyph"), 500

    # Precompute ray directions (every 5 degrees, but only need 0-180 since
    # opposite directions are checked as a pair)
    n_angles = 36
    angles = [i * np.pi / n_angles for i in range(n_angles)]
    directions = [(np.cos(a), np.sin(a)) for a in angles]

    # Interior distance field to find nearest inside point for outside/edge pts
    dist_in = distance_transform_edt(mask)
    outside_mask = ~mask
    dist_out, snap_indices = distance_transform_edt(outside_mask, return_indices=True)
    h, w = mask.shape

    result = []
    for stroke in data['strokes']:
        centered = []
        for p in stroke:
            x, y = p[0], p[1]
            locked = len(p) >= 3 and p[2] == 1
            ix = int(round(min(max(x, 0), w - 1)))
            iy = int(round(min(max(y, 0), h - 1)))

            # If point is outside or right on the glyph edge, nudge it inside
            # so ray-casting works from a valid interior position.
            if not mask[iy, ix]:
                # Snap to nearest inside pixel
                ny = float(snap_indices[0, iy, ix])
                nx = float(snap_indices[1, iy, ix])
                x, y = nx, ny
                ix, iy = int(round(x)), int(round(y))

            if ix < 0 or ix >= w or iy < 0 or iy >= h or not mask[iy, ix]:
                centered.append([p[0], p[1], 1] if locked else [p[0], p[1]])
                continue

            # If very close to edge (dist < 2), nudge inward first
            # so rays can fire in all directions
            if dist_in[iy, ix] < 2:
                # Find nearest pixel at least 2px inside
                search_r = 5
                best_d = float('inf')
                best_xy = (x, y)
                for sy in range(max(0, iy - search_r), min(h, iy + search_r + 1)):
                    for sx in range(max(0, ix - search_r), min(w, ix + search_r + 1)):
                        if dist_in[sy, sx] >= 2:
                            dd = (sx - x) ** 2 + (sy - y) ** 2
                            if dd < best_d:
                                best_d = dd
                                best_xy = (float(sx), float(sy))
                if best_d < float('inf'):
                    x, y = best_xy

            # Cast rays in opposite directions, find shortest crossing line
            best_total = float('inf')
            best_mid = (x, y)

            for dx, dy in directions:
                d_pos = _ray_to_border(mask, x, y, dx, dy)
                d_neg = _ray_to_border(mask, x, y, -dx, -dy)

                if d_pos is not None and d_neg is not None:
                    total = d_pos + d_neg
                    if total < best_total:
                        best_total = total
                        # Midpoint of the crossing line
                        half = (d_pos - d_neg) / 2.0
                        best_mid = (x + dx * half, y + dy * half)

            sp = [best_mid[0], best_mid[1], 1] if locked else [best_mid[0], best_mid[1]]
            centered.append(sp)
        result.append(centered)

    return jsonify(strokes=result)


def _analyze_skeleton(mask):
    """Skeletonize a mask and return adjacency, junction clusters, and endpoints.

    Delegates to stroke_lib.analysis.skeleton.SkeletonAnalyzer for the core analysis.
    """
    analyzer = SLSkeletonAnalyzer(merge_distance=12)
    info = analyzer.analyze(mask)
    if info is None:
        return None

    # Convert SkeletonInfo to dict format for backwards compatibility
    # Convert adj from set-based to list-based (downstream code uses indexing)
    adj_as_lists = defaultdict(list)
    for pixel, neighbors in info.adj.items():
        adj_as_lists[pixel] = list(neighbors)

    return {
        'skel_set': info.skel_set,
        'adj': adj_as_lists,
        'junction_pixels': info.junction_pixels,
        'junction_clusters': info.junction_clusters,
        'assigned': info.assigned,
        'endpoints': info.endpoints,
    }


def skeleton_detect_markers(mask, merge_dist=12):
    """Detect vertex (junction) and termination (endpoint) markers from skeleton.

    Delegates to stroke_lib.analysis.skeleton.SkeletonAnalyzer for marker detection.

    Vertices = centroids of junction clusters (where 3+ branches meet).
    Terminations = skeleton endpoints (degree 1 pixels).
    Nearby vertices are merged. Terminations that fall inside a junction
    cluster are removed (they're part of the junction, not real endpoints).
    """
    analyzer = SLSkeletonAnalyzer(merge_distance=merge_dist)
    sl_markers = analyzer.detect_markers(mask)

    # Convert Marker objects to dict format for backwards compatibility
    return [m.to_dict() for m in sl_markers]


# ============================================================================
# Helper functions for skeleton_to_strokes (moved from nested to module level)
# ============================================================================

def _seg_dir(seg, from_end, n=8):
    """Direction vector at one end of a segment (skip junction pixels)."""
    if from_end:
        pts = seg[-min(n, len(seg)):]
    else:
        pts = seg[:min(n, len(seg))][::-1]
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    length = (dx * dx + dy * dy) ** 0.5
    return (dx / length, dy / length) if length > 0.01 else (0, 0)


def _angle(d1, d2):
    """Angle between two direction vectors."""
    dot = d1[0] * d2[0] + d1[1] * d2[1]
    return np.arccos(max(-1.0, min(1.0, dot)))


def _endpoint_cluster(stroke, from_end, assigned):
    """Which junction cluster does this stroke endpoint belong to?"""
    pt = tuple(stroke[-1]) if from_end else tuple(stroke[0])
    return assigned.get(pt, -1)


def _trace_skeleton_segment(start, neighbor, adj, stop_set, visited_edges):
    """Trace a skeleton path from start through neighbor until a stop point."""
    edge = (min(start, neighbor), max(start, neighbor))
    if edge in visited_edges:
        return None
    visited_edges.add(edge)
    path = [start, neighbor]
    current, prev = neighbor, start
    while True:
        if current in stop_set and len(path) > 2:
            break
        neighbors = [n for n in adj[current] if n != prev]
        # Filter to unvisited edges
        candidates = []
        for n in neighbors:
            e = (min(current, n), max(current, n))
            if e not in visited_edges:
                candidates.append((n, e))
        if not candidates:
            break
        # Pick the neighbor that continues straightest
        if len(candidates) == 1:
            next_pt, next_edge = candidates[0]
        else:
            # Direction of travel: use last few path points for stability
            n_look = min(4, len(path))
            dx_in = current[0] - path[-n_look][0]
            dy_in = current[1] - path[-n_look][1]
            len_in = (dx_in * dx_in + dy_in * dy_in) ** 0.5
            if len_in > 0.01:
                dx_in /= len_in
                dy_in /= len_in
            best_dot = -2
            next_pt, next_edge = candidates[0]
            for n, e in candidates:
                dx_out = n[0] - current[0]
                dy_out = n[1] - current[1]
                len_out = (dx_out * dx_out + dy_out * dy_out) ** 0.5
                if len_out > 0.01:
                    dot = (dx_in * dx_out + dy_in * dy_out) / len_out
                else:
                    dot = 0
                if dot > best_dot:
                    best_dot = dot
                    next_pt, next_edge = n, e
        visited_edges.add(next_edge)
        path.append(next_pt)
        prev, current = current, next_pt
    return path


def _run_merge_pass(strokes, assigned, min_len=0, max_angle=np.pi/4, max_ratio=0):
    """Merge strokes through junction clusters by direction alignment.

    max_ratio > 0 means reject pairs where max(len)/min(len) > ratio.
    """
    changed = True
    while changed:
        changed = False
        cluster_map = defaultdict(list)
        for si, s in enumerate(strokes):
            sc = _endpoint_cluster(s, False, assigned)
            if sc >= 0:
                cluster_map[sc].append((si, 'start'))
            ec = _endpoint_cluster(s, True, assigned)
            if ec >= 0:
                cluster_map[ec].append((si, 'end'))

        best_score = float('inf')
        best_merge = None
        for cid, entries in cluster_map.items():
            if len(entries) < 2:
                continue
            for ai in range(len(entries)):
                si, side_i = entries[ai]
                dir_i = _seg_dir(strokes[si], from_end=(side_i == 'end'))
                for bi in range(ai + 1, len(entries)):
                    sj, side_j = entries[bi]
                    if sj == si:
                        continue
                    li, lj = len(strokes[si]), len(strokes[sj])
                    if min(li, lj) < min_len:
                        continue
                    if max_ratio > 0 and max(li, lj) / max(min(li, lj), 1) > max_ratio:
                        continue
                    # Don't merge with a loop stroke (both endpoints at same cluster)
                    sci = _endpoint_cluster(strokes[si], False, assigned)
                    eci = _endpoint_cluster(strokes[si], True, assigned)
                    scj = _endpoint_cluster(strokes[sj], False, assigned)
                    ecj = _endpoint_cluster(strokes[sj], True, assigned)
                    if sci >= 0 and sci == eci:
                        continue
                    if scj >= 0 and scj == ecj:
                        continue
                    dir_j = _seg_dir(strokes[sj], from_end=(side_j == 'end'))
                    angle = np.pi - _angle(dir_i, dir_j)
                    if angle < max_angle and angle < best_score:
                        best_score = angle
                        best_merge = (si, side_i, sj, side_j)

        if best_merge:
            si, side_i, sj, side_j = best_merge
            seg_i = strokes[si] if side_i == 'end' else list(reversed(strokes[si]))
            seg_j = strokes[sj] if side_j == 'start' else list(reversed(strokes[sj]))
            merged_stroke = seg_i + seg_j[1:]
            hi, lo = max(si, sj), min(si, sj)
            strokes.pop(hi)
            strokes.pop(lo)
            strokes.append(merged_stroke)
            changed = True
    return strokes


def _merge_t_junctions(strokes, junction_clusters, assigned):
    """Merge strokes at T-junctions where a short cross-branch connects two main branches.

    At junctions with 3+ strokes, if the shortest stroke is a cross-branch (both endpoints
    in junction clusters) and much shorter than the main branches, merge the two longest
    with a relaxed angle threshold.
    """
    changed = True
    while changed:
        changed = False
        cluster_map = defaultdict(list)
        for si, s in enumerate(strokes):
            sc = _endpoint_cluster(s, False, assigned)
            if sc >= 0:
                cluster_map[sc].append((si, 'start'))
            ec = _endpoint_cluster(s, True, assigned)
            if ec >= 0:
                cluster_map[ec].append((si, 'end'))

        for cid, entries in cluster_map.items():
            if len(entries) < 3:
                continue
            entries_sorted = sorted(entries, key=lambda e: len(strokes[e[0]]),
                                    reverse=True)
            shortest_idx, shortest_side = entries_sorted[-1]
            shortest_stroke = strokes[shortest_idx]
            second_longest_len = len(strokes[entries_sorted[1][0]])
            # Shortest must be a cross-branch (both ends at junctions)
            s_sc = _endpoint_cluster(shortest_stroke, False, assigned)
            s_ec = _endpoint_cluster(shortest_stroke, True, assigned)
            if s_sc < 0 or s_ec < 0:
                continue
            if len(shortest_stroke) >= second_longest_len * 0.4:
                continue
            # Merge the two longest with relaxed angle (120°)
            si, side_i = entries_sorted[0]
            sj, side_j = entries_sorted[1]
            if si == sj:
                continue
            # Don't merge if result would be a loop
            far_i = _endpoint_cluster(strokes[si], from_end=(side_i != 'end'), assigned=assigned)
            far_j = _endpoint_cluster(strokes[sj], from_end=(side_j != 'end'), assigned=assigned)
            if far_i >= 0 and far_i == far_j:
                continue
            dir_i = _seg_dir(strokes[si], from_end=(side_i == 'end'))
            dir_j = _seg_dir(strokes[sj], from_end=(side_j == 'end'))
            angle = np.pi - _angle(dir_i, dir_j)
            if angle < 2 * np.pi / 3:
                seg_i = strokes[si] if side_i == 'end' else list(reversed(strokes[si]))
                seg_j = strokes[sj] if side_j == 'start' else list(reversed(strokes[sj]))
                merged_stroke = seg_i + seg_j[1:]
                hi, lo = max(si, sj), min(si, sj)
                strokes.pop(hi)
                strokes.pop(lo)
                strokes.append(merged_stroke)
                # Also remove the cross-branch (it's now redundant)
                for sk in range(len(strokes)):
                    s = strokes[sk]
                    s_sc2 = _endpoint_cluster(s, False, assigned)
                    s_ec2 = _endpoint_cluster(s, True, assigned)
                    if s_sc2 >= 0 and s_ec2 >= 0 and len(s) < second_longest_len * 0.4:
                        if s_sc2 == cid or s_ec2 == cid:
                            strokes.pop(sk)
                            break
                changed = True
                break
    return strokes


def _absorb_convergence_stubs(strokes, junction_clusters, assigned, conv_threshold=18):
    """Absorb short convergence stubs into longer strokes at junction clusters.

    A convergence stub is a short stroke with one endpoint in a junction cluster
    and the other end free (e.g. the pointed apex of letter A). Extends all other
    strokes at the cluster toward the stub tip, then removes the stub.
    """
    changed = True
    while changed:
        changed = False
        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) < 2 or len(s) >= conv_threshold:
                continue
            sc = _endpoint_cluster(s, False, assigned)
            ec = _endpoint_cluster(s, True, assigned)

            # One end in a junction cluster, other end free (or same cluster)
            if sc >= 0 and ec < 0:
                cluster_id = sc
                stub_path = list(s)
            elif ec >= 0 and sc < 0:
                cluster_id = ec
                stub_path = list(reversed(s))
            elif sc >= 0 and ec >= 0 and sc == ec:
                cluster_id = sc
                cluster = junction_clusters[sc]
                cx = sum(p[0] for p in cluster) / len(cluster)
                cy = sum(p[1] for p in cluster) / len(cluster)
                d_start = ((s[0][0] - cx) ** 2 + (s[0][1] - cy) ** 2) ** 0.5
                d_end = ((s[-1][0] - cx) ** 2 + (s[-1][1] - cy) ** 2) ** 0.5
                stub_path = list(reversed(s)) if d_start > d_end else list(s)
            else:
                continue

            # Only absorb if other strokes also arrive at this cluster
            others_at_cluster = 0
            for sj in range(len(strokes)):
                if sj == si:
                    continue
                if _endpoint_cluster(strokes[sj], False, assigned) == cluster_id:
                    others_at_cluster += 1
                if _endpoint_cluster(strokes[sj], True, assigned) == cluster_id:
                    others_at_cluster += 1
            if others_at_cluster < 2:
                continue

            # Extend every other stroke at this cluster toward the stub tip
            stub_tip = stub_path[-1]
            cluster = junction_clusters[cluster_id]
            for sj in range(len(strokes)):
                if sj == si:
                    continue
                s2 = strokes[sj]
                at_end = _endpoint_cluster(s2, True, assigned) == cluster_id
                at_start = (not at_end) and _endpoint_cluster(s2, False, assigned) == cluster_id
                if not at_end and not at_start:
                    continue

                # Get the last few points before the junction to determine direction
                if at_end:
                    tail = []
                    for k in range(len(s2) - 1, -1, -1):
                        pt = tuple(s2[k]) if isinstance(s2[k], (list, tuple)) else s2[k]
                        if len(tail) >= 8:
                            break
                        if (int(round(pt[0])), int(round(pt[1]))) not in cluster or not tail:
                            tail.insert(0, pt)
                    leg_end = s2[-1]
                else:
                    tail = []
                    for k in range(len(s2)):
                        pt = tuple(s2[k]) if isinstance(s2[k], (list, tuple)) else s2[k]
                        if len(tail) >= 8:
                            break
                        if (int(round(pt[0])), int(round(pt[1]))) not in cluster or not tail:
                            tail.append(pt)
                    tail = list(reversed(tail))
                    leg_end = s2[0]

                # Use direction from pre-junction points to extrapolate to stub tip
                if len(tail) >= 2:
                    dx = tail[-1][0] - tail[0][0]
                    dy = tail[-1][1] - tail[0][1]
                    leg_len = (dx * dx + dy * dy) ** 0.5
                else:
                    leg_len = 0
                tip_dx = stub_tip[0] - leg_end[0]
                tip_dy = stub_tip[1] - leg_end[1]
                tip_dist = (tip_dx * tip_dx + tip_dy * tip_dy) ** 0.5
                steps = max(1, int(round(tip_dist)))

                if leg_len > 0.01:
                    ux, uy = dx / leg_len, dy / leg_len
                    ext_pts = []
                    for k in range(1, steps + 1):
                        t = k / steps
                        ex = leg_end[0] + ux * k
                        ey = leg_end[1] + uy * k
                        px = ex * (1 - t) + stub_tip[0] * t
                        py = ey * (1 - t) + stub_tip[1] * t
                        ext_pts.append((px, py))
                else:
                    ext_pts = []
                    for k in range(1, steps + 1):
                        t = k / steps
                        ext_pts.append((leg_end[0] + tip_dx * t,
                                        leg_end[1] + tip_dy * t))

                if at_end:
                    s2.extend(ext_pts)
                else:
                    for p in reversed(ext_pts):
                        s2.insert(0, p)

            strokes.pop(si)
            changed = True
            break
    return strokes


def _absorb_junction_stubs(strokes, assigned, stub_threshold=20):
    """Absorb short stubs into neighboring strokes at junction clusters.

    Any stroke shorter than stub_threshold that touches a junction cluster
    gets appended to the longest stroke sharing that junction.
    """
    changed = True
    while changed:
        changed = False
        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) >= stub_threshold:
                continue
            sc = _endpoint_cluster(s, False, assigned)
            ec = _endpoint_cluster(s, True, assigned)
            clusters_touching = set()
            if sc >= 0:
                clusters_touching.add(sc)
            if ec >= 0:
                clusters_touching.add(ec)
            if not clusters_touching:
                continue

            # Find longest other stroke at any shared junction
            best_target = -1
            best_len = 0
            best_target_side = None
            best_stub_side = None
            for cid in clusters_touching:
                for sj in range(len(strokes)):
                    if sj == si:
                        continue
                    s2 = strokes[sj]
                    tc_start = _endpoint_cluster(s2, False, assigned)
                    tc_end = _endpoint_cluster(s2, True, assigned)
                    if tc_start == cid and len(s2) > best_len:
                        best_target = sj
                        best_len = len(s2)
                        best_target_side = 'start'
                        best_stub_side = 'start' if sc == cid else 'end'
                    if tc_end == cid and len(s2) > best_len:
                        best_target = sj
                        best_len = len(s2)
                        best_target_side = 'end'
                        best_stub_side = 'start' if sc == cid else 'end'

            if best_target >= 0:
                stub = s if best_stub_side == 'end' else list(reversed(s))
                target = strokes[best_target]
                if best_target_side == 'end':
                    strokes[best_target] = target + stub[1:]
                else:
                    strokes[best_target] = list(reversed(stub[1:])) + target
                strokes.pop(si)
                changed = True
                break
    return strokes


def _absorb_proximity_stubs(strokes, stub_threshold=20, prox_threshold=20):
    """Absorb short stubs by proximity to longer stroke endpoints.

    Any remaining short stroke whose endpoint is near a longer stroke's endpoint
    gets appended to that stroke.
    """
    changed = True
    while changed:
        changed = False
        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) >= stub_threshold:
                continue
            best_dist = prox_threshold
            best_target = -1
            best_target_side = None
            best_stub_side = None
            for stub_end in [False, True]:
                sp = s[-1] if stub_end else s[0]
                for sj in range(len(strokes)):
                    if sj == si or len(strokes[sj]) < stub_threshold:
                        continue
                    for target_end in [False, True]:
                        tp = strokes[sj][-1] if target_end else strokes[sj][0]
                        d = ((sp[0] - tp[0]) ** 2 + (sp[1] - tp[1]) ** 2) ** 0.5
                        if d < best_dist:
                            best_dist = d
                            best_target = sj
                            best_target_side = 'end' if target_end else 'start'
                            best_stub_side = 'end' if stub_end else 'start'
            if best_target >= 0:
                stub = s if best_stub_side == 'end' else list(reversed(s))
                target = strokes[best_target]
                if best_target_side == 'end':
                    strokes[best_target] = target + stub[1:]
                else:
                    strokes[best_target] = list(reversed(stub[1:])) + target
                strokes.pop(si)
                changed = True
                break
    return strokes


def _remove_orphan_stubs(strokes, assigned, stub_threshold=20):
    """Remove orphaned short stubs that have no neighbors at their junction clusters.

    Any stroke shorter than stub_threshold that has an endpoint at a junction cluster
    where no other stroke touches is an artifact.
    """
    changed = True
    while changed:
        changed = False
        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) >= stub_threshold:
                continue
            sc = _endpoint_cluster(s, False, assigned)
            ec = _endpoint_cluster(s, True, assigned)
            has_neighbor = False
            for sj in range(len(strokes)):
                if sj == si:
                    continue
                if sc >= 0 and (_endpoint_cluster(strokes[sj], False, assigned) == sc or
                                _endpoint_cluster(strokes[sj], True, assigned) == sc):
                    has_neighbor = True
                    break
                if ec >= 0 and (_endpoint_cluster(strokes[sj], False, assigned) == ec or
                                _endpoint_cluster(strokes[sj], True, assigned) == ec):
                    has_neighbor = True
                    break
            if not has_neighbor:
                strokes.pop(si)
                changed = True
                break
    return strokes


def skeleton_to_strokes(mask, min_stroke_len=5):
    """Extract stroke paths from a glyph mask via skeletonization."""
    info = _analyze_skeleton(mask)
    if not info:
        return []

    adj = info['adj']
    junction_pixels = info['junction_pixels']
    junction_clusters = info['junction_clusters']
    assigned = info['assigned']
    endpoints = info['endpoints']

    # For tracing, all junction cluster pixels are stop points
    stop_set = endpoints | junction_pixels

    # Trace initial strokes from endpoints and junction pixels
    visited_edges = set()
    raw_strokes = []
    for start in sorted(endpoints):
        for neighbor in adj[start]:
            p = _trace_skeleton_segment(start, neighbor, adj, stop_set, visited_edges)
            if p and len(p) >= 2:
                raw_strokes.append(p)
    for start in sorted(junction_pixels):
        for neighbor in adj[start]:
            p = _trace_skeleton_segment(start, neighbor, adj, stop_set, visited_edges)
            if p and len(p) >= 2:
                raw_strokes.append(p)

    # Filter tiny stubs
    strokes = [s for s in raw_strokes if len(s) >= min_stroke_len]

    # Pass 1: T-junction merge
    strokes = _merge_t_junctions(strokes, junction_clusters, assigned)

    # Pass 2: standard direction-based merge
    strokes = _run_merge_pass(strokes, assigned, min_len=0)

    # Absorb convergence stubs
    strokes = _absorb_convergence_stubs(strokes, junction_clusters, assigned)

    # Absorb remaining short stubs into neighboring strokes
    strokes = _absorb_junction_stubs(strokes, assigned)

    # Proximity-based stub absorption
    strokes = _absorb_proximity_stubs(strokes)

    # Remove orphaned short stubs
    strokes = _remove_orphan_stubs(strokes, assigned)

    # Convert to float coords
    return [[[float(x), float(y)] for x, y in s] for s in strokes]


@app.route('/api/detect-markers/<int:font_id>', methods=['POST'])
def api_detect_markers(font_id):
    """Auto-detect vertex and termination markers from skeleton."""
    char = request.args.get('c')
    if not char:
        return jsonify(error="Missing ?c= parameter"), 400

    db = get_db()
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?", (font_id,)).fetchone()
    db.close()
    if not font:
        return jsonify(error="Font not found"), 404

    mask = render_glyph_mask(font['file_path'], char)
    if mask is None:
        return jsonify(error="Could not render glyph"), 500

    markers = skeleton_detect_markers(mask)
    return jsonify(markers=markers)


@app.route('/api/clear-shape-cache/<int:font_id>', methods=['POST'])
def api_clear_shape_cache(font_id):
    """Clear cached shape optimization params for a font+char or all chars."""
    char = request.args.get('c')
    db = get_db()
    if char:
        db.execute("""
            UPDATE characters SET shape_params_cache = NULL
            WHERE font_id = ? AND char = ?
        """, (font_id, char))
    else:
        db.execute("""
            UPDATE characters SET shape_params_cache = NULL
            WHERE font_id = ?
        """, (font_id,))
    db.commit()
    db.close()
    return jsonify(ok=True)


@app.route('/api/optimize-stream/<int:font_id>', methods=['GET'])
class OptimizationStreamState:
    """Mutable state container for streaming optimization."""

    def __init__(self, shape_types, slices, glyph_bbox, mask, bounds_lo, bounds_hi):
        self.shape_types = shape_types
        self.slices = slices
        self.glyph_bbox = glyph_bbox
        self.mask = mask
        self.bounds_lo = bounds_lo
        self.bounds_hi = bounds_hi
        self.frame_num = 0
        self.last_emit = 0.0
        self.best_x = None
        self.best_fun = float('inf')

    def clamp(self, x):
        return np.clip(x, self.bounds_lo, self.bounds_hi)

    def update_best(self, x, fun):
        if fun < self.best_fun:
            self.best_x = x.copy()
            self.best_fun = fun

    def is_perfect(self):
        return self.best_fun <= -0.99

    def emit_frame(self, x, fun, phase=''):
        """Build strokes from params and return SSE event JSON."""
        import time as _time
        x = self.clamp(x)
        shapes = _param_vector_to_shapes(x, self.shape_types, self.slices, self.glyph_bbox)
        stroke_list = []
        for pts in shapes:
            pl = [(float(p[0]), float(p[1])) for p in pts]
            pl = _smooth_stroke(pl, sigma=2.0)
            pl = _constrain_to_mask(pl, self.mask)
            if len(pl) >= 2:
                stroke_list.append([[round(px, 1), round(py, 1)] for px, py in pl])
        self.frame_num += 1
        self.last_emit = _time.monotonic()
        return json.dumps({
            'frame': self.frame_num,
            'score': round(float(-fun), 4),
            'phase': phase,
            'strokes': stroke_list,
        })

    def emit_raw_frame(self, stroke_list, score, phase=''):
        """Emit SSE event for raw strokes (not shape-param based)."""
        import time as _time
        self.frame_num += 1
        self.last_emit = _time.monotonic()
        return json.dumps({
            'frame': self.frame_num,
            'score': round(float(score), 4),
            'phase': phase,
            'strokes': stroke_list,
        })


def _stream_phase0_template_matching(font_path, char, state):
    """Phase 0: Try DiffVG and affine template matching.

    Args:
        font_path: Resolved font path
        char: Character to optimize
        state: OptimizationStreamState

    Yields:
        SSE frame strings
        Final yield is tuple (affine_result, done) where affine_result is (strokes, score) or None
    """
    affine_strokes_result = None

    # Try DiffVG first
    diffvg_result = _optimize_diffvg(font_path, char, 224)
    if diffvg_result is not None:
        dv_strokes, dv_score, _, _ = diffvg_result
        if dv_strokes and dv_score > 0:
            affine_strokes_result = (dv_strokes, dv_score)
            yield f"data: {state.emit_raw_frame(dv_strokes, dv_score, 'diffvg')}\n\n"

            if dv_score >= 0.85:
                yield f"data: {state.emit_raw_frame(dv_strokes, dv_score, 'final')}\n\n"
                yield f"data: {json.dumps({'done': True, 'score': round(dv_score, 4), 'frame': state.frame_num})}\n\n"
                yield (affine_strokes_result, True)
                return

    # Also try affine
    affine_result = _optimize_affine(font_path, char, 224)
    if affine_result is not None:
        affine_strokes, affine_score, _, _ = affine_result
        affine_stroke_list = [[[round(float(x), 1), round(float(y), 1)] for x, y in s]
                              for s in affine_strokes if len(s) >= 2]
        if affine_stroke_list:
            if affine_strokes_result and affine_strokes_result[1] >= affine_score:
                yield f"data: {state.emit_raw_frame(affine_stroke_list, affine_score, 'affine template (DiffVG better)')}\n\n"
            else:
                affine_strokes_result = (affine_stroke_list, affine_score)
                yield f"data: {state.emit_raw_frame(affine_stroke_list, affine_score, 'affine template')}\n\n"

            if affine_strokes_result[1] >= 0.85:
                best_strokes, best_score = affine_strokes_result
                yield f"data: {state.emit_raw_frame(best_strokes, best_score, 'final')}\n\n"
                yield f"data: {json.dumps({'done': True, 'score': round(best_score, 4), 'frame': state.frame_num})}\n\n"
                yield (affine_strokes_result, True)
                return

    yield (affine_strokes_result, False)


def api_optimize_stream(font_id):
    """SSE endpoint: streams optimization frames in real time.

    Mirrors the full auto_fit_strokes pipeline but yields frames
    after every improvement so the frontend can show exactly what
    the optimizer is doing.

    Each SSE event is JSON with:
      frame: int, score: float, phase: str, strokes: [[[x,y],...],...]
    """
    char = request.args.get('c')
    if not char:
        return jsonify(error="Missing ?c= parameter"), 400

    db = get_db()
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?",
                       (font_id,)).fetchone()
    db.close()
    if not font:
        return jsonify(error="Font not found"), 404

    def generate():
        import time as _time
        from scipy.optimize import differential_evolution, minimize

        font_path = resolve_font_path(font['file_path'])
        templates = SHAPE_TEMPLATES.get(char)
        if not templates:
            yield f"data: {json.dumps({'error': 'no template'})}\n\n"
            return

        # Setup using shared helper
        setup = _setup_auto_fit(font_path, char, 224, templates)
        if setup is None:
            yield f"data: {json.dumps({'error': 'setup failed'})}\n\n"
            return

        mask = setup['mask']
        glyph_bbox = setup['glyph_bbox']
        cloud = setup['cloud']
        cloud_tree = setup['cloud_tree']
        n_cloud = setup['n_cloud']
        radius = setup['radius']
        h, w = setup['h'], setup['w']
        snap_yi, snap_xi = setup['snap_yi'], setup['snap_xi']
        shape_types = setup['shape_types']
        bounds = setup['bounds']
        slices = setup['slices']
        bounds_lo = setup['bounds_lo']
        bounds_hi = setup['bounds_hi']
        dist_map = setup['dist_map']

        x0 = setup['x0'].copy()

        rel_path = font_path
        if rel_path.startswith(BASE_DIR):
            rel_path = os.path.relpath(font_path, BASE_DIR)
        cached = _load_cached_params(rel_path, char)
        cached_score = None
        if cached is not None and len(cached[0]) == len(x0):
            x0 = cached[0]
            cached_score = cached[1]

        for i, (lo, hi) in enumerate(bounds):
            x0[i] = np.clip(x0[i], lo, hi)

        joint_args = (shape_types, slices, glyph_bbox, cloud_tree, n_cloud,
                      radius, snap_yi, snap_xi, w, h, dist_map)

        # Create state object
        state = OptimizationStreamState(shape_types, slices, glyph_bbox, mask,
                                        bounds_lo, bounds_hi)

        _t_start = _time.monotonic()
        _TIME_BUDGET = 3600.0
        _STALE_THRESHOLD = 0.001
        _STALE_CYCLES = 2

        def _elapsed():
            return _time.monotonic() - _t_start

        # ---- Phase 0: Template matching ----
        affine_strokes_result = None
        for item in _stream_phase0_template_matching(font_path, char, state):
            if isinstance(item, tuple):
                affine_strokes_result, done = item
                if done:
                    return
            else:
                yield item

        # Emit initial frame
        init_fun = _score_all_strokes(x0, *joint_args)
        yield f"data: {state.emit_frame(x0, init_fun, 'initial')}\n\n"

        state.best_x = x0.copy()
        state.best_fun = init_fun

        # ---- Phase 1: Greedy per-shape ----
        greedy_x = x0.copy()
        uncovered_mask = np.ones(n_cloud, dtype=bool)
        n_pts_shape = max(60, int(((glyph_bbox[2]-glyph_bbox[0])**2 +
                                    (glyph_bbox[3]-glyph_bbox[1])**2)**0.5 / 1.5))

        for si in range(len(templates)):
            if _elapsed() >= _TIME_BUDGET * 0.4:
                break
            start, end = slices[si]
            stype = shape_types[si]
            s_bounds = bounds[start:end]
            s_x0 = greedy_x[start:end].copy()

            uncov_idx = np.where(uncovered_mask)[0]
            if len(uncov_idx) < 5:
                break
            uncov_pts = cloud[uncov_idx]
            uncov_tree = cKDTree(uncov_pts)

            s_args = (stype, glyph_bbox, uncov_pts, uncov_tree,
                      len(uncov_pts), radius, snap_yi, snap_xi, w, h)

            s_lo, s_hi = bounds_lo[start:end], bounds_hi[start:end]

            def _greedy_nm_obj(params, _s_lo=s_lo, _s_hi=s_hi, _s_args=s_args):
                return _score_single_shape(np.clip(params, _s_lo, _s_hi), *_s_args)

            nm_r = minimize(_greedy_nm_obj, s_x0, method='Nelder-Mead',
                           options={'maxfev': 800, 'xatol': 0.2, 'fatol': 0.002, 'adaptive': True})
            best_s = np.clip(nm_r.x, s_lo, s_hi).copy()
            best_sf = nm_r.fun

            if _elapsed() < _TIME_BUDGET * 0.35:
                try:
                    de_r = differential_evolution(_score_single_shape, bounds=s_bounds,
                                                  args=s_args, x0=best_s, maxiter=30,
                                                  popsize=12, tol=0.005, polish=False, disp=False)
                    if de_r.fun < best_sf:
                        best_s = de_r.x.copy()
                except Exception:
                    pass

            greedy_x[start:end] = best_s

            # Mark covered points
            pts = SHAPE_FNS[stype](tuple(best_s), glyph_bbox, offset=(0, 0), n_pts=n_pts_shape)
            if len(pts) > 0:
                xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
                yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
                snapped = np.column_stack([snap_xi[yi, xi].astype(float),
                                           snap_yi[yi, xi].astype(float)])
                hits = cloud_tree.query_ball_point(snapped, radius)
                for lst in hits:
                    for idx in lst:
                        uncovered_mask[idx] = False

            greedy_fun = _score_all_strokes(greedy_x, *joint_args)
            state.update_best(greedy_x, greedy_fun)
            uncov_remaining = int(uncovered_mask.sum())
            yield f"data: {state.emit_frame(greedy_x, greedy_fun, f'greedy shape {si+1}/{len(templates)} ({stype}) uncov={uncov_remaining}')}\n\n"

        # Compare with x0
        x0_fun = _score_all_strokes(x0, *joint_args)
        if x0_fun < state.best_fun:
            state.best_x = x0.copy()
            state.best_fun = x0_fun

        class _EarlyStop(Exception):
            pass

        # ---- Phase 2: Refinement cycles ----
        stale_count = 0
        cycle_num = 0
        while not state.is_perfect() and _elapsed() < _TIME_BUDGET and stale_count < _STALE_CYCLES:
            cycle_num += 1
            score_at_cycle_start = state.best_fun

            # NM refinement
            if not state.is_perfect() and _elapsed() < _TIME_BUDGET:
                nm_frames = []
                nm_best = [state.best_x.copy()]
                nm_best_f = [state.best_fun]

                def _nm_obj_stream(params, _nb=nm_best, _nbf=nm_best_f, _nf=nm_frames, _cn=cycle_num):
                    params = state.clamp(params)
                    val = _score_all_strokes(params, *joint_args)
                    now = _time.monotonic()
                    if val < _nbf[0]:
                        _nb[0] = params.copy()
                        _nbf[0] = val
                        _nf.append(state.emit_frame(params, val, f'cycle {_cn} NM improve'))
                    elif (now - state.last_emit) > 1.0:
                        _nf.append(state.emit_frame(_nb[0], _nbf[0], f'cycle {_cn} NM'))
                    return val

                remaining_fev = max(500, int(min(30.0, _TIME_BUDGET - _elapsed()) / 0.0003))
                nm_result = minimize(_nm_obj_stream, state.best_x, method='Nelder-Mead',
                                    options={'maxfev': remaining_fev, 'xatol': 0.2,
                                             'fatol': 0.0005, 'adaptive': True})
                state.update_best(state.clamp(nm_result.x), nm_result.fun)
                for f in nm_frames:
                    yield f"data: {f}\n\n"

            # DE global search
            if not state.is_perfect() and _elapsed() < _TIME_BUDGET:
                nm_x = state.clamp(state.best_x.copy())
                de_frames = []

                def de_cb(xk, convergence=0, _df=de_frames, _cn=cycle_num):
                    val = _score_all_strokes(xk, *joint_args)
                    state.update_best(xk, val)
                    _df.append(state.emit_frame(state.best_x, state.best_fun,
                                                f'cycle {_cn} DE conv={convergence:.3f}'))
                    if state.is_perfect() or _elapsed() >= _TIME_BUDGET:
                        raise _EarlyStop()

                try:
                    de = differential_evolution(_score_all_strokes, bounds=bounds,
                                               args=joint_args, x0=nm_x, maxiter=200,
                                               popsize=20, tol=0.002, mutation=(0.5, 1.0),
                                               recombination=0.7, polish=False, disp=False,
                                               callback=de_cb)
                    state.update_best(de.x, de.fun)
                except _EarlyStop:
                    pass
                for f in de_frames:
                    yield f"data: {f}\n\n"

            # NM polish
            if not state.is_perfect() and _elapsed() < _TIME_BUDGET:
                polish_frames = []
                polish_best = [state.best_x.copy()]
                polish_best_f = [state.best_fun]

                def _polish_obj(params, _pb=polish_best, _pbf=polish_best_f, _pf=polish_frames, _cn=cycle_num):
                    params = state.clamp(params)
                    val = _score_all_strokes(params, *joint_args)
                    if val < _pbf[0]:
                        _pb[0] = params.copy()
                        _pbf[0] = val
                        _pf.append(state.emit_frame(params, val, f'cycle {_cn} polish'))
                    return val

                remaining_fev = max(200, int(min(15.0, _TIME_BUDGET - _elapsed()) / 0.0003))
                nm2 = minimize(_polish_obj, state.best_x, method='Nelder-Mead',
                              options={'maxfev': remaining_fev, 'xatol': 0.1,
                                       'fatol': 0.0005, 'adaptive': True})
                state.update_best(state.clamp(nm2.x), nm2.fun)
                for f in polish_frames:
                    yield f"data: {f}\n\n"

            improvement = score_at_cycle_start - state.best_fun
            if improvement < _STALE_THRESHOLD:
                stale_count += 1
            else:
                stale_count = 0

            current_score = float(-state.best_fun)
            if cached_score is None or current_score > cached_score:
                _save_cached_params(rel_path, char, state.best_x, current_score)

        # Determine stop reason
        if state.is_perfect():
            stop_reason = 'perfect'
        elif stale_count >= _STALE_CYCLES:
            stop_reason = 'converged'
        elif _elapsed() >= _TIME_BUDGET:
            stop_reason = 'time limit'
        else:
            stop_reason = 'done'

        # Final frame
        final_score = float(-state.best_fun)
        if affine_strokes_result and affine_strokes_result[1] > final_score:
            aff_strokes, aff_score = affine_strokes_result
            yield f"data: {state.emit_raw_frame(aff_strokes, aff_score, 'final (affine)')}\n\n"
            final_score = aff_score
        else:
            yield f"data: {state.emit_frame(state.best_x, state.best_fun, 'final')}\n\n"

        if not (affine_strokes_result and affine_strokes_result[1] > float(-state.best_fun)):
            if cached_score is None or final_score > cached_score:
                _save_cached_params(rel_path, char, state.best_x, final_score)

        yield f"data: {json.dumps({'done': True, 'score': round(final_score, 4), 'frame': state.frame_num, 'cycles': cycle_num, 'reason': stop_reason, 'elapsed': round(_elapsed(), 1)})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


@app.route('/api/skeleton/<int:font_id>', methods=['POST'])
def api_skeleton(font_id):
    """Generate strokes from font glyph via template-driven contour midpoints,
    falling back to skeletonization."""
    char = request.args.get('c')
    if not char:
        return jsonify(error="Missing ?c= parameter"), 400

    db = get_db()
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?", (font_id,)).fetchone()
    db.close()
    if not font:
        return jsonify(error="Font not found"), 404

    # Try differential-evolution auto-fit first
    result = auto_fit_strokes(font['file_path'], char, return_markers=True)
    if result and result[0]:
        strokes, markers = result
        return jsonify(strokes=strokes, markers=markers)

    # Try grid-search shape-fitting
    result = shape_fit_to_strokes(font['file_path'], char, return_markers=True)
    if result and result[0]:
        strokes, markers = result
        return jsonify(strokes=strokes, markers=markers)

    # Try template-driven approach
    result = template_to_strokes(font['file_path'], char, return_markers=True)
    if result and result[0]:
        strokes, markers = result
        return jsonify(strokes=strokes, markers=markers)

    # Fall back to skeleton pipeline
    mask = render_glyph_mask(font['file_path'], char)
    if mask is None:
        return jsonify(error="Could not render glyph"), 500

    strokes = skeleton_to_strokes(mask, min_stroke_len=5)
    if not strokes:
        return jsonify(error="No skeleton found"), 500

    strokes = apply_stroke_template(strokes, char)
    strokes = adjust_stroke_paths(strokes, char, mask)
    return jsonify(strokes=strokes)


@app.route('/api/minimal-strokes-batch/<int:font_id>', methods=['POST'])
def api_minimal_strokes_batch(font_id):
    """Generate minimal strokes (template + skeleton) for all characters and save to DB."""
    db = get_db()
    font = db.execute("SELECT * FROM fonts WHERE id = ?", (font_id,)).fetchone()
    if not font:
        db.close()
        return jsonify(error="Font not found"), 404

    # Ensure template_variant column exists
    try:
        db.execute("ALTER TABLE characters ADD COLUMN template_variant TEXT")
        db.commit()
    except:
        pass  # Column already exists

    generated = 0
    skipped = 0
    failed = 0
    variants_used = {}

    for char in CHARS:
        # Skip chars that already have stroke data (unless force=true)
        force = request.args.get('force', '').lower() == 'true'
        if not force:
            existing = db.execute(
                "SELECT id FROM characters WHERE font_id = ? AND char = ? AND strokes_raw IS NOT NULL",
                (font_id, char)
            ).fetchone()
            if existing:
                skipped += 1
                continue

        # Generate minimal strokes using template + skeleton tracing
        strokes, variant = minimal_strokes_from_skeleton(font['file_path'], char, return_variant=True)
        if not strokes:
            failed += 1
            continue

        strokes_json = json.dumps(strokes)
        if variant:
            variants_used[char] = variant

        # Upsert into database
        row = db.execute(
            "SELECT id FROM characters WHERE font_id = ? AND char = ?",
            (font_id, char)
        ).fetchone()
        if row:
            db.execute(
                "UPDATE characters SET strokes_raw = ?, template_variant = ? WHERE id = ?",
                (strokes_json, variant, row['id'])
            )
        else:
            db.execute(
                "INSERT INTO characters (font_id, char, strokes_raw, template_variant) VALUES (?, ?, ?, ?)",
                (font_id, char, strokes_json, variant)
            )
        generated += 1

    db.commit()
    db.close()
    return jsonify(ok=True, generated=generated, skipped=skipped, failed=failed, variants=variants_used)


@app.route('/api/skeleton-batch/<int:font_id>', methods=['POST'])
def api_skeleton_batch(font_id):
    """Generate skeleton strokes for all default characters of a font and save to DB."""
    db = get_db()
    font = db.execute("SELECT * FROM fonts WHERE id = ?", (font_id,)).fetchone()
    if not font:
        db.close()
        return jsonify(error="Font not found"), 404

    results = {}
    for char in CHARS:
        # Skip chars that already have stroke data
        existing = db.execute(
            "SELECT id FROM characters WHERE font_id = ? AND char = ? AND strokes_raw IS NOT NULL",
            (font_id, char)
        ).fetchone()
        if existing:
            results[char] = 'skipped'
            continue

        # Try auto-fit (DE), then grid-search, then template, then skeleton
        strokes = auto_fit_strokes(font['file_path'], char)
        if not strokes:
            strokes = shape_fit_to_strokes(font['file_path'], char)
        if not strokes:
            strokes = template_to_strokes(font['file_path'], char)
        if not strokes:
            # Fall back to skeleton pipeline
            mask = render_glyph_mask(font['file_path'], char)
            if mask is None:
                results[char] = 'no_glyph'
                continue

            strokes = skeleton_to_strokes(mask, min_stroke_len=5)
            if not strokes:
                results[char] = 'no_skeleton'
                continue
            strokes = apply_stroke_template(strokes, char)
            strokes = adjust_stroke_paths(strokes, char, mask)

        total_points = sum(len(s) for s in strokes)
        strokes_json = json.dumps(strokes)

        # Upsert
        row = db.execute(
            "SELECT id FROM characters WHERE font_id = ? AND char = ?",
            (font_id, char)
        ).fetchone()
        if row:
            db.execute(
                "UPDATE characters SET strokes_raw = ?, point_count = ? WHERE font_id = ? AND char = ?",
                (strokes_json, total_points, font_id, char)
            )
        else:
            db.execute(
                "INSERT INTO characters (font_id, char, strokes_raw, point_count) VALUES (?, ?, ?, ?)",
                (font_id, char, strokes_json, total_points)
            )
        results[char] = f'{len(strokes)} strokes'

    db.commit()
    db.close()
    generated = sum(1 for v in results.values() if 'strokes' in v)
    return jsonify(ok=True, generated=generated, results=results)


@app.route('/api/template-variants')
def api_template_variants():
    """Get available template variants for a character."""
    char = request.args.get('c')
    if not char:
        return jsonify(error="Missing ?c= parameter"), 400

    variants = NUMPAD_TEMPLATE_VARIANTS.get(char, {})
    # Return variant names and stroke counts
    result = {}
    for name, template in variants.items():
        result[name] = {
            'stroke_count': len(template),
            'template': [[str(wp) for wp in stroke] for stroke in template]
        }
    return jsonify(variants=result, char=char)


@app.route('/api/minimal-strokes/<int:font_id>')
def api_minimal_strokes(font_id):
    """Generate minimal strokes from template + skeleton analysis.

    Returns simple strokes with just key points (3-5 points per stroke).
    """
    char = request.args.get('c')
    if not char:
        return jsonify(error="Missing ?c= parameter"), 400

    variant = request.args.get('variant')

    db = get_db()
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?", (font_id,)).fetchone()
    db.close()
    if not font:
        return jsonify(error="Font not found"), 404

    font_path = resolve_font_path(font['file_path'])

    # If variant specified, use that template directly
    if variant:
        variants = NUMPAD_TEMPLATE_VARIANTS.get(char, {})
        if variant not in variants:
            return jsonify(error=f"Unknown variant '{variant}' for '{char}'", available=list(variants.keys())), 400
        template = variants[variant]
        strokes = minimal_strokes_from_skeleton(font_path, char, canvas_size=224, template=template)
        if not strokes:
            return jsonify(error=f"Could not generate strokes for '{char}' with variant '{variant}'"), 400
        return jsonify(strokes=strokes, variant=variant)

    strokes, used_variant = minimal_strokes_from_skeleton(font_path, char, canvas_size=224, return_variant=True)

    if not strokes:
        return jsonify(error=f"Could not generate minimal strokes for '{char}'"), 400

    return jsonify(strokes=strokes, variant=used_variant)


@app.route('/api/diffvg/<int:font_id>', methods=['POST'])
def api_diffvg(font_id):
    """Generate or refine strokes using DiffVG gradient-based optimization in Docker/GPU.

    If strokes are provided, refines them. If empty/missing, generates from letter template.
    Query params:
        c: character (required)
        thin: number of topology-preserving thinning iterations (default 0)
    """
    char = request.args.get('c')
    if not char:
        return jsonify(error="Missing ?c= parameter"), 400

    thin_iterations = int(request.args.get('thin', 0))

    if _diffvg_docker is None:
        return jsonify(error="DiffVG Docker not available"), 503

    db = get_db()
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?", (font_id,)).fetchone()
    db.close()
    if not font:
        return jsonify(error="Font not found"), 404

    font_path = resolve_font_path(font['file_path'])

    data = request.get_json() or {}
    strokes = data.get('strokes', [])
    valid_strokes = [s for s in strokes if len(s) >= 2]

    if valid_strokes:
        # Refine existing strokes
        clean_strokes = [[[p[0], p[1]] for p in s] for s in valid_strokes]
        source = 'refined'
    else:
        # Generate minimal strokes from template + skeleton
        clean_strokes = minimal_strokes_from_skeleton(font_path, char, canvas_size=224)
        if not clean_strokes:
            # Fall back to template_to_strokes
            clean_strokes = template_to_strokes(font_path, char, canvas_size=224)
        if not clean_strokes:
            return jsonify(error=f"No template available for '{char}'"), 400
        source = 'generated'
        # Log how minimal the strokes are
        total_pts = sum(len(s) for s in clean_strokes)
        print(f"DiffVG input: {len(clean_strokes)} strokes, {total_pts} total points")

    result = _diffvg_docker.optimize(
        font_path=font_path,
        char=char,
        initial_strokes=clean_strokes,
        canvas_size=224,
        num_iterations=500,
        stroke_width=8.0,
        thin_iterations=thin_iterations,
        timeout=300,
    )

    if 'error' in result:
        return jsonify(error=result['error']), 500

    return jsonify(
        strokes=result.get('strokes', []),
        score=result.get('score', 0),
        elapsed=result.get('elapsed', 0),
        iterations=result.get('iterations', 0),
        source=source,
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
