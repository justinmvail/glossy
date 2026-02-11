"""Core stroke processing functions for font glyph analysis.

This module provides the primary API for extracting stroke representations
from font glyphs. It integrates skeleton analysis, stroke tracing, and
template-based optimization to produce clean stroke paths suitable for
rendering or further processing.

The module serves as the central orchestration layer that combines:
    - Skeleton extraction via SkeletonAnalyzer
    - Stroke path tracing and merging
    - Template-based stroke optimization (MinimalStrokePipeline)
    - Affine transformation optimization (auto_fit)

Key Functions:
    - skel_markers: Detect structural markers (endpoints, junctions) in glyphs
    - skel_strokes: Extract raw stroke paths from skeleton analysis
    - min_strokes: Generate optimized strokes using template matching
    - auto_fit: Apply affine optimization for best stroke fit

Example:
    >>> from stroke_core import min_strokes, skel_strokes
    >>> # Get minimal strokes for letter 'A' in a font
    >>> strokes = min_strokes('/path/to/font.ttf', 'A', canvas_size=224)
    >>> # Or get raw skeleton-based strokes
    >>> mask = render_glyph_mask('/path/to/font.ttf', 'A', 224)
    >>> raw_strokes = skel_strokes(mask)

Notes:
    This module depends on several specialized submodules:
    - stroke_flask: Font path resolution
    - stroke_lib.analysis.skeleton: Skeleton extraction and analysis
    - stroke_merge: Stroke path merging algorithms
    - stroke_pipeline: Template-based stroke generation
    - stroke_rendering: Glyph mask rendering
    - stroke_skeleton: Path tracing utilities
    - stroke_templates: Template definitions for characters
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

# Module logger
logger = logging.getLogger(__name__)

from stroke_flask import resolve_font_path
from stroke_lib.analysis.skeleton import SkeletonAnalyzer
from stroke_merge import MergePipeline
from stroke_rendering import render_glyph_mask
from stroke_scoring import quick_stroke_score
from stroke_skeleton import (
    SKELETON_MERGE_DISTANCE,
    find_skeleton_segments,
    generate_straight_line,
    resample_path,
    trace_segment,
    trace_to_region,
)
from stroke_templates import NUMPAD_TEMPLATE_VARIANTS
from stroke_utils import point_in_region


def _analyze_skeleton_legacy(mask: np.ndarray) -> dict[str, Any] | None:
    """Analyze skeleton and return info dict compatible with legacy format.

    Performs skeleton extraction on a binary mask and returns structural
    information including the skeleton pixel set, adjacency graph, and
    junction/endpoint classifications.

    Args:
        mask: A 2D numpy array representing a binary glyph mask where
            non-zero values indicate foreground pixels.

    Returns:
        A dictionary containing skeleton analysis results with the following keys:
            - 'skel_set': Set of (x, y) tuples representing skeleton pixels
            - 'adj': Dictionary mapping each skeleton pixel to its neighbors
            - 'junction_pixels': Set of pixels with degree >= 3
            - 'junction_clusters': List of clustered junction pixel groups
            - 'assigned': Mapping of pixels to their assigned clusters
            - 'endpoints': Set of pixels with degree == 1

        Returns None if skeleton analysis fails (empty mask, no skeleton).

    Notes:
        Uses a merge_distance of 12 pixels for clustering nearby junction
        points. This helps consolidate complex junction regions into single
        logical junction points.
    """
    info = SkeletonAnalyzer(merge_distance=SKELETON_MERGE_DISTANCE).analyze(mask)
    if not info:
        return None
    from collections import defaultdict
    adj = defaultdict(list)
    for p, n in info.adj.items():
        adj[p] = list(n)
    return {
        'skel_set': info.skel_set,
        'adj': adj,
        'junction_pixels': info.junction_pixels,
        'junction_clusters': info.junction_clusters,
        'assigned': info.assigned,
        'endpoints': info.endpoints,
    }


def skel_markers(mask: np.ndarray) -> list[dict[str, Any]]:
    """Detect skeleton markers from a binary glyph mask.

    Extracts structural markers from the skeleton of a glyph, identifying
    key topological features like endpoints, junctions, and curvature points.

    Args:
        mask: A 2D numpy array representing a binary glyph mask where
            non-zero values indicate foreground pixels.

    Returns:
        A list of marker dictionaries, where each dictionary contains:
            - 'x': X coordinate of the marker
            - 'y': Y coordinate of the marker
            - 'type': Marker type string (e.g., 'endpoint', 'junction')

        Returns an empty list if no markers are detected.

    Notes:
        This function is useful for visualizing glyph structure or for
        providing anchor points in interactive stroke editing interfaces.
    """
    markers = SkeletonAnalyzer(merge_distance=SKELETON_MERGE_DISTANCE).detect_markers(mask)
    return [m.to_dict() for m in markers]


def _trace_skeleton_path(start: tuple, neighbor: tuple, adj: dict,
                          stop_pixels: set, visited_edges: set) -> list | None:
    """Trace a path through the skeleton from start to a stop pixel.

    Walks along skeleton edges, marking them as visited, until reaching
    an endpoint or junction (stop pixel) or running out of unvisited edges.

    Args:
        start: Starting pixel (x, y).
        neighbor: First neighbor to walk to.
        adj: Adjacency dict mapping pixels to their neighbors.
        stop_pixels: Set of pixels where path should terminate.
        visited_edges: Set of already-visited edges (modified in place).

    Returns:
        List of pixels forming the path, or None if edge already visited.
    """
    edge = (min(start, neighbor), max(start, neighbor))
    if edge in visited_edges:
        return None

    visited_edges.add(edge)
    path = [start, neighbor]
    cur, prev = neighbor, start

    while True:
        if cur in stop_pixels and len(path) > 2:
            break
        # Find unvisited neighbors
        candidates = []
        for n in adj.get(cur, []):
            if n != prev:
                candidate_edge = (min(cur, n), max(cur, n))
                if candidate_edge not in visited_edges:
                    candidates.append((n, candidate_edge))
        if not candidates:
            break
        nxt, next_edge = candidates[0]
        visited_edges.add(next_edge)
        path.append(nxt)
        prev, cur = cur, nxt

    return path


def _trace_all_paths(adj: dict, endpoints: set, junction_pixels: set) -> list:
    """Trace all paths in the skeleton graph.

    Starts from endpoints first, then junctions, to ensure complete coverage.

    Args:
        adj: Adjacency dict mapping pixels to their neighbors.
        endpoints: Set of endpoint pixels (degree 1).
        junction_pixels: Set of junction pixels (degree >= 3).

    Returns:
        List of paths, where each path is a list of (x, y) tuples.
    """
    stop_pixels = endpoints | junction_pixels
    visited_edges = set()
    paths = []

    # Trace from endpoints first
    for start_pt in sorted(endpoints):
        for neighbor in adj.get(start_pt, []):
            path = _trace_skeleton_path(start_pt, neighbor, adj, stop_pixels, visited_edges)
            if path and len(path) >= 2:
                paths.append(path)

    # Then from junctions to catch any remaining edges
    for start_pt in sorted(junction_pixels):
        for neighbor in adj.get(start_pt, []):
            path = _trace_skeleton_path(start_pt, neighbor, adj, stop_pixels, visited_edges)
            if path and len(path) >= 2:
                paths.append(path)

    return paths


def skel_strokes(mask: np.ndarray, min_len: int = 5,
                 min_stroke_len: int | None = None) -> list[list[list[float]]]:
    """Extract stroke paths from skeleton analysis with merging.

    Performs complete stroke extraction from a glyph mask by:
    1. Analyzing the skeleton structure
    2. Tracing paths between endpoints and junctions
    3. Applying multiple merge passes to consolidate fragments
    4. Filtering by minimum length

    Args:
        mask: A 2D numpy array representing a binary glyph mask where
            non-zero values indicate foreground pixels.
        min_len: Minimum number of points required for a valid stroke.
            Strokes shorter than this are filtered out. Defaults to 5.
        min_stroke_len: Deprecated alias for min_len. If provided, overrides
            min_len for backward compatibility.

    Returns:
        A list of strokes, where each stroke is a list of [x, y] coordinate
        pairs as floats. Example: [[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], ...]

        Returns an empty list if skeleton analysis fails or no valid strokes
        are found.

    Notes:
        The merging pipeline includes:
        - T-junction merging for strokes meeting at perpendicular intersections
        - General run merge pass for nearby endpoints
        - Convergence stub absorption for short branches at junctions
        - Junction stub cleanup for isolated fragments
        - Proximity-based stub absorption
        - Orphan stub removal for disconnected short strokes

        Strokes are traced starting from endpoints first, then from junction
        pixels, ensuring complete coverage of the skeleton graph.
    """
    if min_stroke_len is not None:
        min_len = min_stroke_len

    info = _analyze_skeleton_legacy(mask)
    if not info:
        return []

    adj = info['adj']
    endpoints = set(info['endpoints'])
    junction_pixels = set(info['junction_pixels'])
    junction_clusters = info['junction_clusters']

    # Trace all paths through the skeleton
    raw_paths = _trace_all_paths(adj, endpoints, junction_pixels)

    # Filter by minimum length and apply merge pipeline
    strokes = [s for s in raw_paths if len(s) >= min_len]
    assigned_clusters = [set(c) for c in junction_clusters]

    # Use MergePipeline for all merge operations
    pipeline = MergePipeline.create_default()
    strokes = pipeline.run(strokes, junction_clusters, assigned_clusters)

    return [[[float(x), float(y)] for x, y in s] for s in strokes]


def _find_closest_endpoint_pair(strokes: list[list[list[float]]]) -> tuple[float, int, int, bool]:
    """Find the closest pair of endpoints between different strokes.

    Uses cKDTree for O(n log n) performance when stroke count is large,
    falling back to simple O(n²) search for small counts where tree
    construction overhead would dominate.

    Args:
        strokes: List of strokes, each a list of [x, y] coordinate pairs.

    Returns:
        Tuple of (distance_squared, stroke_i, stroke_j, reverse_j):
        - distance_squared: Squared distance between closest endpoints
        - stroke_i: Index of first stroke (lower index)
        - stroke_j: Index of second stroke (higher index)
        - reverse_j: True if stroke_j should be reversed when merging
        Returns (inf, -1, -1, False) if no valid pair found.
    """
    n = len(strokes)
    if n < 2:
        return float('inf'), -1, -1, False

    # For small counts, simple O(n²) is faster than tree construction
    if n <= 6:
        best_dist, best_i, best_j, reverse_j = float('inf'), -1, -1, False
        for i in range(n):
            for j in range(i + 1, n):
                for end_i, end_j, rev in [(strokes[i][-1], strokes[j][0], False),
                                          (strokes[i][-1], strokes[j][-1], True),
                                          (strokes[i][0], strokes[j][0], False),
                                          (strokes[i][0], strokes[j][-1], True)]:
                    d = (end_i[0] - end_j[0])**2 + (end_i[1] - end_j[1])**2
                    if d < best_dist:
                        best_dist, best_i, best_j, reverse_j = d, i, j, rev
        return best_dist, best_i, best_j, reverse_j

    # Build array of endpoints: each stroke contributes 2 points
    # Format: [x, y] with metadata tracked separately
    endpoints = []
    metadata = []  # (stroke_idx, is_end)
    for i, s in enumerate(strokes):
        endpoints.append(s[0])   # start
        metadata.append((i, False))
        endpoints.append(s[-1])  # end
        metadata.append((i, True))

    points = np.array(endpoints)
    tree = cKDTree(points)

    # For each endpoint, find nearest from a different stroke
    best_dist, best_i, best_j, reverse_j = float('inf'), -1, -1, False

    for idx, (stroke_idx, is_end) in enumerate(metadata):
        # Query enough neighbors to find one from a different stroke
        dists, indices = tree.query(points[idx], k=min(4, len(points)))

        for d_sq, neighbor_idx in zip(dists**2, indices):
            if neighbor_idx == idx:
                continue
            neighbor_stroke, neighbor_is_end = metadata[neighbor_idx]
            if neighbor_stroke == stroke_idx:
                continue

            if d_sq < best_dist:
                # Ensure i < j for consistent ordering
                si, sj = stroke_idx, neighbor_stroke
                if si > sj:
                    si, sj = sj, si
                    is_end, neighbor_is_end = neighbor_is_end, is_end

                # Determine if j should be reversed based on which endpoints connect
                # We connect: stroke_i[is_end] to stroke_j[neighbor_is_end]
                # If connecting i's end to j's end, j needs reversing
                # If connecting i's end to j's start, no reverse needed
                rev = neighbor_is_end if stroke_idx < neighbor_stroke else is_end

                best_dist, best_i, best_j, reverse_j = d_sq, si, sj, rev
            break  # Found closest from different stroke

    return best_dist, best_i, best_j, reverse_j


def _merge_to_expected_count(strokes: list[list[list[float]]],
                             char: str) -> list[list[list[float]]]:
    """Merge strokes to match expected count from character template.

    Applies greedy endpoint-based merging to reduce the number of strokes
    to match the expected count defined in the character's template variants.

    Args:
        strokes: A list of strokes, where each stroke is a list of [x, y]
            coordinate pairs.
        char: The character being processed (e.g., 'A', '3'). Used to look
            up expected stroke counts from NUMPAD_TEMPLATE_VARIANTS.

    Returns:
        The merged list of strokes. If no template exists for the character,
        or if the current stroke count already matches a template variant,
        the original strokes are returned unchanged.

    Notes:
        The merging algorithm:
        1. Looks up all template variants for the character
        2. If current count matches any variant, returns unchanged
        3. Otherwise, targets the minimum expected count
        4. Greedily merges strokes by finding the closest endpoint pairs
        5. Considers all four endpoint combinations (start-start, start-end,
           end-start, end-end) when computing distances
        6. Reverses stroke direction as needed when merging

        Endpoint combination strategy explained:
            Given strokes A (points a1→a2) and B (points b1→b2), we check
            distances for all four ways to connect them:
                - a2 to b1: A followed by B (no reversal)
                - a2 to b2: A followed by reversed B
                - a1 to b1: reversed A followed by B
                - a1 to b2: reversed A followed by reversed B
            The pair with minimum distance wins. This ensures we find the
            optimal connection regardless of how strokes were originally
            traced (skeleton tracing can produce strokes in either direction).
    """
    variants = NUMPAD_TEMPLATE_VARIANTS.get(char, {})
    if not variants or not strokes:
        return strokes
    expected_counts = [len(t) for t in variants.values()]
    # If current stroke count already matches any template variant, don't merge
    if len(strokes) in expected_counts:
        return strokes
    expected = min(expected_counts)
    if len(strokes) <= expected:
        return strokes
    # Greedy merge: find closest endpoints and merge
    while len(strokes) > expected and len(strokes) > 1:
        best_dist, best_i, best_j, reverse_j = _find_closest_endpoint_pair(strokes)
        if best_i < 0:
            break
        # Merge j into i
        s_j = strokes[best_j][::-1] if reverse_j else strokes[best_j]
        strokes[best_i] = strokes[best_i] + s_j
        strokes.pop(best_j)
    return strokes


def min_strokes(fp: str, c: str, cs: int = 224, tpl: list | None = None,
                ret_var: bool = False) -> list[list[list[float]]] | tuple[list, str] | None:
    """Generate minimal strokes using the template-based pipeline.

    Produces optimized stroke paths for a character by evaluating multiple
    template variants and selecting the best match based on scoring.

    Args:
        fp: Font path string. Can be an absolute path or a font name that
            will be resolved via resolve_font_path.
        c: The character to generate strokes for (single character string).
        cs: Canvas size in pixels for rendering the glyph. Larger values
            provide more detail but increase processing time. Defaults to 224.
        tpl: Optional custom template string to use instead of evaluating
            all variants. When provided, only this template is traced.
        ret_var: If True, returns a tuple of (strokes, variant_name) instead
            of just strokes. Useful for debugging which template was selected.
            Defaults to False.

    Returns:
        If ret_var is False:
            A list of strokes, where each stroke is a list of [x, y]
            coordinate pairs as floats.

        If ret_var is True:
            A tuple of (strokes, variant_name) where variant_name is
            'custom' for custom templates or the name of the best-matching
            template variant.

    Notes:
        The pipeline stages are:
        1. Render glyph mask at specified canvas size
        2. Analyze skeleton structure
        3. Either trace custom template or evaluate all template variants
        4. Score each result based on coverage and path quality
        5. Return the best-scoring variant

        When no custom template is provided, the function evaluates all
        variants defined in NUMPAD_TEMPLATE_VARIANTS for the character
        and returns the one with the best score.
    """
    # Lazy import to avoid circular dependency with stroke_pipeline
    from stroke_pipeline import MinimalStrokePipeline

    pipe = MinimalStrokePipeline(
        fp, c, cs,
        resolve_font_path_fn=resolve_font_path,
        render_glyph_mask_fn=render_glyph_mask,
        analyze_skeleton_fn=_analyze_skeleton_legacy,
        find_skeleton_segments_fn=find_skeleton_segments,
        point_in_region_fn=point_in_region,
        trace_segment_fn=trace_segment,
        trace_to_region_fn=trace_to_region,
        generate_straight_line_fn=generate_straight_line,
        resample_path_fn=resample_path,
        skeleton_to_strokes_fn=skel_strokes,
        apply_stroke_template_fn=_merge_to_expected_count,
        adjust_stroke_paths_fn=lambda st, c, m: st,  # No-op for now
        quick_stroke_score_fn=quick_stroke_score,
    )

    if tpl:
        st = pipe.run(tpl, trace_paths=True)
        logger.debug("min_strokes: char='%s' custom template -> %d strokes", c, len(st) if st else 0)
        return (st, 'custom') if ret_var else st

    result = pipe.evaluate_all_variants()
    logger.debug("min_strokes: char='%s' best variant='%s' score=%.3f strokes=%d",
                 c, result.variant_name, result.score, len(result.strokes) if result.strokes else 0)
    if ret_var:
        return result.strokes, result.variant_name
    return result.strokes


def auto_fit(fp: str, c: str, cs: int = 224,
             ret_mark: bool = False) -> list[list[list[float]]] | tuple[list, list] | None:
    """Auto-fit strokes using affine transformation optimization.

    Generates strokes and optimizes their positions using affine
    transformations (translation, rotation, scaling, shear) to best
    fit the glyph mask.

    Args:
        fp: Font path string. Can be an absolute path or a font name that
            will be resolved via resolve_font_path.
        c: The character to generate strokes for (single character string).
        cs: Canvas size in pixels for rendering the glyph. Defaults to 224.
        ret_mark: If True, returns a tuple of (strokes, markers) where
            markers are the start/stop points of each stroke. Useful for
            visualization or interactive editing. Defaults to False.

    Returns:
        If ret_mark is False:
            A list of strokes, where each stroke is a list of [x, y]
            coordinate pairs as floats. Returns None if optimization fails.

        If ret_mark is True:
            A tuple of (strokes, markers) where:
            - strokes: List of stroke paths (or None if failed)
            - markers: List of marker dictionaries with keys:
                - 'x': X coordinate
                - 'y': Y coordinate
                - 'type': Either 'start' or 'stop'
                - 'stroke_id': Index of the stroke this marker belongs to

    Notes:
        The affine optimization adjusts six parameters:
        - Translation (tx, ty)
        - Scale (sx, sy)
        - Rotation (theta)
        - Shear

        This produces better-fitted strokes than raw skeleton extraction,
        especially for fonts with unusual proportions or stylization.

        Strokes with fewer than 2 points are filtered from the output.
    """
    from stroke_affine import optimize_affine

    result = optimize_affine(fp, c, cs)
    if result is None:
        logger.debug("auto_fit: char='%s' optimization returned None", c)
        return (None, []) if ret_mark else None

    strokes, _score, _, _ = result
    if not strokes:
        logger.debug("auto_fit: char='%s' no strokes from optimization", c)
        return (None, []) if ret_mark else None

    # Convert to list format
    stroke_list = [[[float(x), float(y)] for x, y in s] for s in strokes if len(s) >= 2]
    logger.debug("auto_fit: char='%s' -> %d strokes, score=%.3f", c, len(stroke_list), _score)

    if ret_mark:
        markers = []
        for i, s in enumerate(stroke_list):
            if len(s) >= 2:
                markers.append({'x': s[0][0], 'y': s[0][1], 'type': 'start', 'stroke_id': i})
                markers.append({'x': s[-1][0], 'y': s[-1][1], 'type': 'stop', 'stroke_id': i})
        return stroke_list, markers

    return stroke_list
