"""Service functions for stroke route handlers.

This module provides the business logic for stroke operations, extracted
from stroke_routes_core.py to improve code organization and testability.

Services:
    - Character data services (get, save)
    - Rendering services (glyph, preview, sample)
    - Processing services (smooth, snap, center)
    - Quality check services (font validation)
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont
from scipy.ndimage import distance_transform_edt
from skimage.morphology import thin

from stroke_flask import CHARS, STROKE_COLORS, font_repository
from stroke_rendering import (
    analyze_shape_metrics,
    check_case_mismatch,
    check_char_holes,
    check_char_shape_count,
    get_char_shape_count,
    render_char_image,
    render_glyph_mask,
    render_text_for_analysis,
)

logger = logging.getLogger(__name__)

# Quality check thresholds
MIN_SHAPE_COUNT = 10
MAX_SHAPE_COUNT = 15
MAX_WIDTH_RATIO = 0.225
EXPECTED_EXCLAMATION_SHAPES = 2


@dataclass
class FontQualityResult:
    """Result of font quality check."""
    is_connected: bool
    shape_count: int
    width_ratio: float
    issues: list[str]


@dataclass
class CharacterData:
    """Character stroke data for API responses."""
    strokes: list
    markers: list
    image_base64: str | None = None


def check_font_quality(pil_font: FreeTypeFont, font_path: str) -> FontQualityResult:
    """Check font quality based on standard criteria.

    Args:
        pil_font: Loaded PIL ImageFont object.
        font_path: Path to the font file.

    Returns:
        FontQualityResult with quality metrics and issues.
    """
    issues = []

    # Check for case mismatches
    case_mismatches = check_case_mismatch(font_path)
    if case_mismatches:
        issues.append(f"Case mismatch: {', '.join(case_mismatches)}")

    # Check shape count
    img, count = check_char_shape_count(pil_font, 'm')
    is_connected = count > MAX_SHAPE_COUNT or count < MIN_SHAPE_COUNT

    # Check exclamation point
    _, exc_count = check_char_shape_count(pil_font, '!')
    if exc_count != EXPECTED_EXCLAMATION_SHAPES:
        issues.append(f"Exclamation shapes: {exc_count} (expected {EXPECTED_EXCLAMATION_SHAPES})")

    # Check holes in closed letters
    for char in 'oadgqe':
        if not check_char_holes(pil_font, char):
            issues.append(f"Missing hole in '{char}'")

    # Check width ratio
    metrics = analyze_shape_metrics(font_path, size=60)
    width_ratio = metrics.get('width_ratio', 0) if metrics else 0
    if width_ratio > MAX_WIDTH_RATIO:
        issues.append(f"Width ratio: {width_ratio:.3f} (max {MAX_WIDTH_RATIO})")

    return FontQualityResult(
        is_connected=is_connected,
        shape_count=count,
        width_ratio=width_ratio,
        issues=issues
    )


def get_character_data(font_id: int, char: str, include_image: bool = True) -> CharacterData | None:
    """Get stroke data for a character.

    Args:
        font_id: Font database ID.
        char: Single character to get data for.
        include_image: Whether to include base64 image.

    Returns:
        CharacterData or None if font not found.
    """
    font_row = font_repository.get_font_by_id(font_id)
    if not font_row:
        return None

    char_row = font_repository.get_character(font_id, char)
    strokes = json.loads(char_row['strokes_raw']) if char_row and char_row['strokes_raw'] else []
    markers = json.loads(char_row['markers']) if char_row and char_row['markers'] else []

    image_base64 = None
    if include_image:
        try:
            img = render_char_image(font_row['file_path'], char, size=224)
            if img:
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                image_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
        except Exception as e:
            logger.warning("Failed to render character image: %s", e)

    return CharacterData(strokes=strokes, markers=markers, image_base64=image_base64)


def save_character_data(font_id: int, char: str, strokes: list, markers: list = None) -> bool:
    """Save stroke data for a character.

    Args:
        font_id: Font database ID.
        char: Single character to save data for.
        strokes: List of stroke paths.
        markers: Optional list of markers.

    Returns:
        True if saved successfully.
    """
    font_row = font_repository.get_font_by_id(font_id)
    if not font_row:
        return False

    font_repository.save_character(
        font_id=font_id,
        char=char,
        strokes_raw=json.dumps(strokes),
        markers=json.dumps(markers) if markers else None
    )
    return True


def render_glyph_image(font_path: str, char: str, size: int = 224) -> Image.Image | None:
    """Render a character glyph as an image.

    Args:
        font_path: Path to font file.
        char: Character to render.
        size: Canvas size in pixels.

    Returns:
        PIL Image or None on error.
    """
    try:
        return render_char_image(font_path, char, size=size)
    except Exception as e:
        logger.warning("Failed to render glyph: %s", e)
        return None


def render_thin_preview(font_path: str, char: str, size: int = 224) -> Image.Image | None:
    """Render thinned skeleton preview of a character.

    Args:
        font_path: Path to font file.
        char: Character to render.
        size: Canvas size in pixels.

    Returns:
        PIL Image with skeleton overlay or None on error.
    """
    try:
        mask = render_glyph_mask(font_path, char, size)
        if mask is None:
            return None

        thinned = thin(mask > 0)
        img = Image.fromarray((mask > 0).astype(np.uint8) * 200)
        img = img.convert('RGB')

        pixels = img.load()
        for y in range(thinned.shape[0]):
            for x in range(thinned.shape[1]):
                if thinned[y, x]:
                    pixels[x, y] = (255, 0, 0)

        return img
    except Exception as e:
        logger.warning("Failed to render thin preview: %s", e)
        return None


def render_stroke_preview(font_path: str, char: str, strokes: list,
                          size: int = 224, stroke_width: int = 2) -> Image.Image | None:
    """Render character with stroke overlay.

    Args:
        font_path: Path to font file.
        char: Character to render.
        strokes: List of stroke paths to overlay.
        size: Canvas size in pixels.
        stroke_width: Width of stroke lines.

    Returns:
        PIL Image with strokes overlaid or None on error.
    """
    try:
        img = render_char_image(font_path, char, size=size)
        if img is None:
            return None

        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)

        for i, stroke in enumerate(strokes):
            if len(stroke) < 2:
                continue
            color = STROKE_COLORS[i % len(STROKE_COLORS)]
            points = [(p[0], p[1]) for p in stroke]
            draw.line(points, fill=color, width=stroke_width)

        return img
    except Exception as e:
        logger.warning("Failed to render stroke preview: %s", e)
        return None


def render_font_sample(font_path: str, text: str = "Hello World!",
                       font_size: int = 48) -> Image.Image | None:
    """Render sample text with a font.

    Args:
        font_path: Path to font file.
        text: Text to render.
        font_size: Font size in points.

    Returns:
        PIL Image with rendered text or None on error.
    """
    try:
        return render_text_for_analysis(font_path, text, size=font_size)
    except Exception as e:
        logger.warning("Failed to render font sample: %s", e)
        return None


def process_strokes(strokes: list, smooth: bool = False, smooth_sigma: float = 1.5,
                    connect: bool = True, max_extension: float = 8.0) -> list:
    """Process strokes with smoothing and connection.

    Args:
        strokes: List of stroke paths (each stroke is list of [x, y] or [x, y, locked]).
        smooth: Whether to apply Gaussian smoothing.
        smooth_sigma: Smoothing sigma parameter.
        connect: Whether to extend strokes to connect.
        max_extension: Maximum extension distance.

    Returns:
        Processed strokes list.
    """
    from inksight_vectorizer import InkSightVectorizer

    # Track locked points
    locked_flags = [[len(p) >= 3 and p[2] == 1 for p in s] for s in strokes]

    # Convert to numpy arrays
    np_strokes = [np.array([[p[0], p[1]] for p in s], dtype=float) for s in strokes]

    # Apply smoothing
    if smooth:
        np_strokes = [
            InkSightVectorizer.smooth_gaussian(s, sigma=smooth_sigma * min(1.0, (len(s) - 2) / 30.0))
            if len(s) >= 3 else s
            for s in np_strokes
        ]

    # Apply connection extension
    if connect:
        np_strokes = InkSightVectorizer().extend_to_connect(np_strokes, max_extension=max_extension)

    # Convert back to lists with locked flags preserved
    result = []
    for si, stroke in enumerate(np_strokes):
        stroke_pts = []
        for pi, pt in enumerate(stroke.tolist()):
            if pi < len(locked_flags[si]) and locked_flags[si][pi]:
                stroke_pts.append([pt[0], pt[1], 1])
            else:
                stroke_pts.append([pt[0], pt[1]])
        result.append(stroke_pts)

    return result


def snap_strokes_to_boundary(strokes: list, mask: np.ndarray, max_dist: float = 15.0) -> list:
    """Snap stroke points to glyph boundary.

    Args:
        strokes: List of stroke paths.
        mask: Binary glyph mask.
        max_dist: Maximum snapping distance.

    Returns:
        Strokes with points snapped to boundary.
    """
    # Compute distance transform
    dist = distance_transform_edt(mask == 0)

    result = []
    for stroke in strokes:
        snapped_stroke = []
        for point in stroke:
            x, y = int(round(point[0])), int(round(point[1]))

            # Check bounds
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if dist[y, x] <= max_dist:
                    # Find nearest boundary point
                    best_x, best_y = x, y
                    best_dist = float('inf')

                    search_range = int(max_dist) + 1
                    for dy in range(-search_range, search_range + 1):
                        for dx in range(-search_range, search_range + 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                                if mask[ny, nx] > 0:
                                    d = (dx * dx + dy * dy) ** 0.5
                                    if d < best_dist:
                                        best_dist = d
                                        best_x, best_y = nx, ny

                    snapped_stroke.append([float(best_x), float(best_y)])
                else:
                    snapped_stroke.append([point[0], point[1]])
            else:
                snapped_stroke.append([point[0], point[1]])

        result.append(snapped_stroke)

    return result


def center_strokes_on_glyph(strokes: list, mask: np.ndarray) -> list:
    """Center strokes on glyph bounding box.

    Args:
        strokes: List of stroke paths.
        mask: Binary glyph mask.

    Returns:
        Centered strokes.
    """
    if not strokes:
        return strokes

    # Find glyph bounding box
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return strokes

    glyph_cx = (xs.min() + xs.max()) / 2
    glyph_cy = (ys.min() + ys.max()) / 2

    # Find strokes bounding box
    all_points = []
    for stroke in strokes:
        all_points.extend(stroke)

    if not all_points:
        return strokes

    stroke_xs = [p[0] for p in all_points]
    stroke_ys = [p[1] for p in all_points]
    stroke_cx = (min(stroke_xs) + max(stroke_xs)) / 2
    stroke_cy = (min(stroke_ys) + max(stroke_ys)) / 2

    # Calculate offset
    dx = glyph_cx - stroke_cx
    dy = glyph_cy - stroke_cy

    # Apply offset
    result = []
    for stroke in strokes:
        centered_stroke = [[p[0] + dx, p[1] + dy] + p[2:] for p in stroke]
        result.append(centered_stroke)

    return result


def simon_extract_strokes(font_path: str, char: str, size: int = 224,
                          bezier_error: float = 5, pts_per_segment: int = 8) -> list:
    """Extract strokes using Simon Cozens' skeleton pipeline.

    Pipeline: Lee skeletonize → graph-based path extraction → Bezier curve
    fitting → resample to polylines.

    Args:
        font_path: Path to font file.
        char: Character to extract strokes for.
        size: Canvas size in pixels.
        bezier_error: Error tolerance for Bezier curve fitting (pixels).
        pts_per_segment: Points to sample per Bezier segment.

    Returns:
        List of strokes, each stroke is a list of [x, y] points.
    """
    from beziers.path import BezierPath
    from beziers.point import Point
    from skimage.morphology import skeletonize

    mask = render_glyph_mask(font_path, char, size)
    if mask is None:
        return []

    skel = skeletonize(mask > 0, method='lee') > 0
    if not skel.any():
        return []

    edges = _skeleton_to_graph_edges(skel)
    if not edges:
        return []

    strokes = []
    for edge in edges:
        # Skip tiny junction-to-junction fragments
        if len(edge) < 5:
            continue

        # Convert (row, col) to (x, y) Points
        points = [Point(float(c), float(r)) for r, c in edge]

        try:
            bp = BezierPath.fromPoints(points, error=bezier_error)
            segments = bp.asSegments()

            polyline = []
            for seg in segments:
                for i in range(pts_per_segment):
                    t = i / pts_per_segment
                    pt = seg.pointAtTime(t)
                    polyline.append([round(pt.x, 1), round(pt.y, 1)])
            # Final point of last segment
            pt = segments[-1].pointAtTime(1.0)
            polyline.append([round(pt.x, 1), round(pt.y, 1)])

            # Deduplicate consecutive identical points
            deduped = [polyline[0]]
            for p in polyline[1:]:
                if p != deduped[-1]:
                    deduped.append(p)

            if len(deduped) >= 2:
                strokes.append(deduped)
        except Exception:
            # Fallback: subsample raw skeleton points
            step = max(1, len(edge) // 20)
            raw = [[float(c), float(r)] for r, c in edge[::step]]
            if edge[-1] != edge[-1 * step]:
                raw.append([float(edge[-1][1]), float(edge[-1][0])])
            if len(raw) >= 2:
                strokes.append(raw)

    return strokes


def _skeleton_to_graph_edges(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """Extract graph edges from a binary skeleton image.

    Builds a graph where nodes are endpoints (degree 1) and junctions
    (degree > 2). Each edge is a path of pixels connecting two nodes,
    traced through degree-2 pixels. This produces semantically meaningful
    strokes rather than fragments.

    For skeletons with no junctions (simple curves), traces endpoint
    to endpoint.

    Args:
        skel: Binary skeleton image (True = skeleton pixel).

    Returns:
        List of edges, each edge is a list of (row, col) tuples.
    """
    h, w = skel.shape

    # Compute neighbor count for each skeleton pixel
    neighbor_count = np.zeros((h, w), dtype=int)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            # Manual shift to avoid np.roll wrap-around artifacts
            src_r = slice(max(0, -dy), h + min(0, -dy))
            dst_r = slice(max(0, dy), h + min(0, dy))
            src_c = slice(max(0, -dx), w + min(0, -dx))
            dst_c = slice(max(0, dx), w + min(0, dx))
            neighbor_count[dst_r, dst_c] += skel[src_r, src_c].astype(int)
    neighbor_count *= skel

    # Classify pixels
    node_set = set()  # endpoints and junctions
    endpoint_list = []
    junction_list = []

    rows, cols = np.where(skel)
    for r, c in zip(rows, cols):
        nc = neighbor_count[r, c]
        if nc == 1:
            endpoint_list.append((r, c))
            node_set.add((r, c))
        elif nc > 2:
            junction_list.append((r, c))
            node_set.add((r, c))

    def get_skel_neighbors(r, c):
        nbrs = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                nr, nc_ = r + dy, c + dx
                if 0 <= nr < h and 0 <= nc_ < w and skel[nr, nc_]:
                    nbrs.append((nr, nc_))
        return nbrs

    def trace_edge(start, first_step):
        """Trace from start through first_step until reaching another node."""
        path = [start, first_step]
        prev = start
        cur = first_step
        while cur not in node_set or cur == start:
            nbrs = get_skel_neighbors(cur[0], cur[1])
            # Continue to a neighbor that isn't where we came from
            next_px = None
            for n in nbrs:
                if n != prev:
                    next_px = n
                    break
            if next_px is None:
                break  # dead end (shouldn't happen in clean skeleton)
            prev = cur
            cur = next_px
            path.append(cur)
            if cur in node_set:
                break
        return path

    edges = []
    visited_edges = set()  # Track visited (node, first_step) to avoid duplicates

    # Trace edges from all nodes
    start_points = endpoint_list + junction_list
    for node in start_points:
        nbrs = get_skel_neighbors(node[0], node[1])
        for nbr in nbrs:
            edge_key = (node, nbr)
            if edge_key in visited_edges:
                continue
            path = trace_edge(node, nbr)
            if len(path) >= 2:
                edges.append(path)
                # Mark both directions as visited
                visited_edges.add((node, nbr))
                if len(path) >= 2:
                    visited_edges.add((path[-1], path[-2]))

    # Handle components with no nodes (pure loops — all degree-2)
    if not node_set:
        visited = np.zeros_like(skel, dtype=bool)
        for r, c in zip(rows, cols):
            if visited[r, c]:
                continue
            path = [(r, c)]
            visited[r, c] = True
            cur_r, cur_c = r, c
            while True:
                nbrs = [(nr, nc_) for nr, nc_ in get_skel_neighbors(cur_r, cur_c)
                        if not visited[nr, nc_]]
                if not nbrs:
                    break
                cur_r, cur_c = nbrs[0]
                visited[cur_r, cur_c] = True
                path.append((cur_r, cur_c))
            if len(path) >= 3:
                edges.append(path)

    return edges
