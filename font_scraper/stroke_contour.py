"""Font contour extraction and processing.

This module contains functions for extracting glyph contours from fonts
and converting them to strokes.
"""

import numpy as np
from typing import List, Tuple, Optional
from PIL import ImageFont
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen


def flatten_bezier_quad(p0: Tuple, p1: Tuple, p2: Tuple, steps: int = 15) -> List[Tuple]:
    """Flatten a quadratic Bezier curve to a point sequence."""
    pts = []
    for i in range(1, steps + 1):
        t = i / steps
        x = (1-t)**2*p0[0] + 2*(1-t)*t*p1[0] + t**2*p2[0]
        y = (1-t)**2*p0[1] + 2*(1-t)*t*p1[1] + t**2*p2[1]
        pts.append((x, y))
    return pts


def flatten_bezier_cubic(p0: Tuple, p1: Tuple, p2: Tuple, p3: Tuple, steps: int = 20) -> List[Tuple]:
    """Flatten a cubic Bezier curve to a point sequence."""
    pts = []
    for i in range(1, steps + 1):
        t = i / steps
        x = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
        y = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
        pts.append((x, y))
    return pts


def extract_contours(font_path: str, char: str) -> Tuple[Optional[List], 'TTFont']:
    """Extract glyph contours from a font using fontTools RecordingPen.

    Returns:
        Tuple of (contours, TTFont) where contours is a list of point lists
    """
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
            pts = flatten_bezier_cubic(current_pos, args[0], args[1], args[2])
            current.extend(pts)
            current_pos = args[2]
        elif op == 'qCurveTo':
            if len(args) == 2:
                pts = flatten_bezier_quad(current_pos, args[0], args[1])
                current.extend(pts)
                current_pos = args[1]
            else:
                for i in range(len(args) - 1):
                    if i < len(args) - 2:
                        mid = ((args[i][0]+args[i+1][0])/2, (args[i][1]+args[i+1][1])/2)
                        pts = flatten_bezier_quad(current_pos, args[i], mid)
                        current.extend(pts)
                        current_pos = mid
                    else:
                        pts = flatten_bezier_quad(current_pos, args[i], args[i+1])
                        current.extend(pts)
                        current_pos = args[i+1]
        elif op in ('closePath', 'endPath'):
            if current:
                contours.append(current)
                current = []
    if current:
        contours.append(current)
    return contours, tt


def font_to_pixel_transform(tt: 'TTFont', font_path: str, char: str,
                            canvas_size: int = 224):
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


def get_pixel_contours(font_path: str, char: str, canvas_size: int = 224,
                       resolve_font_path_fn=None) -> List[List[Tuple]]:
    """Extract glyph contours as pixel-space polylines.

    Returns list of polylines, each a list of (x, y) tuples.
    """
    if resolve_font_path_fn:
        font_path = resolve_font_path_fn(font_path)
    contours, tt = extract_contours(font_path, char)
    if not contours:
        return []

    transform = font_to_pixel_transform(tt, font_path, char, canvas_size)

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


def contour_segments(pixel_contours: List[List[Tuple]]) -> List[Tuple]:
    """Build flat list of line segments from pixel contours.

    Returns list of ((x0,y0), (x1,y1)) tuples.
    """
    segments = []
    for contour in pixel_contours:
        for i in range(len(contour) - 1):
            segments.append((contour[i], contour[i + 1]))
    return segments


def contour_to_strokes(font_path: str, char: str, canvas_size: int = 224,
                       resolve_font_path_fn=None) -> Optional[List[List[List[float]]]]:
    """Extract strokes by splitting the outer font contour at top/bottom extremal points.

    Returns two strokes as [[[x,y], ...], [[x,y], ...]] tracing the left and right
    sides of the letter outline.
    """
    if resolve_font_path_fn:
        font_path = resolve_font_path_fn(font_path)
    contours, tt = extract_contours(font_path, char)
    if not contours:
        return None

    transform = font_to_pixel_transform(tt, font_path, char, canvas_size)

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


def contour_detect_markers(font_path: str, char: str, canvas_size: int = 224,
                           resolve_font_path_fn=None) -> List[dict]:
    """Detect termination markers from contour split points (top/bottom of letter)."""
    if resolve_font_path_fn:
        font_path = resolve_font_path_fn(font_path)
    contours, tt = extract_contours(font_path, char)
    if not contours:
        return []

    transform = font_to_pixel_transform(tt, font_path, char, canvas_size)

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


def ray_segment_intersection(origin: Tuple, direction: Tuple,
                             seg_a: Tuple, seg_b: Tuple) -> Optional[float]:
    """Find intersection parameter t of ray with line segment.

    Ray: origin + t * direction (t >= 0)
    Segment: seg_a to seg_b

    Returns t (distance along ray) or None if no intersection.
    """
    ox, oy = origin
    dx, dy = direction
    ax, ay = seg_a
    bx, by = seg_b

    sx, sy = bx - ax, by - ay

    denom = dx * sy - dy * sx
    if abs(denom) < 1e-10:
        return None  # Parallel

    t = ((ax - ox) * sy - (ay - oy) * sx) / denom
    s = ((ax - ox) * dy - (ay - oy) * dx) / denom

    if t > 0.5 and 0 <= s <= 1:
        return t
    return None


def find_cross_section_midpoint(point: Tuple, tangent: Tuple,
                                segments: List[Tuple], mask: np.ndarray) -> Optional[Tuple]:
    """Find the stroke center at a guide path point via cross-section ray casting.

    Cast perpendicular rays in both directions, find nearest contour intersection
    on each side, return midpoint.
    """
    perp = (-tangent[1], tangent[0])
    h, w = mask.shape

    # Cast ray in positive perpendicular direction
    best_t_pos = None
    for seg_a, seg_b in segments:
        t = ray_segment_intersection(point, perp, seg_a, seg_b)
        if t is not None and (best_t_pos is None or t < best_t_pos):
            best_t_pos = t

    # Cast ray in negative perpendicular direction
    neg_perp = (perp[0] * -1, perp[1] * -1)
    best_t_neg = None
    for seg_a, seg_b in segments:
        t = ray_segment_intersection(point, neg_perp, seg_a, seg_b)
        if t is not None and (best_t_neg is None or t < best_t_neg):
            best_t_neg = t

    if best_t_pos is not None and best_t_neg is not None:
        pos_pt = (point[0] + perp[0] * best_t_pos,
                  point[1] + perp[1] * best_t_pos)
        neg_pt = (point[0] + neg_perp[0] * best_t_neg,
                  point[1] + neg_perp[1] * best_t_neg)
        mx = (pos_pt[0] + neg_pt[0]) / 2
        my = (pos_pt[1] + neg_pt[1]) / 2
        mix = int(round(min(max(mx, 0), w - 1)))
        miy = int(round(min(max(my, 0), h - 1)))
        if mask[miy, mix]:
            return (mx, my)

    ix = int(round(min(max(point[0], 0), w - 1)))
    iy = int(round(min(max(point[1], 0), h - 1)))
    if mask[iy, ix]:
        return point
    return None


# Aliases for backwards compatibility
_flatten_bezier_quad = flatten_bezier_quad
_flatten_bezier_cubic = flatten_bezier_cubic
_extract_contours = extract_contours
_font_to_pixel_transform = font_to_pixel_transform
_get_pixel_contours = get_pixel_contours
_contour_segments = contour_segments
_ray_segment_intersection = ray_segment_intersection
_find_cross_section_midpoint = find_cross_section_midpoint
