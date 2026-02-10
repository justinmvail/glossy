"""Font contour extraction and processing.

This module provides functions for extracting glyph outlines from font files
and converting them to stroke representations. It handles the conversion from
font-space Bezier curves to pixel-space polylines.

Algorithm Overview:
    1. Contour Extraction:
        - Uses fontTools RecordingPen to capture drawing operations from the font
        - Flattens quadratic and cubic Bezier curves into discrete point sequences
        - Handles TrueType (quadratic) and PostScript (cubic) outline formats

    2. Coordinate Transformation:
        - Converts from font units (typically 1000 or 2048 units per em) to pixels
        - Centers the glyph on the canvas with appropriate margins
        - Flips Y-axis (font coordinates have Y pointing up, pixels have Y down)

    3. Stroke Generation:
        - Identifies the outer contour (largest by point count)
        - Splits at extremal points (topmost/bottommost) to create stroke pairs
        - Supports cross-section ray casting for finding stroke centerlines

Key Concepts:
    - Contour: A closed path defining part of a glyph outline (outer boundary
      or inner hole like in 'O')
    - Pixel contour: A contour transformed to pixel coordinates
    - Segment: A line segment between two adjacent contour points
    - Cross-section: A perpendicular slice through the glyph at a given point

Typical usage:
    # Get pixel-space contours
    contours = get_pixel_contours(font_path, 'A', canvas_size=224)

    # Convert to strokes (left and right sides of letter)
    strokes = contour_to_strokes(font_path, 'A', canvas_size=224)
"""


import numpy as np
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont
from PIL import ImageFont


def flatten_bezier_quad(p0: tuple, p1: tuple, p2: tuple, steps: int = 15) -> list[tuple]:
    """Flatten a quadratic Bezier curve into a sequence of discrete points.

    Quadratic Bezier curves are defined by three control points and are commonly
    used in TrueType fonts. The curve passes through p0 and p2, with p1
    influencing the curve's shape but not lying on it.

    The parametric formula is:
        B(t) = (1-t)^2 * p0 + 2*(1-t)*t * p1 + t^2 * p2, for t in [0, 1]

    Args:
        p0: Starting point (x, y) of the curve. The curve passes through this point.
        p1: Control point (x, y) that defines the curve's tangent direction at
            the endpoints. The curve does not pass through this point.
        p2: Ending point (x, y) of the curve. The curve passes through this point.
        steps: Number of line segments to approximate the curve. Higher values
            produce smoother approximations but more points. Defaults to 15.

    Returns:
        List of (x, y) tuples representing points along the curve, excluding p0
        (which is assumed to already be in the path) but including p2.
        Returns `steps` points evenly spaced in parameter t from 1/steps to 1.

    Note:
        The first point p0 is not included in the output because it's expected
        to already exist as the current path position. This avoids duplicate
        points when chaining curve segments.
    """
    pts = []
    for i in range(1, steps + 1):
        t = i / steps
        x = (1-t)**2*p0[0] + 2*(1-t)*t*p1[0] + t**2*p2[0]
        y = (1-t)**2*p0[1] + 2*(1-t)*t*p1[1] + t**2*p2[1]
        pts.append((x, y))
    return pts


def flatten_bezier_cubic(p0: tuple, p1: tuple, p2: tuple, p3: tuple, steps: int = 20) -> list[tuple]:
    """Flatten a cubic Bezier curve into a sequence of discrete points.

    Cubic Bezier curves are defined by four control points and are commonly
    used in PostScript/CFF fonts. The curve passes through p0 and p3, with
    p1 and p2 controlling the shape.

    The parametric formula is:
        B(t) = (1-t)^3 * p0 + 3*(1-t)^2*t * p1 + 3*(1-t)*t^2 * p2 + t^3 * p3

    Args:
        p0: Starting point (x, y) of the curve.
        p1: First control point (x, y), influences curve near p0.
        p2: Second control point (x, y), influences curve near p3.
        p3: Ending point (x, y) of the curve.
        steps: Number of line segments to approximate the curve. Defaults to 20
            (more than quadratic due to increased complexity).

    Returns:
        List of (x, y) tuples representing points along the curve, excluding p0
        but including p3. Returns `steps` points.

    Note:
        Cubic curves can represent more complex shapes than quadratic curves,
        including inflection points. The default step count of 20 provides
        adequate smoothness for typical glyph outlines.
    """
    pts = []
    for i in range(1, steps + 1):
        t = i / steps
        x = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
        y = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
        pts.append((x, y))
    return pts


def extract_contours(font_path: str, char: str) -> tuple[list | None, 'TTFont']:
    """Extract raw glyph contours from a font file using fontTools.

    This function uses fontTools' RecordingPen to capture all drawing operations
    for a character's glyph, then processes those operations into a list of
    contours (closed paths).

    Args:
        font_path: Absolute path to the font file (.ttf, .otf, etc.).
        char: Single character to extract contours for.

    Returns:
        A tuple of (contours, tt_font) where:
            - contours: List of contours, where each contour is a list of (x, y)
                points in font units. Points include flattened Bezier curves.
                Returns None if the character is not in the font.
            - tt_font: The opened TTFont object, which contains metrics needed
                for coordinate transformation (unitsPerEm, ascender, etc.).

    Note:
        Coordinates are in font units with Y-axis pointing up. The TTFont object
        should be used with font_to_pixel_transform() to convert to pixel space.

        The function handles both TrueType fonts (quadratic curves via qCurveTo)
        and PostScript/CFF fonts (cubic curves via curveTo). TrueType's qCurveTo
        can have multiple control points with implicit on-curve points between
        them, which this function handles by computing midpoints.
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
    """Build a transformation function from font units to pixel coordinates.

    This function replicates the centering logic used in glyph rendering functions
    (render_glyph_mask, render_char_image) to ensure contours align with rendered
    images.

    The transformation:
        1. Scales from font units to pixels based on font size
        2. Flips Y-axis (font Y-up to pixel Y-down)
        3. Applies offsets to center the glyph on the canvas

    Args:
        tt: TTFont object from fontTools, used to get unitsPerEm and ascender.
        font_path: Path to the font file (used to render with PIL for bbox).
        char: The character being transformed.
        canvas_size: Size of the square canvas in pixels. Defaults to 224.

    Returns:
        A transformation function with signature:
            transform(fx: float, fy: float) -> tuple[float, float]

        Where (fx, fy) are font-unit coordinates and the return value is
        pixel coordinates (px, py).

    Note:
        The function first renders the character with PIL to determine its
        bounding box, then calculates offsets to center it. If the character
        would exceed 90% of the canvas, the font size is scaled down.

        The Y transformation uses the font's ascender value as the reference
        point, converting Y-up to Y-down coordinates.
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
                       resolve_font_path_fn=None) -> list[list[tuple]]:
    """Extract glyph contours as pixel-space polylines.

    This is a convenience function that combines contour extraction with
    coordinate transformation to produce ready-to-use pixel coordinates.

    Args:
        font_path: Path to the font file.
        char: Single character to extract contours for.
        canvas_size: Size of the square canvas in pixels. Defaults to 224.
        resolve_font_path_fn: Optional callable to resolve the font path
            (e.g., handle aliases or relative paths). If None, font_path
            is used directly.

    Returns:
        List of contours, where each contour is a list of (x, y) tuples in
        pixel coordinates. Contours are automatically closed (first point
        appended to end if not already closed).

        Returns an empty list if the character is not found in the font.

    Note:
        Contours are closed if the distance between the first and last points
        exceeds 0.5 pixels. This handles cases where the font's closePath
        operation doesn't explicitly return to the start point.
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


def contour_segments(pixel_contours: list[list[tuple]]) -> list[tuple]:
    """Convert pixel contours to a flat list of line segments.

    This function extracts all line segments from contours for use in
    ray intersection tests and other geometric operations.

    Args:
        pixel_contours: List of contours from get_pixel_contours(), where
            each contour is a list of (x, y) tuples.

    Returns:
        List of line segments, where each segment is a tuple of two points:
        ((x0, y0), (x1, y1)). Segments connect adjacent points in each contour.

    Example:
        >>> contours = [[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]]
        >>> segments = contour_segments(contours)
        >>> len(segments)
        4
        >>> segments[0]
        ((0, 0), (10, 0))
    """
    segments = []
    for contour in pixel_contours:
        for i in range(len(contour) - 1):
            segments.append((contour[i], contour[i + 1]))
    return segments


def contour_to_strokes(font_path: str, char: str, canvas_size: int = 224,
                       resolve_font_path_fn=None) -> list[list[list[float]]] | None:
    """Convert the outer font contour to strokes by splitting at extremal points.

    This function extracts the largest (outer) contour of a glyph and splits it
    into two strokes at the topmost and bottommost points. This creates strokes
    that trace the left and right sides of the letter outline.

    Algorithm:
        1. Extract all contours and find the largest (by point count)
        2. Find the topmost point (minimum Y in pixel coords) and bottommost
        3. Split the contour into two halves at these extremal points
        4. Return both halves as separate strokes

    Args:
        font_path: Path to the font file.
        char: Single character to convert to strokes.
        canvas_size: Size of the square canvas in pixels. Defaults to 224.
        resolve_font_path_fn: Optional callable to resolve the font path.

    Returns:
        List of two strokes, where each stroke is a list of [x, y] coordinate
        pairs (as lists, not tuples). The strokes trace from top to bottom
        on opposite sides of the letter.

        Returns None if:
            - The character is not found in the font
            - No contours are extracted
            - The outer contour has fewer than 4 points

        Returns a single stroke (the entire contour) if top and bottom
        extremal points are the same index.

    Note:
        This method works best for simple letterforms. Complex letters with
        multiple parts (like 'i' or '%') will only return strokes for the
        largest connected outline.
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
                           resolve_font_path_fn=None) -> list[dict]:
    """Detect termination markers from contour extremal points.

    Termination markers indicate where strokes should naturally end, typically
    at the top and bottom of letters. These are extracted from the same
    extremal points used for contour splitting.

    Args:
        font_path: Path to the font file.
        char: Single character to detect markers for.
        canvas_size: Size of the square canvas in pixels. Defaults to 224.
        resolve_font_path_fn: Optional callable to resolve the font path.

    Returns:
        List of marker dictionaries, each containing:
            - 'x': X coordinate rounded to 1 decimal place
            - 'y': Y coordinate rounded to 1 decimal place
            - 'type': Always 'termination' for this function

        Typically returns 2 markers (top and bottom points).
        Returns an empty list if extraction fails.

    Note:
        These markers can be used to guide stroke optimization or to verify
        that generated strokes reach the expected endpoints of the letter.
    """
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


def ray_segment_intersection(origin: tuple, direction: tuple,
                             seg_a: tuple, seg_b: tuple) -> float | None:
    """Find the intersection of a ray with a line segment.

    Uses parametric ray-segment intersection. The ray is defined as:
        point = origin + t * direction, for t >= 0

    The segment is defined by endpoints seg_a and seg_b.

    Args:
        origin: Starting point (x, y) of the ray.
        direction: Direction vector (dx, dy) of the ray (does not need to be
            normalized).
        seg_a: First endpoint (x, y) of the line segment.
        seg_b: Second endpoint (x, y) of the line segment.

    Returns:
        The parameter t (distance along ray in direction units) where the ray
        intersects the segment, or None if:
            - Ray and segment are parallel (denominator near zero)
            - Intersection is behind the ray origin (t <= 0.5, threshold to
              avoid self-intersection)
            - Intersection is outside the segment (s < 0 or s > 1)

    Note:
        The threshold t > 0.5 prevents detecting intersections at or very near
        the ray origin, which is important when casting rays from points on
        the contour to avoid self-intersection with the originating segment.
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


def find_cross_section_midpoint(point: tuple, tangent: tuple,
                                segments: list[tuple], mask: np.ndarray) -> tuple | None:
    """Find the stroke center at a point by casting perpendicular rays.

    This function determines the center of a stroke at a given position by
    finding where perpendicular rays intersect the glyph contour on both sides,
    then returning the midpoint between those intersections.

    Algorithm:
        1. Compute perpendicular direction to the tangent
        2. Cast rays in both perpendicular directions
        3. Find nearest contour intersection on each side
        4. Return midpoint between the two intersections

    Args:
        point: The (x, y) position along the guide path to find the center for.
        tangent: The (dx, dy) tangent direction at that point (normalized or not).
        segments: List of line segments from contour_segments(), representing
            the glyph boundary.
        mask: Binary numpy array where True indicates glyph pixels. Used to
            verify that computed midpoints are inside the glyph.

    Returns:
        The (x, y) midpoint between contour intersections, or None if:
            - No intersection found on one or both sides
            - Computed midpoint is outside the glyph mask

        Falls back to the input point if it's inside the mask but no valid
        midpoint could be computed.

    Note:
        This is useful for generating centerline strokes from contour outlines.
        By sampling along a guide path (e.g., skeleton or medial axis) and
        finding midpoints, you can construct strokes that follow the center
        of the letterform.
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
