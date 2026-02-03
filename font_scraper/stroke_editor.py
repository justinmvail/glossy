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
from collections import defaultdict
from urllib.parse import quote as urlquote
from flask import Flask, render_template, request, jsonify, send_file, Response
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen
from inksight_vectorizer import InkSightVectorizer
try:
    from docker.diffvg_docker import DiffVGDocker
    _diffvg_docker = DiffVGDocker()
except ImportError:
    _diffvg_docker = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
DB_PATH = os.path.join(BASE_DIR, 'fonts.db')


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
NUMPAD_TEMPLATES = {
    # --- Uppercase ---
    'A': [[1, 'v(8)', 3], [4, 6]],
    'B': [[7, 1], [7, 'c(9)', 'v(6)', 'c(3)', 1]],
    'C': [[9, 'c(7)', 'c(1)', 3]],
    'D': [[7, 1], [7, 'c(9)', 'c(3)', 1]],
    'E': [[9, 7, 1, 3], [4, 6]],
    'F': [[9, 7, 1], [4, 6]],
    'G': [[9, 'c(7)', 'c(1)', 3, 6], [6, 5]],
    'H': [[7, 1], [9, 3], [4, 6]],
    'I': [[8, 2]],
    'J': [[8, 'c(1)']],
    'K': [[7, 1], [9, 'v(4)', 3]],
    'L': [[7, 1, 3]],
    'M': [[1, 7, 'v(2)', 9, 3]],
    'N': [[1, 7, 3, 9]],
    'O': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8]],
    'P': [[1, 7, 'c(9)', 'c(6)', 4]],
    'Q': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8], [5, 3]],
    'R': [[1, 7, 'c(9)', 'c(6)', 4], [4, 3]],
    'S': [[9, 'c(7)', 'c(4)', 'c(6)', 'c(3)', 1]],
    'T': [[7, 9], [8, 2]],
    'U': [[7, 'c(1)', 'c(3)', 9]],
    'V': [[7, 'v(2)', 9]],
    'W': [[7, 'v(1)', 'v(5)', 'v(3)', 9]],
    'X': [[7, 3], [9, 1]],
    'Y': [[7, 'v(5)', 9], [5, 2]],
    'Z': [[7, 9, 1, 3]],

    # --- Lowercase ---
    'a': [[9, 'c(7)', 'c(1)', 3], [9, 3]],
    'b': [[7, 1, 'c(3)', 'c(9)', 4]],
    'c': [[9, 'c(7)', 'c(1)', 3]],
    'd': [[9, 'c(7)', 'c(1)', 3], [9, 3]],
    'e': [[6, 4, 'c(1)', 'c(3)', 'c(6)']],
    'f': [[9, 'c(8)', 2], [4, 6]],
    'g': [[9, 'c(7)', 'c(1)', 'c(3)', 9], [9, 'c(3)']],
    'h': [[7, 1], [4, 'c(9)', 3]],
    'i': [[8, 2]],
    'j': [[8, 'c(1)']],
    'k': [[7, 1], [9, 'v(4)', 3]],
    'l': [[8, 2]],
    'm': [[1, 4, 'c(8)', 5], [5, 'c(9)', 3]],
    'n': [[1, 4, 'c(9)', 3]],
    'o': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8]],
    'p': [[4, 'c(7)', 'c(9)', 'c(6)', 1]],
    'q': [[9, 'c(7)', 'c(1)', 'c(3)', 9], [9, 3]],
    'r': [[1, 4, 'c(9)']],
    's': [[9, 'c(7)', 'c(4)', 'c(6)', 'c(3)', 1]],
    't': [[8, 2], [7, 9]],
    'u': [[7, 'c(1)', 'c(3)', 9], [9, 3]],
    'v': [[7, 'v(2)', 9]],
    'w': [[7, 'v(1)', 'v(5)', 'v(3)', 9]],
    'x': [[7, 3], [9, 1]],
    'y': [[7, 'v(5)'], [9, 'v(5)', 'c(1)']],
    'z': [[7, 9, 1, 3]],

    # --- Digits ---
    '0': [[8, 'c(7)', 'c(1)', 'c(3)', 'c(9)', 8]],
    '1': [[7, 8, 2]],
    '2': [[7, 'c(9)', 'c(6)', 'v(4)', 1, 3]],
    '3': [[7, 'c(9)', 'c(6)', 4], [4, 'c(6)', 'c(3)', 1]],
    '4': [[7, 4, 6], [9, 3]],
    '5': [[9, 7, 4, 'c(6)', 'c(3)', 1]],
    '6': [[9, 'c(7)', 'c(1)', 'c(3)', 'c(6)', 4]],
    '7': [[7, 9, 1]],
    '8': [[5, 'c(8)', 'c(7)', 'c(4)', 5], [5, 'c(2)', 'c(3)', 'c(6)', 5]],
    '9': [[6, 'c(9)', 'c(8)', 'c(7)', 'c(4)', 6], [6, 'c(3)', 1]],
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

    def _snap_inside(pos):
        """Snap a position to the nearest mask pixel if outside."""
        ix = int(round(min(max(pos[0], 0), w - 1)))
        iy = int(round(min(max(pos[1], 0), h - 1)))
        if mask[iy, ix]:
            return pos
        ny = float(snap_indices[0, iy, ix])
        nx = float(snap_indices[1, iy, ix])
        return (nx, ny)

    def _snap_deep_inside(pos):
        """Snap a position to be well inside the mask.

        Cast ray from pos toward centroid, find the point with maximum
        distance-from-edge (deepest inside the glyph).
        """
        ix = int(round(min(max(pos[0], 0), w - 1)))
        iy = int(round(min(max(pos[1], 0), h - 1)))
        if mask[iy, ix] and dist_in[iy, ix] >= 5:
            return pos

        # Walk from pos toward centroid, find deepest point
        dx = centroid[0] - pos[0]
        dy = centroid[1] - pos[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1:
            return _snap_inside(pos)

        best_pos = _snap_inside(pos)
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

    def _snap_to_skeleton(region):
        """Find the closest skeleton feature to the numpad region center."""
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

    # Map waypoints to pixel positions — prefer skeleton features
    positions = []
    for region, kind in parsed:
        # Try skeleton feature first
        skel_pos = _snap_to_skeleton(region)
        if skel_pos is not None:
            pos = (float(skel_pos[0]), float(skel_pos[1]))
        elif kind == 'terminal':
            pos = _snap_to_glyph_edge(
                _numpad_to_pixel(region, glyph_bbox), centroid, mask)
            if pos is None:
                return []
        else:
            pos = _snap_deep_inside(_numpad_to_pixel(region, glyph_bbox))
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
            constrained.append(_snap_inside((x, y)))

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
    """Gaussian smooth a stroke's x and y coordinates independently."""
    if len(points) < 3:
        return points
    arr = np.array(points, dtype=float)
    arr[:, 0] = gaussian_filter1d(arr[:, 0], sigma=sigma)
    arr[:, 1] = gaussian_filter1d(arr[:, 1], sigma=sigma)
    return [tuple(p) for p in arr]


def _constrain_to_mask(points, mask):
    """Constrain points to stay inside the glyph mask.

    For any point outside the mask, snap it to the nearest inside pixel.
    """
    h, w = mask.shape
    dist_out, indices = distance_transform_edt(~mask, return_indices=True)
    result = []
    for x, y in points:
        ix = int(round(min(max(x, 0), w - 1)))
        iy = int(round(min(max(y, 0), h - 1)))
        if mask[iy, ix]:
            result.append((x, y))
        else:
            ny = float(indices[0, iy, ix])
            nx = float(indices[1, iy, ix])
            result.append((nx, ny))
    return result


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


def _optimize_affine(font_path, char, canvas_size=224):
    """Optimise template strokes via affine transforms.

    Stage 1: Global affine (6 params) on all strokes together.
    Stage 2: Per-stroke translate+scale refinement.

    Returns (strokes, score, mask, glyph_bbox) or None if no template.
    """
    import time as _time
    from scipy.optimize import minimize, differential_evolution

    strokes_raw = template_to_strokes(font_path, char, canvas_size)
    if not strokes_raw or len(strokes_raw) == 0:
        return None

    font_path = resolve_font_path(font_path)
    mask = render_glyph_mask(font_path, char, canvas_size)
    if mask is None:
        return None
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

    # ---- Stage 1: Global affine (6 params) ----
    # params: tx, ty, sx, sy, theta_deg, shear
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

    # ---- Stage 2: Per-stroke refinement ----
    # Each stroke gets (dx, dy, sx, sy)
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
        best_score = nm2.fun
    else:
        final_strokes = best_strokes

    # Convert back to list format
    result_strokes = [[[round(float(p[0]), 1), round(float(p[1]), 1)]
                       for p in s] for s in final_strokes]

    return result_strokes, float(-best_score), mask, glyph_bbox


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


def auto_fit_strokes(font_path, char, canvas_size=224, return_markers=False):
    """Generate strokes by optimising shape parameters with a greedy per-shape
    approach followed by joint refinement.

    Phase 1: Greedy — optimise each shape against uncovered points
    Phase 2: Joint NM refinement of all params together
    Phase 3: Joint DE global search if time remains

    Caches winning params in DB so subsequent runs start from the best
    known position and can improve further.

    Returns list of strokes [[[x,y], ...], ...] or None.
    If return_markers=True returns (strokes, markers).
    """
    from scipy.optimize import differential_evolution, minimize
    import time as _time

    templates = SHAPE_TEMPLATES.get(char)
    if not templates:
        return None

    font_path = resolve_font_path(font_path)
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

    # Pre-compute nearest-mask-pixel arrays for snap-during-scoring
    dist_map = distance_transform_edt(mask)
    _, snap_indices = distance_transform_edt(~mask, return_indices=True)
    snap_yi = snap_indices[0]
    snap_xi = snap_indices[1]

    shape_types = [t['shape'] for t in templates]
    bounds, slices = _get_param_bounds(templates)

    # Initial guess: prefer cached params, fall back to template defaults
    x0 = []
    for t in templates:
        x0.extend(t['params'])
    x0 = np.array(x0, dtype=float)

    # Strip font_path back to relative for DB lookup
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

    bounds_lo = np.array([b[0] for b in bounds])
    bounds_hi = np.array([b[1] for b in bounds])

    def _clamp(x):
        return np.clip(x, bounds_lo, bounds_hi)

    def _score_all_clamped(params, *args):
        return _score_all_strokes(_clamp(params), *args)

    _t_start = _time.monotonic()
    _TIME_BUDGET = 3600.0  # seconds (1 hour max)
    _STALE_THRESHOLD = 0.001  # stop if score improves less than this over a full cycle
    _STALE_CYCLES = 2  # stop after this many cycles with no meaningful improvement

    def _elapsed():
        return _time.monotonic() - _t_start

    # ---- Phase 0: Template + Affine + DiffVG optimisation ----
    affine_strokes_result = None  # (stroke_list, score) if affine produced something

    # Try DiffVG first (gradient-based, typically better)
    diffvg_result = _optimize_diffvg(font_path, char, canvas_size)
    if diffvg_result is not None:
        dv_strokes, dv_score, _, _ = diffvg_result
        if dv_strokes and dv_score > 0:
            affine_strokes_result = (dv_strokes, dv_score)
            if dv_score >= 0.85:
                if return_markers:
                    markers = []
                    for si, st in enumerate(dv_strokes):
                        markers.append({'x': st[0][0], 'y': st[0][1],
                                        'type': 'start', 'label': 'S', 'stroke_id': si})
                        markers.append({'x': st[-1][0], 'y': st[-1][1],
                                        'type': 'stop', 'label': 'E', 'stroke_id': si})
                    return dv_strokes, markers
                return dv_strokes

    # Fall back to affine optimisation
    affine_result = _optimize_affine(font_path, char, canvas_size)
    if affine_result is not None:
        affine_strokes, affine_score, _, _ = affine_result
        aff_stroke_list = [[[round(float(x), 1), round(float(y), 1)] for x, y in s]
                           for s in affine_strokes if len(s) >= 2]
        if aff_stroke_list:
            # Keep whichever scored better: DiffVG or affine
            if affine_strokes_result and affine_strokes_result[1] >= affine_score:
                pass  # DiffVG already better
            else:
                affine_strokes_result = (aff_stroke_list, affine_score)
            best_strokes, best_score = affine_strokes_result
            if best_score >= 0.85:
                if return_markers:
                    markers = []
                    for si, st in enumerate(best_strokes):
                        markers.append({'x': st[0][0], 'y': st[0][1],
                                        'type': 'start', 'label': 'S', 'stroke_id': si})
                        markers.append({'x': st[-1][0], 'y': st[-1][1],
                                        'type': 'stop', 'label': 'E', 'stroke_id': si})
                    return best_strokes, markers
                return best_strokes

    # ---- Phase 1: Greedy per-shape optimisation ----
    # Optimise each shape in turn against uncovered points.
    # This reduces the problem from N*D dims to D dims per shape.
    greedy_x = x0.copy()
    uncovered_mask = np.ones(n_cloud, dtype=bool)

    for si in range(len(templates)):
        if _elapsed() >= _TIME_BUDGET * 0.4:
            break
        start, end = slices[si]
        stype = shape_types[si]
        s_bounds = bounds[start:end]
        s_x0 = greedy_x[start:end].copy()

        # Build tree from currently uncovered points
        uncov_idx = np.where(uncovered_mask)[0]
        if len(uncov_idx) < 5:
            break
        uncov_pts = cloud[uncov_idx]
        uncov_tree = cKDTree(uncov_pts)

        s_args = (stype, glyph_bbox, uncov_pts, uncov_tree,
                  len(uncov_pts), radius, snap_yi, snap_xi, w, h)

        # NM for this shape (clamp to bounds since NM is unconstrained)
        s_lo = bounds_lo[start:end]
        s_hi = bounds_hi[start:end]
        def _score_single_clamped(p, *a, _lo=s_lo, _hi=s_hi):
            return _score_single_shape(np.clip(p, _lo, _hi), *a)
        nm_r = minimize(
            _score_single_clamped, s_x0, args=s_args, method='Nelder-Mead',
            options={'maxfev': 800, 'xatol': 0.2, 'fatol': 0.002, 'adaptive': True},
        )
        best_s = np.clip(nm_r.x, s_lo, s_hi).copy()
        best_sf = nm_r.fun

        # Quick DE for this shape if time permits
        if _elapsed() < _TIME_BUDGET * 0.35:
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
                    best_sf = de_r.fun
            except Exception:
                pass

        # Store best params for this shape
        greedy_x[start:end] = best_s

        # Mark newly covered points
        n_pts_shape = max(60, int(((glyph_bbox[2]-glyph_bbox[0])**2 +
                                    (glyph_bbox[3]-glyph_bbox[1])**2)**0.5 / 1.5))
        pts = SHAPE_FNS[stype](tuple(best_s), glyph_bbox, offset=(0, 0),
                               n_pts=n_pts_shape)
        if len(pts) > 0:
            xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
            yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
            snapped_x = snap_xi[yi, xi].astype(float)
            snapped_y = snap_yi[yi, xi].astype(float)
            snapped = np.column_stack([snapped_x, snapped_y])
            # Find which cloud points are now covered
            dists, _ = cloud_tree.query(snapped, k=1)
            # Use ball query on the uncov_tree
            hits = cloud_tree.query_ball_point(snapped, radius)
            newly_covered = set()
            for lst in hits:
                newly_covered.update(lst)
            for idx in newly_covered:
                uncovered_mask[idx] = False

    # ---- Phase 2: Joint NM refinement from greedy solution ----
    best_x = greedy_x.copy()
    best_fun = _score_all_strokes(greedy_x, *joint_args)

    # Also compare with original x0
    x0_fun = _score_all_strokes(x0, *joint_args)
    if x0_fun < best_fun:
        best_x = x0.copy()
        best_fun = x0_fun

    def _update_best(x, fun):
        nonlocal best_x, best_fun
        if fun < best_fun:
            best_x = x.copy()
            best_fun = fun

    def _perfect():
        return best_fun <= -0.99

    class _EarlyStop(Exception):
        pass

    # ---- Repeating NM → DE → NM cycle until stagnation or time limit ----
    stale_count = 0
    cycle_num = 0
    while not _perfect() and _elapsed() < _TIME_BUDGET and stale_count < _STALE_CYCLES:
        cycle_num += 1
        score_at_cycle_start = best_fun

        # NM refinement
        if not _perfect() and _elapsed() < _TIME_BUDGET:
            remaining_fev = max(500, int(min(30.0, _TIME_BUDGET - _elapsed()) / 0.0003))
            nm_result = minimize(
                _score_all_clamped, best_x, args=joint_args, method='Nelder-Mead',
                options={'maxfev': remaining_fev, 'xatol': 0.2, 'fatol': 0.0005,
                         'adaptive': True},
            )
            _update_best(_clamp(nm_result.x), nm_result.fun)

        # DE global search
        if not _perfect() and _elapsed() < _TIME_BUDGET:
            nm_x = best_x.copy()
            for i, (lo, hi) in enumerate(bounds):
                nm_x[i] = np.clip(nm_x[i], lo, hi)

            def _de_callback(xk, convergence=0):
                _update_best(xk, _score_all_strokes(xk, *joint_args))
                if _perfect() or _elapsed() >= _TIME_BUDGET:
                    raise _EarlyStop()

            try:
                de_result = differential_evolution(
                    _score_all_strokes,
                    bounds=bounds,
                    args=joint_args,
                    x0=nm_x,
                    maxiter=200,
                    popsize=20,
                    tol=0.002,
                    seed=None,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    polish=False,
                    disp=False,
                    callback=_de_callback,
                )
                _update_best(de_result.x, de_result.fun)
            except _EarlyStop:
                pass

        # NM polish
        if not _perfect() and _elapsed() < _TIME_BUDGET:
            remaining_fev = max(200, int(min(15.0, _TIME_BUDGET - _elapsed()) / 0.0003))
            nm2 = minimize(
                _score_all_clamped, best_x, args=joint_args, method='Nelder-Mead',
                options={'maxfev': remaining_fev, 'xatol': 0.1, 'fatol': 0.0005,
                         'adaptive': True},
            )
            _update_best(_clamp(nm2.x), nm2.fun)

        # Check for stagnation
        improvement = score_at_cycle_start - best_fun  # positive = improved
        if improvement < _STALE_THRESHOLD:
            stale_count += 1
        else:
            stale_count = 0

        # Cache periodically so progress isn't lost
        current_score = float(-best_fun)
        if cached_score is None or current_score > cached_score:
            _save_cached_params(rel_path, char, best_x, current_score)
            cached_score = current_score

    # Compare shape optimisation vs affine — use whichever scored higher
    final_score = float(-best_fun)
    if affine_strokes_result and affine_strokes_result[1] > final_score:
        # Affine was better — return affine strokes directly
        aff_strokes, aff_score = affine_strokes_result
        if return_markers:
            markers = []
            for si, st in enumerate(aff_strokes):
                markers.append({'x': st[0][0], 'y': st[0][1],
                                'type': 'start', 'label': 'S', 'stroke_id': si})
                markers.append({'x': st[-1][0], 'y': st[-1][1],
                                'type': 'stop', 'label': 'E', 'stroke_id': si})
            return aff_strokes, markers
        return aff_strokes

    # Cache the winning params if they improved over what was stored
    if cached_score is None or final_score > cached_score:
        _save_cached_params(rel_path, char, best_x, final_score)

    best_shapes = _param_vector_to_shapes(best_x, shape_types, slices, glyph_bbox)

    # Group shapes into strokes based on 'group' key in templates.
    # Shapes with the same group are joined into one stroke; shapes without
    # a group get their own stroke.
    from collections import OrderedDict
    groups = OrderedDict()  # group_key -> list of (shape_idx, stype, pts)
    _auto_gid = 1000
    for si, (tmpl, stype, pts) in enumerate(zip(templates, shape_types, best_shapes)):
        gid = tmpl.get('group')
        if gid is None:
            gid = _auto_gid
            _auto_gid += 1
        groups.setdefault(gid, []).append((si, stype, pts))

    strokes = []
    all_markers = []
    stroke_idx = 0
    for gid, members in groups.items():
        # Process each shape in the group
        shape_strokes = []  # list of point lists for each shape in group
        for si, stype, pts in members:
            point_list = [(float(p[0]), float(p[1])) for p in pts]
            point_list = _smooth_stroke(point_list, sigma=2.0)
            point_list = _constrain_to_mask(point_list, mask)
            if len(point_list) >= 2:
                shape_strokes.append((si, stype, point_list))

        if not shape_strokes:
            continue

        if len(shape_strokes) == 1:
            # Single shape → single stroke
            si, stype, point_list = shape_strokes[0]
            final_stroke = [[round(x, 1), round(y, 1)] for x, y in point_list]
        else:
            # Multiple shapes → join by finding nearest endpoints.
            # Try both orientations of the first shape and pick the
            # chain with the shortest total gap distance.
            def _build_chain(start_shape, rest):
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
                            best_dist = d_start
                            best_idx = ri
                            best_flip = False
                        if d_end < best_dist:
                            best_dist = d_end
                            best_idx = ri
                            best_flip = True
                    total_gap += best_dist ** 0.5
                    chosen = remaining.pop(best_idx)
                    if best_flip:
                        chosen = (chosen[0], chosen[1], list(reversed(chosen[2])))
                    chain.append(chosen)
                return chain, total_gap

            s0 = shape_strokes[0]
            s0_flip = (s0[0], s0[1], list(reversed(s0[2])))
            rest = shape_strokes[1:]
            chain_fwd, gap_fwd = _build_chain(s0, rest)
            chain_rev, gap_rev = _build_chain(s0_flip, rest)
            chain = chain_fwd if gap_fwd <= gap_rev else chain_rev

            # Concatenate all shape points into one stroke
            combined = []
            for ci, (csi, cstype, cpts) in enumerate(chain):
                if ci > 0:
                    # Add vertex marker at the join point
                    jpt = cpts[0]
                    if return_markers:
                        all_markers.append({
                            'x': round(jpt[0], 1), 'y': round(jpt[1], 1),
                            'type': 'vertex', 'label': 'V', 'stroke_id': stroke_idx
                        })
                    # Skip first few points if they overlap with end of previous
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

        if len(final_stroke) < 2:
            continue
        strokes.append(final_stroke)

        if return_markers:
            all_markers.append({
                'x': final_stroke[0][0], 'y': final_stroke[0][1],
                'type': 'start', 'label': 'S', 'stroke_id': stroke_idx
            })
            all_markers.append({
                'x': final_stroke[-1][0], 'y': final_stroke[-1][1],
                'type': 'stop', 'label': 'E', 'stroke_id': stroke_idx
            })
            # Add curve markers for arcs/loops within this stroke
            for si, stype, point_list in (shape_strokes if len(shape_strokes) == 1
                                          else chain):
                if stype in ('arc_right', 'arc_left', 'loop', 'u_arc'):
                    # Find midpoint of this shape in the final stroke
                    mid_pt = point_list[len(point_list) // 2]
                    all_markers.append({
                        'x': round(mid_pt[0], 1),
                        'y': round(mid_pt[1], 1),
                        'type': 'curve', 'label': 'C', 'stroke_id': stroke_idx
                    })

        stroke_idx += 1

    if not strokes:
        return None
    if return_markers:
        return strokes, all_markers
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


STROKE_COLORS = [
    (255, 80, 80), (80, 180, 255), (80, 220, 80), (255, 180, 40),
    (200, 100, 255), (255, 120, 200), (100, 220, 220), (180, 180, 80),
]


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

    # Draw strokes
    if row and row['strokes_raw']:
        strokes = json.loads(row['strokes_raw'])
        draw = ImageDraw.Draw(bg)
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
    """Skeletonize a mask and return adjacency, junction clusters, and endpoints."""
    skel = skeletonize(mask)
    ys, xs = np.where(skel)
    skel_set = set(zip(xs.tolist(), ys.tolist()))
    if not skel_set:
        return None

    # Build adjacency (8-connected)
    adj = defaultdict(list)
    for (x, y) in skel_set:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                n = (x + dx, y + dy)
                if n in skel_set:
                    adj[(x, y)].append(n)

    # Cluster adjacent junction pixels into single logical junctions
    junction_pixels = set(p for p in skel_set if len(adj[p]) >= 3)
    junction_clusters = []  # list of sets
    assigned = {}  # pixel -> cluster_index
    for jp in junction_pixels:
        if jp in assigned:
            continue
        cluster = set()
        queue = [jp]
        while queue:
            p = queue.pop()
            if p in cluster:
                continue
            cluster.add(p)
            assigned[p] = len(junction_clusters)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    n = (p[0] + dx, p[1] + dy)
                    if n in junction_pixels and n not in cluster:
                        queue.append(n)
        junction_clusters.append(cluster)

    # Merge nearby junction clusters whose centroids are within merge_dist
    merge_dist = 12
    merged_flag = True
    while merged_flag:
        merged_flag = False
        for i in range(len(junction_clusters)):
            ci = junction_clusters[i]
            cx_i = sum(p[0] for p in ci) / len(ci)
            cy_i = sum(p[1] for p in ci) / len(ci)
            for j in range(i + 1, len(junction_clusters)):
                cj = junction_clusters[j]
                cx_j = sum(p[0] for p in cj) / len(cj)
                cy_j = sum(p[1] for p in cj) / len(cj)
                dx = cx_i - cx_j
                dy = cy_i - cy_j
                if (dx * dx + dy * dy) ** 0.5 < merge_dist:
                    # Merge j into i, also absorb bridging skeleton pixels
                    # BFS from ci to find shortest path to any pixel in cj
                    from collections import deque
                    bfs_q = deque()
                    bfs_parent = {}
                    for p in ci:
                        bfs_q.append(p)
                        bfs_parent[p] = None
                    bridge_path = []
                    while bfs_q:
                        p = bfs_q.popleft()
                        if p in cj:
                            # Trace back path
                            cur = p
                            while cur is not None and cur not in ci:
                                bridge_path.append(cur)
                                cur = bfs_parent[cur]
                            break
                        for nb in adj[p]:
                            if nb not in bfs_parent:
                                bfs_parent[nb] = p
                                bfs_q.append(nb)
                    merged_cluster = ci | cj
                    for bp in bridge_path:
                        merged_cluster.add(bp)
                        junction_pixels.add(bp)
                    junction_clusters[i] = merged_cluster
                    junction_clusters.pop(j)
                    # Rebuild assigned for all clusters
                    for p in junction_clusters[i]:
                        assigned[p] = i
                    for k in range(j, len(junction_clusters)):
                        for p in junction_clusters[k]:
                            assigned[p] = k
                    merged_flag = True
                    break
            if merged_flag:
                break

    endpoints = set(p for p in skel_set if len(adj[p]) == 1)

    return {
        'skel_set': skel_set,
        'adj': adj,
        'junction_pixels': junction_pixels,
        'junction_clusters': junction_clusters,
        'assigned': assigned,
        'endpoints': endpoints,
    }


def skeleton_detect_markers(mask, merge_dist=12):
    """Detect vertex (junction) and termination (endpoint) markers from skeleton.

    Vertices = centroids of junction clusters (where 3+ branches meet).
    Terminations = skeleton endpoints (degree 1 pixels).
    Nearby vertices are merged. Terminations that fall inside a junction
    cluster are removed (they're part of the junction, not real endpoints).
    """
    info = _analyze_skeleton(mask)
    if not info:
        return []

    adj = info['adj']
    endpoints = info['endpoints']
    junction_pixels = info['junction_pixels']

    # Vertices: centroid of each junction cluster
    vertices = []
    for cluster in info['junction_clusters']:
        cx = sum(p[0] for p in cluster) / len(cluster)
        cy = sum(p[1] for p in cluster) / len(cluster)
        vertices.append([cx, cy])

    # Merge nearby vertices
    merged = True
    while merged:
        merged = False
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                dx = vertices[i][0] - vertices[j][0]
                dy = vertices[i][1] - vertices[j][1]
                if (dx * dx + dy * dy) ** 0.5 < merge_dist:
                    vertices[i] = [(vertices[i][0] + vertices[j][0]) / 2,
                                   (vertices[i][1] + vertices[j][1]) / 2]
                    vertices.pop(j)
                    merged = True
                    break
            if merged:
                break

    # Classify junction clusters as vertex vs intersection.
    # A vertex has a convergence stub: a short skeleton path from the
    # junction to a nearby endpoint, meaning strokes converge to a point
    # (e.g. apex of A).  The vertex marker moves to the stub tip.
    # An intersection has no convergence stub: strokes cross through it
    # (e.g. where A's crossbar meets its legs).
    assigned = info['assigned']
    stub_max_len = 18  # convergence stubs are short artifacts, not real strokes
    absorbed_endpoints = set()
    is_vertex = [False] * len(info['junction_clusters'])

    for (ex, ey) in endpoints:
        if (ex, ey) in junction_pixels:
            continue
        # Trace from this endpoint toward a junction cluster
        path = [(ex, ey)]
        current = (ex, ey)
        prev = None
        reached_cluster = -1
        for _ in range(stub_max_len):
            neighbors = [n for n in adj[current] if n != prev]
            if not neighbors:
                break
            nxt = neighbors[0]
            path.append(nxt)
            if nxt in junction_pixels:
                reached_cluster = assigned.get(nxt, -1)
                break
            if len(adj[nxt]) != 2:
                break  # branching or dead end
            prev, current = current, nxt
        if reached_cluster < 0:
            continue
        # Collect direction vectors of OTHER branches leaving this cluster
        cluster = info['junction_clusters'][reached_cluster]
        path_set = set(path)
        branch_dirs = []
        for cp in cluster:
            for nb in adj[cp]:
                if nb in cluster or nb in path_set:
                    continue
                # Walk a few steps to get a stable direction
                bx, by = nb[0] - cp[0], nb[1] - cp[1]
                cur, prv = nb, cp
                for _ in range(6):
                    nbs = [n for n in adj[cur] if n != prv and n not in cluster]
                    if not nbs:
                        break
                    nxt = nbs[0]
                    prv, cur = cur, nxt
                bx, by = cur[0] - cp[0], cur[1] - cp[1]
                bl = (bx * bx + by * by) ** 0.5
                if bl > 0.01:
                    branch_dirs.append((bx / bl, by / bl))
        if len(branch_dirs) < 2:
            continue

        # Check if any two branches form a pass-through (roughly opposite
        # directions, dot < -0.5).  If so, strokes cross here → intersection,
        # not a convergence vertex.  Threshold -0.5 corresponds to ~120° apart.
        is_passthrough = False
        for i in range(len(branch_dirs)):
            for j in range(i + 1, len(branch_dirs)):
                dot = (branch_dirs[i][0] * branch_dirs[j][0] +
                       branch_dirs[i][1] * branch_dirs[j][1])
                if dot < -0.5:
                    is_passthrough = True
                    break
            if is_passthrough:
                break

        # Also check convergence: the stub must be opposite ALL branches
        # (all branches fan out from the junction on the opposite side of
        # the stub tip).  If the stub aligns with one branch but not others,
        # it's just extending that branch at a corner, not converging.
        cluster = info['junction_clusters'][reached_cluster]
        ccx = sum(p[0] for p in cluster) / len(cluster)
        ccy = sum(p[1] for p in cluster) / len(cluster)
        sdx, sdy = ex - ccx, ey - ccy
        sl = (sdx * sdx + sdy * sdy) ** 0.5
        if sl > 0.01:
            sdx /= sl; sdy /= sl
        all_branches_opposite = True
        for bd in branch_dirs:
            if sdx * bd[0] + sdy * bd[1] >= -0.5:
                all_branches_opposite = False
                break

        if is_passthrough or not all_branches_opposite:
            # Not a convergence vertex: either strokes pass through, or
            # the stub only opposes some branches (corner, not convergence).
            absorbed_endpoints.add((ex, ey))
            continue

        # Branches converge (no pass-through pair, all opposite stub) → vertex
        is_vertex[reached_cluster] = True
        vertices[reached_cluster] = [float(ex), float(ey)]
        absorbed_endpoints.add((ex, ey))

    # Keep terminations that aren't inside a junction cluster, aren't
    # too close to a vertex, and weren't absorbed as convergence stubs
    near_vertex_dist = 5
    terminations = []
    for (x, y) in endpoints:
        if (x, y) in junction_pixels:
            continue
        if (x, y) in absorbed_endpoints:
            continue
        # Check distance to nearest vertex
        too_close = False
        for v in vertices:
            dx = v[0] - x
            dy = v[1] - y
            if (dx * dx + dy * dy) ** 0.5 < near_vertex_dist:
                too_close = True
                break
        if not too_close:
            terminations.append([float(x), float(y)])

    # Merge terminations that are very close to each other (within 5px)
    merged = True
    while merged:
        merged = False
        for i in range(len(terminations)):
            for j in range(i + 1, len(terminations)):
                dx = terminations[i][0] - terminations[j][0]
                dy = terminations[i][1] - terminations[j][1]
                if (dx * dx + dy * dy) ** 0.5 < 5:
                    terminations[i] = [(terminations[i][0] + terminations[j][0]) / 2,
                                       (terminations[i][1] + terminations[j][1]) / 2]
                    terminations.pop(j)
                    merged = True
                    break
            if merged:
                break

    markers = []
    for i, v in enumerate(vertices):
        mtype = 'vertex' if is_vertex[i] else 'intersection'
        markers.append({'x': round(v[0], 1), 'y': round(v[1], 1), 'type': mtype})
    for t in terminations:
        markers.append({'x': t[0], 'y': t[1], 'type': 'termination'})

    return markers


def skeleton_to_strokes(mask, min_stroke_len=5):
    """Extract stroke paths from a glyph mask via skeletonization."""
    info = _analyze_skeleton(mask)
    if not info:
        return []

    skel_set = info['skel_set']
    adj = info['adj']
    junction_pixels = info['junction_pixels']
    junction_clusters = info['junction_clusters']
    assigned = info['assigned']
    endpoints = info['endpoints']

    # For tracing, all junction cluster pixels are stop points
    stop_set = endpoints | junction_pixels

    visited_edges = set()
    raw_strokes = []

    def trace(start, neighbor):
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
            # Pick the neighbor that continues straightest (least direction
            # change from prev→current to current→next).
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

    # Trace from endpoints first, then junction pixels
    for start in sorted(endpoints):
        for neighbor in adj[start]:
            p = trace(start, neighbor)
            if p and len(p) >= 2:
                raw_strokes.append(p)

    for start in sorted(junction_pixels):
        for neighbor in adj[start]:
            p = trace(start, neighbor)
            if p and len(p) >= 2:
                raw_strokes.append(p)

    # Filter tiny stubs
    strokes = [s for s in raw_strokes if len(s) >= min_stroke_len]

    # --- Merge strokes through junction clusters ---
    # For each junction cluster, find pairs of strokes whose endpoints
    # land in that cluster and whose directions align (continuation).

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
        dot = d1[0] * d2[0] + d1[1] * d2[1]
        return np.arccos(max(-1.0, min(1.0, dot)))

    def _endpoint_cluster(stroke, from_end):
        """Which junction cluster does this stroke endpoint belong to?"""
        pt = tuple(stroke[-1]) if from_end else tuple(stroke[0])
        return assigned.get(pt, -1)

    def _run_merge_pass(strokes, min_len=0, max_angle=np.pi/4,
                        max_ratio=0):
        """Merge strokes through junction clusters by direction alignment.
        max_ratio > 0 means reject pairs where max(len)/min(len) > ratio.
        """
        changed = True
        while changed:
            changed = False
            cluster_map = defaultdict(list)
            for si, s in enumerate(strokes):
                sc = _endpoint_cluster(s, False)
                if sc >= 0:
                    cluster_map[sc].append((si, 'start'))
                ec = _endpoint_cluster(s, True)
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
                        # Don't merge with a loop stroke (both endpoints at
                        # the same junction cluster)
                        sci = _endpoint_cluster(strokes[si], False)
                        eci = _endpoint_cluster(strokes[si], True)
                        scj = _endpoint_cluster(strokes[sj], False)
                        ecj = _endpoint_cluster(strokes[sj], True)
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

    # Pass 1: T-junction merge.  At junctions with 3+ strokes, if the
    # shortest stroke is a cross-branch (both endpoints in junction
    # clusters) and much shorter than the main branches, merge the two
    # longest with a relaxed angle threshold.  This handles letters like
    # B where bumps approach the pinch junction from perpendicular
    # directions but should form the "3" shape.
    changed = True
    while changed:
        changed = False
        cluster_map = defaultdict(list)
        for si, s in enumerate(strokes):
            sc = _endpoint_cluster(s, False)
            if sc >= 0:
                cluster_map[sc].append((si, 'start'))
            ec = _endpoint_cluster(s, True)
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
            s_sc = _endpoint_cluster(shortest_stroke, False)
            s_ec = _endpoint_cluster(shortest_stroke, True)
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
            far_i = _endpoint_cluster(strokes[si], from_end=(side_i != 'end'))
            far_j = _endpoint_cluster(strokes[sj], from_end=(side_j != 'end'))
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
                # Re-find it since indices shifted
                for sk in range(len(strokes)):
                    s = strokes[sk]
                    s_sc2 = _endpoint_cluster(s, False)
                    s_ec2 = _endpoint_cluster(s, True)
                    if s_sc2 >= 0 and s_ec2 >= 0 and len(s) < second_longest_len * 0.4:
                        if s_sc2 == cid or s_ec2 == cid:
                            strokes.pop(sk)
                            break
                changed = True
                break

    # Pass 2: standard direction-based merge.
    strokes = _run_merge_pass(strokes, min_len=0)

    # --- Absorb convergence stubs ---
    # A convergence stub is a short stroke with one endpoint in a
    # junction cluster and the other end free (e.g. the pointed apex
    # of letter A where a stub extends from the junction to the true
    # geometric tip).  We extend every other stroke converging at that
    # cluster along the stub path so each leg reaches the tip, then
    # remove the stub entirely.
    conv_threshold = 18
    changed = True
    while changed:
        changed = False
        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) < 2 or len(s) >= conv_threshold:
                continue
            sc = _endpoint_cluster(s, False)
            ec = _endpoint_cluster(s, True)

            # One end in a junction cluster, other end free (or same cluster)
            if sc >= 0 and ec < 0:
                cluster_id = sc
                # stub_path: from cluster-end (start) toward free tip (end)
                stub_path = list(s)
            elif ec >= 0 and sc < 0:
                cluster_id = ec
                # stub_path: from cluster-end (end) toward free tip (start)
                stub_path = list(reversed(s))
            elif sc >= 0 and ec >= 0 and sc == ec:
                # Both endpoints in same cluster
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
                if _endpoint_cluster(strokes[sj], False) == cluster_id:
                    others_at_cluster += 1
                if _endpoint_cluster(strokes[sj], True) == cluster_id:
                    others_at_cluster += 1
            if others_at_cluster < 2:
                continue

            # Extend every other stroke at this cluster toward the stub tip.
            # Instead of appending the literal stub skeleton pixels (which
            # create a vertical kink/"nipple"), extrapolate each leg's own
            # direction to the tip's y (or x) level so the path stays smooth.
            stub_tip = stub_path[-1]
            for sj in range(len(strokes)):
                if sj == si:
                    continue
                s2 = strokes[sj]
                at_end = _endpoint_cluster(s2, True) == cluster_id
                at_start = (not at_end) and _endpoint_cluster(s2, False) == cluster_id
                if not at_end and not at_start:
                    continue

                # Get the last few points of the leg before the junction
                # cluster to determine its incoming direction. Skip any
                # points that are inside the junction cluster since they
                # don't reflect the leg's true direction.
                cluster = junction_clusters[cluster_id]
                if at_end:
                    # Walk backward from end, skip junction pixels
                    tail = []
                    for k in range(len(s2) - 1, -1, -1):
                        pt = tuple(s2[k]) if isinstance(s2[k], (list, tuple)) else s2[k]
                        if len(tail) >= 8:
                            break
                        if (int(round(pt[0])), int(round(pt[1]))) not in cluster or not tail:
                            tail.insert(0, pt)
                    leg_end = s2[-1]  # actual last point (in the cluster)
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

                # Use the direction from the pre-junction points toward
                # the junction to extrapolate to the stub tip
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
                    # Extrapolate along the leg's direction, blending to tip
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

    # --- Absorb remaining short stubs into neighboring strokes ---
    # Any stroke shorter than stub_threshold that touches a junction cluster
    # gets appended to the longest stroke sharing that junction.
    stub_threshold = 20
    changed = True
    while changed:
        changed = False
        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) >= stub_threshold:
                continue
            # Check which junction clusters this stub touches
            sc = _endpoint_cluster(s, False)
            ec = _endpoint_cluster(s, True)
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
                    tc_start = _endpoint_cluster(s2, False)
                    tc_end = _endpoint_cluster(s2, True)
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

    # --- Proximity-based stub absorption ---
    # Any remaining short stroke whose endpoint is near a longer stroke's endpoint
    # gets appended to that stroke.
    prox_threshold = 20  # max pixel distance to absorb
    changed = True
    while changed:
        changed = False
        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) >= stub_threshold:
                continue
            # Try each endpoint of the stub
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

    # --- Remove orphaned short stubs ---
    # Any stroke shorter than stub_threshold that has an endpoint at a
    # junction cluster where no other stroke touches is an artifact.
    changed = True
    while changed:
        changed = False
        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) >= stub_threshold:
                continue
            sc = _endpoint_cluster(s, False)
            ec = _endpoint_cluster(s, True)
            # Check if any other stroke shares a junction with this stub
            has_neighbor = False
            for sj in range(len(strokes)):
                if sj == si:
                    continue
                if sc >= 0 and (_endpoint_cluster(strokes[sj], False) == sc or
                                _endpoint_cluster(strokes[sj], True) == sc):
                    has_neighbor = True
                    break
                if ec >= 0 and (_endpoint_cluster(strokes[sj], False) == ec or
                                _endpoint_cluster(strokes[sj], True) == ec):
                    has_neighbor = True
                    break
            if not has_neighbor:
                strokes.pop(si)
                changed = True
                break

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
def api_optimize_stream(font_id):
    """SSE endpoint: streams optimization frames in real time.

    Mirrors the full auto_fit_strokes 4-phase pipeline but yields frames
    after every improvement or at regular intervals so the frontend can
    show exactly what the optimizer is doing.

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

        mask = render_glyph_mask(font_path, char, 224)
        if mask is None:
            yield f"data: {json.dumps({'error': 'no mask'})}\n\n"
            return

        rows, cols = np.where(mask)
        if len(rows) == 0:
            yield f"data: {json.dumps({'error': 'empty mask'})}\n\n"
            return

        glyph_bbox = (float(cols.min()), float(rows.min()),
                      float(cols.max()), float(rows.max()))
        cloud = _make_point_cloud(mask, spacing=3)
        if len(cloud) < 10:
            yield f"data: {json.dumps({'error': 'cloud too small'})}\n\n"
            return

        cloud_tree = cKDTree(cloud)
        n_cloud = len(cloud)
        radius = _adaptive_radius(mask, spacing=3)
        h, w = mask.shape
        dist_map = distance_transform_edt(mask)
        _, snap_indices = distance_transform_edt(~mask, return_indices=True)
        snap_yi, snap_xi = snap_indices[0], snap_indices[1]

        shape_types = [t['shape'] for t in templates]
        bounds, slices = _get_param_bounds(templates)

        x0 = []
        for t in templates:
            x0.extend(t['params'])
        x0 = np.array(x0, dtype=float)

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

        bounds_lo = np.array([b[0] for b in bounds])
        bounds_hi = np.array([b[1] for b in bounds])

        def _clamp(x):
            return np.clip(x, bounds_lo, bounds_hi)

        _t_start = _time.monotonic()
        _TIME_BUDGET = 3600.0  # 1 hour max
        _STALE_THRESHOLD = 0.001
        _STALE_CYCLES = 2
        frame_num = [0]
        last_emit = [0.0]

        def _elapsed():
            return _time.monotonic() - _t_start

        def emit_frame(x, fun, phase=''):
            """Build strokes from params and yield SSE event."""
            x = _clamp(x)
            shapes = _param_vector_to_shapes(x, shape_types, slices, glyph_bbox)
            stroke_list = []
            for pts in shapes:
                pl = [(float(p[0]), float(p[1])) for p in pts]
                pl = _smooth_stroke(pl, sigma=2.0)
                pl = _constrain_to_mask(pl, mask)
                if len(pl) >= 2:
                    stroke_list.append([[round(px, 1), round(py, 1)]
                                        for px, py in pl])
            frame_num[0] += 1
            last_emit[0] = _time.monotonic()
            return json.dumps({
                'frame': frame_num[0],
                'score': round(float(-fun), 4),
                'phase': phase,
                'strokes': stroke_list,
            })

        def emit_raw_frame(stroke_list, score, phase=''):
            """Emit SSE event for raw strokes (not shape-param based)."""
            frame_num[0] += 1
            last_emit[0] = _time.monotonic()
            return json.dumps({
                'frame': frame_num[0],
                'score': round(float(score), 4),
                'phase': phase,
                'strokes': stroke_list,
            })

        # ---- Phase 0: Template + Affine + DiffVG optimisation ----
        affine_strokes_result = None  # (stroke_list, score) if best Phase-0 result

        # Try DiffVG first (gradient-based, typically better)
        diffvg_result = _optimize_diffvg(font_path, char, 224)
        if diffvg_result is not None:
            dv_strokes, dv_score, _, _ = diffvg_result
            if dv_strokes and dv_score > 0:
                affine_strokes_result = (dv_strokes, dv_score)
                yield f"data: {emit_raw_frame(dv_strokes, dv_score, 'diffvg')}\n\n"

                if dv_score >= 0.85:
                    yield f"data: {emit_raw_frame(dv_strokes, dv_score, 'final')}\n\n"
                    yield f"data: {json.dumps({'done': True, 'score': round(dv_score, 4), 'frame': frame_num[0]})}\n\n"
                    return

        # Also try affine
        affine_result = _optimize_affine(font_path, char, 224)
        if affine_result is not None:
            affine_strokes, affine_score, _, _ = affine_result
            affine_stroke_list = [[[round(float(x), 1), round(float(y), 1)] for x, y in s]
                                  for s in affine_strokes if len(s) >= 2]
            if affine_stroke_list:
                # Keep whichever scored better
                if affine_strokes_result and affine_strokes_result[1] >= affine_score:
                    yield f"data: {emit_raw_frame(affine_stroke_list, affine_score, 'affine template (DiffVG better)')}\n\n"
                else:
                    affine_strokes_result = (affine_stroke_list, affine_score)
                    yield f"data: {emit_raw_frame(affine_stroke_list, affine_score, 'affine template')}\n\n"

                best_p0_strokes, best_p0_score = affine_strokes_result
                if best_p0_score >= 0.85:
                    yield f"data: {emit_raw_frame(best_p0_strokes, best_p0_score, 'final')}\n\n"
                    yield f"data: {json.dumps({'done': True, 'score': round(best_p0_score, 4), 'frame': frame_num[0]})}\n\n"
                    return

        # Emit initial frame (template defaults or cached params)
        init_fun = _score_all_strokes(x0, *joint_args)
        yield f"data: {emit_frame(x0, init_fun, 'initial')}\n\n"

        best_x = x0.copy()
        best_fun = init_fun

        def _update_best(x, fun):
            nonlocal best_x, best_fun
            if fun < best_fun:
                best_x = x.copy()
                best_fun = fun

        def _perfect():
            return best_fun <= -0.99

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

            # NM for this shape — emit on improvement
            greedy_best_s = [s_x0.copy()]
            greedy_best_f = [_score_single_shape(s_x0, *s_args)]
            greedy_evals = [0]

            def _greedy_nm_obj(params, _si=si, _start=start, _end=end,
                               _s_args=s_args, _gb_s=greedy_best_s,
                               _gb_f=greedy_best_f, _ge=greedy_evals):
                params = np.clip(params, bounds_lo[_start:_end], bounds_hi[_start:_end])
                val = _score_single_shape(params, *_s_args)
                _ge[0] += 1
                if val < _gb_f[0]:
                    _gb_s[0] = params.copy()
                    _gb_f[0] = val
                return val

            nm_r = minimize(
                _greedy_nm_obj, s_x0, method='Nelder-Mead',
                options={'maxfev': 800, 'xatol': 0.2, 'fatol': 0.002,
                         'adaptive': True},
            )
            best_s = np.clip(nm_r.x, bounds_lo[start:end], bounds_hi[start:end]).copy()
            best_sf = nm_r.fun

            # Quick DE for this shape
            if _elapsed() < _TIME_BUDGET * 0.35:
                try:
                    de_r = differential_evolution(
                        _score_single_shape, bounds=s_bounds, args=s_args,
                        x0=best_s, maxiter=30, popsize=12, tol=0.005,
                        seed=None, polish=False, disp=False,
                    )
                    if de_r.fun < best_sf:
                        best_s = de_r.x.copy()
                        best_sf = de_r.fun
                except Exception:
                    pass

            greedy_x[start:end] = best_s

            # Mark covered points
            pts = SHAPE_FNS[stype](tuple(best_s), glyph_bbox, offset=(0, 0),
                                   n_pts=n_pts_shape)
            if len(pts) > 0:
                xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
                yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
                snapped_x = snap_xi[yi, xi].astype(float)
                snapped_y = snap_yi[yi, xi].astype(float)
                snapped = np.column_stack([snapped_x, snapped_y])
                hits = cloud_tree.query_ball_point(snapped, radius)
                newly_covered = set()
                for lst in hits:
                    newly_covered.update(lst)
                for idx in newly_covered:
                    uncovered_mask[idx] = False

            # Emit frame after each shape is placed
            greedy_fun = _score_all_strokes(greedy_x, *joint_args)
            _update_best(greedy_x, greedy_fun)
            uncov_remaining = int(uncovered_mask.sum())
            yield f"data: {emit_frame(greedy_x, greedy_fun, f'greedy shape {si+1}/{len(templates)} ({stype}) uncov={uncov_remaining}')}\n\n"

        # Compare greedy result with x0
        x0_fun = _score_all_strokes(x0, *joint_args)
        if x0_fun < best_fun:
            best_x = x0.copy()
            best_fun = x0_fun

        class _EarlyStop(Exception):
            pass

        # ---- Repeating NM → DE → NM cycle until stagnation or time limit ----
        stale_count = 0
        cycle_num = 0
        while not _perfect() and _elapsed() < _TIME_BUDGET and stale_count < _STALE_CYCLES:
            cycle_num += 1
            score_at_cycle_start = best_fun

            # -- NM refinement --
            if not _perfect() and _elapsed() < _TIME_BUDGET:
                nm_frames = []
                nm_best = [best_x.copy()]
                nm_best_f = [best_fun]

                def _nm_obj_stream(params, _nb=nm_best, _nbf=nm_best_f, _nf=nm_frames):
                    params = _clamp(params)
                    val = _score_all_strokes(params, *joint_args)
                    now = _time.monotonic()
                    if val < _nbf[0]:
                        _nb[0] = params.copy()
                        _nbf[0] = val
                        _nf.append(emit_frame(params, val,
                                   f'cycle {cycle_num} NM improve'))
                    elif (now - last_emit[0]) > 1.0:
                        _nf.append(emit_frame(_nb[0], _nbf[0],
                                   f'cycle {cycle_num} NM'))
                    return val

                remaining_fev = max(500, int(min(30.0, _TIME_BUDGET - _elapsed()) / 0.0003))
                nm_result = minimize(
                    _nm_obj_stream, best_x, method='Nelder-Mead',
                    options={'maxfev': remaining_fev, 'xatol': 0.2, 'fatol': 0.0005,
                             'adaptive': True},
                )
                _update_best(_clamp(nm_result.x), nm_result.fun)

                for f in nm_frames:
                    yield f"data: {f}\n\n"

            # -- DE global search --
            if not _perfect() and _elapsed() < _TIME_BUDGET:
                nm_x = best_x.copy()
                for i, (lo, hi) in enumerate(bounds):
                    nm_x[i] = np.clip(nm_x[i], lo, hi)

                de_frames = []
                def de_cb(xk, convergence=0, _df=de_frames, _cn=cycle_num):
                    val = _score_all_strokes(xk, *joint_args)
                    _update_best(xk, val)
                    _df.append(emit_frame(best_x, best_fun,
                                     f'cycle {_cn} DE conv={convergence:.3f}'))
                    if _perfect() or _elapsed() >= _TIME_BUDGET:
                        raise _EarlyStop()

                try:
                    de = differential_evolution(
                        _score_all_strokes, bounds=bounds, args=joint_args,
                        x0=nm_x, maxiter=200, popsize=20, tol=0.002,
                        seed=None, mutation=(0.5, 1.0), recombination=0.7,
                        polish=False, disp=False, callback=de_cb)
                    _update_best(de.x, de.fun)
                except _EarlyStop:
                    pass

                for f in de_frames:
                    yield f"data: {f}\n\n"

            # -- NM polish --
            if not _perfect() and _elapsed() < _TIME_BUDGET:
                polish_frames = []
                polish_best = [best_x.copy()]
                polish_best_f = [best_fun]

                def _polish_obj(params, _pb=polish_best, _pbf=polish_best_f,
                                _pf=polish_frames, _cn=cycle_num):
                    params = _clamp(params)
                    val = _score_all_strokes(params, *joint_args)
                    if val < _pbf[0]:
                        _pb[0] = params.copy()
                        _pbf[0] = val
                        _pf.append(emit_frame(params, val,
                                   f'cycle {_cn} polish'))
                    return val

                remaining_fev = max(200, int(min(15.0, _TIME_BUDGET - _elapsed()) / 0.0003))
                nm2 = minimize(
                    _polish_obj, best_x, method='Nelder-Mead',
                    options={'maxfev': remaining_fev, 'xatol': 0.1, 'fatol': 0.0005,
                             'adaptive': True},
                )
                _update_best(_clamp(nm2.x), nm2.fun)

                for f in polish_frames:
                    yield f"data: {f}\n\n"

            # Check for stagnation
            improvement = score_at_cycle_start - best_fun  # positive = improved
            if improvement < _STALE_THRESHOLD:
                stale_count += 1
            else:
                stale_count = 0

            # Cache periodically so progress isn't lost
            current_score = float(-best_fun)
            if cached_score is None or current_score > cached_score:
                _save_cached_params(rel_path, char, best_x, current_score)
                cached_score = current_score

        # Determine stop reason
        if _perfect():
            stop_reason = 'perfect'
        elif stale_count >= _STALE_CYCLES:
            stop_reason = 'converged'
        elif _elapsed() >= _TIME_BUDGET:
            stop_reason = 'time limit'
        else:
            stop_reason = 'done'

        # Final frame — pick best of shape optimisation vs affine
        final_score = float(-best_fun)
        if affine_strokes_result and affine_strokes_result[1] > final_score:
            # Affine was better — emit affine strokes as final
            aff_strokes, aff_score = affine_strokes_result
            yield f"data: {emit_raw_frame(aff_strokes, aff_score, 'final (affine)')}\n\n"
            final_score = aff_score
        else:
            yield f"data: {emit_frame(best_x, best_fun, 'final')}\n\n"

        # Save cache (only for shape-param results)
        if not (affine_strokes_result and affine_strokes_result[1] > float(-best_fun)):
            if cached_score is None or final_score > cached_score:
                _save_cached_params(rel_path, char, best_x, final_score)

        yield f"data: {json.dumps({'done': True, 'score': round(final_score, 4), 'frame': frame_num[0], 'cycles': cycle_num, 'reason': stop_reason, 'elapsed': round(_elapsed(), 1)})}\n\n"

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


@app.route('/api/diffvg/<int:font_id>', methods=['POST'])
def api_diffvg(font_id):
    """Refine strokes using DiffVG gradient-based optimization in Docker/GPU."""
    char = request.args.get('c')
    if not char:
        return jsonify(error="Missing ?c= parameter"), 400

    if _diffvg_docker is None:
        return jsonify(error="DiffVG Docker not available"), 503

    data = request.get_json()
    if not data or 'strokes' not in data:
        return jsonify(error="Missing strokes data"), 400

    strokes = data['strokes']
    if not strokes or not any(len(s) >= 2 for s in strokes):
        return jsonify(error="No valid strokes to refine"), 400

    db = get_db()
    font = db.execute("SELECT file_path FROM fonts WHERE id = ?", (font_id,)).fetchone()
    db.close()
    if not font:
        return jsonify(error="Font not found"), 404

    font_path = resolve_font_path(font['file_path'])

    # Strip lock flags from points (DiffVG expects [x, y] only)
    clean_strokes = [[[p[0], p[1]] for p in s] for s in strokes if len(s) >= 2]

    result = _diffvg_docker.optimize(
        font_path=font_path,
        char=char,
        initial_strokes=clean_strokes,
        canvas_size=224,
        num_iterations=500,
        stroke_width=8.0,
        timeout=300,
    )

    if 'error' in result:
        return jsonify(error=result['error']), 500

    return jsonify(
        strokes=result.get('strokes', []),
        score=result.get('score', 0),
        elapsed=result.get('elapsed', 0),
        iterations=result.get('iterations', 0),
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
