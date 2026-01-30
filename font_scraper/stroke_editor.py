#!/usr/bin/env python3
"""Stroke Editor - Web app for viewing and editing InkSight stroke data."""

import sqlite3
import json
import io
import base64
import os
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from collections import defaultdict
from urllib.parse import quote as urlquote
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image, ImageDraw, ImageFont
from inksight_vectorizer import InkSightVectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
DB_PATH = os.path.join(BASE_DIR, 'fonts.db')


@app.template_filter('urlencode')
def urlencode_filter(s):
    return urlquote(str(s), safe='')


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

            # Extend every other stroke at this cluster along the stub
            for sj in range(len(strokes)):
                if sj == si:
                    continue
                s2 = strokes[sj]
                if _endpoint_cluster(s2, True) == cluster_id:
                    for k in range(1, len(stub_path)):
                        s2.append(stub_path[k])
                elif _endpoint_cluster(s2, False) == cluster_id:
                    rev = list(reversed(stub_path))
                    for k in range(1, len(rev)):
                        s2.insert(k - 1, rev[k])

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


@app.route('/api/skeleton/<int:font_id>', methods=['POST'])
def api_skeleton(font_id):
    """Generate strokes from font glyph via skeletonization."""
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

    strokes = skeleton_to_strokes(mask, min_stroke_len=5)
    if not strokes:
        return jsonify(error="No skeleton found"), 500

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

        mask = render_glyph_mask(font['file_path'], char)
        if mask is None:
            results[char] = 'no_glyph'
            continue

        strokes = skeleton_to_strokes(mask, min_stroke_len=5)
        if not strokes:
            results[char] = 'no_skeleton'
            continue

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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
