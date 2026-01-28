#!/usr/bin/env python3
"""Stroke Editor - Web app for viewing and editing InkSight stroke data."""

import sqlite3
import json
import io
import base64
import os
import numpy as np
from scipy.ndimage import distance_transform_edt
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
    fonts = db.execute("""
        SELECT f.id, f.name, f.source, f.file_path,
               COUNT(c.id) as char_count
        FROM fonts f
        JOIN characters c ON c.font_id = f.id
        WHERE c.strokes_raw IS NOT NULL
        GROUP BY f.id
        ORDER BY f.name
    """).fetchall()
    db.close()
    return render_template('font_list.html', fonts=fonts)


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

    # Get all chars for this font (for prev/next navigation)
    all_chars = db.execute("""
        SELECT char FROM characters
        WHERE font_id = ? AND strokes_raw IS NOT NULL
        ORDER BY char
    """, (font_id,)).fetchall()
    char_list = [r['char'] for r in all_chars]
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
        SELECT strokes_raw FROM characters
        WHERE font_id = ? AND char = ?
    """, (font_id, char)).fetchone()
    db.close()

    if not row:
        return jsonify(error="Character not found"), 404

    strokes = json.loads(row['strokes_raw']) if row['strokes_raw'] else []

    # Render font character image
    img_bytes = render_char_image(font['file_path'], char)
    img_b64 = None
    if img_bytes:
        img_b64 = base64.b64encode(img_bytes).decode('ascii')

    return jsonify(strokes=strokes, image=img_b64)


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
    total_points = sum(len(s) for s in strokes)

    db = get_db()
    db.execute("""
        UPDATE characters
        SET strokes_raw = ?, point_count = ?
        WHERE font_id = ? AND char = ?
    """, (json.dumps(strokes), total_points, font_id, char))
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

    # Convert to numpy arrays
    np_strokes = [np.array(s, dtype=float) for s in data['strokes']]

    # Optionally smooth (scale sigma down for short strokes to avoid over-smoothing)
    if smooth:
        smoothed = []
        for s in np_strokes:
            if len(s) < 3:
                smoothed.append(s)
            else:
                # Taper sigma for strokes with fewer points
                effective_sigma = smooth_sigma * min(1.0, (len(s) - 2) / 30.0)
                smoothed.append(InkSightVectorizer.smooth_gaussian(s, sigma=effective_sigma))
        np_strokes = smoothed

    # Extend to connect (optional)
    if data.get('connect', True):
        vectorizer = InkSightVectorizer()
        np_strokes = vectorizer.extend_to_connect(np_strokes, max_extension=max_ext)

    # Convert back to lists
    result = [s.tolist() for s in np_strokes]
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
    dist, indices = distance_transform_edt(outside, return_indices=True)

    canvas_size = mask.shape[0]
    result = []
    for stroke in data['strokes']:
        snapped = []
        for p in stroke:
            x, y = p[0], p[1]
            # Clamp to canvas bounds
            ix = int(round(min(max(x, 0), canvas_size - 1)))
            iy = int(round(min(max(y, 0), canvas_size - 1)))

            if mask[iy, ix]:
                # Already inside glyph
                snapped.append([x, y])
            else:
                # Snap to nearest glyph pixel
                ny = float(indices[0, iy, ix])
                nx = float(indices[1, iy, ix])
                snapped.append([nx, ny])
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

    result = []
    for stroke in data['strokes']:
        centered = []
        for p in stroke:
            x, y = p[0], p[1]
            ix, iy = int(round(x)), int(round(y))

            # If point is outside glyph, keep it as-is
            if ix < 0 or ix >= mask.shape[1] or iy < 0 or iy >= mask.shape[0] or not mask[iy, ix]:
                centered.append([x, y])
                continue

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

            centered.append([best_mid[0], best_mid[1]])
        result.append(centered)

    return jsonify(strokes=result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
