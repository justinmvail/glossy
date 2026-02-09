"""Flask routes for the stroke editor - core routes.

This module contains core Flask route handlers for the stroke editor web app.
"""

import io
import json
import base64
import numpy as np
from flask import render_template, request, jsonify, send_file
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from skimage.morphology import thin

from stroke_flask import (
    app, get_db, CHARS, STROKE_COLORS, resolve_font_path, validate_char_param,
)
from stroke_rendering import (
    render_char_image, render_glyph_mask, check_case_mismatch,
    render_text_for_analysis, analyze_shape_metrics, check_char_holes,
    check_char_shape_count,
)


def _font(fid):
    """Get font by ID from database."""
    db = get_db()
    f = db.execute("SELECT * FROM fonts WHERE id = ?", (fid,)).fetchone()
    db.close()
    return f


@app.route('/')
def font_list():
    db, rej = get_db(), request.args.get('rejected') == '1'
    q = "SELECT f.id, f.name, f.source, f.file_path, COALESCE(cs.char_count, 0) as char_count, {} as rejected FROM fonts f {} LEFT JOIN (SELECT font_id, COUNT(*) as char_count FROM characters WHERE strokes_raw IS NOT NULL GROUP BY font_id) cs ON cs.font_id = f.id {} ORDER BY f.name"
    fonts = db.execute(q.format('1', 'JOIN font_removals fr ON fr.font_id = f.id AND fr.reason_id = 8', '') if rej else q.format('0', 'LEFT JOIN font_removals rej ON rej.font_id = f.id AND rej.reason_id = 8 LEFT JOIN font_removals dup ON dup.font_id = f.id AND dup.reason_id = 2', 'WHERE rej.id IS NULL AND dup.id IS NULL')).fetchall()
    db.close()
    return render_template('font_list.html', fonts=fonts, show_rejected=rej)


@app.route('/font/<int:fid>')
def char_grid(fid):
    db = get_db()
    f = db.execute("SELECT * FROM fonts WHERE id = ?", (fid,)).fetchone()
    if not f:
        db.close()
        return "Font not found", 404
    ch = db.execute("SELECT char, strokes_raw, point_count FROM characters WHERE font_id = ? AND strokes_raw IS NOT NULL ORDER BY char", (fid,)).fetchall()
    db.close()
    return render_template('char_grid.html', font=f, chars=ch if ch else [{'char': c, 'strokes_raw': None, 'point_count': 0} for c in CHARS])


@app.route('/edit/<int:fid>')
def edit_char(fid):
    c = request.args.get('c')
    if not c:
        return "Missing character parameter ?c=", 400
    f = _font(fid)
    return render_template('editor.html', font=f, char=c, char_list=CHARS) if f else ("Font not found", 404)


@app.route('/api/char/<int:fid>')
def api_get_char(fid):
    c = request.args.get('c')
    ok, err = validate_char_param(c)
    if not ok:
        return err
    db = get_db()
    f = db.execute("SELECT * FROM fonts WHERE id = ?", (fid,)).fetchone()
    if not f:
        db.close()
        return jsonify(error="Font not found"), 404
    row = db.execute("SELECT strokes_raw, markers FROM characters WHERE font_id = ? AND char = ?", (fid, c)).fetchone()
    db.close()
    img = render_char_image(f['file_path'], c)
    return jsonify(
        strokes=json.loads(row['strokes_raw']) if row and row['strokes_raw'] else [],
        markers=json.loads(row['markers']) if row and row['markers'] else [],
        image=base64.b64encode(img).decode('ascii') if img else None
    )


@app.route('/api/char/<int:fid>', methods=['POST'])
def api_save_char(fid):
    c, data = request.args.get('c'), request.get_json()
    ok, err = validate_char_param(c)
    if not ok:
        return err
    if not data or 'strokes' not in data:
        return jsonify(error="Missing strokes data"), 400
    st, mk = data['strokes'], data.get('markers', [])
    db = get_db()
    if db.execute("SELECT id FROM characters WHERE font_id = ? AND char = ?", (fid, c)).fetchone():
        db.execute("UPDATE characters SET strokes_raw = ?, point_count = ?, markers = ? WHERE font_id = ? AND char = ?",
                   (json.dumps(st), sum(len(s) for s in st), json.dumps(mk) if mk else None, fid, c))
    else:
        db.execute("INSERT INTO characters (font_id, char, strokes_raw, point_count, markers) VALUES (?, ?, ?, ?, ?)",
                   (fid, c, json.dumps(st), sum(len(s) for s in st), json.dumps(mk) if mk else None))
    db.commit()
    db.close()
    return jsonify(ok=True)


@app.route('/api/render/<int:fid>')
def api_render(fid):
    c = request.args.get('c')
    if not c:
        return "Missing ?c= parameter", 400
    f = _font(fid)
    if not f:
        return "Font not found", 404
    img = render_char_image(f['file_path'], c)
    return send_file(io.BytesIO(img), mimetype='image/png') if img else ("Could not render", 500)


@app.route('/api/thin-preview/<int:fid>')
def api_thin_preview(fid):
    c = request.args.get('c')
    if not c:
        return "Missing ?c= parameter", 400
    f = _font(fid)
    if not f:
        return "Font not found", 404
    m = render_glyph_mask(f['file_path'], c)
    if m is None:
        return "Could not render glyph", 500
    th = thin(m, max_num_iter=int(request.args.get('thin', 5)))
    img = np.full((224, 224, 3), 255, dtype=np.uint8)
    img[m], img[th] = [200, 200, 200], [0, 0, 0]
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/api/check-connected/<int:fid>')
def api_check_connected(fid):
    f = _font(fid)
    if not f:
        return jsonify(error="Font not found"), 404
    fp = resolve_font_path(f['file_path'])
    try:
        pf = ImageFont.truetype(fp, 60)
        arr = render_text_for_analysis(pf, "Hello World")
        if arr is None:
            return jsonify(error="Could not render"), 500
        ns, mw = analyze_shape_metrics(arr, arr.shape[1])
        bad = ns < 10 or ns > 15 or mw > 0.225 or check_char_holes(pf, 'l') or not check_char_shape_count(pf, '!', 2)
        return jsonify(shapes=int(ns), bad=bool(bad), case_mismatches=check_case_mismatch(fp))
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/api/reject-connected', methods=['POST'])
def api_reject_connected():
    db = get_db()
    fonts = db.execute("SELECT f.id, f.file_path FROM fonts f LEFT JOIN font_removals fr ON fr.font_id = f.id AND fr.reason_id = 8 WHERE fr.id IS NULL").fetchall()
    rej, chk = 0, 0
    for f in fonts:
        try:
            pf = ImageFont.truetype(resolve_font_path(f['file_path']), 60)
            arr = render_text_for_analysis(pf, "Hello World")
            if arr is None:
                continue
            ns, mw = analyze_shape_metrics(arr, arr.shape[1])
            chk += 1
            if ns < 10 or ns > 15 or mw > 0.225 or check_char_holes(pf, 'l') or not check_char_shape_count(pf, '!', 2) or check_case_mismatch(resolve_font_path(f['file_path'])):
                db.execute("INSERT INTO font_removals (font_id, reason_id, details) VALUES (?, 8, ?)", (f['id'], f'{ns} shapes'))
                rej += 1
        except (OSError, ValueError, MemoryError):
            continue
    db.commit()
    db.close()
    return jsonify(ok=True, checked=chk, rejected=rej)


@app.route('/api/font-sample/<int:fid>')
def api_font_sample(fid):
    txt, h = request.args.get('text', 'Hello World!'), int(request.args.get('h', 40))
    f = _font(fid)
    if not f:
        return "Font not found", 404
    try:
        pf = ImageFont.truetype(resolve_font_path(f['file_path']), int(h * 0.85))
        bb = pf.getbbox(txt)
        if not bb:
            return "Could not render text", 500
        img = Image.new('RGBA', (bb[2] - bb[0] + 10, h), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((5 - bb[0], (h - (bb[3] - bb[1])) // 2 - bb[1]), txt, fill=(255, 255, 255, 255), font=pf)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return f"Error: {e}", 500


@app.route('/api/preview/<int:fid>')
def api_preview(fid):
    c = request.args.get('c')
    if not c:
        return "Missing ?c= parameter", 400
    db = get_db()
    f = db.execute("SELECT file_path FROM fonts WHERE id = ?", (fid,)).fetchone()
    row = db.execute("SELECT strokes_raw FROM characters WHERE font_id = ? AND char = ?", (fid, c)).fetchone()
    db.close()
    if not f:
        return "Font not found", 404
    img = render_char_image(f['file_path'], c)
    if not img:
        return "Could not render", 500
    arr = np.array(Image.open(io.BytesIO(img)).convert('L'))
    rgba = np.full((*arr.shape, 4), 255, dtype=np.uint8)
    gm = arr < 200
    rgba[gm, :3], rgba[gm, 3] = arr[gm, None], 60
    bg = Image.fromarray(rgba, 'RGBA')
    draw = ImageDraw.Draw(bg)
    if row and row['strokes_raw']:
        for i, s in enumerate(json.loads(row['strokes_raw'])):
            if len(s) >= 2:
                draw.line([(p[0], p[1]) for p in s], fill=STROKE_COLORS[i % len(STROKE_COLORS)] + (255,), width=2)
    buf = io.BytesIO()
    bg.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/api/process/<int:fid>', methods=['POST'])
def api_process(fid):
    from inksight_vectorizer import InkSightVectorizer
    c, data = request.args.get('c'), request.get_json()
    if not c:
        return jsonify(error="Missing ?c= parameter"), 400
    if not data or 'strokes' not in data:
        return jsonify(error="Missing strokes data"), 400
    lf = [[len(p) >= 3 and p[2] == 1 for p in s] for s in data['strokes']]
    nps = [np.array([[p[0], p[1]] for p in s], dtype=float) for s in data['strokes']]
    if data.get('smooth'):
        sig = data.get('smooth_sigma', 1.5)
        nps = [InkSightVectorizer.smooth_gaussian(s, sigma=sig * min(1.0, (len(s) - 2) / 30.0)) if len(s) >= 3 else s for s in nps]
    if data.get('connect', True):
        nps = InkSightVectorizer().extend_to_connect(nps, max_extension=data.get('max_extension', 8.0))
    return jsonify(strokes=[[[pt[0], pt[1], 1] if pi < len(lf[si]) and lf[si][pi] else [pt[0], pt[1]] for pi, pt in enumerate(s.tolist())] for si, s in enumerate(nps)])


@app.route('/api/snap/<int:fid>', methods=['POST'])
def api_snap(fid):
    c, data = request.args.get('c'), request.get_json()
    if not c:
        return jsonify(error="Missing ?c= parameter"), 400
    if not data or 'strokes' not in data:
        return jsonify(error="Missing strokes data"), 400
    f = _font(fid)
    if not f:
        return jsonify(error="Font not found"), 404
    m = render_glyph_mask(f['file_path'], c)
    if m is None:
        return jsonify(error="Could not render glyph"), 500
    _, idx = distance_transform_edt(~m, return_indices=True)
    h, w = m.shape
    res = []
    for st in data['strokes']:
        sn = []
        for p in st:
            x, y, lk = p[0], p[1], len(p) >= 3 and p[2] == 1
            ix, iy = int(round(min(max(x, 0), w - 1))), int(round(min(max(y, 0), h - 1)))
            sn.append(p[:] if m[iy, ix] else ([float(idx[1, iy, ix]), float(idx[0, iy, ix]), 1] if lk else [float(idx[1, iy, ix]), float(idx[0, iy, ix])]))
        res.append(sn)
    return jsonify(strokes=res)


@app.route('/api/center/<int:fid>', methods=['POST'])
def api_center(fid):
    c, data = request.args.get('c'), request.get_json()
    if not c:
        return jsonify(error="Missing ?c= parameter"), 400
    if not data or 'strokes' not in data:
        return jsonify(error="Missing strokes data"), 400
    f = _font(fid)
    if not f:
        return jsonify(error="Font not found"), 404
    m = render_glyph_mask(f['file_path'], c)
    if m is None:
        return jsonify(error="Could not render glyph"), 500
    rows, cols = np.where(m)
    if len(rows) == 0:
        return jsonify(error="Empty glyph"), 500
    gcx, gcy = (cols.min() + cols.max()) / 2.0, (rows.min() + rows.max()) / 2.0
    ax, ay = [p[0] for s in data['strokes'] for p in s], [p[1] for s in data['strokes'] for p in s]
    if not ax:
        return jsonify(strokes=data['strokes'])
    dx, dy = gcx - (min(ax) + max(ax)) / 2.0, gcy - (min(ay) + max(ay)) / 2.0
    return jsonify(strokes=[[[p[0] + dx, p[1] + dy] for p in st] for st in data['strokes']])


@app.route('/api/reject/<int:fid>', methods=['POST'])
def api_reject_font(fid):
    db = get_db()
    if not db.execute("SELECT id FROM fonts WHERE id = ?", (fid,)).fetchone():
        db.close()
        return jsonify(error="Font not found"), 404
    if db.execute("SELECT id FROM font_removals WHERE font_id = ? AND reason_id = 8", (fid,)).fetchone():
        db.close()
        return jsonify(ok=True, status='already_rejected')
    db.execute("INSERT INTO font_removals (font_id, reason_id, details) VALUES (?, 8, 'Rejected in stroke editor')", (fid,))
    db.commit()
    db.close()
    return jsonify(ok=True, status='rejected')


@app.route('/api/unreject/<int:fid>', methods=['POST'])
def api_unreject_font(fid):
    db = get_db()
    db.execute("DELETE FROM font_removals WHERE font_id = ? AND reason_id = 8", (fid,))
    db.commit()
    db.close()
    return jsonify(ok=True, status='unrejected')


@app.route('/api/unreject-all', methods=['POST'])
def api_unreject_all():
    db = get_db()
    r = db.execute("DELETE FROM font_removals WHERE reason_id = 8")
    db.commit()
    db.close()
    return jsonify(ok=True, restored=r.rowcount)
