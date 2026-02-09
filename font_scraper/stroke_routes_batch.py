"""Flask routes for the stroke editor - batch and test routes.

This module contains batch processing and test-related Flask route handlers.
"""

import io
import json
import sqlite3
import numpy as np
from datetime import datetime
from flask import render_template, request, jsonify, send_file
from PIL import Image, ImageDraw

from stroke_flask import (
    app, get_db, ensure_test_tables, CHARS, resolve_font_path,
    DEFAULT_CANVAS_SIZE, DEFAULT_STROKE_WIDTH, DIFFVG_ITERATIONS, DIFFVG_TIMEOUT,
)
from stroke_templates import NUMPAD_TEMPLATE_VARIANTS
from stroke_rendering import render_glyph_mask

try:
    from docker.diffvg_docker import DiffVGDocker
    _diffvg = DiffVGDocker()
except ImportError:
    _diffvg = None


def _font(fid):
    """Get font by ID from database."""
    db = get_db()
    f = db.execute("SELECT * FROM fonts WHERE id = ?", (fid,)).fetchone()
    db.close()
    return f


def _get_stroke_funcs():
    """Import stroke processing functions lazily to avoid circular imports."""
    from stroke_core import skel_strokes, skel_markers, min_strokes, auto_fit
    return skel_strokes, skel_markers, min_strokes, auto_fit


@app.route('/api/test-run/<int:fid>', methods=['POST'])
def api_run_tests(fid):
    from test_minimal_strokes import test_letter
    ensure_test_tables()
    f = _font(fid)
    if not f:
        return jsonify(error="Font not found"), 404
    res = [test_letter(resolve_font_path(f['file_path']), c) for c in CHARS]
    ok = [r for r in res if r['status'] == 'ok']
    n = max(len(ok), 1)
    avg = sum(r['score'] for r in ok) / n
    avg_cov = sum(r['coverage'] for r in ok) / n
    avg_over = sum(r['overshoot'] for r in ok) / n
    avg_strokes = sum(r['stroke_count_score'] for r in ok) / n
    avg_topo = sum(r['topology_score'] for r in ok) / n
    db = get_db()
    db.execute("INSERT INTO test_runs (font_id, run_date, chars_tested, chars_ok, avg_score, avg_coverage, avg_overshoot, avg_stroke_count, avg_topology, results_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
               (fid, datetime.now().isoformat(), len(res), len(ok), avg, avg_cov, avg_over, avg_strokes, avg_topo, json.dumps(res)))
    db.commit()
    db.close()
    return jsonify(ok=True, chars_tested=len(res), chars_ok=len(ok), avg_score=round(avg, 3))


@app.route('/api/test-history/<int:fid>')
def api_test_history(fid):
    ensure_test_tables()
    db = get_db()
    runs = db.execute("SELECT id, run_date, chars_tested, chars_ok, avg_score, avg_coverage, avg_overshoot, avg_stroke_count, avg_topology FROM test_runs WHERE font_id = ? ORDER BY run_date DESC LIMIT ?", (fid, request.args.get('limit', 10, type=int))).fetchall()
    db.close()
    return jsonify(runs=[dict(r) for r in runs])


@app.route('/api/test-run-detail/<int:rid>')
def api_test_run_detail(rid):
    ensure_test_tables()
    db = get_db()
    run = db.execute("SELECT * FROM test_runs WHERE id = ?", (rid,)).fetchone()
    db.close()
    if not run:
        return jsonify(error="Run not found"), 404
    r = dict(run)
    if r.get('results_json'):
        r['results'] = json.loads(r['results_json'])
        del r['results_json']
    return jsonify(r)


@app.route('/api/preview-from-run/<int:rid>')
def api_preview_from_run(rid):
    c, fid = request.args.get('c', 'A'), request.args.get('font_id', type=int)
    ensure_test_tables()
    db = get_db()
    run = db.execute("SELECT * FROM test_runs WHERE id = ?", (rid,)).fetchone()
    if not run:
        db.close()
        return "Run not found", 404
    res = json.loads(run['results_json']) if run['results_json'] else []
    cr = next((r for r in res if r.get('char') == c), None)
    if not cr or 'strokes' not in cr:
        db.close()
        img = Image.new('RGB', (224, 224), (26, 26, 46))
        ImageDraw.Draw(img).text((60, 100), "No stroke data", fill=(100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    f = db.execute("SELECT file_path FROM fonts WHERE id = ?", (fid or run['font_id'],)).fetchone()
    db.close()
    if not f:
        return "Font not found", 404
    m = render_glyph_mask(f['file_path'], c, 224)
    img = Image.new('RGB', (224, 224), (26, 26, 46))
    if m is not None:
        arr = np.array(img)
        arr[m] = [60, 60, 80]
        img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    cols = [(126, 184, 247), (247, 184, 126), (184, 247, 126), (247, 126, 184)]
    for i, s in enumerate(cr['strokes']):
        if len(s) >= 2:
            draw.line([(int(p[0]), int(p[1])) for p in s], fill=cols[i % len(cols)], width=3)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/api/compare-runs')
def api_compare_runs():
    ensure_test_tables()
    db = get_db()
    r1, r2, fid = request.args.get('run1', type=int), request.args.get('run2', type=int), request.args.get('font_id', type=int)
    if fid and not (r1 and r2):
        runs = db.execute("SELECT id FROM test_runs WHERE font_id = ? ORDER BY run_date DESC LIMIT 2", (fid,)).fetchall()
        if len(runs) < 2:
            db.close()
            return jsonify(error="Need at least 2 runs to compare"), 400
        r2, r1 = runs[0]['id'], runs[1]['id']
    if not r1 or not r2:
        db.close()
        return jsonify(error="Must specify run1 & run2, or font_id"), 400
    run1, run2 = db.execute("SELECT * FROM test_runs WHERE id = ?", (r1,)).fetchone(), db.execute("SELECT * FROM test_runs WHERE id = ?", (r2,)).fetchone()
    db.close()
    if not run1 or not run2:
        return jsonify(error="Run not found"), 404
    m1 = {r['char']: r for r in (json.loads(run1['results_json']) if run1['results_json'] else [])}
    m2 = {r['char']: r for r in (json.loads(run2['results_json']) if run2['results_json'] else [])}
    cmp = [{'char': c, 'old_score': m1.get(c, {}).get('score') if m1.get(c, {}).get('status') == 'ok' else None,
            'new_score': m2.get(c, {}).get('score') if m2.get(c, {}).get('status') == 'ok' else None,
            'delta': round((m2.get(c, {}).get('score', 0) - m1.get(c, {}).get('score', 0)), 3) if m1.get(c, {}).get('status') == 'ok' and m2.get(c, {}).get('status') == 'ok' else None}
           for c in sorted(set(m1.keys()) | set(m2.keys()))]
    old_avg = run1['avg_score'] if run1 else None
    new_avg = run2['avg_score'] if run2 else None
    font_id = run1['font_id'] if run1 else (run2['font_id'] if run2 else fid)
    run1_date = run1['run_date'] if run1 else None
    run2_date = run2['run_date'] if run2 else None
    return jsonify(ok=True, run1_id=r1, run2_id=r2, font_id=font_id, old_avg=old_avg, new_avg=new_avg, run1_date=run1_date, run2_date=run2_date, comparisons=sorted(cmp, key=lambda x: x['delta'] or 0))


@app.route('/compare/<int:fid>')
def compare_page(fid):
    f = _font(fid)
    return render_template('compare.html', font=f) if f else ("Font not found", 404)


@app.route('/api/center-borders/<int:fid>', methods=['POST'])
def api_center_borders(fid):
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
    dirs = [(np.cos(i * np.pi / 36), np.sin(i * np.pi / 36)) for i in range(36)]
    h, w = m.shape
    def ray(x, y, dx, dy):
        for st in range(1, 300):
            ix, iy = int(round(x + dx * st)), int(round(y + dy * st))
            if ix < 0 or ix >= w or iy < 0 or iy >= h or not m[iy, ix]:
                return st
        return None
    res = []
    for st in data['strokes']:
        cen = []
        for p in st:
            x, y, lk = p[0], p[1], len(p) >= 3 and p[2] == 1
            ix, iy = int(round(min(max(x, 0), w - 1))), int(round(min(max(y, 0), h - 1)))
            if not m[iy, ix]:
                cen.append([p[0], p[1], 1] if lk else [p[0], p[1]])
                continue
            bt, bm = float('inf'), (x, y)
            for dx, dy in dirs:
                dp, dn = ray(x, y, dx, dy), ray(x, y, -dx, -dy)
                if dp and dn and dp + dn < bt:
                    bt, bm = dp + dn, (x + dx * (dp - dn) / 2.0, y + dy * (dp - dn) / 2.0)
            cen.append([bm[0], bm[1], 1] if lk else [bm[0], bm[1]])
        res.append(cen)
    return jsonify(strokes=res)


@app.route('/api/detect-markers/<int:fid>', methods=['POST'])
def api_detect_markers(fid):
    _, skel_markers, _, _ = _get_stroke_funcs()
    c = request.args.get('c')
    if not c:
        return jsonify(error="Missing ?c= parameter"), 400
    f = _font(fid)
    if not f:
        return jsonify(error="Font not found"), 404
    m = render_glyph_mask(f['file_path'], c)
    return jsonify(markers=skel_markers(m)) if m is not None else (jsonify(error="Could not render glyph"), 500)


@app.route('/api/clear-shape-cache/<int:fid>', methods=['POST'])
def api_clear_shape_cache(fid):
    c, db = request.args.get('c'), get_db()
    db.execute("UPDATE characters SET shape_params_cache = NULL WHERE font_id = ?" + (" AND char = ?" if c else ""), (fid, c) if c else (fid,))
    db.commit()
    db.close()
    return jsonify(ok=True)


@app.route('/api/skeleton/<int:fid>', methods=['POST'])
def api_skeleton(fid):
    skel_strokes, _, _, auto_fit = _get_stroke_funcs()
    c = request.args.get('c')
    if not c:
        return jsonify(error="Missing ?c= parameter"), 400
    f = _font(fid)
    if not f:
        return jsonify(error="Font not found"), 404
    r = auto_fit(f['file_path'], c, ret_mark=True)
    if r and r[0]:
        return jsonify(strokes=r[0], markers=r[1])
    m = render_glyph_mask(f['file_path'], c)
    if m is None:
        return jsonify(error="Could not render glyph"), 500
    st = skel_strokes(m)
    return jsonify(strokes=st) if st else (jsonify(error="No skeleton found"), 500)


@app.route('/api/minimal-strokes-batch/<int:fid>', methods=['POST'])
def api_minimal_strokes_batch(fid):
    _, _, min_strokes, _ = _get_stroke_funcs()
    db = get_db()
    f = db.execute("SELECT * FROM fonts WHERE id = ?", (fid,)).fetchone()
    if not f:
        db.close()
        return jsonify(error="Font not found"), 404
    try:
        db.execute("ALTER TABLE characters ADD COLUMN template_variant TEXT")
        db.commit()
    except sqlite3.OperationalError:
        pass
    gen, skp, fail, force = 0, 0, 0, request.args.get('force', '').lower() == 'true'
    for c in CHARS:
        if not force and db.execute("SELECT id FROM characters WHERE font_id = ? AND char = ? AND strokes_raw IS NOT NULL", (fid, c)).fetchone():
            skp += 1
            continue
        st, var = min_strokes(f['file_path'], c, ret_var=True)
        if not st:
            fail += 1
            continue
        row = db.execute("SELECT id FROM characters WHERE font_id = ? AND char = ?", (fid, c)).fetchone()
        if row:
            db.execute("UPDATE characters SET strokes_raw = ?, template_variant = ? WHERE id = ?", (json.dumps(st), var, row['id']))
        else:
            db.execute("INSERT INTO characters (font_id, char, strokes_raw, template_variant) VALUES (?, ?, ?, ?)", (fid, c, json.dumps(st), var))
        gen += 1
    db.commit()
    db.close()
    return jsonify(ok=True, generated=gen, skipped=skp, failed=fail)


@app.route('/api/skeleton-batch/<int:fid>', methods=['POST'])
def api_skeleton_batch(fid):
    skel_strokes, _, _, auto_fit = _get_stroke_funcs()
    db = get_db()
    f = db.execute("SELECT * FROM fonts WHERE id = ?", (fid,)).fetchone()
    if not f:
        db.close()
        return jsonify(error="Font not found"), 404
    res = {}
    for c in CHARS:
        if db.execute("SELECT id FROM characters WHERE font_id = ? AND char = ? AND strokes_raw IS NOT NULL", (fid, c)).fetchone():
            res[c] = 'skipped'
            continue
        st = auto_fit(f['file_path'], c)
        if not st:
            m = render_glyph_mask(f['file_path'], c)
            st = skel_strokes(m) if m is not None else None
        if not st:
            res[c] = 'no_skeleton'
            continue
        row = db.execute("SELECT id FROM characters WHERE font_id = ? AND char = ?", (fid, c)).fetchone()
        if row:
            db.execute("UPDATE characters SET strokes_raw = ?, point_count = ? WHERE font_id = ? AND char = ?", (json.dumps(st), sum(len(s) for s in st), fid, c))
        else:
            db.execute("INSERT INTO characters (font_id, char, strokes_raw, point_count) VALUES (?, ?, ?, ?)", (fid, c, json.dumps(st), sum(len(s) for s in st)))
        res[c] = f'{len(st)} strokes'
    db.commit()
    db.close()
    return jsonify(ok=True, generated=sum(1 for v in res.values() if 'strokes' in v), results=res)


@app.route('/api/template-variants')
def api_template_variants():
    c = request.args.get('c')
    if not c:
        return jsonify(error="Missing ?c= parameter"), 400
    v = NUMPAD_TEMPLATE_VARIANTS.get(c, {})
    return jsonify(variants={n: {'stroke_count': len(t), 'template': [[str(wp) for wp in s] for s in t]} for n, t in v.items()}, char=c)


@app.route('/api/minimal-strokes/<int:fid>')
def api_minimal_strokes(fid):
    _, _, min_strokes, _ = _get_stroke_funcs()
    c, var = request.args.get('c'), request.args.get('variant')
    if not c:
        return jsonify(error="Missing ?c= parameter"), 400
    f = _font(fid)
    if not f:
        return jsonify(error="Font not found"), 404
    fp = resolve_font_path(f['file_path'])
    if var:
        vs = NUMPAD_TEMPLATE_VARIANTS.get(c, {})
        if var not in vs:
            return jsonify(error=f"Unknown variant '{var}'", available=list(vs.keys())), 400
        st = min_strokes(fp, c, tpl=vs[var])
        return jsonify(strokes=st, variant=var) if st else (jsonify(error="Could not generate strokes"), 400)
    st, uv = min_strokes(fp, c, ret_var=True)
    return jsonify(strokes=st, variant=uv) if st else (jsonify(error=f"Could not generate strokes for '{c}'"), 400)


@app.route('/api/diffvg/<int:fid>', methods=['POST'])
def api_diffvg(fid):
    _, _, min_strokes, _ = _get_stroke_funcs()
    c = request.args.get('c')
    if not c:
        return jsonify(error="Missing ?c= parameter"), 400
    if _diffvg is None:
        return jsonify(error="DiffVG Docker not available"), 503
    f = _font(fid)
    if not f:
        return jsonify(error="Font not found"), 404
    fp, data = resolve_font_path(f['file_path']), request.get_json() or {}
    st = [s for s in data.get('strokes', []) if len(s) >= 2]
    if st:
        cst, src = [[[p[0], p[1]] for p in s] for s in st], 'refined'
    else:
        cst = min_strokes(fp, c)
        if not cst:
            return jsonify(error=f"No template available for '{c}'"), 400
        src = 'generated'
    r = _diffvg.optimize(font_path=fp, char=c, initial_strokes=cst, canvas_size=DEFAULT_CANVAS_SIZE,
                          num_iterations=DIFFVG_ITERATIONS, stroke_width=DEFAULT_STROKE_WIDTH,
                          thin_iterations=int(request.args.get('thin', 0)), timeout=DIFFVG_TIMEOUT)
    return (jsonify(error=r['error']), 500) if 'error' in r else jsonify(strokes=r.get('strokes', []), score=r.get('score', 0), elapsed=r.get('elapsed', 0), source=src)
