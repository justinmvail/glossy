"""Flask routes for the stroke editor - batch and test routes.

This module contains batch processing and test-related Flask route handlers
for the stroke editor application. It provides endpoints for:

- Running automated tests on font stroke generation
- Viewing and comparing test run history
- Batch stroke generation using skeleton and template methods
- DiffVG optimization integration
- Stroke centering and marker detection utilities

The routes in this module are designed for operations that process multiple
characters at once or perform computationally intensive tasks that don't
require real-time streaming feedback.

Example:
    The routes are automatically registered with the Flask app on import::

        from stroke_routes_batch import *

    Then access endpoints like::

        POST /api/test-run/1        # Run tests for font ID 1
        GET /api/test-history/1     # Get test history for font ID 1
        POST /api/skeleton-batch/1  # Generate skeletons for all chars

Attributes:
    _diffvg (DiffVGDocker | None): Docker client for DiffVG optimization.
        None if the docker module is not available.
"""

import io
import json
import sqlite3
from datetime import datetime

import numpy as np
from flask import jsonify, render_template, request, send_file
from PIL import Image, ImageDraw
from stroke_flask import (
    CHARS,
    DEFAULT_CANVAS_SIZE,
    DEFAULT_STROKE_WIDTH,
    DIFFVG_ITERATIONS,
    DIFFVG_TIMEOUT,
    app,
    ensure_test_tables,
    get_db,
    resolve_font_path,
)
from stroke_rendering import render_glyph_mask
from stroke_templates import NUMPAD_TEMPLATE_VARIANTS

try:
    from docker.diffvg_docker import DiffVGDocker
    _diffvg = DiffVGDocker()
except ImportError:
    _diffvg = None


def _font(fid):
    """Retrieve a font record from the database by ID.

    Args:
        fid: The integer ID of the font to retrieve.

    Returns:
        sqlite3.Row | None: A dictionary-like row containing font data with
        keys 'id', 'file_path', 'name', etc., or None if not found.
    """
    db = get_db()
    f = db.execute("SELECT * FROM fonts WHERE id = ?", (fid,)).fetchone()
    db.close()
    return f


def _get_stroke_funcs():
    """Import stroke processing functions lazily to avoid circular imports.

    This function defers imports from stroke_core until they're actually
    needed, preventing circular dependency issues during module initialization.

    Returns:
        tuple: A tuple containing (skel_strokes, skel_markers, min_strokes, auto_fit)
            functions from the stroke_core module.
    """
    from stroke_core import auto_fit, min_strokes, skel_markers, skel_strokes
    return skel_strokes, skel_markers, min_strokes, auto_fit


@app.route('/api/test-run/<int:fid>', methods=['POST'])
def api_run_tests(fid):
    """Run automated stroke generation tests for all characters of a font.

    Executes the test_letter function for each character in CHARS, computes
    aggregate statistics, and stores the results in the test_runs table.

    Args:
        fid: Font ID from URL path.

    Returns:
        Response: JSON response with test results.

    Request:
        POST /api/test-run/<fid>

        No request body required.

    Response:
        Success (200)::

            {
                "ok": true,
                "chars_tested": 62,
                "chars_ok": 58,
                "avg_score": 0.847
            }

        Error (404)::

            {"error": "Font not found"}

    Note:
        This endpoint creates the test tables if they don't exist.
        Results are stored in the database for later comparison.
    """
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
    """Retrieve test run history for a font.

    Returns a list of recent test runs with aggregate statistics,
    ordered by run date descending.

    Args:
        fid: Font ID from URL path.

    Returns:
        Response: JSON response with test run history.

    Request:
        GET /api/test-history/<fid>?limit=10

    Query Parameters:
        limit (int, optional): Maximum number of runs to return. Defaults to 10.

    Response:
        Success (200)::

            {
                "runs": [
                    {
                        "id": 42,
                        "run_date": "2024-01-15T10:30:00",
                        "chars_tested": 62,
                        "chars_ok": 58,
                        "avg_score": 0.847,
                        "avg_coverage": 0.92,
                        "avg_overshoot": 0.03,
                        "avg_stroke_count": 0.85,
                        "avg_topology": 0.88
                    },
                    ...
                ]
            }
    """
    ensure_test_tables()
    db = get_db()
    runs = db.execute("SELECT id, run_date, chars_tested, chars_ok, avg_score, avg_coverage, avg_overshoot, avg_stroke_count, avg_topology FROM test_runs WHERE font_id = ? ORDER BY run_date DESC LIMIT ?", (fid, request.args.get('limit', 10, type=int))).fetchall()
    db.close()
    return jsonify(runs=[dict(r) for r in runs])


@app.route('/api/test-run-detail/<int:rid>')
def api_test_run_detail(rid):
    """Retrieve detailed results for a specific test run.

    Returns the full test run record including per-character results
    parsed from JSON.

    Args:
        rid: Test run ID from URL path.

    Returns:
        Response: JSON response with detailed test results.

    Request:
        GET /api/test-run-detail/<rid>

    Response:
        Success (200)::

            {
                "id": 42,
                "font_id": 1,
                "run_date": "2024-01-15T10:30:00",
                "chars_tested": 62,
                "chars_ok": 58,
                "avg_score": 0.847,
                "results": [
                    {
                        "char": "A",
                        "status": "ok",
                        "score": 0.92,
                        "coverage": 0.95,
                        "overshoot": 0.02,
                        "stroke_count_score": 1.0,
                        "topology_score": 0.9,
                        "strokes": [[[x, y], ...], ...]
                    },
                    ...
                ]
            }

        Error (404)::

            {"error": "Run not found"}
    """
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
    """Generate a preview image of strokes from a test run.

    Renders the glyph mask with colored stroke overlays for visual
    inspection of test results.

    Args:
        rid: Test run ID from URL path.

    Returns:
        Response: PNG image file.

    Request:
        GET /api/preview-from-run/<rid>?c=A&font_id=1

    Query Parameters:
        c (str): Character to preview. Defaults to 'A'.
        font_id (int, optional): Override font ID. Uses run's font if not provided.

    Response:
        Success (200): PNG image (224x224 pixels) with:
            - Dark background (RGB: 26, 26, 46)
            - Glyph region highlighted (RGB: 60, 60, 80)
            - Colored stroke lines (blue, orange, green, pink cycle)

        Error (404): Plain text "Run not found" or "Font not found"

    Note:
        If no stroke data exists for the character, returns a placeholder
        image with "No stroke data" text.
    """
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
    """Compare two test runs and show per-character score differences.

    Calculates score deltas between runs to identify improvements
    or regressions in stroke generation quality.

    Returns:
        Response: JSON response with comparison data.

    Request:
        GET /api/compare-runs?run1=41&run2=42

        OR

        GET /api/compare-runs?font_id=1

    Query Parameters:
        run1 (int, optional): First (older) test run ID.
        run2 (int, optional): Second (newer) test run ID.
        font_id (int, optional): If run1/run2 not specified, compares the
            two most recent runs for this font.

    Response:
        Success (200)::

            {
                "ok": true,
                "run1_id": 41,
                "run2_id": 42,
                "font_id": 1,
                "old_avg": 0.82,
                "new_avg": 0.85,
                "run1_date": "2024-01-14T10:30:00",
                "run2_date": "2024-01-15T10:30:00",
                "comparisons": [
                    {"char": "A", "old_score": 0.85, "new_score": 0.90, "delta": 0.05},
                    {"char": "B", "old_score": 0.80, "new_score": 0.78, "delta": -0.02},
                    ...
                ]
            }

        Error (400)::

            {"error": "Need at least 2 runs to compare"}
            {"error": "Must specify run1 & run2, or font_id"}

        Error (404)::

            {"error": "Run not found"}

    Note:
        Comparisons are sorted by delta ascending (worst regressions first).
        Characters that failed in either run will have null scores/delta.
    """
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
    """Render the test run comparison page for a font.

    Args:
        fid: Font ID from URL path.

    Returns:
        Response: Rendered HTML template or 404 error.

    Request:
        GET /compare/<fid>

    Response:
        Success (200): Rendered 'compare.html' template with font context.
        Error (404): Plain text "Font not found"
    """
    f = _font(fid)
    return render_template('compare.html', font=f) if f else ("Font not found", 404)


@app.route('/api/center-borders/<int:fid>', methods=['POST'])
def api_center_borders(fid):
    """Center stroke points within the glyph borders using ray casting.

    For each point in the provided strokes, casts rays in 36 directions
    (every 5 degrees) to find the narrowest glyph width, then moves the
    point to the center of that width.

    Args:
        fid: Font ID from URL path.

    Returns:
        Response: JSON response with centered strokes.

    Request:
        POST /api/center-borders/<fid>?c=A

        Body (JSON)::

            {
                "strokes": [
                    [[100, 50], [100, 150]],
                    [[50, 100, 1], [150, 100]]
                ]
            }

        Note: Points can optionally have a third value (1) indicating
        the point is "locked" and should preserve that flag.

    Query Parameters:
        c (str, required): Character to render for border detection.

    Response:
        Success (200)::

            {
                "strokes": [
                    [[102.5, 51.2], [99.8, 149.5]],
                    [[52.1, 100.3, 1], [148.2, 100.1]]
                ]
            }

        Error (400)::

            {"error": "Missing ?c= parameter"}
            {"error": "Missing strokes data"}

        Error (404)::

            {"error": "Font not found"}

        Error (500)::

            {"error": "Could not render glyph"}
    """
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
    """Detect skeleton markers (endpoints and junctions) for a character.

    Uses skeletonization to identify key structural points in the glyph
    that can guide stroke placement.

    Args:
        fid: Font ID from URL path.

    Returns:
        Response: JSON response with detected markers.

    Request:
        POST /api/detect-markers/<fid>?c=A

    Query Parameters:
        c (str, required): Character to analyze.

    Response:
        Success (200)::

            {
                "markers": {
                    "endpoints": [[x1, y1], [x2, y2], ...],
                    "junctions": [[x3, y3], [x4, y4], ...]
                }
            }

        Error (400)::

            {"error": "Missing ?c= parameter"}

        Error (404)::

            {"error": "Font not found"}

        Error (500)::

            {"error": "Could not render glyph"}
    """
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
    """Clear cached shape parameters for a font's characters.

    Removes the shape_params_cache column values to force regeneration
    on next access.

    Args:
        fid: Font ID from URL path.

    Returns:
        Response: JSON response confirming operation.

    Request:
        POST /api/clear-shape-cache/<fid>
        POST /api/clear-shape-cache/<fid>?c=A

    Query Parameters:
        c (str, optional): Specific character to clear. If omitted,
            clears cache for all characters of the font.

    Response:
        Success (200)::

            {"ok": true}
    """
    c, db = request.args.get('c'), get_db()
    db.execute("UPDATE characters SET shape_params_cache = NULL WHERE font_id = ?" + (" AND char = ?" if c else ""), (fid, c) if c else (fid,))
    db.commit()
    db.close()
    return jsonify(ok=True)


@app.route('/api/skeleton/<int:fid>', methods=['POST'])
def api_skeleton(fid):
    """Generate skeleton-based strokes for a single character.

    First attempts auto_fit which uses template matching, then falls back
    to pure skeleton extraction if templates fail.

    Args:
        fid: Font ID from URL path.

    Returns:
        Response: JSON response with generated strokes.

    Request:
        POST /api/skeleton/<fid>?c=A

    Query Parameters:
        c (str, required): Character to process.

    Response:
        Success (200) with markers (from auto_fit)::

            {
                "strokes": [[[x1, y1], [x2, y2], ...], ...],
                "markers": {
                    "endpoints": [...],
                    "junctions": [...]
                }
            }

        Success (200) without markers (from skeleton fallback)::

            {
                "strokes": [[[x1, y1], [x2, y2], ...], ...]
            }

        Error (400)::

            {"error": "Missing ?c= parameter"}

        Error (404)::

            {"error": "Font not found"}

        Error (500)::

            {"error": "Could not render glyph"}
            {"error": "No skeleton found"}
    """
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
    """Generate minimal strokes for all characters of a font in batch.

    Uses template-based stroke generation (min_strokes) for each character
    in CHARS and stores results in the database.

    Args:
        fid: Font ID from URL path.

    Returns:
        Response: JSON response with batch operation results.

    Request:
        POST /api/minimal-strokes-batch/<fid>
        POST /api/minimal-strokes-batch/<fid>?force=true

    Query Parameters:
        force (str, optional): If 'true', regenerates strokes even for
            characters that already have strokes_raw data.

    Response:
        Success (200)::

            {
                "ok": true,
                "generated": 45,
                "skipped": 12,
                "failed": 5
            }

        Error (404)::

            {"error": "Font not found"}

    Note:
        This endpoint adds a 'template_variant' column to the characters
        table if it doesn't exist (for tracking which template was used).
    """
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
    """Generate skeleton-based strokes for all characters in batch.

    For each character, first tries auto_fit (template matching), then
    falls back to pure skeletonization. Results are stored in the database.

    Args:
        fid: Font ID from URL path.

    Returns:
        Response: JSON response with batch operation results.

    Request:
        POST /api/skeleton-batch/<fid>

    Response:
        Success (200)::

            {
                "ok": true,
                "generated": 45,
                "results": {
                    "A": "3 strokes",
                    "B": "2 strokes",
                    "C": "skipped",
                    "D": "no_skeleton",
                    ...
                }
            }

        Error (404)::

            {"error": "Font not found"}

    Note:
        Characters with existing strokes_raw data are skipped.
        The 'results' dict shows status for each character.
    """
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
    """Get available template variants for a character.

    Returns all stroke templates available for a given character,
    including stroke count and template structure for each variant.

    Returns:
        Response: JSON response with template variants.

    Request:
        GET /api/template-variants?c=A

    Query Parameters:
        c (str, required): Character to get variants for.

    Response:
        Success (200)::

            {
                "char": "A",
                "variants": {
                    "default": {
                        "stroke_count": 3,
                        "template": [
                            ["S", "NE"],
                            ["NW", "SE"],
                            ["W", "E"]
                        ]
                    },
                    "alt1": {
                        "stroke_count": 2,
                        "template": [...]
                    }
                }
            }

        Error (400)::

            {"error": "Missing ?c= parameter"}
    """
    c = request.args.get('c')
    if not c:
        return jsonify(error="Missing ?c= parameter"), 400
    v = NUMPAD_TEMPLATE_VARIANTS.get(c, {})
    return jsonify(variants={n: {'stroke_count': len(t), 'template': [[str(wp) for wp in s] for s in t]} for n, t in v.items()}, char=c)


@app.route('/api/minimal-strokes/<int:fid>')
def api_minimal_strokes(fid):
    """Generate minimal strokes for a single character.

    Uses template-based stroke generation with optional variant selection.

    Args:
        fid: Font ID from URL path.

    Returns:
        Response: JSON response with generated strokes.

    Request:
        GET /api/minimal-strokes/<fid>?c=A
        GET /api/minimal-strokes/<fid>?c=A&variant=alt1

    Query Parameters:
        c (str, required): Character to generate strokes for.
        variant (str, optional): Specific template variant to use.
            If omitted, the best variant is automatically selected.

    Response:
        Success (200)::

            {
                "strokes": [[[x1, y1], [x2, y2], ...], ...],
                "variant": "default"
            }

        Error (400)::

            {"error": "Missing ?c= parameter"}
            {"error": "Unknown variant 'foo'", "available": ["default", "alt1"]}
            {"error": "Could not generate strokes for 'X'"}

        Error (404)::

            {"error": "Font not found"}
    """
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
    """Optimize strokes using DiffVG differentiable rendering.

    Uses Docker-containerized DiffVG to refine stroke positions through
    gradient-based optimization against the target glyph.

    Args:
        fid: Font ID from URL path.

    Returns:
        Response: JSON response with optimized strokes.

    Request:
        POST /api/diffvg/<fid>?c=A&thin=0

        Body (JSON, optional)::

            {
                "strokes": [[[x1, y1], [x2, y2], ...], ...]
            }

        If strokes are provided, they are used as initial guess.
        Otherwise, strokes are generated using min_strokes.

    Query Parameters:
        c (str, required): Character to optimize.
        thin (int, optional): Number of thinning iterations. Defaults to 0.

    Response:
        Success (200)::

            {
                "strokes": [[[x1, y1], [x2, y2], ...], ...],
                "score": 0.92,
                "elapsed": 5.3,
                "source": "refined"  // or "generated" if no input strokes
            }

        Error (400)::

            {"error": "Missing ?c= parameter"}
            {"error": "No template available for 'X'"}

        Error (404)::

            {"error": "Font not found"}

        Error (500)::

            {"error": "DiffVG optimization failed: <details>"}

        Error (503)::

            {"error": "DiffVG Docker not available"}
    """
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
