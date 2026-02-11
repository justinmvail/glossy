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
from typing import Any

import numpy as np
from flask import Response, jsonify, render_template, request, send_file
from PIL import Image, ImageDraw

from stroke_flask import (
    CHARS,
    DEFAULT_CANVAS_SIZE,
    DEFAULT_STROKE_WIDTH,
    DIFFVG_ITERATIONS,
    DIFFVG_TIMEOUT,
    STROKE_COLORS,
    app,
    ensure_test_tables,
    get_db,
    get_db_context,
    get_font,
    get_font_and_mask,
    get_font_or_error,
    resolve_font_path,
    send_pil_image_as_png,
    test_run_repository,
)
from stroke_rendering import render_glyph_mask
from stroke_templates import NUMPAD_TEMPLATE_VARIANTS

# Lazy-loaded DiffVG instance (avoids import overhead if not used)
_diffvg = None
_diffvg_initialized = False


def get_diffvg() -> Any:
    """Get the DiffVG Docker instance, lazily initializing on first use.

    Returns:
        DiffVGDocker instance, or None if unavailable.
    """
    global _diffvg, _diffvg_initialized
    if not _diffvg_initialized:
        try:
            from docker.diffvg_docker import DiffVGDocker
            _diffvg = DiffVGDocker()
        except ImportError:
            _diffvg = None
        _diffvg_initialized = True
    return _diffvg


# Alias for backward compatibility
_font = get_font


def _get_stroke_funcs() -> tuple:
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
def api_run_tests(fid: int) -> Response | tuple[str, int]:
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
    test_run_repository.save_run(
        fid, datetime.now().isoformat(), len(res), len(ok),
        avg, avg_cov, avg_over, avg_strokes, avg_topo, json.dumps(res)
    )
    return jsonify(ok=True, chars_tested=len(res), chars_ok=len(ok), avg_score=round(avg, 3))


@app.route('/api/test-history/<int:fid>')
def api_test_history(fid: int) -> Response | tuple[str, int]:
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
    limit = request.args.get('limit', 10, type=int)
    if limit is None or limit < 1 or limit > 100:
        limit = 10
    runs = test_run_repository.get_history(fid, limit)
    return jsonify(runs=[dict(r) for r in runs])


@app.route('/api/test-run-detail/<int:rid>')
def api_test_run_detail(rid: int) -> Response | tuple[str, int]:
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
    run = test_run_repository.get_run(rid)
    if not run:
        return jsonify(error="Run not found"), 404
    r = dict(run)
    if r.get('results_json'):
        r['results'] = json.loads(r['results_json'])
        del r['results_json']
    return jsonify(r)


@app.route('/api/preview-from-run/<int:rid>')
def api_preview_from_run(rid: int) -> Response | tuple[str, int]:
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
    run = test_run_repository.get_run(rid)
    if not run:
        return "Run not found", 404
    res = json.loads(run['results_json']) if run['results_json'] else []
    cr = next((r for r in res if r.get('char') == c), None)
    if not cr or 'strokes' not in cr:
        img = Image.new('RGB', (224, 224), (26, 26, 46))
        ImageDraw.Draw(img).text((60, 100), "No stroke data", fill=(100, 100, 100))
        return send_pil_image_as_png(img)
    from stroke_flask import font_repository
    f = font_repository.get_font_by_id(fid or run['font_id'])
    if not f:
        return "Font not found", 404
    m = render_glyph_mask(f['file_path'], c, 224)
    img = Image.new('RGB', (224, 224), (26, 26, 46))
    if m is not None:
        arr = np.array(img)
        arr[m] = [60, 60, 80]
        img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    for i, s in enumerate(cr['strokes']):
        if len(s) >= 2:
            color = STROKE_COLORS[i % len(STROKE_COLORS)]
            draw.line([(int(p[0]), int(p[1])) for p in s], fill=color, width=3)
    return send_pil_image_as_png(img)


@app.route('/api/compare-runs')
def api_compare_runs() -> Response:
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
    r1, r2, fid = request.args.get('run1', type=int), request.args.get('run2', type=int), request.args.get('font_id', type=int)
    if fid and not (r1 and r2):
        recent = test_run_repository.get_recent_runs(fid, 2)
        if len(recent) < 2:
            return jsonify(error="Need at least 2 runs to compare"), 400
        r2, r1 = recent[0], recent[1]
    if not r1 or not r2:
        return jsonify(error="Must specify run1 & run2, or font_id"), 400
    run1, run2 = test_run_repository.get_run(r1), test_run_repository.get_run(r2)
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
def compare_page(fid: int) -> str | tuple[str, int]:
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


def _cast_ray(mask: np.ndarray, x: float, y: float, dx: float, dy: float,
               max_dist: int = 300) -> int | None:
    """Cast a ray from a point in a direction until hitting mask boundary.

    Args:
        mask: Binary mask where True = inside glyph.
        x: Starting x coordinate.
        y: Starting y coordinate.
        dx: X component of direction (normalized).
        dy: Y component of direction (normalized).
        max_dist: Maximum distance to search.

    Returns:
        Distance to boundary, or None if max_dist reached.
    """
    h, w = mask.shape
    for dist in range(1, max_dist):
        ix = int(round(x + dx * dist))
        iy = int(round(y + dy * dist))
        if ix < 0 or ix >= w or iy < 0 or iy >= h or not mask[iy, ix]:
            return dist
    return None


def _center_point_in_glyph(x: float, y: float, mask: np.ndarray,
                            directions: list) -> tuple[float, float]:
    """Find the centered position for a point within the glyph.

    Casts rays in all directions to find the narrowest width,
    then returns the center of that width.

    Args:
        x: Point x coordinate.
        y: Point y coordinate.
        mask: Binary mask where True = inside glyph.
        directions: List of (dx, dy) direction tuples.

    Returns:
        Tuple (new_x, new_y) of centered position.
    """
    best_width = float('inf')
    best_pos = (x, y)

    for dx, dy in directions:
        dist_pos = _cast_ray(mask, x, y, dx, dy)
        dist_neg = _cast_ray(mask, x, y, -dx, -dy)
        if dist_pos and dist_neg:
            total_width = dist_pos + dist_neg
            if total_width < best_width:
                best_width = total_width
                # Move to center of this width
                offset = (dist_pos - dist_neg) / 2.0
                best_pos = (x + dx * offset, y + dy * offset)

    return best_pos


# Pre-compute 36 ray directions (every 10 degrees)
_RAY_DIRECTIONS = [(np.cos(i * np.pi / 18), np.sin(i * np.pi / 18)) for i in range(36)]


@app.route('/api/center-borders/<int:fid>', methods=['POST'])
def api_center_borders(fid: int) -> Response | tuple[str, int]:
    """Center stroke points within the glyph borders using ray casting.

    For each point in the provided strokes, casts rays in 36 directions
    to find the narrowest glyph width, then moves the point to the center.

    Args:
        fid: Font ID from URL path.

    Returns:
        Response: JSON response with centered strokes.

    Request:
        POST /api/center-borders/<fid>?c=A

        Body (JSON): {"strokes": [[[x, y], ...], ...]}

    Query Parameters:
        c (str, required): Character to render for border detection.
    """
    c, data = request.args.get('c'), request.get_json()
    if not c:
        return jsonify(error="Missing ?c= parameter"), 400
    if not data or 'strokes' not in data:
        return jsonify(error="Missing strokes data"), 400

    _f, mask, err = get_font_and_mask(fid, c)
    if err:
        return err

    h, w = mask.shape
    result_strokes = []

    for stroke in data['strokes']:
        centered_points = []
        for point in stroke:
            x, y = point[0], point[1]
            is_locked = len(point) >= 3 and point[2] == 1

            # Clamp to mask bounds
            ix = int(round(min(max(x, 0), w - 1)))
            iy = int(round(min(max(y, 0), h - 1)))

            # If point is outside glyph, keep original position
            if not mask[iy, ix]:
                centered_points.append([x, y, 1] if is_locked else [x, y])
                continue

            # Find centered position
            new_x, new_y = _center_point_in_glyph(x, y, mask, _RAY_DIRECTIONS)
            centered_points.append([new_x, new_y, 1] if is_locked else [new_x, new_y])

        result_strokes.append(centered_points)

    return jsonify(strokes=result_strokes)


@app.route('/api/detect-markers/<int:fid>', methods=['POST'])
def api_detect_markers(fid: int) -> Response | tuple[str, int]:
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
    _f, m, err = get_font_and_mask(fid, c)
    if err:
        return err
    return jsonify(markers=skel_markers(m))


@app.route('/api/clear-shape-cache/<int:fid>', methods=['POST'])
def api_clear_shape_cache(fid: int) -> Response | tuple[str, int]:
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
def api_skeleton(fid: int) -> Response | tuple[str, int]:
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
    f, err = get_font_or_error(fid)
    if err:
        return err
    r = auto_fit(f['file_path'], c, ret_mark=True)
    if r and r[0]:
        return jsonify(strokes=r[0], markers=r[1])
    m = render_glyph_mask(f['file_path'], c)
    if m is None:
        return jsonify(error="Could not render glyph"), 500
    st = skel_strokes(m)
    return jsonify(strokes=st) if st else (jsonify(error="No skeleton found"), 500)


@app.route('/api/minimal-strokes-batch/<int:fid>', methods=['POST'])
def api_minimal_strokes_batch(fid: int) -> Response | tuple[str, int]:
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
    with get_db_context() as db:
        f = db.execute("SELECT * FROM fonts WHERE id = ?", (fid,)).fetchone()
        if not f:
            return jsonify(error="Font not found"), 404
        # Ensure template_variant column exists
        cursor = db.execute("PRAGMA table_info(characters)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'template_variant' not in columns:
            try:
                db.execute("ALTER TABLE characters ADD COLUMN template_variant TEXT")
                db.commit()
            except sqlite3.OperationalError as e:
                # Log but continue - column might have been added by another process
                import logging
                logging.getLogger(__name__).warning(
                    "Could not add template_variant column: %s", e
                )
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
    return jsonify(ok=True, generated=gen, skipped=skp, failed=fail)


@app.route('/api/skeleton-batch/<int:fid>', methods=['POST'])
def api_skeleton_batch(fid: int) -> Response | tuple[str, int]:
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
def api_template_variants() -> Response:
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
def api_minimal_strokes(fid: int) -> Response | tuple[str, int]:
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
def api_diffvg(fid: int) -> Response | tuple[str, int]:
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
    diffvg = get_diffvg()
    if diffvg is None:
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
    r = diffvg.optimize(font_path=fp, char=c, initial_strokes=cst, canvas_size=DEFAULT_CANVAS_SIZE,
                        num_iterations=DIFFVG_ITERATIONS, stroke_width=DEFAULT_STROKE_WIDTH,
                        thin_iterations=request.args.get('thin', 0, type=int) or 0, timeout=DIFFVG_TIMEOUT)
    return (jsonify(error=r['error']), 500) if 'error' in r else jsonify(strokes=r.get('strokes', []), score=r.get('score', 0), elapsed=r.get('elapsed', 0), source=src)
