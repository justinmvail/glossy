"""Core Flask routes for the Stroke Editor web application.

This module contains the primary route handlers for the Stroke Editor, providing
endpoints for font browsing, character editing, stroke manipulation, and font
quality checking. These routes handle the main user interactions with the
application.

Route Categories:
    Page Routes (HTML):
        - / : Font list page showing all available fonts
        - /font/<fid> : Character grid for a specific font
        - /edit/<fid> : Character editor page for drawing strokes

    Character Data API:
        - GET /api/char/<fid> : Retrieve stroke data for a character
        - POST /api/char/<fid> : Save stroke data for a character

    Rendering API:
        - /api/render/<fid> : Render a character as PNG image
        - /api/thin-preview/<fid> : Render thinned skeleton preview
        - /api/preview/<fid> : Render character with stroke overlay
        - /api/font-sample/<fid> : Render sample text with a font

    Stroke Processing API:
        - POST /api/process/<fid> : Process strokes (smooth, connect)
        - POST /api/snap/<fid> : Snap stroke points to glyph boundary
        - POST /api/center/<fid> : Center strokes on glyph

    Font Quality API:
        - /api/check-connected/<fid> : Check if font has connected letters
        - POST /api/reject-connected : Batch reject connected fonts
        - POST /api/reject/<fid> : Mark a font as rejected
        - POST /api/unreject/<fid> : Remove rejection from a font
        - POST /api/unreject-all : Remove all rejections

Request Parameters:
    Most endpoints use the following common parameters:
        - fid (path): Font ID as integer
        - c (query): Single character to operate on

Response Formats:
    - HTML pages use Jinja2 templates from the templates/ directory
    - API endpoints return JSON with 'ok', 'error', or data fields
    - Image endpoints return PNG with mimetype 'image/png'

Example:
    Typical API usage flow::

        # Get character data
        GET /api/char/42?c=A
        Response: {"strokes": [...], "markers": [...], "image": "base64..."}

        # Save edited strokes
        POST /api/char/42?c=A
        Body: {"strokes": [[[x1,y1], [x2,y2], ...], ...]}
        Response: {"ok": true}
"""

import base64
import io
import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

# Module logger
logger = logging.getLogger(__name__)
from flask import Response, jsonify, render_template, request, send_file
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont
from scipy.ndimage import distance_transform_edt
from skimage.morphology import thin

from stroke_flask import (
    CHARS,
    STROKE_COLORS,
    app,
    data_response,
    error_response,
    font_repository,
    get_char_param_or_error,
    get_font,
    get_font_and_mask,
    get_font_or_error,
    resolve_font_path,
    send_pil_image_as_png,
    success_response,
    validate_char_param,
)
from stroke_rendering import (
    analyze_shape_metrics,
    check_case_mismatch,
    check_char_holes,
    check_char_shape_count,
    render_char_image,
    render_glyph_mask,
    render_text_for_analysis,
)
from stroke_services_core import (
    check_font_quality,
    get_character_data,
    save_character_data,
    render_glyph_image,
    render_thin_preview,
    render_stroke_preview,
    render_font_sample,
    process_strokes,
    snap_strokes_to_boundary,
    center_strokes_on_glyph,
    MIN_SHAPE_COUNT,
    MAX_SHAPE_COUNT,
    MAX_WIDTH_RATIO,
    EXPECTED_EXCLAMATION_SHAPES,
    get_char_shape_count,
)



def _check_font_quality(pil_font: FreeTypeFont, font_path: str) -> dict:
    """Check font quality based on standard criteria.

    Args:
        pil_font: Loaded PIL ImageFont object.
        font_path: Path to the font file (for case mismatch check).

    Returns:
        Dict with quality check results:
            - is_bad: True if font fails quality checks
            - shape_count: Number of shapes in "Hello World" rendering
            - max_width: Maximum width ratio from analysis
            - case_mismatches: List of case mismatch issues
            - l_has_hole: True if 'l' character has holes
            - exclaim_shapes: Number of shapes in '!' character
            - exclaim_ok: True if '!' has expected shape count
    """
    arr = render_text_for_analysis(pil_font, "Hello World")
    if arr is None:
        return {
            'is_bad': True, 'shape_count': 0, 'max_width': 0,
            'case_mismatches': [], 'l_has_hole': False,
            'exclaim_shapes': -1, 'exclaim_ok': False
        }

    shape_count, max_width = analyze_shape_metrics(arr, arr.shape[1])
    case_mismatches = check_case_mismatch(font_path)
    l_has_hole = check_char_holes(pil_font, 'l')
    exclaim_shapes = get_char_shape_count(pil_font, '!')
    exclaim_ok = exclaim_shapes == EXPECTED_EXCLAMATION_SHAPES

    is_bad = (
        shape_count < MIN_SHAPE_COUNT or
        shape_count > MAX_SHAPE_COUNT or
        max_width > MAX_WIDTH_RATIO or
        l_has_hole or
        not exclaim_ok or
        bool(case_mismatches)
    )

    return {
        'is_bad': bool(is_bad),
        'shape_count': int(shape_count),
        'max_width': float(max_width),
        'case_mismatches': case_mismatches,
        'l_has_hole': bool(l_has_hole),
        'exclaim_shapes': int(exclaim_shapes),
        'exclaim_ok': bool(exclaim_ok)
    }


@app.route('/')
def font_list() -> str:
    """Display the font list page with alphabet navigation.

    Renders the font browser page. Fonts are loaded via JavaScript/AJAX
    when user clicks on a letter in the alphabet bar.

    Query Parameters:
        rejected (str, optional): Set to '1' to show only rejected fonts.

    Returns:
        str: Rendered HTML template 'font_list.html' with context:
            - show_rejected: Boolean indicating if viewing rejected fonts.

    Example:
        View all non-rejected fonts::

            GET /

        View rejected fonts only::

            GET /?rejected=1
    """
    show_rejected = request.args.get('rejected') == '1'
    return render_template('font_list.html', show_rejected=show_rejected)


@app.route('/font/<int:fid>')
def char_grid(fid: int) -> str | tuple[str, int]:
    """Display the character grid page for a specific font.

    Shows a grid of all characters (A-Z, a-z, 0-9) for the font, with visual
    indicators for which characters have stroke data defined.

    Args:
        fid: The font ID from the URL path.

    Returns:
        str: Rendered HTML template 'char_grid.html' on success.
        tuple: ("Font not found", 404) if font ID does not exist.

    Template Context:
        font (sqlite3.Row): The font record with id, name, file_path, etc.
        chars (list): List of character records, each containing:
            - char (str): The character itself
            - strokes_raw (str|None): JSON stroke data or None if not defined
            - point_count (int): Total number of points across all strokes

    Note:
        If no characters have stroke data, a default list is generated from
        the CHARS constant with empty stroke data for each character.

    Example:
        View character grid for font ID 42::

            GET /font/42
    """
    f = font_repository.get_font_by_id(fid)
    if not f:
        return "Font not found", 404
    ch = font_repository.get_font_characters(fid)
    return render_template('char_grid.html', font=f, chars=ch if ch else [{'char': c, 'strokes_raw': None, 'point_count': 0} for c in CHARS])


@app.route('/edit/<int:fid>')
def edit_char(fid: int) -> str | tuple[str, int]:
    """Display the stroke editor page for a specific character.

    Opens the interactive editor interface where users can draw and edit
    stroke paths for a character in the specified font.

    Args:
        fid: The font ID from the URL path.

    Query Parameters:
        c (str, required): The single character to edit.

    Returns:
        str: Rendered HTML template 'editor.html' on success.
        tuple: Error message and status code on failure:
            - ("Missing character parameter ?c=", 400) if c is not provided
            - ("Font not found", 404) if font ID does not exist

    Template Context:
        font (sqlite3.Row): The font record with id, name, file_path, etc.
        char (str): The character being edited.
        char_list (str): The full CHARS string for navigation between characters.

    Example:
        Edit the letter 'A' for font ID 42::

            GET /edit/42?c=A
    """
    c = request.args.get('c')
    if not c:
        return "Missing character parameter ?c=", 400
    f = get_font(fid)
    return render_template('editor.html', font=f, char=c, char_list=CHARS) if f else ("Font not found", 404)


@app.route('/api/char/<int:fid>')
def api_get_char(fid: int) -> Response | tuple[str, int]:
    """Retrieve stroke data and rendered image for a character.

    Fetches the stored stroke data, markers, and a rendered glyph image for
    the specified character in the specified font.

    Args:
        fid: The font ID from the URL path.

    Query Parameters:
        c (str, required): The single character to retrieve.

    Returns:
        flask.Response: JSON response with one of the following:

        Success (200):
            {
                "strokes": [[point, ...], ...],  // Array of strokes
                "markers": [...],                // Array of markers
                "image": "base64string"          // PNG image as base64 or null
            }

            Where each point is [x, y] or [x, y, 1] if locked.

        Error (400):
            {"error": "Missing ?c= parameter"}
            {"error": "Character must be a single character"}

        Error (404):
            {"error": "Font not found"}

    Note:
        The image is rendered from the font file, not from saved data.
        Returns empty arrays for strokes/markers if no data exists for
        the character.

    Example:
        Request::

            GET /api/char/42?c=A

        Response::

            {
                "strokes": [[[100, 50], [112, 150, 1], [124, 50]]],
                "markers": [],
                "image": "iVBORw0KGgoAAAANSUhEUgAA..."
            }
    """
    c = request.args.get('c')
    ok, err = validate_char_param(c)
    if not ok:
        return err
    f = font_repository.get_font_by_id(fid)
    if not f:
        return jsonify(error="Font not found"), 404
    row = font_repository.get_character(fid, c)
    img = render_char_image(f['file_path'], c)
    return jsonify(
        strokes=json.loads(row['strokes_raw']) if row and row['strokes_raw'] else [],
        markers=json.loads(row['markers']) if row and row['markers'] else [],
        image=base64.b64encode(img).decode('ascii') if img else None
    )


@app.route('/api/char/<int:fid>', methods=['POST'])
def api_save_char(fid: int) -> Response | tuple[str, int]:
    """Save stroke data for a character.

    Stores or updates the stroke data and optional markers for a character
    in the database. Performs an upsert operation (insert if new, update if
    exists).

    Args:
        fid: The font ID from the URL path.

    Query Parameters:
        c (str, required): The single character to save.

    Request Body:
        JSON object with the following structure::

            {
                "strokes": [[point, ...], ...],  // Required: array of strokes
                "markers": [...]                  // Optional: array of markers
            }

        Where each point is [x, y] or [x, y, 1] for locked points.

    Returns:
        flask.Response: JSON response with one of the following:

        Success (200):
            {"ok": true}

        Error (400):
            {"error": "Missing ?c= parameter"}
            {"error": "Character must be a single character"}
            {"error": "Missing strokes data"}

    Note:
        The point_count field is automatically calculated as the total number
        of points across all strokes.

    Example:
        Request::

            POST /api/char/42?c=A
            Content-Type: application/json

            {
                "strokes": [
                    [[100, 50], [112, 150, 1], [124, 50]],
                    [[106, 100], [118, 100]]
                ],
                "markers": []
            }

        Response::

            {"ok": true}
    """
    c, data = request.args.get('c'), request.get_json()
    ok, err = validate_char_param(c)
    if not ok:
        return err
    if not data or 'strokes' not in data:
        return jsonify(error="Missing strokes data"), 400
    st, mk = data['strokes'], data.get('markers', [])
    point_count = sum(len(s) for s in st)
    font_repository.save_character(
        fid, c,
        strokes_raw=json.dumps(st),
        point_count=point_count,
        markers=json.dumps(mk) if mk else None
    )
    logger.info("Saved char '%s' for font %d: %d strokes, %d points",
                c, fid, len(st), point_count)
    return success_response()


@app.route('/api/render/<int:fid>')
def api_render(fid: int) -> Response | tuple[str, int]:
    """Render a character glyph as a PNG image.

    Renders the specified character using the font file and returns it as
    a PNG image suitable for display in the editor.

    Args:
        fid: The font ID from the URL path.

    Query Parameters:
        c (str, required): The single character to render.

    Returns:
        flask.Response: PNG image with mimetype 'image/png' on success.
        tuple: Error message and status code on failure:
            - ("Missing ?c= parameter", 400) if c is not provided
            - ("Font not found", 404) if font ID does not exist
            - ("Could not render", 500) if rendering fails

    Note:
        The image is rendered at the default canvas size (224x224) and font
        size (200) defined in stroke_flask.py constants.

    Example:
        Render the letter 'A' for font ID 42::

            GET /api/render/42?c=A

        Use in HTML::

            <img src="/api/render/42?c=A" alt="Letter A">
    """
    c, err = get_char_param_or_error(response_format='text')
    if err:
        return err
    f, err = get_font_or_error(fid, response_format='text')
    if err:
        return err
    img = render_char_image(f['file_path'], c)
    return send_file(io.BytesIO(img), mimetype='image/png') if img else error_response("Could not render", 500, 'text')


@app.route('/api/thin-preview/<int:fid>')
def api_thin_preview(fid: int) -> Response | tuple[str, int]:
    """Generate a thinned skeleton preview of a character glyph.

    Renders the character and applies morphological thinning to produce a
    skeleton representation. Useful for visualizing the center-line structure
    that stroke paths should follow.

    Args:
        fid: The font ID from the URL path.

    Query Parameters:
        c (str, required): The single character to process.
        thin (int, optional): Maximum number of thinning iterations.
            Defaults to 5. Higher values produce thinner skeletons.

    Returns:
        flask.Response: PNG image (224x224) with mimetype 'image/png' on success.
            The image shows:
            - White background (255, 255, 255)
            - Light gray glyph area (200, 200, 200)
            - Black skeleton lines (0, 0, 0)

        tuple: Error message and status code on failure:
            - ("Missing ?c= parameter", 400) if c is not provided
            - ("Font not found", 404) if font ID does not exist
            - ("Could not render glyph", 500) if rendering fails

    Note:
        Uses scikit-image's thin() function for morphological thinning.
        The thin parameter controls how many iterations of thinning to apply.

    Example:
        Get skeleton preview with default thinning::

            GET /api/thin-preview/42?c=A

        Get more aggressive thinning::

            GET /api/thin-preview/42?c=A&thin=10
    """
    c, err = get_char_param_or_error(response_format='text')
    if err:
        return err
    f, err = get_font_or_error(fid, response_format='text')
    if err:
        return err
    m = render_glyph_mask(f['file_path'], c)
    if m is None:
        return error_response("Could not render glyph", 500, 'text')
    thin_iter = request.args.get('thin', 5, type=int)
    if thin_iter is None or thin_iter < 0 or thin_iter > 100:
        return error_response("Invalid thin parameter", 400, 'text')
    th = thin(m, max_num_iter=thin_iter)
    img = np.full((224, 224, 3), 255, dtype=np.uint8)
    img[m], img[th] = [200, 200, 200], [0, 0, 0]
    return send_pil_image_as_png(Image.fromarray(img))


@app.route('/api/check-connected/<int:fid>')
def api_check_connected(fid: int) -> Response | tuple[str, int]:
    """Check if a font has connected/script letters or quality issues.

    Analyzes a font by rendering sample text and checking for various issues
    that indicate the font may not be suitable for stroke extraction, such as
    connected letters (cursive/script fonts) or incorrect shape counts.

    Args:
        fid: The font ID from the URL path.

    Returns:
        flask.Response: JSON response with one of the following:

        Success (200)::

            {
                "shapes": 11,           // Number of distinct shapes in "Hello World"
                "bad": false,           // True if font fails quality checks
                "case_mismatches": []   // List of case mismatch issues
            }

        Error (404):
            {"error": "Font not found"}

        Error (500):
            {"error": "Could not render"}
            {"error": "<exception message>"}

    Quality Checks:
        A font is marked as "bad" if any of the following are true:
        - Shape count for "Hello World" is less than 10 or more than 15
          (indicates connected letters or missing characters)
        - Maximum stroke width ratio exceeds 0.225
        - The letter 'l' has holes (indicates decorative font)
        - The character '!' does not have exactly 2 shapes

    Example:
        Request::

            GET /api/check-connected/42

        Response for a good font::

            {"shapes": 11, "bad": false, "case_mismatches": []}

        Response for a script font::

            {"shapes": 3, "bad": true, "case_mismatches": []}
    """
    f = get_font(fid)
    if not f:
        return jsonify(error="Font not found"), 404
    fp = resolve_font_path(f['file_path'])
    try:
        pf = ImageFont.truetype(fp, 60)
        result = _check_font_quality(pf, fp)
        if result['shape_count'] == 0:
            return jsonify(error="Could not render"), 500
        return jsonify(
            shapes=result['shape_count'],
            bad=result['is_bad'],
            case_mismatches=result['case_mismatches'],
            case_mismatch_count=len(result['case_mismatches']),
            max_width_pct=round(result['max_width'] * 100, 1),
            l_has_hole=result['l_has_hole'],
            exclaim_shapes=result['exclaim_shapes'],
            exclaim_ok=result['exclaim_ok']
        )
    except Exception as e:
        logger.warning("Font quality check failed for font %d: %s", fid, e)
        return jsonify(error="Could not check font"), 500


@app.route('/api/verify-all-stream')
def api_verify_all_stream() -> Response:
    """Stream font quality verification results via SSE.

    Checks all non-rejected fonts server-side and streams results as
    Server-Sent Events, avoiding 57K+ individual HTTP round trips.

    SSE Events:
        progress: {"checked": N, "failed": N, "total": N, "name": "...", "status": "ok"|"fail", "issues": [...]}
        done: {"done": true, "checked": N, "failed": N, "failed_fonts": [...]}

    Returns:
        Response: SSE event stream.
    """
    from stroke_flask import format_sse_event, get_db_context

    def generate():
        with get_db_context() as db:
            fonts = db.execute('''
                SELECT f.id, f.name, f.file_path
                FROM fonts f
                WHERE f.id NOT IN (SELECT font_id FROM font_removals)
                ORDER BY f.name
            ''').fetchall()

        total = len(fonts)
        checked = 0
        failed = 0
        failed_fonts = []

        yield format_sse_event({'type': 'start', 'total': total})

        for font in fonts:
            fid, name, file_path = font['id'], font['name'], font['file_path']
            checked += 1

            try:
                fp = resolve_font_path(file_path)
                pf = ImageFont.truetype(fp, 60)
                result = _check_font_quality(pf, fp)

                issues = []
                if result['shape_count'] == 0:
                    issues.append('render_failed')
                else:
                    if result['shape_count'] < MIN_SHAPE_COUNT or result['shape_count'] > MAX_SHAPE_COUNT:
                        issues.append(f"shapes={result['shape_count']}")
                    if result['max_width'] > MAX_WIDTH_RATIO:
                        issues.append(f"width={round(result['max_width'] * 100, 1)}%")
                    if result['l_has_hole']:
                        issues.append('l_has_hole')
                    if result['exclaim_shapes'] != EXPECTED_EXCLAMATION_SHAPES:
                        issues.append(f"exclaim_shapes={result['exclaim_shapes']}")
                    if result['case_mismatches']:
                        issues.append(f"case_issues={','.join(result['case_mismatches'])}")

                if issues:
                    failed += 1
                    failed_fonts.append({
                        'font_id': fid, 'name': name, 'issues': issues,
                        'shapes': result['shape_count'],
                        'max_width_pct': round(result['max_width'] * 100, 1),
                        'l_has_hole': result['l_has_hole'],
                        'exclaim_shapes': result['exclaim_shapes'],
                        'case_mismatches': result.get('case_mismatches', []),
                    })

                yield format_sse_event({
                    'type': 'progress',
                    'checked': checked, 'failed': failed, 'total': total,
                    'name': name,
                    'status': 'fail' if issues else 'ok',
                    'issues': issues,
                })

            except Exception as e:
                failed += 1
                failed_fonts.append({
                    'font_id': fid, 'name': name,
                    'issues': [f'error: {e}'],
                })
                yield format_sse_event({
                    'type': 'progress',
                    'checked': checked, 'failed': failed, 'total': total,
                    'name': name, 'status': 'error',
                    'issues': [str(e)],
                })

        yield format_sse_event({
            'type': 'done', 'done': True,
            'checked': checked, 'failed': failed,
            'failed_fonts': failed_fonts,
        })

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/reject-failed-stream')
def api_reject_failed_stream() -> Response:
    """Re-verify all fonts and reject failures server-side via SSE.

    Avoids sending large payloads from browser. Runs verification and
    inserts rejections directly into the database in batches.

    SSE Events:
        progress: {"checked": N, "failed": N, "rejected": N, "total": N, "name": "..."}
        done: {"done": true, "checked": N, "rejected": N}
    """
    from stroke_flask import format_sse_event, get_db_context

    def generate():
        with get_db_context() as db:
            fonts = db.execute('''
                SELECT f.id, f.name, f.file_path
                FROM fonts f
                WHERE f.id NOT IN (SELECT font_id FROM font_removals)
                ORDER BY f.name
            ''').fetchall()

        total = len(fonts)
        checked = 0
        failed = 0
        rejected = 0
        batch = []
        BATCH_SIZE = 500

        yield format_sse_event({'type': 'start', 'total': total})

        for font in fonts:
            fid, name, file_path = font['id'], font['name'], font['file_path']
            checked += 1

            try:
                fp = resolve_font_path(file_path)
                pf = ImageFont.truetype(fp, 60)
                result = _check_font_quality(pf, fp)

                issues = []
                if result['shape_count'] == 0:
                    issues.append('render_failed')
                else:
                    if result['shape_count'] < MIN_SHAPE_COUNT or result['shape_count'] > MAX_SHAPE_COUNT:
                        issues.append(f"shapes={result['shape_count']}")
                    if result['max_width'] > MAX_WIDTH_RATIO:
                        issues.append(f"width={round(result['max_width'] * 100, 1)}%")
                    if result['l_has_hole']:
                        issues.append('l_has_hole')
                    if result['exclaim_shapes'] != EXPECTED_EXCLAMATION_SHAPES:
                        issues.append(f"exclaim_shapes={result['exclaim_shapes']}")
                    if result['case_mismatches']:
                        issues.append(f"case_issues={','.join(result['case_mismatches'])}")

                if issues:
                    failed += 1
                    details = json.dumps({
                        'shapes': result['shape_count'],
                        'max_width_pct': round(result['max_width'] * 100, 1),
                        'l_has_hole': result['l_has_hole'],
                        'exclaim_shapes': result['exclaim_shapes'],
                        'case_mismatches': result.get('case_mismatches', []),
                    })
                    batch.append((fid, details))

            except Exception as e:
                failed += 1
                batch.append((fid, json.dumps({'error': str(e)})))

            # Flush batch to DB
            if len(batch) >= BATCH_SIZE:
                with get_db_context() as db:
                    for b_fid, b_details in batch:
                        db.execute(
                            "INSERT OR IGNORE INTO font_removals (font_id, reason_id, details) VALUES (?, 8, ?)",
                            (b_fid, b_details)
                        )
                rejected += len(batch)
                batch = []

                yield format_sse_event({
                    'type': 'progress',
                    'checked': checked, 'failed': failed,
                    'rejected': rejected, 'total': total,
                    'name': name,
                })

            elif checked % 100 == 0:
                yield format_sse_event({
                    'type': 'progress',
                    'checked': checked, 'failed': failed,
                    'rejected': rejected, 'total': total,
                    'name': name,
                })

        # Flush remaining
        if batch:
            with get_db_context() as db:
                for b_fid, b_details in batch:
                    db.execute(
                        "INSERT OR IGNORE INTO font_removals (font_id, reason_id, details) VALUES (?, 8, ?)",
                        (b_fid, b_details)
                    )
            rejected += len(batch)

        yield format_sse_event({
            'type': 'done', 'done': True,
            'checked': checked, 'rejected': rejected,
        })

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/reject-batch', methods=['POST'])
def api_reject_batch() -> Response:
    """Reject multiple fonts by ID.

    Accepts a list of font IDs and marks them all as rejected with reason_id 8.
    This is used by the UI to reject fonts that have already been verified as bad.

    Request Body:
        JSON object with a 'font_ids' array::

            {"font_ids": [1, 2, 3]}

    Returns:
        flask.Response: JSON response::

            {
                "ok": true,
                "rejected": 3    // Number of fonts rejected
            }

    Note:
        - Skips fonts that are already rejected
        - Invalid font IDs are silently ignored

    Example:
        Request::

            POST /api/reject-batch
            {"font_ids": [42, 57, 123]}

        Response::

            {"ok": true, "rejected": 3}

    Request Body (with details)::

        {
            "rejections": [
                {"font_id": 42, "shapes": 5, "max_width_pct": 30.5, "case_mismatches": ["r", "n"]},
                {"font_id": 57, "shapes": 11, "l_has_hole": true}
            ]
        }
    """
    data = request.get_json() or {}

    from stroke_flask import get_db_context

    # Support both old format (font_ids) and new format (rejections with details)
    rejections = data.get('rejections', [])
    if rejections:
        # Check if a specific reason is provided (e.g., 'font_variation', 'duplicate')
        reason_code = rejections[0].get('reason') if rejections else None

        if reason_code and reason_code != 'manual':
            # Use the specific reason code - need to look up reason_id
            with get_db_context() as db:
                # Ensure the reason exists
                db.execute(
                    "INSERT OR IGNORE INTO removal_reasons (code, description) VALUES (?, ?)",
                    (reason_code, f"Rejected: {reason_code}")
                )
                row = db.execute(
                    "SELECT id FROM removal_reasons WHERE code = ?",
                    (reason_code,)
                ).fetchone()
                if not row:
                    return error_response(f"Unknown reason: {reason_code}")
                reason_id = row[0]

                # Insert rejections with this reason
                count = 0
                for r in rejections:
                    fid = r.get('font_id')
                    if not fid:
                        continue
                    details = json.dumps({k: v for k, v in r.items() if k not in ('font_id', 'reason')})
                    try:
                        db.execute(
                            "INSERT OR IGNORE INTO font_removals (font_id, reason_id, details) VALUES (?, ?, ?)",
                            (fid, reason_id, details)
                        )
                        count += 1
                    except Exception:
                        pass
            return success_response(rejected=count)

        to_reject = []
        for r in rejections:
            fid = r.get('font_id')
            if not fid:
                continue
            # Store quality check results as JSON details
            details = json.dumps({
                'shapes': r.get('shapes'),
                'max_width_pct': r.get('max_width_pct'),
                'l_has_hole': r.get('l_has_hole'),
                'exclaim_ok': r.get('exclaim_ok'),
                'exclaim_shapes': r.get('exclaim_shapes'),
                'case_mismatches': r.get('case_mismatches', []),
                'error': r.get('error')
            })
            to_reject.append((fid, details))
    else:
        # Fallback to old format
        font_ids = data.get('font_ids', [])
        if not font_ids:
            return success_response(rejected=0)
        to_reject = [(fid, "Failed quality check") for fid in font_ids]

    rej = font_repository.reject_fonts_batch(to_reject)
    return success_response(rejected=rej)


@app.route('/api/font-sample/<int:fid>')
def api_font_sample(fid: int) -> Response | tuple[str, int]:
    """Render sample text using a specific font.

    Creates a PNG image of custom text rendered with the specified font.
    Useful for previewing how a font looks before editing individual
    characters.

    Args:
        fid: The font ID from the URL path.

    Query Parameters:
        text (str, optional): The text to render. Defaults to "Hello World!".
        h (int, optional): The height of the output image in pixels.
            Defaults to 40. Font size is set to 85% of this height.

    Returns:
        flask.Response: PNG image with mimetype 'image/png' on success.
            The image has:
            - Transparent background
            - White text
            - Width automatically sized to fit the text
            - Height as specified by the 'h' parameter

        tuple: Error message and status code on failure:
            - ("Font not found", 404) if font ID does not exist
            - ("Could not render text", 500) if text has no bounding box
            - ("Error: <message>", 500) for other rendering errors

    Example:
        Render default sample text::

            GET /api/font-sample/42

        Render custom text with larger height::

            GET /api/font-sample/42?text=ABCDEF&h=60

        Use in HTML::

            <img src="/api/font-sample/42?text=Preview" alt="Font preview">
    """
    txt = request.args.get('text', 'Hello World!')
    if len(txt) > 500:
        return "Text parameter too long (max 500 characters)", 400
    h = request.args.get('h', 40, type=int)
    if h is None or h < 10 or h > 500:
        return "Invalid height parameter", 400
    f = get_font(fid)
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
        logger.warning("Font sample render failed for font %d: %s", fid, e)
        return "Could not render font sample", 500


@app.route('/api/preview/<int:fid>')
def api_preview(fid: int) -> Response | tuple[str, int]:
    """Generate a preview image showing strokes overlaid on the glyph.

    Creates a composite image with the character glyph as a semi-transparent
    background and the saved stroke paths drawn on top in distinct colors.
    Useful for quickly reviewing stroke quality across multiple characters.

    Args:
        fid: The font ID from the URL path.

    Query Parameters:
        c (str, required): The single character to preview.

    Returns:
        flask.Response: PNG image (224x224) with mimetype 'image/png' on success.
            The image contains:
            - White background
            - Semi-transparent glyph (alpha 60)
            - Colored stroke lines (2px width) cycling through STROKE_COLORS

        tuple: Error message and status code on failure:
            - ("Missing ?c= parameter", 400) if c is not provided
            - ("Font not found", 404) if font ID does not exist
            - ("Could not render", 500) if rendering fails

    Note:
        Each stroke is drawn in a different color from STROKE_COLORS,
        cycling through the 8 available colors. Strokes with fewer than
        2 points are skipped.

    Example:
        Get preview for letter 'A'::

            GET /api/preview/42?c=A

        Use in an image grid::

            <img src="/api/preview/42?c=A" alt="A preview">
            <img src="/api/preview/42?c=B" alt="B preview">
    """
    c, err = get_char_param_or_error(response_format='text')
    if err:
        return err
    f, err = get_font_or_error(fid, response_format='text')
    if err:
        return err
    row = font_repository.get_character_strokes(fid, c)
    img = render_char_image(f['file_path'], c)
    if not img:
        return error_response("Could not render", 500, 'text')
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
    return send_pil_image_as_png(bg)


@app.route('/api/process/<int:fid>', methods=['POST'])
def api_process(fid: int) -> Response | tuple[str, int]:
    """Process strokes with optional smoothing and connection extension.

    Applies post-processing algorithms to the provided stroke data, including
    Gaussian smoothing to reduce noise and connection extension to close gaps
    between stroke endpoints.

    Args:
        fid: The font ID from the URL path (used for context, not processing).

    Query Parameters:
        c (str, required): The single character being processed.

    Request Body:
        JSON object with the following structure::

            {
                "strokes": [[point, ...], ...],   // Required: array of strokes
                "smooth": true,                    // Optional: apply Gaussian smoothing
                "smooth_sigma": 1.5,               // Optional: smoothing sigma (default 1.5)
                "connect": true,                   // Optional: extend to connect (default true)
                "max_extension": 8.0               // Optional: max extension distance (default 8.0)
            }

    Returns:
        flask.Response: JSON response with one of the following:

        Success (200)::

            {
                "strokes": [[point, ...], ...]  // Processed strokes
            }

        Error (400):
            {"error": "Missing ?c= parameter"}
            {"error": "Missing strokes data"}

    Processing Details:
        - Smoothing: Applies Gaussian filter with adaptive sigma based on
          stroke length. Shorter strokes get proportionally less smoothing.
        - Connection: Extends stroke endpoints toward each other to close
          small gaps, up to max_extension pixels.
        - Locked points (marked with p[2]=1) are preserved in the output.

    Example:
        Request::

            POST /api/process/42?c=A
            Content-Type: application/json

            {
                "strokes": [[[100, 50], [112, 150], [124, 50]]],
                "smooth": true,
                "smooth_sigma": 2.0,
                "connect": true,
                "max_extension": 10.0
            }

        Response::

            {
                "strokes": [[[100.5, 52.1], [112.0, 149.8], [123.5, 51.2]]]
            }
    """
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
def api_snap(fid: int) -> Response | tuple[str, int]:
    """Snap stroke points to the nearest glyph boundary.

    Moves any stroke points that are outside the glyph mask to the nearest
    point inside the glyph. Uses Euclidean distance transform to efficiently
    find the closest boundary point.

    Args:
        fid: The font ID from the URL path.

    Query Parameters:
        c (str, required): The single character to snap points to.

    Request Body:
        JSON object with the following structure::

            {
                "strokes": [[point, ...], ...]  // Required: array of strokes
            }

        Where each point is [x, y] or [x, y, 1] for locked points.

    Returns:
        flask.Response: JSON response with one of the following:

        Success (200)::

            {
                "strokes": [[point, ...], ...]  // Strokes with snapped points
            }

        Error (400):
            {"error": "Missing ?c= parameter"}
            {"error": "Missing strokes data"}

        Error (404):
            {"error": "Font not found"}

        Error (500):
            {"error": "Could not render glyph"}

    Algorithm:
        1. Render the glyph as a binary mask
        2. Compute distance transform with index mapping
        3. For each point outside the mask, replace with the nearest
           point inside the mask
        4. Points already inside the mask are unchanged
        5. Locked status (p[2]=1) is preserved

    Example:
        Request::

            POST /api/snap/42?c=A
            Content-Type: application/json

            {
                "strokes": [[[50, 50], [200, 200]]]  // Point outside glyph
            }

        Response::

            {
                "strokes": [[[50, 50], [150, 180]]]  // Snapped to boundary
            }
    """
    c, data = request.args.get('c'), request.get_json()
    if not c:
        return jsonify(error="Missing ?c= parameter"), 400
    if not data or 'strokes' not in data:
        return jsonify(error="Missing strokes data"), 400
    _f, m, err = get_font_and_mask(fid, c)
    if err:
        return err
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
def api_center(fid: int) -> Response | tuple[str, int]:
    """Center strokes on the glyph bounding box.

    Translates all stroke points so that the center of the stroke bounding
    box aligns with the center of the glyph bounding box. Useful for
    repositioning strokes that were drawn slightly off-center.

    Args:
        fid: The font ID from the URL path.

    Query Parameters:
        c (str, required): The single character to center strokes on.

    Request Body:
        JSON object with the following structure::

            {
                "strokes": [[point, ...], ...]  // Required: array of strokes
            }

        Where each point is [x, y] or [x, y, 1] for locked points.

    Returns:
        flask.Response: JSON response with one of the following:

        Success (200)::

            {
                "strokes": [[point, ...], ...]  // Centered strokes
            }

        Error (400):
            {"error": "Missing ?c= parameter"}
            {"error": "Missing strokes data"}

        Error (404):
            {"error": "Font not found"}

        Error (500):
            {"error": "Could not render glyph"}
            {"error": "Empty glyph"}

    Algorithm:
        1. Compute glyph bounding box center from the mask
        2. Compute stroke bounding box center from all points
        3. Translate all points by the difference
        4. Note: Locked point status is NOT preserved (points become unlocked)

    Note:
        If there are no stroke points, returns the original strokes unchanged.

    Example:
        Request::

            POST /api/center/42?c=A
            Content-Type: application/json

            {
                "strokes": [[[80, 80], [100, 180], [120, 80]]]
            }

        Response (if glyph is centered at 112, 112)::

            {
                "strokes": [[[92, 44], [112, 144], [132, 44]]]
            }
    """
    c, data = request.args.get('c'), request.get_json()
    if not c:
        return jsonify(error="Missing ?c= parameter"), 400
    if not data or 'strokes' not in data:
        return jsonify(error="Missing strokes data"), 400
    _f, m, err = get_font_and_mask(fid, c)
    if err:
        return err
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
def api_reject_font(fid: int) -> Response | tuple[str, int]:
    """Mark a font as rejected in the stroke editor.

    Adds a rejection record to the font_removals table with reason_id 8
    (stroke editor rejection). Rejected fonts are hidden from the main
    font list.

    Args:
        fid: The font ID from the URL path.

    Returns:
        flask.Response: JSON response with one of the following:

        Success (200) - font was rejected::

            {"ok": true, "status": "rejected"}

        Success (200) - font was already rejected::

            {"ok": true, "status": "already_rejected"}

        Error (404):
            {"error": "Font not found"}

    Note:
        The rejection reason_id 8 corresponds to "Rejected in stroke editor"
        and is distinct from other rejection reasons like duplicates (2).

    Example:
        Request::

            POST /api/reject/42

        Response::

            {"ok": true, "status": "rejected"}
    """
    # Note: There's a theoretical TOCTOU race condition between the existence check
    # and the reject operation. This is benign because:
    # 1. reject_font() handles missing fonts gracefully (returns False)
    # 2. Database foreign key constraints prevent orphaned rejection records
    # 3. Single-user web UI makes concurrent font deletion extremely unlikely
    if not font_repository.get_font_by_id(fid):
        return error_response("Font not found", 404)
    if font_repository.reject_font(fid, 'Rejected in stroke editor'):
        return success_response(status='rejected')
    return success_response(status='already_rejected')


@app.route('/api/unreject/<int:fid>', methods=['POST'])
def api_unreject_font(fid: int) -> Response | tuple[str, int]:
    """Remove the rejection status from a font.

    Deletes the rejection record for the specified font (reason_id 8 only).
    The font will reappear in the main font list after un-rejection.

    Args:
        fid: The font ID from the URL path.

    Returns:
        flask.Response: JSON response::

            {"ok": true, "status": "unrejected"}

    Note:
        This only removes stroke editor rejections (reason_id 8).
        Other rejection reasons (like duplicates) are not affected.
        No error is returned if the font was not rejected.

    Example:
        Request::

            POST /api/unreject/42

        Response::

            {"ok": true, "status": "unrejected"}
    """
    font_repository.unreject_font(fid)
    return success_response(status='unrejected')


@app.route('/api/unreject-all', methods=['POST'])
def api_unreject_all() -> Response:
    """Remove rejection status from all fonts.

    Deletes all rejection records with reason_id 8 (stroke editor rejections)
    from the database, effectively restoring all previously rejected fonts
    to the active font list.

    Returns:
        flask.Response: JSON response::

            {
                "ok": true,
                "restored": 23  // Number of fonts un-rejected
            }

    Note:
        This only removes stroke editor rejections (reason_id 8).
        Other rejection reasons (like duplicates with reason_id 2) are
        not affected.

    Warning:
        This is a bulk operation that cannot be easily undone. Consider
        using api_unreject_font for individual fonts when possible.

    Example:
        Request::

            POST /api/unreject-all

        Response::

            {"ok": true, "restored": 23}
    """
    restored = font_repository.unreject_all_fonts()
    return success_response(restored=restored)


@app.route('/data-management')
def data_management_page() -> str:
    """Data management page for font quality verification and scraping.

    Provides a UI for:
    - Viewing database statistics
    - Running font quality verification
    - Managing rejected fonts
    - Reset and re-scrape all fonts with real-time progress

    Returns:
        str: Rendered HTML template for the data management page.
    """
    from stroke_flask import get_db_context

    with get_db_context() as db:
        total_fonts = db.execute('SELECT COUNT(*) FROM fonts').fetchone()[0]

        # Count rejected fonts (reason_id=8 is quality rejection)
        rejected_fonts = db.execute('''
            SELECT COUNT(DISTINCT font_id) FROM font_removals WHERE reason_id = 8
        ''').fetchone()[0]

        active_fonts = total_fonts - rejected_fonts

        # Count fonts with stroke data
        with_strokes = db.execute('''
            SELECT COUNT(DISTINCT font_id) FROM characters
            WHERE strokes_raw IS NOT NULL OR strokes_processed IS NOT NULL
        ''').fetchone()[0]

    return render_template('data_management.html',
                           total_fonts=total_fonts,
                           active_fonts=active_fonts,
                           rejected_fonts=rejected_fonts,
                           with_strokes=with_strokes)


@app.route('/api/fonts-list')
def api_fonts_list() -> Response:
    """Get list of all active (non-rejected) fonts.

    Returns a JSON array of font objects for use in verification.

    Returns:
        flask.Response: JSON array of font objects with id, name, source.
    """
    from stroke_flask import get_db_context

    with get_db_context() as db:
        fonts = db.execute('''
            SELECT f.id, f.name, f.source
            FROM fonts f
            WHERE f.id NOT IN (
                SELECT font_id FROM font_removals
            )
            ORDER BY f.name
        ''').fetchall()

    return jsonify([{'id': f[0], 'name': f[1], 'source': f[2]} for f in fonts])


@app.route('/api/fonts-by-letter')
def api_fonts_by_letter() -> Response:
    """Get fonts filtered by first letter.

    Query Parameters:
        letter: Single letter A-Z, or '#' for numbers/symbols
        rejected: '1' to show rejected fonts instead

    Returns:
        flask.Response: JSON with fonts array and counts.
    """
    from stroke_flask import get_db_context

    letter = request.args.get('letter', 'A').upper()
    show_rejected = request.args.get('rejected') == '1'

    with get_db_context() as db:
        if show_rejected:
            # Show rejected fonts
            if letter == '#':
                # Numbers and symbols
                fonts = db.execute('''
                    SELECT f.id, f.name, f.source,
                           (SELECT COUNT(*) FROM characters c WHERE c.font_id = f.id) as char_count
                    FROM fonts f
                    JOIN font_removals fr ON f.id = fr.font_id
                    WHERE f.name NOT GLOB '[A-Za-z]*'
                    ORDER BY f.name
                ''').fetchall()
            else:
                fonts = db.execute('''
                    SELECT f.id, f.name, f.source,
                           (SELECT COUNT(*) FROM characters c WHERE c.font_id = f.id) as char_count
                    FROM fonts f
                    JOIN font_removals fr ON f.id = fr.font_id
                    WHERE UPPER(SUBSTR(f.name, 1, 1)) = ?
                    ORDER BY f.name
                ''', (letter,)).fetchall()
        else:
            # Show active fonts
            if letter == '#':
                fonts = db.execute('''
                    SELECT f.id, f.name, f.source,
                           (SELECT COUNT(*) FROM characters c WHERE c.font_id = f.id) as char_count
                    FROM fonts f
                    WHERE f.id NOT IN (SELECT font_id FROM font_removals)
                    AND f.name NOT GLOB '[A-Za-z]*'
                    ORDER BY f.name
                ''').fetchall()
            else:
                fonts = db.execute('''
                    SELECT f.id, f.name, f.source,
                           (SELECT COUNT(*) FROM characters c WHERE c.font_id = f.id) as char_count
                    FROM fonts f
                    WHERE f.id NOT IN (SELECT font_id FROM font_removals)
                    AND UPPER(SUBSTR(f.name, 1, 1)) = ?
                    ORDER BY f.name
                ''', (letter,)).fetchall()

        # Get counts per letter for the alphabet bar
        if show_rejected:
            counts = db.execute('''
                SELECT UPPER(SUBSTR(f.name, 1, 1)) as letter, COUNT(*) as cnt
                FROM fonts f
                JOIN font_removals fr ON f.id = fr.font_id
                WHERE f.name GLOB '[A-Za-z]*'
                GROUP BY UPPER(SUBSTR(f.name, 1, 1))
            ''').fetchall()
            other_count = db.execute('''
                SELECT COUNT(*) FROM fonts f
                JOIN font_removals fr ON f.id = fr.font_id
                WHERE f.name NOT GLOB '[A-Za-z]*'
            ''').fetchone()[0]
        else:
            counts = db.execute('''
                SELECT UPPER(SUBSTR(f.name, 1, 1)) as letter, COUNT(*) as cnt
                FROM fonts f
                WHERE f.id NOT IN (SELECT font_id FROM font_removals)
                AND f.name GLOB '[A-Za-z]*'
                GROUP BY UPPER(SUBSTR(f.name, 1, 1))
            ''').fetchall()
            other_count = db.execute('''
                SELECT COUNT(*) FROM fonts f
                WHERE f.id NOT IN (SELECT font_id FROM font_removals)
                AND f.name NOT GLOB '[A-Za-z]*'
            ''').fetchone()[0]

    letter_counts = {row[0]: row[1] for row in counts}
    letter_counts['#'] = other_count

    return jsonify({
        'fonts': [{'id': f[0], 'name': f[1], 'source': f[2], 'char_count': f[3]} for f in fonts],
        'letter_counts': letter_counts,
        'current_letter': letter
    })


@app.route('/api/find-variations')
def api_find_variations() -> Response:
    """Find font variations (Bold, Italic, Light, etc.) to remove.

    Identifies fonts with variation keywords in their names that should
    be removed to keep only regular variants.

    Returns:
        JSON with list of font variations to remove.
    """
    from stroke_flask import get_db_context

    print("=== find-variations endpoint called ===")  # Debug

    # Keywords that indicate a font variation (not a regular font)
    # Simple substring matching - case insensitive
    variation_keywords = [
        'bold', 'italic', 'light', 'medium', 'thin', 'black', 'heavy',
        'semibold', 'extrabold', 'extralight', 'ultralight', 'ultrabold',
        'demibold', 'condensed', 'expanded', 'narrow', 'wide', 'oblique',
        'slant', 'regular',  # Regular also indicates a variant exists
    ]

    try:
        with get_db_context() as db:
            fonts = db.execute('''
                SELECT f.id, f.name, f.source
                FROM fonts f
                WHERE f.id NOT IN (
                    SELECT font_id FROM font_removals
                )
                ORDER BY f.name
            ''').fetchall()

        logger.info("Found %d fonts to check for variations", len(fonts))

        variations = []
        for fid, name, source in fonts:
            name_lower = name.lower()
            # Check if any variation keyword appears in the font name
            for kw in variation_keywords:
                if kw in name_lower:
                    logger.info("Font '%s' matches keyword '%s'", name, kw)
                    variations.append({
                        'id': fid,
                        'name': name,
                        'source': source
                    })
                    break

        logger.info("Found %d variations", len(variations))
        return jsonify({'ok': True, 'variations': variations})

    except Exception as e:
        logger.error("Failed to find variations: %s", e)
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/find-duplicates', methods=['POST'])
def api_find_duplicates() -> Response:
    """Find duplicate fonts using perceptual hashing.

    Computes perceptual hashes for all active fonts and identifies
    groups of visually similar fonts. Streams progress via SSE.

    Returns:
        flask.Response: SSE stream with progress and final duplicate groups.
    """
    from stroke_flask import get_db_context

    def generate():
        try:
            from font_utils import FontDeduplicator

            # Get all active fonts
            with get_db_context() as db:
                fonts = db.execute('''
                    SELECT f.id, f.name, f.file_path
                    FROM fonts f
                    WHERE f.id NOT IN (
                        SELECT font_id FROM font_removals
                    )
                    ORDER BY f.name
                ''').fetchall()

            total = len(fonts)
            yield f'data: {json.dumps({"status": "starting", "message": f"Computing hashes for {total} fonts...", "total": total})}\n\n'

            dedup = FontDeduplicator(threshold=2)
            font_data = []
            base_dir = Path(__file__).parent

            for i, (fid, name, file_path) in enumerate(fonts, 1):
                full_path = base_dir / file_path
                if not full_path.exists():
                    continue

                phash = dedup.compute_phash(str(full_path))
                if phash:
                    font_data.append({
                        'id': fid,
                        'name': name,
                        'path': file_path,
                        'phash': phash
                    })

                if i % 5 == 0 or i == total:
                    yield f'data: {json.dumps({"status": "hashing", "message": f"Hashed {i}/{total}: {name}", "current": i, "total": total})}\n\n'

            yield f'data: {json.dumps({"status": "analyzing", "message": "Finding duplicate groups..."})}\n\n'

            # Find duplicates
            duplicate_groups = dedup.find_duplicates(font_data)

            # Format results
            groups = []
            for group in duplicate_groups:
                groups.append([{
                    'id': f['id'],
                    'name': f['name'],
                    'path': f['path']
                } for f in group])

            total_duplicates = sum(len(g) - 1 for g in groups)
            yield f'data: {json.dumps({"status": "complete", "message": f"Found {len(groups)} duplicate groups ({total_duplicates} duplicates)", "groups": groups, "total_groups": len(groups), "total_duplicates": total_duplicates})}\n\n'

        except Exception as e:
            logger.error("Deduplication failed: %s", e)
            yield f'data: {json.dumps({"status": "error", "message": f"Error: {str(e)}"})}\n\n'

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/reset-and-scrape', methods=['POST'])
def api_reset_and_scrape() -> Response:
    """Reset database and re-scrape all fonts.

    WARNING: This is a destructive operation that:
    1. Deletes all data from fonts.db
    2. Runs all scrapers (DaFont, FontSpace, Google Fonts)
    3. Re-populates the database with fresh font data

    This operation runs in the background and streams progress via SSE.

    Returns:
        flask.Response: SSE stream with progress updates.

    Example:
        Request::

            POST /api/reset-and-scrape

        SSE Events::

            data: {"status": "starting", "message": "Resetting database..."}
            data: {"status": "scraping", "message": "Running DaFont scraper..."}
            data: {"status": "scraping", "message": "Running FontSpace scraper..."}
            data: {"status": "complete", "message": "Done! Scraped 2500 fonts."}
    """
    import os
    import shutil
    import subprocess
    import sqlite3
    from pathlib import Path

    def generate():
        try:
            # Step 1: Reset database
            yield f'data: {json.dumps({"status": "starting", "message": "Resetting database..."})}\n\n'

            db_path = Path(__file__).parent / 'fonts.db'

            # Close any existing connections and delete the file
            if db_path.exists():
                os.remove(db_path)

            # Recreate database with schema
            yield f'data: {json.dumps({"status": "starting", "message": "Creating fresh database..."})}\n\n'
            result = subprocess.run(
                ['python3', 'setup_database.py'],
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                yield f'data: {json.dumps({"status": "error", "message": f"Database setup failed: {result.stderr}"})}\n\n'
                return

            # Step 1b: Delete existing font directories to start fresh
            yield f'data: {json.dumps({"status": "starting", "message": "Clearing font directories..."})}\n\n'
            font_dirs_to_clear = ['fonts/dafont', 'fonts/fontspace', 'fonts/google']
            for font_dir in font_dirs_to_clear:
                font_path = Path(__file__).parent / font_dir
                if font_path.exists():
                    shutil.rmtree(font_path)
                    logger.info("Deleted font directory: %s", font_path)

            # Signal that everything is cleared - UI should update now
            yield f'data: {json.dumps({"status": "cleared", "message": "Database and fonts cleared. Starting scrapers..."})}\n\n'

            # Step 2: Run scrapers using observer pattern for real-time updates
            # TODO: Remove max_fonts=10 after testing
            from queue import Queue
            from font_source import ScraperConfig, ScraperEvent, ScraperEventType
            from dafont_scraper import DaFontScraper
            from fontspace_scraper import FontSpaceScraper
            from google_fonts_scraper import GoogleFontsScraper

            # Event queue for collecting scraper events
            event_queue = Queue()

            class SSEObserver:
                """Observer that collects events for SSE streaming."""
                def on_event(self, event: ScraperEvent) -> None:
                    event_queue.put(event)

            observer = SSEObserver()

            # Configure scrapers
            base_dir = Path(__file__).parent
            scraper_configs = [
                ('dafont', 'DaFont', DaFontScraper(str(base_dir / 'fonts/dafont')),
                 ScraperConfig()),  # Scrapes all categories until exhausted
                ('fontspace', 'FontSpace', FontSpaceScraper(str(base_dir / 'fonts/fontspace')),
                 ScraperConfig()),  # Scrapes all categories until exhausted
                ('google', 'Google Fonts', GoogleFontsScraper(str(base_dir / 'fonts/google')),
                 ScraperConfig()),  # Fetches all Handwriting fonts from metadata
            ]

            scraper_counts = {}
            for i, (key, name, scraper, config) in enumerate(scraper_configs):
                progress = (i / len(scraper_configs)) * 0.8
                yield f'data: {json.dumps({"status": "scraping", "scraper": key, "message": f"Starting {name} scraper...", "progress": progress})}\n\n'

                # Add observer and run scraper
                scraper.add_observer(observer)

                try:
                    # Run scraper (this populates the queue via observer)
                    import threading

                    def run_scraper():
                        scraper.scrape_and_download(config)

                    thread = threading.Thread(target=run_scraper)
                    thread.start()

                    # Yield events as they come in
                    while thread.is_alive() or not event_queue.empty():
                        try:
                            event = event_queue.get(timeout=0.1)
                            event_data = {
                                "status": "scraping",
                                "scraper": key,
                                "message": f"{name}: {event.message}",
                                "event_type": event.event_type.value
                            }
                            if event.data:
                                event_data.update(event.data)
                            yield f'data: {json.dumps(event_data)}\n\n'
                        except:
                            pass  # Queue timeout, continue checking thread

                    thread.join()

                    # Get final count
                    count = len(scraper.downloaded)
                    scraper_counts[key] = count
                    yield f'data: {json.dumps({"status": "complete", "scraper": key, "count": count, "message": f"{name} complete: {count} fonts downloaded"})}\n\n'

                except Exception as e:
                    logger.error("Scraper %s failed: %s", name, e)
                    scraper_counts[key] = 0
                    yield f'data: {json.dumps({"status": "error", "scraper": key, "message": f"{name} failed: {e}"})}\n\n'

                finally:
                    scraper.remove_observer(observer)

            # Step 3: Scan font directories and register fonts in database
            yield f'data: {json.dumps({"status": "registering", "message": "Registering fonts in database...", "progress": 0.85})}\n\n'

            font_dirs = [
                ('fonts/dafont', 'dafont'),
                ('fonts/fontspace', 'fontspace'),
                ('fonts/google', 'google'),
            ]

            db = sqlite3.connect(str(db_path))
            total_registered = 0
            source_counts = {}

            for font_dir, source in font_dirs:
                font_path = Path(__file__).parent / font_dir
                source_count = 0
                if not font_path.exists():
                    source_counts[source] = 0
                    continue

                # Collect all font files (ttf, otf, woff2 - various cases)
                font_files = (
                    list(font_path.glob('*.[ot]tf')) +
                    list(font_path.glob('*.[OT]TF')) +
                    list(font_path.glob('*.woff2')) +
                    list(font_path.glob('*.woff'))
                )

                for font_file in font_files:
                    name = font_file.stem
                    rel_path = f"{font_dir}/{font_file.name}"

                    try:
                        db.execute("""
                            INSERT OR IGNORE INTO fonts (name, file_path, source)
                            VALUES (?, ?, ?)
                        """, (name, rel_path, source))
                        source_count += 1
                        total_registered += 1
                    except Exception as e:
                        logger.warning("Failed to register font %s: %s", rel_path, e)

                source_counts[source] = source_count
                yield f'data: {json.dumps({"status": "registering", "scraper": source, "count": source_count, "message": f"Registered {source_count} fonts from {source}"})}\n\n'

            db.commit()

            # Count total fonts in database
            total_fonts = db.execute('SELECT COUNT(*) FROM fonts').fetchone()[0]
            db.close()

            # Final summary
            summary = f"Done! {total_fonts} fonts total"
            for source, count in source_counts.items():
                summary += f" | {source}: {count}"
            yield f'data: {json.dumps({"status": "complete", "message": summary, "progress": 1.0})}\n\n'

        except subprocess.TimeoutExpired:
            yield f'data: {json.dumps({"status": "error", "message": "Scraper timed out after 2 hours"})}\n\n'
        except Exception as e:
            yield f'data: {json.dumps({"status": "error", "message": f"Error: {str(e)}"})}\n\n'

    return Response(generate(), mimetype='text/event-stream')


# =============================================================================
# New Background Scraper API Endpoints
# =============================================================================

@app.route('/api/start-scrape', methods=['POST'])
def api_start_scrape() -> Response:
    """Start background scrapers for all sources.

    Creates scraper jobs and starts background workers for each source.
    Returns immediately with job IDs while scrapers run in background.

    Uses the new queue-based architecture with:
    - Discovery thread: Scrapes font listings continuously
    - Download workers: Pool of workers using work-stealing pattern
    - Full recovery: Can resume from any failure point

    Request Body (optional):
        JSON object with options:
            - sources: List of sources to start (default: all)
            - num_workers: Number of download workers (default: 4)

    Returns:
        flask.Response: JSON with job information::

            {
                "ok": true,
                "jobs": {
                    "dafont": {"job_id": 1, "status": "pending"},
                    "fontspace": {"job_id": 2, "status": "pending"},
                    "google": {"job_id": 3, "status": "pending"}
                }
            }
    """
    from scraper_queue import start_scraper

    data = request.get_json(force=False, silent=True) or {}
    sources = data.get('sources', ['dafont', 'fontspace', 'google'])
    num_workers = data.get('num_workers', 4)

    jobs = {}
    for source in sources:
        if source not in ['dafont', 'fontspace', 'google']:
            continue
        try:
            job_id = start_scraper(source, num_workers=num_workers)
            jobs[source] = {'job_id': job_id, 'status': 'discovering'}
        except Exception as e:
            logger.error("Failed to start %s scraper: %s", source, e)
            jobs[source] = {'error': str(e)}

    return success_response(jobs=jobs)


@app.route('/api/scraper-status')
def api_scraper_status() -> Response:
    """Get status of all scraper jobs.

    Returns status information for all active and recent jobs.
    The frontend polls this endpoint to display progress.

    Returns:
        flask.Response: JSON with job statuses::

            {
                "jobs": [
                    {
                        "id": 1,
                        "source": "dafont",
                        "status": "discovering",
                        "current_category": "Script - Handwritten",
                        "categories_total": 44,
                        "categories_done": 12,
                        "fonts_found": 2341,
                        "fonts_downloaded": 1892,
                        "fonts_pending": 449,
                        "fonts_failed": 23,
                        "started_at": "2024-01-15T10:30:00",
                        "completed_at": null,
                        "error_message": null
                    },
                    ...
                ]
            }
    """
    from scraper_queue import QueueRepository

    repo = QueueRepository()
    jobs = repo.get_recent_jobs(limit=20)
    return jsonify(jobs=jobs)


@app.route('/api/scraper-logs')
def api_scraper_logs() -> Response:
    """Get recent scraper log entries.

    Query params:
        job_id: Optional job ID to filter logs
        limit: Max number of logs to return (default 100)

    Returns:
        flask.Response: JSON with log entries.
    """
    from scraper_queue import QueueRepository

    job_id = request.args.get('job_id', type=int)
    limit = request.args.get('limit', 100, type=int)

    repo = QueueRepository()
    logs = repo.get_logs(job_id=job_id, limit=limit)

    # Reverse to show oldest first (for display)
    logs = list(reversed(logs))

    return jsonify(logs=logs)


@app.route('/api/scraper-status/<int:job_id>')
def api_scraper_job_status(job_id: int) -> Response:
    """Get detailed status for a specific scraper job.

    Args:
        job_id: The job ID.

    Returns:
        flask.Response: JSON with detailed job status including
            font counts by status.
    """
    from scraper_queue import QueueRepository

    repo = QueueRepository()
    job = repo.get_job(job_id)
    if not job:
        return error_response("Job not found", 404)

    # Get font status breakdown
    font_counts = repo.get_font_counts(job_id)

    result = dict(job)
    result['font_counts'] = font_counts

    return jsonify(result)


@app.route('/api/stop-scraper/<int:job_id>', methods=['POST'])
def api_stop_scraper(job_id: int) -> Response:
    """Stop a running scraper job.

    Args:
        job_id: The job ID to stop.

    Returns:
        flask.Response: JSON with result::

            {"ok": true, "status": "cancelled"}
            or
            {"ok": true, "status": "not_running"}
    """
    from scraper_queue import stop_scraper

    if stop_scraper(job_id):
        return success_response(status='cancelled')
    return success_response(status='not_running')


@app.route('/api/stop-all-scrapers', methods=['POST'])
def api_stop_all_scrapers() -> Response:
    """Stop all running scraper jobs.

    Returns:
        flask.Response: JSON with count of stopped jobs.
    """
    from scraper_queue import QueueRepository, stop_scraper

    repo = QueueRepository()
    jobs = repo.get_active_jobs()
    stopped = 0
    for job in jobs:
        if stop_scraper(job['id']):
            stopped += 1

    return success_response(stopped=stopped)


@app.route('/api/pause-scraper', methods=['POST'])
def api_pause_scraper() -> Response:
    """Pause the running scraper (can be resumed later).

    Returns:
        flask.Response: JSON with result::

            {"ok": true, "status": "paused"}
            or
            {"ok": true, "status": "not_running"}
    """
    from scraper_queue import pause_scraper

    if pause_scraper():
        return success_response(status='paused')
    return success_response(status='not_running')


@app.route('/api/resume-scraper/<int:job_id>', methods=['POST'])
def api_resume_scraper(job_id: int) -> Response:
    """Resume a paused or interrupted scraper job.

    Args:
        job_id: The job ID to resume.

    Request Body (optional):
        JSON object with options:
            - num_workers: Number of download workers (default: 4)

    Returns:
        flask.Response: JSON with result::

            {"ok": true, "status": "resumed"}
            or
            {"ok": false, "error": "Job not found or cannot be resumed"}
    """
    from scraper_queue import resume_scraper

    data = request.get_json(force=False, silent=True) or {}
    num_workers = data.get('num_workers', 4)

    if resume_scraper(job_id, num_workers=num_workers):
        return success_response(status='resumed')
    return error_response("Job not found or cannot be resumed", 400)


@app.route('/api/reset-database', methods=['POST'])
def api_reset_database() -> Response:
    """Reset the database (delete all font data).

    This is a destructive operation that:
    1. Stops all running scrapers
    2. Deletes all data from fonts, scraper_jobs, scraper_fonts tables
    3. Deletes downloaded font files

    Returns:
        flask.Response: JSON with result.
    """
    from stroke_flask import get_db_context
    from scraper_queue import QueueRepository, stop_scraper

    try:
        # Stop all running scrapers first
        repo = QueueRepository()
        jobs = repo.get_active_jobs()
        for job in jobs:
            stop_scraper(job['id'])

        # Clear database tables
        with get_db_context() as db:
            db.execute("DELETE FROM scraper_logs")
            db.execute("DELETE FROM scraper_fonts")
            db.execute("DELETE FROM scraper_jobs")
            db.execute("DELETE FROM characters")
            db.execute("DELETE FROM font_checks")
            db.execute("DELETE FROM font_removals")
            db.execute("DELETE FROM fonts")

        # Delete font directories
        font_dirs = ['fonts/dafont', 'fonts/fontspace', 'fonts/google']
        for font_dir in font_dirs:
            font_path = Path(__file__).parent / font_dir
            if font_path.exists():
                shutil.rmtree(font_path)
                logger.info("Deleted font directory: %s", font_path)

        return success_response(message="Database reset complete")

    except Exception as e:
        logger.error("Database reset failed: %s", e)
        return error_response(f"Reset failed: {str(e)}", 500)


# =============================================================================
# Export / Import Fonts API Endpoints
# =============================================================================

@app.route('/api/export-fonts')
def api_export_fonts() -> Response:
    """Export all accepted fonts as a ZIP file.

    Creates a ZIP archive containing:
    - manifest.json: Metadata for all exported fonts
    - fonts/<source>/<filename>: Font files organized by source

    An "accepted" font is one with NO entry in the font_removals table.

    Returns:
        flask.Response: ZIP file as attachment download.

    Example:
        GET /api/export-fonts

        Returns: fonts_export_20260213_193000.zip
    """
    from stroke_flask import get_db_context

    base_dir = Path(__file__).parent

    try:
        # Query accepted fonts (not in font_removals)
        with get_db_context() as db:
            fonts = db.execute('''
                SELECT f.id, f.name, f.source, f.category, f.url, f.file_path
                FROM fonts f
                WHERE f.id NOT IN (
                    SELECT font_id FROM font_removals
                )
                ORDER BY f.source, f.name
            ''').fetchall()

        if not fonts:
            return error_response("No fonts to export", 400)

        # Build manifest
        manifest = {
            'exported_at': datetime.now().isoformat(),
            'font_count': len(fonts),
            'fonts': []
        }

        # Create ZIP in memory
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for font in fonts:
                font_path = base_dir / font['file_path']

                if not font_path.exists():
                    logger.warning("Font file not found, skipping: %s", font_path)
                    continue

                # Determine archive path (fonts/<source>/<filename>)
                source = font['source'] or 'unknown'
                archive_path = f"fonts/{source}/{font_path.name}"

                # Add font file to ZIP
                zf.write(font_path, archive_path)

                # Add to manifest
                manifest['fonts'].append({
                    'name': font['name'],
                    'source': source,
                    'category': font['category'],
                    'file_path': archive_path,
                    'url': font['url']
                })

        # Update manifest font count (may differ if some files were missing)
        manifest['font_count'] = len(manifest['fonts'])

        if manifest['font_count'] == 0:
            return error_response("No font files found to export", 400)

        # Add manifest to ZIP
        buf.seek(0)
        # Re-open to add manifest
        with zipfile.ZipFile(buf, 'a', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('manifest.json', json.dumps(manifest, indent=2))

        buf.seek(0)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'fonts_export_{timestamp}.zip'

        logger.info("Exported %d fonts to %s", manifest['font_count'], filename)

        return send_file(
            buf,
            mimetype='application/zip',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.error("Export failed: %s", e)
        return error_response(f"Export failed: {str(e)}", 500)


@app.route('/api/import-fonts', methods=['POST'])
def api_import_fonts() -> Response:
    """Import fonts from an uploaded ZIP file.

    Extracts font files from the ZIP and registers them in the database.
    Expects a ZIP with the structure created by /api/export-fonts:
    - manifest.json (optional but recommended)
    - fonts/<source>/<filename>

    Request:
        POST with multipart/form-data containing 'file' field with ZIP.

    Returns:
        flask.Response: JSON with import statistics::

            {
                "ok": true,
                "imported": 150,
                "skipped": 5,
                "errors": 2
            }

    Example:
        POST /api/import-fonts
        Content-Type: multipart/form-data
        file: fonts_export_20260213_193000.zip
    """
    from stroke_flask import get_db_context

    if 'file' not in request.files:
        return error_response("No file uploaded", 400)

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return error_response("No file selected", 400)

    if not uploaded_file.filename.endswith('.zip'):
        return error_response("File must be a ZIP archive", 400)

    base_dir = Path(__file__).parent
    imported = 0
    skipped = 0
    errors = 0

    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
            uploaded_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            with zipfile.ZipFile(tmp_path, 'r') as zf:
                # Read manifest if present
                manifest = None
                if 'manifest.json' in zf.namelist():
                    manifest_data = zf.read('manifest.json')
                    manifest = json.loads(manifest_data)
                    logger.info("Found manifest with %d fonts", manifest.get('font_count', 0))

                # Get list of font files in ZIP
                font_files = [
                    name for name in zf.namelist()
                    if name.startswith('fonts/') and
                    name.lower().endswith(('.ttf', '.otf', '.woff', '.woff2'))
                ]

                if not font_files:
                    return error_response("No font files found in ZIP", 400)

                # Build manifest lookup for metadata
                manifest_lookup = {}
                if manifest and manifest.get('fonts'):
                    for font_meta in manifest['fonts']:
                        manifest_lookup[font_meta['file_path']] = font_meta

                # Extract and register fonts
                with get_db_context() as db:
                    for archive_path in font_files:
                        try:
                            # Determine destination path
                            # archive_path is like: fonts/dafont/FontName.ttf
                            dest_path = base_dir / archive_path

                            # Create directory if needed
                            dest_path.parent.mkdir(parents=True, exist_ok=True)

                            # Check if file already exists
                            if dest_path.exists():
                                logger.debug("Font file already exists: %s", dest_path)
                                # Still try to register in DB if not present
                            else:
                                # Extract file
                                with zf.open(archive_path) as src:
                                    with open(dest_path, 'wb') as dst:
                                        dst.write(src.read())

                            # Get metadata from manifest or derive from path
                            if archive_path in manifest_lookup:
                                meta = manifest_lookup[archive_path]
                                name = meta['name']
                                source = meta['source']
                                category = meta.get('category')
                                url = meta.get('url')
                            else:
                                # Derive from path
                                parts = archive_path.split('/')
                                source = parts[1] if len(parts) > 2 else 'unknown'
                                name = dest_path.stem
                                category = None
                                url = None

                            # Check if font already in database (by file_path)
                            existing = db.execute(
                                "SELECT id FROM fonts WHERE file_path = ?",
                                (archive_path,)
                            ).fetchone()

                            if existing:
                                skipped += 1
                                continue

                            # Insert into database
                            db.execute("""
                                INSERT INTO fonts (name, file_path, source, category, url)
                                VALUES (?, ?, ?, ?, ?)
                            """, (name, archive_path, source, category, url))
                            imported += 1

                        except Exception as e:
                            logger.warning("Failed to import %s: %s", archive_path, e)
                            errors += 1

        finally:
            # Clean up temp file
            os.unlink(tmp_path)

        logger.info("Import complete: %d imported, %d skipped, %d errors",
                    imported, skipped, errors)

        return success_response(imported=imported, skipped=skipped, errors=errors)

    except zipfile.BadZipFile:
        return error_response("Invalid ZIP file", 400)
    except Exception as e:
        logger.error("Import failed: %s", e)
        return error_response(f"Import failed: {str(e)}", 500)


@app.route('/api/import-default', methods=['POST'])
def api_import_default() -> Response:
    """Import fonts from default export ZIPs in exports/.

    Looks for exports/fonts_export_part*.zip (split archives) or
    exports/fonts_export_default.zip (single archive) and imports all.

    Returns:
        flask.Response: JSON with import statistics.
    """
    from stroke_flask import get_db_context

    base_dir = Path(__file__).parent
    exports_dir = base_dir / 'exports'

    # Find zip files: prefer split parts, fall back to single default
    zip_files = sorted(exports_dir.glob('fonts_export_part*.zip'))
    if not zip_files:
        default_zip = exports_dir / 'fonts_export_default.zip'
        if default_zip.exists():
            zip_files = [default_zip]

    if not zip_files:
        return error_response("No export ZIPs found in exports/", 404)

    imported = 0
    skipped = 0
    errors = 0

    try:
        for zip_path in zip_files:
            logger.info("Importing from %s", zip_path.name)

            with zipfile.ZipFile(zip_path, 'r') as zf:
                manifest = None
                if 'manifest.json' in zf.namelist():
                    manifest_data = zf.read('manifest.json')
                    manifest = json.loads(manifest_data)

                font_files = [
                    name for name in zf.namelist()
                    if name.startswith('fonts/') and
                    name.lower().endswith(('.ttf', '.otf', '.woff', '.woff2'))
                ]

                manifest_lookup = {}
                if manifest and manifest.get('fonts'):
                    for font_meta in manifest['fonts']:
                        manifest_lookup[font_meta['file_path']] = font_meta

                with get_db_context() as db:
                    for archive_path in font_files:
                        try:
                            dest_path = base_dir / archive_path
                            dest_path.parent.mkdir(parents=True, exist_ok=True)

                            if not dest_path.exists():
                                with zf.open(archive_path) as src:
                                    with open(dest_path, 'wb') as dst:
                                        dst.write(src.read())

                            if archive_path in manifest_lookup:
                                meta = manifest_lookup[archive_path]
                                name = meta['name']
                                source = meta['source']
                                category = meta.get('category')
                                url = meta.get('url')
                            else:
                                parts = archive_path.split('/')
                                source = parts[1] if len(parts) > 2 else 'unknown'
                                name = dest_path.stem
                                category = None
                                url = None

                            existing = db.execute(
                                "SELECT id FROM fonts WHERE file_path = ?",
                                (archive_path,)
                            ).fetchone()

                            if existing:
                                skipped += 1
                                continue

                            db.execute("""
                                INSERT INTO fonts (name, file_path, source, category, url)
                                VALUES (?, ?, ?, ?, ?)
                            """, (name, archive_path, source, category, url))
                            imported += 1

                        except Exception as e:
                            logger.warning("Failed to import %s: %s", archive_path, e)
                            errors += 1

        logger.info("Default import complete: %d imported, %d skipped, %d errors",
                    imported, skipped, errors)

        return success_response(imported=imported, skipped=skipped, errors=errors)

    except zipfile.BadZipFile:
        return error_response("Export ZIP file is corrupt", 400)
    except Exception as e:
        logger.error("Default import failed: %s", e)
        return error_response(f"Import failed: {str(e)}", 500)


@app.route('/api/default-export-exists')
def api_default_export_exists() -> Response:
    """Check if default export ZIPs exist in exports/.

    Returns:
        flask.Response: JSON with exists flag and file info.
    """
    base_dir = Path(__file__).parent
    exports_dir = base_dir / 'exports'

    parts = sorted(exports_dir.glob('fonts_export_part*.zip'))
    if parts:
        total_mb = sum(p.stat().st_size for p in parts) / (1024 * 1024)
        return success_response(
            exists=True, parts=len(parts), size_mb=round(total_mb, 1)
        )

    default_zip = exports_dir / 'fonts_export_default.zip'
    if default_zip.exists():
        size_mb = default_zip.stat().st_size / (1024 * 1024)
        return success_response(exists=True, parts=1, size_mb=round(size_mb, 1))

    return success_response(exists=False)


@app.route('/api/accepted-font-count')
def api_accepted_font_count() -> Response:
    """Get count of accepted fonts (not rejected).

    Returns:
        flask.Response: JSON with count::

            {"count": 203}
    """
    from stroke_flask import get_db_context

    with get_db_context() as db:
        count = db.execute('''
            SELECT COUNT(*) FROM fonts f
            WHERE f.id NOT IN (
                SELECT font_id FROM font_removals
            )
        ''').fetchone()[0]

    return jsonify(count=count)


@app.route('/api/register-existing-fonts', methods=['POST'])
def api_register_existing_fonts() -> Response:
    """Register font files on disk that aren't in the database.

    Scans font directories and registers any font files that exist on disk
    but aren't in the fonts table. Useful after database resets or when
    fonts were downloaded but not registered.

    Returns:
        flask.Response: JSON with registration statistics::

            {
                "ok": true,
                "registered": 150,
                "already_registered": 1200,
                "errors": 5
            }
    """
    from stroke_flask import get_db_context

    base_dir = Path(__file__).parent
    registered = 0
    already_registered = 0
    errors = 0

    font_dirs = [
        ('fonts/dafont', 'dafont'),
        ('fonts/fontspace', 'fontspace'),
        ('fonts/google', 'google'),
    ]

    try:
        with get_db_context() as db:
            for font_dir, source in font_dirs:
                font_path = base_dir / font_dir
                if not font_path.exists():
                    continue

                # Get all font files
                font_files = (
                    list(font_path.glob('*.[ot]tf')) +
                    list(font_path.glob('*.[OT]TF')) +
                    list(font_path.glob('*.woff2')) +
                    list(font_path.glob('*.woff'))
                )

                for font_file in font_files:
                    rel_path = f"{font_dir}/{font_file.name}"

                    # Check if already registered
                    existing = db.execute(
                        "SELECT id FROM fonts WHERE file_path = ?",
                        (rel_path,)
                    ).fetchone()

                    if existing:
                        already_registered += 1
                        continue

                    # Register the font
                    try:
                        name = font_file.stem
                        db.execute("""
                            INSERT INTO fonts (name, file_path, source)
                            VALUES (?, ?, ?)
                        """, (name, rel_path, source))
                        registered += 1
                    except Exception as e:
                        logger.warning("Failed to register %s: %s", rel_path, e)
                        errors += 1

        logger.info("Registered %d existing fonts, %d already registered, %d errors",
                    registered, already_registered, errors)

        return success_response(
            registered=registered,
            already_registered=already_registered,
            errors=errors
        )

    except Exception as e:
        logger.error("Font registration failed: %s", e)
        return error_response(f"Registration failed: {str(e)}", 500)
