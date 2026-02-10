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

import numpy as np
from flask import jsonify, render_template, request, send_file
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from skimage.morphology import thin
from stroke_flask import (
    CHARS,
    STROKE_COLORS,
    app,
    get_db,
    get_font,
    resolve_font_path,
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


# Alias for backward compatibility
_font = get_font


@app.route('/')
def font_list():
    """Display the font list page showing all available fonts.

    Renders a list of fonts from the database with character count statistics.
    Can optionally show only rejected fonts or filter out rejected/duplicate fonts.

    Query Parameters:
        rejected (str, optional): Set to '1' to show only rejected fonts.
            When not set or set to any other value, shows non-rejected fonts
            excluding duplicates.

    Returns:
        str: Rendered HTML template 'font_list.html' with context:
            - fonts: List of font records with id, name, source, file_path,
              char_count (number of characters with stroke data), and rejected flag.
            - show_rejected: Boolean indicating if viewing rejected fonts.

    Template Context:
        fonts (list[sqlite3.Row]): Font records with the following fields:
            - id (int): Font database ID
            - name (str): Font name
            - source (str): Font source/origin
            - file_path (str): Path to font file
            - char_count (int): Number of characters with stroke data
            - rejected (int): 1 if viewing rejected fonts, 0 otherwise
        show_rejected (bool): True if ?rejected=1 was passed

    Example:
        View all non-rejected fonts::

            GET /

        View rejected fonts only::

            GET /?rejected=1
    """
    db, rej = get_db(), request.args.get('rejected') == '1'
    q = "SELECT f.id, f.name, f.source, f.file_path, COALESCE(cs.char_count, 0) as char_count, {} as rejected FROM fonts f {} LEFT JOIN (SELECT font_id, COUNT(*) as char_count FROM characters WHERE strokes_raw IS NOT NULL GROUP BY font_id) cs ON cs.font_id = f.id {} ORDER BY f.name"
    fonts = db.execute(q.format('1', 'JOIN font_removals fr ON fr.font_id = f.id AND fr.reason_id = 8', '') if rej else q.format('0', 'LEFT JOIN font_removals rej ON rej.font_id = f.id AND rej.reason_id = 8 LEFT JOIN font_removals dup ON dup.font_id = f.id AND dup.reason_id = 2', 'WHERE rej.id IS NULL AND dup.id IS NULL')).fetchall()
    db.close()
    return render_template('font_list.html', fonts=fonts, show_rejected=rej)


@app.route('/font/<int:fid>')
def char_grid(fid):
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
    f = _font(fid)
    return render_template('editor.html', font=f, char=c, char_list=CHARS) if f else ("Font not found", 404)


@app.route('/api/char/<int:fid>')
def api_get_char(fid):
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
    """Batch check and reject all fonts with connected letters or quality issues.

    Iterates through all non-rejected fonts in the database, applies the same
    quality checks as api_check_connected, and automatically marks failing
    fonts as rejected with reason_id 8.

    Returns:
        flask.Response: JSON response::

            {
                "ok": true,
                "checked": 150,   // Number of fonts successfully checked
                "rejected": 23    // Number of fonts rejected in this run
            }

    Quality Checks:
        Same as api_check_connected:
        - Shape count for "Hello World" must be between 10-15
        - Maximum stroke width ratio must not exceed 0.225
        - Letter 'l' must not have holes
        - Character '!' must have exactly 2 shapes
        - No case mismatches between upper and lower case letters

    Note:
        - Skips fonts that are already rejected (reason_id = 8)
        - Silently skips fonts that cannot be loaded or rendered
        - This is a potentially long-running operation for large font databases

    Example:
        Request::

            POST /api/reject-connected

        Response::

            {"ok": true, "checked": 150, "rejected": 23}
    """
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
def api_snap(fid):
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
    db = get_db()
    db.execute("DELETE FROM font_removals WHERE font_id = ? AND reason_id = 8", (fid,))
    db.commit()
    db.close()
    return jsonify(ok=True, status='unrejected')


@app.route('/api/unreject-all', methods=['POST'])
def api_unreject_all():
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
    db = get_db()
    r = db.execute("DELETE FROM font_removals WHERE reason_id = 8")
    db.commit()
    db.close()
    return jsonify(ok=True, restored=r.rowcount)
