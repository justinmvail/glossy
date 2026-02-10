"""Flask routes for SSE streaming optimization.

This module provides Server-Sent Events (SSE) endpoints for real-time
optimization streaming to the frontend. It enables progressive updates
during computationally intensive stroke optimization operations.

The streaming approach allows the frontend to:
- Display intermediate optimization results in real-time
- Show progress through different optimization phases
- Provide visual feedback during long-running operations
- Cancel or interrupt optimization if needed

SSE Event Format:
    All events are sent as JSON-encoded data in the standard SSE format::

        data: {"phase": "...", "frame": N, "score": 0.XXX, ...}

    Common event fields:
        - phase (str): Current optimization phase name
        - frame (int): Sequential frame number for ordering
        - score (float): Current optimization score (0-1, higher is better)
        - strokes (list): Current stroke data as nested coordinate lists
        - elapsed (float): Time elapsed since start in seconds
        - done (bool): True only in final event
        - error (str): Error message if operation failed
        - reason (str): Completion reason in final event ('perfect', 'converged', 'time limit')

Example SSE Stream:
    A typical optimization stream produces events like::

        data: {"phase": "Initializing", "frame": 0, "score": 0}

        data: {"phase": "Initial", "frame": 1, "score": 0.65, "strokes": [...]}

        data: {"phase": "Global affine (NM)", "frame": 1, "score": 0.65}

        data: {"phase": "NM iter 50", "frame": 2, "score": 0.72, "strokes": [...]}

        data: {"phase": "Complete", "frame": 15, "score": 0.89, "strokes": [...],
               "done": true, "reason": "converged", "elapsed": 2.34, "cycles": 15}

Attributes:
    None (all state is local to request handlers).

See Also:
    - stroke_routes_batch: Non-streaming batch operations
    - stroke_pipeline_stream: Streaming pipeline for minimal strokes
"""

import json
import time
from collections.abc import Generator

import numpy as np
from flask import Response, request
from scipy.ndimage import distance_transform_edt
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree
from stroke_core import min_strokes, skel_strokes
from stroke_flask import app, get_db, get_font, resolve_font_path
from stroke_rendering import render_glyph_mask
from stroke_scoring import score_raw_strokes
from stroke_shapes import adaptive_radius, make_point_cloud


# Alias for backward compatibility
_font = get_font


def _sse_event(data: dict) -> str:
    """Format a dictionary as a Server-Sent Events data line.

    Args:
        data: Dictionary to serialize as JSON event data.

    Returns:
        str: SSE-formatted string with 'data: ' prefix and double newline.

    Example:
        >>> _sse_event({'phase': 'Init', 'score': 0.5})
        'data: {"phase": "Init", "score": 0.5}\\n\\n'
    """
    return f'data: {json.dumps(data)}\n\n'


def _prepare_optimization(mask: np.ndarray, strokes_raw: list) -> tuple | None:
    """Prepare data structures needed for stroke optimization.

    Creates spatial index, distance map, and other structures used
    by the scoring function during optimization iterations.

    Args:
        mask: Binary numpy array where True indicates glyph pixels.
            Shape should be (height, width).
        strokes_raw: List of strokes, where each stroke is a list of
            [x, y] coordinate pairs.

    Returns:
        tuple | None: If successful, returns a tuple containing:
            - stroke_arrays (list[np.ndarray]): Strokes as numpy arrays
            - centroid (tuple[float, float]): Center of glyph mass (x, y)
            - score_args (tuple): Arguments for score_raw_strokes function
            - w (int): Mask width
            - h (int): Mask height

        Returns None if:
            - Mask has no foreground pixels
            - Point cloud has fewer than 10 points
            - No valid strokes (each needs at least 2 points)
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None

    h, w = mask.shape
    cloud = make_point_cloud(mask, spacing=3)
    if len(cloud) < 10:
        return None

    cloud_tree = cKDTree(cloud)
    n_cloud = len(cloud)
    radius = adaptive_radius(mask, spacing=3)
    dist_map = distance_transform_edt(mask)
    _, snap_indices = distance_transform_edt(~mask, return_indices=True)
    snap_yi, snap_xi = snap_indices[0], snap_indices[1]

    score_args = (cloud_tree, n_cloud, radius, snap_yi, snap_xi, w, h, dist_map)

    # Convert raw strokes to numpy arrays
    stroke_arrays = []
    for s in strokes_raw:
        if len(s) >= 2:
            stroke_arrays.append(np.array([[float(p[0]), float(p[1])] for p in s]))

    if not stroke_arrays:
        return None

    centroid = (float(cols.mean()), float(rows.mean()))
    return stroke_arrays, centroid, score_args, w, h


def _affine_transform(strokes: list[np.ndarray], params: tuple,
                      centroid: tuple[float, float]) -> list[np.ndarray]:
    """Apply an affine transformation to a list of strokes.

    Transforms strokes by applying translation, scaling, rotation,
    and shear around a center point.

    Args:
        strokes: List of stroke arrays, each with shape (N, 2) for N points.
        params: Transformation parameters as a 6-tuple:
            - tx (float): X translation in pixels
            - ty (float): Y translation in pixels
            - sx (float): X scale factor (1.0 = no change)
            - sy (float): Y scale factor (1.0 = no change)
            - theta_deg (float): Rotation angle in degrees
            - shear (float): Shear factor for skewing
        centroid: Center point (x, y) around which to apply rotation/scaling.

    Returns:
        list[np.ndarray]: Transformed strokes as new numpy arrays.
        Original arrays are not modified.
    """
    tx, ty, sx, sy, theta_deg, shear = params
    theta = np.radians(theta_deg)
    cx, cy = centroid
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    result = []
    for stroke in strokes:
        pts = stroke.copy()
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        rx = dx * cos_t - dy * sin_t + shear * dy
        ry = dx * sin_t + dy * cos_t
        pts[:, 0] = cx + sx * rx + tx
        pts[:, 1] = cy + sy * ry + ty
        result.append(pts)
    return result


def _strokes_to_list(strokes: list[np.ndarray]) -> list:
    """Convert numpy stroke arrays to JSON-serializable nested lists.

    Args:
        strokes: List of numpy arrays with shape (N, 2) each.

    Returns:
        list: Nested list structure suitable for JSON serialization.
        Coordinates are rounded to 1 decimal place for compactness.

    Example:
        >>> arr = [np.array([[100.123, 50.456], [200.789, 150.012]])]
        >>> _strokes_to_list(arr)
        [[[100.1, 50.5], [200.8, 150.0]]]
    """
    return [[[round(float(p[0]), 1), round(float(p[1]), 1)] for p in s] for s in strokes]


def optimize_stream_generator(font_path: str, char: str,
                              canvas_size: int = 224) -> Generator[str, None, None]:
    """Generator that yields SSE events during multi-phase stroke optimization.

    Performs progressive optimization of strokes to match a target glyph,
    yielding intermediate results as SSE events for real-time display.

    Optimization Phases:
        1. **Initialization**: Generate initial strokes from template or skeleton
        2. **Global Affine (NM)**: Nelder-Mead optimization of affine transform
        3. **Global Affine (DE)**: Differential Evolution refinement
        4. **Per-stroke Refine**: Individual stroke position/scale adjustment
        5. **Complete**: Final result with completion metadata

    Args:
        font_path: Path to the font file (TTF/OTF).
        char: Single character to optimize strokes for.
        canvas_size: Size of the square canvas in pixels. Defaults to 224.

    Yields:
        str: SSE-formatted event strings. Each event contains a JSON object
        with varying fields depending on the phase.

    SSE Event Types:
        Phase start events::

            {"phase": "Initializing", "frame": 0, "score": 0}
            {"phase": "Global affine (NM)", "frame": N, "score": X.XXX}

        Progress events (include current strokes)::

            {
                "phase": "NM iter 50",
                "frame": N,
                "score": X.XXX,
                "strokes": [[[x, y], ...], ...]
            }

        Completion event::

            {
                "phase": "Complete",
                "frame": N,
                "score": X.XXX,
                "strokes": [...],
                "done": true,
                "reason": "converged",  // or "perfect", "time limit", "simple"
                "elapsed": X.XX,
                "cycles": N
            }

        Error events::

            {"error": "Failed to get initial strokes: <details>"}
            {"error": "No strokes generated"}
            {"error": "Could not render glyph mask"}

    Note:
        - The generator may terminate early if initial stroke generation fails
        - Score values are negated internally (minimization) but reported as
          positive values (higher = better)
        - Progress events are throttled to avoid overwhelming the client
    """
    start_time = time.time()
    frame = 0

    # Phase 0: Get initial strokes
    yield _sse_event({'phase': 'Initializing', 'frame': frame, 'score': 0})

    try:
        strokes_raw = min_strokes(font_path, char, canvas_size)
    except Exception as e:
        yield _sse_event({'error': f'Failed to get initial strokes: {e}'})
        return

    if not strokes_raw:
        # Try skeleton fallback
        try:
            mask = render_glyph_mask(resolve_font_path(font_path), char, canvas_size)
            if mask is not None:
                strokes_raw = skel_strokes(mask, min_len=5)
        except Exception:
            pass

    if not strokes_raw:
        yield _sse_event({'error': 'No strokes generated'})
        return

    # Render mask and prepare optimization
    mask = render_glyph_mask(resolve_font_path(font_path), char, canvas_size)
    if mask is None:
        yield _sse_event({'error': 'Could not render glyph mask'})
        return

    setup = _prepare_optimization(mask, strokes_raw)
    if setup is None:
        # Return initial strokes without optimization
        yield _sse_event({
            'strokes': strokes_raw,
            'frame': 1,
            'score': 0.5,
            'phase': 'No optimization needed',
            'done': True,
            'reason': 'simple',
            'elapsed': round(time.time() - start_time, 2),
            'cycles': 0
        })
        return

    stroke_arrays, centroid, score_args, _w, _h = setup

    # Emit initial strokes
    frame += 1
    initial_score = -score_raw_strokes(stroke_arrays, *score_args)
    yield _sse_event({
        'strokes': _strokes_to_list(stroke_arrays),
        'frame': frame,
        'score': round(initial_score, 3),
        'phase': 'Initial'
    })

    best_strokes = stroke_arrays
    best_score = -initial_score  # negative for minimization

    # Phase 1: Global affine optimization with Nelder-Mead
    yield _sse_event({'phase': 'Global affine (NM)', 'frame': frame, 'score': round(-best_score, 3)})

    def affine_obj(params):
        transformed = _affine_transform(stroke_arrays, params, centroid)
        return score_raw_strokes(transformed, *score_args)

    x0 = np.array([0, 0, 1, 1, 0, 0], dtype=float)
    nm_evals = [0]
    last_emit = [time.time()]

    def nm_callback(xk):
        nm_evals[0] += 1
        # Emit every 50 evaluations or 0.1 seconds
        if nm_evals[0] % 50 == 0 or time.time() - last_emit[0] > 0.1:
            transformed = _affine_transform(stroke_arrays, tuple(xk), centroid)
            score = -score_raw_strokes(transformed, *score_args)
            nonlocal frame
            frame += 1
            last_emit[0] = time.time()
            return _sse_event({
                'strokes': _strokes_to_list(transformed),
                'frame': frame,
                'score': round(score, 3),
                'phase': f'NM iter {nm_evals[0]}'
            })
        return None

    # Run NM with periodic yields
    try:
        nm = minimize(affine_obj, x0, method='Nelder-Mead',
                      options={'maxfev': 400, 'xatol': 0.2, 'fatol': 0.005, 'adaptive': True})
        if nm.fun < best_score:
            best_score = nm.fun
            best_strokes = _affine_transform(stroke_arrays, tuple(nm.x), centroid)
            frame += 1
            yield _sse_event({
                'strokes': _strokes_to_list(best_strokes),
                'frame': frame,
                'score': round(-best_score, 3),
                'phase': 'NM complete'
            })
    except Exception:
        pass

    # Phase 2: Differential Evolution refinement
    yield _sse_event({'phase': 'Global affine (DE)', 'frame': frame, 'score': round(-best_score, 3)})

    affine_bounds = [(-15, 15), (-15, 15), (0.75, 1.25), (0.75, 1.25), (-10, 10), (-0.2, 0.2)]
    de_gens = [0]

    def de_callback(xk, convergence=None):
        de_gens[0] += 1
        transformed = _affine_transform(stroke_arrays, tuple(xk), centroid)
        score = -score_raw_strokes(transformed, *score_args)
        nonlocal frame, best_score, best_strokes
        if score > -best_score:
            best_score = -score
            best_strokes = transformed
        frame += 1
        return False  # Don't stop

    try:
        de = differential_evolution(affine_obj, bounds=affine_bounds,
                                    x0=nm.x if 'nm' in dir() else x0,
                                    maxiter=15, popsize=8, tol=0.01, polish=False,
                                    callback=de_callback)
        if de.fun < best_score:
            best_score = de.fun
            best_strokes = _affine_transform(stroke_arrays, tuple(de.x), centroid)

        frame += 1
        yield _sse_event({
            'strokes': _strokes_to_list(best_strokes),
            'frame': frame,
            'score': round(-best_score, 3),
            'phase': f'DE gen {de_gens[0]}'
        })
    except Exception:
        pass

    # Phase 3: Per-stroke refinement
    yield _sse_event({'phase': 'Per-stroke refine', 'frame': frame, 'score': round(-best_score, 3)})

    n_strokes = len(best_strokes)

    def per_stroke_obj(params):
        adjusted = []
        for si, base in enumerate(best_strokes):
            dx, dy, sx, sy = params[si * 4:(si + 1) * 4]
            pts = base.copy()
            c = pts.mean(axis=0)
            pts[:, 0] = c[0] + sx * (pts[:, 0] - c[0]) + dx
            pts[:, 1] = c[1] + sy * (pts[:, 1] - c[1]) + dy
            adjusted.append(pts)
        return score_raw_strokes(adjusted, *score_args)

    x0_per = np.array([0, 0, 1, 1] * n_strokes, dtype=float)

    try:
        nm2 = minimize(per_stroke_obj, x0_per, method='Nelder-Mead',
                       options={'maxfev': 800, 'xatol': 0.15, 'fatol': 0.005, 'adaptive': True})

        if nm2.fun < best_score:
            final_strokes = []
            for si, base in enumerate(best_strokes):
                dx, dy, sx, sy = nm2.x[si * 4:(si + 1) * 4]
                pts = base.copy()
                c = pts.mean(axis=0)
                pts[:, 0] = c[0] + sx * (pts[:, 0] - c[0]) + dx
                pts[:, 1] = c[1] + sy * (pts[:, 1] - c[1]) + dy
                final_strokes.append(pts)
            best_strokes = final_strokes
            best_score = nm2.fun

        frame += 1
        yield _sse_event({
            'strokes': _strokes_to_list(best_strokes),
            'frame': frame,
            'score': round(-best_score, 3),
            'phase': 'Per-stroke complete'
        })
    except Exception:
        pass

    # Final result
    elapsed = round(time.time() - start_time, 2)
    final_score = round(-best_score, 3)

    # Determine completion reason
    if final_score >= 0.95:
        reason = 'perfect'
    elif elapsed > 30:
        reason = 'time limit'
    else:
        reason = 'converged'

    yield _sse_event({
        'strokes': _strokes_to_list(best_strokes),
        'frame': frame,
        'score': final_score,
        'phase': 'Complete',
        'done': True,
        'reason': reason,
        'elapsed': elapsed,
        'cycles': frame
    })


@app.route('/api/optimize-stream/<int:fid>')
def api_optimize_stream(fid):
    """SSE endpoint for streaming stroke optimization progress.

    Performs multi-phase optimization of strokes for a character,
    streaming intermediate results as Server-Sent Events for real-time
    visualization in the frontend.

    Args:
        fid: Font ID from URL path.

    Returns:
        Response: SSE stream with optimization progress events.

    Request:
        GET /api/optimize-stream/<fid>?c=A

    Query Parameters:
        c (str, required): Character to optimize strokes for.

    Response:
        Content-Type: text/event-stream

        The response is a stream of SSE events. Each event contains
        JSON data with the following possible structures:

        Phase announcement::

            data: {"phase": "Global affine (NM)", "frame": 5, "score": 0.72}

        Progress update (includes current strokes)::

            data: {
                "phase": "NM iter 100",
                "frame": 6,
                "score": 0.75,
                "strokes": [[[100, 50], [100, 150]], [[50, 100], [150, 100]]]
            }

        Final result::

            data: {
                "phase": "Complete",
                "frame": 15,
                "score": 0.89,
                "strokes": [...],
                "done": true,
                "reason": "converged",
                "elapsed": 3.45,
                "cycles": 15
            }

        Error (stream terminates after)::

            data: {"error": "Font not found"}
            data: {"error": "Missing ?c= parameter"}

    Response Headers:
        - Cache-Control: no-cache
        - X-Accel-Buffering: no (disables nginx buffering)
        - Connection: keep-alive

    Note:
        The client should use EventSource API to consume this endpoint::

            const es = new EventSource('/api/optimize-stream/1?c=A');
            es.onmessage = (e) => {
                const data = JSON.parse(e.data);
                if (data.done) es.close();
            };
    """
    c = request.args.get('c')
    if not c:
        return Response(_sse_event({'error': 'Missing ?c= parameter'}),
                        mimetype='text/event-stream')

    f = _font(fid)
    if not f:
        return Response(_sse_event({'error': 'Font not found'}),
                        mimetype='text/event-stream')

    def generate():
        try:
            yield from optimize_stream_generator(f['file_path'], c)
        except Exception as e:
            yield _sse_event({'error': str(e)})

    return Response(generate(), mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'X-Accel-Buffering': 'no',  # Disable nginx buffering
                        'Connection': 'keep-alive'
                    })


@app.route('/api/minimal-strokes-stream/<int:fid>')
def api_minimal_strokes_stream(fid):
    """SSE endpoint for streaming minimal strokes generation with debug info.

    Streams the step-by-step process of generating minimal strokes,
    allowing frame-by-frame visualization of the stroke fitting pipeline.

    Args:
        fid: Font ID from URL path.

    Returns:
        Response: SSE stream with stroke generation progress events.

    Request:
        GET /api/minimal-strokes-stream/<fid>?c=A

    Query Parameters:
        c (str, required): Character to generate strokes for.

    Response:
        Content-Type: text/event-stream

        Events from the underlying stream_minimal_strokes generator are
        augmented with frame numbers and elapsed time:

        Progress events::

            data: {
                "step": "skeleton",
                "frame": 1,
                "elapsed": 0.05,
                "skeleton_points": [[x, y], ...],
                ...
            }

            data: {
                "step": "template_match",
                "frame": 2,
                "elapsed": 0.12,
                "template": "default",
                "strokes": [...],
                ...
            }

        Final result::

            data: {
                "step": "complete",
                "frame": 5,
                "elapsed": 0.34,
                "strokes": [...],
                "variant": "default",
                "done": true
            }

        Error events::

            data: {"error": "Font not found"}
            data: {"error": "<exception message>", "done": true}

    Response Headers:
        - Cache-Control: no-cache
        - X-Accel-Buffering: no (disables nginx buffering)
        - Connection: keep-alive

    Note:
        This endpoint is primarily useful for debugging the stroke
        generation pipeline. For production use, prefer the non-streaming
        /api/minimal-strokes endpoint.

    See Also:
        stroke_pipeline_stream.stream_minimal_strokes: The underlying
        generator that produces the step events.
    """
    from stroke_pipeline_stream import stream_minimal_strokes

    c = request.args.get('c')
    if not c:
        return Response(_sse_event({'error': 'Missing ?c= parameter'}),
                        mimetype='text/event-stream')

    f = _font(fid)
    if not f:
        return Response(_sse_event({'error': 'Font not found'}),
                        mimetype='text/event-stream')

    def generate():
        frame = 0
        start_time = time.time()
        try:
            for step in stream_minimal_strokes(f['file_path'], c):
                frame += 1
                step['frame'] = frame
                step['elapsed'] = round(time.time() - start_time, 2)
                yield _sse_event(step)
        except Exception as e:
            yield _sse_event({'error': str(e), 'done': True})

    return Response(generate(), mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'X-Accel-Buffering': 'no',
                        'Connection': 'keep-alive'
                    })
