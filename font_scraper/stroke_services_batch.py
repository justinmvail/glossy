"""Service functions for batch stroke route handlers.

This module provides the business logic for batch stroke operations,
extracted from stroke_routes_batch.py to improve code organization.

Services:
    - DiffVG integration (lazy loading)
    - Ray casting for point centering
    - Skeleton and stroke generation
    - Test run management
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from stroke_flask import (
    CHARS,
    DEFAULT_CANVAS_SIZE,
    DEFAULT_STROKE_WIDTH,
    STROKE_COLORS,
    test_run_repository,
)
from stroke_rendering import render_glyph_mask
from stroke_templates import NUMPAD_TEMPLATE_VARIANTS

# Lazy-loaded DiffVG instance
_diffvg = None
_diffvg_initialized = False
_diffvg_lock = threading.Lock()


def get_diffvg() -> Any:
    """Get the DiffVG Docker instance, lazily initializing on first use.

    Thread-safe initialization using double-checked locking pattern.

    Returns:
        DiffVGDocker instance, or None if unavailable.
    """
    global _diffvg, _diffvg_initialized
    if not _diffvg_initialized:
        with _diffvg_lock:
            if not _diffvg_initialized:
                try:
                    from docker.diffvg_docker import DiffVGDocker
                    _diffvg = DiffVGDocker()
                except ImportError:
                    _diffvg = None
                _diffvg_initialized = True
    return _diffvg


def get_stroke_funcs() -> tuple:
    """Import stroke processing functions lazily to avoid circular imports.

    Returns:
        Tuple of (min_strokes, skel_strokes, auto_fit) functions.
    """
    from stroke_core import auto_fit, min_strokes, skel_strokes
    return min_strokes, skel_strokes, auto_fit


# Pre-compute 36 ray directions (every 10 degrees)
RAY_DIRECTIONS = [(np.cos(i * np.pi / 18), np.sin(i * np.pi / 18)) for i in range(36)]


def cast_ray(mask: np.ndarray, x: float, y: float, dx: float, dy: float,
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


def center_point_in_glyph(x: float, y: float, mask: np.ndarray,
                          directions: list = None) -> tuple[float, float]:
    """Find the centered position for a point within the glyph.

    Casts rays in all directions to find the narrowest width,
    then returns the center of that width.

    Args:
        x: Point x coordinate.
        y: Point y coordinate.
        mask: Binary mask where True = inside glyph.
        directions: List of (dx, dy) direction tuples. Defaults to RAY_DIRECTIONS.

    Returns:
        Tuple (new_x, new_y) of centered position.
    """
    if directions is None:
        directions = RAY_DIRECTIONS

    best_width = float('inf')
    best_pos = (x, y)

    for dx, dy in directions:
        dist_pos = cast_ray(mask, x, y, dx, dy)
        dist_neg = cast_ray(mask, x, y, -dx, -dy)
        if dist_pos and dist_neg:
            total_width = dist_pos + dist_neg
            if total_width < best_width:
                best_width = total_width
                offset = (dist_pos - dist_neg) / 2.0
                best_pos = (x + dx * offset, y + dy * offset)

    return best_pos


def center_strokes_in_borders(strokes: list, mask: np.ndarray) -> list:
    """Center all stroke points within glyph borders.

    Args:
        strokes: List of stroke paths.
        mask: Binary glyph mask.

    Returns:
        Strokes with points centered within glyph.
    """
    result = []
    for stroke in strokes:
        centered_stroke = []
        for point in stroke:
            x, y = point[0], point[1]
            new_x, new_y = center_point_in_glyph(x, y, mask)
            if len(point) > 2:
                centered_stroke.append([new_x, new_y] + list(point[2:]))
            else:
                centered_stroke.append([new_x, new_y])
        result.append(centered_stroke)
    return result


@dataclass
class TestRunResult:
    """Result of a test run."""
    run_id: int
    font_id: int
    timestamp: str
    results: dict
    summary: dict


def create_test_run(font_id: int, font_path: str, chars: str = None,
                    canvas_size: int = DEFAULT_CANVAS_SIZE) -> TestRunResult:
    """Run stroke generation tests on a font.

    Args:
        font_id: Database font ID.
        font_path: Path to font file.
        chars: Characters to test. Defaults to CHARS.
        canvas_size: Canvas size for rendering.

    Returns:
        TestRunResult with test outcomes.
    """
    min_strokes, skel_strokes, auto_fit = get_stroke_funcs()

    if chars is None:
        chars = CHARS

    results = {}
    for c in chars:
        try:
            strokes = min_strokes(font_path, c, cs=canvas_size)
            if strokes:
                results[c] = {
                    'status': 'ok',
                    'stroke_count': len(strokes),
                    'point_count': sum(len(s) for s in strokes)
                }
            else:
                results[c] = {'status': 'no_strokes'}
        except Exception as e:
            results[c] = {'status': 'error', 'message': str(e)}

    # Calculate summary
    ok_count = sum(1 for r in results.values() if r.get('status') == 'ok')
    summary = {
        'total': len(results),
        'ok': ok_count,
        'failed': len(results) - ok_count
    }

    # Save to database
    run_id = test_run_repository.create_run(
        font_id=font_id,
        results=json.dumps(results),
        summary=json.dumps(summary)
    )

    return TestRunResult(
        run_id=run_id,
        font_id=font_id,
        timestamp=datetime.now().isoformat(),
        results=results,
        summary=summary
    )


def get_test_history(font_id: int, limit: int = 10) -> list[dict]:
    """Get test run history for a font.

    Args:
        font_id: Database font ID.
        limit: Maximum number of runs to return.

    Returns:
        List of test run summaries.
    """
    runs = test_run_repository.get_runs_for_font(font_id, limit=limit)
    return [
        {
            'id': run['id'],
            'timestamp': run['created_at'],
            'summary': json.loads(run['summary']) if run['summary'] else {}
        }
        for run in runs
    ]


def get_test_run_detail(run_id: int) -> dict | None:
    """Get detailed results for a test run.

    Args:
        run_id: Test run database ID.

    Returns:
        Dict with run details or None if not found.
    """
    run = test_run_repository.get_run(run_id)
    if not run:
        return None

    return {
        'id': run['id'],
        'font_id': run['font_id'],
        'timestamp': run['created_at'],
        'results': json.loads(run['results']) if run['results'] else {},
        'summary': json.loads(run['summary']) if run['summary'] else {}
    }


def generate_skeleton_strokes(font_path: str, char: str,
                              canvas_size: int = DEFAULT_CANVAS_SIZE) -> list | None:
    """Generate strokes using skeleton analysis.

    Args:
        font_path: Path to font file.
        char: Character to process.
        canvas_size: Canvas size for rendering.

    Returns:
        List of stroke paths or None on failure.
    """
    _, skel_strokes, _ = get_stroke_funcs()
    try:
        mask = render_glyph_mask(font_path, char, canvas_size)
        if mask is None:
            return None
        return skel_strokes(mask)
    except Exception:
        return None


def generate_minimal_strokes(font_path: str, char: str,
                             canvas_size: int = DEFAULT_CANVAS_SIZE,
                             return_variant: bool = False) -> list | tuple | None:
    """Generate strokes using minimal stroke pipeline.

    Args:
        font_path: Path to font file.
        char: Character to process.
        canvas_size: Canvas size for rendering.
        return_variant: If True, return (strokes, variant_name).

    Returns:
        List of stroke paths, or (strokes, variant_name) if return_variant=True.
        None on failure.
    """
    min_strokes, _, _ = get_stroke_funcs()
    try:
        return min_strokes(font_path, char, cs=canvas_size, ret_var=return_variant)
    except Exception:
        return None


def generate_batch_skeletons(font_path: str, chars: str = None,
                             canvas_size: int = DEFAULT_CANVAS_SIZE) -> dict:
    """Generate skeleton strokes for multiple characters.

    Args:
        font_path: Path to font file.
        chars: Characters to process. Defaults to CHARS.
        canvas_size: Canvas size for rendering.

    Returns:
        Dict mapping character to strokes list.
    """
    if chars is None:
        chars = CHARS

    results = {}
    for c in chars:
        strokes = generate_skeleton_strokes(font_path, c, canvas_size)
        if strokes:
            results[c] = strokes

    return results


def generate_batch_minimal_strokes(font_path: str, chars: str = None,
                                   canvas_size: int = DEFAULT_CANVAS_SIZE) -> dict:
    """Generate minimal strokes for multiple characters.

    Args:
        font_path: Path to font file.
        chars: Characters to process. Defaults to CHARS.
        canvas_size: Canvas size for rendering.

    Returns:
        Dict mapping character to (strokes, variant_name) tuples.
    """
    if chars is None:
        chars = CHARS

    results = {}
    for c in chars:
        result = generate_minimal_strokes(font_path, c, canvas_size, return_variant=True)
        if result:
            strokes, variant = result
            if strokes:
                results[c] = {'strokes': strokes, 'variant': variant}

    return results


def get_template_variants() -> dict:
    """Get all available template variants.

    Returns:
        Dict mapping character to dict of variant names to templates.
    """
    return {
        char: {name: template for name, template in variants.items()}
        for char, variants in NUMPAD_TEMPLATE_VARIANTS.items()
    }


def render_strokes_preview(strokes: list, size: int = DEFAULT_CANVAS_SIZE,
                           stroke_width: int = DEFAULT_STROKE_WIDTH,
                           background: str = 'white') -> Image.Image:
    """Render strokes to an image.

    Args:
        strokes: List of stroke paths.
        size: Canvas size in pixels.
        stroke_width: Width of stroke lines.
        background: Background color.

    Returns:
        PIL Image with rendered strokes.
    """
    img = Image.new('RGB', (size, size), background)
    draw = ImageDraw.Draw(img)

    for i, stroke in enumerate(strokes):
        if len(stroke) < 2:
            continue
        color = STROKE_COLORS[i % len(STROKE_COLORS)]
        points = [(p[0], p[1]) for p in stroke]
        draw.line(points, fill=color, width=stroke_width)

    return img
