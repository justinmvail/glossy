"""Affine transformation and optimization for strokes.

This module contains functions for optimizing stroke positions using
affine transformations (translation, rotation, scaling, shear).
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

from stroke_shapes import make_point_cloud, adaptive_radius
from stroke_scoring import score_raw_strokes

logger = logging.getLogger(__name__)


def affine_transform_strokes(strokes: List[np.ndarray], params: Tuple,
                             centroid: Tuple[float, float]) -> List[np.ndarray]:
    """Apply affine transform to strokes around a centroid.

    Args:
        strokes: List of Nx2 numpy arrays
        params: (tx, ty, sx, sy, theta_deg, shear)
        centroid: (cx, cy) center point for transform

    Returns:
        List of transformed Nx2 numpy arrays.
    """
    tx, ty, sx, sy, theta_deg, shear = params
    theta = np.radians(theta_deg)
    cx, cy = centroid
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    result = []
    for stroke in strokes:
        pts = np.array(stroke, dtype=float)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        rx = dx * cos_t - dy * sin_t + shear * dy
        ry = dx * sin_t + dy * cos_t
        pts[:, 0] = cx + sx * rx + tx
        pts[:, 1] = cy + sy * ry + ty
        result.append(pts)
    return result


def prepare_affine_optimization(font_path: str, char: str, canvas_size: int,
                                strokes_raw: List, mask: np.ndarray,
                                smooth_stroke_fn, constrain_to_mask_fn) -> Optional[Tuple]:
    """Setup optimization data structures for affine stroke optimization.

    Args:
        font_path: Path to font file
        char: Character being optimized
        canvas_size: Canvas size for rendering
        strokes_raw: Raw strokes from template
        mask: Glyph mask array
        smooth_stroke_fn: Function to smooth strokes
        constrain_to_mask_fn: Function to constrain to mask

    Returns:
        Tuple of (stroke_arrays, centroid, glyph_bbox, score_args) or None if setup fails
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None

    glyph_bbox = (float(cols.min()), float(rows.min()),
                  float(cols.max()), float(rows.max()))
    centroid = (float(cols.mean()), float(rows.mean()))
    cloud = make_point_cloud(mask, spacing=3)
    if len(cloud) < 10:
        return None
    cloud_tree = cKDTree(cloud)
    n_cloud = len(cloud)
    radius = adaptive_radius(mask, spacing=3)
    h, w = mask.shape
    dist_map = distance_transform_edt(mask)
    _, snap_indices = distance_transform_edt(~mask, return_indices=True)
    snap_yi, snap_xi = snap_indices[0], snap_indices[1]

    score_args = (cloud_tree, n_cloud, radius, snap_yi, snap_xi, w, h, dist_map)

    # Convert raw strokes to numpy arrays (pre-smooth once)
    stroke_arrays = []
    for s in strokes_raw:
        pl = [(float(p[0]), float(p[1])) for p in s]
        pl = smooth_stroke_fn(pl, sigma=2.0)
        pl = constrain_to_mask_fn(pl, mask)
        if len(pl) >= 2:
            stroke_arrays.append(np.array(pl))
    if not stroke_arrays:
        return None

    return stroke_arrays, centroid, glyph_bbox, score_args


def run_global_affine(stroke_arrays: List[np.ndarray], centroid: Tuple[float, float],
                      score_args: Tuple) -> Tuple[List[np.ndarray], np.ndarray, float]:
    """Run Stage 1 global affine optimization on all strokes together.

    Args:
        stroke_arrays: List of Nx2 numpy arrays, one per stroke
        centroid: (x, y) center point for affine transform
        score_args: Tuple of scoring function arguments

    Returns:
        Tuple of (best_strokes, best_params, best_score)
    """
    from scipy.optimize import minimize, differential_evolution

    affine_bounds = [(-20, 20), (-20, 20),  # translate
                     (0.7, 1.3), (0.7, 1.3),  # scale
                     (-15, 15),  # rotation degrees
                     (-0.3, 0.3)]  # shear

    def _affine_obj(params):
        transformed = affine_transform_strokes(stroke_arrays, params, centroid)
        return score_raw_strokes(transformed, *score_args)

    # Quick NM from identity
    x0 = np.array([0, 0, 1, 1, 0, 0], dtype=float)
    nm = minimize(_affine_obj, x0, method='Nelder-Mead',
                  options={'maxfev': 800, 'xatol': 0.1, 'fatol': 0.002,
                           'adaptive': True})
    best_params = nm.x.copy()
    best_score = nm.fun

    # DE refinement (quick)
    try:
        de = differential_evolution(_affine_obj, bounds=affine_bounds,
                                    x0=best_params, maxiter=20, popsize=10,
                                    tol=0.005, polish=False)
        if de.fun < best_score:
            best_params = de.x.copy()
            best_score = de.fun
    except (ValueError, RuntimeError) as e:
        logger.debug("DE optimization failed in global affine: %s", e)

    # Apply best global affine
    best_strokes = affine_transform_strokes(stroke_arrays, best_params, centroid)
    return best_strokes, best_params, best_score


def run_per_stroke_refinement(best_strokes: List[np.ndarray], best_score: float,
                              score_args: Tuple) -> Tuple[List[np.ndarray], float]:
    """Run Stage 2 per-stroke translate+scale refinement.

    Args:
        best_strokes: List of Nx2 numpy arrays after global affine
        best_score: Score from global affine stage
        score_args: Tuple of scoring function arguments

    Returns:
        Tuple of (final_strokes, final_score)
    """
    from scipy.optimize import minimize

    n_strokes = len(best_strokes)

    def _per_stroke_obj(params):
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
    nm2 = minimize(_per_stroke_obj, x0_per, method='Nelder-Mead',
                   options={'maxfev': 1500, 'xatol': 0.1, 'fatol': 0.002,
                            'adaptive': True})

    if nm2.fun < best_score:
        final_strokes = []
        for si, base in enumerate(best_strokes):
            dx, dy, sx, sy = nm2.x[si * 4:(si + 1) * 4]
            pts = base.copy()
            c = pts.mean(axis=0)
            pts[:, 0] = c[0] + sx * (pts[:, 0] - c[0]) + dx
            pts[:, 1] = c[1] + sy * (pts[:, 1] - c[1]) + dy
            final_strokes.append(pts)
        return final_strokes, nm2.fun
    else:
        return best_strokes, best_score


def optimize_affine(font_path: str, char: str, canvas_size: int,
                    template_to_strokes_fn, render_glyph_mask_fn,
                    resolve_font_path_fn, smooth_stroke_fn,
                    constrain_to_mask_fn) -> Optional[Tuple]:
    """Optimise template strokes via affine transforms.

    Stage 1: Global affine (6 params) on all strokes together.
    Stage 2: Per-stroke translate+scale refinement.

    Returns (strokes, score, mask, glyph_bbox) or None if no template.
    """
    strokes_raw = template_to_strokes_fn(font_path, char, canvas_size)
    if not strokes_raw or len(strokes_raw) == 0:
        return None

    font_path = resolve_font_path_fn(font_path)
    mask = render_glyph_mask_fn(font_path, char, canvas_size)
    if mask is None:
        return None

    # Setup optimization data
    setup = prepare_affine_optimization(font_path, char, canvas_size, strokes_raw, mask,
                                        smooth_stroke_fn, constrain_to_mask_fn)
    if setup is None:
        return None
    stroke_arrays, centroid, glyph_bbox, score_args = setup

    # Stage 1: Global affine
    best_strokes, best_params, best_score = run_global_affine(stroke_arrays, centroid, score_args)

    # Stage 2: Per-stroke refinement
    final_strokes, final_score = run_per_stroke_refinement(best_strokes, best_score, score_args)

    # Convert back to list format
    result_strokes = [[[round(float(p[0]), 1), round(float(p[1]), 1)]
                       for p in s] for s in final_strokes]

    return result_strokes, float(-final_score), mask, glyph_bbox


def optimize_diffvg(font_path: str, char: str, canvas_size: int,
                    diffvg_docker, template_to_strokes_fn,
                    render_glyph_mask_fn, logger_ref,
                    num_iterations: int = 500, stroke_width: float = 8.0,
                    timeout: int = 300) -> Optional[Tuple]:
    """Optimise strokes using DiffVG differentiable rendering in Docker.

    Returns same format as optimize_affine: (strokes, score, mask, bbox)
    or None on failure.
    """
    if diffvg_docker is None:
        return None

    # Get initial strokes from template
    tpl = template_to_strokes_fn(font_path, char, canvas_size)
    if tpl is None:
        return None

    initial_strokes = tpl if not isinstance(tpl, tuple) else tpl[0]
    if not initial_strokes:
        return None

    result = diffvg_docker.optimize(
        font_path=font_path,
        char=char,
        initial_strokes=initial_strokes,
        canvas_size=canvas_size,
        num_iterations=num_iterations,
        stroke_width=stroke_width,
        timeout=timeout,
    )

    if 'error' in result:
        logger_ref.warning('DiffVG error for %s: %s', char, result['error'])
        return None

    diffvg_strokes = result.get('strokes', [])
    diffvg_score = result.get('score', 0.0)

    if not diffvg_strokes or diffvg_score <= 0:
        return None

    # Render glyph mask and bbox
    mask = render_glyph_mask_fn(font_path, char, canvas_size)
    if mask is None:
        return None
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    glyph_bbox = (float(xs.min()), float(ys.min()),
                  float(xs.max()), float(ys.max()))

    return diffvg_strokes, diffvg_score, mask, glyph_bbox


# Aliases for backwards compatibility
_affine_transform_strokes = affine_transform_strokes
_prepare_affine_optimization = prepare_affine_optimization
_run_global_affine = run_global_affine
_run_per_stroke_refinement = run_per_stroke_refinement
_optimize_affine = optimize_affine
_optimize_diffvg = optimize_diffvg
