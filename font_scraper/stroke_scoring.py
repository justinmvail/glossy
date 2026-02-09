"""Stroke scoring functions for optimization.

This module contains functions for scoring strokes against point clouds
and glyph masks during shape fitting optimization.
"""


import numpy as np
from scipy.spatial import cKDTree
from stroke_shapes import (
    SHAPE_FNS,
)
from stroke_shapes import (
    param_vector_to_shapes as _param_vector_to_shapes,
)


def score_all_strokes(param_vector: np.ndarray, shape_types: list[str],
                      slices: list[tuple[int, int]], bbox: tuple,
                      cloud_tree: cKDTree, n_cloud: int, radius: float,
                      snap_yi: np.ndarray, snap_xi: np.ndarray,
                      w: int, h: int, dist_map: np.ndarray = None) -> float:
    """Objective for optimisation (minimisation -> returns -score).

    Snaps stroke points to nearest mask pixel before scoring so the
    optimiser sees the same benefit as the post-processing pipeline.

    Score = coverage - snap_penalty - edge_penalty
      coverage:       fraction of cloud points within radius of any stroke
      snap_penalty:   fraction of stroke points in white space (hard penalty)
      edge_penalty:   penalises strokes hugging the glyph edge instead of
                      running through the interior
    """
    all_shapes = _param_vector_to_shapes(param_vector, shape_types, slices, bbox)
    all_pts = np.concatenate(all_shapes, axis=0)
    if len(all_pts) == 0:
        return 0.0

    # Snap all stroke points to nearest mask pixel
    xi = np.clip(np.round(all_pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(all_pts[:, 1]).astype(int), 0, h - 1)
    snapped_x = snap_xi[yi, xi].astype(float)
    snapped_y = snap_yi[yi, xi].astype(float)
    snapped = np.column_stack([snapped_x, snapped_y])

    # White-space penalty: fraction of stroke points that lie outside the mask.
    snap_dist = np.sqrt((all_pts[:, 0] - snapped_x) ** 2 +
                        (all_pts[:, 1] - snapped_y) ** 2)
    off_mask = float(np.mean(snap_dist > 0.5))
    snap_penalty = 0.5 * off_mask

    # Edge penalty: penalise stroke points near the glyph boundary.
    edge_penalty = 0.0
    if dist_map is not None:
        sxi = np.clip(np.round(snapped_x).astype(int), 0, w - 1)
        syi = np.clip(np.round(snapped_y).astype(int), 0, h - 1)
        dt_vals = dist_map[syi, sxi]
        near_edge = float(np.mean(dt_vals < 1.5))
        edge_penalty = 0.1 * near_edge

    # Per-shape coverage sets
    per_shape = []
    offset = 0
    for i in range(len(shape_types)):
        n = len(all_shapes[i])
        shape_snapped = snapped[offset:offset + n]
        offset += n
        hits = cloud_tree.query_ball_point(shape_snapped, radius)
        sc = set()
        for lst in hits:
            sc.update(lst)
        per_shape.append(sc)

    covered_all = set().union(*per_shape) if per_shape else set()
    coverage = len(covered_all) / n_cloud

    # Overlap penalty
    FREE_OVERLAP = 0.25
    overlap_excess = 0.0
    n_shapes = len(per_shape)
    if n_shapes > 1:
        for i in range(n_shapes):
            if not per_shape[i]:
                continue
            others = set()
            for j in range(n_shapes):
                if j != i:
                    others |= per_shape[j]
            frac = len(per_shape[i] & others) / len(per_shape[i])
            if frac > FREE_OVERLAP:
                overlap_excess += (frac - FREE_OVERLAP)
        overlap_excess /= n_shapes

    overlap_penalty = 0.5 * overlap_excess

    return -(coverage - overlap_penalty - snap_penalty - edge_penalty)


def score_raw_strokes(stroke_arrays: list[np.ndarray], cloud_tree: cKDTree,
                      n_cloud: int, radius: float, snap_yi: np.ndarray,
                      snap_xi: np.ndarray, w: int, h: int,
                      dist_map: np.ndarray = None, mask: np.ndarray = None) -> float:
    """Score pre-built stroke point arrays against the target point cloud.

    Like score_all_strokes but accepts raw Nx2 arrays (one per stroke)
    instead of a shape-parameter vector.
    """
    if not stroke_arrays or all(len(s) == 0 for s in stroke_arrays):
        return 0.0

    processed = [s for s in stroke_arrays if len(s) >= 2]
    if not processed:
        return 0.0

    all_pts = np.concatenate(processed, axis=0)
    xi = np.clip(np.round(all_pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(all_pts[:, 1]).astype(int), 0, h - 1)
    snapped_x = snap_xi[yi, xi].astype(float)
    snapped_y = snap_yi[yi, xi].astype(float)
    snapped = np.column_stack([snapped_x, snapped_y])

    snap_dist = np.sqrt((all_pts[:, 0] - snapped_x) ** 2 +
                        (all_pts[:, 1] - snapped_y) ** 2)
    off_mask = float(np.mean(snap_dist > 0.5))
    snap_penalty = 0.5 * off_mask

    edge_penalty = 0.0
    if dist_map is not None:
        sxi = np.clip(np.round(snapped_x).astype(int), 0, w - 1)
        syi = np.clip(np.round(snapped_y).astype(int), 0, h - 1)
        near_edge = float(np.mean(dist_map[syi, sxi] < 1.5))
        edge_penalty = 0.1 * near_edge

    per_shape = []
    offset = 0
    for arr in processed:
        n = len(arr)
        shape_snapped = snapped[offset:offset + n]
        offset += n
        hits = cloud_tree.query_ball_point(shape_snapped, radius)
        sc = set()
        for lst in hits:
            sc.update(lst)
        per_shape.append(sc)

    covered_all = set().union(*per_shape) if per_shape else set()
    coverage = len(covered_all) / n_cloud

    FREE_OVERLAP = 0.25
    overlap_excess = 0.0
    n_shapes = len(per_shape)
    if n_shapes > 1:
        for i in range(n_shapes):
            if not per_shape[i]:
                continue
            others = set()
            for j in range(n_shapes):
                if j != i:
                    others |= per_shape[j]
            frac = len(per_shape[i] & others) / len(per_shape[i])
            if frac > FREE_OVERLAP:
                overlap_excess += (frac - FREE_OVERLAP)
        overlap_excess /= n_shapes

    overlap_penalty = 0.5 * overlap_excess
    return -(coverage - overlap_penalty - snap_penalty - edge_penalty)


def score_single_shape(params: np.ndarray, shape_type: str, bbox: tuple,
                       uncovered_pts: np.ndarray, uncovered_tree: cKDTree,
                       n_uncovered: int, radius: float,
                       snap_yi: np.ndarray, snap_xi: np.ndarray,
                       w: int, h: int, n_pts: int | None = None) -> float:
    """Score a single shape against uncovered points only (for greedy fitting).

    Returns negative coverage of uncovered points minus snap penalty.
    """
    if n_pts is None:
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        n_pts = max(60, int((bw * bw + bh * bh) ** 0.5 / 1.5))
    pts = SHAPE_FNS[shape_type](tuple(params), bbox, offset=(0, 0), n_pts=n_pts)
    if len(pts) == 0:
        return 0.0
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    snapped_x = snap_xi[yi, xi].astype(float)
    snapped_y = snap_yi[yi, xi].astype(float)
    snapped = np.column_stack([snapped_x, snapped_y])
    snap_dist = np.sqrt((pts[:, 0] - snapped_x) ** 2 +
                        (pts[:, 1] - snapped_y) ** 2)
    off_mask = float(np.mean(snap_dist > 0.5))
    snap_penalty = 0.5 * off_mask
    hits = uncovered_tree.query_ball_point(snapped, radius)
    covered = set()
    for lst in hits:
        covered.update(lst)
    coverage = len(covered) / max(n_uncovered, 1)
    return -(coverage - snap_penalty)


def quick_stroke_score(strokes: list[list[list[float]]], mask: np.ndarray) -> float:
    """Quick scoring for stroke quality - coverage of glyph mask.

    Returns fraction of mask pixels covered by strokes (0 to 1).
    """
    from scipy.ndimage import distance_transform_edt

    if not strokes or mask is None:
        return 0.0

    h, w = mask.shape

    # Get all stroke points
    all_pts = []
    for stroke in strokes:
        for pt in stroke:
            all_pts.append((int(round(pt[0])), int(round(pt[1]))))

    if not all_pts:
        return 0.0

    # Build a mask of stroke coverage using distance transform
    stroke_mask = np.zeros_like(mask, dtype=bool)
    for x, y in all_pts:
        if 0 <= x < w and 0 <= y < h:
            stroke_mask[y, x] = True

    # Dilate stroke mask using distance transform
    dist = distance_transform_edt(~stroke_mask)
    # Coverage radius based on average stroke width
    radius = 6.0  # Approximate stroke half-width

    covered = dist <= radius
    glyph_pixels = np.sum(mask)
    if glyph_pixels == 0:
        return 0.0

    covered_glyph = np.sum(mask & covered)
    return float(covered_glyph) / float(glyph_pixels)


def score_shape_coverage(shape_pts: np.ndarray, tree: cKDTree, radius: float,
                         claimed: set | None = None) -> float:
    """Count cloud points within radius of shape path.

    Gives a bonus weight for unclaimed points.
    """
    if len(shape_pts) == 0:
        return 0
    indices = tree.query_ball_point(shape_pts, radius)
    hit = set()
    for idx_list in indices:
        hit.update(idx_list)
    if claimed is None:
        return len(hit)
    unclaimed = hit - claimed
    return len(unclaimed) * 1.0 + len(hit & claimed) * 0.3


# Aliases for backwards compatibility
_score_all_strokes = score_all_strokes
_score_raw_strokes = score_raw_strokes
_score_single_shape = score_single_shape
_quick_stroke_score = quick_stroke_score
