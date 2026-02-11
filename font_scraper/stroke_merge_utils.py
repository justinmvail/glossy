"""Utility functions for stroke merging.

This module provides low-level utility functions used by the stroke merging
algorithms. These include geometry calculations, cluster indexing, and
caching helpers.

Extracted from stroke_merge.py to improve code organization.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Stroke Tail Constants
# ---------------------------------------------------------------------------

# Maximum number of points to include in the tail segment when extracting
# the approach direction of a stroke toward a junction cluster.
TAIL_POINT_LIMIT = 8


def seg_dir(stroke: list[tuple], from_end: bool = False, n_samples: int = 5) -> tuple[float, float]:
    """Compute the direction vector of a stroke segment.

    Calculates a normalized direction vector based on the first or last few
    points of the stroke. This is used to determine stroke alignment for
    merging decisions.

    Args:
        stroke: List of (x, y) coordinate tuples representing the stroke path.
        from_end: If True, compute direction from the end of the stroke going
            backward. If False (default), compute from the start going forward.
        n_samples: Number of points to use for direction calculation. More
            samples provide a more stable direction estimate but may miss
            sharp turns near endpoints. Defaults to 5.

    Returns:
        A normalized (unit length) direction vector as (dx, dy). Returns (0, 0)
        if the stroke is too short to compute a direction.

    Note:
        The direction is computed by averaging the displacement between
        consecutive points in the sampled region, then normalizing. This
        smooths out noise from individual point jitter.
    """
    if len(stroke) < 2:
        return (0.0, 0.0)

    n = min(n_samples, len(stroke) - 1)

    if from_end:
        segment = stroke[-n-1:]
        dx = segment[-1][0] - segment[0][0]
        dy = segment[-1][1] - segment[0][1]
    else:
        segment = stroke[:n+1]
        dx = segment[-1][0] - segment[0][0]
        dy = segment[-1][1] - segment[0][1]

    length = (dx**2 + dy**2)**0.5
    if length < 1e-9:
        return (0.0, 0.0)
    return (dx / length, dy / length)


def angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    """Compute the angle between two direction vectors.

    Args:
        v1: First direction vector as (dx, dy). Should be normalized.
        v2: Second direction vector as (dx, dy). Should be normalized.

    Returns:
        Angle in radians, in the range [0, pi]. Returns 0 for parallel vectors,
        pi for antiparallel vectors.

    Note:
        Uses the dot product formula: cos(angle) = v1 . v2 / (|v1| |v2|).
        The dot product is clamped to [-1, 1] to handle floating-point errors
        that might produce values slightly outside this range.
    """
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    dot = max(-1.0, min(1.0, dot))
    return np.arccos(dot)


def endpoint_cluster(stroke: list[tuple], from_end: bool, assigned: list[set]) -> int:
    """Find which junction cluster contains a stroke endpoint.

    Checks if the specified endpoint of a stroke lies within any of the
    assigned junction clusters, with a small tolerance for nearby pixels.

    Args:
        stroke: List of (x, y) coordinate tuples.
        from_end: If True, check the last point; if False, check the first.
        assigned: List of sets, where each set contains (x, y) integer tuples
            representing pixels in a junction cluster.

    Returns:
        The index of the cluster containing the endpoint (0-indexed), or -1
        if the endpoint is not in any cluster.

    Note:
        The function checks the exact pixel position and all 8 neighboring
        pixels (3x3 neighborhood) to handle slight misalignments between
        stroke endpoints and cluster centers.
    """
    if not stroke:
        return -1

    pt = stroke[-1] if from_end else stroke[0]
    if isinstance(pt, (list, tuple)):
        pt_int = (int(round(pt[0])), int(round(pt[1])))
    else:
        pt_int = pt

    for i, cluster in enumerate(assigned):
        if pt_int in cluster:
            return i
        # Check nearby
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (pt_int[0] + dx, pt_int[1] + dy) in cluster:
                    return i
    return -1


def _build_cluster_endpoint_map(strokes: list[list[tuple]],
                                 assigned: list[set],
                                 cluster_cache: dict = None) -> dict[int, list[tuple]]:
    """Build a mapping from cluster indices to stroke endpoints.

    Args:
        strokes: List of stroke paths.
        assigned: List of junction cluster sets.
        cluster_cache: Optional dict to populate with (stroke_idx, from_end) -> cluster_id.
            If provided, will be filled during the map building to avoid redundant lookups.

    Returns:
        Dict mapping cluster index to list of (stroke_index, 'start'/'end') tuples.
    """
    cluster_map = defaultdict(list)
    for si, s in enumerate(strokes):
        sc = endpoint_cluster(s, False, assigned)
        ec = endpoint_cluster(s, True, assigned)
        if cluster_cache is not None:
            cluster_cache[(si, False)] = sc
            cluster_cache[(si, True)] = ec
        if sc >= 0:
            cluster_map[sc].append((si, 'start'))
        if ec >= 0:
            cluster_map[ec].append((si, 'end'))
    return dict(cluster_map)


def _build_cluster_index(strokes: list[list[tuple]], assigned: list[set]) -> dict[int, set[int]]:
    """Build an index mapping cluster IDs to stroke indices with endpoints there.

    Returns:
        Dict mapping cluster_id -> set of stroke indices.
    """
    index = {}
    for si, s in enumerate(strokes):
        for is_end in [False, True]:
            cid = endpoint_cluster(s, is_end, assigned)
            if cid >= 0:
                if cid not in index:
                    index[cid] = set()
                index[cid].add(si)
    return index


def _build_detailed_cluster_index(strokes: list[list[tuple]], assigned: list[set]) -> dict[int, list[tuple[int, bool]]]:
    """Build an index mapping cluster IDs to (stroke_index, is_end) pairs.

    This provides more detail than _build_cluster_index by tracking which
    endpoint (start or end) of each stroke is at each cluster.

    Args:
        strokes: List of stroke paths.
        assigned: List of assigned junction cluster sets.

    Returns:
        Dict mapping cluster_id -> list of (stroke_index, is_end) tuples.
    """
    index = {}
    for si, s in enumerate(strokes):
        for is_end in [False, True]:
            cid = endpoint_cluster(s, is_end, assigned)
            if cid >= 0:
                if cid not in index:
                    index[cid] = []
                index[cid].append((si, is_end))
    return index


def _build_endpoint_cache(strokes: list[list[tuple]], assigned: list[set]) -> dict[tuple[int, bool], int]:
    """Build a cache mapping (stroke_index, is_end) to cluster_id.

    This is the reverse of _build_detailed_cluster_index and provides O(1)
    lookup of which cluster a stroke endpoint belongs to, avoiding repeated
    calls to endpoint_cluster().

    Args:
        strokes: List of stroke paths.
        assigned: List of assigned junction cluster sets.

    Returns:
        Dict mapping (stroke_index, is_end) -> cluster_id.
        Missing keys indicate the endpoint is not at any cluster (cluster_id = -1).
    """
    cache = {}
    for si, s in enumerate(strokes):
        for is_end in [False, True]:
            cid = endpoint_cluster(s, is_end, assigned)
            cache[(si, is_end)] = cid
    return cache


def _get_cached_cluster(cache: dict[tuple[int, bool], int], stroke_idx: int, is_end: bool) -> int:
    """Get cluster ID from cache, returning -1 if not found."""
    return cache.get((stroke_idx, is_end), -1)


def _is_loop_stroke_cached(stroke_idx: int, cluster_cache: dict) -> bool:
    """Check if a stroke forms a loop (both ends at same cluster) using cache.

    Args:
        stroke_idx: Index of the stroke to check.
        cluster_cache: Dict mapping (stroke_idx, from_end) -> cluster_id.

    Returns:
        True if both endpoints are at the same valid cluster (>= 0).
    """
    sc = cluster_cache.get((stroke_idx, False), -1)
    ec = cluster_cache.get((stroke_idx, True), -1)
    return sc >= 0 and sc == ec


def _is_loop_stroke(stroke: list[tuple], assigned: list[set]) -> bool:
    """Check if a stroke forms a loop (both ends at same cluster).

    Args:
        stroke: List of (x, y) coordinate tuples.
        assigned: List of junction cluster sets.

    Returns:
        True if both endpoints are at the same valid cluster (>= 0).
    """
    sc = endpoint_cluster(stroke, False, assigned)
    ec = endpoint_cluster(stroke, True, assigned)
    return sc >= 0 and sc == ec


def get_stroke_tail(stroke: list[tuple], at_end: bool, cluster: set) -> tuple[list[tuple], tuple]:
    """Extract the tail portion of a stroke before entering a junction cluster.

    The "tail" is the segment of the stroke leading up to (but not inside) the
    junction cluster, used to determine the stroke's approach direction and
    for extending strokes toward stub tips.

    Args:
        stroke: List of (x, y) coordinate tuples.
        at_end: If True, get the tail at the end of the stroke; if False,
            at the start.
        cluster: Set of (x, y) integer tuples representing the junction cluster.

    Returns:
        A tuple of (tail_points, leg_end_point) where:
            - tail_points: List of up to 8 points from the stroke leading to
                the junction, ordered from interior toward the junction.
            - leg_end_point: The endpoint of the stroke (inside or at the
                cluster boundary).

    Note:
        Points inside the cluster are excluded from the tail (except the first
        point, which establishes the connection). This ensures the tail
        represents the stroke's approach direction without cluster noise.
    """
    if at_end:
        tail = []
        for k in range(len(stroke) - 1, -1, -1):
            pt = tuple(stroke[k]) if isinstance(stroke[k], (list, tuple)) else stroke[k]
            if len(tail) >= TAIL_POINT_LIMIT:
                break
            if (int(round(pt[0])), int(round(pt[1]))) not in cluster or not tail:
                tail.insert(0, pt)
        return tail, stroke[-1]
    else:
        tail = []
        for k in range(len(stroke)):
            pt = tuple(stroke[k]) if isinstance(stroke[k], (list, tuple)) else stroke[k]
            if len(tail) >= TAIL_POINT_LIMIT:
                break
            if (int(round(pt[0])), int(round(pt[1]))) not in cluster or not tail:
                tail.append(pt)
        return tail, stroke[0]


def extend_stroke_to_tip(stroke: list[tuple], at_end: bool, tail: list[tuple],
                          leg_end: tuple, stub_tip: tuple) -> None:
    """Extend a stroke toward a stub tip point.

    Modifies the stroke in place by adding extension points that continue
    the stroke's direction toward the stub tip. This is used during stub
    absorption to extend neighboring strokes.

    Args:
        stroke: List of (x, y) coordinate tuples. Modified in place.
        at_end: If True, extend from the end of the stroke; if False, from start.
        tail: The tail segment of the stroke (used for direction calculation).
        leg_end: The current endpoint of the stroke leg being extended.
        stub_tip: The target point to extend toward (the stub's free tip).

    Note:
        Extension points are interpolated between the leg end and stub tip,
        maintaining the stroke's approach direction. The number of extension
        points is proportional to the distance to ensure smooth continuation.
    """
    if len(tail) < 2:
        return

    # Calculate direction from tail
    dx = tail[-1][0] - tail[0][0]
    dy = tail[-1][1] - tail[0][1]
    length = (dx**2 + dy**2)**0.5
    if length < 1e-9:
        return

    # Calculate distance to stub tip
    tip_dx = stub_tip[0] - leg_end[0]
    tip_dy = stub_tip[1] - leg_end[1]
    tip_dist = (tip_dx**2 + tip_dy**2)**0.5

    if tip_dist < 1:
        return

    # Generate extension points
    n_pts = max(2, int(tip_dist / 2))
    ext_pts = []
    for i in range(1, n_pts + 1):
        t = i / n_pts
        x = leg_end[0] + t * tip_dx
        y = leg_end[1] + t * tip_dy
        ext_pts.append((x, y))

    if at_end:
        stroke.extend(ext_pts)
    else:
        for p in reversed(ext_pts):
            stroke.insert(0, p)
