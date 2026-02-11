"""Skeleton analysis and stroke tracing functions.

This module provides functions for analyzing glyph skeletons and tracing
stroke paths through them. It supports the stroke extraction pipeline by
identifying skeleton topology (endpoints, junctions, segments) and providing
path-finding algorithms for connecting waypoints along skeleton pixels.

Key functionality:
    - Skeleton analysis: Extract endpoints, junctions, and segments from
      morphological skeletons
    - Junction clustering: Merge nearby junction pixels into logical nodes
    - Path tracing: BFS-based path finding along skeleton pixels with
      directional bias and avoidance constraints
    - Path utilities: Straight line generation, path resampling

Typical usage:
    from stroke_skeleton import analyze_skeleton, trace_skeleton_path

    # Analyze a glyph skeleton
    info = analyze_skeleton(mask)
    segments = find_skeleton_segments(info)

    # Trace a path between two points
    path = trace_skeleton_path(start, end, info['adj'], info['skel_set'])
"""

from collections import defaultdict, deque

import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

# Import shared utilities from stroke_lib (canonical implementations)
from stroke_lib.utils.geometry import (
    generate_straight_line,
    resample_path,
)

# Constants
SKELETON_MERGE_DISTANCE = 12
MIN_STROKE_LENGTH = 5
DIRECTION_BIAS_STEPS = 15  # Apply direction bias only for first N steps
DIRECTION_BIAS_WEIGHT = 10  # Multiplier for direction matching bonus
AVOID_PENALTY = 1000  # Penalty score for stepping on avoided pixels

# Direction vectors for path tracing
DIRECTION_VECTORS = {
    'down': (0, 1),
    'up': (0, -1),
    'left': (-1, 0),
    'right': (1, 0),
}


def analyze_skeleton(mask: np.ndarray, merge_dist: int = SKELETON_MERGE_DISTANCE) -> dict | None:
    """Analyze a glyph skeleton to find topological features.

    Performs morphological skeletonization and classifies pixels as endpoints
    (degree 1), regular pixels (degree 2), or junction pixels (degree 3+).
    Nearby junctions are clustered together.

    Args:
        mask: Binary numpy array where True indicates glyph pixels.
            Shape should be (height, width).
        merge_dist: Maximum distance for merging junction pixels into
            clusters. Junctions within this distance are considered part
            of the same logical node. Defaults to SKELETON_MERGE_DISTANCE (12).

    Returns:
        A dictionary containing:
            - 'skel_set': Set of (x, y) tuples for all skeleton pixels
            - 'adj': Dict mapping each pixel to its 8-connected neighbors
            - 'endpoints': List of (x, y) pixels with exactly 1 neighbor
            - 'junction_pixels': List of (x, y) pixels with 3+ neighbors
            - 'junction_clusters': List of sets, each containing pixels
              belonging to the same junction cluster
        Returns None if the skeleton is empty.

    Notes:
        - Uses 8-connectivity for neighbor relationships.
        - Junction clustering helps handle complex junctions that span
          multiple pixels due to skeleton artifacts.
    """
    skel = skeletonize(mask)
    ys, xs = np.where(skel)
    if len(xs) == 0:
        return None

    skel_set = set(zip(xs.tolist(), ys.tolist()))

    # Build 8-connected adjacency
    adj = defaultdict(list)
    for (x, y) in skel_set:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                n = (x + dx, y + dy)
                if n in skel_set:
                    adj[(x, y)].append(n)

    # Classify pixels
    endpoints = []
    junction_pixels = []
    for p in skel_set:
        n_neighbors = len(adj[p])
        if n_neighbors == 1:
            endpoints.append(p)
        elif n_neighbors >= 3:
            junction_pixels.append(p)

    # Cluster junctions using BFS with merge distance
    junction_clusters = _merge_junction_clusters(junction_pixels, skel_set, adj, merge_dist)

    return {
        'skel_set': skel_set,
        'adj': adj,
        'endpoints': endpoints,
        'junction_pixels': junction_pixels,
        'junction_clusters': junction_clusters,
    }


def _merge_junction_clusters(junction_pixels: list[tuple], skel_set: set,
                              adj: dict, merge_dist: int) -> list[set]:
    """Merge nearby junction pixels into clusters.

    Uses BFS with KD-tree spatial indexing to group junction pixels that are
    within merge_dist of each other along the skeleton.

    Args:
        junction_pixels: List of (x, y) junction pixel positions.
        skel_set: Set of all skeleton pixel positions.
        adj: Adjacency dict mapping pixels to their neighbors.
        merge_dist: Maximum Euclidean distance for merging junctions.

    Returns:
        A list of sets, where each set contains (x, y) tuples of pixels
        belonging to the same junction cluster.

    Notes:
        - Empty list is returned if no junction pixels exist.
        - A pixel can only belong to one cluster.
        - Uses KD-tree for O(n log n) neighbor queries instead of O(n²).
    """
    if not junction_pixels:
        return []

    # Build KD-tree for efficient spatial queries
    skel_list = list(skel_set)
    skel_array = np.array(skel_list)
    skel_tree = cKDTree(skel_array)
    skel_index_to_point = {i: tuple(skel_list[i]) for i in range(len(skel_list))}

    junction_set = set(junction_pixels)
    visited = set()
    clusters = []

    for jp in junction_pixels:
        if jp in visited:
            continue

        cluster = set()
        queue = deque([jp])

        while queue:
            p = queue.popleft()
            if p in visited:
                continue
            visited.add(p)

            if p in junction_set or p in cluster:
                cluster.add(p)
                # Use KD-tree to find skeleton pixels within merge distance
                indices = skel_tree.query_ball_point(p, merge_dist)
                for idx in indices:
                    sp = skel_index_to_point[idx]
                    if sp in visited:
                        continue
                    if sp in junction_set or len(adj[sp]) >= 3:
                        queue.append(sp)

        if cluster:
            clusters.append(cluster)

    return clusters


def _build_pixel_to_junction_map(junction_clusters: list[set]) -> dict:
    """Map junction pixels to their cluster index.

    Args:
        junction_clusters: List of sets of (x, y) junction pixel positions.

    Returns:
        Dict mapping each junction pixel to its cluster index.
    """
    pixel_to_junction = {}
    for i, cluster in enumerate(junction_clusters):
        for p in cluster:
            pixel_to_junction[p] = i
    return pixel_to_junction


def _collect_segment_start_points(junction_clusters: list[set], endpoints: list,
                                   junction_pixels: list, adj: dict) -> list[tuple]:
    """Collect starting points for segment tracing.

    Args:
        junction_clusters: List of junction cluster sets.
        endpoints: List of endpoint pixels.
        junction_pixels: List of junction pixel positions.
        adj: Adjacency dict.

    Returns:
        List of (start, first_step, junction_idx) tuples for tracing.
    """
    start_points = []
    # Junction cluster border pixels
    for i, cluster in enumerate(junction_clusters):
        for p in cluster:
            for nb in adj[p]:
                if nb not in junction_pixels:
                    start_points.append((p, nb, i))
    # Endpoints
    for ep in endpoints:
        if ep not in junction_pixels:
            start_points.append((ep, None, -1))
    return start_points


def _trace_segment_from_start(start: tuple, first_step: tuple, adj: dict,
                               junction_pixels: list, endpoints: list) -> list[tuple]:
    """Trace a segment path from a starting point.

    Args:
        start: Starting pixel position.
        first_step: First step direction (or None to use first neighbor).
        adj: Adjacency dict.
        junction_pixels: List of junction pixel positions.
        endpoints: List of endpoint pixels.

    Returns:
        List of (x, y) tuples representing the traced path.
    """
    if first_step is None:
        neighbors = adj[start]
        if not neighbors:
            return []
        first_step = neighbors[0]

    path = [start, first_step]
    current = first_step
    prev = start

    while current not in junction_pixels and current not in endpoints:
        neighbors = [n for n in adj[current] if n != prev]
        if not neighbors or len(neighbors) > 1:
            break
        prev = current
        current = neighbors[0]
        path.append(current)

    return path


def _create_segment_dict(path: list[tuple], start_junc: int, end_junc: int) -> dict:
    """Create a segment dictionary from a traced path.

    Args:
        path: List of (x, y) points along the segment.
        start_junc: Junction cluster index at start (-1 for endpoint).
        end_junc: Junction cluster index at end (-1 for endpoint).

    Returns:
        Segment dict with path, endpoints, junctions, angle, and length.
    """
    dx = path[-1][0] - path[0][0]
    dy = path[-1][1] - path[0][1]
    angle = np.degrees(np.arctan2(dy, dx))

    return {
        'path': path,
        'start': path[0],
        'end': path[-1],
        'start_junction': start_junc,
        'end_junction': end_junc,
        'angle': angle,
        'length': len(path),
    }


def find_skeleton_segments(info: dict) -> list[dict]:
    """Find and classify skeleton path segments between junctions.

    Traces paths along the skeleton from junction borders and endpoints,
    recording the connectivity and geometric properties of each segment.

    Args:
        info: Skeleton analysis dict from analyze_skeleton(), containing
            'adj', 'junction_pixels', 'junction_clusters', and 'endpoints'.

    Returns:
        A list of segment dictionaries, each containing:
            - 'path': List of (x, y) points along the segment
            - 'start': First point of the segment
            - 'end': Last point of the segment
            - 'start_junction': Index into junction_clusters, or -1 for endpoint
            - 'end_junction': Index into junction_clusters, or -1 for endpoint
            - 'angle': Direction in degrees (0=right, 90=down)
            - 'length': Number of pixels in the path

    Notes:
        - Segments connect either two junctions, a junction and an endpoint,
          or two endpoints.
        - Duplicate segments (same start/end pair) are filtered out.
        - Segments shorter than 2 pixels are not included.
    """
    adj = info['adj']
    junction_pixels = info['junction_pixels']
    junction_clusters = info['junction_clusters']
    endpoints = info['endpoints']

    # Build lookup structures
    pixel_to_junction = _build_pixel_to_junction_map(junction_clusters)
    start_points = _collect_segment_start_points(
        junction_clusters, endpoints, junction_pixels, adj
    )

    segments = []
    visited_pairs = set()

    for start, first_step, start_junc in start_points:
        path = _trace_segment_from_start(start, first_step, adj,
                                          junction_pixels, endpoints)
        if len(path) < 2:
            continue

        current = path[-1]

        # Determine end junction
        if current in junction_pixels:
            end_junc = pixel_to_junction.get(current, -1)
        else:
            end_junc = -1

        # Avoid duplicate segments
        pair = (min(start_junc, end_junc), max(start_junc, end_junc),
                min(start, current), max(start, current))
        if pair in visited_pairs:
            continue
        visited_pairs.add(pair)

        segments.append(_create_segment_dict(path, start_junc, end_junc))

    return segments


def snap_to_skeleton(point: tuple, skel_set: set, skel_tree: cKDTree = None,
                     skel_list: list = None) -> tuple:
    """Find the nearest skeleton pixel to a given point.

    Args:
        point: The (x, y) coordinates to snap.
        skel_set: Set of (x, y) skeleton pixel positions.
        skel_tree: Optional pre-built KD-tree for O(log n) lookup.
            If provided, skel_list must also be provided.
        skel_list: Optional list of skeleton points corresponding to skel_tree.
            Required if skel_tree is provided.

    Returns:
        The (x, y) coordinates of the nearest skeleton pixel.
        Returns the original point if it is already on the skeleton
        or if the skeleton set is empty.

    Notes:
        - If skel_tree is provided, uses O(log n) KD-tree query.
        - Otherwise falls back to O(n) brute-force search.
    """
    point = tuple(point) if not isinstance(point, tuple) else point
    if point in skel_set or not skel_set:
        return point

    # Use KD-tree if available for O(log n) lookup
    if skel_tree is not None and skel_list is not None:
        _, idx = skel_tree.query(point)
        return tuple(skel_list[idx])

    # Fallback to brute-force O(n) search
    min_dist = float('inf')
    nearest = point
    for p in skel_set:
        d = (p[0] - point[0])**2 + (p[1] - point[1])**2
        if d < min_dist:
            min_dist = d
            nearest = p
    return nearest


def _compute_neighbor_score(neighbor: tuple, current: tuple, end: tuple,
                            dir_vec: tuple | None, steps: int,
                            avoid_pixels: set) -> float:
    """Compute priority score for a neighbor pixel in BFS path finding.

    Lower scores are preferred. The score combines distance to target,
    optional direction bias, and avoidance penalties.

    Args:
        neighbor: The neighbor pixel position (x, y).
        current: The current pixel position (x, y).
        end: The target position (x, y).
        dir_vec: Optional direction vector for bias, or None.
        steps: Current step count in the path.
        avoid_pixels: Set of pixels to penalize.

    Returns:
        Priority score (lower = better).
    """
    # Base score: distance to end
    to_end = ((end[0] - neighbor[0])**2 + (end[1] - neighbor[1])**2)**0.5
    score = to_end

    # Direction bias (only for first few pixels)
    if dir_vec and steps < DIRECTION_BIAS_STEPS:
        dx, dy = neighbor[0] - current[0], neighbor[1] - current[1]
        dot = dx * dir_vec[0] + dy * dir_vec[1]
        score -= dot * DIRECTION_BIAS_WEIGHT

    # Penalty for avoid_pixels
    if neighbor in avoid_pixels:
        score += AVOID_PENALTY

    return score


def _bfs_trace_path(start: tuple, end: tuple, adj: dict, max_steps: int,
                    avoid_pixels: set, dir_vec: tuple | None) -> list[tuple] | None:
    """BFS path finding with neighbor priority scoring.

    Args:
        start: Starting position (already snapped to skeleton).
        end: Target position (already snapped to skeleton).
        adj: Adjacency dict mapping pixels to neighbors.
        max_steps: Maximum path length.
        avoid_pixels: Set of pixels to penalize.
        dir_vec: Direction vector for bias, or None.

    Returns:
        List of (x, y) tuples forming the path, or None if not found.
    """
    if start == end:
        return [start]

    queue = deque([(start, [start], 0)])
    visited = {start}

    while queue:
        current, path, steps = queue.popleft()

        if steps >= max_steps:
            continue

        if current == end:
            return path

        neighbors = adj.get(current, [])
        sorted_neighbors = sorted(
            neighbors,
            key=lambda n: _compute_neighbor_score(n, current, end, dir_vec, steps, avoid_pixels)
        )

        for nb in sorted_neighbors:
            if nb in visited:
                continue
            visited.add(nb)
            queue.append((nb, [*path, nb], steps + 1))

    return None


def trace_skeleton_path(start: tuple, end: tuple, adj: dict, skel_set: set,
                        max_steps: int = 500, avoid_pixels: set | None = None,
                        direction: str | None = None, skel_tree: cKDTree = None,
                        skel_list: list = None) -> list[tuple] | None:
    """Trace a path along skeleton pixels from start to end using BFS.

    Finds the shortest path along the skeleton between two points, with
    optional directional bias and pixel avoidance.

    Args:
        start: Starting (x, y) position. Will be snapped to skeleton.
        end: Target (x, y) position. Will be snapped to skeleton.
        adj: Adjacency dict mapping skeleton pixels to their neighbors.
        skel_set: Set of all skeleton pixel positions.
        max_steps: Maximum path length allowed. Defaults to 500.
        avoid_pixels: Optional set of pixels to avoid during path finding.
            If no path can be found with avoidance, retries without it.
        direction: Optional directional bias for the first few steps.
            One of 'down', 'up', 'left', 'right'. Helps choose the correct
            branch at junctions near the start.
        skel_tree: Optional pre-built KD-tree for O(log n) snap lookups.
        skel_list: Optional list of skeleton points for KD-tree index mapping.

    Returns:
        A list of (x, y) tuples representing the path from start to end,
        inclusive of both endpoints. Returns None if no path exists.

    Notes:
        - Uses BFS with neighbor sorting for efficient path finding.
        - Direction bias only applies to the first DIRECTION_BIAS_STEPS pixels.
        - Avoided pixels incur a penalty but don't completely block paths.
    """
    if avoid_pixels is None:
        avoid_pixels = set()

    start = snap_to_skeleton(start, skel_set, skel_tree, skel_list)
    end = snap_to_skeleton(end, skel_set, skel_tree, skel_list)
    dir_vec = DIRECTION_VECTORS.get(direction)

    # Try with avoidance first
    path = _bfs_trace_path(start, end, adj, max_steps, avoid_pixels, dir_vec)

    # Retry without avoidance if needed
    if path is None and avoid_pixels:
        path = _bfs_trace_path(start, end, adj, max_steps, set(), dir_vec)

    return path


def trace_segment(start: tuple, end: tuple, config, adj: dict, skel_set: set,
                  avoid_pixels: set | None = None, fallback_avoid: set | None = None) -> list[tuple] | None:
    """Trace a segment between two points along the skeleton.

    Wrapper around trace_skeleton_path that extracts direction hints from
    a configuration object and supports fallback avoidance sets.

    Args:
        start: Starting (x, y) position.
        end: Target (x, y) position.
        config: Configuration object with optional 'direction' attribute
            specifying the preferred initial direction.
        adj: Adjacency dict mapping skeleton pixels to their neighbors.
        skel_set: Set of all skeleton pixel positions.
        avoid_pixels: Primary set of pixels to avoid.
        fallback_avoid: Secondary avoidance set used if primary tracing fails.

    Returns:
        A list of (x, y) tuples representing the path, or None if no path
        can be found even with the fallback.
    """
    direction = config.direction if hasattr(config, 'direction') else None

    result = trace_skeleton_path(start, end, adj, skel_set,
                                 avoid_pixels=avoid_pixels, direction=direction)

    if result is None and fallback_avoid:
        result = trace_skeleton_path(start, end, adj, skel_set,
                                     avoid_pixels=fallback_avoid, direction=direction)

    return result


def trace_to_region(start: tuple, region: int, bbox: tuple, adj: dict,
                    skel_set: set, avoid_pixels: set | None = None,
                    point_in_region_fn=None, skel_tree: cKDTree = None,
                    skel_list: list = None) -> list[tuple] | None:
    """Trace a path from start until it reaches a target numpad region.

    Uses BFS to explore the skeleton, stopping when a point falls within
    the specified region of the bounding box.

    Args:
        start: Starting (x, y) position. Will be snapped to skeleton.
        region: Target numpad region (1-9) where the path should end.
        bbox: Bounding box as (x0, y0, x1, y1) for region calculation.
        adj: Adjacency dict mapping skeleton pixels to their neighbors.
        skel_set: Set of all skeleton pixel positions.
        avoid_pixels: Optional set of pixels to avoid (after initial escape).
        point_in_region_fn: Function(point, region, bbox) -> bool to check
            if a point is in the target region.
        skel_tree: Optional pre-built KD-tree for O(log n) snap lookups.
        skel_list: Optional list of skeleton points for KD-tree index mapping.

    Returns:
        A list of (x, y) tuples from start to a point in the target region,
        or None if no such path exists within the step limit.

    Notes:
        - Avoidance is relaxed for the first 5 pixels to allow escaping
          from congested areas.
        - Maximum path length is 500 pixels.
    """
    if avoid_pixels is None:
        avoid_pixels = set()

    start = snap_to_skeleton(start, skel_set, skel_tree, skel_list)

    # BFS to find path to region
    queue = deque([(start, [start])])
    visited = {start}
    max_steps = 500

    while queue:
        current, path = queue.popleft()

        if len(path) >= max_steps:
            continue

        # Check if we've reached the target region
        if point_in_region_fn and point_in_region_fn(current, region, bbox):
            return path

        neighbors = adj.get(current, [])

        for nb in neighbors:
            if nb in visited:
                continue
            if nb in avoid_pixels and len(path) > 5:  # Allow initial escape
                continue
            visited.add(nb)
            queue.append((nb, [*path, nb]))

    return None


# Note: generate_straight_line and resample_path are imported from
# stroke_lib.utils.geometry (canonical implementations) at the top of this file.
