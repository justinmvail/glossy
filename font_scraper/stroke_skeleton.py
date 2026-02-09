"""Skeleton analysis and stroke tracing functions.

This module contains functions for analyzing glyph skeletons and
tracing stroke paths through them.
"""

from collections import defaultdict, deque

import numpy as np
from skimage.morphology import skeletonize

# Constants
SKELETON_MERGE_DISTANCE = 12
MIN_STROKE_LENGTH = 5


def analyze_skeleton(mask: np.ndarray, merge_dist: int = SKELETON_MERGE_DISTANCE) -> dict | None:
    """Analyze skeleton to find endpoints and junction clusters.

    Args:
        mask: Binary glyph mask
        merge_dist: Distance threshold for merging junction clusters

    Returns:
        Dict with 'skel_set', 'adj', 'endpoints', 'junction_pixels', 'junction_clusters'
        or None if skeleton is empty.
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
    """Merge nearby junction pixels into clusters."""
    if not junction_pixels:
        return []

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

            if p in junction_pixels or p in cluster:
                cluster.add(p)
                # Check skeleton pixels within merge distance
                for sp in skel_set:
                    if sp in visited:
                        continue
                    d = ((sp[0] - p[0])**2 + (sp[1] - p[1])**2)**0.5
                    if d <= merge_dist and (sp in junction_pixels or len(adj[sp]) >= 3):
                        queue.append(sp)

        if cluster:
            clusters.append(cluster)

    return clusters


def find_skeleton_segments(info: dict) -> list[dict]:
    """Find and classify skeleton path segments between junctions.

    Returns list of segments, each with:
    - 'path': list of (x,y) points
    - 'start_junction': index or -1 for endpoint
    - 'end_junction': index or -1 for endpoint
    - 'angle': direction in degrees (0=right, 90=down)
    - 'length': number of pixels
    """
    adj = info['adj']
    junction_pixels = info['junction_pixels']
    junction_clusters = info['junction_clusters']
    endpoints = info['endpoints']

    segments = []

    # Map pixels to their junction cluster index
    pixel_to_junction = {}
    for i, cluster in enumerate(junction_clusters):
        for p in cluster:
            pixel_to_junction[p] = i

    # Find all segments between junctions/endpoints
    visited_pairs = set()

    # Start points: junction cluster border pixels and endpoints
    start_points = []
    for i, cluster in enumerate(junction_clusters):
        for p in cluster:
            for nb in adj[p]:
                if nb not in junction_pixels:
                    start_points.append((p, nb, i))  # (start, next, junction_idx)
    for ep in endpoints:
        if ep not in junction_pixels:
            start_points.append((ep, None, -1))

    for start, first_step, start_junc in start_points:
        if first_step is None:
            neighbors = adj[start]
            if not neighbors:
                continue
            first_step = neighbors[0]

        # Trace path until we hit another junction or endpoint
        path = [start, first_step]
        current = first_step
        prev = start

        while current not in junction_pixels and current not in endpoints:
            neighbors = [n for n in adj[current] if n != prev]
            if not neighbors:
                break
            if len(neighbors) > 1:
                break
            prev = current
            current = neighbors[0]
            path.append(current)

        # Determine end junction/endpoint
        if current in junction_pixels:
            end_junc = pixel_to_junction.get(current, -1)
        elif current in endpoints:
            end_junc = -1
        else:
            end_junc = -1

        # Avoid duplicate segments
        pair = (min(start_junc, end_junc), max(start_junc, end_junc),
                min(start, current), max(start, current))
        if pair in visited_pairs:
            continue
        visited_pairs.add(pair)

        if len(path) >= 2:
            dx = path[-1][0] - path[0][0]
            dy = path[-1][1] - path[0][1]
            angle = np.degrees(np.arctan2(dy, dx))

            segments.append({
                'path': path,
                'start': path[0],
                'end': path[-1],
                'start_junction': start_junc,
                'end_junction': end_junc,
                'angle': angle,
                'length': len(path),
            })

    return segments


def snap_to_skeleton(point: tuple, skel_set: set) -> tuple:
    """Find the nearest skeleton pixel to a point."""
    point = tuple(point) if not isinstance(point, tuple) else point
    if point in skel_set or not skel_set:
        return point

    min_dist = float('inf')
    nearest = point
    for p in skel_set:
        d = (p[0] - point[0])**2 + (p[1] - point[1])**2
        if d < min_dist:
            min_dist = d
            nearest = p
    return nearest


def trace_skeleton_path(start: tuple, end: tuple, adj: dict, skel_set: set,
                        max_steps: int = 500, avoid_pixels: set | None = None,
                        direction: str | None = None) -> list[tuple] | None:
    """Trace a path along skeleton pixels from start to end using BFS.

    Args:
        start: (x, y) starting point
        end: (x, y) ending point
        adj: adjacency dict
        skel_set: set of skeleton pixels
        max_steps: maximum path length
        avoid_pixels: set of pixels to avoid
        direction: 'down', 'up', 'left', 'right' - bias for initial direction

    Returns:
        List of (x, y) points along the path, or None if no path found.
    """
    if avoid_pixels is None:
        avoid_pixels = set()

    start = snap_to_skeleton(start, skel_set)
    end = snap_to_skeleton(end, skel_set)

    if start == end:
        return [start]

    # BFS with direction bias
    queue = deque([(start, [start], 0)])
    visited = {start}

    # Direction vectors for bias
    dir_vectors = {
        'down': (0, 1),
        'up': (0, -1),
        'left': (-1, 0),
        'right': (1, 0),
    }
    dir_vec = dir_vectors.get(direction)

    while queue:
        current, path, steps = queue.popleft()

        if steps >= max_steps:
            continue

        if current == end:
            return path

        neighbors = adj.get(current, [])

        # Sort neighbors by preference
        def neighbor_score(n):
            # Prefer end direction
            to_end = ((end[0] - n[0])**2 + (end[1] - n[1])**2)**0.5
            score = to_end

            # Direction bias (only for first few pixels)
            if dir_vec and steps < 15:
                dx, dy = n[0] - current[0], n[1] - current[1]
                dot = dx * dir_vec[0] + dy * dir_vec[1]
                score -= dot * 10  # Bonus for matching direction

            # Penalty for avoid_pixels
            if n in avoid_pixels:
                score += 1000

            return score

        sorted_neighbors = sorted(neighbors, key=neighbor_score)

        for nb in sorted_neighbors:
            if nb in visited:
                continue
            visited.add(nb)
            queue.append((nb, [*path, nb], steps + 1))

    # No path found - try without avoidance
    if avoid_pixels:
        return trace_skeleton_path(start, end, adj, skel_set, max_steps, None, direction)

    return None


def trace_segment(start: tuple, end: tuple, config, adj: dict, skel_set: set,
                  avoid_pixels: set | None = None, fallback_avoid: set | None = None) -> list[tuple] | None:
    """Trace a segment between two points along the skeleton.

    Args:
        start: Starting point
        end: Ending point
        config: SegmentConfig with direction hints
        adj: Adjacency dict
        skel_set: Set of skeleton pixels
        avoid_pixels: Pixels to avoid
        fallback_avoid: Fallback avoidance set if primary fails
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
                    point_in_region_fn=None) -> list[tuple] | None:
    """Trace a path from start until it reaches a target numpad region.

    Args:
        start: Starting point
        region: Target numpad region (1-9)
        bbox: Glyph bounding box
        adj: Adjacency dict
        skel_set: Set of skeleton pixels
        avoid_pixels: Pixels to avoid
        point_in_region_fn: Function to check if point is in region
    """
    if avoid_pixels is None:
        avoid_pixels = set()

    start = snap_to_skeleton(start, skel_set)

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


def generate_straight_line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Generate a straight line of pixels from start to end using Bresenham's algorithm."""
    x0, y0 = int(round(start[0])), int(round(start[1]))
    x1, y1 = int(round(end[0])), int(round(end[1]))

    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


def resample_path(path: list[tuple], num_points: int = 30) -> list[tuple]:
    """Resample a path to have a fixed number of evenly-spaced points."""
    if len(path) < 2:
        return list(path)

    if len(path) <= num_points:
        return list(path)

    # Calculate cumulative distances
    cumulative = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        cumulative.append(cumulative[-1] + (dx*dx + dy*dy)**0.5)

    total_length = cumulative[-1]
    if total_length < 1e-6:
        return [path[0], path[-1]]

    # Resample at even intervals
    result = [path[0]]
    current_idx = 1
    for i in range(1, num_points - 1):
        target_dist = total_length * i / (num_points - 1)

        while current_idx < len(cumulative) and cumulative[current_idx] < target_dist:
            current_idx += 1

        if current_idx >= len(path):
            result.append(path[-1])
            continue

        # Interpolate
        d_prev = cumulative[current_idx - 1]
        d_curr = cumulative[current_idx]
        if d_curr - d_prev < 1e-6:
            t = 0.0
        else:
            t = (target_dist - d_prev) / (d_curr - d_prev)

        x = path[current_idx-1][0] + t * (path[current_idx][0] - path[current_idx-1][0])
        y = path[current_idx-1][1] + t * (path[current_idx][1] - path[current_idx-1][1])
        result.append((x, y))

    result.append(path[-1])
    return result


# Aliases for backwards compatibility
_analyze_skeleton = analyze_skeleton
_find_skeleton_segments = find_skeleton_segments
_snap_to_skeleton = snap_to_skeleton
_trace_skeleton_path = trace_skeleton_path
_trace_to_region = trace_to_region
_generate_straight_line = generate_straight_line
_resample_path = resample_path
