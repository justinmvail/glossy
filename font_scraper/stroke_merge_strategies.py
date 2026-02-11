"""Strategy implementations for stroke merging.

This module contains the implementation functions for each merge strategy.
These are called by the MergeStrategy subclasses in stroke_merge.py.

Extracted from stroke_merge.py to improve code organization.

Default Thresholds:
    The magic number defaults in this module match the named constants in
    stroke_merge.py. When calling these functions via MergeStrategy classes,
    the constants are passed explicitly. For direct calls, these defaults
    provide sensible behavior.

    - conv_threshold=18: DEFAULT_CONV_THRESHOLD (convergence stub max length)
    - stub_threshold=20: DEFAULT_STUB_THRESHOLD (junction stub max length)
    - prox_threshold=20: DEFAULT_PROX_THRESHOLD (proximity merge distance)
"""

from __future__ import annotations

import numpy as np

from stroke_merge_utils import (
    seg_dir,
    angle_between,
    endpoint_cluster,
    _build_cluster_endpoint_map,
    _build_cluster_index,
    _build_detailed_cluster_index,
    _build_endpoint_cache,
    _get_cached_cluster,
    _is_loop_stroke_cached,
    get_stroke_tail,
    extend_stroke_to_tip,
)


def _is_valid_merge_candidate(strokes: list, stroke_idx: int, min_len: int,
                               cluster_cache: dict) -> bool:
    """Check if a stroke is a valid candidate for merging.

    Args:
        strokes: List of stroke paths.
        stroke_idx: Index of the stroke to check.
        min_len: Minimum stroke length requirement.
        cluster_cache: Cache for loop stroke detection.

    Returns:
        True if the stroke can be considered for merging.
    """
    if len(strokes[stroke_idx]) < min_len:
        return False
    if _is_loop_stroke_cached(stroke_idx, cluster_cache):
        return False
    return True


def _compute_merge_angle(strokes: list, si: int, side_i: str,
                         sj: int, side_j: str) -> float:
    """Compute the merge angle between two strokes.

    Args:
        strokes: List of stroke paths.
        si, sj: Indices of the two strokes.
        side_i, side_j: Which side ('start' or 'end') of each stroke.

    Returns:
        Angle in radians (0 = perfectly aligned, pi = opposite directions).
    """
    dir_i = seg_dir(strokes[si], from_end=(side_i == 'end'))
    dir_j = seg_dir(strokes[sj], from_end=(side_j == 'end'))
    return np.pi - angle_between(dir_i, dir_j)


def _find_best_merge_pair(strokes: list[list[tuple]], cluster_map: dict,
                           assigned: list[set], min_len: int, max_angle: float,
                           max_ratio: float, cluster_cache: dict) -> tuple | None:
    """Find the best pair of strokes to merge at a junction.

    Searches all junction clusters for pairs of strokes that can be merged
    based on direction alignment and length ratio constraints.

    Args:
        strokes: List of stroke paths.
        cluster_map: Mapping from cluster ID to list of (stroke_index, side) tuples.
        assigned: List of junction cluster sets.
        min_len: Minimum stroke length for merge consideration.
        max_angle: Maximum angle (radians) between stroke directions for merge.
        max_ratio: Maximum length ratio between strokes (shorter/longer).
        cluster_cache: Dict mapping (stroke_idx, from_end) -> cluster_id.

    Returns:
        Tuple (si, side_i, sj, side_j) identifying the best merge pair, or None
        if no valid pair was found.
    """
    best_angle = max_angle
    best_merge = None

    for cluster_id, entries in cluster_map.items():
        if len(entries) < 2:
            continue

        for i in range(len(entries)):
            si, side_i = entries[i]
            if not _is_valid_merge_candidate(strokes, si, min_len, cluster_cache):
                continue

            for j in range(i + 1, len(entries)):
                sj, side_j = entries[j]
                if not _is_valid_merge_candidate(strokes, sj, min_len, cluster_cache):
                    continue

                # Check length ratio constraint
                len_i, len_j = len(strokes[si]), len(strokes[sj])
                if min(len_i, len_j) / max(len_i, len_j) < max_ratio:
                    continue

                angle = _compute_merge_angle(strokes, si, side_i, sj, side_j)
                if angle < best_angle:
                    best_angle = angle
                    best_merge = (si, side_i, sj, side_j)

    return best_merge


def _execute_merge(strokes: list[list[tuple]], si: int, side_i: str,
                    sj: int, side_j: str) -> None:
    """Execute a merge between two strokes.

    Combines stroke sj into stroke si, then removes sj from the list.
    The merge respects the specified sides (start/end) for proper ordering.

    Args:
        strokes: List of stroke paths. Modified in place.
        si: Index of first stroke (will receive merged result).
        side_i: Which side of stroke si to connect ('start' or 'end').
        sj: Index of second stroke (will be removed).
        side_j: Which side of stroke sj to connect ('start' or 'end').
    """
    s_i = strokes[si]
    s_j = strokes[sj]

    if side_i == 'end' and side_j == 'start':
        strokes[si] = s_i + s_j
    elif side_i == 'end' and side_j == 'end':
        strokes[si] = s_i + s_j[::-1]
    elif side_i == 'start' and side_j == 'start':
        strokes[si] = s_j[::-1] + s_i
    else:  # start-end
        strokes[si] = s_j + s_i

    strokes.pop(sj)


def run_merge_pass(strokes: list[list[tuple]], assigned: list[set],
                    min_len: int = 5, max_angle: float = np.pi/4,
                    max_ratio: float = 0.2) -> list[list[tuple]]:
    """Run a single pass of direction-based stroke merging.

    Iteratively merges pairs of strokes that meet at junction clusters with
    aligned directions, until no more merges are possible.

    Args:
        strokes: List of stroke paths. Modified in place.
        assigned: List of junction cluster sets.
        min_len: Minimum stroke length to consider for merging. Default 5.
        max_angle: Maximum angle between directions for merge. Default pi/4.
        max_ratio: Minimum length ratio for merge. Default 0.2.

    Returns:
        The modified strokes list.
    """
    changed = True
    while changed:
        changed = False
        cluster_cache = {}
        cluster_map = _build_cluster_endpoint_map(strokes, assigned, cluster_cache)

        best_merge = _find_best_merge_pair(
            strokes, cluster_map, assigned, min_len, max_angle, max_ratio, cluster_cache
        )

        if best_merge:
            si, side_i, sj, side_j = best_merge
            _execute_merge(strokes, si, side_i, sj, side_j)
            changed = True

    return strokes


def _find_t_junction_candidate(strokes: list[list[tuple]], cluster_map: dict,
                                assigned: list[set],
                                endpoint_cache: dict[tuple[int, bool], int] = None) -> tuple | None:
    """Find a T-junction candidate for merging.

    Args:
        strokes: List of stroke paths.
        cluster_map: Cluster to endpoints mapping.
        assigned: List of junction cluster sets.
        endpoint_cache: Optional pre-built cache from _build_endpoint_cache().

    Returns:
        Tuple (cid, si, side_i, sj, side_j, second_longest_len) if found, else None.
    """
    T_JUNCTION_MAX_ANGLE = 2 * np.pi / 3  # 120 degrees

    for cid, entries in cluster_map.items():
        if len(entries) < 3:
            continue

        entries_sorted = sorted(entries, key=lambda e: len(strokes[e[0]]), reverse=True)
        shortest_idx, _shortest_side = entries_sorted[-1]
        second_longest_len = len(strokes[entries_sorted[1][0]])

        # Check shortest stroke has valid junctions (use cache if available)
        if endpoint_cache:
            s_sc = _get_cached_cluster(endpoint_cache, shortest_idx, False)
            s_ec = _get_cached_cluster(endpoint_cache, shortest_idx, True)
        else:
            s_sc = endpoint_cluster(strokes[shortest_idx], False, assigned)
            s_ec = endpoint_cluster(strokes[shortest_idx], True, assigned)
        if s_sc < 0 or s_ec < 0:
            continue
        if len(strokes[shortest_idx]) >= second_longest_len * 0.4:
            continue

        # Get the two longest strokes
        si, side_i = entries_sorted[0]
        sj, side_j = entries_sorted[1]
        if si == sj:
            continue

        # Check they don't form a loop (use cache if available)
        if endpoint_cache:
            far_i = _get_cached_cluster(endpoint_cache, si, side_i != 'end')
            far_j = _get_cached_cluster(endpoint_cache, sj, side_j != 'end')
        else:
            far_i = endpoint_cluster(strokes[si], from_end=(side_i != 'end'), assigned=assigned)
            far_j = endpoint_cluster(strokes[sj], from_end=(side_j != 'end'), assigned=assigned)
        if far_i >= 0 and far_i == far_j:
            continue

        # Check angle alignment
        dir_i = seg_dir(strokes[si], from_end=(side_i == 'end'))
        dir_j = seg_dir(strokes[sj], from_end=(side_j == 'end'))
        angle = np.pi - angle_between(dir_i, dir_j)

        if angle < T_JUNCTION_MAX_ANGLE:
            return (cid, si, side_i, sj, side_j, second_longest_len)

    return None


def _remove_short_cross_strokes(strokes: list[list[tuple]], cid: int,
                                 threshold_len: float, assigned: list[set],
                                 endpoint_cache: dict[tuple[int, bool], int] = None) -> bool:
    """Remove short cross-strokes at a junction.

    Args:
        strokes: List of stroke paths (modified in place).
        cid: Cluster ID to check.
        threshold_len: Length threshold (40% of second longest).
        assigned: List of junction cluster sets.
        endpoint_cache: Optional pre-built cache from _build_endpoint_cache().

    Returns:
        True if a stroke was removed.
    """
    for sk in range(len(strokes)):
        s = strokes[sk]
        if endpoint_cache:
            s_sc = _get_cached_cluster(endpoint_cache, sk, False)
            s_ec = _get_cached_cluster(endpoint_cache, sk, True)
        else:
            s_sc = endpoint_cluster(s, False, assigned)
            s_ec = endpoint_cluster(s, True, assigned)
        if s_sc >= 0 and s_ec >= 0 and len(s) < threshold_len and (s_sc == cid or s_ec == cid):
            strokes.pop(sk)
            return True
    return False


def merge_t_junctions(strokes: list[list[tuple]], junction_clusters: list[set],
                      assigned: list[set]) -> list[list[tuple]]:
    """Merge strokes at T-junctions by connecting the main through-stroke.

    T-junctions occur where three or more strokes meet, with two forming a
    main "through" path and others being branches. This function identifies
    the shortest stroke at such junctions and, if it's significantly shorter
    than the main branches, merges the two longest strokes while optionally
    removing the short cross-branch.

    Args:
        strokes: List of stroke paths. Modified in place.
        junction_clusters: Original list of junction cluster sets.
        assigned: List of assigned junction cluster sets.

    Returns:
        The modified strokes list.
    """
    changed = True
    while changed:
        changed = False
        # Build caches once per iteration (O(n) instead of O(n²))
        cluster_map = _build_cluster_endpoint_map(strokes, assigned)
        endpoint_cache = _build_endpoint_cache(strokes, assigned)
        candidate = _find_t_junction_candidate(strokes, cluster_map, assigned, endpoint_cache)

        if candidate:
            cid, si, side_i, sj, side_j, second_longest_len = candidate
            _execute_merge(strokes, si, side_i, sj, side_j)
            # Rebuild cache after merge since stroke indices changed
            endpoint_cache = _build_endpoint_cache(strokes, assigned)
            _remove_short_cross_strokes(strokes, cid, second_longest_len * 0.4, assigned, endpoint_cache)
            changed = True

    return strokes


def absorb_convergence_stubs(strokes: list[list[tuple]], junction_clusters: list[set],
                             assigned: list[set], conv_threshold: int = 18) -> list[list[tuple]]:
    """Absorb short convergence stubs into longer strokes at junction clusters.

    A convergence stub is a short stroke with one endpoint at a junction cluster
    and the other end free (not at any junction). These often appear at apex
    points of letters like 'A', 'V', 'W' where multiple strokes converge.

    Args:
        strokes: List of stroke paths. Modified in place.
        junction_clusters: List of original junction cluster sets.
        assigned: List of assigned junction cluster sets.
        conv_threshold: Maximum length for a stroke to be considered a stub.

    Returns:
        The modified strokes list.
    """
    changed = True
    while changed:
        changed = False
        # Build detailed cluster index once per iteration (O(n) instead of O(n²))
        detailed_index = _build_detailed_cluster_index(strokes, assigned)
        # Build endpoint cache for O(1) cluster lookups
        endpoint_cache = _build_endpoint_cache(strokes, assigned)

        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) < 2 or len(s) >= conv_threshold:
                continue
            sc = _get_cached_cluster(endpoint_cache, si, False)
            ec = _get_cached_cluster(endpoint_cache, si, True)

            if sc >= 0 and ec < 0:
                cluster_id = sc
                stub_path = list(s)
            elif ec >= 0 and sc < 0:
                cluster_id = ec
                stub_path = list(reversed(s))
            elif sc >= 0 and ec >= 0 and sc == ec:
                cluster_id = sc
                cluster = junction_clusters[sc]
                cx = sum(p[0] for p in cluster) / len(cluster)
                cy = sum(p[1] for p in cluster) / len(cluster)
                d_start = ((s[0][0] - cx) ** 2 + (s[0][1] - cy) ** 2) ** 0.5
                d_end = ((s[-1][0] - cx) ** 2 + (s[-1][1] - cy) ** 2) ** 0.5
                stub_path = list(reversed(s)) if d_start > d_end else list(s)
            else:
                continue

            # Count other endpoints at cluster using index (O(k) instead of O(n))
            endpoints_at_cluster = detailed_index.get(cluster_id, [])
            others_at_cluster = sum(1 for (idx, _) in endpoints_at_cluster if idx != si)
            if others_at_cluster < 2:
                continue

            stub_tip = stub_path[-1]
            cluster = junction_clusters[cluster_id]

            # Extend only strokes at this cluster using index (O(k) instead of O(n))
            strokes_extended = set()
            for sj, is_end in endpoints_at_cluster:
                if sj == si or sj in strokes_extended:
                    continue
                strokes_extended.add(sj)
                s2 = strokes[sj]
                at_end = is_end
                # Check if the other endpoint is also at this cluster
                if not at_end:
                    other_end_cid = _get_cached_cluster(endpoint_cache, sj, True)
                    if other_end_cid == cluster_id:
                        at_end = True  # Prefer extending from the end
                        at_start = False
                    else:
                        at_start = True
                else:
                    at_start = False

                tail, leg_end = get_stroke_tail(s2, at_end, cluster)
                extend_stroke_to_tip(s2, at_end, tail, leg_end, stub_tip)

            strokes.pop(si)
            changed = True
            break
    return strokes


def absorb_junction_stubs(strokes: list[list[tuple]], assigned: list[set],
                          stub_threshold: int = 20) -> list[list[tuple]]:
    """Absorb short stubs into neighboring strokes at junction clusters.

    Unlike convergence stubs (which have one free end), junction stubs have
    both endpoints at junction clusters. This function merges such stubs into
    the longest neighboring stroke at one of their junction clusters.

    Args:
        strokes: List of stroke paths. Modified in place.
        assigned: List of assigned junction cluster sets.
        stub_threshold: Maximum length for a stroke to be considered a stub.

    Returns:
        The modified strokes list.
    """
    changed = True
    while changed:
        changed = False
        # Build detailed cluster index once per iteration (O(n) instead of O(n²))
        detailed_index = _build_detailed_cluster_index(strokes, assigned)
        # Build endpoint cache for O(1) cluster lookups
        endpoint_cache = _build_endpoint_cache(strokes, assigned)

        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) >= stub_threshold:
                continue
            sc = _get_cached_cluster(endpoint_cache, si, False)
            ec = _get_cached_cluster(endpoint_cache, si, True)
            clusters_touching = set()
            if sc >= 0:
                clusters_touching.add(sc)
            if ec >= 0:
                clusters_touching.add(ec)
            if not clusters_touching:
                continue

            best_target = -1
            best_len = 0
            best_target_side = None
            best_stub_side = None

            # Use index to find strokes at each cluster (O(k) instead of O(n))
            for cid in clusters_touching:
                endpoints_at_cluster = detailed_index.get(cid, [])
                for sj, is_end in endpoints_at_cluster:
                    if sj == si:
                        continue
                    s2 = strokes[sj]
                    if len(s2) > best_len:
                        best_target = sj
                        best_len = len(s2)
                        best_target_side = 'end' if is_end else 'start'
                        best_stub_side = 'start' if sc == cid else 'end'

            if best_target >= 0:
                stub = s if best_stub_side == 'end' else list(reversed(s))
                target = strokes[best_target]
                if best_target_side == 'end':
                    strokes[best_target] = target + stub[1:]
                else:
                    strokes[best_target] = list(reversed(stub[1:])) + target
                strokes.pop(si)
                changed = True
                break
    return strokes


def _find_proximity_merge_target(stub: list[tuple], stub_idx: int, strokes: list[list[tuple]],
                                  stub_threshold: int, prox_threshold: int) -> tuple:
    """Find best merge target for a stub based on endpoint proximity.

    Returns:
        Tuple of (target_idx, target_side, stub_side, distance) or (-1, None, None, inf) if none found.
    """
    best_dist = prox_threshold
    best_target = -1
    best_target_side = None
    best_stub_side = None

    for stub_end in [False, True]:
        sp = stub[-1] if stub_end else stub[0]
        for sj, target in enumerate(strokes):
            if sj == stub_idx or len(target) < stub_threshold:
                continue
            for target_end in [False, True]:
                tp = target[-1] if target_end else target[0]
                d = ((sp[0] - tp[0]) ** 2 + (sp[1] - tp[1]) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_target = sj
                    best_target_side = 'end' if target_end else 'start'
                    best_stub_side = 'end' if stub_end else 'start'

    return best_target, best_target_side, best_stub_side, best_dist


def absorb_proximity_stubs(strokes: list[list[tuple]], stub_threshold: int = 20,
                           prox_threshold: int = 20) -> list[list[tuple]]:
    """Absorb short stubs by proximity to longer stroke endpoints.

    This function handles stubs that may not be at recognized junction clusters
    but are close to endpoints of longer strokes.

    Args:
        strokes: List of stroke paths. Modified in place.
        stub_threshold: Maximum length for a stroke to be considered a stub.
        prox_threshold: Maximum distance for proximity merging.

    Returns:
        The modified strokes list.
    """
    changed = True
    while changed:
        changed = False
        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) >= stub_threshold:
                continue

            best_target, best_target_side, best_stub_side, _ = _find_proximity_merge_target(
                s, si, strokes, stub_threshold, prox_threshold)

            if best_target < 0:
                continue

            stub = s if best_stub_side == 'end' else list(reversed(s))
            target = strokes[best_target]
            if best_target_side == 'end':
                strokes[best_target] = target + stub[1:]
            else:
                strokes[best_target] = list(reversed(stub[1:])) + target
            strokes.pop(si)
            changed = True
            break
    return strokes


def remove_orphan_stubs(strokes: list[list[tuple]], assigned: list[set],
                        stub_threshold: int = 20) -> list[list[tuple]]:
    """Remove orphaned short stubs with no neighbors at their junction clusters.

    An orphan stub is a short stroke at a junction cluster where no other
    strokes have endpoints.

    Args:
        strokes: List of stroke paths. Modified in place.
        assigned: List of assigned junction cluster sets.
        stub_threshold: Maximum length for a stroke to be considered for removal.

    Returns:
        The modified strokes list.
    """
    changed = True
    while changed:
        changed = False
        # Build caches once per iteration (O(n) instead of O(n²))
        cluster_index = _build_cluster_index(strokes, assigned)
        endpoint_cache = _build_endpoint_cache(strokes, assigned)

        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) >= stub_threshold:
                continue
            sc = _get_cached_cluster(endpoint_cache, si, False)
            ec = _get_cached_cluster(endpoint_cache, si, True)

            orphan = False
            for cid in [sc, ec]:
                if cid < 0:
                    continue
                strokes_at_cluster = cluster_index.get(cid, set())
                if len(strokes_at_cluster) <= 1:
                    orphan = True
                    break

            if orphan:
                strokes.pop(si)
                changed = True
                break

    return strokes
