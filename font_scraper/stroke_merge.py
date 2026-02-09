"""Stroke merging and absorption functions.

This module contains functions for merging strokes at junctions,
absorbing short stubs, and cleaning up stroke paths.
"""

from collections import defaultdict

import numpy as np


def seg_dir(stroke: list[tuple], from_end: bool = False, n_samples: int = 5) -> tuple[float, float]:
    """Get direction vector for end of stroke (averaged over last n_samples)."""
    if len(stroke) < 2:
        return (1.0, 0.0)

    pts = stroke[-n_samples:] if from_end else stroke[:n_samples]
    if len(pts) < 2:
        pts = stroke[-2:] if from_end else stroke[:2]

    if from_end:
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
    else:
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]

    length = (dx*dx + dy*dy)**0.5
    if length < 1e-6:
        return (1.0, 0.0)
    return (dx/length, dy/length)


def angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    """Compute angle between two direction vectors (in radians)."""
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    dot = max(-1.0, min(1.0, dot))
    return np.arccos(dot)


def endpoint_cluster(stroke: list[tuple], from_end: bool, assigned: list[set]) -> int:
    """Find which junction cluster (if any) contains the stroke endpoint.

    Args:
        stroke: List of (x, y) points
        from_end: True to check end, False to check start
        assigned: List of junction cluster sets

    Returns:
        Cluster index, or -1 if not at a junction
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


def run_merge_pass(strokes: list[list[tuple]], assigned: list[set],
                   min_len: int = 0, max_angle: float = np.pi/4,
                   max_ratio: float = 0) -> list[list[tuple]]:
    """Merge strokes through junction clusters by direction alignment.

    Args:
        strokes: List of stroke paths
        assigned: List of junction cluster sets
        min_len: Minimum stroke length to consider
        max_angle: Maximum angle difference for merging
        max_ratio: If > 0, reject pairs where max(len)/min(len) > ratio
    """
    changed = True
    while changed:
        changed = False
        cluster_map = defaultdict(list)
        for si, s in enumerate(strokes):
            sc = endpoint_cluster(s, False, assigned)
            if sc >= 0:
                cluster_map[sc].append((si, 'start'))
            ec = endpoint_cluster(s, True, assigned)
            if ec >= 0:
                cluster_map[ec].append((si, 'end'))

        best_score = float('inf')
        best_merge = None
        for _cid, entries in cluster_map.items():
            if len(entries) < 2:
                continue
            for ai in range(len(entries)):
                si, side_i = entries[ai]
                dir_i = seg_dir(strokes[si], from_end=(side_i == 'end'))
                for bi in range(ai + 1, len(entries)):
                    sj, side_j = entries[bi]
                    if sj == si:
                        continue
                    li, lj = len(strokes[si]), len(strokes[sj])
                    if min(li, lj) < min_len:
                        continue
                    if max_ratio > 0 and max(li, lj) / max(min(li, lj), 1) > max_ratio:
                        continue
                    # Don't merge with a loop stroke
                    sci = endpoint_cluster(strokes[si], False, assigned)
                    eci = endpoint_cluster(strokes[si], True, assigned)
                    scj = endpoint_cluster(strokes[sj], False, assigned)
                    ecj = endpoint_cluster(strokes[sj], True, assigned)
                    if sci >= 0 and sci == eci:
                        continue
                    if scj >= 0 and scj == ecj:
                        continue
                    dir_j = seg_dir(strokes[sj], from_end=(side_j == 'end'))
                    angle = np.pi - angle_between(dir_i, dir_j)
                    if angle < max_angle and angle < best_score:
                        best_score = angle
                        best_merge = (si, side_i, sj, side_j)

        if best_merge:
            si, side_i, sj, side_j = best_merge
            seg_i = strokes[si] if side_i == 'end' else list(reversed(strokes[si]))
            seg_j = strokes[sj] if side_j == 'start' else list(reversed(strokes[sj]))
            merged_stroke = seg_i + seg_j[1:]
            hi, lo = max(si, sj), min(si, sj)
            strokes.pop(hi)
            strokes.pop(lo)
            strokes.append(merged_stroke)
            changed = True
    return strokes


def merge_t_junctions(strokes: list[list[tuple]], junction_clusters: list[set],
                      assigned: list[set]) -> list[list[tuple]]:
    """Merge strokes at T-junctions.

    At junctions with 3+ strokes, if the shortest stroke is a cross-branch
    and much shorter than the main branches, merge the two longest with
    a relaxed angle threshold.
    """
    changed = True
    while changed:
        changed = False
        cluster_map = defaultdict(list)
        for si, s in enumerate(strokes):
            sc = endpoint_cluster(s, False, assigned)
            if sc >= 0:
                cluster_map[sc].append((si, 'start'))
            ec = endpoint_cluster(s, True, assigned)
            if ec >= 0:
                cluster_map[ec].append((si, 'end'))

        for cid, entries in cluster_map.items():
            if len(entries) < 3:
                continue
            entries_sorted = sorted(entries, key=lambda e: len(strokes[e[0]]), reverse=True)
            shortest_idx, _shortest_side = entries_sorted[-1]
            shortest_stroke = strokes[shortest_idx]
            second_longest_len = len(strokes[entries_sorted[1][0]])

            s_sc = endpoint_cluster(shortest_stroke, False, assigned)
            s_ec = endpoint_cluster(shortest_stroke, True, assigned)
            if s_sc < 0 or s_ec < 0:
                continue
            if len(shortest_stroke) >= second_longest_len * 0.4:
                continue

            si, side_i = entries_sorted[0]
            sj, side_j = entries_sorted[1]
            if si == sj:
                continue

            far_i = endpoint_cluster(strokes[si], from_end=(side_i != 'end'), assigned=assigned)
            far_j = endpoint_cluster(strokes[sj], from_end=(side_j != 'end'), assigned=assigned)
            if far_i >= 0 and far_i == far_j:
                continue

            dir_i = seg_dir(strokes[si], from_end=(side_i == 'end'))
            dir_j = seg_dir(strokes[sj], from_end=(side_j == 'end'))
            angle = np.pi - angle_between(dir_i, dir_j)

            if angle < 2 * np.pi / 3:
                seg_i = strokes[si] if side_i == 'end' else list(reversed(strokes[si]))
                seg_j = strokes[sj] if side_j == 'start' else list(reversed(strokes[sj]))
                merged_stroke = seg_i + seg_j[1:]
                hi, lo = max(si, sj), min(si, sj)
                strokes.pop(hi)
                strokes.pop(lo)
                strokes.append(merged_stroke)

                for sk in range(len(strokes)):
                    s = strokes[sk]
                    s_sc2 = endpoint_cluster(s, False, assigned)
                    s_ec2 = endpoint_cluster(s, True, assigned)
                    if s_sc2 >= 0 and s_ec2 >= 0 and len(s) < second_longest_len * 0.4:
                        if s_sc2 == cid or s_ec2 == cid:
                            strokes.pop(sk)
                            break
                changed = True
                break
    return strokes


def get_stroke_tail(stroke: list[tuple], at_end: bool, cluster: set) -> tuple[list[tuple], tuple]:
    """Get the tail points of a stroke before a junction cluster.

    Returns:
        Tuple of (tail_points, leg_end_point)
    """
    if at_end:
        tail = []
        for k in range(len(stroke) - 1, -1, -1):
            pt = tuple(stroke[k]) if isinstance(stroke[k], (list, tuple)) else stroke[k]
            if len(tail) >= 8:
                break
            if (int(round(pt[0])), int(round(pt[1]))) not in cluster or not tail:
                tail.insert(0, pt)
        return tail, stroke[-1]
    else:
        tail = []
        for k in range(len(stroke)):
            pt = tuple(stroke[k]) if isinstance(stroke[k], (list, tuple)) else stroke[k]
            if len(tail) >= 8:
                break
            if (int(round(pt[0])), int(round(pt[1]))) not in cluster or not tail:
                tail.append(pt)
        return list(reversed(tail)), stroke[0]


def extend_stroke_to_tip(stroke: list[tuple], at_end: bool, tail: list[tuple],
                         leg_end: tuple, stub_tip: tuple) -> None:
    """Extend a stroke from its junction end toward a stub tip (modifies in place)."""
    if len(tail) >= 2:
        dx = tail[-1][0] - tail[0][0]
        dy = tail[-1][1] - tail[0][1]
        leg_len = (dx * dx + dy * dy) ** 0.5
    else:
        dx, dy = 0, 0
        leg_len = 0

    tip_dx = stub_tip[0] - leg_end[0]
    tip_dy = stub_tip[1] - leg_end[1]
    tip_dist = (tip_dx * tip_dx + tip_dy * tip_dy) ** 0.5
    steps = max(1, int(round(tip_dist)))

    if leg_len > 0.01:
        ux, uy = dx / leg_len, dy / leg_len
        ext_pts = []
        for k in range(1, steps + 1):
            t = k / steps
            ex = leg_end[0] + ux * k
            ey = leg_end[1] + uy * k
            px = ex * (1 - t) + stub_tip[0] * t
            py = ey * (1 - t) + stub_tip[1] * t
            ext_pts.append((px, py))
    else:
        ext_pts = []
        for k in range(1, steps + 1):
            t = k / steps
            ext_pts.append((leg_end[0] + tip_dx * t, leg_end[1] + tip_dy * t))

    if at_end:
        stroke.extend(ext_pts)
    else:
        for p in reversed(ext_pts):
            stroke.insert(0, p)


def absorb_convergence_stubs(strokes: list[list[tuple]], junction_clusters: list[set],
                             assigned: list[set], conv_threshold: int = 18) -> list[list[tuple]]:
    """Absorb short convergence stubs into longer strokes at junction clusters.

    A convergence stub is a short stroke with one endpoint in a junction cluster
    and the other end free (e.g. the pointed apex of letter A).
    """
    changed = True
    while changed:
        changed = False
        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) < 2 or len(s) >= conv_threshold:
                continue
            sc = endpoint_cluster(s, False, assigned)
            ec = endpoint_cluster(s, True, assigned)

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

            others_at_cluster = 0
            for sj in range(len(strokes)):
                if sj == si:
                    continue
                if endpoint_cluster(strokes[sj], False, assigned) == cluster_id:
                    others_at_cluster += 1
                if endpoint_cluster(strokes[sj], True, assigned) == cluster_id:
                    others_at_cluster += 1
            if others_at_cluster < 2:
                continue

            stub_tip = stub_path[-1]
            cluster = junction_clusters[cluster_id]
            for sj in range(len(strokes)):
                if sj == si:
                    continue
                s2 = strokes[sj]
                at_end = endpoint_cluster(s2, True, assigned) == cluster_id
                at_start = (not at_end) and endpoint_cluster(s2, False, assigned) == cluster_id
                if not at_end and not at_start:
                    continue

                tail, leg_end = get_stroke_tail(s2, at_end, cluster)
                extend_stroke_to_tip(s2, at_end, tail, leg_end, stub_tip)

            strokes.pop(si)
            changed = True
            break
    return strokes


def absorb_junction_stubs(strokes: list[list[tuple]], assigned: list[set],
                          stub_threshold: int = 20) -> list[list[tuple]]:
    """Absorb short stubs into neighboring strokes at junction clusters."""
    changed = True
    while changed:
        changed = False
        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) >= stub_threshold:
                continue
            sc = endpoint_cluster(s, False, assigned)
            ec = endpoint_cluster(s, True, assigned)
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
            for cid in clusters_touching:
                for sj in range(len(strokes)):
                    if sj == si:
                        continue
                    s2 = strokes[sj]
                    tc_start = endpoint_cluster(s2, False, assigned)
                    tc_end = endpoint_cluster(s2, True, assigned)
                    if tc_start == cid and len(s2) > best_len:
                        best_target = sj
                        best_len = len(s2)
                        best_target_side = 'start'
                        best_stub_side = 'start' if sc == cid else 'end'
                    if tc_end == cid and len(s2) > best_len:
                        best_target = sj
                        best_len = len(s2)
                        best_target_side = 'end'
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


def absorb_proximity_stubs(strokes: list[list[tuple]], stub_threshold: int = 20,
                           prox_threshold: int = 20) -> list[list[tuple]]:
    """Absorb short stubs by proximity to longer stroke endpoints."""
    changed = True
    while changed:
        changed = False
        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) >= stub_threshold:
                continue
            best_dist = prox_threshold
            best_target = -1
            best_target_side = None
            best_stub_side = None
            for stub_end in [False, True]:
                sp = s[-1] if stub_end else s[0]
                for sj in range(len(strokes)):
                    if sj == si or len(strokes[sj]) < stub_threshold:
                        continue
                    for target_end in [False, True]:
                        tp = strokes[sj][-1] if target_end else strokes[sj][0]
                        d = ((sp[0] - tp[0]) ** 2 + (sp[1] - tp[1]) ** 2) ** 0.5
                        if d < best_dist:
                            best_dist = d
                            best_target = sj
                            best_target_side = 'end' if target_end else 'start'
                            best_stub_side = 'end' if stub_end else 'start'
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


def remove_orphan_stubs(strokes: list[list[tuple]], assigned: list[set],
                        stub_threshold: int = 20) -> list[list[tuple]]:
    """Remove orphaned short stubs that have no neighbors at their junction clusters."""
    changed = True
    while changed:
        changed = False
        for si in range(len(strokes)):
            s = strokes[si]
            if len(s) >= stub_threshold:
                continue
            sc = endpoint_cluster(s, False, assigned)
            ec = endpoint_cluster(s, True, assigned)

            orphan = False
            for cid in [sc, ec]:
                if cid < 0:
                    continue
                neighbors = 0
                for sj in range(len(strokes)):
                    if sj == si:
                        continue
                    if endpoint_cluster(strokes[sj], False, assigned) == cid:
                        neighbors += 1
                    if endpoint_cluster(strokes[sj], True, assigned) == cid:
                        neighbors += 1
                if neighbors == 0:
                    orphan = True
                    break

            if orphan:
                strokes.pop(si)
                changed = True
                break
    return strokes


# Aliases for backwards compatibility
_seg_dir = seg_dir
_angle = angle_between
_endpoint_cluster = endpoint_cluster
_run_merge_pass = run_merge_pass
_merge_t_junctions = merge_t_junctions
_get_stroke_tail = get_stroke_tail
_extend_stroke_to_tip = extend_stroke_to_tip
_absorb_convergence_stubs = absorb_convergence_stubs
_absorb_junction_stubs = absorb_junction_stubs
_absorb_proximity_stubs = absorb_proximity_stubs
_remove_orphan_stubs = remove_orphan_stubs
