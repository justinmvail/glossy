"""Core stroke processing functions.

This module contains the main stroke generation and processing functions
used by the Flask routes and other modules.
"""

from stroke_flask import resolve_font_path
from stroke_lib.analysis.skeleton import SkeletonAnalyzer
from stroke_merge import (
    absorb_convergence_stubs,
    absorb_junction_stubs,
    absorb_proximity_stubs,
    merge_t_junctions,
    remove_orphan_stubs,
    run_merge_pass,
)
from stroke_pipeline import MinimalStrokePipeline
from stroke_rendering import render_glyph_mask
from stroke_scoring import quick_stroke_score
from stroke_skeleton import (
    find_skeleton_segments,
    generate_straight_line,
    resample_path,
    trace_segment,
    trace_to_region,
)
from stroke_templates import NUMPAD_TEMPLATE_VARIANTS
from stroke_utils import point_in_region


def _skel(mask):
    """Analyze skeleton and return info dict compatible with legacy format."""
    info = SkeletonAnalyzer(merge_distance=12).analyze(mask)
    if not info:
        return None
    from collections import defaultdict
    adj = defaultdict(list)
    for p, n in info.adj.items():
        adj[p] = list(n)
    return {
        'skel_set': info.skel_set,
        'adj': adj,
        'junction_pixels': info.junction_pixels,
        'junction_clusters': info.junction_clusters,
        'assigned': info.assigned,
        'endpoints': info.endpoints,
    }


def skel_markers(mask):
    """Detect skeleton markers from mask."""
    markers = SkeletonAnalyzer(merge_distance=12).detect_markers(mask)
    return [m.to_dict() for m in markers]


def skel_strokes(mask, min_len=5, min_stroke_len=None):
    """Extract stroke paths from skeleton."""
    if min_stroke_len is not None:
        min_len = min_stroke_len
    info = _skel(mask)
    if not info:
        return []

    adj, eps = info['adj'], set(info['endpoints'])
    jps, jcs = set(info['junction_pixels']), info['junction_clusters']
    stops, vedges, raw = eps | jps, set(), []

    def trace(s, nb):
        e = (min(s, nb), max(s, nb))
        if e in vedges:
            return None
        vedges.add(e)
        path, cur, prev = [s, nb], nb, s
        while True:
            if cur in stops and len(path) > 2:
                break
            cands = [(n, (min(cur, n), max(cur, n))) for n in adj.get(cur, [])
                     if n != prev and (min(cur, n), max(cur, n)) not in vedges]
            if not cands:
                break
            nxt, ne = cands[0]
            vedges.add(ne)
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    for st in sorted(eps):
        for nb in adj.get(st, []):
            if (p := trace(st, nb)) and len(p) >= 2:
                raw.append(p)
    for st in sorted(jps):
        for nb in adj.get(st, []):
            if (p := trace(st, nb)) and len(p) >= 2:
                raw.append(p)

    strokes = [s for s in raw if len(s) >= min_len]
    asgn = [set(c) for c in jcs]
    strokes = merge_t_junctions(strokes, jcs, asgn)
    strokes = run_merge_pass(strokes, asgn, min_len=0)
    strokes = absorb_convergence_stubs(strokes, jcs, asgn)
    strokes = absorb_junction_stubs(strokes, asgn)
    strokes = absorb_proximity_stubs(strokes)
    strokes = remove_orphan_stubs(strokes, asgn)

    return [[[float(x), float(y)] for x, y in s] for s in strokes]


def _merge_to_expected_count(strokes, char):
    """Merge strokes to match expected count from template."""
    variants = NUMPAD_TEMPLATE_VARIANTS.get(char, {})
    if not variants or not strokes:
        return strokes
    expected = min(len(t) for t in variants.values())
    if len(strokes) <= expected:
        return strokes
    # Greedy merge: find closest endpoints and merge
    while len(strokes) > expected and len(strokes) > 1:
        best_dist, best_i, best_j, reverse_j = float('inf'), -1, -1, False
        for i in range(len(strokes)):
            for j in range(i + 1, len(strokes)):
                # Check all endpoint combinations
                for end_i, end_j, rev in [(strokes[i][-1], strokes[j][0], False),
                                          (strokes[i][-1], strokes[j][-1], True),
                                          (strokes[i][0], strokes[j][0], False),
                                          (strokes[i][0], strokes[j][-1], True)]:
                    d = (end_i[0] - end_j[0])**2 + (end_i[1] - end_j[1])**2
                    if d < best_dist:
                        best_dist, best_i, best_j, reverse_j = d, i, j, rev
        if best_i < 0:
            break
        # Merge j into i
        s_j = strokes[best_j][::-1] if reverse_j else strokes[best_j]
        strokes[best_i] = strokes[best_i] + s_j
        strokes.pop(best_j)
    return strokes


def min_strokes(fp, c, cs=224, tpl=None, ret_var=False):
    """Generate minimal strokes using the pipeline."""
    pipe = MinimalStrokePipeline(
        fp, c, cs,
        resolve_font_path_fn=resolve_font_path,
        render_glyph_mask_fn=render_glyph_mask,
        analyze_skeleton_fn=_skel,
        find_skeleton_segments_fn=find_skeleton_segments,
        point_in_region_fn=point_in_region,
        trace_segment_fn=trace_segment,
        trace_to_region_fn=trace_to_region,
        generate_straight_line_fn=generate_straight_line,
        resample_path_fn=resample_path,
        skeleton_to_strokes_fn=skel_strokes,
        apply_stroke_template_fn=_merge_to_expected_count,
        adjust_stroke_paths_fn=lambda st, c, m: st,  # No-op for now
        quick_stroke_score_fn=quick_stroke_score,
    )

    if tpl:
        st = pipe.run(tpl, trace_paths=True)
        return (st, 'custom') if ret_var else st

    result = pipe.evaluate_all_variants()
    if ret_var:
        return result.strokes, result.variant_name
    return result.strokes


def auto_fit(fp, c, cs=224, ret_mark=False):
    """Auto-fit strokes using affine optimization."""
    from stroke_affine import optimize_affine

    result = optimize_affine(fp, c, cs)
    if result is None:
        return (None, []) if ret_mark else None

    strokes, _score, _, _ = result
    if not strokes:
        return (None, []) if ret_mark else None

    # Convert to list format
    stroke_list = [[[float(x), float(y)] for x, y in s] for s in strokes if len(s) >= 2]

    if ret_mark:
        markers = []
        for i, s in enumerate(stroke_list):
            if len(s) >= 2:
                markers.append({'x': s[0][0], 'y': s[0][1], 'type': 'start', 'stroke_id': i})
                markers.append({'x': s[-1][0], 'y': s[-1][1], 'type': 'stop', 'stroke_id': i})
        return stroke_list, markers

    return stroke_list
