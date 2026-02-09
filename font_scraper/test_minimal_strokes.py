#!/usr/bin/env python3
"""
Test suite for minimal_strokes_from_skeleton.

Evaluates stroke quality across multiple letters and fonts to catch regressions.
Run with: python3 test_minimal_strokes.py
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from stroke_core import min_strokes as minimal_strokes_from_skeleton, _skel as _analyze_skeleton
from stroke_rendering import render_glyph_mask
from stroke_flask import resolve_font_path
from stroke_templates import NUMPAD_TEMPLATES, NUMPAD_POS


def compute_stroke_coverage(strokes, mask, stroke_width=8):
    """Compute what fraction of the glyph is covered by strokes.

    Returns:
        coverage: fraction of glyph pixels covered (0-1)
        overshoot: fraction of stroke pixels outside glyph (0-1)
    """
    from PIL import Image, ImageDraw

    h, w = mask.shape
    stroke_img = Image.new('L', (w, h), 255)
    draw = ImageDraw.Draw(stroke_img)

    for stroke in strokes:
        if len(stroke) >= 2:
            points = [(p[0], p[1]) for p in stroke]
            draw.line(points, fill=0, width=stroke_width)

    stroke_mask = np.array(stroke_img) < 128
    glyph_mask = mask > 0

    # Coverage: fraction of glyph covered by strokes
    glyph_pixels = glyph_mask.sum()
    if glyph_pixels == 0:
        return 0.0, 1.0

    covered = (glyph_mask & stroke_mask).sum()
    coverage = covered / glyph_pixels

    # Overshoot: fraction of stroke pixels outside glyph
    stroke_pixels = stroke_mask.sum()
    if stroke_pixels == 0:
        return 0.0, 0.0

    outside = (stroke_mask & ~glyph_mask).sum()
    overshoot = outside / stroke_pixels

    return float(coverage), float(overshoot)


def compute_stroke_count_score(strokes, char):
    """Check if stroke count matches template.

    Returns:
        score: 1.0 if count matches, 0.0 otherwise
        expected: expected stroke count from template
        actual: actual stroke count
    """
    template = NUMPAD_TEMPLATES.get(char)
    if template is None:
        return 0.0, 0, len(strokes) if strokes else 0

    expected = len(template)
    actual = len(strokes) if strokes else 0
    score = 1.0 if actual == expected else 0.0

    return score, expected, actual


def _parse_waypoint(wp):
    """Parse a waypoint into (region_int, kind)."""
    import re

    # Handle tuple format: (position, hint)
    if isinstance(wp, tuple):
        wp_val = wp[0]
    else:
        wp_val = wp

    if isinstance(wp_val, int):
        return (wp_val, 'terminal')
    m = re.match(r'^v\((\d)\)$', str(wp_val))
    if m:
        return (int(m.group(1)), 'vertex')
    m = re.match(r'^c\((\d)\)$', str(wp_val))
    if m:
        return (int(m.group(1)), 'curve')
    raise ValueError(f"Unknown waypoint format: {wp}")


def _point_in_region(point, region, bbox, tolerance=0.2):
    """Check if a point is within a numpad region (with tolerance).

    Args:
        point: (x, y) pixel coordinates
        region: numpad region 1-9
        bbox: (x_min, y_min, x_max, y_max) glyph bounding box
        tolerance: fraction of bbox size for region matching

    Returns:
        True if point is within the region (with tolerance)
    """
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    w, h = x_max - x_min, y_max - y_min

    if w == 0 or h == 0:
        return False

    # Normalize point to 0-1 range
    nx = (x - x_min) / w
    ny = (y - y_min) / h

    # Get expected position for region
    rx, ry = NUMPAD_POS[region]

    # Check if within tolerance
    return abs(nx - rx) <= tolerance and abs(ny - ry) <= tolerance


def compute_topology_score(strokes, char, bbox):
    """Check if strokes pass through expected waypoint regions.

    For each stroke in the template, checks if the corresponding generated
    stroke passes through the expected numpad regions.

    Args:
        strokes: list of strokes [[[x,y], ...], ...]
        char: character being tested
        bbox: (x_min, y_min, x_max, y_max) glyph bounding box

    Returns:
        score: fraction of waypoints hit (0-1)
        details: dict with hit/miss info per stroke
    """
    template = NUMPAD_TEMPLATES.get(char)
    if template is None or not strokes:
        return 0.0, {'error': 'no template or strokes'}

    if len(strokes) != len(template):
        # Can't match topology if stroke count differs
        return 0.0, {'error': f'stroke count mismatch: {len(strokes)} vs {len(template)}'}

    total_waypoints = 0
    waypoints_hit = 0
    details = []

    for stroke_idx, (stroke, template_stroke) in enumerate(zip(strokes, template)):
        stroke_details = {
            'stroke_idx': stroke_idx,
            'waypoints': [],
            'hits': 0,
            'total': len(template_stroke)
        }

        for wp in template_stroke:
            region, kind = _parse_waypoint(wp)
            total_waypoints += 1

            # Check if any point in the stroke is in this region
            hit = False
            for point in stroke:
                if _point_in_region(point, region, bbox):
                    hit = True
                    break

            if hit:
                waypoints_hit += 1
                stroke_details['hits'] += 1

            stroke_details['waypoints'].append({
                'region': region,
                'kind': kind,
                'hit': hit
            })

        details.append(stroke_details)

    score = waypoints_hit / total_waypoints if total_waypoints > 0 else 0.0
    return score, {'strokes': details, 'total': total_waypoints, 'hits': waypoints_hit}


def compute_combined_score(coverage, overshoot, stroke_count_score, topology_score,
                           weights=None):
    """Compute weighted combined score.

    Args:
        coverage: glyph coverage (0-1)
        overshoot: stroke overshoot (0-1, lower is better)
        stroke_count_score: 1 if count matches, 0 otherwise
        topology_score: fraction of waypoints hit (0-1)
        weights: dict with keys 'coverage', 'overshoot', 'stroke_count', 'topology'
                 default: equal weights

    Returns:
        Combined score (0-1)
    """
    if weights is None:
        weights = {
            'coverage': 0.35,
            'overshoot': 0.15,
            'stroke_count': 0.20,
            'topology': 0.30
        }

    score = (
        weights['coverage'] * coverage +
        weights['overshoot'] * (1.0 - overshoot) +
        weights['stroke_count'] * stroke_count_score +
        weights['topology'] * topology_score
    )

    return score


def test_letter(font_path, char, canvas_size=224, verbose=False):
    """Test minimal strokes for a single letter.

    Returns dict with test results including all four scoring metrics.
    """
    result = {
        'char': char,
        'font': Path(font_path).name,
        'status': 'unknown',
        # Individual metrics
        'coverage': 0.0,
        'overshoot': 0.0,
        'stroke_count_score': 0.0,
        'topology_score': 0.0,
        # Combined score
        'score': 0.0,
        # Additional info
        'num_strokes': 0,
        'expected_strokes': 0,
        'total_points': 0,
        'topology_hits': 0,
        'topology_total': 0,
        'error': None,
    }

    try:
        font_path = resolve_font_path(font_path)

        # Generate strokes
        strokes = minimal_strokes_from_skeleton(font_path, char, canvas_size)

        if strokes is None:
            result['status'] = 'no_template'
            result['error'] = 'No template for this character'
            return result

        result['num_strokes'] = len(strokes)
        result['total_points'] = sum(len(s) for s in strokes)

        # Render glyph mask for scoring
        mask = render_glyph_mask(font_path, char, canvas_size)
        if mask is None:
            result['status'] = 'no_glyph'
            result['error'] = 'Could not render glyph'
            return result

        # Get glyph bounding box for topology scoring
        rows, cols = np.where(mask)
        if len(rows) == 0:
            result['status'] = 'no_glyph'
            result['error'] = 'Empty glyph mask'
            return result
        bbox = (float(cols.min()), float(rows.min()),
                float(cols.max()), float(rows.max()))

        # 1. Coverage and overshoot
        coverage, overshoot = compute_stroke_coverage(strokes, mask)

        # 2. Stroke count
        stroke_count_score, expected, actual = compute_stroke_count_score(strokes, char)
        result['expected_strokes'] = expected

        # 3. Topology
        topology_score, topology_details = compute_topology_score(strokes, char, bbox)
        if isinstance(topology_details, dict) and 'hits' in topology_details:
            result['topology_hits'] = topology_details['hits']
            result['topology_total'] = topology_details['total']

        # 4. Combined score
        combined = compute_combined_score(coverage, overshoot, stroke_count_score, topology_score)

        result['status'] = 'ok'
        result['coverage'] = round(coverage, 3)
        result['overshoot'] = round(overshoot, 3)
        result['stroke_count_score'] = round(stroke_count_score, 3)
        result['topology_score'] = round(topology_score, 3)
        result['score'] = round(combined, 3)
        result['strokes'] = strokes

        if verbose:
            print(f"  {char}: score={combined:.3f} cov={coverage:.3f} over={overshoot:.3f} "
                  f"strokes={actual}/{expected} topo={topology_score:.2f} "
                  f"({result['topology_hits']}/{result['topology_total']})")

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        if verbose:
            print(f"  {char}: ERROR - {e}")
            import traceback
            traceback.print_exc()

    return result


def run_test_suite(font_path=None, chars=None, verbose=True):
    """Run test suite on multiple characters.

    Args:
        font_path: Path to font file (default: Engineer's Hand)
        chars: List of characters to test (default: A-Z)
        verbose: Print progress

    Returns:
        List of test result dicts
    """
    if font_path is None:
        font_path = "fonts/dafont/003 Engineer_'s Hand.ttf"

    if chars is None:
        chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    if verbose:
        print(f"Testing {len(chars)} characters from {Path(font_path).name}")
        print("-" * 60)

    results = []
    for char in chars:
        result = test_letter(font_path, char, verbose=verbose)
        results.append(result)

    # Summary
    if verbose:
        print("-" * 60)
        ok_results = [r for r in results if r['status'] == 'ok']
        if ok_results:
            n = len(ok_results)
            avg_score = sum(r['score'] for r in ok_results) / n
            avg_coverage = sum(r['coverage'] for r in ok_results) / n
            avg_overshoot = sum(r['overshoot'] for r in ok_results) / n
            avg_stroke_count = sum(r['stroke_count_score'] for r in ok_results) / n
            avg_topology = sum(r['topology_score'] for r in ok_results) / n

            print(f"Results: {n}/{len(results)} characters OK")
            print(f"Combined Score:    {avg_score:.3f}")
            print(f"  - Coverage:      {avg_coverage:.3f}")
            print(f"  - Overshoot:     {avg_overshoot:.3f}")
            print(f"  - Stroke Count:  {avg_stroke_count:.3f}")
            print(f"  - Topology:      {avg_topology:.3f}")

        failed = [r for r in results if r['status'] != 'ok']
        if failed:
            print(f"Failed: {[r['char'] for r in failed]}")

    return results


def save_baseline(results, path='test_baseline.json'):
    """Save test results as baseline for regression testing."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved baseline to {path}")


def compare_to_baseline(results, baseline_path='test_baseline.json', threshold=0.05):
    """Compare results to baseline and report regressions.

    Args:
        results: Current test results
        baseline_path: Path to baseline JSON
        threshold: Score drop threshold to report as regression

    Returns:
        List of regression dicts
    """
    try:
        with open(baseline_path) as f:
            baseline = json.load(f)
    except FileNotFoundError:
        print(f"No baseline found at {baseline_path}")
        return []

    baseline_map = {r['char']: r for r in baseline}
    regressions = []

    metrics = ['score', 'coverage', 'overshoot', 'stroke_count_score', 'topology_score']

    for result in results:
        char = result['char']
        if char not in baseline_map:
            continue

        base = baseline_map[char]
        if result['status'] != 'ok' and base['status'] == 'ok':
            regressions.append({
                'char': char,
                'type': 'status_change',
                'was': base['status'],
                'now': result['status'],
            })
        elif result['status'] == 'ok' and base['status'] == 'ok':
            # Check each metric for regression
            for metric in metrics:
                if metric not in base:
                    continue  # Skip if metric not in baseline (old format)

                base_val = base[metric]
                curr_val = result[metric]

                # For overshoot, higher is worse (so regression is increase)
                if metric == 'overshoot':
                    change = curr_val - base_val
                    if change > threshold:
                        regressions.append({
                            'char': char,
                            'type': f'{metric}_regression',
                            'was': base_val,
                            'now': curr_val,
                            'change': round(change, 3),
                        })
                else:
                    # For other metrics, lower is worse
                    change = base_val - curr_val
                    if change > threshold:
                        regressions.append({
                            'char': char,
                            'type': f'{metric}_regression',
                            'was': base_val,
                            'now': curr_val,
                            'change': round(change, 3),
                        })

    if regressions:
        print(f"\n⚠️  REGRESSIONS DETECTED ({len(regressions)}):")
        for r in regressions:
            if r['type'] == 'status_change':
                print(f"  {r['char']}: status {r['was']} -> {r['now']}")
            else:
                direction = '↑' if 'overshoot' in r['type'] else '↓'
                print(f"  {r['char']}: {r['type'].replace('_regression', '')} "
                      f"{r['was']:.3f} -> {r['now']:.3f} ({direction}{abs(r['change']):.3f})")
    else:
        print("\n✓ No regressions detected")

    return regressions


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test minimal stroke generation')
    parser.add_argument('--font', default="fonts/dafont/003 Engineer_'s Hand.ttf",
                       help='Font file to test')
    parser.add_argument('--chars', default='ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                       help='Characters to test')
    parser.add_argument('--save-baseline', action='store_true',
                       help='Save results as new baseline')
    parser.add_argument('--check', action='store_true',
                       help='Compare to baseline and report regressions')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')

    args = parser.parse_args()

    results = run_test_suite(args.font, list(args.chars), verbose=not args.json)

    if args.json:
        print(json.dumps(results, indent=2))

    if args.save_baseline:
        save_baseline(results)

    if args.check:
        regressions = compare_to_baseline(results)
        sys.exit(1 if regressions else 0)
