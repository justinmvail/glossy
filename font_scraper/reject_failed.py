#!/usr/bin/env python3
"""Verify all fonts and reject failures directly in the database.

Uses multiprocessing to speed up the CPU-bound quality checks.
"""
import json
import os
import sqlite3
import sys
import time
from multiprocessing import Pool, cpu_count

# Must be importable from font_scraper directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import ImageFont
from stroke_rendering import (
    resolve_font_path,
    analyze_shape_metrics,
    check_case_mismatch,
    check_char_holes,
    render_text_for_analysis,
    get_char_shape_count,
)
from stroke_services_core import (
    MIN_SHAPE_COUNT,
    MAX_SHAPE_COUNT,
    MAX_WIDTH_RATIO,
    EXPECTED_EXCLAMATION_SHAPES,
)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts.db')


def check_font(args):
    """Check a single font. Returns (font_id, details_json) if bad, else None."""
    fid, name, file_path = args
    try:
        fp = resolve_font_path(file_path)
        pf = ImageFont.truetype(fp, 60)

        arr = render_text_for_analysis(pf, "Hello World")
        if arr is None:
            return (fid, json.dumps({'error': 'render_failed'}))

        shape_count, max_width = analyze_shape_metrics(arr, arr.shape[1])
        case_mismatches = check_case_mismatch(fp)
        l_has_hole = check_char_holes(pf, 'l')
        exclaim_shapes = get_char_shape_count(pf, '!')

        issues = []
        if shape_count < MIN_SHAPE_COUNT or shape_count > MAX_SHAPE_COUNT:
            issues.append(f"shapes={shape_count}")
        if max_width > MAX_WIDTH_RATIO:
            issues.append(f"width={round(max_width * 100, 1)}%")
        if l_has_hole:
            issues.append('l_has_hole')
        if exclaim_shapes != EXPECTED_EXCLAMATION_SHAPES:
            issues.append(f"exclaim_shapes={exclaim_shapes}")
        if case_mismatches:
            issues.append(f"case_issues={','.join(case_mismatches)}")

        if issues:
            details = json.dumps({
                'shapes': int(shape_count),
                'max_width_pct': round(float(max_width) * 100, 1),
                'l_has_hole': bool(l_has_hole),
                'exclaim_shapes': int(exclaim_shapes),
                'case_mismatches': case_mismatches,
            })
            return (fid, details)

        return None
    except Exception as e:
        return (fid, json.dumps({'error': str(e)}))


def main():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    fonts = db.execute('''
        SELECT f.id, f.name, f.file_path
        FROM fonts f
        WHERE f.id NOT IN (SELECT font_id FROM font_removals)
        ORDER BY f.name
    ''').fetchall()

    total = len(fonts)
    print(f"Checking {total} fonts...")

    args_list = [(f['id'], f['name'], f['file_path']) for f in fonts]

    workers = max(1, cpu_count() - 1)
    print(f"Using {workers} workers")

    start = time.time()
    rejected = 0
    checked = 0
    batch = []
    BATCH_SIZE = 500

    with Pool(workers) as pool:
        for result in pool.imap_unordered(check_font, args_list, chunksize=20):
            checked += 1

            if result is not None:
                batch.append(result)

            if len(batch) >= BATCH_SIZE:
                db.executemany(
                    "INSERT OR IGNORE INTO font_removals (font_id, reason_id, details) VALUES (?, 8, ?)",
                    batch
                )
                db.commit()
                rejected += len(batch)
                batch = []

            if checked % 500 == 0:
                elapsed = time.time() - start
                rate = checked / elapsed
                remaining = (total - checked) / rate
                eta = f"{remaining / 60:.1f}m" if remaining > 60 else f"{remaining:.0f}s"
                print(f"  {checked}/{total} ({checked*100//total}%) — {rate:.1f}/sec — ETA {eta} — {rejected} rejected")

    # Flush remaining
    if batch:
        db.executemany(
            "INSERT OR IGNORE INTO font_removals (font_id, reason_id, details) VALUES (?, 8, ?)",
            batch
        )
        db.commit()
        rejected += len(batch)

    elapsed = time.time() - start
    print(f"\nDone! Checked {checked} fonts, rejected {rejected} in {elapsed:.1f}s")
    db.close()


if __name__ == '__main__':
    main()
