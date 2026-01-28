#!/usr/bin/env python3
"""
Filter fonts by checking if rendered text has connected letters.

"Hello World!" should have ~11 separate components (10 letters + !)
Cursive fonts will have far fewer (connected letters form single components)

Usage:
    python3 run_connectivity_filter.py          # Preview only
    python3 run_connectivity_filter.py --update # Update database
"""

import argparse
import sqlite3
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import ndimage
from pathlib import Path

DB_PATH = 'fonts.db'
SAMPLE_TEXT = "Hello World!"
FONT_SIZE = 48

# Expected components: H,e,l,l,o, W,o,r,l,d,! = 11 base
# Some letters have 2 parts (i,j,!,?) so allow some variance
# Threshold: if components < 6, it's likely cursive (letters connected)
MIN_COMPONENTS = 6


def count_components(font_path: str) -> tuple:
    """Render text and count connected components."""
    try:
        font = ImageFont.truetype(font_path, FONT_SIZE)
    except Exception as e:
        return None, str(e)

    # Create image
    temp_img = Image.new('L', (1, 1), 255)
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), SAMPLE_TEXT, font=font)
    width = bbox[2] - bbox[0] + 40
    height = bbox[3] - bbox[1] + 40

    img = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(img)
    draw.text((20 - bbox[0], 20 - bbox[1]), SAMPLE_TEXT, font=font, fill=0)

    # Convert to binary array (1 = ink, 0 = background)
    arr = np.array(img)
    binary = (arr < 128).astype(np.uint8)

    # Count connected components
    labeled, num_components = ndimage.label(binary)

    return num_components, None


def main(update_db=False):
    print("=" * 60)
    print("CONNECTIVITY FILTER")
    print("=" * 60)
    print(f"Sample text: '{SAMPLE_TEXT}'")
    print(f"Minimum components: {MIN_COMPONENTS}")
    print()

    # Get fonts that passed OCR prefilter
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT f.id, f.name, f.file_path, fc.prefilter_confidence
        FROM fonts f
        JOIN font_checks fc ON f.id = fc.font_id
        WHERE fc.prefilter_passed = 1
        ORDER BY f.name
    """)

    fonts = [dict(row) for row in cursor.fetchall()]
    print(f"Checking {len(fonts)} fonts...")
    print()

    # Check each font
    results = []
    for font in fonts:
        components, error = count_components(font['file_path'])
        if error:
            results.append({**font, 'components': None, 'error': error, 'connected': None})
        else:
            connected = components < MIN_COMPONENTS
            results.append({**font, 'components': components, 'error': None, 'connected': connected})

    # Sort by components
    valid_results = [r for r in results if r['components'] is not None]
    valid_results.sort(key=lambda x: x['components'])

    # Show most connected (fewest components = most cursive)
    print("MOST CONNECTED (likely cursive - fewest components):")
    print("-" * 60)
    for r in valid_results[:15]:
        status = "FAIL" if r['connected'] else "PASS"
        print(f"  {r['components']:2d} components [{status}] {r['name'][:40]}")

    print()
    print("LEAST CONNECTED (clearly print - most components):")
    print("-" * 60)
    for r in valid_results[-10:]:
        status = "FAIL" if r['connected'] else "PASS"
        print(f"  {r['components']:2d} components [{status}] {r['name'][:40]}")

    # Count pass/fail
    connected_fonts = [r for r in valid_results if r['connected']]
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total fonts checked: {len(fonts)}")
    print(f"  Connected (FAIL, < {MIN_COMPONENTS} components): {len(connected_fonts)}")
    print(f"  Print (PASS): {len(valid_results) - len(connected_fonts)}")

    # Update database if requested
    if connected_fonts:
        print()
        print("Fonts to be marked as connected/cursive:")
        for r in connected_fonts:
            print(f"  - {r['name']} ({r['components']} components)")

        if update_db:
            print()
            for r in connected_fonts:
                # Update prefilter_passed to False
                cursor.execute("""
                    UPDATE font_checks
                    SET prefilter_passed = 0
                    WHERE font_id = ?
                """, (r['id'],))

                # Add removal reason
                cursor.execute("SELECT id FROM removal_reasons WHERE code = 'cursive'")
                reason_id = cursor.fetchone()[0]
                cursor.execute("""
                    INSERT INTO font_removals (font_id, reason_id, details)
                    VALUES (?, ?, ?)
                """, (r['id'], reason_id, f"connected text: {r['components']} components"))

            conn.commit()
            print(f"Updated {len(connected_fonts)} fonts as failed.")
        else:
            print()
            print("Run with --update to mark these as failed.")

    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter fonts by text connectivity')
    parser.add_argument('--update', action='store_true', help='Update database')
    parser.add_argument('--threshold', type=int, default=MIN_COMPONENTS,
                        help=f'Min components threshold (default: {MIN_COMPONENTS})')
    args = parser.parse_args()

    MIN_COMPONENTS = args.threshold
    main(update_db=args.update)
