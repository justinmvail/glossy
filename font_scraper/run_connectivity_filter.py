#!/usr/bin/env python3
"""Filter fonts by checking if rendered text has connected letters.

This script identifies cursive or connected fonts by rendering sample text and
counting the number of connected components in the resulting image. Print fonts
will have many separate components (one per letter), while cursive fonts will
have fewer components because letters connect together.

The script uses "Hello World!" as sample text, which should produce approximately
11 separate components for print fonts (10 letters + exclamation mark). Fonts
with fewer than a threshold number of components (default: 6) are flagged as
cursive/connected and can optionally be marked as failed in the database.

Example:
    Preview mode (no database changes)::

        $ python3 run_connectivity_filter.py

    Update database with results::

        $ python3 run_connectivity_filter.py --update

    Custom threshold::

        $ python3 run_connectivity_filter.py --update --threshold 5

Attributes:
    DB_PATH (str): Default path to the SQLite fonts database.
    SAMPLE_TEXT (str): Text rendered for connectivity analysis.
    FONT_SIZE (int): Font size in points for rendering.
    MIN_COMPONENTS (int): Minimum number of components for a font to pass.
        Fonts with fewer components are considered cursive.
"""

import argparse
import sqlite3

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

DB_PATH = 'fonts.db'
SAMPLE_TEXT = "Hello World!"
FONT_SIZE = 48

# Expected components: H,e,l,l,o, W,o,r,l,d,! = 11 base
# Some letters have 2 parts (i,j,!,?) so allow some variance
# Threshold: if components < 6, it's likely cursive (letters connected)
MIN_COMPONENTS = 6


def count_components(font_path: str) -> tuple:
    """Render text with a font and count connected components.

    Renders the sample text using the specified font file and performs
    connected component analysis to count the number of separate ink
    regions. Print fonts will have many components (one per letter),
    while cursive fonts will have fewer (connected letters merge).

    Args:
        font_path: Absolute path to the font file (.ttf, .otf, etc.).

    Returns:
        A tuple of (num_components, error_message) where:
            - num_components (int or None): Number of connected components
              in the rendered text, or None if an error occurred.
            - error_message (str or None): Error description if the font
              could not be loaded, or None on success.

    Example:
        >>> components, error = count_components('/path/to/font.ttf')
        >>> if error:
        ...     print(f"Failed: {error}")
        ... else:
        ...     print(f"Found {components} components")
    """
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


def main(update_db: bool = False) -> None:
    """Run the connectivity filter on all fonts that passed OCR prefilter.

    Queries the database for fonts that passed the prefilter stage, renders
    sample text with each font, counts connected components, and identifies
    fonts that appear to be cursive based on low component counts.

    Results are displayed showing the most and least connected fonts, along
    with pass/fail status. If update_db is True, fonts identified as
    cursive are marked as failed in the database with an appropriate
    removal reason.

    Args:
        update_db: If True, update the database to mark cursive fonts as
            failed. If False (default), only preview results without
            making any database changes.

    Returns:
        None. Results are printed to stdout and optionally written to
        the database.
    """
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
