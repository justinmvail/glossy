#!/usr/bin/env python3
"""Render sample text with all fonts that passed the OCR prefilter.

This module creates a visual catalog of all fonts in the database that have
passed OCR quality checks. Each font is rendered with sample text "Hello World!"
along with its name and OCR confidence score, organized in a multi-column layout.

The visualization is useful for:
    - Reviewing the pool of fonts available for training
    - Quick visual quality assessment of passing fonts
    - Identifying fonts that may need manual review
    - Comparing rendering quality across different fonts

Output:
    Saves a PNG image (all_passing_fonts.png) with all passing fonts
    rendered in a two-column layout, showing font name, confidence score,
    and sample text. Opens the result with xdg-open.

Example:
    Run as a script to generate the catalog::

        $ python render_all_passing.py

Note:
    Requires a populated fonts.db database with font check results in the
    font_checks table. Fonts with prefilter_passed=1 will be included.
"""

import sqlite3

from PIL import Image, ImageDraw, ImageFont

from filter_config import DB_PATH, SAMPLE_TEXT

# Configuration constants - inherit from shared config, override as needed
OUTPUT_PATH = 'all_passing_fonts.png'
"""str: Path for the output visualization image."""

FONT_SIZE = 36  # Smaller than filter_config.FONT_SIZE for compact display
"""int: Point size for rendering sample text."""

ROW_HEIGHT = 60
"""int: Pixel height allocated for each font row."""

LABEL_WIDTH = 300
"""int: Pixel width reserved for font name and confidence."""

SAMPLE_WIDTH = 400
"""int: Pixel width reserved for sample text rendering."""

COLS = 2
"""int: Number of columns in the output layout."""

# Get passing fonts
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
conn.close()

print(f"Found {len(fonts)} passing fonts")

# Calculate image size
col_width = LABEL_WIDTH + SAMPLE_WIDTH
rows_per_col = (len(fonts) + COLS - 1) // COLS
img_width = col_width * COLS + 20
img_height = rows_per_col * ROW_HEIGHT + 40

print(f"Creating image: {img_width}x{img_height}")

# Create image
img = Image.new('RGB', (img_width, img_height), 'white')
draw = ImageDraw.Draw(img)

# Load a default font for labels
try:
    label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except OSError:
    label_font = ImageFont.load_default()

# Render each font
for i, font in enumerate(fonts):
    col = i // rows_per_col
    row = i % rows_per_col

    x_offset = col * col_width + 10
    y = row * ROW_HEIGHT + 20

    # Draw label (font name + confidence)
    conf = font['prefilter_confidence']
    label = f"{font['name'][:35]}"
    draw.text((x_offset, y + 5), label, font=label_font, fill='gray')

    # Draw confidence
    conf_text = f"{conf:.0%}"
    draw.text((x_offset, y + 22), conf_text, font=label_font, fill='green' if conf >= 0.9 else 'orange')

    # Render sample with the font
    try:
        sample_font = ImageFont.truetype(font['file_path'], FONT_SIZE)
        draw.text((x_offset + LABEL_WIDTH, y + 10), SAMPLE_TEXT, font=sample_font, fill='black')
    except Exception as e:
        draw.text((x_offset + LABEL_WIDTH, y + 15), f"[Error: {str(e)[:30]}]", font=label_font, fill='red')

# Draw column separator
if COLS > 1:
    for c in range(1, COLS):
        x = c * col_width + 5
        draw.line([(x, 10), (x, img_height - 10)], fill='lightgray', width=1)

# Save
img.save(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")

# Open it
import subprocess  # noqa: E402

subprocess.run(['xdg-open', OUTPUT_PATH])
