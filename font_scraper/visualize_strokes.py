#!/usr/bin/env python3
"""Visualize stroke data from the font database.

This module creates a grid visualization of selected characters showing their
stroke paths. It displays a 2x5 grid of common letters (A, B, C, a, b, c, H, e, l, o)
from the first font in the database, with each stroke rendered in a different color.

The visualization is useful for:
    - Quick visual inspection of stroke extraction quality
    - Verifying stroke segmentation and ordering
    - Debugging stroke path generation
    - Comparing stroke styles across different characters

Output:
    Saves a PNG image grid (stroke_preview.png) with 10 character cells
    and opens it with the system default image viewer via xdg-open.

Example:
    Run as a script to generate the preview::

        $ python visualize_strokes.py

Note:
    Requires a populated fonts.db database with stroke data in the
    characters.strokes_raw column. Characters without sufficient points
    (fewer than 2) are skipped during rendering.
"""

import sqlite3
import json
from PIL import Image, ImageDraw

# Database connection with Row factory for dict-like access
conn = sqlite3.connect('fonts.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get a few different letters from first font
cursor.execute("""
    SELECT f.name, c.char, c.strokes_raw
    FROM characters c
    JOIN fonts f ON c.font_id = f.id
    WHERE c.strokes_raw IS NOT NULL
      AND c.char IN ('A', 'B', 'C', 'a', 'b', 'c', 'H', 'e', 'l', 'o')
    ORDER BY f.id, c.char
    LIMIT 10
""")

rows = cursor.fetchall()
font_name = rows[0]['name']

# Create image grid
cell_size = 150
cols = 5
rows_count = 2
img = Image.new('RGB', (cols * cell_size, rows_count * cell_size), 'white')
draw = ImageDraw.Draw(img)

for i, row in enumerate(rows):
    col = i % cols
    r = i // cols

    x_off = col * cell_size
    y_off = r * cell_size

    # Draw border
    draw.rectangle([x_off, y_off, x_off + cell_size - 1, y_off + cell_size - 1], outline='lightgray')

    # Draw label
    draw.text((x_off + 5, y_off + 5), f"'{row['char']}'", fill='gray')

    # Parse strokes
    strokes = json.loads(row['strokes_raw'])

    if not strokes:
        continue

    # Find bounds
    all_points = []
    for stroke in strokes:
        all_points.extend(stroke)

    if not all_points:
        continue

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Scale to fit cell (with padding)
    padding = 25
    available = cell_size - 2 * padding

    width = max_x - min_x if max_x > min_x else 1
    height = max_y - min_y if max_y > min_y else 1
    scale = min(available / width, available / height)

    # Center in cell
    scaled_w = width * scale
    scaled_h = height * scale
    offset_x = x_off + padding + (available - scaled_w) / 2
    offset_y = y_off + padding + (available - scaled_h) / 2

    # Draw strokes
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

    for si, stroke in enumerate(strokes):
        if len(stroke) < 2:
            continue

        color = colors[si % len(colors)]
        points = []
        for p in stroke:
            px = offset_x + (p[0] - min_x) * scale
            py = offset_y + (p[1] - min_y) * scale
            points.append((px, py))

        draw.line(points, fill=color, width=2)

# Save
output = 'stroke_preview.png'
img.save(output)
print(f"Saved to {output}")
print(f"Font: {font_name}")

conn.close()

# Open it
import subprocess
subprocess.run(['xdg-open', output])
