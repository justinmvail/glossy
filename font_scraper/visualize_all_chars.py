#!/usr/bin/env python3
"""Visualize all stroke data for characters from a single font.

This module creates a grid visualization of all characters from the first font
in the database that has stroke data. Each character is rendered in a cell
showing its stroke paths with color-coded strokes for easy visual inspection.

The visualization is useful for:
    - Quality checking extracted stroke data
    - Verifying stroke ordering and connectivity
    - Debugging stroke extraction algorithms
    - Visual comparison of different fonts

Output:
    Saves a PNG image grid (all_chars_preview.png) and opens it with the
    system default image viewer via xdg-open.

Example:
    Run as a script to generate the visualization::

        $ python visualize_all_chars.py

Note:
    Requires a populated fonts.db database with stroke data in the
    characters.strokes_raw column.
"""

import sqlite3
import json
from PIL import Image, ImageDraw, ImageFont

# Database connection with Row factory for dict-like access
conn = sqlite3.connect('fonts.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get first font with data
cursor.execute("""
    SELECT DISTINCT f.id, f.name
    FROM fonts f
    JOIN characters c ON f.id = c.font_id
    WHERE c.strokes_raw IS NOT NULL
    ORDER BY f.id
    LIMIT 1
""")
font = cursor.fetchone()
font_id = font['id']
font_name = font['name']

# Get all characters for this font
cursor.execute("""
    SELECT char, strokes_raw, point_count
    FROM characters
    WHERE font_id = ? AND strokes_raw IS NOT NULL
    ORDER BY char
""", (font_id,))

chars = cursor.fetchall()
print(f"Font: {font_name}")
print(f"Characters: {len(chars)}")

# Create image grid
cell_size = 80
cols = 10
rows_count = (len(chars) + cols - 1) // cols
img = Image.new('RGB', (cols * cell_size, rows_count * cell_size + 40), 'white')
draw = ImageDraw.Draw(img)

# Title
try:
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
except:
    title_font = ImageFont.load_default()
    label_font = ImageFont.load_default()

draw.text((10, 10), f"{font_name} - All Characters", fill='black', font=title_font)

for i, row in enumerate(chars):
    col = i % cols
    r = i // cols

    x_off = col * cell_size
    y_off = r * cell_size + 40

    # Draw border
    draw.rectangle([x_off, y_off, x_off + cell_size - 1, y_off + cell_size - 1], outline='#eee')

    # Draw label
    char_display = row['char'] if row['char'] != ' ' else 'SPC'
    draw.text((x_off + 3, y_off + 2), char_display, fill='#999', font=label_font)

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
    padding = 15
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
            # Draw point
            if len(stroke) == 1:
                p = stroke[0]
                px = offset_x + (p[0] - min_x) * scale
                py = offset_y + (p[1] - min_y) * scale
                draw.ellipse([px-2, py-2, px+2, py+2], fill=colors[si % len(colors)])
            continue

        color = colors[si % len(colors)]
        points = []
        for p in stroke:
            px = offset_x + (p[0] - min_x) * scale
            py = offset_y + (p[1] - min_y) * scale
            points.append((px, py))

        draw.line(points, fill=color, width=2)

# Save
output = 'all_chars_preview.png'
img.save(output)
print(f"Saved to {output}")

conn.close()

# Open it
import subprocess
subprocess.run(['xdg-open', output])
