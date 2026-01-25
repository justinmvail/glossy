"""
Parse SVG single-line fonts and extract stroke data for SDT training.
"""

import re
import os
import numpy as np
from pathlib import Path

def parse_svg_font(svg_path):
    """Parse an SVG font file and extract glyph stroke data."""
    with open(svg_path, 'r') as f:
        content = f.read()

    # Extract font metadata
    font_id_match = re.search(r'<font id="([^"]+)"', content)
    font_id = font_id_match.group(1) if font_id_match else Path(svg_path).stem

    # Extract all glyphs - handle different attribute orders
    # Pattern matches glyph tags with both unicode and d attributes
    glyph_pattern = r'<glyph[^>]*unicode="([^"]*)"[^>]*d="([^"]*)"[^>]*/>'
    # Also try alternate order: d before unicode
    glyph_pattern2 = r'<glyph[^>]*d="([^"]*)"[^>]*unicode="([^"]*)"[^>]*/>'
    glyphs = {}

    for match in re.finditer(glyph_pattern, content, re.DOTALL):
        char = match.group(1)
        path_data = match.group(2)
        if char and path_data.strip():
            # Decode HTML entities
            decoded_char = decode_html_entities(char)
            strokes = parse_path_to_strokes(path_data)
            if strokes is not None and len(strokes) > 0:
                glyphs[decoded_char] = strokes

    # Try alternate pattern for fonts with different attribute order
    for match in re.finditer(glyph_pattern2, content, re.DOTALL):
        path_data = match.group(1)
        char = match.group(2)
        if char and path_data.strip() and char not in glyphs:
            decoded_char = decode_html_entities(char)
            strokes = parse_path_to_strokes(path_data)
            if strokes is not None and len(strokes) > 0:
                glyphs[decoded_char] = strokes

    # Also extract horiz-adv-x for spacing
    adv_pattern = r'<glyph[^>]*unicode="([^"]*)"[^>]*horiz-adv-x="([^"]*)"'
    advances = {}
    for match in re.finditer(adv_pattern, content):
        char = match.group(1)
        char = decode_html_entities(char)
        try:
            advances[char] = float(match.group(2))
        except:
            pass

    return font_id, glyphs, advances


def decode_html_entities(char):
    """Decode common HTML entities in glyph unicode values."""
    char = char.replace('&#x22;', '"')
    char = char.replace('&#x27;', "'")
    char = char.replace('&amp;', '&')
    char = char.replace('&apos;', "'")
    char = char.replace('&quot;', '"')
    char = char.replace('&#x3c;', '<')
    char = char.replace('&#x3e;', '>')
    char = char.replace('&lt;', '<')
    char = char.replace('&gt;', '>')
    # Handle hex unicode escapes like &#xc6;
    hex_pattern = r'&#x([0-9a-fA-F]+);'
    for m in re.finditer(hex_pattern, char):
        try:
            char = char.replace(m.group(0), chr(int(m.group(1), 16)))
        except:
            pass
    return char


def bezier_point(t, p0, p1, p2, p3=None):
    """Calculate point on bezier curve at parameter t."""
    if p3 is None:  # Quadratic
        return (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
    else:  # Cubic
        return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3


def sample_bezier(x0, y0, x1, y1, x2, y2, x3=None, y3=None, num_samples=8):
    """Sample points along a bezier curve."""
    points = []
    p0 = np.array([x0, y0])
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    p3 = np.array([x3, y3]) if x3 is not None else None

    for i in range(1, num_samples + 1):
        t = i / num_samples
        pt = bezier_point(t, p0, p1, p2, p3)
        points.append(pt)

    return points


def parse_path_to_strokes(path_data):
    """
    Parse SVG path data into stroke coordinates.
    Returns array of shape (N, 3): [x, y, pen_up]
    pen_up = 1 means pen lifted before this point

    Supports: M, m, L, l, H, h, V, v, C, c, S, s, Q, q, T, t, Z, z
    """
    # Tokenize path data - handle negative numbers properly
    tokens = re.findall(r'[MLmlHhVvCcSsQqTtAaZz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', path_data)

    points = []
    current_x, current_y = 0, 0
    start_x, start_y = 0, 0  # For Z command
    pen_up = 1  # Start with pen up
    last_control = None  # For smooth curves
    last_cmd = None

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Check if it's a command or a number
        if token in 'MLmlHhVvCcSsQqTtAaZz':
            cmd = token
            i += 1
        else:
            # Implicit command - repeat last command
            if last_cmd in ['M', 'm']:
                cmd = 'L' if last_cmd == 'M' else 'l'
            else:
                cmd = last_cmd
            if cmd is None:
                i += 1
                continue

        if cmd == 'M':  # Move to (absolute)
            current_x = float(tokens[i])
            current_y = float(tokens[i + 1])
            start_x, start_y = current_x, current_y
            points.append([current_x, current_y, 1])  # pen up
            pen_up = 0
            i += 2
            last_control = None

        elif cmd == 'm':  # Move to (relative)
            current_x += float(tokens[i])
            current_y += float(tokens[i + 1])
            start_x, start_y = current_x, current_y
            points.append([current_x, current_y, 1])  # pen up
            pen_up = 0
            i += 2
            last_control = None

        elif cmd == 'L':  # Line to (absolute)
            current_x = float(tokens[i])
            current_y = float(tokens[i + 1])
            points.append([current_x, current_y, pen_up])
            pen_up = 0
            i += 2
            last_control = None

        elif cmd == 'l':  # Line to (relative)
            current_x += float(tokens[i])
            current_y += float(tokens[i + 1])
            points.append([current_x, current_y, pen_up])
            pen_up = 0
            i += 2
            last_control = None

        elif cmd == 'H':  # Horizontal line (absolute)
            current_x = float(tokens[i])
            points.append([current_x, current_y, pen_up])
            pen_up = 0
            i += 1
            last_control = None

        elif cmd == 'h':  # Horizontal line (relative)
            current_x += float(tokens[i])
            points.append([current_x, current_y, pen_up])
            pen_up = 0
            i += 1
            last_control = None

        elif cmd == 'V':  # Vertical line (absolute)
            current_y = float(tokens[i])
            points.append([current_x, current_y, pen_up])
            pen_up = 0
            i += 1
            last_control = None

        elif cmd == 'v':  # Vertical line (relative)
            current_y += float(tokens[i])
            points.append([current_x, current_y, pen_up])
            pen_up = 0
            i += 1
            last_control = None

        elif cmd == 'C':  # Cubic bezier (absolute)
            x1, y1 = float(tokens[i]), float(tokens[i + 1])
            x2, y2 = float(tokens[i + 2]), float(tokens[i + 3])
            x3, y3 = float(tokens[i + 4]), float(tokens[i + 5])
            for pt in sample_bezier(current_x, current_y, x1, y1, x2, y2, x3, y3):
                points.append([pt[0], pt[1], pen_up])
                pen_up = 0
            current_x, current_y = x3, y3
            last_control = (x2, y2)
            i += 6

        elif cmd == 'c':  # Cubic bezier (relative)
            x1, y1 = current_x + float(tokens[i]), current_y + float(tokens[i + 1])
            x2, y2 = current_x + float(tokens[i + 2]), current_y + float(tokens[i + 3])
            x3, y3 = current_x + float(tokens[i + 4]), current_y + float(tokens[i + 5])
            for pt in sample_bezier(current_x, current_y, x1, y1, x2, y2, x3, y3):
                points.append([pt[0], pt[1], pen_up])
                pen_up = 0
            current_x, current_y = x3, y3
            last_control = (x2, y2)
            i += 6

        elif cmd == 'S':  # Smooth cubic bezier (absolute)
            # First control point is reflection of last control
            if last_control and last_cmd in ['C', 'c', 'S', 's']:
                x1 = 2 * current_x - last_control[0]
                y1 = 2 * current_y - last_control[1]
            else:
                x1, y1 = current_x, current_y
            x2, y2 = float(tokens[i]), float(tokens[i + 1])
            x3, y3 = float(tokens[i + 2]), float(tokens[i + 3])
            for pt in sample_bezier(current_x, current_y, x1, y1, x2, y2, x3, y3):
                points.append([pt[0], pt[1], pen_up])
                pen_up = 0
            current_x, current_y = x3, y3
            last_control = (x2, y2)
            i += 4

        elif cmd == 's':  # Smooth cubic bezier (relative)
            if last_control and last_cmd in ['C', 'c', 'S', 's']:
                x1 = 2 * current_x - last_control[0]
                y1 = 2 * current_y - last_control[1]
            else:
                x1, y1 = current_x, current_y
            x2 = current_x + float(tokens[i])
            y2 = current_y + float(tokens[i + 1])
            x3 = current_x + float(tokens[i + 2])
            y3 = current_y + float(tokens[i + 3])
            for pt in sample_bezier(current_x, current_y, x1, y1, x2, y2, x3, y3):
                points.append([pt[0], pt[1], pen_up])
                pen_up = 0
            current_x, current_y = x3, y3
            last_control = (x2, y2)
            i += 4

        elif cmd == 'Q':  # Quadratic bezier (absolute)
            x1, y1 = float(tokens[i]), float(tokens[i + 1])
            x2, y2 = float(tokens[i + 2]), float(tokens[i + 3])
            for pt in sample_bezier(current_x, current_y, x1, y1, x2, y2):
                points.append([pt[0], pt[1], pen_up])
                pen_up = 0
            current_x, current_y = x2, y2
            last_control = (x1, y1)
            i += 4

        elif cmd == 'q':  # Quadratic bezier (relative)
            x1 = current_x + float(tokens[i])
            y1 = current_y + float(tokens[i + 1])
            x2 = current_x + float(tokens[i + 2])
            y2 = current_y + float(tokens[i + 3])
            for pt in sample_bezier(current_x, current_y, x1, y1, x2, y2):
                points.append([pt[0], pt[1], pen_up])
                pen_up = 0
            current_x, current_y = x2, y2
            last_control = (x1, y1)
            i += 4

        elif cmd == 'T':  # Smooth quadratic bezier (absolute)
            if last_control and last_cmd in ['Q', 'q', 'T', 't']:
                x1 = 2 * current_x - last_control[0]
                y1 = 2 * current_y - last_control[1]
            else:
                x1, y1 = current_x, current_y
            x2, y2 = float(tokens[i]), float(tokens[i + 1])
            for pt in sample_bezier(current_x, current_y, x1, y1, x2, y2):
                points.append([pt[0], pt[1], pen_up])
                pen_up = 0
            current_x, current_y = x2, y2
            last_control = (x1, y1)
            i += 2

        elif cmd == 't':  # Smooth quadratic bezier (relative)
            if last_control and last_cmd in ['Q', 'q', 'T', 't']:
                x1 = 2 * current_x - last_control[0]
                y1 = 2 * current_y - last_control[1]
            else:
                x1, y1 = current_x, current_y
            x2 = current_x + float(tokens[i])
            y2 = current_y + float(tokens[i + 1])
            for pt in sample_bezier(current_x, current_y, x1, y1, x2, y2):
                points.append([pt[0], pt[1], pen_up])
                pen_up = 0
            current_x, current_y = x2, y2
            last_control = (x1, y1)
            i += 2

        elif cmd in ['Z', 'z']:  # Close path
            if (current_x, current_y) != (start_x, start_y):
                points.append([start_x, start_y, pen_up])
            current_x, current_y = start_x, start_y
            pen_up = 1
            last_control = None

        elif cmd in ['A', 'a']:  # Arc - approximate with line for simplicity
            # Arc has 7 params: rx ry x-axis-rotation large-arc sweep x y
            if cmd == 'A':
                current_x = float(tokens[i + 5])
                current_y = float(tokens[i + 6])
            else:
                current_x += float(tokens[i + 5])
                current_y += float(tokens[i + 6])
            points.append([current_x, current_y, pen_up])
            pen_up = 0
            i += 7
            last_control = None

        else:
            # Unknown command, skip
            i += 1
            continue

        last_cmd = cmd

    if not points:
        return None

    return np.array(points, dtype=np.float32)


def strokes_to_relative(strokes):
    """
    Convert absolute coordinates to relative (dx, dy, pen_up).
    This is the format SDT expects.
    """
    if strokes is None or len(strokes) < 2:
        return None

    relative = np.zeros((len(strokes), 3), dtype=np.float32)
    relative[0] = [0, 0, strokes[0, 2]]  # First point: no movement

    for i in range(1, len(strokes)):
        dx = strokes[i, 0] - strokes[i-1, 0]
        dy = strokes[i, 1] - strokes[i-1, 1]
        pen_up = strokes[i, 2]
        relative[i] = [dx, dy, pen_up]

    return relative


def generate_word_strokes(word, glyphs, advances, spacing=50):
    """
    Generate stroke data for a word by concatenating glyph strokes.
    """
    all_strokes = []
    x_offset = 0

    for char in word:
        if char not in glyphs:
            # Skip unknown characters, add space
            x_offset += spacing
            continue

        char_strokes = glyphs[char].copy()

        # Offset x coordinates
        char_strokes[:, 0] += x_offset

        # Mark first point as pen-up (moving to new character)
        if len(all_strokes) > 0:
            char_strokes[0, 2] = 1

        all_strokes.append(char_strokes)

        # Advance x position
        adv = advances.get(char, 400)  # Default advance
        x_offset += adv

    if not all_strokes:
        return None

    combined = np.concatenate(all_strokes, axis=0)
    return combined


def normalize_strokes(strokes, target_height=64):
    """Normalize strokes to fit in target height while preserving aspect ratio."""
    if strokes is None or len(strokes) == 0:
        return None

    # Get bounding box
    min_x, min_y = strokes[:, 0].min(), strokes[:, 1].min()
    max_x, max_y = strokes[:, 0].max(), strokes[:, 1].max()

    height = max_y - min_y
    if height == 0:
        height = 1

    scale = target_height / height

    # Normalize
    normalized = strokes.copy()
    normalized[:, 0] = (strokes[:, 0] - min_x) * scale
    normalized[:, 1] = (strokes[:, 1] - min_y) * scale

    return normalized


if __name__ == "__main__":
    # Test with one font
    font_path = "/home/server/svg-fonts/fonts/EMS/EMSCasualHand.svg"

    font_id, glyphs, advances = parse_svg_font(font_path)
    print(f"Font: {font_id}")
    print(f"Glyphs: {len(glyphs)}")
    print(f"Sample chars: {list(glyphs.keys())[:20]}")

    # Generate a word
    word = "hello"
    word_strokes = generate_word_strokes(word, glyphs, advances)

    if word_strokes is not None:
        print(f"\nWord '{word}':")
        print(f"  Points: {len(word_strokes)}")
        print(f"  Bounds: x=[{word_strokes[:,0].min():.1f}, {word_strokes[:,0].max():.1f}], y=[{word_strokes[:,1].min():.1f}, {word_strokes[:,1].max():.1f}]")

        # Normalize and convert to relative
        norm_strokes = normalize_strokes(word_strokes)
        rel_strokes = strokes_to_relative(norm_strokes)
        print(f"  Relative coords shape: {rel_strokes.shape}")
        print(f"  First 5 points:\n{rel_strokes[:5]}")
