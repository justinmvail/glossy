"""Glyph rendering utilities.

This module contains functions for rendering font glyphs to images and masks.
"""

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stroke_flask import DEFAULT_CANVAS_SIZE, DEFAULT_FONT_SIZE, resolve_font_path


def render_char_image(font_path, char, font_size=DEFAULT_FONT_SIZE, canvas_size=DEFAULT_CANVAS_SIZE):
    """Render a character to a grayscale PNG.

    Returns image bytes or None on failure.
    """
    font_path = resolve_font_path(font_path)
    try:
        pil_font = ImageFont.truetype(font_path, font_size)
    except OSError:
        return None

    # Get bbox and scale if needed
    bbox = pil_font.getbbox(char)
    if bbox:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w > canvas_size * 0.9 or h > canvas_size * 0.9:
            scale = min(canvas_size * 0.9 / w, canvas_size * 0.9 / h)
            font_size = int(font_size * scale)
            try:
                pil_font = ImageFont.truetype(font_path, font_size)
            except OSError:
                return None
            bbox = pil_font.getbbox(char)
            if bbox:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]

    img = Image.new('L', (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)

    if bbox:
        x = (canvas_size - w) // 2 - bbox[0]
        y = (canvas_size - h) // 2 - bbox[1]
    else:
        x, y = 0, 0

    draw.text((x, y), char, fill=0, font=pil_font)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def render_glyph_mask(font_path, char, canvas_size=DEFAULT_CANVAS_SIZE):
    """Render a glyph as a binary mask.

    Returns numpy boolean array where True = glyph pixel.
    """
    font_path = resolve_font_path(font_path)
    try:
        pil_font = ImageFont.truetype(font_path, 200)
    except OSError:
        return None

    bbox = pil_font.getbbox(char)
    font_size = 200
    if bbox:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w > canvas_size * 0.9 or h > canvas_size * 0.9:
            scale = min(canvas_size * 0.9 / w, canvas_size * 0.9 / h)
            font_size = int(font_size * scale)
            try:
                pil_font = ImageFont.truetype(font_path, font_size)
            except OSError:
                return None
            bbox = pil_font.getbbox(char)
            if bbox:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]

    img = Image.new('L', (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)

    if bbox:
        x = (canvas_size - w) // 2 - bbox[0]
        y = (canvas_size - h) // 2 - bbox[1]
    else:
        x, y = 0, 0

    draw.text((x, y), char, fill=0, font=pil_font)
    return np.array(img) < 128


def check_case_mismatch(font_path, threshold=0.80):
    """Check if lowercase letters match their uppercase counterparts.

    Returns a list of lowercase letters that appear identical to uppercase.
    Normalizes glyphs to same size before comparing to catch small-caps fonts.
    """
    font_path = resolve_font_path(font_path)
    letters_to_check = 'abcdefghijklmnopqrstuvwxyz'
    mismatched = []

    try:
        font_size = 100
        pil_font = ImageFont.truetype(font_path, font_size)
    except OSError:
        return []

    norm_size = 64

    for lower in letters_to_check:
        upper = lower.upper()

        try:
            l_bbox = pil_font.getbbox(lower)
            u_bbox = pil_font.getbbox(upper)

            if not l_bbox or not u_bbox:
                continue

            l_w = l_bbox[2] - l_bbox[0]
            l_h = l_bbox[3] - l_bbox[1]
            u_w = u_bbox[2] - u_bbox[0]
            u_h = u_bbox[3] - u_bbox[1]

            if l_w < 5 or l_h < 5 or u_w < 5 or u_h < 5:
                continue

            # Render lowercase at its natural size
            l_img = Image.new('L', (l_w + 10, l_h + 10), 255)
            l_draw = ImageDraw.Draw(l_img)
            l_draw.text((5 - l_bbox[0], 5 - l_bbox[1]), lower, fill=0, font=pil_font)

            # Render uppercase at its natural size
            u_img = Image.new('L', (u_w + 10, u_h + 10), 255)
            u_draw = ImageDraw.Draw(u_img)
            u_draw.text((5 - u_bbox[0], 5 - u_bbox[1]), upper, fill=0, font=pil_font)

            # Scale both to same normalized size
            l_scaled = l_img.resize((norm_size, norm_size), Image.Resampling.BILINEAR)
            u_scaled = u_img.resize((norm_size, norm_size), Image.Resampling.BILINEAR)

            l_arr = np.array(l_scaled) < 128
            u_arr = np.array(u_scaled) < 128

            # Compare using IoU
            intersection = np.sum(l_arr & u_arr)
            union = np.sum(l_arr | u_arr)

            if union > 0:
                iou = intersection / union
                if iou >= threshold:
                    mismatched.append(lower)

        except (ValueError, MemoryError, OSError):
            continue

    return mismatched


def render_text_for_analysis(pil_font, text):
    """Render text and return as numpy array for shape analysis."""
    canvas = 400
    img = Image.new('L', (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)

    try:
        bbox = pil_font.getbbox(text)
        if not bbox:
            return None
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (canvas - w) // 2 - bbox[0]
        y = (canvas - h) // 2 - bbox[1]
        draw.text((x, y), text, fill=0, font=pil_font)
    except Exception:
        return None

    return np.array(img)


def analyze_shape_metrics(arr, width):
    """Analyze shape metrics from rendered text array.

    Returns (shape_count, max_width_pct) tuple.
    """
    from scipy.ndimage import label

    binary = arr < 128
    _labeled, num_shapes = label(binary)

    # Find max contiguous width percentage
    max_width = 0
    rows, cols = np.where(binary)
    if len(rows) > 0:
        for r in np.unique(rows):
            row_cols = cols[rows == r]
            if len(row_cols) > 0:
                span = row_cols.max() - row_cols.min() + 1
                max_width = max(max_width, span)

    return num_shapes, max_width / width if width > 0 else 0


def check_char_holes(pil_font, char):
    """Check if a character has holes (inner contours).

    Returns True if character has holes.
    """
    from scipy.ndimage import label

    canvas = 150
    img = Image.new('L', (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)

    try:
        bbox = pil_font.getbbox(char)
        if not bbox:
            return False
        draw.text((-bbox[0] + 5, -bbox[1] + 5), char, fill=0, font=pil_font)
    except Exception:
        return False

    arr = np.array(img)
    binary = arr < 128

    # Label background components
    _bg_labeled, bg_count = label(~binary)

    # If more than 1 background region, there are holes
    return bg_count > 1


def check_char_shape_count(pil_font, char, expected):
    """Check if character has expected number of connected components.

    Returns True if shape count matches expected.
    """
    from scipy.ndimage import label

    canvas = 150
    img = Image.new('L', (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)

    try:
        bbox = pil_font.getbbox(char)
        if not bbox:
            return False
        draw.text((-bbox[0] + 5, -bbox[1] + 5), char, fill=0, font=pil_font)
    except Exception:
        return False

    arr = np.array(img)
    binary = arr < 128
    _, num_shapes = label(binary)

    return num_shapes == expected
