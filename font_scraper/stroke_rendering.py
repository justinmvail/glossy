"""Glyph rendering utilities for font analysis.

This module provides functions for rendering font glyphs to images and binary
masks for analysis. It supports the stroke extraction pipeline by generating
the input images and masks that are processed by skeleton and scoring modules.

Key functionality:
    - Character rendering: Generate grayscale PNG images of characters
    - Mask generation: Create binary masks for glyph pixels
    - Font analysis: Detect small-caps fonts, check for holes and shape counts
    - Shape metrics: Analyze rendered text for connected components

Typical usage:
    from stroke_rendering import render_glyph_mask, render_char_image

    # Get a binary mask of a character
    mask = render_glyph_mask('path/to/font.ttf', 'A')

    # Render a character to PNG bytes
    png_bytes = render_char_image('path/to/font.ttf', 'A')

    # Check for small-caps font
    mismatched = check_case_mismatch('path/to/font.ttf')
"""

from __future__ import annotations
import io
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont
from stroke_flask import DEFAULT_CANVAS_SIZE, DEFAULT_FONT_SIZE, resolve_font_path


# LRU cache for font objects to avoid repeated disk I/O
@lru_cache(maxsize=32)
def _cached_font(font_path: str, size: int) -> FreeTypeFont:
    """Load a font with caching to avoid repeated disk I/O."""
    return ImageFont.truetype(font_path, size)

# Constants for rendering
CANVAS_FILL_THRESHOLD = 0.9  # Max fraction of canvas a glyph can fill
BINARIZATION_THRESHOLD = 128  # Grayscale threshold for mask generation


def _scale_font_to_fit(font_path: str, pil_font: FreeTypeFont, char: str,
                       canvas_size: int, threshold: float = CANVAS_FILL_THRESHOLD
                       ) -> tuple[FreeTypeFont | None, tuple | None]:
    """Scale a font to fit within the canvas bounds.

    Args:
        font_path: Path to the font file.
        pil_font: Currently loaded PIL font object.
        char: Character to measure.
        canvas_size: Target canvas size in pixels.
        threshold: Maximum fraction of canvas the glyph can fill.

    Returns:
        Tuple of (scaled_font, bbox) where scaled_font may be the same as
        pil_font if no scaling was needed. Returns (None, None) on error.
    """
    bbox = pil_font.getbbox(char)
    if not bbox:
        return pil_font, bbox

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    max_dim = canvas_size * threshold

    if w <= max_dim and h <= max_dim:
        return pil_font, bbox

    # Need to scale down
    scale = min(max_dim / w, max_dim / h)
    current_size = pil_font.size
    new_size = int(current_size * scale)

    try:
        scaled_font = _cached_font(font_path, new_size)
        new_bbox = scaled_font.getbbox(char)
        return scaled_font, new_bbox
    except OSError:
        return None, None


def _compute_centered_position(bbox: tuple, canvas_size: int) -> tuple[int, int]:
    """Compute the position to center a bounding box in a canvas.

    Args:
        bbox: Bounding box as (x0, y0, x1, y1) from font.getbbox().
        canvas_size: Width/height of the square canvas.

    Returns:
        Tuple of (x, y) offset for draw.text() to center the glyph.
    """
    if not bbox:
        return 0, 0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (canvas_size - w) // 2 - bbox[0]
    y = (canvas_size - h) // 2 - bbox[1]
    return x, y


def render_char_image(font_path: str, char: str, font_size: int = DEFAULT_FONT_SIZE,
                      canvas_size: int = DEFAULT_CANVAS_SIZE) -> bytes | None:
    """Render a character to a centered grayscale PNG image.

    Creates a square grayscale image with the character drawn in black on
    a white background, centered within the canvas.

    Args:
        font_path: Path to the font file (.ttf, .otf, etc.). Can be absolute
            or relative to the module's base directory.
        char: The character to render (single character string).
        font_size: Initial font size in points. Defaults to DEFAULT_FONT_SIZE.
            May be scaled down if the glyph exceeds canvas bounds.
        canvas_size: Width and height of the output image in pixels.
            Defaults to DEFAULT_CANVAS_SIZE.

    Returns:
        PNG image data as bytes, or None if the font file cannot be loaded.

    Notes:
        - The glyph is automatically scaled down if it exceeds 90% of the
          canvas size in either dimension.
        - Font path is resolved using resolve_font_path() for relative paths.
        - Output is 8-bit grayscale (mode 'L').
    """
    font_path = resolve_font_path(font_path)
    try:
        pil_font = _cached_font(font_path, font_size)
    except OSError:
        return None

    # Scale font to fit and get centered position
    pil_font, bbox = _scale_font_to_fit(font_path, pil_font, char, canvas_size)
    if pil_font is None:
        return None

    img = Image.new('L', (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    x, y = _compute_centered_position(bbox, canvas_size)
    draw.text((x, y), char, fill=0, font=pil_font)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


# Cache for rendered glyph masks - key is (font_path, char, canvas_size)
_mask_cache = {}
_MASK_CACHE_MAX_SIZE = 256


def render_glyph_mask(font_path: str, char: str,
                      canvas_size: int = DEFAULT_CANVAS_SIZE) -> np.ndarray | None:
    """Render a glyph as a binary mask for image processing.

    Creates a boolean numpy array where True indicates pixels that are
    part of the glyph (ink pixels). Results are cached for performance.

    Args:
        font_path: Path to the font file. Can be absolute or relative.
        char: The character to render.
        canvas_size: Width and height of the output mask in pixels.
            Defaults to DEFAULT_CANVAS_SIZE.

    Returns:
        A 2D boolean numpy array of shape (canvas_size, canvas_size) where
        True indicates glyph pixels. Returns None if the font cannot be loaded.

    Notes:
        - Uses a fixed initial font size of 200 points for good resolution.
        - Automatically scales down for large glyphs (>90% of canvas).
        - Threshold of 128 is used to binarize the grayscale rendering.
        - The glyph is centered within the canvas.
        - Results are cached (up to 256 entries) for repeated calls.
    """
    font_path = resolve_font_path(font_path)

    # Check cache first
    cache_key = (font_path, char, canvas_size)
    if cache_key in _mask_cache:
        return _mask_cache[cache_key].copy()  # Return copy to prevent mutation

    try:
        pil_font = _cached_font(font_path, 200)
    except OSError:
        return None

    # Scale font to fit and get centered position
    pil_font, bbox = _scale_font_to_fit(font_path, pil_font, char, canvas_size)
    if pil_font is None:
        return None

    img = Image.new('L', (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    x, y = _compute_centered_position(bbox, canvas_size)
    draw.text((x, y), char, fill=0, font=pil_font)

    result = np.array(img) < BINARIZATION_THRESHOLD

    # Cache the result (with simple LRU eviction)
    if len(_mask_cache) >= _MASK_CACHE_MAX_SIZE:
        # Remove oldest entry (first key)
        oldest = next(iter(_mask_cache))
        del _mask_cache[oldest]
    _mask_cache[cache_key] = result.copy()

    return result


def check_case_mismatch(font_path: str, threshold: float = 0.80) -> list[str]:
    """Check if lowercase letters match their uppercase counterparts.

    Detects small-caps fonts where lowercase letters are rendered as
    scaled-down versions of uppercase letters. Compares each letter pair
    by normalizing to the same size and computing IoU (Intersection over
    Union) similarity.

    Args:
        font_path: Path to the font file to analyze.
        threshold: IoU threshold above which letters are considered matching.
            Defaults to 0.80 (80% similarity).

    Returns:
        A list of lowercase letters (e.g., ['a', 'b', 'c']) that appear
        visually identical to their uppercase counterparts. Empty list if
        the font cannot be loaded or no mismatches are found.

    Notes:
        - Letters with bounding boxes smaller than 5x5 pixels are skipped.
        - Each letter pair is rendered at natural size, then scaled to a
          common 64x64 size for fair comparison.
        - Uses IoU metric: intersection / union of binarized glyph pixels.
        - Useful for filtering out fonts unsuitable for case-sensitive
          character recognition.
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


def render_text_for_analysis(pil_font: FreeTypeFont, text: str) -> np.ndarray | None:
    """Render text and return as a numpy array for shape analysis.

    Creates a centered rendering of the given text string suitable for
    further image analysis operations.

    Args:
        pil_font: A PIL ImageFont object (already loaded).
        text: The text string to render.

    Returns:
        A 2D numpy array of shape (400, 400) containing the grayscale
        rendering (0=black/ink, 255=white/background). Returns None if
        the text cannot be rendered (e.g., missing glyphs).

    Notes:
        - Uses a fixed 400x400 canvas size.
        - Text is centered based on its bounding box.
        - Exceptions during rendering are caught and result in None return.
    """
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


def analyze_shape_metrics(arr: np.ndarray, width: int) -> tuple[int, float]:
    """Analyze shape metrics from a rendered text array.

    Computes basic shape statistics useful for character classification
    and validation.

    Args:
        arr: 2D numpy array containing rendered text (grayscale).
        width: Reference width for computing width percentage.

    Returns:
        A tuple of (shape_count, max_width_pct) where:
            - shape_count: Number of connected components in the binarized image
            - max_width_pct: Maximum horizontal span of any row as a fraction
              of the reference width

    Notes:
        - Uses scipy.ndimage.label for connected component analysis.
        - Threshold of 128 is used for binarization.
        - max_width_pct can exceed 1.0 if the rendered text is wider than
          the reference width.
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


def check_char_holes(pil_font: FreeTypeFont, char: str) -> bool:
    """Check if a character has holes (inner contours).

    Determines whether the rendered character contains enclosed regions
    (like the inside of 'O', 'A', '8', etc.).

    Args:
        pil_font: A PIL ImageFont object.
        char: The character to analyze.

    Returns:
        True if the character has at least one hole (enclosed background
        region), False otherwise.

    Notes:
        - Uses connected component labeling on the background (non-glyph)
          pixels to detect holes.
        - A character has holes if there is more than one background region
          (the outer background plus at least one inner hole).
        - Uses a 150x150 canvas for rendering.
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


def check_char_shape_count(pil_font: FreeTypeFont, char: str, expected: int) -> bool:
    """Check if a character has the expected number of connected components.

    Useful for validating that characters like 'i' (2 components) or
    '%' (3 components) are rendered correctly.

    Args:
        pil_font: A PIL ImageFont object.
        char: The character to analyze.
        expected: The expected number of connected components.

    Returns:
        True if the number of connected components matches expected,
        False otherwise.

    Notes:
        - Uses scipy.ndimage.label for connected component counting.
        - Returns False if the character cannot be rendered.
        - Uses a 150x150 canvas for rendering.
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
