"""Glyph rendering utilities.

This module provides functions for rendering font glyphs as binary masks
and extracting information from rendered images. These utilities are used
throughout the stroke editing package for glyph analysis and visualization.

The module provides the following functions:
    render_glyph_mask: Render a single character as a binary mask.
    get_glyph_bbox: Extract bounding box from a binary mask.
    render_text_for_analysis: Render text for shape analysis.

Example usage:
    Rendering a glyph::

        from stroke_lib.utils.rendering import render_glyph_mask, get_glyph_bbox

        # Render character 'A' at 224x224
        mask = render_glyph_mask('/fonts/arial.ttf', 'A', canvas_size=224)

        if mask is not None:
            # Get the bounding box
            bbox = get_glyph_bbox(mask)
            print(f"Glyph size: {bbox.width} x {bbox.height}")

            # Use the mask for analysis
            import numpy as np
            pixel_count = np.sum(mask)
            print(f"Glyph has {pixel_count} pixels")

    Analyzing text shapes::

        from stroke_lib.utils.rendering import render_text_for_analysis

        arr = render_text_for_analysis('/fonts/arial.ttf', 'Hello', size=100)
        if arr is not None:
            from scipy import ndimage
            labeled, num_components = ndimage.label(arr)
            print(f"Text has {num_components} connected components")
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from ..domain.geometry import BBox


def render_glyph_mask(font_path: str, char: str, canvas_size: int = 224) -> Optional[np.ndarray]:
    """Render a glyph as a binary mask.

    Renders the specified character from the font file onto a square
    canvas, centered and scaled to fit. Returns a binary mask where
    True values indicate glyph pixels.

    The rendering process:
        1. Load the font at size 200 (or smaller if needed to fit)
        2. Calculate character bounding box
        3. Scale down if character is too large for canvas
        4. Center the character on the canvas
        5. Render and convert to binary mask

    Args:
        font_path: Path to the font file (TTF, OTF, etc.).
        char: Single character to render.
        canvas_size: Size of the square canvas in pixels. Default is 224.
            The character will be centered and scaled to fit within
            90% of this size.

    Returns:
        Binary numpy array of shape (canvas_size, canvas_size) where
        True indicates glyph pixels and False indicates background.
        Returns None if the font cannot be loaded or the character
        has no visible representation.

    Example:
        >>> mask = render_glyph_mask('/fonts/arial.ttf', 'A')
        >>> mask.shape
        (224, 224)
        >>> mask.dtype
        dtype('bool')
    """
    font_size = 200
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return None

    # Create image with white background
    img = Image.new('L', (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)

    # Get character bounding box for centering
    bbox = font.getbbox(char)
    if not bbox:
        return None

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Scale down if glyph is too large for canvas
    if w > canvas_size * 0.9 or h > canvas_size * 0.9:
        scale = min(canvas_size * 0.9 / w, canvas_size * 0.9 / h)
        font_size = int(font_size * scale)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            return None
        bbox = font.getbbox(char)
        if not bbox:
            return None
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

    # Center the character
    x = (canvas_size - w) // 2 - bbox[0]
    y = (canvas_size - h) // 2 - bbox[1]

    draw.text((x, y), char, fill=0, font=font)

    # Convert to binary mask: True where glyph is (dark pixels)
    return np.array(img) < 128


def get_glyph_bbox(mask: np.ndarray) -> Optional[BBox]:
    """Get bounding box of non-zero pixels in mask.

    Finds the smallest axis-aligned rectangle that contains all
    non-zero (True) pixels in the mask.

    Args:
        mask: Binary numpy array where non-zero values indicate
            glyph pixels.

    Returns:
        BBox object with x_min, y_min, x_max, y_max representing
        the bounding box of the glyph. Returns None if the mask
        contains no non-zero pixels.

    Example:
        >>> import numpy as np
        >>> mask = np.zeros((100, 100), dtype=bool)
        >>> mask[20:80, 30:70] = True
        >>> bbox = get_glyph_bbox(mask)
        >>> bbox.to_tuple()
        (30.0, 20.0, 69.0, 79.0)
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None

    return BBox(
        x_min=float(cols.min()),
        y_min=float(rows.min()),
        x_max=float(cols.max()),
        y_max=float(rows.max())
    )


def render_text_for_analysis(font_path: str, text: str, size: int = 100) -> Optional[np.ndarray]:
    """Render text and return as binary array for shape analysis.

    Renders a text string (can be multiple characters) and returns
    a binary array suitable for connected component analysis and
    shape metrics computation.

    Unlike render_glyph_mask, this function:
        - Sizes the canvas to fit the text (not fixed square)
        - Does not center or scale the text
        - Uses white text on black background (for component labeling)

    Args:
        font_path: Path to the font file.
        text: Text string to render (can be multiple characters).
        size: Font size in points. Default is 100.

    Returns:
        Binary numpy array where True indicates text pixels.
        The array size is determined by the text dimensions plus
        padding. Returns None if the font cannot be loaded.

    Example:
        >>> arr = render_text_for_analysis('/fonts/arial.ttf', 'Hello')
        >>> from scipy import ndimage
        >>> labeled, n = ndimage.label(arr)
        >>> print(f"Found {n} connected components")
    """
    try:
        pil_font = ImageFont.truetype(font_path, size=size)
    except Exception:
        return None

    # Get text size
    dummy_img = Image.new('L', (1, 1), 0)
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=pil_font)
    width = bbox[2] - bbox[0] + 20
    height = bbox[3] - bbox[1] + 20

    # Render
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    draw.text((10 - bbox[0], 10 - bbox[1]), text, font=pil_font, fill=255)

    return np.array(img) > 127
