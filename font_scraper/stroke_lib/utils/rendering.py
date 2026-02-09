"""Glyph rendering utilities."""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from ..domain.geometry import BBox


def render_glyph_mask(font_path: str, char: str, canvas_size: int = 224) -> Optional[np.ndarray]:
    """Render a glyph as a binary mask.

    Args:
        font_path: Path to the font file
        char: Character to render
        canvas_size: Size of the square canvas

    Returns:
        Binary numpy array (H, W) where True = glyph pixels, or None on failure
    """
    try:
        font = ImageFont.truetype(font_path, size=int(canvas_size * 0.8))
    except Exception:
        return None

    # Create image and draw character
    img = Image.new('L', (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(img)

    # Get character bounding box for centering
    bbox = draw.textbbox((0, 0), char, font=font)
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]

    # Center the character
    x = (canvas_size - char_width) // 2 - bbox[0]
    y = (canvas_size - char_height) // 2 - bbox[1]

    draw.text((x, y), char, font=font, fill=255)

    # Convert to binary mask
    mask = np.array(img) > 127
    return mask


def get_glyph_bbox(mask: np.ndarray) -> Optional[BBox]:
    """Get bounding box of non-zero pixels in mask.

    Args:
        mask: Binary numpy array

    Returns:
        BBox or None if mask is empty
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

    Args:
        font_path: Path to the font file
        text: Text to render
        size: Font size

    Returns:
        Binary numpy array or None on failure
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
