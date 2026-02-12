"""Glyph rendering utilities for font analysis.

This module provides functions and classes for rendering font glyphs to images and binary
masks for analysis. It supports the stroke extraction pipeline by generating
the input images and masks that are processed by skeleton and scoring modules.

Key functionality:
    - GlyphRenderer: Unified class for all rendering operations
    - Character rendering: Generate grayscale PNG images of characters
    - Mask generation: Create binary masks for glyph pixels
    - Font analysis: Detect small-caps fonts, check for holes and shape counts
    - Shape metrics: Analyze rendered text for connected components

Typical usage:
    from stroke_rendering import render_glyph_mask, render_char_image, GlyphRenderer

    # Get a binary mask of a character
    mask = render_glyph_mask('path/to/font.ttf', 'A')

    # Render a character to PNG bytes
    png_bytes = render_char_image('path/to/font.ttf', 'A')

    # Use GlyphRenderer for more control
    renderer = GlyphRenderer('path/to/font.ttf', font_size=200)
    img = renderer.render_char('A', canvas_size=224)
    mask = renderer.render_mask('A', canvas_size=224)
    text_img = renderer.render_text('Hello', canvas_size=400)

    # Check for small-caps font
    mismatched = check_case_mismatch('path/to/font.ttf')
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

# Module directory for resolving relative paths
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default rendering parameters
DEFAULT_CANVAS_SIZE = 224
DEFAULT_FONT_SIZE = 200
SMALLCAPS_NORM_SIZE = 64  # Normalization size for small-caps detection
SMALLCAPS_MIN_SIZE = 5  # Minimum glyph size (pixels) for small-caps detection
SMALLCAPS_PADDING = 5  # Padding around glyphs for small-caps rendering
HOLE_ANALYSIS_CANVAS = 400  # Canvas size for hole analysis rendering


def resolve_font_path(font_path: str) -> str:
    """Resolve a font file path to an absolute path.

    Converts relative font paths to absolute paths by joining them with
    the module's base directory. Absolute paths are returned unchanged.

    Args:
        font_path: The font file path, either absolute or relative.

    Returns:
        str: The absolute path to the font file.
    """
    if os.path.isabs(font_path):
        return font_path
    return os.path.join(_BASE_DIR, font_path)


# LRU cache size for font objects - larger cache reduces disk I/O
# when processing many characters from the same fonts
FONT_CACHE_SIZE = 256

@lru_cache(maxsize=FONT_CACHE_SIZE)
def _cached_font(font_path: str, size: int) -> FreeTypeFont:
    """Load a font with caching to avoid repeated disk I/O."""
    return ImageFont.truetype(font_path, size)

# Constants for rendering
CANVAS_FILL_THRESHOLD = 0.9  # Max fraction of canvas a glyph can fill
BINARIZATION_THRESHOLD = 128  # Grayscale threshold for mask generation


def binarize_image(arr: np.ndarray, threshold: int = BINARIZATION_THRESHOLD,
                   as_uint8: bool = False, invert: bool = False) -> np.ndarray:
    """Convert a grayscale image array to binary (black/white).

    Args:
        arr: Grayscale numpy array (0-255 values).
        threshold: Pixel values below this are considered "ink/foreground".
            Default BINARIZATION_THRESHOLD (128).
        as_uint8: If True, return uint8 array (0/1). If False, return bool.
        invert: If True, return True for background pixels instead of ink.

    Returns:
        Binary numpy array. By default, True/1 = ink (dark pixels),
        False/0 = background (light pixels).

    Example:
        >>> arr = np.array(img)
        >>> mask = binarize_image(arr)  # Boolean mask of ink pixels
        >>> binary = binarize_image(arr, as_uint8=True)  # uint8 0/1 array
        >>> bg_mask = binarize_image(arr, invert=True)  # Background pixels
    """
    if invert:
        binary = arr >= threshold
    else:
        binary = arr < threshold

    if as_uint8:
        return binary.astype(np.uint8)
    return binary


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


# ---------------------------------------------------------------------------
# GlyphRenderer Class - Unified Rendering Abstraction
# ---------------------------------------------------------------------------

@dataclass
class RenderConfig:
    """Configuration for glyph rendering.

    Attributes:
        canvas_size: Width and height of the output canvas in pixels.
        font_size: Initial font size in points (may be scaled down to fit).
        fill_threshold: Maximum fraction of canvas a glyph can fill (0.0-1.0).
        binarize_threshold: Grayscale threshold for mask generation (0-255).
        background: Background color for grayscale images (0=black, 255=white).
        ink_color: Ink color for drawing (0=black on L mode, 255=white on inverted).
    """
    canvas_size: int = DEFAULT_CANVAS_SIZE
    font_size: int = DEFAULT_FONT_SIZE
    fill_threshold: float = CANVAS_FILL_THRESHOLD
    binarize_threshold: int = BINARIZATION_THRESHOLD
    background: int = 255
    ink_color: int = 0


class GlyphRenderer:
    """Unified class for rendering font glyphs.

    This class consolidates common rendering patterns used throughout the codebase,
    providing a consistent interface for:
    - Rendering single characters to images
    - Generating binary masks for image processing
    - Rendering text strings for analysis
    - Computing bounding boxes and positions

    The renderer caches the loaded font and provides methods for common operations.

    Attributes:
        font_path: Resolved absolute path to the font file.
        font_size: Default font size in points.
        config: RenderConfig with default rendering parameters.

    Example:
        >>> renderer = GlyphRenderer('fonts/arial.ttf', font_size=200)
        >>> img = renderer.render_char('A', canvas_size=224)
        >>> mask = renderer.render_mask('A', canvas_size=224)
        >>> text_arr = renderer.render_text('Hello', canvas_size=400)
        >>> bbox = renderer.get_bbox('A')
    """

    def __init__(self, font_path: str, font_size: int = DEFAULT_FONT_SIZE,
                 config: RenderConfig = None):
        """Initialize the renderer with a font.

        Args:
            font_path: Path to the font file (TTF, OTF, etc.).
            font_size: Default font size in points.
            config: Optional RenderConfig to customize rendering.

        Raises:
            OSError: If the font file cannot be loaded.
        """
        self.font_path = resolve_font_path(font_path)
        self.font_size = font_size
        self.config = config or RenderConfig(font_size=font_size)
        # Validate font can be loaded
        self._font = _cached_font(self.font_path, self.font_size)

    @property
    def font(self) -> FreeTypeFont:
        """Get the loaded PIL font object."""
        return self._font

    def get_bbox(self, text: str) -> tuple | None:
        """Get the bounding box for text.

        Args:
            text: Text string to measure.

        Returns:
            Bounding box as (x0, y0, x1, y1), or None if not available.
        """
        return self._font.getbbox(text)

    def get_size(self, text: str) -> tuple[int, int]:
        """Get the width and height of rendered text.

        Args:
            text: Text string to measure.

        Returns:
            Tuple of (width, height) in pixels.
        """
        bbox = self.get_bbox(text)
        if not bbox:
            return 0, 0
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def render_char(self, char: str, canvas_size: int = None,
                    mode: Literal['L', 'RGB', 'RGBA'] = 'L',
                    auto_scale: bool = True) -> Image.Image | None:
        """Render a single character to an image.

        Args:
            char: Single character to render.
            canvas_size: Canvas width/height in pixels. Defaults to config value.
            mode: PIL image mode ('L' for grayscale, 'RGB' for color).
            auto_scale: If True, scale font down if glyph exceeds canvas.

        Returns:
            PIL Image with the rendered character, or None on error.
        """
        canvas_size = canvas_size or self.config.canvas_size
        fill = self.config.background
        ink = self.config.ink_color

        # Use RGB values for non-grayscale modes
        if mode in ('RGB', 'RGBA'):
            fill = (255, 255, 255) if self.config.background == 255 else (0, 0, 0)
            ink = (0, 0, 0) if self.config.ink_color == 0 else (255, 255, 255)

        pil_font = self._font
        bbox = pil_font.getbbox(char)
        if not bbox:
            return None

        # Scale font to fit if needed
        if auto_scale:
            pil_font, bbox = _scale_font_to_fit(
                self.font_path, pil_font, char, canvas_size, self.config.fill_threshold
            )
            if pil_font is None:
                return None

        img = Image.new(mode, (canvas_size, canvas_size), fill)
        draw = ImageDraw.Draw(img)
        x, y = _compute_centered_position(bbox, canvas_size)
        draw.text((x, y), char, fill=ink, font=pil_font)
        return img

    def render_char_png(self, char: str, canvas_size: int = None) -> bytes | None:
        """Render a character to PNG bytes.

        Args:
            char: Single character to render.
            canvas_size: Canvas width/height in pixels.

        Returns:
            PNG image data as bytes, or None on error.
        """
        img = self.render_char(char, canvas_size=canvas_size, mode='L')
        if img is None:
            return None
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def render_mask(self, char: str, canvas_size: int = None,
                    inverted: bool = False) -> np.ndarray | None:
        """Render a character as a binary mask.

        Args:
            char: Single character to render.
            canvas_size: Canvas width/height in pixels.
            inverted: If True, background is black (0) and ink is white (255).
                     Default is ink=True in mask.

        Returns:
            Boolean numpy array where True = glyph pixels, or None on error.
        """
        canvas_size = canvas_size or self.config.canvas_size

        # For inverted rendering (black bg, white ink)
        if inverted:
            img = Image.new('L', (canvas_size, canvas_size), 0)
            draw = ImageDraw.Draw(img)
            pil_font, bbox = _scale_font_to_fit(
                self.font_path, self._font, char, canvas_size, self.config.fill_threshold
            )
            if pil_font is None or not bbox:
                return None
            x, y = _compute_centered_position(bbox, canvas_size)
            draw.text((x, y), char, fill=255, font=pil_font)
            return np.array(img) > self.config.binarize_threshold

        # Standard rendering (white bg, black ink)
        img = self.render_char(char, canvas_size=canvas_size, mode='L')
        if img is None:
            return None
        return np.array(img) < self.config.binarize_threshold

    def render_mask_with_bbox(self, char: str, canvas_size: int = None
                              ) -> tuple[np.ndarray | None, tuple | None]:
        """Render a character mask and compute its tight bounding box.

        Args:
            char: Single character to render.
            canvas_size: Canvas width/height in pixels.

        Returns:
            Tuple of (mask, bbox):
                - mask: Boolean numpy array where True = glyph pixels
                - bbox: Tight bounding box as (col_min, row_min, col_max, row_max)
            Returns (None, None) on error.
        """
        mask = self.render_mask(char, canvas_size=canvas_size)
        if mask is None:
            return None, None

        # Find tight bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return None, None

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return mask, (cmin, rmin, cmax, rmax)

    def render_text(self, text: str, canvas_size: int = None,
                    mode: Literal['L', 'RGB'] = 'L') -> np.ndarray | None:
        """Render a text string to a numpy array.

        Args:
            text: Text string to render.
            canvas_size: Canvas width/height in pixels.
            mode: PIL image mode.

        Returns:
            Numpy array of the rendered image, or None on error.
        """
        canvas_size = canvas_size or HOLE_ANALYSIS_CANVAS
        fill = 255 if mode == 'L' else (255, 255, 255)
        ink = 0 if mode == 'L' else (0, 0, 0)

        img = Image.new(mode, (canvas_size, canvas_size), fill)
        draw = ImageDraw.Draw(img)

        bbox = self._font.getbbox(text)
        if not bbox:
            return None

        x, y = _compute_centered_position(bbox, canvas_size)
        draw.text((x, y), text, fill=ink, font=self._font)
        return np.array(img)

    def render_for_phash(self, text: str, canvas_width: int = 256,
                         canvas_height: int = 64) -> Image.Image | None:
        """Render text for perceptual hashing.

        Creates a consistent rendering suitable for computing perceptual hashes
        for font deduplication.

        Args:
            text: Text to render.
            canvas_width: Width of the canvas.
            canvas_height: Height of the canvas.

        Returns:
            PIL Image suitable for hashing, or None on error.
        """
        try:
            img = Image.new('L', (canvas_width, canvas_height), 255)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), text, font=self._font, fill=0)
            return img
        except Exception:
            return None

    @staticmethod
    def bbox_size(bbox: tuple) -> tuple[int, int]:
        """Compute width and height from a bounding box.

        Args:
            bbox: Bounding box as (x0, y0, x1, y1).

        Returns:
            Tuple of (width, height).
        """
        if not bbox:
            return 0, 0
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    @staticmethod
    def centered_position(bbox: tuple, canvas_size: int) -> tuple[int, int]:
        """Compute position to center a bounding box in a canvas.

        Static wrapper for _compute_centered_position.

        Args:
            bbox: Bounding box as (x0, y0, x1, y1).
            canvas_size: Width/height of the square canvas.

        Returns:
            Tuple of (x, y) offset for draw.text().
        """
        return _compute_centered_position(bbox, canvas_size)


# Factory function for creating renderers
def create_renderer(font_path: str, font_size: int = DEFAULT_FONT_SIZE,
                    **config_kwargs) -> GlyphRenderer | None:
    """Create a GlyphRenderer with optional configuration.

    Factory function that handles errors gracefully.

    Args:
        font_path: Path to the font file.
        font_size: Font size in points.
        **config_kwargs: Additional arguments for RenderConfig.

    Returns:
        GlyphRenderer instance, or None if font cannot be loaded.
    """
    try:
        config = RenderConfig(font_size=font_size, **config_kwargs)
        return GlyphRenderer(font_path, font_size=font_size, config=config)
    except OSError:
        return None


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


def _render_letter_normalized(pil_font: FreeTypeFont, char: str) -> np.ndarray | None:
    """Render a letter and return as normalized binary array.

    Renders the character at its natural size with padding, scales to
    SMALLCAPS_NORM_SIZE, and binarizes.

    Args:
        pil_font: Loaded PIL font object.
        char: Single character to render.

    Returns:
        Binary numpy array of shape (SMALLCAPS_NORM_SIZE, SMALLCAPS_NORM_SIZE),
        or None if the character cannot be rendered or is too small.
    """
    bbox = pil_font.getbbox(char)
    if not bbox:
        return None

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    if w < SMALLCAPS_MIN_SIZE or h < SMALLCAPS_MIN_SIZE:
        return None

    # Render at natural size with padding
    img = Image.new('L', (w + SMALLCAPS_PADDING * 2, h + SMALLCAPS_PADDING * 2), 255)
    draw = ImageDraw.Draw(img)
    draw.text((SMALLCAPS_PADDING - bbox[0], SMALLCAPS_PADDING - bbox[1]), char, fill=0, font=pil_font)

    # Scale to normalized size and binarize
    scaled = img.resize((SMALLCAPS_NORM_SIZE, SMALLCAPS_NORM_SIZE), Image.Resampling.BILINEAR)
    return np.array(scaled) < BINARIZATION_THRESHOLD


def _compute_iou(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Compute Intersection over Union between two binary arrays.

    Args:
        arr1: First binary array.
        arr2: Second binary array (same shape as arr1).

    Returns:
        IoU value from 0.0 (no overlap) to 1.0 (identical).
        Returns 0.0 if union is empty.
    """
    intersection = np.sum(arr1 & arr2)
    union = np.sum(arr1 | arr2)
    return intersection / union if union > 0 else 0.0


# Letters that are naturally symmetric between upper/lower case
# These are excluded from case mismatch detection since many fonts
# legitimately render them identically (just scaled)
CASE_SYMMETRIC_LETTERS = frozenset('cvwxosz')


def check_case_mismatch(font_path: str, threshold: float = 0.70) -> list[str]:
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
        - Letters with bounding boxes smaller than SMALLCAPS_MIN_SIZE are skipped.
        - Each letter pair is rendered at natural size, then scaled to
          SMALLCAPS_NORM_SIZE for fair comparison.
        - Uses IoU metric: intersection / union of binarized glyph pixels.
        - Useful for filtering out fonts unsuitable for case-sensitive
          character recognition.
        - Naturally symmetric letters (c, o, s, v, w, x, z) are excluded
          since they often look identical between cases in many fonts.
    """
    font_path = resolve_font_path(font_path)
    mismatched = []

    try:
        pil_font = _cached_font(font_path, 100)
    except OSError:
        return []

    for lower in 'abcdefghijklmnopqrstuvwxyz':
        # Skip naturally symmetric letters
        if lower in CASE_SYMMETRIC_LETTERS:
            continue
        try:
            l_arr = _render_letter_normalized(pil_font, lower)
            u_arr = _render_letter_normalized(pil_font, lower.upper())

            if l_arr is None or u_arr is None:
                continue

            if _compute_iou(l_arr, u_arr) >= threshold:
                mismatched.append(lower)

        except (ValueError, MemoryError, OSError):
            continue

    return mismatched


def render_text_to_image(pil_font: FreeTypeFont, text: str,
                         canvas_size: tuple[int, int] | int = None,
                         padding: int = 10,
                         background: int = 255,
                         fill: int = 0,
                         center: bool = True) -> Image.Image | None:
    """Render text to a PIL Image with configurable options.

    A general-purpose utility for rendering text using a loaded font.
    Handles bounding box computation and optional centering automatically.

    Args:
        pil_font: A PIL ImageFont object (already loaded).
        text: The text string to render.
        canvas_size: Canvas dimensions. Can be:
            - tuple (width, height): Use exact dimensions
            - int: Use as both width and height (square canvas)
            - None: Auto-size to fit text with padding
        padding: Padding around text when auto-sizing or for positioning.
        background: Background color (0-255). Default 255 (white).
        fill: Text color (0-255). Default 0 (black).
        center: If True and canvas_size is specified, center the text.

    Returns:
        PIL Image with the rendered text, or None on failure.

    Example:
        >>> font = ImageFont.truetype("font.ttf", 48)
        >>> img = render_text_to_image(font, "Hello", canvas_size=100)
        >>> img = render_text_to_image(font, "A", canvas_size=(64, 64))
        >>> img = render_text_to_image(font, "Test")  # Auto-sized
    """
    try:
        bbox = pil_font.getbbox(text)
        if not bbox:
            return None

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Determine canvas size
        if canvas_size is None:
            # Auto-size to fit text
            img_w = text_w + padding * 2
            img_h = text_h + padding * 2
            x = padding - bbox[0]
            y = padding - bbox[1]
        elif isinstance(canvas_size, int):
            img_w = img_h = canvas_size
            if center:
                x = (canvas_size - text_w) // 2 - bbox[0]
                y = (canvas_size - text_h) // 2 - bbox[1]
            else:
                x = padding - bbox[0]
                y = padding - bbox[1]
        else:
            img_w, img_h = canvas_size
            if center:
                x = (img_w - text_w) // 2 - bbox[0]
                y = (img_h - text_h) // 2 - bbox[1]
            else:
                x = padding - bbox[0]
                y = padding - bbox[1]

        img = Image.new('L', (img_w, img_h), background)
        draw = ImageDraw.Draw(img)
        draw.text((x, y), text, fill=fill, font=pil_font)
        return img

    except OSError:
        return None


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
    img = render_text_to_image(pil_font, text, canvas_size=HOLE_ANALYSIS_CANVAS)
    if img is None:
        return None
    return np.array(img)


def analyze_shape_metrics(arr: np.ndarray, width: int) -> tuple[int, float]:
    """Analyze shape metrics from a rendered text array.

    Computes basic shape statistics useful for character classification
    and validation. Used to detect connected/script fonts where individual
    letters span a large portion of the total text width.

    Args:
        arr: 2D numpy array containing rendered text (grayscale).
        width: Reference width for computing width percentage.

    Returns:
        A tuple of (shape_count, max_component_width_pct) where:
            - shape_count: Number of connected components in the binarized image
            - max_component_width_pct: Width of the largest connected component
              as a fraction of the reference width. For normal fonts, each letter
              is a separate component. For script/cursive fonts, connected letters
              form larger components that span more of the total width.

    Notes:
        - Uses scipy.ndimage.label for connected component analysis.
        - Threshold of BINARIZATION_THRESHOLD (128) is used for binarization.
        - A max_component_width_pct > 0.225 typically indicates a script font
          where letters are connected.
    """
    from scipy.ndimage import label

    binary = arr < BINARIZATION_THRESHOLD
    labeled, num_shapes = label(binary)

    # Find the width of the largest connected component
    max_width_pct = 0.0
    if num_shapes > 0:
        for i in range(1, num_shapes + 1):
            component = (labeled == i)
            # Get columns that contain this component
            cols_with_component = np.any(component, axis=0)
            comp_width = np.sum(cols_with_component)
            width_pct = comp_width / width if width > 0 else 0
            if width_pct > max_width_pct:
                max_width_pct = width_pct

    return num_shapes, max_width_pct


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
    binary = arr < BINARIZATION_THRESHOLD

    # Label background components
    _bg_labeled, bg_count = label(~binary)

    # If more than 1 background region, there are holes
    return bg_count > 1


def get_char_shape_count(pil_font: FreeTypeFont, char: str) -> int:
    """Get the number of connected components for a character.

    Args:
        pil_font: A PIL ImageFont object.
        char: The character to analyze.

    Returns:
        Number of connected components, or -1 if rendering fails.
    """
    from scipy.ndimage import label

    canvas = 150
    img = Image.new('L', (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)

    try:
        bbox = pil_font.getbbox(char)
        if not bbox:
            return -1
        draw.text((-bbox[0] + 5, -bbox[1] + 5), char, fill=0, font=pil_font)
    except Exception:
        return -1

    arr = np.array(img)
    binary = arr < 128
    _, num_shapes = label(binary)

    return num_shapes


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
    num_shapes = get_char_shape_count(pil_font, char)
    return num_shapes == expected
