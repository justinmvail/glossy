"""Dataset for loading glyph images from fonts for stroke prediction training.

Reads font file paths from SQLite database, renders glyphs on-the-fly using
PIL/Pillow. No stroke annotations needed — the glyph raster is the ground truth.
"""

import os
import sqlite3
import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont

from model import CANVAS_SIZE, CHARS_LIST, char_to_index

logger = logging.getLogger(__name__)


class GlyphDataset(Dataset):
    """Dataset that renders glyph images on-the-fly from font files.

    Each sample is a (glyph_image_tensor, char_label_index) pair.
    No stroke labels needed — the rendering loss provides supervision.

    Args:
        db_path: Path to SQLite database with fonts table.
        font_dir: Base directory for resolving relative font paths.
        canvas_size: Size of the square canvas in pixels.
        chars: String of characters to include.
        max_fonts: If set, limit to this many fonts (for debugging).
        cache_dir: If set, cache rendered images to this directory.
        augment: Whether to apply random augmentation.
    """

    def __init__(self, db_path: str, font_dir: str,
                 canvas_size: int = CANVAS_SIZE,
                 chars: str = None, max_fonts: int = None,
                 cache_dir: str = None, augment: bool = False):
        self.canvas_size = canvas_size
        self.chars = list(chars or ''.join(CHARS_LIST))
        self.font_dir = font_dir
        self.cache_dir = cache_dir
        self.augment = augment

        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        # Load font paths from database
        self.font_paths = self._load_font_paths(db_path, max_fonts)

        # Build index: (font_idx, char_idx) pairs
        self.samples = []
        for fi in range(len(self.font_paths)):
            for ci in range(len(self.chars)):
                self.samples.append((fi, ci))

        logger.info("GlyphDataset: %d fonts x %d chars = %d samples",
                     len(self.font_paths), len(self.chars), len(self.samples))

    def _load_font_paths(self, db_path: str, max_fonts: int = None) -> list:
        """Load font file paths from SQLite database.

        Filters to fonts that passed prefilter and are not rejected.
        """
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = """
                SELECT f.file_path FROM fonts f
                LEFT JOIN font_checks fc ON f.id = fc.font_id
                LEFT JOIN font_removals fr ON f.id = fr.font_id
                WHERE fr.font_id IS NULL
                AND (fc.prefilter_passed = 1 OR fc.prefilter_passed IS NULL)
                AND (fc.is_cursive = 0 OR fc.is_cursive IS NULL)
            """
            if max_fonts:
                query += f" LIMIT {max_fonts}"

            rows = conn.execute(query).fetchall()
            paths = []
            for row in rows:
                fp = row['file_path']
                if not os.path.isabs(fp):
                    fp = os.path.join(self.font_dir, fp)
                if os.path.exists(fp):
                    paths.append(fp)

            logger.info("Loaded %d valid font paths from %s", len(paths), db_path)
            return paths
        finally:
            conn.close()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        """Get a single training sample.

        Returns:
            Tuple of (image_tensor, char_index):
                image_tensor: (1, H, W) float32 tensor, 1.0=white, 0.0=black.
                char_index: int, index into CHARS_LIST.
        """
        fi, ci = self.samples[idx]
        font_path = self.font_paths[fi]
        char = self.chars[ci]
        char_idx = char_to_index(char)

        # Try cache first
        if self.cache_dir:
            cache_path = os.path.join(
                self.cache_dir, f"{fi}_{ci}.npy",
            )
            if os.path.exists(cache_path):
                try:
                    arr = np.load(cache_path)
                    img_tensor = torch.from_numpy(arr).float().unsqueeze(0)
                    return img_tensor, char_idx
                except (EOFError, ValueError):
                    pass  # Corrupt cache file, fall through to render

        # Render glyph
        mask = self._render_glyph(font_path, char)
        if mask is None:
            # Return blank white image if rendering fails
            img_tensor = torch.ones(1, self.canvas_size, self.canvas_size)
            return img_tensor, char_idx

        # Convert to tensor: 1.0=white background, 0.0=black glyph
        arr = mask.astype(np.float32)  # mask is True where glyph
        img_arr = 1.0 - arr  # invert: 1=white, 0=glyph
        img_tensor = torch.from_numpy(img_arr).float().unsqueeze(0)

        # Apply augmentation
        if self.augment:
            img_tensor = self._augment(img_tensor)

        # Cache to disk
        if self.cache_dir:
            try:
                np.save(cache_path, img_arr)
            except OSError:
                pass

        return img_tensor, char_idx

    def get_glyph_mask(self, idx: int) -> torch.Tensor:
        """Get the binary glyph mask for a sample (for loss computation).

        Returns:
            (H, W) float32 tensor, 1.0=glyph, 0.0=background.
        """
        fi, ci = self.samples[idx]
        font_path = self.font_paths[fi]
        char = self.chars[ci]

        mask = self._render_glyph(font_path, char)
        if mask is None:
            return torch.zeros(self.canvas_size, self.canvas_size)

        return torch.from_numpy(mask.astype(np.float32))

    def _render_glyph(self, font_path: str, char: str) -> np.ndarray | None:
        """Render a single glyph as a binary mask.

        Renders the character at the largest font size that fits in 90% of
        the canvas, centered. Same approach as optimize_strokes.py.

        Returns:
            Boolean numpy array (H, W), True where glyph, False where bg.
            None if the font can't render this character.
        """
        img = Image.new('L', (self.canvas_size, self.canvas_size), 255)
        draw = ImageDraw.Draw(img)

        for font_size in range(200, 20, -5):
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                continue
            bbox = font.getbbox(char)
            if bbox is None:
                return None
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            max_dim = self.canvas_size * 0.9
            if tw <= max_dim and th <= max_dim:
                x = (self.canvas_size - tw) / 2 - bbox[0]
                y = (self.canvas_size - th) / 2 - bbox[1]
                draw.text((x, y), char, fill=0, font=font)
                arr = np.array(img)
                return arr < 128  # Boolean mask

        return None

    def _augment(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Apply random affine augmentation.

        Small random rotation (+-5 deg), scale (+-10%), translate (+-5px).
        """
        import torchvision.transforms.functional as TF
        import random

        angle = random.uniform(-5, 5)
        scale = random.uniform(0.9, 1.1)
        tx = random.uniform(-5, 5)
        ty = random.uniform(-5, 5)

        # Apply affine: rotation + scale
        img_tensor = TF.affine(
            img_tensor, angle=angle, translate=[tx, ty],
            scale=scale, shear=0, fill=1.0,
        )
        return img_tensor


def collate_with_masks(batch: list) -> tuple:
    """Custom collate function that also generates glyph masks from images.

    The glyph mask is derived from the image: pixels below 0.5 are glyph.

    Args:
        batch: List of (image_tensor, char_index) tuples from GlyphDataset.

    Returns:
        Tuple of (images, char_indices, glyph_masks):
            images: (B, 1, H, W) float32 tensor.
            char_indices: (B,) long tensor.
            glyph_masks: (B, H, W) float32 tensor (1=glyph, 0=bg).
    """
    images = torch.stack([b[0] for b in batch])
    char_indices = torch.tensor([b[1] for b in batch], dtype=torch.long)

    # Glyph mask: image is white bg (1.0), black glyph (0.0)
    # Mask should be 1 where glyph (dark), 0 where background (bright)
    glyph_masks = (images.squeeze(1) < 0.5).float()

    return images, char_indices, glyph_masks
