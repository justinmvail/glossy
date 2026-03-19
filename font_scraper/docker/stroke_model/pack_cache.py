"""Pre-render all glyphs and pack into a single memory-mapped binary.

Renders every font x character combination as a 224x224 uint8 image and stores
them in a single memmap file for fast random access during training.

Usage:
    python3 pack_cache.py --cache-dir /app/cache --db /data/fonts.db --font-dir /
"""

import argparse
import logging
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

CANVAS_SIZE = 224
SAMPLE_BYTES = CANVAS_SIZE * CANVAS_SIZE


def render_glyph(font_path, char, canvas_size=CANVAS_SIZE):
    """Render a single glyph as a uint8 image (255=white, 0=black)."""
    img = Image.new('L', (canvas_size, canvas_size), 255)
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

        max_dim = canvas_size * 0.9
        if tw <= max_dim and th <= max_dim:
            x = (canvas_size - tw) / 2 - bbox[0]
            y = (canvas_size - th) / 2 - bbox[1]
            draw.text((x, y), char, fill=0, font=font)
            return np.array(img, dtype=np.uint8)

    return None


def pack(cache_dir, font_paths, chars):
    """Render all glyphs and pack into memmap. Sequential, low memory."""
    num_fonts = len(font_paths)
    num_chars = len(chars)
    total = num_fonts * num_chars
    packed_path = os.path.join(cache_dir, 'glyphs_packed.bin')
    valid_path = os.path.join(cache_dir, 'glyphs_valid.npy')

    # Pre-allocate file
    total_bytes = total * SAMPLE_BYTES
    if not os.path.exists(packed_path) or os.path.getsize(packed_path) != total_bytes:
        logger.info("Allocating %.1f GB packed file...", total_bytes / 1e9)
        with open(packed_path, 'wb') as f:
            f.seek(total_bytes - 1)
            f.write(b'\0')

    mmap = np.memmap(
        packed_path, dtype=np.uint8, mode='r+',
        shape=(total, CANVAS_SIZE, CANVAS_SIZE),
    )
    valid = np.zeros(total, dtype=np.bool_)
    white = np.full((CANVAS_SIZE, CANVAS_SIZE), 255, dtype=np.uint8)

    done = 0
    for fi in range(num_fonts):
        font_path = font_paths[fi]
        for ci in range(num_chars):
            idx = fi * num_chars + ci

            # Try existing .npy cache first
            npy_path = os.path.join(cache_dir, f'{fi}_{ci}.npy')
            if os.path.exists(npy_path):
                try:
                    arr = np.load(npy_path)
                    mmap[idx] = (arr * 255).clip(0, 255).astype(np.uint8)
                    valid[idx] = True
                    done += 1
                    continue
                except (EOFError, ValueError):
                    pass

            # Render fresh
            arr = render_glyph(font_path, chars[ci])
            if arr is not None:
                mmap[idx] = arr
            else:
                mmap[idx] = white
            valid[idx] = True
            done += 1

        if (fi + 1) % 500 == 0:
            mmap.flush()
            logger.info("Progress: %d/%d fonts, %d/%d samples (%.0f%%)",
                        fi + 1, num_fonts, done, total, 100 * done / total)

    mmap.flush()
    del mmap
    np.save(valid_path, valid)
    logger.info("Done: %d/%d samples, saved to %s (%.1f GB)",
                done, total, packed_path, os.path.getsize(packed_path) / 1e9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--db', required=True)
    parser.add_argument('--font-dir', default='/')
    args = parser.parse_args()

    from dataset import GlyphDataset
    ds = GlyphDataset(
        db_path=args.db,
        font_dir=args.font_dir,
        cache_dir=None,
    )
    font_paths = list(ds.font_paths)
    chars = list(ds.chars)
    del ds

    logger.info("Pre-rendering: %d fonts x %d chars = %d samples",
                len(font_paths), len(chars), len(font_paths) * len(chars))
    pack(args.cache_dir, font_paths, chars)


if __name__ == '__main__':
    main()
