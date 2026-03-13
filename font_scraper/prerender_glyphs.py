#!/usr/bin/env python3
"""Pre-render all glyph images for ML training.

Renders every (font, character) pair as a 224x224 binary mask and stores
them in compressed .npz archive files. These can be loaded directly by
the training dataset instead of rendering from TTF on-the-fly.

Output: exports/glyph_cache_partN.npz files (~200MB total)
Each .npz contains arrays keyed by "{font_idx}_{char_idx}" with uint8 masks.

Usage:
    python3 prerender_glyphs.py                    # Full render
    python3 prerender_glyphs.py --max-fonts 100    # Debug subset
    python3 prerender_glyphs.py --workers 8        # Parallel rendering
"""

import argparse
import logging
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

CANVAS_SIZE = 224
CHARS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
# Max samples per archive file (~50MB each uncompressed, ~25MB compressed)
SAMPLES_PER_ARCHIVE = 100_000

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / 'fonts.db'
EXPORT_DIR = BASE_DIR / 'exports'


def render_glyph(font_path: str, char: str, canvas_size: int = CANVAS_SIZE) -> np.ndarray | None:
    """Render a single glyph as a binary uint8 mask (0=bg, 1=glyph)."""
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
            arr = np.array(img)
            return (arr < 128).astype(np.uint8)

    return None


def render_font_batch(args):
    """Render all characters for one font. Called in worker process."""
    font_idx, font_path, chars = args
    results = {}
    for ci, char in enumerate(chars):
        mask = render_glyph(font_path, char)
        if mask is not None:
            results[f"{font_idx}_{ci}"] = mask
    return results


def load_font_paths(db_path: str, max_fonts: int = None) -> list:
    """Load eligible font paths from database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    query = """
        SELECT f.id, f.file_path FROM fonts f
        LEFT JOIN font_checks fc ON f.id = fc.font_id
        LEFT JOIN font_removals fr ON f.id = fr.font_id
        WHERE fr.font_id IS NULL
        AND (fc.prefilter_passed = 1 OR fc.prefilter_passed IS NULL)
        AND (fc.is_cursive = 0 OR fc.is_cursive IS NULL)
    """
    if max_fonts:
        query += f" LIMIT {max_fonts}"

    rows = conn.execute(query).fetchall()
    conn.close()

    paths = []
    for row in rows:
        fp = row['file_path']
        if not os.path.isabs(fp):
            fp = os.path.join(str(BASE_DIR), fp)
        if os.path.exists(fp):
            paths.append(fp)

    return paths


def main():
    parser = argparse.ArgumentParser(description='Pre-render glyph images for training')
    parser.add_argument('--max-fonts', type=int, default=None, help='Limit fonts (debug)')
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers')
    parser.add_argument('--output-dir', type=str, default=str(EXPORT_DIR), help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load fonts
    font_paths = load_font_paths(str(DB_PATH), args.max_fonts)
    total_samples = len(font_paths) * len(CHARS)
    logger.info("Fonts: %d, Chars: %d, Total samples: %d", len(font_paths), len(CHARS), total_samples)

    # Render in parallel
    t0 = time.time()
    all_masks = {}
    rendered = 0
    failed = 0

    work = [(fi, fp, CHARS) for fi, fp in enumerate(font_paths)]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_font_batch, w): w[0] for w in work}

        for i, future in enumerate(as_completed(futures)):
            results = future.result()
            all_masks.update(results)
            rendered += len(results)
            failed += len(CHARS) - len(results)

            if (i + 1) % 500 == 0 or i + 1 == len(futures):
                elapsed = time.time() - t0
                rate = rendered / elapsed
                eta = (total_samples - rendered) / rate if rate > 0 else 0
                logger.info(
                    "Progress: %d/%d fonts | %d rendered | %d failed | %.0f/s | ETA %.0fs",
                    i + 1, len(font_paths), rendered, failed, rate, eta,
                )

    elapsed = time.time() - t0
    logger.info("Rendering complete: %d masks in %.1fs (%.0f/s)", rendered, elapsed, rendered / elapsed)

    # Save as compressed archives
    keys = sorted(all_masks.keys(), key=lambda k: (int(k.split('_')[0]), int(k.split('_')[1])))
    n_archives = (len(keys) + SAMPLES_PER_ARCHIVE - 1) // SAMPLES_PER_ARCHIVE

    logger.info("Saving %d masks across %d archive(s)...", len(keys), n_archives)

    # Also save the font path index so we can map font_idx back
    index = {
        'canvas_size': CANVAS_SIZE,
        'chars': ''.join(CHARS),
        'n_fonts': len(font_paths),
        'n_samples': rendered,
        'font_paths': font_paths,
    }

    for part in range(n_archives):
        start = part * SAMPLES_PER_ARCHIVE
        end = min(start + SAMPLES_PER_ARCHIVE, len(keys))
        chunk_keys = keys[start:end]

        archive_path = output_dir / f"glyph_cache_part{part + 1}.npz"
        chunk = {k: all_masks[k] for k in chunk_keys}

        np.savez_compressed(str(archive_path), **chunk)
        size = archive_path.stat().st_size
        logger.info("  Saved %s: %d samples, %.1f MB", archive_path.name, len(chunk), size / 1e6)

    # Save index
    import json
    index_path = output_dir / "glyph_cache_index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    logger.info("Saved index: %s", index_path.name)

    total_size = sum(
        (output_dir / f"glyph_cache_part{p + 1}.npz").stat().st_size
        for p in range(n_archives)
    )
    logger.info("Total cache size: %.1f MB", total_size / 1e6)
    logger.info("Done! Glyphs ready for training.")


if __name__ == '__main__':
    main()
