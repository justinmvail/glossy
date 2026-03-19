#!/usr/bin/env python3
"""Precompute pseudo-ground-truth stroke decompositions for training.

Uses medial axis skeletonization to extract stroke paths from font glyphs.
Results are saved as .npz files in a cache directory that gets mounted
into the Docker training container.

Usage:
    python3 precompute_pseudo_gt.py --db fonts.db --output docker/stroke_model/pseudo_gt/
    python3 precompute_pseudo_gt.py --db fonts.db --output docker/stroke_model/pseudo_gt/ --max-fonts 100
"""

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

CANVAS_SIZE = 224
CHARS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
MAX_STROKES = 8
MAX_POINTS = 40


def render_glyph(font_path, char, canvas_size=CANVAS_SIZE):
    """Render a character using a font file. Returns binary mask or None."""
    try:
        for size in range(200, 15, -5):
            try:
                font = ImageFont.truetype(font_path, size)
            except Exception:
                continue
            bbox = font.getbbox(char)
            if bbox is None:
                continue
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < 5 or h < 5:
                continue
            max_dim = max(w, h)
            target = int(canvas_size * 0.85)
            if max_dim > target:
                continue

            img = Image.new('L', (canvas_size, canvas_size), 255)
            draw = ImageDraw.Draw(img)
            x = (canvas_size - w) // 2 - bbox[0]
            y = (canvas_size - h) // 2 - bbox[1]
            draw.text((x, y), char, font=font, fill=0)
            arr = np.array(img)
            if arr.min() < 128:
                return arr < 128  # Boolean mask
            break
    except Exception:
        pass
    return None


def extract_strokes_skeleton(mask, canvas_size=CANVAS_SIZE):
    """Extract stroke paths from a binary mask using skeletonization.

    Returns list of (points_array, estimated_width) tuples.
    """
    try:
        from skimage.morphology import skeletonize
        from skimage.measure import label, regionprops
    except ImportError:
        logger.error("scikit-image required. Install: pip install scikit-image")
        return []

    if mask is None or mask.sum() < 10:
        return []

    # Skeletonize
    skel = skeletonize(mask)

    # Find connected components in skeleton
    labeled = label(skel)
    regions = regionprops(labeled)

    strokes = []
    for region in regions:
        coords = region.coords  # (N, 2) in row, col format
        if len(coords) < 2:
            continue

        # Order points by tracing the path (greedy nearest neighbor)
        ordered = [coords[0]]
        remaining = set(range(1, len(coords)))
        while remaining:
            cur = ordered[-1]
            best_dist = float('inf')
            best_idx = -1
            for idx in remaining:
                d = np.sum((coords[idx] - cur) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            if best_dist > 25:  # gap too large, stop this stroke
                break
            ordered.append(coords[best_idx])
            remaining.remove(best_idx)

        if len(ordered) < 2:
            continue

        # Convert to (x, y) format and simplify
        points = np.array(ordered, dtype=np.float32)
        points = points[:, ::-1].copy()  # row,col -> x,y

        # Simplify to MAX_POINTS using uniform sampling
        if len(points) > MAX_POINTS:
            indices = np.linspace(0, len(points) - 1, MAX_POINTS, dtype=int)
            points = points[indices]

        # Estimate width from distance transform at skeleton points
        from scipy.ndimage import distance_transform_edt
        dt = distance_transform_edt(mask)
        skel_y = np.clip(np.array(ordered)[:, 0], 0, canvas_size - 1).astype(int)
        skel_x = np.clip(np.array(ordered)[:, 1], 0, canvas_size - 1).astype(int)
        widths_at_skel = dt[skel_y, skel_x]
        avg_width = float(np.median(widths_at_skel) * 2)  # diameter = 2 * radius
        avg_width = max(1.0, min(avg_width, 30.0))

        strokes.append((points, avg_width))

    # Sort by length (longest first) and limit to MAX_STROKES
    strokes.sort(key=lambda s: len(s[0]), reverse=True)
    return strokes[:MAX_STROKES]


def font_char_hash(font_path, char_idx):
    """Stable hash for a font+character combination."""
    key = f"{font_path}:{char_idx}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description='Precompute pseudo-GT strokes')
    parser.add_argument('--db', type=str, required=True, help='Path to fonts.db')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--font-dir', type=str, default='.', help='Base font directory')
    parser.add_argument('--max-fonts', type=int, default=None, help='Limit fonts')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load font paths
    conn = sqlite3.connect(args.db)
    rows = conn.execute('''
        SELECT f.file_path FROM fonts f
        LEFT JOIN font_checks fc ON f.id = fc.font_id
        LEFT JOIN font_removals fr ON f.id = fr.font_id
        WHERE fr.font_id IS NULL
        AND (fc.prefilter_passed = 1 OR fc.prefilter_passed IS NULL)
        AND (fc.is_cursive = 0 OR fc.is_cursive IS NULL)
    ''').fetchall()
    conn.close()

    font_paths = [r[0] for r in rows]
    if args.max_fonts:
        font_paths = font_paths[:args.max_fonts]

    logger.info("Processing %d fonts x %d chars = %d glyphs",
                len(font_paths), len(CHARS), len(font_paths) * len(CHARS))

    # Build manifest of what to process
    total = 0
    skipped = 0
    failed = 0

    for fi, font_path in enumerate(font_paths):
        full_path = os.path.join(args.font_dir, font_path)
        if not os.path.exists(full_path):
            skipped += 1
            continue

        for ci, char in enumerate(CHARS):
            out_file = os.path.join(args.output, f"{font_char_hash(font_path, ci)}.npz")
            if os.path.exists(out_file):
                skipped += 1
                continue

            # Render glyph
            mask = render_glyph(full_path, char)
            if mask is None:
                failed += 1
                continue

            # Extract strokes
            strokes = extract_strokes_skeleton(mask)
            if not strokes:
                failed += 1
                continue

            # Save as npz
            points_list = []
            widths_list = []
            n_points_list = []
            for pts, w in strokes:
                # Normalize to [0, 1]
                pts_norm = pts / CANVAS_SIZE
                # Pad to MAX_POINTS
                padded = np.zeros((MAX_POINTS, 2), dtype=np.float32)
                n = min(len(pts_norm), MAX_POINTS)
                padded[:n] = pts_norm[:n]
                points_list.append(padded)
                widths_list.append(w)
                n_points_list.append(n)

            # Pad to MAX_STROKES
            while len(points_list) < MAX_STROKES:
                points_list.append(np.zeros((MAX_POINTS, 2), dtype=np.float32))
                widths_list.append(0.0)
                n_points_list.append(2)

            np.savez_compressed(out_file,
                                points=np.array(points_list[:MAX_STROKES]),
                                widths=np.array(widths_list[:MAX_STROKES], dtype=np.float32),
                                n_points=np.array(n_points_list[:MAX_STROKES], dtype=np.int32),
                                n_strokes=len(strokes),
                                font_path=font_path,
                                char=char)
            total += 1

        if (fi + 1) % 100 == 0:
            logger.info("Progress: %d/%d fonts, %d saved, %d skipped, %d failed",
                        fi + 1, len(font_paths), total, skipped, failed)

    logger.info("Done: %d saved, %d skipped, %d failed", total, skipped, failed)

    # Write manifest
    manifest = {
        'n_glyphs': total,
        'canvas_size': CANVAS_SIZE,
        'max_strokes': MAX_STROKES,
        'max_points': MAX_POINTS,
        'chars': CHARS,
    }
    with open(os.path.join(args.output, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s/manifest.json", args.output)


if __name__ == '__main__':
    main()
