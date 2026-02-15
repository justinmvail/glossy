#!/usr/bin/env python3
"""Prototype: Distance Transform Ridge-based centerline extraction.

Instead of morphological skeletonization, this finds the ridges (local maxima)
of the distance transform — the set of points equidistant from two or more
boundary points. These ridges ARE the medial axis / centerline.

Advantages over skeletonize():
- Smoother results (no pixel-stepping artifacts)
- Naturally provides stroke width at every point
- Ridge strength helps with pruning (weak ridges = spurious branches)
"""
import sys
import os
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, label
from skimage.morphology import skeletonize
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stroke_rendering import render_glyph_mask, resolve_font_path


def ridge_centerline(mask: np.ndarray, ridge_threshold: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Extract centerline via distance transform ridges.

    Args:
        mask: Binary mask (True = glyph pixels)
        ridge_threshold: Minimum distance value to keep (prunes thin features)

    Returns:
        ridge_mask: Boolean array of ridge pixels
        dist: Distance transform values (stroke half-width at each point)
    """
    # Distance transform: each interior pixel gets distance to nearest boundary
    dist = distance_transform_edt(mask)

    # Find ridges: points that are local maxima in at least one direction
    # A ridge point is where the distance function has a local max perpendicular
    # to the stroke direction.
    #
    # Method: Compare each pixel to its neighbors. A ridge pixel is one where
    # the distance value is >= at least one pair of opposing neighbors.

    # Pad to handle borders
    d = dist

    # Check all 4 axis-aligned and diagonal directions for ridge condition
    # A pixel is a ridge if it's a local max along ANY axis
    ridge = np.zeros_like(mask, dtype=bool)

    # Horizontal ridge (local max in horizontal direction)
    ridge |= (d >= np.roll(d, 1, axis=1)) & (d >= np.roll(d, -1, axis=1))
    # Vertical ridge
    ridge |= (d >= np.roll(d, 1, axis=0)) & (d >= np.roll(d, -1, axis=0))
    # Diagonal /
    ridge |= (d >= np.roll(np.roll(d, 1, axis=0), 1, axis=1)) & \
             (d >= np.roll(np.roll(d, -1, axis=0), -1, axis=1))
    # Diagonal \
    ridge |= (d >= np.roll(np.roll(d, 1, axis=0), -1, axis=1)) & \
             (d >= np.roll(np.roll(d, -1, axis=0), 1, axis=1))

    # Only keep ridges inside the shape and above threshold
    ridge &= mask
    ridge &= (dist >= ridge_threshold)

    # The raw ridge is thick — thin it to 1px using skeletonize
    # (ironic, but this is just thinning the ridge, not the original shape)
    if ridge.any():
        ridge = skeletonize(ridge)

    return ridge, dist


def ridge_centerline_v2(mask: np.ndarray, ridge_threshold: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Improved ridge extraction using gradient divergence.

    The medial axis is where the gradient of the distance field is discontinuous
    (the distance "flow" diverges). We detect this by looking for pixels where
    the Laplacian of the distance field is strongly negative.
    """
    dist = distance_transform_edt(mask).astype(np.float64)

    # Laplacian of distance field — strongly negative at ridges
    from scipy.ndimage import laplace
    lap = laplace(dist)

    # Ridge pixels: inside shape, negative laplacian, sufficient distance
    ridge = mask & (lap < -0.5) & (dist >= ridge_threshold)

    # Thin to 1px
    if ridge.any():
        ridge = skeletonize(ridge)

    return ridge, dist


def ridge_centerline_v3(mask: np.ndarray, ridge_threshold: float = 1.5,
                        gradient_thresh: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Ridge extraction via flux analysis.

    Compute the average outward flux of the distance gradient. The medial axis
    has strongly negative flux (gradient vectors point away from it).
    """
    dist = distance_transform_edt(mask).astype(np.float64)

    # Gradient of distance field
    gy, gx = np.gradient(dist)

    # Normalize gradient
    mag = np.sqrt(gx**2 + gy**2) + 1e-10
    gx_norm = gx / mag
    gy_norm = gy / mag

    # Divergence of normalized gradient = average outward flux
    div = np.gradient(gx_norm, axis=1) + np.gradient(gy_norm, axis=0)

    # Medial axis has strongly negative divergence
    ridge = mask & (div < -gradient_thresh) & (dist >= ridge_threshold)

    # Thin to 1px
    if ridge.any():
        ridge = skeletonize(ridge)

    return ridge, dist


def trace_ridges_to_strokes(ridge: np.ndarray, dist: np.ndarray,
                            min_length: int = 5) -> list[list[list[float]]]:
    """Convert ridge pixels to ordered stroke paths.

    Args:
        ridge: Binary ridge mask (1px thin)
        dist: Distance transform (for stroke width info)
        min_length: Minimum stroke length in pixels

    Returns:
        List of strokes, each stroke is [[x,y,width], ...]
    """
    # Build adjacency
    ys, xs = np.where(ridge)
    if len(ys) == 0:
        return []

    pixel_set = set(zip(ys.tolist(), xs.tolist()))
    adj = {}
    for y, x in pixel_set:
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in pixel_set:
                    neighbors.append((ny, nx))
        adj[(y, x)] = neighbors

    # Find endpoints (degree 1) and junctions (degree >= 3)
    endpoints = [p for p, n in adj.items() if len(n) == 1]
    junctions = set(p for p, n in adj.items() if len(n) >= 3)

    # Trace paths from endpoints
    visited_edges = set()
    strokes = []

    def trace_from(start):
        path = [start]
        visited_edges.add(start)
        current = start
        while True:
            neighbors = [n for n in adj.get(current, [])
                        if n not in visited_edges or n in junctions]
            if not neighbors:
                break
            # Prefer unvisited
            unvisited = [n for n in neighbors if n not in visited_edges]
            if unvisited:
                nxt = unvisited[0]
            else:
                break
            visited_edges.add(nxt)
            path.append(nxt)
            if nxt in junctions:
                break
            current = nxt
        return path

    # Trace from each endpoint
    for ep in endpoints:
        if ep in visited_edges:
            continue
        path = trace_from(ep)
        if len(path) >= min_length:
            # Convert to [x, y, width] format
            stroke = [[float(x), float(y), float(dist[y, x])]
                     for y, x in path]
            strokes.append(stroke)

    # Handle cycles (no endpoints)
    remaining = pixel_set - visited_edges
    while remaining:
        start = next(iter(remaining))
        path = trace_from(start)
        remaining -= visited_edges
        if len(path) >= min_length:
            stroke = [[float(x), float(y), float(dist[y, x])]
                     for y, x in path]
            strokes.append(stroke)

    return strokes


def visualize(mask, ridge, dist, strokes, title=""):
    """Create a visualization image showing mask, ridge, and strokes."""
    h, w = mask.shape

    # Create 4-panel image
    panel_w = w
    img = Image.new('RGB', (panel_w * 4, h), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Panel 1: Original mask
    mask_img = np.zeros((h, w, 3), dtype=np.uint8)
    mask_img[mask] = [200, 200, 200]
    img.paste(Image.fromarray(mask_img), (0, 0))

    # Panel 2: Distance transform (heatmap)
    if dist.max() > 0:
        dist_norm = (dist / dist.max() * 255).astype(np.uint8)
        # Apply colormap
        dist_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        dist_rgb[:, :, 0] = dist_norm  # Red channel
        dist_rgb[:, :, 2] = 255 - dist_norm  # Blue channel (inverted)
        dist_rgb[~mask] = [30, 30, 30]
        img.paste(Image.fromarray(dist_rgb), (panel_w, 0))

    # Panel 3: Ridge overlay on mask
    ridge_img = np.zeros((h, w, 3), dtype=np.uint8)
    ridge_img[mask] = [60, 60, 60]
    ridge_img[ridge] = [0, 255, 0]
    img.paste(Image.fromarray(ridge_img), (panel_w * 2, 0))

    # Panel 4: Traced strokes with width
    stroke_img = np.zeros((h, w, 3), dtype=np.uint8)
    stroke_img[mask] = [40, 40, 40]
    pil_stroke = Image.fromarray(stroke_img)
    d = ImageDraw.Draw(pil_stroke)
    colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255),
              (255, 255, 80), (255, 80, 255), (80, 255, 255)]
    for i, stroke in enumerate(strokes):
        color = colors[i % len(colors)]
        pts = [(p[0], p[1]) for p in stroke]
        if len(pts) >= 2:
            d.line(pts, fill=color, width=2)
    img.paste(pil_stroke, (panel_w * 3, 0))

    return img


def compare_methods(font_path: str, char: str, canvas_size: int = 224):
    """Compare ridge methods against morphological skeleton."""
    mask = render_glyph_mask(font_path, char, canvas_size)
    if mask is None:
        print(f"Could not render {char}")
        return

    print(f"\nCharacter: '{char}'")
    print(f"Mask pixels: {mask.sum()}")

    # Method 1: Classic skeleton (baseline)
    skel = skeletonize(mask)
    skel_pixels = skel.sum()

    # Method 2: Local max ridge
    ridge_v1, dist = ridge_centerline(mask, ridge_threshold=1.5)
    v1_pixels = ridge_v1.sum()

    # Method 3: Laplacian ridge
    ridge_v2, _ = ridge_centerline_v2(mask, ridge_threshold=1.5)
    v2_pixels = ridge_v2.sum()

    # Method 4: Flux ridge
    ridge_v3, _ = ridge_centerline_v3(mask, ridge_threshold=1.5, gradient_thresh=0.7)
    v3_pixels = ridge_v3.sum()

    print(f"  Skeleton:       {skel_pixels:5d} pixels")
    print(f"  Ridge (local):  {v1_pixels:5d} pixels")
    print(f"  Ridge (lapl):   {v2_pixels:5d} pixels")
    print(f"  Ridge (flux):   {v3_pixels:5d} pixels")

    # Trace strokes from each
    for name, ridge in [("local", ridge_v1), ("laplacian", ridge_v2), ("flux", ridge_v3)]:
        strokes = trace_ridges_to_strokes(ridge, dist)
        total_pts = sum(len(s) for s in strokes)
        print(f"  {name:10s} strokes: {len(strokes):3d} ({total_pts} points)")

    # Generate comparison images
    strokes_v1 = trace_ridges_to_strokes(ridge_v1, dist)
    strokes_v2 = trace_ridges_to_strokes(ridge_v2, dist)
    strokes_v3 = trace_ridges_to_strokes(ridge_v3, dist)

    # Create comparison: skeleton vs best ridge method
    h, w = mask.shape
    comp = Image.new('RGB', (w * 3, h), (30, 30, 30))

    # Skeleton
    skel_img = np.zeros((h, w, 3), dtype=np.uint8)
    skel_img[mask] = [60, 60, 60]
    skel_img[skel] = [255, 255, 0]
    comp.paste(Image.fromarray(skel_img), (0, 0))

    # Ridge v2 (laplacian)
    r2_img = np.zeros((h, w, 3), dtype=np.uint8)
    r2_img[mask] = [60, 60, 60]
    r2_img[ridge_v2] = [0, 255, 0]
    comp.paste(Image.fromarray(r2_img), (w, 0))

    # Ridge v3 (flux)
    r3_img = np.zeros((h, w, 3), dtype=np.uint8)
    r3_img[mask] = [60, 60, 60]
    r3_img[ridge_v3] = [0, 128, 255]
    comp.paste(Image.fromarray(r3_img), (w * 2, 0))

    return comp, {
        'skeleton': skel,
        'ridge_local': ridge_v1,
        'ridge_laplacian': ridge_v2,
        'ridge_flux': ridge_v3,
        'dist': dist,
        'mask': mask,
    }


def main():
    import sqlite3

    db = sqlite3.connect('fonts.db')
    db.row_factory = sqlite3.Row
    fonts = db.execute('''
        SELECT f.id, f.name, f.file_path FROM fonts f
        WHERE f.id NOT IN (SELECT font_id FROM font_removals)
        ORDER BY f.name LIMIT 3
    ''').fetchall()
    db.close()

    test_chars = ['A', 'B', 'R', 'g', 'e', 'W']

    for font in fonts:
        fp = resolve_font_path(font['file_path'])
        print(f"\n{'='*60}")
        print(f"Font: {font['name']}")
        print(f"{'='*60}")

        for char in test_chars:
            result = compare_methods(fp, char, canvas_size=224)
            if result:
                comp_img, data = result
                safe_name = font['name'].replace(' ', '_').replace('/', '_')
                out_path = f"/tmp/ridge_{safe_name}_{char}.png"
                comp_img.save(out_path)
                print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
