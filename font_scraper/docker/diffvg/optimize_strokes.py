#!/usr/bin/env python3
"""
DiffVG-based differentiable stroke optimizer.

Runs inside Docker container with GPU. Reads JSON input from a file,
optimizes polyline strokes to match a font glyph mask using gradient
descent through a differentiable rasterizer, outputs JSON result.

Usage:
    python3 optimize_strokes.py /app/input.json
"""

import json
import sys
import time

import numpy as np
import pydiffvg
import torch
from PIL import Image, ImageDraw, ImageFont


def render_glyph_mask(font_path, char, canvas_size=224):
    """Render a single character glyph as a binary mask."""
    img = Image.new('L', (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)

    # Try font sizes from large to small until it fits
    for font_size in range(200, 20, -5):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            continue
        bbox = font.getbbox(char)
        if bbox is None:
            continue
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= 0 or th <= 0:
            continue
        # Check if it fits within 90% of canvas
        max_dim = canvas_size * 0.9
        if tw <= max_dim and th <= max_dim:
            x = (canvas_size - tw) / 2 - bbox[0]
            y = (canvas_size - th) / 2 - bbox[1]
            draw.text((x, y), char, fill=0, font=font)
            arr = np.array(img)
            return arr < 128  # Boolean mask
    return None


def build_scene(stroke_points_list, stroke_widths, canvas_size, device):
    """Build pydiffvg scene from stroke point tensors.

    Args:
        stroke_points_list: list of (N, 2) tensors (learnable point positions)
        stroke_widths: list of scalar tensors (learnable widths)
        canvas_size: int
        device: torch device

    Returns:
        (shapes, shape_groups) for pydiffvg rendering
    """
    shapes = []
    shape_groups = []

    for i, (points, width) in enumerate(zip(stroke_points_list, stroke_widths)):
        n_pts = points.shape[0]
        # num_control_points = 0 for each segment means straight lines (polyline)
        num_control_points = torch.zeros(n_pts - 1, dtype=torch.int32, device=device)
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=width,
            is_closed=False,
        )
        shapes.append(path)

        # Black stroke, no fill
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i]),
            fill_color=None,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        shape_groups.append(path_group)

    return shapes, shape_groups


def render_scene(shapes, shape_groups, canvas_size, device):
    """Render shapes to a canvas using pydiffvg."""
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_size, canvas_size, shapes, shape_groups,
    )
    render = pydiffvg.RenderFunction.apply
    # White background is essential — without it the canvas is all black (zeros)
    bg = torch.ones(canvas_size, canvas_size, 4, device=device)
    img = render(
        canvas_size,  # width
        canvas_size,  # height
        2,            # num_samples_x
        2,            # num_samples_y
        0,            # seed
        bg,           # background_image (white)
        *scene_args,
    )
    # img shape: (canvas_size, canvas_size, 4) RGBA
    # Convert to grayscale: take the mean of RGB channels
    return img[:, :, :3].mean(dim=2)


def compute_loss(rendered, target, stroke_points_list, stroke_widths=None,
                 glyph_weight=10.0, outside_weight=2.0, smoothness_weight=0.001):
    """Compute optimization loss with region-weighted MSE.

    The glyph typically covers only ~10% of canvas pixels. Plain MSE is
    dominated by the 90% background, causing strokes to vanish. We weight
    glyph-region pixels much higher so the optimizer is incentivized to
    cover the glyph.

    Args:
        rendered: (H, W) float tensor, 0=black stroke, 1=white background
        target: (H, W) float tensor, 0=glyph, 1=background
        stroke_points_list: list of (N, 2) tensors for smoothness regularization
        glyph_weight: extra weight for pixels inside the glyph region
        outside_weight: penalty weight for strokes outside the glyph
        smoothness_weight: weight for curvature penalty
    """
    glyph_mask = (target < 0.5).float()   # 1 where glyph, 0 where bg
    bg_mask = 1.0 - glyph_mask

    # Per-pixel squared error
    sq_err = (rendered - target) ** 2

    # Weighted MSE: glyph pixels weighted much higher
    weighted_err = sq_err * (glyph_mask * glyph_weight + bg_mask)
    loss_mse = weighted_err.mean()

    # Outside penalty: strokes rendering in background area
    stroke_pixels = (1.0 - rendered).clamp(min=0)  # how dark each pixel is
    outside_penalty = (stroke_pixels * bg_mask).sum() / bg_mask.sum().clamp(min=1)

    # Smoothness: penalize high curvature in polylines
    smoothness = torch.tensor(0.0, device=rendered.device)
    for points in stroke_points_list:
        if points.shape[0] >= 3:
            d1 = points[1:] - points[:-1]
            d2 = d1[1:] - d1[:-1]
            smoothness = smoothness + (d2 ** 2).mean()

    return loss_mse + outside_weight * outside_penalty + smoothness_weight * smoothness


def compute_score(rendered, target):
    """Compute a coverage score (0-1, higher = better) similar to host scoring."""
    # Fraction of glyph pixels that are covered by strokes
    glyph_mask = target < 0.5       # True where glyph is
    stroke_mask = rendered < 0.5    # True where strokes render

    if glyph_mask.sum() == 0:
        return 0.0

    covered = (glyph_mask & stroke_mask).sum().float()
    total_glyph = glyph_mask.sum().float()
    coverage = (covered / total_glyph).item()

    # Penalty for strokes outside glyph
    outside = (stroke_mask & ~glyph_mask).sum().float()
    total_stroke = stroke_mask.sum().float().clamp(min=1)
    overshoot = (outside / total_stroke).item()

    return max(0.0, coverage - 0.5 * overshoot)


def downsample_stroke(points, max_points=40):
    """Downsample a polyline to at most max_points, keeping first and last."""
    n = len(points)
    if n <= max_points:
        return points
    # Evenly spaced indices, always including first and last
    indices = np.round(np.linspace(0, n - 1, max_points)).astype(int)
    return [points[i] for i in indices]


def optimize(config):
    """Run the DiffVG optimization loop.

    Args:
        config: dict with font_path, char, canvas_size, initial_strokes,
                num_iterations, stroke_width, lr
    Returns:
        dict with strokes, score, iterations, final_loss, elapsed
    """
    t0 = time.monotonic()

    font_path = config['font_path']
    char = config['char']
    canvas_size = config.get('canvas_size', 224)
    num_iterations = config.get('num_iterations', 500)
    initial_width = config.get('stroke_width', 8.0)
    lr = config.get('lr', 1.0)
    max_pts_per_stroke = config.get('max_points_per_stroke', 40)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    # Render target glyph mask
    mask = render_glyph_mask(font_path, char, canvas_size)
    if mask is None:
        return {'error': f'Could not render glyph for {char}'}

    # Target: 0.0 where glyph is, 1.0 for background
    target = torch.tensor(
        (~mask).astype(np.float32), device=device,
    )

    # Initialize stroke points
    initial_strokes = config.get('initial_strokes')
    if not initial_strokes:
        return {'error': 'No initial strokes provided'}

    stroke_points_list = []
    stroke_widths = []

    total_pts = 0
    for stroke in initial_strokes:
        # Downsample strokes with too many points
        stroke = downsample_stroke(stroke, max_pts_per_stroke)
        total_pts += len(stroke)
        pts = torch.tensor(stroke, dtype=torch.float32, device=device,
                           requires_grad=True)
        stroke_points_list.append(pts)

        width = torch.tensor(initial_width, dtype=torch.float32, device=device,
                             requires_grad=True)
        stroke_widths.append(width)

    print(f'Strokes: {len(stroke_points_list)}, total points: {total_pts}, '
          f'width: {initial_width}, lr: {lr}, iters: {num_iterations}',
          file=sys.stderr, flush=True)

    # Optimizers: separate LRs for points and widths
    optimizer_pts = torch.optim.Adam(stroke_points_list, lr=lr)
    optimizer_w = torch.optim.Adam(stroke_widths, lr=0.5)

    best_loss = float('inf')
    best_points = None
    best_widths = None
    best_score = 0.0
    stagnant_count = 0
    prev_best_loss = float('inf')

    for it in range(num_iterations):
        optimizer_pts.zero_grad()
        optimizer_w.zero_grad()

        shapes, shape_groups = build_scene(
            stroke_points_list, stroke_widths, canvas_size, device,
        )
        rendered = render_scene(shapes, shape_groups, canvas_size, device)

        loss = compute_loss(rendered, target, stroke_points_list, stroke_widths)
        loss.backward()

        optimizer_pts.step()
        optimizer_w.step()

        loss_val = loss.item()

        # Clamp points to canvas and widths to reasonable range every 10 iters
        if it % 10 == 0:
            with torch.no_grad():
                for pts in stroke_points_list:
                    pts.clamp_(0, canvas_size - 1)
                for w in stroke_widths:
                    w.clamp_(2.0, 25.0)

        # Track best
        if loss_val < best_loss:
            best_loss = loss_val
            best_points = [pts.detach().clone() for pts in stroke_points_list]
            best_widths = [w.detach().clone() for w in stroke_widths]

        # Log progress and compute score
        if it % 50 == 0:
            with torch.no_grad():
                score = compute_score(rendered.detach(), target)
                best_score = max(best_score, score)
                widths_str = ', '.join(f'{w.item():.1f}' for w in stroke_widths)
            print(f'iter {it:4d}  loss={loss_val:.4f}  score={score:.3f}  '
                  f'widths=[{widths_str}]',
                  file=sys.stderr, flush=True)

            # Early stopping on stagnation
            if it > 0:
                improvement = prev_best_loss - best_loss
                if improvement < 1e-5:
                    stagnant_count += 1
                else:
                    stagnant_count = 0
                if stagnant_count >= 6:  # 300 iters with no progress
                    print(f'Converged at iter {it}', file=sys.stderr, flush=True)
                    break
                prev_best_loss = best_loss

    # Final score computation with best points and widths
    if best_points is not None:
        with torch.no_grad():
            for orig, best in zip(stroke_points_list, best_points):
                orig.data.copy_(best)
            if best_widths is not None:
                for orig, best in zip(stroke_widths, best_widths):
                    orig.data.copy_(best)
            shapes, shape_groups = build_scene(
                stroke_points_list, stroke_widths, canvas_size, device,
            )
            rendered = render_scene(shapes, shape_groups, canvas_size, device)
            best_score = compute_score(rendered, target)

    elapsed = time.monotonic() - t0

    # Convert best points to output format
    result_strokes = []
    for pts in best_points or stroke_points_list:
        stroke = [[round(float(p[0]), 1), round(float(p[1]), 1)]
                  for p in pts.cpu().numpy()]
        result_strokes.append(stroke)

    return {
        'strokes': result_strokes,
        'score': round(best_score, 4),
        'iterations': num_iterations,
        'final_loss': round(best_loss, 6),
        'elapsed': round(elapsed, 1),
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 optimize_strokes.py <input.json>', file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = json.load(f)

    result = optimize(config)
    print(json.dumps(result), flush=True)
